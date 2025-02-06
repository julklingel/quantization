import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from models.backbone.backbone import ResnetBackbone


class MaskHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
        )
        self.deconv = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.mask_predictor = nn.Conv2d(256, 2, 1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.deconv(x)
        return self.mask_predictor(x)

class Resnet18_Segmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResnetBackbone()
        self.anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )
        self.roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )
        mask_head = MaskHead(512)
        self.model = MaskRCNN(
            self.backbone,
            num_classes=2,
            rpn_anchor_generator=self.anchor_generator,
            box_roi_pool=self.roi_pooler,
            mask_roi_pool=self.roi_pooler,
            mask_head=mask_head,
            mask_predictor=MaskRCNNPredictor(256, 256, 2),
            min_size=224,
            max_size=224
        )

    def forward(self, x, targets=None):
        return self.model(x, targets)