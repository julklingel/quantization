
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from collections import OrderedDict
from torchvision.models import resnet18, ResNet18_Weights
from models.backbone.backbone import ResnetBackbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class BoxHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv_layers(x)
        return self.avgpool(x).flatten(1)

class Resnet18_Detection(nn.Module):
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
        box_head = BoxHead(512)
        self.model = FasterRCNN(
            self.backbone,
            rpn_anchor_generator=self.anchor_generator,
            box_roi_pool=self.roi_pooler,
            box_head=box_head,
            box_predictor=FastRCNNPredictor(box_head.conv_layers[-2].out_channels, 2),
            min_size=224,
            max_size=224
        )

    def forward(self, x, targets=None):
        return self.model(x, targets)