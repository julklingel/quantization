import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import resnet18

class Resnet18_Detection(nn.Module):
    def __init__(self):
        super().__init__()
     
        backbone = resnet18(pretrained=False)
        
        # Remove the fully connected layer
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        backbone.out_channels = 512
        
   
        rpn_anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )
        

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'], output_size=7, sampling_ratio=2
        )
        

        self.model = FasterRCNN(
            backbone,
            num_classes=2,  # 1 class + background
            rpn_anchor_generator=rpn_anchor_generator,
            box_roi_pool=roi_pooler
        )
    
    def forward(self, x):
        return self.model(x)