
from torchvision.models import resnet18, ResNet18_Weights
import torch
import torch.nn as nn
from models.backbone.backbone import ResnetBackbone


class Resnet18_Classification(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResnetBackbone()
        self.head = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        features = self.backbone(x)['0']
        return self.head(features)