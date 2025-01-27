
from torchvision.models import resnet18, ResNet18_Weights
import torch
import torch.nn as nn


class Resnet18_Classification(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.resnet18 = resnet18(weights=weights, progress=False)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, 2) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_pred = self.resnet18(x)
        return y_pred