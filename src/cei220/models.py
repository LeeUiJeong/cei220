from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(torch.flatten(features, 1))


def resnet18_small(in_channels: int, num_classes: int = 10) -> nn.Module:
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        in_channels,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_model(name: str, in_channels: int, num_classes: int = 10) -> nn.Module:
    if name == "smallcnn":
        return SmallCNN(in_channels=in_channels, num_classes=num_classes)
    if name == "resnet18_small":
        return resnet18_small(in_channels=in_channels, num_classes=num_classes)
    raise ValueError(f"Unsupported model: {name}")
