"""Production candidate architecture for the three-hour ApuCast ResNet."""

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, inputs, width):
        super().__init__()
        self.conv1 = nn.Conv2d(inputs, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.skip = nn.Conv2d(inputs, width, 1) if inputs != width else nn.Identity()

    def forward(self, x):
        return F.gelu(self.conv2(F.gelu(self.conv1(x))) + self.skip(x))


def make_model(ns, variant):
    if variant != "resnet":
        raise ValueError(f"Arquitectura no disponible: {variant}")
    model = ns["Net"]()
    model.enc = ResidualBlock(14, 64)
    model.dec = nn.Sequential(ResidualBlock(73, 64), nn.Conv2d(64, 4, 1))
    return model.to(device=ns["dev"], memory_format=torch.channels_last)
