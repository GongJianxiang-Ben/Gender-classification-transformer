"""
ResNet18 with Dilated Convolutions (dilation=2) in stages 3 and 4.
Replaces standard 3x3 conv with dilated conv to increase receptive field.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Type


class BasicBlockDilated(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None,
        dilation: int = 1,
    ) -> None:
        super(BasicBlockDilated, self).__init__()
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels * expansion,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels * expansion)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet18Dilated(nn.Module):
    def __init__(self, img_channels: int = 3, num_classes: int = 2) -> None:
        super(ResNet18Dilated, self).__init__()
        self.expansion = 1
        self.in_channels = 64

        # Stage 0 — same as vanilla
        self.conv1 = nn.Conv2d(img_channels, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stages 1-2 — standard (dilation=1)
        self.layer1 = self._make_layer(64,  2, stride=1, dilation=1)
        self.layer2 = self._make_layer(128, 2, stride=2, dilation=1)

        # Stages 3-4 — dilated (dilation=2)
        self.layer3 = self._make_layer(256, 2, stride=2, dilation=2)
        self.layer4 = self._make_layer(512, 2, stride=2, dilation=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512 * self.expansion, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride, dilation):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * self.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = [BasicBlockDilated(self.in_channels, out_channels,
                                    stride=stride, expansion=self.expansion,
                                    downsample=downsample, dilation=dilation)]
        self.in_channels = out_channels * self.expansion
        for _ in range(1, num_blocks):
            layers.append(BasicBlockDilated(self.in_channels, out_channels,
                                            expansion=self.expansion,
                                            dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x