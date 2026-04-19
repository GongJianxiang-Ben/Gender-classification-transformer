"""
ResNet18 with Deformable Convolutions in stages 3 and 4.
Uses torchvision.ops.deform_conv2d.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Type
from torchvision.ops import deform_conv2d


class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, bias=False):
        super(DeformableConv2d, self).__init__()
        self.stride  = stride
        self.padding = padding
        self.weight  = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )
        # offset: 2 * kernel_size^2 channels (x and y offsets for each kernel position)
        self.offset_conv = nn.Conv2d(
            in_channels, 2 * kernel_size * kernel_size,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=True
        )
        # mask: kernel_size^2 channels
        self.mask_conv = nn.Conv2d(
            in_channels, kernel_size * kernel_size,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=True
        )
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        nn.init.zeros_(self.mask_conv.weight)
        nn.init.zeros_(self.mask_conv.bias)

    def forward(self, x: Tensor) -> Tensor:
        offset = self.offset_conv(x)
        mask   = torch.sigmoid(self.mask_conv(x))
        return deform_conv2d(x, offset, self.weight, mask=mask,
                             stride=self.stride, padding=self.padding)


class BasicBlockDeformable(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                 expansion=1, downsample=None, use_deform=False):
        super(BasicBlockDeformable, self).__init__()
        self.expansion = expansion
        self.downsample = downsample

        if use_deform:
            self.conv1 = DeformableConv2d(in_channels, out_channels,
                                          kernel_size=3, stride=stride, padding=1)
            self.conv2 = DeformableConv2d(out_channels, out_channels * expansion,
                                          kernel_size=3, stride=1, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
            self.conv2 = nn.Conv2d(out_channels, out_channels * expansion,
                                   kernel_size=3, padding=1, bias=False)

        self.bn1  = nn.BatchNorm2d(out_channels)
        self.bn2  = nn.BatchNorm2d(out_channels * expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet18Deformable(nn.Module):
    def __init__(self, img_channels: int = 3, num_classes: int = 2) -> None:
        super(ResNet18Deformable, self).__init__()
        self.expansion   = 1
        self.in_channels = 64

        self.conv1   = nn.Conv2d(img_channels, 64, kernel_size=7,
                                 stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stages 1-2 — standard conv
        self.layer1 = self._make_layer(64,  2, stride=1, use_deform=False)
        self.layer2 = self._make_layer(128, 2, stride=2, use_deform=False)

        # Stages 3-4 — deformable conv
        self.layer3 = self._make_layer(256, 2, stride=2, use_deform=True)
        self.layer4 = self._make_layer(512, 2, stride=2, use_deform=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512 * self.expansion, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride, use_deform):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * self.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = [BasicBlockDeformable(self.in_channels, out_channels,
                                       stride=stride, expansion=self.expansion,
                                       downsample=downsample,
                                       use_deform=use_deform)]
        self.in_channels = out_channels * self.expansion
        for _ in range(1, num_blocks):
            layers.append(BasicBlockDeformable(self.in_channels, out_channels,
                                               expansion=self.expansion,
                                               use_deform=use_deform))
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