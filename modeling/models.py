import torch
import torchvision
import torch.nn as nn

import torch.nn.functional as F

from typing import List

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int=10):
        """
        ---------
        Arguments
        ---------
        num_classes: int
            an integer indicating the number of classes in the dataset
        """
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )

        self.fc_block = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block(x)
        x = torch.flatten(x, 1)
        x = self.fc_block(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """
        ---------
        Arguments
        ---------
        in_channels: int
            an integer indicating the number of input channels of the input to the residual block
        out_channels: int
            an integer indicating the number of output channels of the output of the residual block
        """
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.elu = nn.ELU()
        self.out_channels = out_channels

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.elu(out)
        return out


class SimpleResNet(nn.Module):
    def __init__(self, num_classes: int=10, list_num_res_units_per_block: List=[3, 3]):
        """
        ---------
        Arguments
        ---------
        num_classes: int
            an integer indicating the number of classes in the dataset
        list_num_res_units_per_block: list
            a list of integers representing number of residual units per block
        """
        super().__init__()
        self.list_num_res_units_per_block = list_num_res_units_per_block

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
        )

        self.residual_block_1 = nn.Sequential(
            *[
                ResidualBlock(32, 32)
                for i in range(self.list_num_res_units_per_block[0])
            ]
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
        )

        self.residual_block_2 = nn.Sequential(
            *[
                ResidualBlock(64, 64)
                for i in range(self.list_num_res_units_per_block[1])
            ]
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block_1(x)
        x = self.residual_block_1(x)
        x = self.conv_block_2(x)
        x = self.residual_block_2(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x


class ComplexResNet(nn.Module):
    def __init__(self, num_classes: int=10, list_num_res_units_per_block: List=[4, 4, 4]):
        """
        ---------
        Arguments
        ---------
        num_classes: int
            an integer indicating the number of classes in the dataset
        list_num_res_units_per_block: list
            a list of integers representing number of residual units per block
        """
        super().__init__()
        self.list_num_res_units_per_block = list_num_res_units_per_block

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
        )

        self.residual_block_1 = nn.Sequential(
            *[
                ResidualBlock(32, 32)
                for i in range(self.list_num_res_units_per_block[0])
            ]
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
        )

        self.residual_block_2 = nn.Sequential(
            *[
                ResidualBlock(64, 64)
                for i in range(self.list_num_res_units_per_block[1])
            ]
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
        )

        self.residual_block_3 = nn.Sequential(
            *[
                ResidualBlock(128, 128)
                for i in range(self.list_num_res_units_per_block[2])
            ]
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(128, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        return

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv_block_1(x)
        x = self.residual_block_1(x)
        x = self.conv_block_2(x)
        x = self.residual_block_2(x)
        x = self.conv_block_3(x)
        x = self.residual_block_3(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x
