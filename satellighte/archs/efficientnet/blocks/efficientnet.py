import copy
import math
from ast import Str
from typing import Any, Callable, List, Optional

import torch
from torch import nn

from .layers import ConvNormActivation, MBConvConfig, FusedMBConvConfig


class Efficient_Net(nn.Module):
    # pylint: disable=no-member
    def __init__(
        self,
        inverted_residual_setting: Str,
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        width_mult=1.0,
        depth_mult=1.0,
        **kwargs: Any,
    ) -> None:
        """
        EfficientNet V1 and V2 main class

        Args:
            inverted_residual_setting: Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use

        This implementation is taken from torchvision repositories.
        `torchvision.models.efficientnet`
        Here is a link for it: `github.com/pytorch/vision`
        """
        super().__init__()

        if inverted_residual_setting == "v1":
            inverted_residual_setting = [
                MBConvConfig(
                    1, 3, 1, 32, 16, 1, width_mult=width_mult, depth_mult=depth_mult
                ),
                MBConvConfig(
                    6, 3, 2, 16, 24, 2, width_mult=width_mult, depth_mult=depth_mult
                ),
                MBConvConfig(
                    6, 5, 2, 24, 40, 2, width_mult=width_mult, depth_mult=depth_mult
                ),
                MBConvConfig(
                    6, 3, 2, 40, 80, 3, width_mult=width_mult, depth_mult=depth_mult
                ),
                MBConvConfig(
                    6, 5, 1, 80, 112, 3, width_mult=width_mult, depth_mult=depth_mult
                ),
                MBConvConfig(
                    6, 5, 2, 112, 192, 4, width_mult=width_mult, depth_mult=depth_mult
                ),
                MBConvConfig(
                    6, 3, 1, 192, 320, 1, width_mult=width_mult, depth_mult=depth_mult
                ),
            ]
            last_channel = None
        elif inverted_residual_setting == "v2_s":
            inverted_residual_setting = [
                FusedMBConvConfig(1, 3, 1, 24, 24, 2),
                FusedMBConvConfig(4, 3, 2, 24, 48, 4),
                FusedMBConvConfig(4, 3, 2, 48, 64, 4),
                MBConvConfig(
                    4, 3, 2, 64, 128, 6, width_mult=width_mult, depth_mult=depth_mult
                ),
                MBConvConfig(
                    6, 3, 1, 128, 160, 9, width_mult=width_mult, depth_mult=depth_mult
                ),
                MBConvConfig(
                    6, 3, 2, 160, 256, 15, width_mult=width_mult, depth_mult=depth_mult
                ),
            ]
            last_channel = 1280
        elif inverted_residual_setting == "v2_m":
            inverted_residual_setting = [
                FusedMBConvConfig(1, 3, 1, 24, 24, 3),
                FusedMBConvConfig(4, 3, 2, 24, 48, 5),
                FusedMBConvConfig(4, 3, 2, 48, 80, 5),
                MBConvConfig(
                    4, 3, 2, 80, 160, 7, width_mult=width_mult, depth_mult=depth_mult
                ),
                MBConvConfig(
                    6, 3, 1, 160, 176, 14, width_mult=width_mult, depth_mult=depth_mult
                ),
                MBConvConfig(
                    6, 3, 2, 176, 304, 18, width_mult=width_mult, depth_mult=depth_mult
                ),
                MBConvConfig(
                    6, 3, 1, 304, 512, 5, width_mult=width_mult, depth_mult=depth_mult
                ),
            ]
            last_channel = 1280
        elif inverted_residual_setting == "v2_l":
            inverted_residual_setting = [
                FusedMBConvConfig(1, 3, 1, 32, 32, 4),
                FusedMBConvConfig(4, 3, 2, 32, 64, 7),
                FusedMBConvConfig(4, 3, 2, 64, 96, 7),
                MBConvConfig(
                    4, 3, 2, 96, 192, 10, width_mult=width_mult, depth_mult=depth_mult
                ),
                MBConvConfig(
                    6, 3, 1, 192, 224, 19, width_mult=width_mult, depth_mult=depth_mult
                ),
                MBConvConfig(
                    6, 3, 2, 224, 384, 25, width_mult=width_mult, depth_mult=depth_mult
                ),
                MBConvConfig(
                    6, 3, 1, 384, 640, 7, width_mult=width_mult, depth_mult=depth_mult
                ),
            ]
            last_channel = 1280
        else:
            raise ValueError(f"Unknown network structure: {inverted_residual_setting}")

        if "block" in kwargs:
            if kwargs["block"] is not None:
                for s in inverted_residual_setting:
                    if isinstance(s, MBConvConfig):
                        s.block = kwargs["block"]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            ConvNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = (
                    stochastic_depth_prob * float(stage_block_id) / total_stage_blocks
                )

                stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = (
            last_channel if last_channel is not None else 4 * lastconv_input_channels
        )
        layers.append(
            ConvNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
