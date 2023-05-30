from typing import List

from torch import nn

from .layers import MBConv, Transformer, conv_3x3_bn


class CoatNet(nn.Module):
    # pylint: disable=no-member

    def __init__(
        self,
        input_size: int,
        input_channel: int,
        number_of_blocks: List[int],
        hidden_dimension: List[int],
        block_types: List[str] = ["C", "C", "T", "T"],
        num_classes: int = 1000,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        block = {"C": MBConv, "T": Transformer}

        self.s0 = self._make_layer(
            conv_3x3_bn,
            input_channel,
            hidden_dimension[0],
            number_of_blocks[0],
            (input_size // 2, input_size // 2),
        )
        self.s1 = self._make_layer(
            block[block_types[0]],
            hidden_dimension[0],
            hidden_dimension[1],
            number_of_blocks[1],
            (input_size // 4, input_size // 4),
        )
        self.s2 = self._make_layer(
            block[block_types[1]],
            hidden_dimension[1],
            hidden_dimension[2],
            number_of_blocks[2],
            (input_size // 8, input_size // 8),
        )
        self.s3 = self._make_layer(
            block[block_types[2]],
            hidden_dimension[2],
            hidden_dimension[3],
            number_of_blocks[3],
            (input_size // 16, input_size // 16),
        )
        self.s4 = self._make_layer(
            block[block_types[3]],
            hidden_dimension[3],
            hidden_dimension[4],
            number_of_blocks[4],
            (input_size // 32, input_size // 32),
        )

        self.pool = nn.AvgPool2d(input_size // 32, 1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dimension[-1], num_classes),
        )

    def _make_layer(
        self,
        block,
        inp,
        oup,
        depth,
        image_size,
    ) -> nn.Sequential:
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size, downsample=True))
            else:
                layers.append(block(oup, oup, image_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)

        x = self.pool(x).view(-1, x.shape[1])
        x = self.classifier(x)

        return x
