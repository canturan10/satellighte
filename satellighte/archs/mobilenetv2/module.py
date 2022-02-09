import torch.nn as nn
import torch
import torchvision
import os
from typing import Dict, List
from .blocks.mobilenetv2 import MobileNet_V2


class MobileNetV2(nn.Module):
    """
    Implementation of MobileNetV2: Inverted Residuals and Linear Bottlenecks

    The network is the one described in arxiv.org/abs/1801.04381v4 .
    """

    __CONFIGS__ = {
        "default": {
            "input": {
                "input_size": 224,
                "normalized_input": True,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "model": {
                "width_mult": 1.0,
                "inverted_residual_setting": None,
                "round_nearest": 8,
                "block": None,
                "norm_layer": None,
                "dropout": 0.2,
                "pretrained": False,
            },
        },
    }

    def __init__(
        self,
        config: Dict,
        labels: List[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.labels = labels
        self.num_classes = len(self.labels)

        self.backbone = MobileNet_V2(
            num_classes=self.num_classes,
            width_mult=self.config["model"]["width_mult"],
            inverted_residual_setting=self.config["model"]["inverted_residual_setting"],
            round_nearest=self.config["model"]["round_nearest"],
            block=self.config["model"]["block"],
            norm_layer=self.config["model"]["norm_layer"],
            dropout=self.config["model"]["dropout"],
        )

        if self.config["model"]["pretrained"]:
            self.backbone.load_state_dict(
                torchvision.models.mobilenet_v2(pretrained=True).state_dict(),
                # strict=False,
            )

    def forward(self, inputs):
        return self.backbone(inputs)

    def logits_to_preds(self, logits: List[torch.Tensor]):
        return torch.softmax(logits, axis=-1)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: str = "",
        *args,
        **kwargs,
    ) -> nn.Module:

        *_, full_model_name, _ = model_path.split(os.path.sep)
        st = torch.load(os.path.join(model_path, f"{full_model_name}.pt"))
        with open(os.path.join(model_path, "labels.txt"), "r") as foo:
            labels = [line.rstrip("\n") for line in foo]

        model = cls(
            config=MobileNetV2.__CONFIGS__[config], labels=labels, *args, **kwargs
        )
        model.load_state_dict(st, strict=False)
        return model

    def compute_loss(
        self,
        batch: torch.Tensor,
        logits: List[torch.Tensor],
        targets: List,
        hparams: Dict = {},
    ):
        pass

    def configure_optimizers(self, hparams: Dict):
        pass
