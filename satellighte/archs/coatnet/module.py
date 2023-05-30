import os
from typing import Dict, List

import torch
import torch.nn as nn

from .blocks.coatnet import CoatNet


class CoAtNet(nn.Module):
    """
    Implementation of CoAtNet: Marrying Convolution and Attention for All Data Sizes

    The network is the one described in arxiv.org/abs/2106.04803v2 .
    """

    # pylint: disable=no-member

    __CONFIGS__ = {
        "0": {
            "input": {
                "input_size": 224,
                "input_channel": 3,
                "normalized_input": True,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "model": {
                "number_of_blocks": [2, 2, 3, 5, 2],
                "hidden_dimension": [64, 96, 192, 384, 768],
                "dropout": 0.3,
                "pretrained": False,
            },
        },
        "1": {
            "input": {
                "input_size": 224,
                "input_channel": 3,
                "normalized_input": True,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "model": {
                "number_of_blocks": [2, 2, 6, 14, 2],
                "hidden_dimension": [64, 96, 192, 384, 768],
                "dropout": 0.3,
                "pretrained": False,
            },
        },
        "2": {
            "input": {
                "input_size": 224,
                "input_channel": 3,
                "normalized_input": True,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "model": {
                "number_of_blocks": [2, 2, 6, 14, 2],
                "hidden_dimension": [128, 128, 256, 512, 1026],
                "dropout": 0.3,
                "pretrained": False,
            },
        },
        "3": {
            "input": {
                "input_size": 224,
                "input_channel": 3,
                "normalized_input": True,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "model": {
                "number_of_blocks": [2, 2, 6, 14, 2],
                "hidden_dimension": [192, 192, 384, 768, 1536],
                "dropout": 0.3,
                "pretrained": False,
            },
        },
        "4": {
            "input": {
                "input_size": 224,
                "input_channel": 3,
                "normalized_input": True,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "model": {
                "number_of_blocks": [2, 2, 12, 28, 2],
                "hidden_dimension": [192, 192, 384, 768, 1536],
                "dropout": 0.3,
                "pretrained": False,
            },
        },
    }

    def __init__(
        self,
        config: Dict,
        labels: List[str] = None,
    ):
        super().__init__()
        self.config = config
        self.labels = labels
        self.num_classes = len(self.labels)

        self.backbone = CoatNet(
            input_size=self.config["input"]["input_size"],
            input_channel=self.config["input"]["input_channel"],
            number_of_blocks=self.config["model"]["number_of_blocks"],
            hidden_dimension=self.config["model"]["hidden_dimension"],
            dropout=self.config["model"]["dropout"],
        )

    def forward(self, inputs):
        """
        Forward pass of the model.
        """
        return self.backbone(inputs)

    def logits_to_preds(self, logits: List[torch.Tensor]):
        """
        Convert logits to predictions.
        """
        return torch.softmax(logits, axis=-1)

    @classmethod
    def build(
        cls,
        config: str = "",
        labels: List[str] = None,
        **kwargs,
    ) -> nn.Module:
        """
        Build the model with random weights.

        Args:
            config (str, optional): Configuration name. Defaults to "".
            labels (List[str], optional): List of labels. Defaults to None.

        Returns:
            nn.Module: Model with random weights.
        """
        # return model with random weight initialization
        labels = ["cls1", "cls2"] if labels is None else labels

        return cls(
            config=CoAtNet.__CONFIGS__[config],
            labels=labels,
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: str,
        *args,
        **kwargs,
    ) -> nn.Module:
        """
        Load a model from a pre-trained model.

        Args:
            model_path (str): Path to the pre-trained model
            config (str): Configuration of the model

        Returns:
            nn.Module: Model with pretrained weights
        """

        *_, full_model_name, _ = model_path.split(os.path.sep)

        s_dict = torch.load(
            os.path.join(model_path, f"{full_model_name}.pth"), map_location="cpu"
        )

        model = cls(
            config=CoAtNet.__CONFIGS__[config],
            labels=s_dict["labels"],
            *args,
            **kwargs,
        )

        model.load_state_dict(s_dict["state_dict"], strict=True)

        return model

    def compute_loss(
        self,
        logits: List[torch.Tensor],
        targets: List,
        hparams: Dict,
    ):
        """
        Compute the loss for the model.

        Args:
            logits (List[torch.Tensor]): _description_
            targets (List): _description_
            hparams (Dict, optional): _description_. Defaults to {}.

        Raises:
            ValueError: Unknown criterion

        Returns:  Loss
        """
        if hparams.get("criterion", "cross_entropy") == "cross_entropy":
            loss_fcn = nn.CrossEntropyLoss()
        else:
            raise ValueError("Unknown criterion")

        return {"loss": loss_fcn(logits, targets)}

    def configure_optimizers(self, hparams: Dict):
        """
        Configure optimizers for the model.

        Args:
            hparams (Dict): Hyperparameters

        Raises:
            ValueError: Unknown optimizer
            ValueError: Unknown scheduler

        Returns: optimizers and scheduler
        """
        hparams_optimizer = hparams.get("optimizer", "sgd")
        if hparams_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=hparams.get("learning_rate", 1e-1),
                momentum=hparams.get("momentum", 0.9),
                weight_decay=hparams.get("weight_decay", 1e-5),
            )
        elif hparams_optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=hparams.get("learning_rate", 1e-1),
                betas=hparams.get("betas", (0.9, 0.999)),
                eps=hparams.get("eps", 1e-08),
                weight_decay=hparams.get("weight_decay", 1e-5),
            )
        elif hparams_optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=hparams.get("learning_rate", 1e-1),
                betas=hparams.get("betas", (0.9, 0.999)),
                eps=hparams.get("eps", 1e-08),
                weight_decay=hparams.get("weight_decay", 1e-5),
            )
        else:
            raise ValueError("Unknown optimizer")

        hparams_scheduler = hparams.get("scheduler", "steplr")
        if hparams_scheduler == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=hparams.get("step_size", 4),
                gamma=hparams.get("gamma", 0.5),
            )
        elif hparams_scheduler == "multisteplr":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                gamma=hparams.get("gamma", 0.5),
                milestones=hparams.get("milestones", [500000, 1000000, 1500000]),
            )
        else:
            raise ValueError("Unknown scheduler")

        return [optimizer], [scheduler]
