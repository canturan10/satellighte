import os
from typing import Dict, List
from functools import partial

import torch
import torch.nn as nn

from .blocks.efficientnet import Efficient_Net


class EfficientNet(nn.Module):
    """
    Implementation of EfficientNet: Deep Residual Learning for Image Recognition

    The network is the one described in arxiv.org/abs/1512.03385 .
    """

    # pylint: disable=no-member

    __CONFIGS__ = {
        "b0": {
            "input": {
                "input_size": 224,
                "normalized_input": True,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "model": {
                "inverted_residual_setting": "v1",
                "compound_coef": 0,
                "dropout": 0.2,
                "width_mult": 1.0,
                "depth_mult": 1.0,
                "pretrained": False,
                "norm_layer": {
                    "eps": 1e-5,
                    "momentum": 0.1,
                },
            },
        },
        "b1": {
            "input": {
                "input_size": 240,
                "normalized_input": True,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "model": {
                "inverted_residual_setting": "v1",
                "dropout": 0.25,
                "width_mult": 1.0,
                "depth_mult": 1.1,
                "pretrained": False,
                "norm_layer": {
                    "eps": 1e-5,
                    "momentum": 0.1,
                },
            },
        },
        "b2": {
            "input": {
                "input_size": 288,
                "normalized_input": True,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "model": {
                "inverted_residual_setting": "v1",
                "dropout": 0.3,
                "width_mult": 1.1,
                "depth_mult": 1.2,
                "pretrained": False,
                "norm_layer": {
                    "eps": 1e-5,
                    "momentum": 0.1,
                },
            },
        },
        "b3": {
            "input": {
                "input_size": 300,
                "normalized_input": True,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "model": {
                "inverted_residual_setting": "v1",
                "dropout": 0.3,
                "width_mult": 1.2,
                "depth_mult": 1.4,
                "pretrained": False,
                "norm_layer": {
                    "eps": 1e-5,
                    "momentum": 0.1,
                },
            },
        },
        "b4": {
            "input": {
                "input_size": 380,
                "normalized_input": True,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "model": {
                "inverted_residual_setting": "v1",
                "dropout": 0.4,
                "width_mult": 1.4,
                "depth_mult": 1.8,
                "pretrained": False,
                "norm_layer": {
                    "eps": 1e-5,
                    "momentum": 0.1,
                },
            },
        },
        "b5": {
            "input": {
                "input_size": 456,
                "normalized_input": True,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "model": {
                "inverted_residual_setting": "v1",
                "dropout": 0.4,
                "width_mult": 1.6,
                "depth_mult": 2.2,
                "pretrained": False,
                "norm_layer": {
                    "eps": 0.001,
                    "momentum": 0.01,
                },
            },
        },
        "b6": {
            "input": {
                "input_size": 528,
                "normalized_input": True,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "model": {
                "inverted_residual_setting": "v1",
                "dropout": 0.5,
                "width_mult": 1.8,
                "depth_mult": 2.6,
                "pretrained": False,
                "norm_layer": {
                    "eps": 0.001,
                    "momentum": 0.01,
                },
            },
        },
        "b7": {
            "input": {
                "input_size": 600,
                "normalized_input": True,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "model": {
                "inverted_residual_setting": "v1",
                "dropout": 0.5,
                "width_mult": 2.0,
                "depth_mult": 3.1,
                "pretrained": False,
                "norm_layer": {
                    "eps": 0.001,
                    "momentum": 0.01,
                },
            },
        },
        "v2-s": {
            "input": {
                "input_size": 384,
                "normalized_input": True,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "model": {
                "inverted_residual_setting": "v2_s",
                "dropout": 0.2,
                "width_mult": 1.0,
                "depth_mult": 1.0,
                "pretrained": False,
                "norm_layer": {
                    "eps": 1e-3,
                    "momentum": 0.1,
                },
            },
        },
        "v2-m": {
            "input": {
                "input_size": 480,
                "normalized_input": True,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "model": {
                "inverted_residual_setting": "v2_m",
                "dropout": 0.3,
                "width_mult": 1.0,
                "depth_mult": 1.0,
                "pretrained": False,
                "norm_layer": {
                    "eps": 1e-3,
                    "momentum": 0.1,
                },
            },
        },
        "v2-l": {
            "input": {
                "input_size": 480,
                "normalized_input": True,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "model": {
                "inverted_residual_setting": "v2_l",
                "dropout": 0.4,
                "width_mult": 1.0,
                "depth_mult": 1.0,
                "pretrained": False,
                "norm_layer": {
                    "eps": 1e-3,
                    "momentum": 0.1,
                },
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

        self.norm_layer = partial(
            nn.BatchNorm2d,
            eps=self.config["model"]["norm_layer"]["eps"],
            momentum=self.config["model"]["norm_layer"]["momentum"],
        )

        self.backbone = Efficient_Net(
            inverted_residual_setting=self.config["model"]["inverted_residual_setting"],
            dropout=self.config["model"]["dropout"],
            stochastic_depth_prob=0.2,
            num_classes=self.num_classes,
            width_mult=self.config["model"]["width_mult"],
            depth_mult=self.config["model"]["depth_mult"],
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
            config=EfficientNet.__CONFIGS__[config],
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
            config=EfficientNet.__CONFIGS__[config],
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
