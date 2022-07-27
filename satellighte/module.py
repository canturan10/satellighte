import os
from typing import Dict, List, Union

import numpy as np
import pytorch_lightning as pl
import torch

from . import api
from .core import _get_arch_cls, _get_model_dir, _parse_saved_model_name
from .utils import configure_batch, convert_json
import torchmetrics as tm


class Classifier(pl.LightningModule):
    """Generic pl.LightningModule definition for image classification"""

    # pylint: disable=no-member
    # pylint: disable=not-callable

    def __init__(
        self,
        model: torch.nn.Module,
        hparams: Dict = None,
    ):
        super().__init__()
        self.model = model
        self.__metrics = {}

        self.save_hyperparameters(hparams)
        self.configure_preprocess()

    @property
    def input_size(self):
        return self.model.config.get("input").get("input_size")

    @property
    def labels(self):
        return self.model.labels

    # WARNING: This function should only be used during training. not inference
    def forward(
        self,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            batch (torch.Tensor): Batch of tensors of shape (B x C x H x W).

        Returns:
            torch.Tensor: Prediction of the model.
        """

        # Apply preprocess with the help of registered buffer
        batch = ((batch / self.normalizer) - self.mean) / self.std

        with torch.no_grad():
            # Get logits from the model
            logits = self.model.forward(batch)

        # Apply postprocess for the logits that are returned from model and get predictions
        preds = self.model.logits_to_preds(logits)

        return preds

    @torch.jit.unused
    def predict(
        self,
        data: Union[np.ndarray, List],
        target_size: int = None,
    ):
        """
        Perform image classification using given image or images.

        Args:
            data (Union[np.ndarray, List]): Numpy array or list of numpy arrays. In the form of RGB.
            target_size (int, optional): If it is not None, the image will be resized to the target size. Defaults to None.

        Returns:
            [type]: [description]
        """

        # Converts given image or list of images to list of tensors
        batch = self.to_tensor(data)

        # Override target_size if input_size is given and target_size is None
        if self.input_size and (target_size is None):
            target_size = self.input_size

        # Configure batch for the required size
        batch = configure_batch(
            batch,
            target_size=target_size,
            adaptive_batch=target_size is None,
        )

        # Get predictions from the model
        preds = self.forward(batch)

        # Convert predictions to json format
        json_preds = convert_json(preds, self.labels)
        return json_preds

    @classmethod
    def build(
        cls,
        arch: str,
        config: str = None,
        hparams: Dict = {},
        **kwargs,
    ) -> pl.LightningModule:
        """
        Build the model with given architecture and configuration.

        Args:
            arch (str): Model architecture name.
            config (str, optional): Model configuration. Defaults to None.
            hparams (Dict, optional): Hyperparameters. Defaults to {}.

        Returns:
            pl.LightningModule: Model instance with randomly initialized weights.
        """

        model = cls.build_arch(arch, config=config, **kwargs)
        return cls(model, hparams=hparams)

    @classmethod
    def build_arch(
        cls,
        arch: str,
        config: str = None,
        **kwargs,
    ) -> torch.nn.Module:
        """
        Build the architecture model with given configuration.

        Args:
            arch (str): Model architecture name.
            config (str, optional): Model configuration. Defaults to None.

        Returns:
            torch.nn.Module: Architecture model instance with randomly initialized weights.
        """

        arch_cls = _get_arch_cls(arch)
        return arch_cls.build(config=config, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        version: int = None,
        hparams: Dict = {},
    ) -> pl.LightningModule:
        """
        [summary]

        Args:
            model_name (str): Model name in the format of {arch}_{config}_{dataset}
            version (int, optional): Model version. Defaults to None.
            hparams (Dict, optional): Hyperparameters. Defaults to {}.

        Returns:
            pl.LightningModule: Model instance.
        """

        model = cls.from_pretrained_arch(model_name, version=version)
        return cls(model, hparams=hparams)

    @classmethod
    def from_pretrained_arch(
        cls,
        model_name: str,
        version: int = None,
    ) -> torch.nn.Module:
        """
        Get pretrained arch model from the model name.

        Args:
            model_name (str): Model name in the format of {arch}_{config}_{dataset}
            version (int, optional): Model version. Defaults to None.

        Returns:
            torch.nn.Module: Architecture model instance.
        """

        # Check if version is not given then get the latest version
        if not version:
            version = api.get_model_latest_version(model_name)

        # Get arch name and config name from the given model_name
        arch, config, _ = _parse_saved_model_name(model_name)

        # Get arch class
        arch_cls = _get_arch_cls(arch)

        api.get_saved_model(model_name, version)

        # Get pretrained model pat
        model_path = os.path.join(_get_model_dir(), model_name, f"v{version}")

        return arch_cls.from_pretrained(model_path, config=config)

    def training_step(self, batch, batch_idx):
        batch, targets = batch

        # Apply preprocess with the help of registered buffer
        batch = ((batch / self.normalizer) - self.mean) / self.std

        # Get logits from the model
        logits = self.model.forward(batch)

        # Compute loss
        loss = self.model.compute_loss(
            logits,
            targets,
            hparams=self.hparams,
        )

        return loss

    def training_epoch_end(self, outputs):
        losses = {}
        for output in outputs:
            if isinstance(output, dict):
                for k, v in output.items():
                    if k not in losses:
                        losses[k] = []
                    losses[k].append(v)
            else:
                if "loss" not in losses:
                    losses["loss"] = []
                losses["loss"].append(output)

        for name, loss in losses.items():
            self.log("{}/training".format(name), sum(loss) / len(loss))

    def on_validation_epoch_start(self):
        for metric in self.__metrics.values():
            metric.reset()

    def validation_step(self, batch, batch_idx):
        batch, targets = batch

        # Apply preprocess with the help of registered buffer
        batch = ((batch / self.normalizer) - self.mean) / self.std

        with torch.no_grad():
            # Get logits from the model
            logits = self.model.forward(batch)

            # Compute loss
            loss = self.model.compute_loss(
                logits,
                targets,
                hparams=self.hparams,
            )

        # Apply postprocess for the logits that are returned from model and get predictions
        preds = self.model.logits_to_preds(logits)

        for metric in self.__metrics.values():
            metric.update(preds.cpu(), targets.cpu())

        return loss

    def validation_epoch_end(self, outputs):
        losses = {}
        for output in outputs:
            if isinstance(output, dict):
                for k, v in output.items():
                    if k not in losses:
                        losses[k] = []
                    losses[k].append(v)
            else:
                if "loss" not in losses:
                    losses["loss"] = []
                losses["loss"].append(output)

        for name, loss in losses.items():
            self.log("{}/validation".format(name), sum(loss) / len(loss))

        for name, metric in self.__metrics.items():
            self.log(
                "metrics/{}".format(name),
                metric.compute(),
                prog_bar=True,
            )

    def on_test_epoch_start(self):
        for metric in self.__metrics.values():
            metric.reset()

    def test_step(self, batch, batch_idx):
        batch, targets = batch

        # Apply preprocess with the help of registered buffer
        batch = ((batch / self.normalizer) - self.mean) / self.std

        with torch.no_grad():
            # Get logits from the model
            logits = self.model.forward(batch)

            # Compute loss
            loss = self.model.compute_loss(
                logits,
                targets,
                hparams=self.hparams,
            )

        # Apply postprocess for the logits that are returned from model and get predictions
        preds = self.model.logits_to_preds(logits)

        for metric in self.__metrics.values():
            metric.update(preds.cpu(), targets.cpu())
        return loss

    def test_epoch_end(self, outputs):
        metric_results = {}
        for name, metric in self.__metrics.items():
            metric_results[name] = metric.compute()

        for name, metric in self.__metrics.items():
            self.log(
                "metrics/{}".format(name),
                metric.compute(),
                prog_bar=True,
            )
        return metric_results

    def configure_optimizers(self):
        return self.model.configure_optimizers(hparams=self.hparams)

    def configure_preprocess(self):
        """
        Configure preprocess for the model.
        """

        # Get information from config of model
        normalized_input = self.model.config.get("input", {}).get(
            "normalized_input", True
        )
        mean = self.model.config.get("input", {}).get("mean", 0.0)
        std = self.model.config.get("input", {}).get("std", 1.0)

        # Check dimension of std and mean
        if isinstance(mean, list):
            assert len(mean) == 3, "mean dimension must be 3 not {}".format(len(mean))
            mean = [float(m) for m in mean]
        else:
            mean = [float(mean) for _ in range(3)]

        if isinstance(std, list):
            assert len(std) == 3, "std dimension must be 3 not {}".format(len(std))
            std = [float(m) for m in std]
        else:
            std = [float(std) for _ in range(3)]

        # Register the tensors as a buffer
        # Now we can access self.normalizer anywhere in the module
        self.register_buffer(
            "normalizer",
            torch.tensor(255.0) if normalized_input else torch.tensor(1.0),
            persistent=False,
        )

        # Now we can access self.mean anywhere in the module
        self.register_buffer(
            "mean",
            torch.tensor(mean).view(-1, 1, 1).contiguous(),
            persistent=False,
        )

        # Now we can access self.std anywhere in the module
        self.register_buffer(
            "std",
            torch.tensor(std).view(-1, 1, 1).contiguous(),
            persistent=False,
        )

    def add_metric(self, name: str, metric: tm.Metric):
        """Adds given metric with name key

        Args:
                name (str): name of the metric
                metric (tm.Metric): Metric object
        """
        # TODO add warnings if override happens
        self.__metrics[name] = metric

    def get_metrics(self) -> Dict[str, tm.Metric]:
        """Return metrics defined in the `FaceDetector` instance

        Returns:
                Dict[str, tm.Metric]: defined model metrics with names
        """
        return {k: v for k, v in self.__metrics.items()}

    def to_tensor(self, images: Union[np.ndarray, List]) -> List[torch.Tensor]:
        """Converts given image or list of images to list of tensors
        Args:
                images (Union[np.ndarray, List]): RGB image or list of RGB images
        Returns:
                List[torch.Tensor]: list of torch.Tensor(C x H x W)

        This method is taken from fastface repositories.
        `fastface.module.to_tensor`
        Here is a link for it: `github.com/borhanMorphy/fastface`
        """
        assert isinstance(
            images, (list, np.ndarray)
        ), "give images must be eather list of numpy arrays or numpy array"

        if isinstance(images, np.ndarray):
            images = [images]

        batch: List[torch.Tensor] = []

        for img in images:
            assert (
                len(img.shape) == 3
            ), "image shape must be channel, height\
                , with length of 3 but found {}".format(
                len(img.shape)
            )
            assert (
                img.shape[2] == 3
            ), "channel size of the image must be 3 but found {}".format(img.shape[2])

            batch.append(
                # h,w,c => c,h,w
                torch.tensor(img, dtype=self.dtype, device=self.device).permute(2, 0, 1)
            )

        return batch
