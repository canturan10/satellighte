from typing import Dict, List, Tuple, Union

import torch
import torch.nn.functional as F


def configure_batch(
    batch: List[torch.Tensor],
    target_size: int,
    adaptive_batch: bool = False,
) -> torch.Tensor:
    """
    Configure batch for the required size

    Args:
        batch (List[torch.Tensor]): List of torch.Tensor(C, H, W)
        target_size (int): Max dimension of the target image
        adaptive_batch (bool, optional): If true, the batch will be adaptive (target_size is the max dimension of the each image), otherwise it will use given `target_size`. Defaults to False.

    Returns:
        torch.Tensor: batched inputs as B x C x target_size x target_size
    """

    for i, img in enumerate(batch):

        # Check adaptive batch is given
        if adaptive_batch:
            # Get max dimension of the image
            target_size: int = max(img.size(1), img.size(2))

        # Apply interpolation to the image
        img_h: int = img.size(-2)
        img_w: int = img.size(-1)

        scale_factor: float = min(target_size / img_h, target_size / img_w)

        img = F.interpolate(
            img.unsqueeze(0),
            scale_factor=scale_factor,
            mode="nearest",
            recompute_scale_factor=False,
            align_corners=False,
        )

        new_h: int = img.size(-2)
        new_w: int = img.size(-1)

        # Apply padding to the image
        pad_left = (target_size - new_w) // 2
        pad_right = pad_left + (target_size - new_w) % 2

        pad_top = (target_size - new_h) // 2
        pad_bottom = pad_top + (target_size - new_h) % 2

        img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), value=0)

        batch[i] = img

    batch = torch.cat(batch, dim=0).contiguous()

    return batch


def convert_json(
    self,
    preds: List[torch.Tensor],
    label: List[str],
) -> List[Dict]:
    pass