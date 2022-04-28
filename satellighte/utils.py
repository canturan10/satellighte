from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


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
    # pylint: disable=no-member

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
            mode="bilinear",
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
    preds: List[torch.Tensor],
    label: List[str],
) -> List[Dict]:
    """
    Convert the predictions to a json format

    Args:
        preds (List[torch.Tensor]): List of torch.Tensor
        label (List[str]): List of labels

    Returns:
        List[Dict]: List of dictionaries with the label as key and the score as value
    """
    outputs = []
    for scores in preds.cpu().numpy().tolist():
        output = {
            label: round(score, 2)
            for score, label in zip(
                scores,
                label,
            )
        }
        outputs.append(
            dict(sorted(output.items(), key=lambda item: item[1], reverse=True))
        )

    return outputs


def visualize(img: np.ndarray, preds: List[Dict]) -> Image:
    """
    Virtualize the image with the predictions

    Args:
        img (np.ndarray): 3 channel np.ndarray
        preds (List[Dict]): Predictions. Each prediction is a dictionary with the label as key and the score as value

    Returns:
        Image: 3 channel PIL Image that will be shown on screen
    """
    pil_img = Image.fromarray(img)

    old_size = pil_img.size
    desired_size = 512
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    pil_img = pil_img.resize(new_size, Image.ANTIALIAS)

    font = ImageFont.load_default()

    for pred in preds:
        max_key = max(pred, key=pred.get)
        model_info = f" {max_key} : {pred[max_key]}"

    text_width, text_height = font.getsize(model_info)
    margin = np.ceil(0.05 * text_height)

    ImageDraw.Draw(pil_img).text(
        (margin, text_height - margin),
        model_info,
        fill="white",
        font=font,
    )

    return pil_img
