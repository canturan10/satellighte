import argparse

import imageio
import torch

import satellighte as sat


def parse_arguments():
    """
    Parse command line arguments.

    Returns: Parsed arguments
    """
    arg = argparse.ArgumentParser()

    arg.add_argument(
        "--device",
        type=int,
        default=1 if torch.cuda.is_available() else 0,
        choices=[0, 1],
        help="GPU device to use",
    )
    arg.add_argument(
        "--model_name",
        type=str,
        default=sat.available_models()[0],
        choices=sat.available_models(),
        help="Model architecture",
    )
    arg.add_argument(
        "--source",
        "-s",
        type=str,
        required=True,
        help="Path to the image file",
    )
    return arg.parse_args()


def main(args):
    """
    Main function.

    Args:
        args : Parsed arguments
    """
    model = sat.Classifier.from_pretrained("mobilenetv2_default_eurosat")
    model.eval()
    img = imageio.imread(args.source)
    results = model.predict(img)
    pil_img = sat.utils.visualize(img, results)
    pil_img.show()
    print(results)


if __name__ == "__main__":
    pa = parse_arguments()
    main(pa)