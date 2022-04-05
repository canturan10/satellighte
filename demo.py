import argparse

import imageio
import torch
import os
import satellighte as sat
import time


def parse_arguments():
    """
    Parse command line arguments.

    Returns: Parsed arguments
    """
    arg = argparse.ArgumentParser()

    arg.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
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
        help="Path to the image file or directory",
    )
    return arg.parse_args()


def main(args):
    """
    Main function.

    Args:
        args : Parsed arguments
    """
    model = sat.Classifier.from_pretrained(args.model_name)
    model.eval()
    model.to(args.device)
    # print(model.summarize(max_depth=1))

    if os.path.isdir(args.source):
        for file in os.listdir(args.source):
            file_path = os.path.join(args.source, file)
            if os.path.isfile(file_path):
                img = imageio.imread(file_path)
                results = model.predict(img)
                pil_img = sat.utils.visualize(img, results)
                pil_img.show()
                print(results)
                time.sleep(1)
    else:
        if os.path.isfile(args.source):
            img = imageio.imread(args.source)
            results = model.predict(img)
            pil_img = sat.utils.visualize(img, results)
            pil_img.show()
            print(results)


if __name__ == "__main__":
    pa = parse_arguments()
    main(pa)
