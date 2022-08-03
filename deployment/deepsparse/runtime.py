import argparse

import imageio as imageio
import numpy as np
import onnxruntime as ort

from deepsparse import compile_model


def parse_arguments():
    """
    Parse command line arguments.

    Returns: Parsed arguments
    """
    arg = argparse.ArgumentParser()
    arg.add_argument(
        "--model-path",
        "-m",
        type=str,
        required=True,
        help="Model path",
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
    img = imageio.imread(args.source)[:, :, :3]

    batch = np.expand_dims(np.transpose(img, (2, 0, 1)), 0).astype(np.float32)

    engine = compile_model(
        args.model_path,
        len(batch),
    )
    preds = engine.run([np.ascontiguousarray(batch, dtype=np.float32)])[0]

    sess = ort.InferenceSession(args.model_path)
    meta = sess.get_modelmeta()
    labels = meta.custom_metadata_map["labels"].split("\n")

    outputs = []
    for scores in preds.tolist():
        output = {
            label: round(score, 2)
            for score, label in zip(
                scores,
                labels,
            )
        }
        outputs.append(
            dict(sorted(output.items(), key=lambda item: item[1], reverse=True))
        )

    print(outputs)


if __name__ == "__main__":
    pa = parse_arguments()
    main(pa)
