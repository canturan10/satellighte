import argparse
import os

import imageio as imageio
import numpy as np
import onnxruntime as ort

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf


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

    model = tf.saved_model.load(args.model_path)
    model.trainable = False

    input_tensor = tf.convert_to_tensor(batch)
    preds = model(**{"input_data": input_tensor})

    # Just getting the labels from the saved model, no need this if you have labels already
    sess = ort.InferenceSession(f"{args.model_path[:-11]}.onnx")
    meta = sess.get_modelmeta()
    labels = meta.custom_metadata_map["labels"].split("\n")

    outputs = []
    for x in range(len(preds)):
        output = {
            label: round(score, 2)
            for score, label in zip(
                preds["preds"][x].numpy(),
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
