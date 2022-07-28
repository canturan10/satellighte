import argparse
import os

import imageio as imageio
import numpy as np

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

    arg.add_argument(
        "--labels",
        "-l",
        type=str,
        required=True,
        help="Delimited list of labels",
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

    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=args.model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], batch)

    interpreter.invoke()

    # get_tensor() returns a copy of the tensor data
    # use tensor() in order to get a pointer to the tensor
    preds = interpreter.get_tensor(output_details[0]["index"])

    labels = np.asarray([str(item) for item in args.labels.split(",")])

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
