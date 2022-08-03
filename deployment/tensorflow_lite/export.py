import argparse
import os

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
        "--target",
        "-t",
        type=str,
        help="Target path to save the model",
    )
    arg.add_argument(
        "--quantize",
        "-q",
        action="store_true",
        help="Quantize the model",
    )

    return arg.parse_args()


def main(args):
    """
    Main function.

    Args:
        args : Parsed arguments
    """
    # pylint: disable=no-member

    converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(
        args.model_path,
        input_shapes={"serving_default_input_data": [1, 3, 64, 64]},
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if args.quantize:
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_model = converter.convert()

    if args.target:
        target_path = args.target
    else:
        target_path = f"{args.model_path}.tflite"

    if args.quantize:
        target_path = "quantized_" + target_path

    print(f"Target Path: {target_path}")
    open(target_path, "wb").write(tflite_model)
    print("Model saved")


if __name__ == "__main__":
    pa = parse_arguments()
    main(pa)
