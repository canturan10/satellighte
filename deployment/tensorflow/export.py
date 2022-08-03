import argparse
import os

import onnx
from onnx_tf.backend import prepare

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


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
    return arg.parse_args()


def main(args):
    """
    Main function.

    Args:
        args : Parsed arguments
    """
    # pylint: disable=no-member

    if args.target:
        target_path = args.target
    else:
        filename, _ = os.path.splitext(args.model_path)
        target_path = f"{filename}_tensorflow"

    print(f"Target Path: {target_path}")
    onnx_model = onnx.load(args.model_path)  # load onnx model
    tf_rep = prepare(onnx_model)  # creating TensorflowRep object
    tf_rep.export_graph(target_path)  # exporting the graph to a protobuf file
    print("Model saved")


if __name__ == "__main__":
    pa = parse_arguments()
    main(pa)
