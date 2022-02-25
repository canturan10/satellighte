import os
import tempfile
import torch
import argparse
import onnx

import torch

import satellighte as sat


def parse_arguments():
    """
    Parse command line arguments.

    Returns: Parsed arguments
    """
    arg = argparse.ArgumentParser()
    arg.add_argument(
        "--model_name",
        type=str,
        default=sat.available_models()[0],
        choices=sat.available_models(),
        help="Model architecture",
    )
    arg.add_argument(
        "--version",
        type=str,
        default=sat.get_model_latest_version(sat.available_models()[0]),
        choices=sat.get_model_latest_version(sat.available_models()[0]),
        help="Model architecture",
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
    arg.add_argument(
        "--opset-version",
        type=int,
        default=11,
        help="Onnx opset version",
    )
    return arg.parse_args()


def main(args):
    """
    Main function.

    Args:
        args : Parsed arguments
    """
    # pylint: disable=no-member

    model = sat.Classifier.from_pretrained(
        args.model_name,
        version=args.version,
    )
    model.eval()

    if args.target:
        target_path = args.target
    else:
        target_path = os.path.join(
            sat.core._get_model_dir(),
            args.model_name,
            args.version,
        )

    print(target_path)
    dynamic_axes = {
        "input_data": {0: "batch", 2: "height", 3: "width"},  # write axis names
        "preds": {0: "batch"},
    }
    input_names = ["input_data"]
    output_names = ["preds"]

    input_sample = torch.rand(1, 3, model.input_size, model.input_size)

    if args.quantize:

        try:
            from onnxruntime.quantization import quantize_qat
        except ImportError:
            raise AssertionError("run `pip install onnxruntime`")

        target_model_path = os.path.join(
            target_path,
            "{}_quantize.onnx".format(args.model_name),
        )
        with tempfile.NamedTemporaryFile(suffix=".onnx") as temp:
            model.to_onnx(
                temp.name,
                input_sample=input_sample,
                opset_version=args.opset_version,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                export_params=True,
            )
            quantize_qat(temp.name, target_model_path)
    else:
        target_model_path = os.path.join(
            target_path,
            "{}.onnx".format(args.model_name),
        )
        model.to_onnx(
            target_model_path,
            input_sample=input_sample,
            opset_version=args.opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_params=True,
        )

    onnx_model = onnx.load(target_model_path)
    meta = onnx_model.metadata_props.add()
    meta.key = "labels"
    meta.value = "\n".join(model.labels)
    onnx.save(onnx_model, target_model_path)


if __name__ == "__main__":
    pa = parse_arguments()
    main(pa)
