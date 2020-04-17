#!/usr/bin/env python3
import onnx
import argparse
from cvi_toolkit.transform.onnx_converter import OnnxConverter
from cvi_toolkit.utils.log_setting import setup_logger

logger = setup_logger('root', log_level="INFO")
CVI_SupportFramework = [
    "onnx"
]

def Convert(args):
    if args.model_type not in CVI_SupportFramework:
        raise ValueError("Not support {} type".format(args.model_type))
    if args.model_type == "onnx":
        onnx_model = onnx.load(args.model_path)
        c = OnnxConverter(args.model_name, onnx_model, args.mlir_file_path)
        c.run()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        help="model path."
    )
    parser.add_argument(
        "--model_name",
        help="model name."
    )
    parser.add_argument(
        "--model_type",
        help="model, type ex. {}".format(CVI_SupportFramework)
    )
    parser.add_argument(
        "--mlir_file_path",
        help="path to store mlir file",
        default=""
    )
    args = parser.parse_args()
    Convert(args)


if __name__ == "__main__":
    main()
