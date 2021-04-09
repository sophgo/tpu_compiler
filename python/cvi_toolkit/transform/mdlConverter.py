#!/usr/bin/env python3

import sys
import onnx
import argparse
from cvi_toolkit.transform.onnx_converter import OnnxConverter
from cvi_toolkit.utils.log_setting import setup_logger

logger = setup_logger('root', log_level="INFO")


def Convert(args):
    onnx_model = onnx.load(args.model_path)
    c = OnnxConverter(args.model_name, onnx_model,
                      args.mlir_file_path, batch_size=args.batch_size
                     )
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
        "--mlir_file_path",
        help="path to store mlir file",
        default=""
    )
    parser.add_argument(
        "--batch_size",
        help="input batch size",
        type=int,
        default=1,
    )
    args = parser.parse_args()
    Convert(args)


if __name__ == "__main__":
    print(sys.argv)
    main()
