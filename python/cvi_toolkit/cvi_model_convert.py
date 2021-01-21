#!/usr/bin/env python3

import sys
import onnx
import argparse
from cvi_toolkit.transform.onnx_converter import OnnxConverter
from cvi_toolkit.transform.tflite_converter_int8 import TFLiteConverter as TFLiteInt8Converter
from cvi_toolkit.transform.tensorflow_converter import TFConverter
from cvi_toolkit.transform.caffe_converter import CaffeConverter
from cvi_toolkit.utils.log_setting import setup_logger
from cvi_toolkit.data.preprocess import add_preprocess_parser, preprocess

logger = setup_logger('root', log_level="INFO")
CVI_SupportFramework = [
    "caffe",
    "onnx",
    "tflite_int8",
    "tensorflow",
]


def Convert(args):
    if args.model_type not in CVI_SupportFramework:
        raise ValueError("Not support {} type".format(args.model_type))
    preprocessor = preprocess()
    preprocessor.config(**vars(args))

    if args.model_type == "onnx":
        onnx_model = onnx.load(args.model_path)
        c = OnnxConverter(args.model_name, onnx_model,
                          args.mlir_file_path, batch_size=args.batch_size,
                          preprocess_args=preprocessor.to_dict())
    elif args.model_type == "tflite_int8":
        c = TFLiteInt8Converter(args.model_name,
                                args.model_path, args.mlir_file_path,
                                preprocess_args=preprocessor.to_dict())
    elif args.model_type == "tensorflow":
        c = TFConverter(args.model_name, args.model_path,
                        args.mlir_file_path, batch_size=args.batch_size,
                        preprocess_args=preprocessor.to_dict())
    elif args.model_type == "caffe":
        c = CaffeConverter(args.model_name, args.model_path, args.model_dat,
                           args.mlir_file_path, batch_size=args.batch_size,
                           preprocess_args=preprocessor.to_dict())
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
        "--model_dat",
        help="model weights, only caffemodel need",
        default=""
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
    parser.add_argument(
        "--input_shape",
        help="""Only fused aspect ratio preprocess need set this config,
                Fusing aspect ratio preprocess need to caculate padding size with each input shape,
                can ignore if don't use fusing aspect ratio preprocess
        """,
        type=str,
        default="0,0,0,0"
    )
    parser = add_preprocess_parser(parser)
    args = parser.parse_args()
    Convert(args)


if __name__ == "__main__":
    print(sys.argv)
    main()
