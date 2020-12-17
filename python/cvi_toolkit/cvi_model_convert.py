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
    preprocessor.config(net_input_dims=args.net_input_dims,
                        resize_dims=args.image_resize_dims,
                        mean=args.mean,
                        mean_file=args.mean_file,
                        input_scale=args.input_scale,
                        raw_scale=args.raw_scale,
                        std=args.std,
                        pixel_format=args.pixel_format,
                        rgb_order=args.model_channel_order,
                        crop_method=args.crop_method,
                        data_format=args.data_format,
                        only_aspect_ratio_img=args.only_aspect_ratio_img,
                        bgray=args.bgray)

    input_shape = [int(i) for i in args.input_shape.split(",")]

    if args.model_type == "onnx":
        onnx_model = onnx.load(args.model_path)
        c = OnnxConverter(args.model_name, onnx_model,
                          args.mlir_file_path, batch_size=args.batch_size,
                          convert_preprocess=args.convert_preprocess, preprocess_args=preprocessor.to_dict(
                              input_h=input_shape[1], input_w=input_shape[2], preprocess_input_data_format=args.preprocess_input_data_format),
                          )
    elif args.model_type == "tflite_int8":
        c = TFLiteInt8Converter(
            args.model_name, args.model_path, args.mlir_file_path)
    elif args.model_type == "tensorflow":
        c = TFConverter(args.model_name, args.model_path,
                        args.mlir_file_path, batch_size=args.batch_size)
    elif args.model_type == "caffe":
        c = CaffeConverter(args.model_name, args.model_path, args.model_dat,
                           args.mlir_file_path, batch_size=args.batch_size,
                           convert_preprocess=args.convert_preprocess, preprocess_args=preprocessor.to_dict(
                               input_h=input_shape[1], input_w=input_shape[2], preprocess_input_data_format=args.preprocess_input_data_format)
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
        "--pixel_format",
        help="""Pixel format, default is RGB mode, user can set to YUV420.
                If YUV420 used, input shape will be [N 6 h/2 w/2]
        """,
        type=str,
        default="RGB"
    )
    parser.add_argument(
        "--convert_preprocess",
        help="Conbine mlir model with preprocess inference",
        type=int,
        default=0,
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
    parser.add_argument(
        "--preprocess_input_data_format",
        help="""
                Fusing preprocess input data_format,
        """,
        type=str,
        default="nhwc"
    )
    parser = add_preprocess_parser(parser)
    args = parser.parse_args()
    Convert(args)


if __name__ == "__main__":
    print(sys.argv)
    main()
