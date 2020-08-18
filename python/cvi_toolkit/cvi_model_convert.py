#!/usr/bin/env python3

import sys
import onnx
import argparse
from cvi_toolkit.transform.onnx_converter import OnnxConverter
from cvi_toolkit.transform.tflite_converter import TFLiteConverter
from cvi_toolkit.transform.tensorflow_converter import TFConverter
from cvi_toolkit.transform.caffe_converter import CaffeConverter
from cvi_toolkit.utils.log_setting import setup_logger
from cvi_toolkit.data.preprocess import add_preprocess_parser, preprocess

logger = setup_logger('root', log_level="INFO")
CVI_SupportFramework = [
    "caffe",
    "onnx",
    "tflite",
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
                        rgb_order=args.model_channel_order,
                        crop_method=args.crop_method,
                        only_aspect_ratio_img=args.only_aspect_ratio_img)
    # only caffe support
    preprocess_args = {
        "swap_channel": args.swap_channel,
        "raw_scale": args.raw_scale,
        "mean": args.mean,
        "scale": args.scale,
        "input_scale": args.input_scale,
        "std": args.std,
        "rgb_order": args.model_channel_order,
        "data_format": args.data_format,
    }

    input_shape = [int(i) for i in args.input_shape.split(",")]

    if args.model_type == "onnx":
        onnx_model = onnx.load(args.model_path)
        c = OnnxConverter(args.model_name, onnx_model,
                          args.mlir_file_path, batch_size=args.batch_size,
                          convert_preprocess=args.convert_preprocess, preprocess_args=preprocessor.to_dict(input_h=input_shape[1], input_w=input_shape[2])
                        )
    elif args.model_type == "tflite":
        c = TFLiteConverter(
            args.model_name, args.model_path, args.mlir_file_path)
    elif args.model_type == "tensorflow":
        c = TFConverter(args.model_name, args.model_path,
                        args.mlir_file_path, batch_size=args.batch_size)
    elif args.model_type == "caffe":
        c = CaffeConverter(args.model_name, args.model_path, args.model_dat,
                           args.mlir_file_path, batch_size=args.batch_size, preprocess=preprocess_args,
                           convert_preprocess=args.convert_preprocess, preprocess_args=preprocessor.to_dict(
                               input_h=input_shape[1], input_w=input_shape[2])
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
        "--swap_channel",
        help="Do preprocess, specify the channel order to swap",
        type=str,
        default="",
    )
    parser.add_argument(
        "--scale",
        help="Do preprocess, specify the scale",
        type=float,
        default=1.0,
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
    parser = add_preprocess_parser(parser)
    args = parser.parse_args()
    Convert(args)


if __name__ == "__main__":
    print(sys.argv)
    main()
