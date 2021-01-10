#!/usr/bin/env python3
import sys
import numpy as np
import argparse
import onnxruntime
import onnx
import os
from onnx import helper
from PIL import Image
from torchvision import transforms
from cvi_toolkit.data.preprocess import get_preprocess_parser, preprocess
from cvi_toolkit.model.ModelFactory import ModelFactory

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        help="Input npz or jpg file."
    )
    parser.add_argument(
        "--output_file",
        help="Output npz"
    )
    parser.add_argument(
        "--dump_tensors",
        help="Dump all output tensors into a file in npz format"
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Input batch size."
    )
    parser.add_argument(
        "--model_type",
        default='caffe',
        type=str,
        help="Input model type"
    )
    parser.add_argument(
        "--model_def",
        default=None,
        type=str,
        help="Model definition file"
    )
    parser.add_argument(
        "--pretrained_model",
        default=None,
        type=str,
        help="Load weights from previously saved parameters."
    )
    parser.add_argument(
        "--mlir_file",
        default=None,
        type=str,
        help="mlir file."
    )

    parser = get_preprocess_parser(existed_parser=parser)
    args = parser.parse_args()

    net = ModelFactory()
    net.load_model(args.model_type, model_file=args.model_def,
             weight_file=args.pretrained_model, mlirfile=args.mlir_file
             )

    preprocessor = preprocess()
    if args.model_type == 'tensorflow' or \
       args.model_type == "tflite_int8":
        args.data_format = 'nhwc'
    preprocessor.config(**vars(args))

    input=None
    file_extension = args.input_file.split(".")[-1].lower()
    if file_extension == "jpg":
        input = preprocessor.run(args.input_file, batch=args.batch_size)
    elif file_extension == "npz":
        input = np.load(args.input_file)['input']
    else:
        print("Not support {} extension")

    output = net.inference(input)

    np.savez(args.output_file, **{'output': output})
    if args.dump_tensors:
        # dump all inferneced tensor
        all_tensor = net.get_all_tensor(input)
        np.savez(args.dump_tensors, **all_tensor)
        print("dump all tensor at ", args.dump_tensors)

if __name__ == '__main__':
    main(sys.argv)

