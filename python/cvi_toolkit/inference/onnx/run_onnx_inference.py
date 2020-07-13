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

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def inference(input, model_path):
    """
        input: np.ndarray
        model_path: str
    """
    ort_session = onnxruntime.InferenceSession(model_path)
    ort_inputs = {ort_session.get_inputs()[0].name: input}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs

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
        "--model_path",
        help="onnx model path."
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

    parser = get_preprocess_parser(existed_parser=parser)

    args = parser.parse_args()

    preprocessor = preprocess()
    preprocessor.config(net_input_dims=args.net_input_dims,
                    resize_dims=args.image_resize_dims,
                    mean=args.mean,
                    mean_file=args.mean_file,
                    input_scale=args.input_scale,
                    raw_scale=args.raw_scale,
                    std=args.std,
                    rgb_order=args.model_channel_order,
                    data_format=args.data_format,
                    bgray=args.bgray
                    )
    input=None
    file_extension = args.input_file.split(".")[-1].lower()
    if file_extension == "jpg" or file_extension == "png":
        mean = [float(x) for x in args.mean.split(",")]
        if args.std:
            std = [float(x) for x in args.std.split(",")]
        else:
            std = [1, 1, 1]
        net_input_dims = [int(x) for x in args.net_input_dims.split(",")]
        if args.image_resize_dims:
            image_resize_dims = [int(x) for x in args.image_resize_dims.split(",")]
        else:
            image_resize_dims = net_input_dims

        input = preprocessor.run(args.input_file, output_channel_order=args.model_channel_order)
        inputs = input
        for i in range(1, args.batch_size):
            inputs = np.append(inputs, input, axis=0)
        input = inputs
    elif file_extension == "npz":
        input = np.load(args.input_file)['input']
    else:
        print("Not support {} extension")

    ort_outs = inference(input, args.model_path)
    np.savez(args.output_file, **{'output': ort_outs[0]})

    if args.dump_tensors:
        # second pass for dump all output
        # plz refre https://github.com/microsoft/onnxruntime/issues/1455
        output_keys = ['output']
        model = onnx.load(args.model_path)

        # tested commited #c3cea486d https://github.com/microsoft/onnxruntime.git
        for x in model.graph.node:
            if x.op_type == 'Split':
                continue
            _intermediate_tensor_name = list(x.output)
            intermediate_tensor_name = ",".join(_intermediate_tensor_name)
            intermediate_layer_value_info = helper.ValueInfoProto()
            intermediate_layer_value_info.name = intermediate_tensor_name
            model.graph.output.append(intermediate_layer_value_info)
            output_keys.append(intermediate_layer_value_info.name + '_' + x.op_type)
        model_file = args.model_path.split("/")[-1]

        dump_all_onnx = "all_{}".format(model_file)

        if not os.path.exists(dump_all_onnx):
            onnx.save(model, dump_all_onnx)
        else:
            print("{} is exitsed!".format(dump_all_onnx))
        print("dump multi-output onnx all tensor at ", dump_all_onnx)

        # dump all inferneced tensor
        ort_outs = inference(input, dump_all_onnx)
        tensor_all_dict = dict(zip(output_keys, map(np.ndarray.flatten, ort_outs)))
        tensor_all_dict['input'] = input
        np.savez(args.dump_tensors, **tensor_all_dict)
        print("dump all tensor at ", args.dump_tensors)

if __name__ == '__main__':
    main(sys.argv)

