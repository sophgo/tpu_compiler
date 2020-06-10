#!/usr/bin/env python3
import argparse
import os

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor

import onnx
import onnxruntime

def to_numpy(tensor):
    return tensor.deatch().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Super Resolution')
    parser.add_argument(
        "--input_file",
        help="Input npz or jpg file."
    )
    parser.add_argument(
        "--output_file",
        help="Output npz"
    )
    parser.add_argument(
        "--model_def",
        help="onnx model path."
    )
    parser.add_argument(
        "--net_input_dims",
        help="input dims"
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
    args = parser.parse_args()

    input_dims = [ int(x) for x in args.net_input_dims.split(',') ]

    ort_session = onnxruntime.InferenceSession(args.model_def)

    img = Image.open(args.input_file)
    img = img.resize((input_dims[0],input_dims[1]), Image.ANTIALIAS).convert("RGB")
    img = ToTensor()(img).view(1, -1, img.size[1], img.size[0])
    inputs = to_numpy(img)
    for i in range(1, args.batch_size):
      inputs = np.append(inputs, to_numpy(img), axis=0)

    print(inputs.shape)
    ort_inputs = {ort_session.get_inputs()[0].name: inputs}
    ort_outs = ort_session.run(None, ort_inputs)
    np.savez(args.output_file, **{'output': ort_outs[0]})

    if args.dump_tensors:
        # second pass for dump all output
        # plz refre https://github.com/microsoft/onnxruntime/issues/1455
        output_keys = []
        for i in range(len(ort_outs)):
            output_keys.append('output_{}'.format(i))

        model = onnx.load(args.model_def)

        # tested commited #c3cea486d https://github.com/microsoft/onnxruntime.git
        for x in model.graph.node:
            _intermediate_tensor_name = list(x.output)
            intermediate_tensor_name = ",".join(_intermediate_tensor_name)
            intermediate_layer_value_info = onnx.helper.ValueInfoProto()
            intermediate_layer_value_info.name = intermediate_tensor_name
            model.graph.output.append(intermediate_layer_value_info)
            output_keys.append(intermediate_layer_value_info.name + '_' + x.op_type)

        dump_all_onnx = "dump_all.onnx"
        if not os.path.exists(dump_all_onnx):
            onnx.save(model, dump_all_onnx)
        else:
            print("{} is exitsed!".format(dump_all_onnx))
        print("dump multi-output onnx all tensor at ", dump_all_onnx)

        # dump all inferneced tensor
        ort_session = onnxruntime.InferenceSession(dump_all_onnx)
        ort_outs = ort_session.run(None, ort_inputs)
        tensor_all_dict = dict(zip(output_keys, ort_outs))
        tensor_all_dict['input'] = inputs
        np.savez(args.dump_tensors, **tensor_all_dict)
        print("dump all tensor at ", args.dump_tensors)