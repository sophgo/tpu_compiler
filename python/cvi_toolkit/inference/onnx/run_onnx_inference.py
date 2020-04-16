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

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def preprocess(image_file, mean, std, resize, centor_crop):
    """
        image_file: str
        mean: list
        std: list
        resize: list
        centor_crop: list

    """
    normalize = transforms.Normalize(mean=mean,
                                     std=std)
    tfms = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(centor_crop),
            transforms.ToTensor(),
            normalize,
    ])
    return tfms(Image.open(image_file)).unsqueeze(0).numpy()

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
        "--mean",
        help="Per Channel image mean values"
    )
    parser.add_argument(
        "--std",
        help="Per Channel image mean values"
    )
    parser.add_argument(
        "--net_input_dims",
        default='224,224',
        help="'height,width' dimensions of network input spatial dimensions."
    )
    parser.add_argument(
        "--image_resize_dims",
        default='256,256',
        help="To resize to this size first, then crop to net_input_dims."
    )
    args = parser.parse_args()


    file_extension = args.input_file.split(".")[-1].lower()
    if file_extension == "jpg":
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
        input = preprocess(args.input_file, mean, std, image_resize_dims, net_input_dims)
    elif file_extension == "npz":
        input = np.load(args.input_file)['input']
    else:
        print("Not support {} extension")

    ort_outs = inference(input, args.model_path)
    np.savez(args.output_file, **{'output': ort_outs[0]})
    #print("org ort_outs", ort_outs)

    if args.dump_tensors:
        # second pass for dump all output
        # plz refre https://github.com/microsoft/onnxruntime/issues/1455
        output_keys = ['output']
        model = onnx.load(args.model_path)

        # tested commited #c3cea486d https://github.com/microsoft/onnxruntime.git
        for x in model.graph.node:
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

