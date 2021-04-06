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
    preprocessor.config(**vars(args))

    input = None
    file_extension = args.input_file.split(".")[-1].lower()
    if file_extension == "jpg" or file_extension == "png":
        input = preprocessor.run(args.input_file, batch=args.batch_size)
    elif file_extension == "npz":
        input = np.load(args.input_file)['input']
    else:
        print("Not support {} extension")

    ort_outs = inference(input, args.model_path)
    output_num = len(ort_outs)

    np.savez(args.output_file, **{'output': ort_outs[0]})

    if args.dump_tensors:
        # second pass for dump all output
        # plz refre https://github.com/microsoft/onnxruntime/issues/1455
        output_keys = []
        model = onnx.load(args.model_path)
        no_list = ["Cast", "Shape", "Unsqueeze",
                   "Gather", "Split", "Constant", "GRU"]
        # tested commited #c3cea486d https://github.com/microsoft/onnxruntime.git
        for x in model.graph.node:
            if x.op_type in no_list:
                continue
            _intermediate_tensor_name = list(x.output)
            intermediate_tensor_name = ",".join(_intermediate_tensor_name)
            intermediate_layer_value_info = helper.ValueInfoProto()
            intermediate_layer_value_info.name = intermediate_tensor_name
            model.graph.output.append(intermediate_layer_value_info)
            output_keys.append(
                intermediate_layer_value_info.name + '_' + x.op_type)
        model_file = args.model_path.split("/")[-1]

        dump_all_onnx = "all_{}".format(model_file)

        if not os.path.exists(dump_all_onnx):
            onnx.save(model, dump_all_onnx)
        else:
            print("{} is exitsed!".format(dump_all_onnx))
        print("dump multi-output onnx all tensor at ", dump_all_onnx)

        # dump all inferneced tensor
        ort_outs = inference(input, dump_all_onnx)
        ort_outs = ort_outs[output_num:]
        tensor_all_dict = dict(
            zip(output_keys, map(np.ndarray.flatten, ort_outs)))
        tensor_all_dict['input'] = input
        np.savez(args.dump_tensors, **tensor_all_dict)
        print("dump all tensor at ", args.dump_tensors)


if __name__ == '__main__':
    main(sys.argv)
