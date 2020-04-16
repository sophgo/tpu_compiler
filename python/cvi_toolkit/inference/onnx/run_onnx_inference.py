#!/usr/bin/env python3
import sys
import numpy as np
import argparse
import onnxruntime
import onnx
import os
from onnx import helper

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def inference(args, model_path):
    input_npz = args.input_file
    input = np.load(input_npz)['input']
    print(model_path)
    ort_session = onnxruntime.InferenceSession(model_path)
    ort_inputs = {ort_session.get_inputs()[0].name: input}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        help="Input npz."
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

    args = parser.parse_args()
    ort_outs = inference(args, args.model_path)
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
        ort_outs = inference(args, dump_all_onnx)
        np.savez(args.dump_tensors, **dict(zip(output_keys, map(np.ndarray.flatten, ort_outs))))
        print("dump all tensor at ", args.dump_tensors)

if __name__ == '__main__':
    main(sys.argv)

