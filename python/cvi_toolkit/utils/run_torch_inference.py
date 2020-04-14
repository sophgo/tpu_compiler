#!/usr/bin/env python3
import sys
import numpy as np
import torch
import argparse

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def to_onnx(torch_model, input, model_path):
    torch.onnx.export(torch_model,               # model being run
                  input,                         # model input (or a tuple for multiple inputs)
                  model_path,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )

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
        help="Pytorch model path."
    )
    parser.add_argument(
        "--gen_onnx_model_path",
        help="onnx_model store path",
        type=str, default=''
    )
    args = parser.parse_args()
    input_npz = args.input_file
    input = np.load(input_npz)['input']
    input = torch.from_numpy(input)
    torch_model = torch.load(args.model_path)
    torch_model.eval()
    torch_out = torch_model(input)
    if args.gen_onnx_model != "":
        to_onnx(torch_model, input, args.gen_onnx_model)

    np.savez(args.output_file, **{'output': to_numpy(torch_out)})
   
if __name__ == '__main__':
    main(sys.argv)
