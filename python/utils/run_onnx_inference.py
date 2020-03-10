#!/usr/bin/env python3
import sys
import numpy as np
import argparse
import onnxruntime

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


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
 
    args = parser.parse_args()
    input_npz = args.input_file
    input = np.load(input_npz)['input']
    print(args.model_path)
    ort_session = onnxruntime.InferenceSession(args.model_path)
    ort_inputs = {ort_session.get_inputs()[0].name: input}
    ort_outs = ort_session.run(None, ort_inputs)

    np.savez(args.output_file, **{'output': ort_outs[0]})
   
if __name__ == '__main__':
    main(sys.argv)
