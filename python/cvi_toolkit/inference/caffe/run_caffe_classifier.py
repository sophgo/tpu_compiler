#!/usr/bin/env python3
"""
classify.py is an out-of-the-box image classifer callable from the command line.

By default it configures and runs the Caffe reference ImageNet model.
"""
import numpy as np
import os
import sys
import argparse
import glob
import time
import skimage
import caffe
from cvi_toolkit.data.preprocess import get_preprocess_parser, preprocess
from cvi_toolkit.model import CaffeModel

def check_files(args):
    if not os.path.isfile(args.model_def):
        print("cannot find the file %s", args.model_def)
        sys.exit(1)

    if not os.path.isfile(args.pretrained_model):
        print("cannot find the file %s", args.pretrained_model)
        sys.exit(1)

    if not os.path.isfile(args.input_file):
        print("cannot find the file %s", args.input_file)
        sys.exit(1)

def main(argv):
    pycaffe_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "input_file",
        help="Input image, directory, or npy."
    )
    parser.add_argument(
        "output_file",
        help="Output npy filename."
    )
    # Optional arguments.
    parser.add_argument(
        "--model_def",
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
        help="Trained model weights file."
    )
    parser.add_argument(
        "--label_file",
        help="Labels file"
    )
    parser.add_argument(
        "--dump_blobs",
        help="Dump all blobs into a file in npz format"
    )
    parser.add_argument(
        "--dump_blobs_with_inplace",
        type=bool, default=False,
        help="Dump all blobs including inplace blobs (takes much longer time)"
    )
    parser.add_argument(
        "--batch_size",
        type=int, default=1,
        help="Set batch size"
    )
    parser = get_preprocess_parser(existed_parser=parser)

    args = parser.parse_args()
    check_files(args)
    caffemodel = CaffeModel()
    caffemodel.load_model(args.model_def, args.pretrained_model)
    if args.net_input_dims:
        net_input_dims = args.net_input_dims
    else:
        # read from caffe
        net_input_dims = caffemodel.get_input_shape()
    preprocessor = preprocess()
    preprocessor.config(**vars(args))

    # Load image file.
    args.input_file = os.path.expanduser(args.input_file)
    print("Loading file: %s" % args.input_file)

    inputs = preprocessor.run(args.input_file, batch=args.batch_size)
    print("inputs:", inputs.shape)
    # Classify.
    start = time.time()
    predictions = caffemodel.inference(inputs)
    print("Done in %.2f s." % (time.time() - start))

    # Save
    print("Saving results into %s" % args.output_file)
    np.save(args.output_file, predictions)

    # Get all tensor
    all_tensor_dict = caffemodel.get_all_tensor(inputs,
        args.dump_blobs_with_inplace)
    np.savez(args.dump_blobs, **all_tensor_dict)
    # Print
    for ix, in_ in enumerate(inputs):
      print("batch : ", ix)
      print(predictions[ix].argmax())
      if args.label_file:
         labels = np.loadtxt(args.label_file, str, delimiter='\t')
         top_k = predictions[ix].flatten().argsort()[-1:-6:-1]
         print(top_k)
         if max(top_k) > 1000:
             pass # no need to show top_k > 1000
         else:
             print(labels[top_k])
         prob = np.squeeze(predictions[ix].flatten())
         idx = np.argsort(-prob)
         for i in range(5):
             label = idx[i]
             # TODO: non-classifier case's label i invalid
             if label < 1000:
                 print('%d - %.2f - %s' % (idx[i], prob[label], labels[label]))

if __name__ == '__main__':
    main(sys.argv)
