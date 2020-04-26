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
from cvi_toolkit import preprocess
from cvi_toolkit.model import CaffeModel


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
        "--net_input_dims",
        help="'height,width' dimensions of network input spatial dimensions."
    )
    parser.add_argument(
        "--image_resize_dims",
        default='256,256',
        help="To resize to this size first, then crop to net_input_dims."
    )
    parser.add_argument(
        "--mean_file",
        help="Data set image mean of [Channels x Height x Width] dimensions " +
             "(numpy array). Set to '' for no mean subtraction."
    )
    parser.add_argument(
        "--mean",
        help="Per Channel image mean values"
    )
    parser.add_argument(
        "--std",
        help="Per Channel image std values",
        default='1,1,1'
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        default=1.0,
        help="Multiply input features by this scale to finish preprocessing."
    )
    parser.add_argument(
        "--raw_scale",
        type=float,
        default=255.0,
        help="Multiply raw input by this scale before preprocessing."
    )
    parser.add_argument(
        "--channel_swap",
        default='2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV."
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
        "--dump_weights",
        help="Dump all weights into a file in npz format"
    )
    parser.add_argument(
        "--batch_size",
        type=int, default=1,
        help="Set batch size"
    )
    parser.add_argument(
        "--force_input",
        help="Force the input blob data, in npy format"
    )
    args = parser.parse_args()
    caffemodel = CaffeModel()
    caffemodel.load_model(args.model_def, args.pretrained_model)
    if args.net_input_dims:
        net_input_dims = args.net_input_dims
    else:
        # read from caffe
        net_input_dims = caffemodel.get_input_shape()
    preprocessor = preprocess()
    preprocessor.config(net_input_dims=net_input_dims,
                    resize_dims=args.image_resize_dims,
                    mean=args.mean,
                    mean_file=args.mean_file,
                    input_scale=args.input_scale,
                    raw_scale=args.raw_scale,
                    std=args.std, batch=args.batch_size)

    # Load image file.
    args.input_file = os.path.expanduser(args.input_file)
    print("Loading file: %s" % args.input_file)

    input_data = preprocessor.run(args.input_file)
    inputs = input_data
    for i in range(1, args.batch_size):
      inputs = np.append(inputs, input_data, axis=0)

    # Classify.
    start = time.time()
    predictions = caffemodel.inference(inputs)
    print("Done in %.2f s." % (time.time() - start))

    # Save
    print("Saving results into %s" % args.output_file)
    np.save(args.output_file, predictions)

    # Get all tensor
    all_tensor_dict = caffemodel.get_all_tensor(inputs)
    np.savez(args.dump_blobs, **all_tensor_dict)
    # Print
    for ix, in_ in enumerate(inputs):
      print("batch : ", ix)
      print(predictions[ix].argmax())
      if args.label_file:
         labels = np.loadtxt(args.label_file, str, delimiter='\t')
         top_k = predictions[ix].flatten().argsort()[-1:-6:-1]
         print(labels[top_k])
         print(top_k)
         prob = np.squeeze(predictions[ix].flatten())
         idx = np.argsort(-prob)
         for i in range(5):
             label = idx[i]
             print('%d - %.2f - %s' % (idx[i], prob[label], labels[label]))

if __name__ == '__main__':
    main(sys.argv)
