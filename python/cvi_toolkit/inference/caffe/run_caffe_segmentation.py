#!/usr/bin/env python3

import os
import numpy as np
from argparse import ArgumentParser
from os.path import join
import argparse
import sys
import caffe
import cv2
import time
from cvi_toolkit.model import CaffeModel

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        help="Input image, directory, or npy."
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
        "--colours",
        help="LUT file"
    )
    parser.add_argument(
        "--net_input_dims", default='512,1024',
        help="'height,width' dimensions of net input tensors."
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help='0: cpu mode active, else cpu mode inactive'
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
        "--output",
        type=str, default='',
        help="Indicate output name"
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
        "--draw_image", type=str, default='',
        help="Draw results on image"
    )
    return parser

if __name__ == '__main__':
    parser1 = make_parser()
    args = parser1.parse_args()
    if args.gpu == 1:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    net_input_dims = [int(s) for s in args.net_input_dims.split(',')]
    label_colours = cv2.imread(args.colours, 1).astype(np.uint8)
    input_data = cv2.imread(args.input_file, 1).astype(np.float32)

    input_data = cv2.resize(input_data, (net_input_dims[1], net_input_dims[0]))
    input_data = input_data.transpose((2, 0, 1))
    input_data = np.asarray([input_data])

    inputs = input_data
    for i in range(1, args.batch_size):
      inputs = np.append(inputs, input_data, axis=0)

    start = time.time()
    caffemodel = CaffeModel()
    caffemodel.load_model(args.model_def, args.pretrained_model)
    caffemodel.inference(inputs)
    outputs = caffemodel.net.blobs
    print("Done in %.2f s." % (time.time() - start))

#    # Save
#    print("Saving results into %s" % args.output_file)
#    np.save(args.output_file, predictions)

    # Get all tensor
    all_tensor_dict = caffemodel.get_all_tensor(inputs,
        args.dump_blobs_with_inplace)
    np.savez(args.dump_blobs, **all_tensor_dict)

    # dump weight to file
    if args.dump_weights is not None:
        print("Save Weights:", args.dump_weights)
        weights_dict = caffemodel.get_all_weights()
        np.savez(args.dump_weights, **weights_dict)


    ##--decoder: Change the name of output blobs according to your prototxts--

    if (args.draw_image != ''):
        output_shape = outputs[args.output].data.shape
        prediction = outputs[args.output].data
        if caffemodel.net.layer_dict[args.output].type != "ArgMax":
            prediction = prediction[0].argmax(axis=0)

        prediction = np.squeeze(prediction)
        prediction = np.resize(prediction, (3, output_shape[2], output_shape[3]))
        prediction = prediction.transpose(1, 2, 0).astype(np.uint8)

        prediction_rgb = np.zeros(prediction.shape, dtype=np.uint8)
        label_colours_bgr = label_colours[..., ::-1]
        cv2.LUT(prediction, label_colours_bgr, prediction_rgb)
        cv2.imwrite(args.draw_image, prediction_rgb)
