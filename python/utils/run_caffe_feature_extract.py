#!/usr/bin/env python

import numpy as np
import os
import sys
import argparse
import glob
import time
import cv2
import caffe

def parse_args():
    parser = argparse.ArgumentParser(description='Eval YOLO networks.')
    parser.add_argument('--model_def', type=str, default='',
                        help="Model definition file")
    parser.add_argument('--pretrained_model', type=str, default='',
                        help='Load weights from previously saved parameters.')
    parser.add_argument("--net_input_dims", default='112,112',
                        help="'height,width' dimensions of net input tensors.")
    parser.add_argument("--input_file", type=str, default='',
                        help="Input image for testing")
    parser.add_argument("--dump_blobs",
                        help="Dump all blobs into a file in npz format")
    parser.add_argument("--dump_weights",
                        help="Dump all weights into a file in npz format")


    args = parser.parse_args()
    return args

def main(argv):
    args = parse_args()

    
    net_input_dims = [int(s) for s in args.net_input_dims.split(',')]
    print("net_input_dims", net_input_dims)

    net = caffe.Net(args.model_def, args.pretrained_model, caffe.TEST)
    img = cv2.imread(args.input_file)
    # Normalization
    _scale = 0.0078125
    _bias  = np.array([-0.99609375, -0.99609375, -0.99609375], dtype=np.float32)
    img = img * _scale
    img += _bias
    img = np.transpose(img, (2, 0, 1))
    # print(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)
    print("Save Weights:", args.dump_weights)
    weights_dict = {}
    for name, param in net.params.iteritems():
        for i in range(len(param)):
            weights_dict[name + "_" + str(i)] = param[i].data
    np.savez(args.dump_weights, **weights_dict)

    print("Save Blobs: ", args.dump_blobs)
    blobs_dict = {}
    # for name, blob in self.blobs.iteritems():
    #     blobs_dict[name] = blob.data
    for name, layer in net.layer_dict.iteritems():
        print("layer : " + str(name))
        print("  type = " + str(layer.type))
        print("  top -> " + str(net.top_names[name]))
        if layer.type == "Split":
            print("  skip Split")
            continue
        assert(len(net.top_names[name]) == 1)
        if layer.type == "Input":
            blobs_dict[name] = img
            continue
        #out = self.forward(None, prev_name, name, **{prev_name: prev_data})
        out = net.forward(None, None, name, **{net.inputs[0]: img})
        blobs_dict[name] = out[net.top_names[name][0]].copy()
    np.savez(args.dump_blobs, **blobs_dict)



if __name__ == '__main__':
    main(sys.argv)
