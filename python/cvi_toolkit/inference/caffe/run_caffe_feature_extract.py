#!/usr/bin/env python

import numpy as np
import os
import sys
import argparse
import glob
import time
import cv2
import caffe

suport_model = [
    "arcface_res50",
    "bmface_v3",
    "liveness"
]

def parse_args():
    parser = argparse.ArgumentParser(description='feature extract networks.')
    parser.add_argument('--model_def', type=str, default='',
                        help="Model definition file")
    parser.add_argument('--pretrained_model', type=str, default='',
                        help='Load weights from previously saved parameters.')
    parser.add_argument("--input_file", type=str, default='',
                        help="Input image for testing")
    parser.add_argument("--dump_blobs",
                        help="Dump all blobs into a file in npz format")
    parser.add_argument("--dump_weights",
                        help="Dump all weights into a file in npz format")
    parser.add_argument("--model_type", type=str, default='', 
                        help="bmface_v3, liveness")
    parser.add_argument("--batch_size", type=int, default=1, help="Set batch size")


    args = parser.parse_args()
    return args

def main(argv):
    args = parse_args()

    input = None
    net = caffe.Net(args.model_def, args.pretrained_model, caffe.TEST)
    if args.model_type == "bmface_v3":
        input_x = cv2.imread(args.input_file)
        input = input_x
        for i in range(1, args.batch_size):
          input = np.append(input, input_x, axis=0)

        # Normalization
        _scale = 0.0078125
        _bias  = np.array([-0.99609375, -0.99609375, -0.99609375], dtype=np.float32)
        input = input * _scale
        input += _bias
        input = np.transpose(input, (2, 0, 1))
        input = np.expand_dims(input, axis=0)
        input = input.astype(np.float32)
        input = input.reshape((args.batch_size, 3, 112, 112))
    elif args.model_type == "arcface_res50":
        input_x = cv2.imread(args.input_file)
        # from bgr to rgb
        input_x[:,:,0], input_x[:,:,2] = input_x[:,:,2], input_x[:,:,0]
        input = input_x
        for i in range(1, args.batch_size):
          input = np.append(input, input_x, axis=0)
        # normalize
        _scale = 0.0078125
        _bias = np.array([-127.5, -127.5, -127.5], dtype=np.float32)
        input = input.astype(np.float32)
        input += _bias
        input = input * _scale
        # from hwc to chw
        input = np.transpose(input, (2, 0, 1))
        input = np.expand_dims(input, axis=0)
        input = input.reshape((args.batch_size, 3, 112, 112))
    elif args.model_type == "liveness":
        input_x = np.fromfile(args.input_file, dtype=np.float32)
        input = input_x
        for i in range(1, args.batch_size):
          input = np.append(input, input_x, axis=0)
        input = input.reshape((args.batch_size, 6, 32, 32))
    else:
        print("Now only support:")
        for i in support_model:
            print("    > {}".format(i))
        exit(-1)
    print("Save Weights:", args.dump_weights)
    weights_dict = {}
    for name, param in net.params.iteritems():
        for i in range(len(param)):
            weights_dict[name + "_" + str(i)] = param[i].data
    np.savez(args.dump_weights, **weights_dict)

    print("Save Blobs: ", args.dump_blobs)
    blobs_dict = {}
    # reshape blobs
    in_ = net.inputs[0]
    net.blobs[in_].reshape(args.batch_size, net.blobs[in_].data.shape[1], net.blobs[in_].data.shape[2], net.blobs[in_].data.shape[3])
    # for name, blob in self.blobs.iteritems():
    #     blobs_dict[name] = blob.data
    for name, layer in net.layer_dict.iteritems():
        print("layer : " + str(name))
        print("  type = " + str(layer.type))
        print("  top -> " + str(net.top_names[name]))
        if layer.type == "Split":
            print("  skip Split")
            continue
        
        if layer.type == "Input":
            blobs_dict[name] = input
            continue
        #out = self.forward(None, prev_name, name, **{prev_name: prev_data})
        out = net.forward(None, None, name, **{net.inputs[0]: input})
        blobs_dict[name] = out[net.top_names[name][0]].copy()
    np.savez(args.dump_blobs, **blobs_dict)



if __name__ == '__main__':
    main(sys.argv)
