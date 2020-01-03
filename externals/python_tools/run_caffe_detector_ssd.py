#!/usr/bin/env python

import numpy as np
import os
import sys
import argparse
import glob
import time
import cv2
import caffe
from ssd_util import preprocess, postprocess, draw

def parse_args():
    parser = argparse.ArgumentParser(description='Eval YOLO networks.')
    parser.add_argument('--model_def', type=str, default='',
                        help="Model definition file")
    parser.add_argument('--pretrained_model', type=str, default='',
                        help='Load weights from previously saved parameters.')
    parser.add_argument("--net_input_dims", default='300,300',
                        help="'height,width' dimensions of net input tensors.")
    parser.add_argument("--input_file", type=str, default='',
                        help="Input image for testing")
    parser.add_argument("--label_file", type=str, default='',
                        help="coco lable file in txt format")
    parser.add_argument("--draw_image", type=str, default='',
                        help="Draw results on image")
    parser.add_argument("--dump_blobs",
                        help="Dump all blobs into a file in npz format")
    parser.add_argument("--dump_weights",
                        help="Dump all weights into a file in npz format")
    parser.add_argument("--force_input",
                        help="Force the input blob data, in npy format")
    parser.add_argument("--obj_threshold", type=float, default=0.3,
                        help="Object confidence threshold")
    parser.add_argument("--nms_threshold", type=float, default=0.5,
                        help="NMS threshold")

    args = parser.parse_args()
    return args

def ssd_detect(net, image_path, net_input_dims, obj_threshold, nms_threshold,
                  dump_blobs=None, dump_weights=None):
    
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))  # row to col, (HWC -> CHW)
    transformer.set_mean('data', np.array([104, 117, 123], dtype=np.float32))
    transformer.set_raw_scale('data', 255)  # [0,1] to [0,255]
    transformer.set_channel_swap('data', (2, 1, 0))  # RGB to BGR

    image = caffe.io.load_image(image_path)  # range from 0 to 1

    net.blobs['data'].reshape(1, 3, net_input_dims[0], net_input_dims[1])
    net.blobs['data'].data[...] = transformer.preprocess('data', image)
    net.blobs['data'].data[...].tofile("test_dog_in_fp32.bin")

    confidence = net.forward()['mbox_conf_flatten']

    # [DEBUG] dump blobs to file
    if dump_blobs is not None:
        print("Save Blobs: ", dump_blobs)
        blobs_dict = {}
        for name, blob in net.blobs.iteritems():
            blobs_dict[name] = blob.data
        np.savez(dump_blobs, **blobs_dict)
    if dump_weights is not None:
        print("Save Weights:", dump_weights)
        weights_dict = {}
        for name, param in net.params.iteritems():
            for i in range(len(param)):
                weights_dict[name + "_" + str(i)] = param[i].data
        np.savez(dump_weights, **weights_dict)

    return confidence

    # out_feat = {}
    # out_feat['layer82-conv'] = y['layer82-conv']
    # out_feat['layer94-conv'] = y['layer94-conv']
    # out_feat['layer106-conv'] = y['layer106-conv']
    # batched_predictions = postprocess(out_feat, image.shape, net_input_dims,
    #                           obj_threshold, nms_threshold, batch=1)
    # # batch = 1
    # predictions = batched_predictions[0]
    # return predictions

def main(argv):
    args = parse_args()

    # Make Detector
    net_input_dims = [int(s) for s in args.net_input_dims.split(',')]
    obj_threshold = float(args.obj_threshold)
    nms_threshold = float(args.nms_threshold)
    print("net_input_dims", net_input_dims)
    print("obj_threshold", obj_threshold)
    print("nms_threshold", nms_threshold)

    net = caffe.Net(args.model_def, args.pretrained_model, caffe.TEST)

    # Load image
    if (args.input_file != '') :
        image = cv2.imread(args.input_file)
        predictions = ssd_detect(net, args.input_file, net_input_dims,
                                    obj_threshold, nms_threshold,
                                    args.dump_blobs, args.dump_weights)
        print(predictions)
        if (args.draw_image != ''):
            image = draw(image, predictions, args.label_file)
            cv2.imwrite(args.draw_image, image)
    else :
        print("No input_file specified")
        exit(1)


if __name__ == '__main__':
    main(sys.argv)
