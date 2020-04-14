#!/usr/bin/env python

import numpy as np
import os
import sys
import argparse
import glob
import time
import cv2
import caffe
from cvi_toolkit.utils.yolov3_util import preprocess, postprocess, draw

def parse_args():
    parser = argparse.ArgumentParser(description='Eval YOLO networks.')
    parser.add_argument('--model_def', type=str, default='',
                        help="Model definition file")
    parser.add_argument('--pretrained_model', type=str, default='',
                        help='Load weights from previously saved parameters.')
    parser.add_argument("--net_input_dims", default='416,416',
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
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Set batch size")

    args = parser.parse_args()
    return args

def yolov3_detect(net, image, net_input_dims, obj_threshold, nms_threshold,
                  dump_blobs=None, dump_weights=None, batch=1):
    image_x = preprocess(image, net_input_dims)
    # net.blobs['data'].data[...] = x
    # y = net.forward()
    image_x = np.expand_dims(image_x, axis=0)
    x = image_x
    for i in range(1, batch):
      x = np.append(x, image_x, axis=0)

    net.blobs['data'].reshape(batch, x.shape[1], x.shape[2], x.shape[3])
    net.blobs['data'].data[...] = x

    print("input shape", x.shape)
    y = net.forward_all(**{net.inputs[0]: x})

    # dump blobs to file
    # this dump all blobs even if some of them are `in_place` blobs
    if dump_blobs is not None:
        print("Save Blobs: ", dump_blobs)
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
                blobs_dict[name] = x
                continue
            #out = self.forward(None, prev_name, name, **{prev_name: prev_data})
            out = net.forward(None, None, name, **{net.inputs[0]: x})
            blobs_dict[name] = out[net.top_names[name][0]].copy()
        np.savez(dump_blobs, **blobs_dict)

    # dump weight to file
    if dump_weights is not None:
        print("Save Weights:", dump_weights)
        weights_dict = {}
        for name, param in net.params.iteritems():
            for i in range(len(param)):
                weights_dict[name + "_" + str(i)] = param[i].data
        np.savez(dump_weights, **weights_dict)

    out_feat = {}
    out_feat['layer82-conv'] = net.blobs['layer82-conv'].data
    out_feat['layer94-conv'] = net.blobs['layer94-conv'].data
    out_feat['layer106-conv'] = net.blobs['layer106-conv'].data
    batched_predictions = postprocess(out_feat, image.shape, net_input_dims,
                              obj_threshold, nms_threshold, batch)
    return batched_predictions[0]

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
        predictions = yolov3_detect(net, image, net_input_dims,
                                    obj_threshold, nms_threshold,
                                    args.dump_blobs, args.dump_weights, args.batch_size)
        print(predictions)
        if (args.draw_image != ''):
            for i in range(1, args.batch_size):
                image = draw(image, predictions[i], args.label_file)
                cv2.imwrite(args.draw_image, image)
    else :
        print("No input_file specified")
        exit(1)


if __name__ == '__main__':
    main(sys.argv)
