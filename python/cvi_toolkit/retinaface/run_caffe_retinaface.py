#!/usr/bin/env python3

import numpy as np
import os
import sys
import argparse
import time
import cv2
import caffe
from retinaface_util import RetinaFace

g_default_fp32_model_path = os.path.join(os.environ['MODEL_PATH'], 'face_detection/retinaface/caffe')
g_default_proto = os.path.join(g_default_fp32_model_path, 'R50-0000.prototxt')
g_default_weight_fp32 = os.path.join(g_default_fp32_model_path, 'R50-0000.caffemodel')
g_default_input_image = '/workspace/llvm-project/llvm/projects/mlir/externals/python_tools/data/faces/test.jpg'
g_detector = RetinaFace()

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

def parse_args():
    parser = argparse.ArgumentParser(description='Eval Retinaface networks.')
    parser.add_argument('--model_def', type=str, default=g_default_proto,
                        help="Model definition file")
    parser.add_argument('--pretrained_model', type=str, default=g_default_weight_fp32,
                        help='Load weights from previously saved parameters.')
    parser.add_argument("--net_input_dims", default='600,600',
                        help="'height,width' dimensions of net input tensors.")
    parser.add_argument("--input_file", type=str, default=g_default_input_image,
                        help="Input image for testing")
    parser.add_argument("--draw_image", type=str, default='',
                        help="Draw results on image")
    parser.add_argument("--obj_threshold", type=float, default=0.5,
                        help="Object confidence threshold")
    parser.add_argument("--nms_threshold", type=float, default=0.4,
                        help="NMS threshold")
    parser.add_argument("--dump_blobs",
                        help="Dump all blobs into a file in npz format")
    parser.add_argument("--dump_weights",
                        help="Dump all weights into a file in npz format")
    parser.add_argument("--batch_size", type=int, default=1, help="Set batch size")

    args = parser.parse_args()
    check_files(args)
    return args


def retinaface_detect(net, img_bgr, net_input_dims, net_batch, obj_threshold, nms_threshold):
    image_x = g_detector.preprocess(img_bgr, net_input_dims[0], net_input_dims[1])
    x = image_x
    for i in range(1, net_batch):
      x = np.append(x, image_x, axis=0)

    net.blobs['data'].reshape(net_batch, 3, x.shape[2], x.shape[3])
    net.blobs['data'].data[...] = x

    y = net.forward()
    faces, landmarks = g_detector.postprocess(y, net_input_dims[0], net_input_dims[1])

    return faces, landmarks


def dump_weights(net, dump_weights):
    weights_dict = {}
    for name, param in net.params.items():
        for i in range(len(param)):
            weights_dict[name + "_" + str(i)] = param[i].data
    np.savez(dump_weights, **weights_dict)


def dump_blobs(net, dump_blobs, img_bgr, net_input_dims, net_batch):
    image_x = g_detector.preprocess(img_bgr, *net_input_dims)
    x = image_x
    for i in range(1, net_batch):
      x = np.append(x, image_x, axis=0)

    net.blobs['data'].reshape(net_batch, 3, x.shape[2], x.shape[3])
    net.blobs['data'].data[...] = x
    print("Save Blobs: ", dump_blobs)
    blobs_dict = {}

    for layer_name, layer in net.layer_dict.items():
        pred = net.forward(start=layer_name, end=layer_name)
        print("layer : " + str(layer_name))
        print("  type = " + str(layer.type))
        print("  top -> " + str(net.top_names[layer_name]))
        if layer.type == "Split":
            print("  skip Split")
            continue
        assert(len(net.top_names[layer_name]) == 1)
        if layer.type == "Input":
            blobs_dict[layer_name] = x
            continue
        blobs_dict[layer_name] = pred[net.top_names[layer_name][0]].copy()

    np.savez(dump_blobs, **blobs_dict)


def main(argv):
    args = parse_args()

    # Make Detector
    net_input_dims = [int(s) for s in args.net_input_dims.split(',')]
    obj_threshold = float(args.obj_threshold)
    nms_threshold = float(args.nms_threshold)

    net = caffe.Net(args.model_def, args.pretrained_model, caffe.TEST)

    # Load image
    image = cv2.imread(args.input_file)

    if args.dump_weights is not None:
        print("Save Weights:", args.dump_weights)
        dump_weights(net, args.dump_weights)

    if args.dump_blobs is not None:
        dump_blobs(net, args.dump_blobs, image, net_input_dims, args.batch_size) 

    if (args.draw_image != ''):
        faces, landmarks = retinaface_detect(net, image, net_input_dims, args.batch_size,
                                obj_threshold, nms_threshold)
        draw_image = g_detector.draw(image, faces, landmarks, True)
        cv2.imwrite(args.draw_image, draw_image)
        cv2.imshow('face', draw_image)
        cv2.waitKey(0)


if __name__ == '__main__':
    main(sys.argv)
