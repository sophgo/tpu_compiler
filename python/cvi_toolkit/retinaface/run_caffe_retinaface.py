#!/usr/bin/env python3

import numpy as np
import os
import sys
import argparse
import time
import cv2
import caffe
from retinaface_util import RetinaFace
from cvi_toolkit.model import CaffeModel

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
    parser.add_argument("--batch_size", type=int, default=1, help="Set batch size")

    args = parser.parse_args()
    check_files(args)
    return args

def retinaface_detect(net, image, net_input_dims, net_batch, obj_threshold, nms_threshold):
    net.blobs['data'].reshape(1, 3, net_input_dims[0], net_input_dims[1])
    net.blobs['data'].data[...] = image

    y = net.forward()
    faces, landmarks = g_detector.postprocess(y, net_input_dims[0], net_input_dims[1])

    return faces, landmarks

def main(argv):
    args = parse_args()

    # Make Detector
    net_input_dims = [int(s) for s in args.net_input_dims.split(',')]
    obj_threshold = float(args.obj_threshold)
    nms_threshold = float(args.nms_threshold)

    # Load image
    image = cv2.imread(args.input_file)
    image = g_detector.preprocess(image, *net_input_dims)
    image = np.expand_dims(image, axis=0)
    inputs = image
    for i in range(1, args.batch_size):
      inputs = np.append(inputs, image, axis=0)
    inputs = inputs.reshape(args.batch_size, 3, net_input_dims[0], net_input_dims[1])

    caffemodel = CaffeModel()
    caffemodel.load_model(args.model_def, args.pretrained_model)
    caffemodel.inference(inputs)

    if args.dump_blobs:
        print("Save Blobs: ", args.dump_blobs)
        blobs_dict = caffemodel.get_all_tensor(inputs, True)
        np.savez(args.dump_blobs, **blobs_dict)

    if (args.draw_image != ''):
        faces, landmarks = retinaface_detect(caffemodel.net, image, net_input_dims, args.batch_size,
                                obj_threshold, nms_threshold)
        draw_image = g_detector.draw(image, faces, landmarks, True)
        cv2.imwrite(args.draw_image, draw_image)
        cv2.imshow('face', draw_image)
        cv2.waitKey(0)


if __name__ == '__main__':
    main(sys.argv)
