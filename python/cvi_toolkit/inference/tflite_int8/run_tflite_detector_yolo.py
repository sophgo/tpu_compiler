#!/usr/bin/env python3

import numpy as np
import os
import sys
import argparse
import glob
import time
import tensorflow as tf
import cv2
from cvi_toolkit.utils.yolov3_util import preprocess, postprocess_v2, postprocess_v3, postprocess_v4_tiny, draw


def check_files(args):
    if not os.path.isfile(args.model_def):
        print("cannot find the file %s", args.model_def)
        sys.exit(1)

    if not os.path.isfile(args.input_file):
        print("cannot find the file %s", args.input_file)
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description='Eval YOLO networks.')
    parser.add_argument('--model_def', type=str, default='',
                        help="Model definition file")
    parser.add_argument("--net_input_dims", default='416,416',
                        help="'height,width' dimensions of net input tensors.")
    parser.add_argument("--input_file", type=str, default='',
                        help="Input image for testing")
    parser.add_argument("--label_file", type=str, default='',
                        help="coco lable file in txt format")
    parser.add_argument("--draw_image", type=str, default='',
                        help="Draw results on image")
    parser.add_argument("--gen_input_npz",
                        help="generate input npz")
    parser.add_argument("--obj_threshold", type=float, default=0.3,
                        help="Object confidence threshold")
    parser.add_argument("--nms_threshold", type=float, default=0.5,
                        help="NMS threshold")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Set batch size")
    parser.add_argument("--yolov3", type=str, default='yes',
                        help="yolov2 or yolov3")
    parser.add_argument("--yolov4-tiny", type=str, default='false',
                        help="set to yolov4")

    args = parser.parse_args()
    check_files(args)
    return args


def main(argv):
    args = parse_args()

    # Make Detector
    net_input_dims = [int(s) for s in args.net_input_dims.split(',')]
    obj_threshold = float(args.obj_threshold)
    nms_threshold = float(args.nms_threshold)
    yolov3 = True if args.yolov3 == 'yes' else False
    print("net_input_dims", net_input_dims)
    print("obj_threshold", obj_threshold)
    print("nms_threshold", nms_threshold)
    print("yolov3", yolov3)

    image = cv2.imread(args.input_file)
    image_x = preprocess(image, net_input_dims)

    image_x = np.expand_dims(image_x, axis=0)
    inputs = image_x
    # for i in range(1, args.batch_size):
    #   inputs = np.append(inputs, image_x, axis=0)
    net = tf.lite.Interpreter(
        model_path=args.model_def)
    input_details = net.get_input_details()
    output_details = net.get_output_details()
    net.allocate_tensors()
    np.savez(args.gen_input_npz, input=inputs)
    inputs = np.transpose(inputs, (0, 2, 3, 1))
    net.set_tensor(input_details[0]['index'], inputs)
    net.invoke()
    featrue = list()
    for i in range(3):
        featrue.append(net.get_tensor(output_details[2-i]['index']))

    out_feat = {}

    out_feat['layer82-conv'] = np.transpose(featrue[0], (0, 3, 1, 2))
    out_feat['layer94-conv'] = np.transpose(featrue[1], (0, 3, 1, 2))
    out_feat['layer106-conv'] = np.transpose(featrue[2], (0, 3, 1, 2))
    batched_predictions = postprocess_v3(out_feat, image.shape, net_input_dims,
                                            obj_threshold, nms_threshold, False, args.batch_size)
    print(batched_predictions[0])
    if args.draw_image:
        image = draw(image, batched_predictions[0], args.label_file)
        cv2.imwrite(args.draw_image, image)

if __name__ == '__main__':
    main(sys.argv)
