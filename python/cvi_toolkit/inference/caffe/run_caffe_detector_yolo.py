#!/usr/bin/env python3

import numpy as np
import os
import sys
import argparse
import glob
import time
import cv2
import caffe
from cvi_toolkit.model import CaffeModel
from cvi_toolkit.utils.yolov3_util import preprocess, postprocess_v2, postprocess_v3, postprocess_v3_tiny, draw

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
    parser.add_argument("--dump_blobs_with_inplace",
                        type=bool, default=False,
                        help="Dump all blobs including inplace blobs (takes much longer time)")
    parser.add_argument("--force_input",
                        help="Force the input blob data, in npy format")
    parser.add_argument("--obj_threshold", type=float, default=0.3,
                        help="Object confidence threshold")
    parser.add_argument("--nms_threshold", type=float, default=0.5,
                        help="NMS threshold")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Set batch size")
    parser.add_argument("--yolov3", type=str, default='yes',
                        help="yolov2 or yolov3")
    parser.add_argument("--spp_net", type=str, default="false",
                        help="yolov3 spp")
    parser.add_argument("--tiny", type=str, default="false",
                        help="yolov3 tiny")

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
    spp_net = True if args.spp_net == "true" else False
    tiny = True if args.tiny == "true" else False
    print("net_input_dims", net_input_dims)
    print("obj_threshold", obj_threshold)
    print("nms_threshold", nms_threshold)
    print("yolov3", yolov3)
    print("spp_net", spp_net)
    print("tiny", tiny)

    image = cv2.imread(args.input_file)
    image_x = preprocess(image, net_input_dims)

    image_x = np.expand_dims(image_x, axis=0)
    inputs = image_x
    for i in range(1, args.batch_size):
      inputs = np.append(inputs, image_x, axis=0)

    caffemodel = CaffeModel()
    caffemodel.load_model(args.model_def, args.pretrained_model)
    caffemodel.inference(inputs)
    outputs = caffemodel.net.blobs

    all_tensor_dict = caffemodel.get_all_tensor(inputs, args.dump_blobs_with_inplace)
    np.savez(args.dump_blobs, **all_tensor_dict)

    # dump weight to file
    if args.dump_weights is not None:
        print("Save Weights:", args.dump_weights)
        weights_dict = caffemodel.get_all_weights()
        np.savez(args.dump_weights, **weights_dict)

    out_feat = {}
    if yolov3 == True:
        if tiny:
            out_feat['layer16-conv'] = outputs['layer16-conv'].data
            out_feat['layer23-conv'] = outputs['layer23-conv'].data
            batched_predictions = postprocess_v3_tiny(out_feat, image.shape, net_input_dims,
                                    obj_threshold, nms_threshold, args.batch_size)
        else:
            if not spp_net:
                out_feat['layer82-conv'] = outputs['layer82-conv'].data
                out_feat['layer94-conv'] = outputs['layer94-conv'].data
                out_feat['layer106-conv'] = outputs['layer106-conv'].data
                batched_predictions = postprocess_v3(out_feat, image.shape, net_input_dims,
                                        obj_threshold, nms_threshold, spp_net, args.batch_size)
            else:
                out_feat['layer89-conv'] = outputs['layer89-conv'].data
                out_feat['layer101-conv'] = outputs['layer101-conv'].data
                out_feat['layer113-conv'] = outputs['layer113-conv'].data
                batched_predictions = postprocess_v3(out_feat, image.shape, net_input_dims,
                                        obj_threshold, nms_threshold, spp_net, args.batch_size)
    else:
        out_feat['conv22'] = outputs['conv22'].data
        batched_predictions = postprocess_v2(out_feat, image.shape, net_input_dims,
                                obj_threshold, nms_threshold, args.batch_size)
    print(batched_predictions[0])
    if (args.draw_image != ''):
        image = draw(image, batched_predictions[0], args.label_file)
        cv2.imwrite(args.draw_image, image)


if __name__ == '__main__':
    main(sys.argv)
