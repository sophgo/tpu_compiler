#!/usr/bin/env python3

import numpy as np
import os
import sys
import argparse
import glob
import time
import cv2
import pymlir
from cvi_toolkit.data.preprocess import get_preprocess_parser, preprocess
from cvi_toolkit.utils.yolov3_util import draw
from cvi_toolkit.utils.yolov3_util import postprocess_v2
from cvi_toolkit.utils.yolov3_util import postprocess_v3, postprocess_v3_tiny
from cvi_toolkit.utils.yolov3_util import postprocess_v4, postprocess_v4_tiny

def check_files(args):
    if not os.path.isfile(args.model):
        print("cannot find the file %s", args.model)
        sys.exit(1)

    if not os.path.isfile(args.input_file):
        print("cannot find the file %s", args.input_file)
        sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description='Eval YOLO networks.')
    parser = get_preprocess_parser(parser)
    parser.add_argument('--model', type=str, default='',
                        help="MLIR Model file")
    parser.add_argument("--input_file", type=str, default='',
                        help="Input image for testing")
    parser.add_argument("--label_file", type=str, default='',
                        help="coco lable file in txt format")
    parser.add_argument("--draw_image", type=str, default='',
                        help="Draw results on image")
    parser.add_argument("--dump_blobs",
                        help="Dump all blobs into a file in npz format")
    parser.add_argument("--dump_blobs_with_inplace",
                        type=bool, default=False,
                        help="Dump all blobs including inplace blobs (takes much longer time)")
    parser.add_argument("--obj_threshold", type=float, default=0.3,
                        help="Object confidence threshold")
    parser.add_argument("--nms_threshold", type=float, default=0.5,
                        help="NMS threshold")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Set batch size")
    parser.add_argument("--yolov3", type=str, default='yes',
                        help="yolov2 or yolov3")
    parser.add_argument("--yolov4", type=str, default='false',
                        help="yolov4")
    parser.add_argument("--spp_net", type=str, default="false",
                        help="yolov3 spp")
    parser.add_argument("--tiny", type=str, default="false",
                        help="yolov3 tiny")

    args = parser.parse_args()
    check_files(args)
    return args

def main(argv):
    args = parse_args()
    preprocessor = preprocess()
    args.pixel_format = 'RGB_PLANAR'
    args.keep_aspect_ratio = True
    args.raw_scale = 1.0
    preprocessor.config(**vars(args))
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
    inputs = preprocessor.run(image, batch=args.batch_size)

    module = pymlir.module()
    module.load(args.model)
    _ = module.run(inputs)
    all_tensor_dict = module.get_all_tensor()
    outputs = all_tensor_dict

    if args.dump_blobs:
        np.savez(args.dump_blobs, **all_tensor_dict)

    out_feat = {}
    if yolov3 == True:
        if tiny:
            out_feat['layer16-conv'] = outputs['layer16-conv']
            out_feat['layer23-conv'] = outputs['layer23-conv']
            batched_predictions = postprocess_v3_tiny(out_feat, image.shape, net_input_dims,
                                    obj_threshold, nms_threshold, args.batch_size)
        else:
            if not spp_net:
                out_feat['layer82-conv'] = outputs['layer82-conv']
                out_feat['layer94-conv'] = outputs['layer94-conv']
                out_feat['layer106-conv'] = outputs['layer106-conv']
                batched_predictions = postprocess_v3(out_feat, image.shape, net_input_dims,
                                        obj_threshold, nms_threshold, spp_net, args.batch_size)
            else:
                out_feat['layer89-conv'] = outputs['layer89-conv']
                out_feat['layer101-conv'] = outputs['layer101-conv']
                out_feat['layer113-conv'] = outputs['layer113-conv']
                batched_predictions = postprocess_v3(out_feat, image.shape, net_input_dims,
                                        obj_threshold, nms_threshold, spp_net, args.batch_size)
    elif args.yolov4 == 'true':
        if tiny:
            #out_feat['layer30-conv'] = outputs['layer30-conv']
            #out_feat['layer37-conv'] = outputs['layer37-conv']
            out_feat[0] = outputs['layer30-conv']
            out_feat[1] = outputs['layer37-conv']
            batched_predictions = postprocess_v4_tiny(out_feat, image.shape, net_input_dims,
                                    obj_threshold, nms_threshold, args.batch_size)
        else:
            out_feat['layer139-conv'] = outputs['layer139-conv']
            out_feat['layer150-conv'] = outputs['layer150-conv']
            out_feat['layer161-conv'] = outputs['layer161-conv']
            batched_predictions = postprocess_v4(out_feat, image.shape, net_input_dims,
                    obj_threshold, nms_threshold, spp_net, args.batch_size)

    else:
        out_feat['conv22'] = outputs['conv22']
        batched_predictions = postprocess_v2(out_feat, image.shape, net_input_dims,
                                obj_threshold, nms_threshold, args.batch_size)
    print(batched_predictions[0])
    if (args.draw_image != ''):
        image = draw(image, batched_predictions[0], args.label_file)
        cv2.imwrite(args.draw_image, image)


if __name__ == '__main__':
    main(sys.argv)
