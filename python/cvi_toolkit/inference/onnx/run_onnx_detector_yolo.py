#!/usr/bin/env python3

import numpy as np
import os
import sys
import argparse
import glob
import time
import onnx
import onnxruntime
import cv2
import caffe
from cvi_toolkit.model import OnnxModel
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
    parser.add_argument("--dump_blobs",
                        help="Dump all blobs into a file in npz format")
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
    yolov4_tiny = True if args.yolov4_tiny == 'yes' else False
    print("net_input_dims", net_input_dims)
    print("obj_threshold", obj_threshold)
    print("nms_threshold", nms_threshold)
    print("yolov3", yolov3)
    print("yolov4_tiny", yolov4_tiny)

    image = cv2.imread(args.input_file)
    image_x = preprocess(image, net_input_dims)

    image_x = np.expand_dims(image_x, axis=0)
    inputs = image_x
    for i in range(1, args.batch_size):
      inputs = np.append(inputs, image_x, axis=0)
    input_shape = np.array([net_input_dims[0], net_input_dims[1]], dtype=np.float32).reshape(1, 2)
    ort_session = onnxruntime.InferenceSession(args.model_def)
    ort_inputs = {'input': inputs}
    ort_outs = ort_session.run(None, ort_inputs)

    out_feat = {}
    if yolov4_tiny:
        batched_predictions = postprocess_v4_tiny(ort_outs, image.shape, net_input_dims,
                                obj_threshold, nms_threshold, args.batch_size)
    else:
        out_feat['layer82-conv'] = ort_outs[0]
        out_feat['layer94-conv'] = ort_outs[1]
        out_feat['layer106-conv'] = ort_outs[2]
        batched_predictions = postprocess_v3(out_feat, image.shape, net_input_dims,
                                    obj_threshold, nms_threshold, False, args.batch_size)
    print(batched_predictions[0])
    if args.draw_image:
        image = draw(image, batched_predictions[0], args.label_file)
        cv2.imwrite(args.draw_image, image)

    if args.dump_blobs:
        # second pass for dump all output
        # plz refre https://github.com/microsoft/onnxruntime/issues/1455
        output_keys = []
        for i in range(len(ort_outs)):
            output_keys.append('output_{}'.format(i))

        model = onnx.load(args.model_def)

        # tested commited #c3cea486d https://github.com/microsoft/onnxruntime.git
        for x in model.graph.node:
            _intermediate_tensor_name = list(x.output)
            intermediate_tensor_name = ",".join(_intermediate_tensor_name)
            intermediate_layer_value_info = onnx.helper.ValueInfoProto()
            intermediate_layer_value_info.name = intermediate_tensor_name
            model.graph.output.append(intermediate_layer_value_info)
            output_keys.append(intermediate_layer_value_info.name + '_' + x.op_type)

        dump_all_onnx = "dump_all.onnx"
        if not os.path.exists(dump_all_onnx):
            onnx.save(model, dump_all_onnx)
        else:
            print("{} is exitsed!".format(dump_all_onnx))
        print("dump multi-output onnx all tensor at ", dump_all_onnx)

        # dump all inferneced tensor
        ort_session = onnxruntime.InferenceSession(dump_all_onnx)
        ort_outs = ort_session.run(None, ort_inputs)
        tensor_all_dict = dict(zip(output_keys, map(np.ndarray.flatten, ort_outs)))
        tensor_all_dict['input'] = inputs
        np.savez(args.dump_blobs, **tensor_all_dict)
        print("dump all tensor at ", args.dump_blobs)


if __name__ == '__main__':
    main(sys.argv)
