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
import tensorflow as tf
from cvi_toolkit.utils.yolov3_util import preprocess, postprocess_v2, postprocess_v3, draw
from cvi_toolkit.utils.tf_utils import from_saved_model, tf_session, tf_node_name
tf_reset_default_graph = tf.compat.v1.reset_default_graph

def check_files(args):
    if not os.path.isfile(args.model_def):
        print("cannot find the file %s", args.model_def)
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
    image_x = np.transpose(image_x, (0, 2, 3, 1))
    tf_graph, inputs, outputs = from_saved_model(args.model_def)
    net = tf.saved_model.load(args.model_def)

    features = net(image_x)

    out_feat = dict()
    out_feat['layer82-conv'] = np.transpose(features[0], (0, 3, 1, 2))
    out_feat['layer94-conv'] = np.transpose(features[1], (0, 3, 1, 2))
    out_feat['layer106-conv'] = np.transpose(features[2], (0, 3, 1, 2))
    batched_predictions = postprocess_v3(out_feat, image.shape, net_input_dims,
                                obj_threshold, nms_threshold, False, args.batch_size)

    if args.draw_image:
        image = draw(image, batched_predictions[0], args.label_file)
        cv2.imwrite(args.draw_image, image)

    if args.dump_blobs:
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(tf_graph, name='')
        with tf_session(graph=graph):
            node_list = graph.get_operations()
        all_op_info = [i.name for i in node_list]
        valid_op = all_op_info[:]

        tf_reset_default_graph()
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(tf_graph, name='')
        with tf_session(graph=graph) as sess:
            output_tensor_list = list()
            input_tensor = sess.graph.get_tensor_by_name(inputs[0])
            for op in all_op_info:
                try:
                    output_tensor_list.append(
                        sess.graph.get_tensor_by_name("{}:0".format(op)))
                except KeyError as key_err:
                    print("skip op {}".format(op))
                    valid_op.remove(op)
            output = sess.run(tuple(output_tensor_list),
                              feed_dict={input_tensor: image_x})

        all_tensor_dict = dict(zip(valid_op, output))
        all_tensor_dict['input'] = all_tensor_dict[tf_node_name(inputs[0])]
        all_tensor_dict['output'] = output[-1]
        tf_reset_default_graph()

        np.savez(args.dump_blobs, **all_tensor_dict)


if __name__ == '__main__':
    main(sys.argv)
