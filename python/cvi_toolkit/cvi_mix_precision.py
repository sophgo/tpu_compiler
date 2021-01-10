#!/usr/bin/env python3
import csv
import argparse
import os, math
import numpy as np
import time

from cvi_toolkit.data.preprocess import get_preprocess_parser, preprocess
from cvi_toolkit.utils.yolov3_util import preprocess as _preprocess_yolov3
from cvi_toolkit.utils.math_function import cal_sigmoid, cal_sqnr
from cvi_toolkit.utils.mlir_shell import gen_bf16_mlir
from cvi_toolkit.mix_precision.MixPrecision import MixPrecisior

import logging


from cvi_toolkit.utils.log_setting import setup_logger
logger = setup_logger('root')
log_flag = logger.level <= logging.DEBUG

def parse_args():
    parser = argparse.ArgumentParser(description="Generate bf16 table")
    parser.add_argument('fp32_cali_mlir_file', metavar='fp32_cali_mlir_file', help='Model file')
    parser.add_argument('image_list_file', metavar='image_list_file', help='Input image list file')
    parser.add_argument('output_bf16_table', metavar='output_bf16_layer_file', help='Output bf16 layer table')
    parser.add_argument('--output_mlir', metavar='output mix precision mlir', help='output mix precision mlir')
    parser.add_argument('--model_name', metavar='model_name', help='Model name', default='generic')
    parser.add_argument('--number_bf16', metavar='number of swich int8 to bf16', help='number of bf16 layer', type=int, default=10)
    parser.add_argument('--input_num', metavar='input_num', help='Calibration data number', type=int, default=10)
    parser.add_argument('--loss_table', metavar='loss_table', help='Loss table', type=str, default='loss_table')
    parser = get_preprocess_parser(existed_parser=parser)
    args = parser.parse_args()

    return args


def generic_loss(bf16_preds, int8_dequant_preds):
    ret = 0
    neuron_count = 0

    for op_name in bf16_preds:
        bf16_pred = bf16_preds[op_name]
        int8_dequant_pred = int8_dequant_preds[op_name]
        loss = cal_sqnr(bf16_pred, int8_dequant_pred)
        if not math.isinf(loss):
            ret += -loss * bf16_pred.size
            neuron_count += bf16_pred.size

    if ret == 0 and neuron_count == 0:
        return -math.inf
    else:
        return ret / neuron_count


def yolo_loss(bf16_preds, int8_dequant_preds, yolo_w, yolo_h, yolo_config):
    ret = 0
    effective_loss = 0

    def yolo_bbox_loc(pred, yolo_w, yolo_h, anchors=None):
        num_boxes_per_cell = 3
        num_of_class = 80

        grid_size = pred.shape[2]
        out = np.transpose(pred, (0, 2, 3, 1))
        out = np.reshape(out, (grid_size, grid_size, num_boxes_per_cell, 5 + num_of_class))

        if anchors == None:
            return out[..., 0:4]

        anchors_tensor = np.array(anchors).reshape(1, 1, 3, 2)
        box_xy = cal_sigmoid(out[..., :2])
        box_wh = np.exp(out[..., 2:4]) * anchors_tensor

        col = np.tile(np.arange(0, grid_size), grid_size).reshape(-1, grid_size)
        row = np.tile(np.arange(0, grid_size).reshape(-1, 1), grid_size)

        col = col.reshape(grid_size, grid_size, 1, 1).repeat(3, axis=-2)
        row = row.reshape(grid_size, grid_size, 1, 1).repeat(3, axis=-2)
        grid = np.concatenate((col, row), axis=-1)

        box_xy += grid
        box_xy /= (grid_size, grid_size)
        box_wh /= (yolo_w, yolo_h)

        boxes = np.concatenate((box_xy, box_wh), axis=-1)
        return boxes.flatten()

    for op_name, anchors in yolo_config:
        bf16_pred = yolo_bbox_loc(bf16_preds[op_name], yolo_w, yolo_h, anchors)
        int8_dequant_pred = yolo_bbox_loc(int8_dequant_preds[op_name], yolo_w, yolo_h, anchors)
        loss = cal_sqnr(bf16_pred, int8_dequant_pred)
        if not math.isinf(loss):
            ret += -loss
            effective_loss += 1

    return ret / effective_loss


def preprocess_yolov3(bgr_img, net_input_dims):
    y = _preprocess_yolov3(bgr_img, net_input_dims)
    y = np.expand_dims(y, axis=0)
    return y


if __name__ == '__main__':
    args = parse_args()

    if (args.model_name == 'generic'):
        preprocessor = preprocess()
        preprocessor.config(**vars(args))
        # read with opencv, bgr, hwc
        p_func = lambda input_tensor: preprocessor.run(input_tensor)
        mix_precisior = MixPrecisior(args.fp32_cali_mlir_file, generic_loss, args.image_list_file,
                                        precrocess_func=p_func, input_num=args.input_num)
    elif (args.model_name == 'yolo_v3_320'):
        preprocess = lambda input_tensor: preprocess_yolov3(input_tensor, [320, 320])
        yolo_config = [('layer82-conv_dequant', [116, 90, 156, 198, 373, 326]),
            ('layer94-conv_dequant', [30, 61, 62, 45, 59, 119]),
            ('layer106-conv_dequant', [10, 13, 16, 30, 33, 23])]
        yolov3_320_loss = lambda bf16, int8_dequant: yolo_loss(bf16, int8_dequant, 320, 320, yolo_config)

        mix_precisior = MixPrecisior(args.fp32_cali_mlir_file, yolov3_320_loss, args.image_list_file,
                                        precrocess_func=preprocess, input_num=args.input_num)
    elif (args.model_name == 'mobilenet_yolo_v3_320'):
        preprocess = lambda input_tensor: preprocess_yolov3(input_tensor, [320, 320])
        yolo_config = [('layer35-conv_dequant', [116, 90, 156, 198, 373, 326]),
            ('layer48-conv_dequant', [30, 61, 62, 45, 59, 119]),
            ('layer61-conv_dequant', [10, 13, 16, 30, 33, 23])]
        yolov3_320_loss = lambda bf16, int8_dequant: yolo_loss(bf16, int8_dequant, 320, 320, yolo_config)

        mix_precisior = MixPrecisior(args.fp32_cali_mlir_file, yolov3_320_loss, args.image_list_file,
                                        precrocess_func=preprocess, input_num=args.input_num)
    elif (args.model_name == 'yolo_v3_416'):
        preprocess = lambda input_tensor: preprocess_yolov3(input_tensor, [416, 416])
        yolo_config = [('layer82-conv_dequant', [116, 90, 156, 198, 373, 326]),
            ('layer94-conv_dequant', [30, 61, 62, 45, 59, 119]),
            ('layer106-conv_dequant', [10, 13, 16, 30, 33, 23])]
        yolov3_416_loss = lambda bf16, int8_dequant: yolo_loss(bf16, int8_dequant, 416, 416, yolo_config)

        mix_precisior = MixPrecisior(args.fp32_cali_mlir_file, yolov3_416_loss, args.image_list_file,
                                        precrocess_func=preprocess, input_num=args.input_num)
    else:
        assert(False)

    sort_bf16_layers = mix_precisior.run()
    with open(args.loss_table, "w") as f:
        for idx, layer in enumerate(sort_bf16_layers):
            loss_msg = "No.{:<4}: Layer: {:<50}\t\tLoss: {}".format(idx, layer[0], layer[1])
            f.write("{}\n".format(loss_msg))
            print(loss_msg)


    with open(args.output_bf16_table, "w") as f:
        sort_bf16_layers = sort_bf16_layers[:args.number_bf16]
        for i in sort_bf16_layers:
            f.write("{}\n".format(i[0]))
    print("Output bf16 table to {}".format(args.output_bf16_table))

    if args.output_mlir:
        gen_bf16_mlir(args.fp32_cali_mlir_file , args.output_mlir, args.output_bf16_table)
        print("gen bf16 mix precision mlir => {}".format(args.output_bf16_table))
