#!/usr/bin/env python3
##
# Copyright (C) Cristal Vision Technologies Inc.
# All Rights Reserved.
##

import argparse
import sys
import os
import cv2
import numpy as np


from cvi_toolkit.calibration.base_calibrator import Base_Calibrator
from cvi_toolkit.calibration.kld_calibrator import KLD_Calibrator
from cvi_toolkit.calibration.tuner import Tuner_v2
from cvi_toolkit import preprocess
from cvi_toolkit.data.preprocess import get_preprocess_parser


def preprocess_func_faster_rcnn(image_path):
    image = cv2.imread(str(image_path).rstrip())
    image = cv2.resize(image, (800, 600))
    image = image.astype(np.float32, copy=True)
    image -= np.array([[[102.9801, 115.9465, 122.7717]]])
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    return image


def preprocess_func_gaitset(image_path):
    image = cv2.imread(str(image_path).rstrip(), cv2.IMREAD_GRAYSCALE)
    image = image.astype(np.float32)
    image = cv2.resize(image, (64, 64))
    x = image / 255.0
    x = np.expand_dims(x, axis=0)
    x = np.expand_dims(x, axis=1)
    return x


def preprocess_func_unet(image_path):
    image = cv2.imread(str(image_path).rstrip(), cv2.IMREAD_GRAYSCALE)
    image = image.astype(np.float32)
    x = cv2.resize(image, (256, 256))
    x = np.expand_dims(x, axis=0)
    x = np.expand_dims(x, axis=1)
    return x


def preprocess_func_espcn(image_path):
    image = cv2.imread(str(image_path).rstrip())
    image = cv2.resize(image, (85, 85))
    image = image / 255.0
    image[:, :, 0], image[:, :, 2] = image[:, :, 2], image[:, :, 0]
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image


def preprocess_func_arcface(image_path):
    image = cv2.imread(str(image_path).rstrip())
    image[:, :, 0], image[:, :, 2] = image[:, :, 2], image[:, :, 0]
    image = image.astype(np.float32)
    image = image - 127.5
    image = image * 0.0078125
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image


def preprocess_func_ssd300_face(image_path):
    imgnet_mean = np.array([104, 177, 123], dtype=np.float32)
    image = cv2.imread(str(image_path).rstrip())
    image = image.astype(np.float32)
    image -= imgnet_mean
    x = cv2.resize(image, (300, 300))
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0)
    return x


def preprocess_func_alphapose(npz_path):
    x = np.load(str(npz_path).rstrip())
    return x


def preprocess_yolov3(image_path, net_input_dims=(416, 416)):
    bgr_img = cv2.imread(str(image_path).rstrip())
    yolo_w = net_input_dims[1]
    yolo_h = net_input_dims[0]
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    rgb_img = rgb_img / 255.0

    ih = rgb_img.shape[0]
    iw = rgb_img.shape[1]

    scale = min(float(yolo_w) / iw, float(yolo_h) / ih)
    rescale_w = int(iw * scale)
    rescale_h = int(ih * scale)

    resized_img = cv2.resize(
        rgb_img, (rescale_w, rescale_h), interpolation=cv2.INTER_LINEAR)
    new_image = np.full((yolo_h, yolo_w, 3), 0, dtype=np.float32)
    paste_w = (yolo_w - rescale_w) // 2
    paste_h = (yolo_h - rescale_h) // 2

    new_image[paste_h:paste_h + rescale_h,
              paste_w: paste_w + rescale_w, :] = resized_img
    # row to col, (HWC -> CHW)
    new_image = np.transpose(new_image, (2, 0, 1))
    new_image = np.expand_dims(new_image, axis=0)
    return new_image


def center_crop(img, crop_dim):
    # print(img.shape)
    h, w, _ = img.shape
    cropy, cropx = crop_dim
    startx = w//2-(cropx//2)
    starty = h//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx, :]


def preprocess_generic(image_path, args):
    image_resize_dims = [int(s) for s in args.image_resize_dims.split(',')]
    net_input_dims = [int(s) for s in args.net_input_dims.split(',')]
    image_resize_dims = [max(x, y)
                         for (x, y) in zip(image_resize_dims, net_input_dims)]

    if args.raw_scale:
        raw_scale = float(args.raw_scale)
    else:
        raw_scale = 255.0
    if args.mean:
        mean = np.array([float(s)
                         for s in args.mean.split(',')], dtype=np.float32)
        # print('mean', mean)
        mean = mean[:, np.newaxis, np.newaxis]
    else:
        mean = np.array([])
    if args.input_scale:
        input_scale = float(args.input_scale)
    else:
        input_scale = 1.0

    image = cv2.imread(str(image_path).rstrip())
    image = image.astype(np.float32)
    # resize
    x = cv2.resize(image, (image_resize_dims[1], image_resize_dims[0]))  # w,h
    # Take center crop.
    x = center_crop(x, net_input_dims)
    # transpose
    x = np.transpose(x, (2, 0, 1))
    # preprocess
    x = x * raw_scale / 255.0
    if mean.size != 0:
        x -= mean
    if input_scale != 1.0:
        x *= input_scale
    x = np.expand_dims(x, axis=0)
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', metavar='model_file', help='Model file')
    parser.add_argument(
        'image_list_file', metavar='image_list_file', help='Input image list file')
    parser.add_argument(
        '--output_file', metavar='output_file', help='Output file')
    parser.add_argument('--output_density_table', metavar='output_density_table',
                        help='Output density file for look-up table')
    parser.add_argument('--model_name', metavar='model_name',
                        help='Model name', default='generic')
    parser.add_argument('--calibrator', metavar='calibrator',
                        help='Calibration method', default='KLD')
    parser.add_argument('--asymmetric', action='store_true',
                        help='Type of quantization ragne')
    parser.add_argument('--math_lib_path', metavar='math_path',
                        help='Calibration math library path', default='calibration_math.so')
    parser.add_argument('--input_num', metavar='input_num',
                        help='Calibration data number', default=10)
    parser.add_argument('--histogram_bin_num', metavar='histogram_bin_num', help='Specify histogram bin numer for kld calculate',
                        default=2048)
    parser.add_argument('--auto_tune', action='store_true',
                        help='Enable auto tune or not')
    parser.add_argument(
        '--binary_path', metavar='binary_path', help='MLIR binary path')
    parser.add_argument('--tune_iteration', metavar='iteration', type=int,
                        help='The number of data using in tuning process', default=10)
    parser.add_argument('--dataset_file_path', metavar='dataset_file_path',
                        type=str, help='file path that recode dataset images path')
    parser.add_argument('--custom_op_plugin', metavar='custom_op_plugin',
                        help='set file path of custom op plugin', default='')
    parser = get_preprocess_parser(existed_parser=parser)
    args = parser.parse_args()

    if (args.model_name == 'generic'):
        preprocessor = preprocess()
        preprocessor.config(net_input_dims=args.net_input_dims,
                            resize_dims=args.image_resize_dims,
                            mean=args.mean,
                            mean_file=args.mean_file,
                            input_scale=args.input_scale,
                            raw_scale=args.raw_scale,
                            std=args.std,
                            rgb_order=args.model_channel_order,
                            bgray=args.bgray)

        def p_func(input_file): return preprocessor.run(
            input_file, output_channel_order=args.model_channel_order)
    elif (args.model_name == 'yolo_v3'):
        p_func = preprocess_yolov3
    elif (args.model_name == 'ssd300_face'):
        p_func = preprocess_func_ssd300_face
    elif (args.model_name == 'alpha_pose'):
        p_func = preprocess_func_alphapose
    elif (args.model_name == 'arcface_res50'):
        p_func = preprocess_func_arcface
    elif (args.model_name == 'espcn'):
        p_func = preprocess_func_espcn
    elif (args.model_name == 'unet'):
        p_func = preprocess_func_unet
    elif (args.model_name == 'faster_rcnn'):
        p_func = preprocess_func_faster_rcnn
    elif (args.model_name == 'gaitset'):
        p_func = preprocess_func_gaitset
    else:
        assert(False)

    if args.calibrator == 'KLD':
        calibrator = KLD_Calibrator(image_list_file=args.image_list_file,
                                    mlir_file=args.model_file,
                                    preprocess_func=p_func,
                                    input_num=args.input_num,
                                    histogram_bin_num=args.histogram_bin_num,
                                    math_lib_path=args.math_lib_path,
                                    custom_op_plugin=args.custom_op_plugin)
    else:
        raise RuntimeError("Now only support kld calibrator")

    if args.output_file:
        threshold_table = args.output_file
    else:
        threshold_table = '{}_threshold_table'.format(args.model_name)

    calibrator.create_calibration_table(threshold_table)

    # # export density table for hw look-up table
    # density_table = args.output_density_table if args.output_density_table else '{}_density_table'.format(args.model_name)

    # calibrator.dump_density_table(density_table, calibrator.get_raw_min(), thresholds)

    def autotune_preprocess(image_path):
        return p_func(image_path, None)

    if args.auto_tune == True:
        args.input_threshold_table = threshold_table
        tuner = Tuner_v2(args.model_file, threshold_table, args.dataset_file_path,
                         tune_iteration=args.tune_iteration, preprocess_func=autotune_preprocess)
        tuner.run_tune()


if __name__ == '__main__':
    main()
