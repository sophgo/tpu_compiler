#!/usr/bin/env python

import numpy as np
import os
import sys
import argparse
import glob
import time
import cv2
import caffe
import pymlir
from caffe.proto import caffe_pb2
from google.protobuf import text_format

def parse_args():
    parser = argparse.ArgumentParser(description='Eval ResNet-10 networks.')
    parser.add_argument('--model', type=str, default='',
                        help="MLIR Model file")
    parser.add_argument("--net_input_dims", default='300,300',
                        help="'height,width' dimensions of net input tensors.")
    parser.add_argument("--input_file", type=str, default='',
                        help="Input image for testing")
    parser.add_argument("--draw_image", type=str, default='',
                        help="Draw results on image")
    parser.add_argument("--obj_threshold", type=float, default=0.5,
                        help="Object confidence threshold")


    args = parser.parse_args()
    return args


def draw(image, top_confs, bboxs,verbose):
    image = np.copy(image)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # https://github.com/amikelive/coco-labels

    for i  in range(len(bboxs)):
        xmin, ymin, xmax, ymax = bboxs[i]
        score = top_confs[i]

        x1 = max(0, np.floor(xmin + 0.5).astype(int))
        y1 = max(0, np.floor(ymin + 0.5).astype(int))

        x2 = min(image.shape[1], np.floor(xmax + 0.5).astype(int))
        y2 = min(image.shape[0], np.floor(ymax + 0.5 ).astype(int))

        cv2.rectangle(image, (x1,y1), (x2,y2), (255, 0, 0), 2)
        # cv2.putText(image, '{0} {1:.2f}'.format(cls, score),
                        # (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX,
                        # 0.6, (0,0,255), 1, cv2.LINE_AA)

        if verbose:
            # print('class: {0}, score: {1:.2f}'.format(cls, score))
            print('box coordinate x, y, w, h: {0}'.format(bboxs[i]))

    return image

def parse_top_detection(resolution, detections, conf_threshold=0.6):
    det_conf = detections[0, 0, :, 2]
    det_xmin = detections[0, 0, :, 3]
    det_ymin = detections[0, 0, :, 4]
    det_xmax = detections[0, 0, :, 5]
    det_ymax = detections[0, 0, :, 6]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_threshold]

    top_conf = det_conf[top_indices]
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    bboxs = np.zeros((top_conf.shape[0], 4), dtype=int)
    for i in range(top_conf.shape[0]):
        bboxs[i][0] = int(round(top_xmin[i] * resolution[1]))
        bboxs[i][1] = int(round(top_ymin[i] * resolution[0]))
        bboxs[i][2] = int(round(top_xmax[i] * resolution[1]))
        bboxs[i][3] = int(round(top_ymax[i] * resolution[0]))

    return top_conf, bboxs

def ssd300_face_detect(model, image_path, net_input_dims):
    
    module = pymlir.module()
    module.load(model)

    transformer = caffe.io.Transformer({'data': (1,3,net_input_dims[0],net_input_dims[1])})
    transformer.set_transpose('data', (2, 0, 1))  # row to col, (HWC -> CHW)
    transformer.set_mean('data', np.array([104, 177, 123], dtype=np.float32))
    transformer.set_raw_scale('data', 255)  # [0,1] to [0,255]

    image = caffe.io.load_image(image_path)  # range from 0 to 1

    x = transformer.preprocess('data', image)
    x = np.expand_dims(x, axis=0)
    _ = module.run(x)
    data = module.get_all_tensor()
    detections = data['detection_out']
    return detections

def main(argv):
    args = parse_args()

    # Make Detector
    net_input_dims = [int(s) for s in args.net_input_dims.split(',')]
    obj_threshold = float(args.obj_threshold)

    print("net_input_dims", net_input_dims)
    print("obj_threshold", obj_threshold)

    # Load image
    if (args.input_file != '') :
        image = cv2.imread(args.input_file)
        predictions = ssd300_face_detect(args.model, args.input_file, net_input_dims)

        top_conf, bboxs = parse_top_detection(image.shape, predictions, obj_threshold)


        if (args.draw_image != ''):
            image = draw(image, top_conf, bboxs,True)
            cv2.imwrite(args.draw_image, image)

    else :
        print("No input_file specified")
        exit(1)


if __name__ == '__main__':
    main(sys.argv)
