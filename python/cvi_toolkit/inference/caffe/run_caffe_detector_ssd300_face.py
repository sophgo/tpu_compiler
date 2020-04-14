#!/usr/bin/env python3

import numpy as np
import os
import sys
import argparse
import glob
import time
import cv2
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format

def parse_args():
    parser = argparse.ArgumentParser(description='Eval ResNet-10 networks.')
    parser.add_argument('--model_def', type=str, default='',
                        help="Model definition file")
    parser.add_argument('--pretrained_model', type=str, default='',
                        help='Load weights from previously saved parameters.')
    parser.add_argument("--net_input_dims", default='300,300',
                        help="'height,width' dimensions of net input tensors.")
    parser.add_argument("--input_file", type=str, default='',
                        help="Input image for testing")
    parser.add_argument("--draw_image", type=str, default='',
                        help="Draw results on image")
    parser.add_argument("--dump_blobs",
                        help="Dump all blobs into a file in npz format")
    parser.add_argument("--dump_weights",
                        help="Dump all weights into a file in npz format")
    parser.add_argument("--force_input",
                        help="Force the input blob data, in npy format")
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

def ssd300_face_detect(net, image_path, net_input_dims,
                  dump_blobs=None, dump_weights=None):
    
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))  # row to col, (HWC -> CHW)
    transformer.set_mean('data', np.array([104, 177, 123], dtype=np.float32))
    transformer.set_raw_scale('data', 255)  # [0,1] to [0,255]

    image = caffe.io.load_image(image_path)  # range from 0 to 1

    net.blobs['data'].reshape(1, 3, net_input_dims[0], net_input_dims[1])
    net.blobs['data'].data[...] = transformer.preprocess('data', image)
    net.blobs['data'].data[...].tofile("test_girl_in_fp32.bin")

    detections = net.forward()['detection_out']
    x = net.blobs['data'].data[...]

    # [DEBUG] dump blobs to file
    if dump_blobs is not None:
        print("Save Blobs: ", dump_blobs)
        blobs_dict = {}
        # for name, blob in self.blobs.items():
        #     blobs_dict[name] = blob.data
        for name, layer in net.layer_dict.items():
            print("layer : " + str(name))
            print("  type = " + str(layer.type))
            print("  top -> " + str(net.top_names[name]))
            if layer.type == "Split":
                print("  skip Split")
                continue
            assert(len(net.top_names[name]) == 1)
            if layer.type == "Input":
                blobs_dict['data'] = x
                continue
            #out = self.forward(None, prev_name, name, **{prev_name: prev_data})
            out = net.forward(None, None, name, **{net.inputs[0]: x})
            blobs_dict[name] = out[net.top_names[name][0]].copy()
        np.savez(dump_blobs, **blobs_dict)

    if dump_weights is not None:
        print("Save Weights:", dump_weights)
        weights_dict = {}
        for name, param in net.params.items():
            for i in range(len(param)):
                weights_dict[name + "_" + str(i)] = param[i].data
        np.savez(dump_weights, **weights_dict)

    return detections

def main(argv):
    args = parse_args()

    # Make Detector
    net_input_dims = [int(s) for s in args.net_input_dims.split(',')]
    obj_threshold = float(args.obj_threshold)

    print("net_input_dims", net_input_dims)
    print("obj_threshold", obj_threshold)

    net = caffe.Net(args.model_def, args.pretrained_model, caffe.TEST)

    # Load image
    if (args.input_file != '') :
        image = cv2.imread(args.input_file)
        predictions = ssd300_face_detect(net, args.input_file, net_input_dims,
                                    args.dump_blobs, args.dump_weights)

        top_conf, bboxs = parse_top_detection(image.shape, predictions, obj_threshold)


        if (args.draw_image != ''):
            image = draw(image, top_conf, bboxs,True)
            cv2.imwrite(args.draw_image, image)

    else :
        print("No input_file specified")
        exit(1)


if __name__ == '__main__':
    main(sys.argv)
