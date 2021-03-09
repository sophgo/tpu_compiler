#!/usr/bin/env python3
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
ssd_mean = np.array([127.5, 127.5, 127.5], dtype=np.float32)
input_scale = 0.007843

def parse_args():
    parser = argparse.ArgumentParser(description='Eval YOLO networks.')
    parser.add_argument("--net_input_dims", default='300,300',
                        help="'height,width' dimensions of net input tensors.")
    parser.add_argument("--input_file", type=str, default='',
                        help="Input image for testing")
    parser.add_argument("--label_file", type=str, default='',
                        help="coco lable file in txt format")
    parser.add_argument("--draw_image", type=str, default='',
                        help="Draw results on image")
    parser.add_argument("--dump_blobs",
                        help="Dump all blobs into a file in npz format")
    parser.add_argument("--obj_threshold", type=float, default=0.1,
                        help="Object confidence threshold")

    parser.add_argument('--model', type=str, default='',
                        help="MLIR Model file")
    args = parser.parse_args()
    return args


def draw(image, top_label_names,top_confs, bboxs,verbose):
    image = np.copy(image)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # https://github.com/amikelive/coco-labels

    for i  in range(len(bboxs)):
        x, y, w, h = bboxs[i]
        score = top_confs[i]
        cls = top_label_names[i]

        x1 = max(0, np.floor(x + 0.5).astype(int))
        y1 = max(0, np.floor(y + 0.5).astype(int))

        x2 = min(image.shape[0], np.floor(x + w ).astype(int))
        y2 = min(image.shape[1], np.floor(y + h ).astype(int))

        cv2.rectangle(image, (x1,y1), (x2,y2), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(cls, score),
                        (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,0,255), 1, cv2.LINE_AA)

        if verbose:
            print('class: {0}, score: {1:.2f}'.format(cls, score))
            print('box coordinate x, y, w, h: {0}'.format(bboxs[i]))

    return image




def parse_top_detection(resolution, detections, conf_threshold=0.1):
    det_label = detections[0, 0, :, 1]
    det_conf = detections[0, 0, :, 2]
    det_xmin = detections[0, 0, :, 3]
    det_ymin = detections[0, 0, :, 4]
    det_xmax = detections[0, 0, :, 5]
    det_ymax = detections[0, 0, :, 6]

    #print(det_label,det_conf,det_xmin,det_ymin,det_xmax,det_ymax)
    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_threshold]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].astype(int).tolist()
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

    return top_label_indices, top_conf, bboxs

def get_label_name(labelmap, labels):
    num_labels = len(labelmap.item)
    label_names = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        for i in range(0, num_labels):
            if label == labelmap.item[i].label:
                label_names.append(labelmap.item[i].display_name)
                break
    return label_names

def ssd_detect(model,image_path, net_input_dims,
                  dump_blobs=None):

    image = caffe.io.load_image(image_path)  # range from 0 to 1

    # net.blobs['data'].reshape(1, 3, net_input_dims[0], net_input_dims[1])
    # net.blobs['data'].data[...] = transformer.preprocess('data', image)
    # detections = net.forward()['detection_out']
    module = pymlir.module()
    module.load(model)


    net_input_dims = [int(s) for s in args.net_input_dims.split(',')]

    transformer = caffe.io.Transformer({'data': (1,3,net_input_dims[0],net_input_dims[1])})
    transformer.set_transpose('data', (2, 0, 1))  # row to col, (HWC -> CHW)
    transformer.set_mean('data', ssd_mean)
    transformer.set_raw_scale('data', 255)  # [0,1] to [0,255]
    transformer.set_input_scale('data', input_scale)
    transformer.set_channel_swap('data', (2, 1, 0))  # RGB to BGR

    x = transformer.preprocess('data', image)
    x = np.expand_dims(x, axis=0)
    _ = module.run(x)
    data = module.get_all_tensor()
    detections = data['detection_out']


    # # [DEBUG] dump blobs to file
    # if dump_blobs is not None:
    #     print("Save Blobs: ", dump_blobs)
    #     blobs_dict = {}
    #     for name, blob in net.blobs.items():
    #         blobs_dict[name] = blob.data
    #     np.savez(dump_blobs, **blobs_dict)

    return detections



def main(argv):

    # Make Detector
    net_input_dims = [int(s) for s in args.net_input_dims.split(',')]
    obj_threshold = float(args.obj_threshold)

    print("net_input_dims", net_input_dims)
    print("obj_threshold", obj_threshold)

    labelmap = caffe_pb2.LabelMap()

    file = open(args.label_file, 'r')
    text_format.Merge(str(file.read()), labelmap)
    file.close()

    # Load image
    if (args.input_file != '') :
        image = cv2.imread(args.input_file)
        predictions = ssd_detect(args.model,args.input_file, net_input_dims,
                                    args.dump_blobs)

        top_label_indices, top_conf, bboxs = parse_top_detection(image.shape, predictions, obj_threshold)
        top_label_name = get_label_name(labelmap, top_label_indices)

        # print(top_label_name)
        # print(top_label_indices, top_conf, bboxs)

        if (args.draw_image != ''):
            image = draw(image, top_label_name,top_conf, bboxs,True)
            cv2.imwrite(args.draw_image, image)

    else :
        print("No input_file specified")
        exit(1)


if __name__ == '__main__':

    args = parse_args()

    main(sys.argv)
