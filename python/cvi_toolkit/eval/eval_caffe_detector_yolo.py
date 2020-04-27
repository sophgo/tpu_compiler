#!/usr/bin/env python3

import numpy as np
import os
import sys
import argparse
import glob
import time
import cv2
import caffe
from cvi_toolkit.utils.yolov3_util import preprocess, postprocess, draw
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
from pathlib import Path
from tqdm import tqdm

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
    parser.add_argument("--force_input",
                        help="Force the input blob data, in npy format")
    parser.add_argument("--obj_threshold", type=float, default=0.3,
                        help="Object confidence threshold")
    parser.add_argument("--nms_threshold", type=float, default=0.5,
                        help="NMS threshold")
    parser.add_argument("--dataset", type=str, default='./val2017_5k.txt',
                        help="dataset image list file")
    parser.add_argument("--annotations", type=str, default='./instances_val2017.json',
                        help="annotations file")
    parser.add_argument("--result_json", type=str,
                        help="Result json file")
    parser.add_argument("--pre_result_json", type=str,
                        help="when present, use pre detected result file, skip detection")
    parser.add_argument("--count", type=int, default=-1)

    args = parser.parse_args()
    return args

def yolov3_detect(net, image, net_input_dims, obj_threshold, nms_threshold):
    x = preprocess(image, net_input_dims)
    net.blobs['data'].data[...] = x
    y = net.forward()
    out_feat = {}
    out_feat['layer82-conv'] = y['layer82-conv']
    out_feat['layer94-conv'] = y['layer94-conv']
    out_feat['layer106-conv'] = y['layer106-conv']
    batched_predictions = postprocess(out_feat, image.shape, net_input_dims,
                              obj_threshold, nms_threshold, batch=1)
    # batch = 1
    predictions = batched_predictions[0]
    return predictions

def get_image_id_in_path(image_path):
    stem = Path(image_path).stem
    # in val2014, name would be like COCO_val2014_000000xxxxxx.jpg
    # in val2017, name would be like 000000xxxxxx.jpg
    if (stem.rfind('_') == -1):
        id = int(stem)
    else:
        id = int(stem[stem.rfind('_') + 1:])
    return id

def clip_box(box, image_shape):
    x, y, w, h = box
    xmin = max(0, x)
    ymin = max(0, y)
    xmax = min(image_shape[1], x + w)
    ymax = min(image_shape[0], y + h)

    bx = xmin
    by = ymin
    bw = xmax - xmin
    bh = ymax - ymin
    return np.array([bx, by, bw, bh])

def eval_detector(net, result_json_file, dataset_path, net_input_dims,
                  obj_threshold, nms_threshold, count=-1):
    coco_ids= [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 13, 14, 15, 16, 17,
               18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
               37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
               54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
               74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

    jdict = []
    image_list = os.listdir(dataset_path)
    i=0
    with tqdm(total= count if count > 0 else len(image_list)) as pbar:
        for image_name in image_list:
            image_path = os.path.join(dataset_path, image_name)
            # print(image_path)
            if (not os.path.exists(image_path)):
                print("image not exit,", image_path)
                exit(-1)
            image_id = get_image_id_in_path(image_path)
            image = cv2.imread(image_path)
            predictions = yolov3_detect(net, image, net_input_dims,
                                        obj_threshold, nms_threshold)
            for pred in predictions:
                clipped_box = clip_box(pred[0], image.shape)
                x, y, w, h = clipped_box
                score = pred[1]
                cls = pred[2]
                jdict.append({
                    "image_id": image_id,
                    "category_id": coco_ids[cls],
                    "bbox": [x, y, w, h],
                    "score": float(score)
                })
            i += 1
            if (i == count):
                break
            pbar.update(1)

    if os.path.exists(result_json_file):
        os.remove(result_json_file)
    with open(result_json_file, 'w') as file:
        #json.dump(jdict, file, indent=4)
        json.dump(jdict, file)

def get_img_id(file_name):
    ls = []
    myset = []
    annos = json.load(open(file_name, 'r'))
    for anno in annos:
        ls.append(anno['image_id'])
    myset = {}.fromkeys(ls).keys()
    return myset

def cal_coco_result(annotations_file, result_json_file):
    # https://github.com/muyiguangda/tensorflow-keras-yolov3/blob/master/pycocoEval.py
    if not os.path.exists(result_json_file):
        print("result file not exist,", result_json_file)
        exit(-1)
    if not os.path.exists(annotations_file):
        print("annotations_file file not exist,", annotations_file)
        exit(-1)

    annType = ['segm', 'bbox', 'keypoints']
    annType = annType[1]
    cocoGt = COCO(annotations_file)
    imgIds = get_img_id(result_json_file)
    print (len(imgIds))
    cocoDt = cocoGt.loadRes(result_json_file)
    imgIds = sorted(imgIds)
    imgIds = imgIds[0:5000]
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

def main(argv):
    args = parse_args()

    if (args.pre_result_json) :
        cal_coco_result(args.annotations, args.pre_result_json)
        return

    # Make Detector
    net_input_dims = [int(s) for s in args.net_input_dims.split(',')]
    obj_threshold = float(args.obj_threshold)
    nms_threshold = float(args.nms_threshold)
    print("net_input_dims", net_input_dims)
    print("obj_threshold", obj_threshold)
    print("nms_threshold", nms_threshold)

    net = caffe.Net(args.model_def, args.pretrained_model, caffe.TEST)

    # Load image
    if (args.input_file != '') :
        image = cv2.imread(args.input_file)
        predictions = yolov3_detect(net, image, net_input_dims,
                                    obj_threshold, nms_threshold)
        print(predictions)
        if (args.draw_image != ''):
            image = draw(image, predictions, args.label_file)
            cv2.imwrite(args.draw_image, image)
        return

    # eval
    result_json_file = args.result_json
    eval_detector(net, result_json_file, args.dataset, net_input_dims,
                  obj_threshold, nms_threshold, count=args.count)
    cal_coco_result(args.annotations, result_json_file)

if __name__ == '__main__':
    main(sys.argv)
