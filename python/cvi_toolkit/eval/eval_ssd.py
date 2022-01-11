#!/usr/bin/env python3

import cv2
import caffe
#from caffe.proto import caffe_pb2
# from google.protobuf import text_format
import numpy as np
import os
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pymlir
import argparse
from tqdm import tqdm
from pathlib import Path

ssd_mean = np.array([104, 117, 123], dtype=np.float32)

def parse_args():
    parser = argparse.ArgumentParser(description='Eval YOLO networks.')
    parser.add_argument('--model', type=str, default='',
                        help="MLIR Model file")
    parser.add_argument("--net_input_dims", default='300,300',
                        help="'height,width' dimensions of net input tensors.")
    parser.add_argument("--coco_image_path", type=str, default='',
                        help="Input coco image path for testing")
    parser.add_argument("--coco_annotation", type=str, default='',
                        help="coco lable file in txt format")

    parser.add_argument("--coco_result_jason_file", type=str,
                        help="Result json file")
    parser.add_argument("--count", type=int, default=-1)

    parser.add_argument("--pre_result_json", type=str,
                        help="when present, use pre detected result file, skip detection")
    parser.add_argument('--model_do_preprocess', type=int, default=0)

    args = parser.parse_args()
    return args

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def get_image_id_in_path(image_path):
    stem = Path(image_path).stem
    # in val2014, name would be like COCO_val2014_000000xxxxxx.jpg
    # in val2017, name would be like 000000xxxxxx.jpg
    if (stem.rfind('_') == -1):
        id = int(stem)
    else:
        id = int(stem[stem.rfind('_') + 1:])
    return id

def write_result_to_json(image_name, predictions, fp, line):
    image_id = get_image_id_in_path(image_name)
    coco_ids = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
        43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
        62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
        85, 86, 87, 88, 89, 90
    ]

    boxes, classes, scores = predictions
    cnt = 0
    for box, cls, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = box

        bx = x1
        by = y1
        bw = x2 - x1
        bh = y2 - y1

        res = {
            "image_id": image_id,
            "category_id": coco_ids[cls],
            "bbox": [bx, by, bw, bh],
            "score": float(score)
        }
        common = "" if line == 0 and cnt == 0 else ","
        fp.write("{}{}".format(common, json.dumps(res, cls=NpEncoder)))
        fp.write('\n')
        cnt += 1

def get_img_id(file_name):
    ls = []
    myset = []
    annos = json.load(open(file_name, 'r'))
    for anno in annos:
        ls.append(anno['image_id'])
    myset = {}.fromkeys(ls).keys()
    return myset

def cal_coco_result_from_json(annotations_file, result_json_file):
    # https://github.com/muyiguangda/tensorflow-keras-yolov3/blob/master/pycocoEval.py
    if not os.path.exists(result_json_file):
        print("result file not exist,", result_json_file)
        exit(-1)
    if not os.path.exists(annotations_file):
        print("annotations_file file not exist,", annotations_file)
        exit(-1)
    print("load json name:",result_json_file)

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


def parse_top_detection(resolution, detections, conf_threshold=0.6):
    det_label = detections[0, 0, :, 1]
    det_conf = detections[0, 0, :, 2]
    det_xmin = detections[0, 0, :, 3]
    det_ymin = detections[0, 0, :, 4]
    det_xmax = detections[0, 0, :, 5]
    det_ymax = detections[0, 0, :, 6]

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


def dump_coco_json():
    module = pymlir.module()
    module.load(args.model)

    net_input_dims = [int(s) for s in args.net_input_dims.split(',')]
    do_preprocess = not args.model_do_preprocess
    # transformer = caffe.io.Transformer({'data': (1,3,net_input_dims[0],net_input_dims[1])})
    # transformer.set_transpose('data', (2, 0, 1))  # row to col, (HWC -> CHW)
    # transformer.set_mean('data', ssd_mean)
    # transformer.set_raw_scale('data', 255)  # [0,1] to [0,255]
    # transformer.set_channel_swap('data', (2, 1, 0))  # RGB to BGR
    if os.path.exists(args.coco_result_jason_file):
        os.remove(args.coco_result_jason_file)
    with open(args.coco_result_jason_file, 'w') as coco_result:
        coco_result.write('[\n')
        coco_result.flush()

        imgs_name = os.listdir(args.coco_image_path)
        i=0
        with tqdm(total= args.count if args.count > 0 else len(imgs_name)) as pbar:
            for img_name in imgs_name:
                image_path = os.path.join(args.coco_image_path, img_name)
                image_x = cv2.imread(image_path)
                image_ori = image_x
                image_x = cv2.resize(image_x, (net_input_dims[1], net_input_dims[0]))
                image_x = image_x.astype(np.float32)
                image_x = np.transpose(image_x, [2,0,1])
                if do_preprocess:
                    mean = ssd_mean[:, np.newaxis, np.newaxis]
                    image_x = image_x - mean
                image = image_x
                image = np.expand_dims(image, axis=0)
                _ = module.run(image)
                data = module.get_all_tensor()
                detections = data['detection_out']
                classes, scores, boxes = parse_top_detection(image_ori.shape, detections, 0.01)
                if len(boxes) > 0:
                    write_result_to_json(img_name, [boxes, classes, scores], coco_result, i)

                coco_result.flush()
                i += 1
                if (i == args.count):
                    break
                pbar.update(1)

        #coco_result.seek(-2, 1)
        coco_result.write('\n]')


if __name__ == "__main__":
    args = parse_args()
    if (args.pre_result_json) :
        cal_coco_result_from_json(args.coco_annotation, args.pre_result_json)
    else:
        dump_coco_json()
        cal_coco_result(args.coco_annotation, args.coco_result_jason_file)
