#!/usr/bin/env python3

import numpy as np
import os
import sys
import argparse
import glob
import time
import cv2
import caffe
from cvi_toolkit.utils.yolov3_util import preprocess, postprocess_v3, postprocess_v3_tiny, postprocess_v2, draw
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
from pathlib import Path
from tqdm import tqdm
import pymlir

def convert(o):
    if isinstance(o, np.int64): return float(o)
    raise TypeError

def parse_args():
    parser = argparse.ArgumentParser(description='Eval YOLO networks.')
    parser.add_argument('--model', type=str, default='',
                        help="MLIR Model file")
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
    parser.add_argument("--model_do_preprocess", type=bool, default=False)
    parser.add_argument("--yolov3", type=str, default='yes')
    parser.add_argument("--spp_net", type=str, default="false")
    parser.add_argument("--tiny", type=str, default="false")

    args = parser.parse_args()
    return args

def yolo_detect(module, image, net_input_dims, obj_threshold, nms_threshold, yolov3, do_preprocess):
    x = preprocess(image, net_input_dims, do_preprocess)
    x = np.expand_dims(x, axis=0)
    res = module.run(x)
    data = module.get_all_tensor()

    out_feat = {}
    if yolov3 == True:
        if tiny:
            if 'layer16-conv' in res.keys():
                out_feat['layer16-conv'] = res['layer16-conv']
                out_feat['layer23-conv'] = res['layer23-conv']
            elif 'layer16-conv_dequant' in res.keys():
                out_feat['layer16-conv'] = res['layer16-conv_dequant']
                out_feat['layer23-conv'] = res['layer23-conv_dequant']
            elif 'layer16-conv_dequant_cast' in res.keys():
                out_feat['layer16-conv'] = res['layer16-conv_dequant_cast']
                out_feat['layer23-conv'] = res['layer23-conv_dequant_cast']
            else:
                assert(0)
        else:
            if not spp_net:
                if ('layer82-conv' in res.keys()):
                    out_feat['layer82-conv'] = res['layer82-conv']
                    out_feat['layer94-conv'] = res['layer94-conv']
                    out_feat['layer106-conv'] = res['layer106-conv']
                elif ('layer82-conv_dequant' in res.keys()):
                    out_feat['layer82-conv'] = res['layer82-conv_dequant']
                    out_feat['layer94-conv'] = res['layer94-conv_dequant']
                    out_feat['layer106-conv'] = res['layer106-conv_dequant']
                elif ('layer82-conv_dequant_cast' in res.keys()):
                    out_feat['layer82-conv'] = res['layer82-conv_dequant_cast']
                    out_feat['layer94-conv'] = res['layer94-conv_dequant_cast']
                    out_feat['layer106-conv'] = res['layer106-conv_dequant_cast']
                else:
                    assert(False)
            else:
                if ('layer89-conv' in res.keys()):
                    out_feat['layer89-conv'] = res['layer89-conv']
                    out_feat['layer101-conv'] = res['layer101-conv']
                    out_feat['layer113-conv'] = res['layer113-conv']
                elif ('layer89-conv_dequant' in res.keys()):
                    out_feat['layer89-conv'] = res['layer89-conv_dequant']
                    out_feat['layer101-conv'] = res['layer101-conv_dequant']
                    out_feat['layer113-conv'] = res['layer113-conv_dequant']
                elif ('layer89-conv_dequant_cast' in res.keys()):
                    out_feat['layer89-conv'] = res['layer89-conv_dequant_cast']
                    out_feat['layer101-conv'] = res['layer101-conv_dequant_cast']
                    out_feat['layer113-conv'] = res['layer113-conv_dequant_cast']
                else:
                    assert(False)
    else:
        if ('conv22' in res.keys()):
            out_feat['conv22'] = res['conv22']
        elif ('conv22_dequant' in res.keys()):
            out_feat['conv22'] = res['conv22_dequant']
        elif ('conv22_dequant_cast' in res.keys()):
            out_feat['conv22'] = res['conv22_dequant_cast']
        else:
            assert(False)

    if yolov3 == True:
        if tiny:
            batched_predictions = postprocess_v3_tiny(out_feat, image.shape, net_input_dims,
                        obj_threshold, nms_threshold, args.batch_size)
        else:
            batched_predictions = postprocess_v3(out_feat, image.shape, net_input_dims,
                                    obj_threshold, nms_threshold, spp_net, batch=1)
    else:
        batched_predictions = postprocess_v2(out_feat, image.shape, net_input_dims,
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

def eval_detector(module, result_json_file, dataset_path, net_input_dims,
                  obj_threshold, nms_threshold, yolov3, do_preprocess, count=-1):
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
            predictions = yolo_detect(module, image, net_input_dims,
                                        obj_threshold, nms_threshold, yolov3, do_preprocess)
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
        json.dump(jdict, file, default=convert)

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
    do_preprocess = not args.model_do_preprocess
    yolov3 = True if args.yolov3 == 'yes' else False
    spp_net = True if args.spp_net == "true" else False
    tiny = True if args.tiny == "true" else False
    print("net_input_dims", net_input_dims)
    print("obj_threshold", obj_threshold)
    print("nms_threshold", nms_threshold)

    module = pymlir.module()
    print('load module ', args.model)
    module.load(args.model)
    print('load module done')

    # Load image
    if (args.input_file != '') :
        image = cv2.imread(args.input_file)
        predictions = yolo_detect(module, image, net_input_dims,
                                    obj_threshold, nms_threshold, yolov3, do_preprocess)
        print(predictions)
        if (args.draw_image != ''):
            image = draw(image, predictions, args.label_file)
            cv2.imwrite(args.draw_image, image)
        return

    # eval
    result_json_file = args.result_json
    eval_detector(module, result_json_file, args.dataset, net_input_dims,
                  obj_threshold, nms_threshold, yolov3, do_preprocess, count=args.count)
    cal_coco_result(args.annotations, result_json_file)

if __name__ == '__main__':
    main(sys.argv)
