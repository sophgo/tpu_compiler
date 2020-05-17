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
import pymlir
import hashlib

def md5(fileName):
    """Compute md5 hash of the specified file"""
    m = hashlib.md5()
    try:
        fd = open(fileName,"rb")
    except IOError:
        print ("Reading file has problem:", filename)
        return
    x = fd.read()
    fd.close()
    m.update(x)
    return m.hexdigest()

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


def parse_args():
    parser = argparse.ArgumentParser(description='Eval YOLO networks via board, default user/password is root@cvitek')
    parser.add_argument("--is_remote_control", type=bool, default=False)
    parser.add_argument('--model', type=str, default='',
                        help="MLIR Model file")
    parser.add_argument('--board_path', type=str, default='/mnt/data/arvin/yolo_hs',
            help="*.cvimodel abs path in board")
    parser.add_argument('--board_ip', type=str, default='',
            help="board ip address")
    parser.add_argument('--model_runner_path', type=str, default='/mnt/data/arvin/old/cvitek_tpu_sdk/bin/model_runner',
            help="abs path of model_runner")
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

def yolov3_detect(module, image, net_input_dims, obj_threshold, nms_threshold, args):
    x = preprocess(image, net_input_dims)
    x = np.expand_dims(x, axis=0)

    if not args.is_remote_control:
      res = module.run(x)
      data = module.get_all_tensor()
    else:
      input_name = "in.npz"
      out_name = "out.npz"
      np.savez(input_name, input=x)

      #print("input_name:", input_name, md5(input_name))
      cmd = "{} --input {} --model {} --output {}".format(
          args.model_runner_path, input_name, args.model, out_name)

      # send input to board
      _cmd = "sshpass -p 'cvitek' scp -r {} root@{}:{}".format(
          input_name, args.board_ip, args.board_path)

      if os.system(_cmd) != 0:
        print('Cmd {} execute failed'.format(_cmd))
        exit(-1)

      # do inference
      cmd = "sshpass -p 'cvitek' ssh -t root@{} 'cd {} && {}'".format(
          args.board_ip, args.board_path, cmd)

      if os.system(cmd) != 0:
        print('Cmd {} execute failed'.format(cmd))
        exit(-1)

      # get result
      _cmd = "sshpass -p 'cvitek' scp -r root@{}:{}/{} . ".format(
          args.board_ip, args.board_path, out_name)

      if os.system(_cmd) != 0:
        print('Cmd {} execute failed'.format(_cmd))
        exit(-1)

      res = np.load(out_name)
      print("out_name md5:", md5(out_name), "files is ", res.files)

    out_feat = {}
    if ('layer82-conv' in res.keys()):
      out_feat['layer82-conv'] = res['layer82-conv']
      out_feat['layer94-conv'] = res['layer94-conv']
      out_feat['layer106-conv'] = res['layer106-conv']
    elif ('layer82-conv_dequant' in res.keys()):
      out_feat['layer82-conv'] = res['layer82-conv_dequant']
      out_feat['layer94-conv'] = res['layer94-conv_dequant']
      out_feat['layer106-conv'] = res['layer106-conv_dequant']
    else:
      assert(False)

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

def eval_detector(module, result_json_file, dataset_path, net_input_dims,
                  obj_threshold, nms_threshold, args, count=-1):
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
            predictions = yolov3_detect(module, image, net_input_dims,
                                        obj_threshold, nms_threshold, args)
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
        json.dump(jdict, file, cls=NpEncoder)

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

    module = None
    if args.is_remote_control:
      print('module in board', args.model)
      #print('ms5 cvi model is', args.model, md5(args.model))
    else:
      # load model
      module = pymlir.module()
      print('load module ', args.model)
      module.load(args.model)
      print('load module done')

    # Load image
    if (args.input_file != '') :
        image = cv2.imread(args.input_file)
        predictions = yolov3_detect(module, image, net_input_dims,
                                    obj_threshold, nms_threshold)
        print(predictions)
        if (args.draw_image != ''):
            image = draw(image, predictions, args.label_file)
            cv2.imwrite(args.draw_image, image)
        return

    # eval
    result_json_file = args.result_json
    eval_detector(module, result_json_file, args.dataset, net_input_dims,
                  obj_threshold, nms_threshold, args, args.count)
    cal_coco_result(args.annotations, result_json_file)

if __name__ == '__main__':
    main(sys.argv)
