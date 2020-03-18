#!/usr/bin/env python

import numpy as np
import os
import sys
import argparse
import glob
import time
import cv2
import caffe
from utils.yolov3_util import preprocess, postprocess, draw
from utils.alphapose_util import preprocess as pose_preprocess, postprocess as pose_postprocess,draw as draw_pose
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
from pathlib import Path
from tqdm import tqdm
import pymlir

def parse_args():
    parser = argparse.ArgumentParser(description='Eval YOLO networks.')
    parser.add_argument('--pose_model', type=str, default='',
                        help="MLIR Open Pose Model file")
    parser.add_argument('--yolov3_model', type=str, default='',
                        help="MLIR Yolov3 Model file")
    parser.add_argument("--net_input_dims", default='416,416',
                        help="'height,width' dimensions of net input tensors.")
    parser.add_argument("--pose_net_input_dims", default='256,192',
                    help="'height,width' dimensions of pose detect input tensors.")
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

def yolov3_detect(module, image, net_input_dims, obj_threshold, nms_threshold):
    x = preprocess(image, net_input_dims)
    x = np.expand_dims(x, axis=0)
    res = module.run(x)
    data = module.get_all_tensor()

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

def alphapose_detect(yolo_module,pose_module, image,net_input_dims,pose_net_input_dims,
                  obj_threshold, nms_threshold,result_image):


    predictions = yolov3_detect(yolo_module, image, net_input_dims,
                                obj_threshold, nms_threshold)

    human_preds = []

    for i in range(len(predictions)):
        preds = predictions[i]
        if preds[2] == 0:
            human_preds.append(preds)

    for idx in range(len(human_preds)):
        pose_pred = fastpose_detect_one(pose_module, image, human_preds[idx], pose_net_input_dims)
        image = draw_pose(image, pose_pred)

    cv2.imshow('AlphaPose result', image)
    cv2.waitKey(0)
    cv2.imwrite(result_image, image)
    print('##### Pose detection result is saved to  {} #####'.format(result_image))



def keep_human_bbox(yolo_preds):
    human_preds = []
    for i in range(len(yolo_preds)):
        pred = yolo_preds[i]
        if pred[2] == 0:
            human_preds.append(pred)
    return human_preds

def fastpose_detect(module, bgr_img, yolo_preds,pose_net_input_dims):

    y = []
    align_bbox_list = []

    for yolo_pred in yolo_preds:
        bbox = yolo_pred[0]
        x, align_bbox= pose_preprocess(bgr_img, bbox, pose_net_input_dims[0], pose_net_input_dims[1])
        x = np.expand_dims(x, axis=0)
        res = module.run(x)
        data = module.get_all_tensor()

        if ('Convolution56' in res.keys()):
            pose_pred = res['Convolution56']
        else:
            assert(False)

        y.append(pose_pred.copy())
        align_bbox_list.append(align_bbox)

    return pose_postprocess(y, align_bbox_list, yolo_preds)

def fastpose_detect_one(module, bgr_img, yolo_pred,pose_net_input_dims):

    y = []
    align_bbox_list = []

    bbox = yolo_pred[0]
    # print(bbox)
    x, align_bbox= pose_preprocess(bgr_img, bbox, pose_net_input_dims[0], pose_net_input_dims[1])
    # print(align_bbox)
    x = np.expand_dims(x, axis=0)
    res = module.run(x)
    data = module.get_all_tensor()

    if ('output_Conv' in res.keys()):
        pose_pred = res['output_Conv']
    else:
        assert(False)

    y.append(pose_pred.copy())
    align_bbox_list.append(align_bbox)

    yolo_preds = []
    yolo_preds.append(yolo_pred)

    return pose_postprocess(y, align_bbox_list, yolo_preds)

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

def eval_detector(yolo_module, pose_module, result_json_file, dataset_path, net_input_dims,pose_net_input_dims,
                  obj_threshold, nms_threshold, count=-1):
    coco_ids= [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 13, 14, 15, 16, 17,
               18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
               37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
               54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
               74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

    jdict = []
    image_list = os.listdir(dataset_path)
    counter=0

    with tqdm(total=count if count > 0 else len(image_list)) as pbar:
        for image_name in image_list:
            image_path = os.path.join(dataset_path, image_name)

            # print(image_path)
            if (not os.path.exists(image_path)):
                print("image not exit,", image_path)
                exit(-1)
            image_id = get_image_id_in_path(image_path)
            image = cv2.imread(image_path)
            predictions = yolov3_detect(yolo_module, image, net_input_dims,
                                        obj_threshold, nms_threshold)

            human_preds = []
            for i in range(len(predictions)):
                preds = predictions[i]
                if preds[2] == 0:
                    human_preds.append(preds)
            for idx in range(len(human_preds)):
                pose_pred = fastpose_detect_one(pose_module, image, human_preds[idx], pose_net_input_dims)
                for human in pose_pred:
                    kp_preds = human['keypoints']
                    # print(kp_preds)
                    kp_scores = human['kp_score']
                    keypoints = np.concatenate((kp_preds, kp_scores), axis=1)

                    keypoints = keypoints.reshape(-1).tolist()
                    #nothing to do ,  just call to get align_bbox
                    x_, align_bbox= pose_preprocess(image, human_preds[idx][0], pose_net_input_dims[0], pose_net_input_dims[1])

                    clipped_box = clip_box(human_preds[idx][0], image.shape)
                    x, y, w, h = align_bbox

                    # print(np.mean(kp_scores) + np.max(kp_scores))
                    #Not sure if human_detect score should be added.
                    # human_detect_score = human_preds[idx][1]
                    jdict.append({
                        "bbox": [x, y, w, h],
                        "image_id": image_id,
                        "score":float(np.mean(kp_scores) + np.max(kp_scores)),
                        "category_id": 1,
                        "keypoints":keypoints
                    })

            counter += 1
            if (counter == count):
                break
            pbar.update(1)

    if os.path.exists(result_json_file):
        os.remove(result_json_file)
    with open(result_json_file, 'w') as file:
        json.dump(jdict, file)

def get_img_id(file_name):
    ls = []
    myset = []
    annos = json.load(open(file_name, 'r'))
    for anno in annos:
        ls.append(anno['image_id'])
    myset = {}.fromkeys(ls).keys()
    return myset

def evaluate_mAP(annotations, res_file, ann_type='bbox', ann_file='person_keypoints_val2017.json', silence=True):
    """Evaluate mAP result for coco dataset.

    Parameters
    ----------
    res_file: str
        Path to result json file.
    ann_type: str
        annotation type, including: `bbox`, `segm`, `keypoints`.
    ann_file: str
        Path to groundtruth file.
    silence: bool
        True: disable running log.

    """
    class NullWriter(object):
        def write(self, arg):
            pass

    if silence:
        nullwrite = NullWriter()
        oldstdout = sys.stdout
        sys.stdout = nullwrite  # disable output

    cocoGt = COCO(annotations)
    cocoDt = cocoGt.loadRes(res_file)

    cocoEval = COCOeval(cocoGt, cocoDt, ann_type)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    if silence:
        sys.stdout = oldstdout  # enable output

    stats_names = ['AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)',
                   'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']
    info_str = {}
    for ind, name in enumerate(stats_names):
        info_str[name] = cocoEval.stats[ind]

    return info_str

def main(argv):
    args = parse_args()

    if (args.pre_result_json) :
        detbox_AP = evaluate_mAP(args.annotations,args.pre_result_json, ann_type='keypoints')
        print('##### det box: {} mAP #####'.format(detbox_AP['AP']))
        return

    # Make Detector
    net_input_dims = [int(s) for s in args.net_input_dims.split(',')]
    pose_net_input_dims = [int(s) for s in args.pose_net_input_dims.split(',')]
    obj_threshold = float(args.obj_threshold)
    nms_threshold = float(args.nms_threshold)
    print("yolov3 net_input_dims", net_input_dims)
    print("alpha pose net_input_dims", pose_net_input_dims)
    print("obj_threshold", obj_threshold)
    print("nms_threshold", nms_threshold)

    yolo_module = pymlir.module()
    print('load module ', args.yolov3_model)
    yolo_module.load(args.yolov3_model)
    print('load module done')


    pose_module = pymlir.module()
    print('load module ', args.pose_model)
    pose_module.load(args.pose_model)
    print('load module done')


    # Load image
    if (args.input_file != '') :
        image = cv2.imread(args.input_file)
        alphapose_detect(yolo_module,pose_module, image,net_input_dims,pose_net_input_dims,
                  obj_threshold, nms_threshold,args.draw_image)
        return

    # eval
    result_json_file = args.result_json
    eval_detector(yolo_module,pose_module, result_json_file, args.dataset, net_input_dims,pose_net_input_dims,
                  obj_threshold, nms_threshold, count=args.count)

    detbox_AP = evaluate_mAP(args.annotations,result_json_file, ann_type='keypoints')
    # print('##### det box: {} mAP #####'.format(detbox_AP))
    print('AP:     {}   AR:     {}'.format(detbox_AP['AP'],detbox_AP['AR']))
    print('AP .5:  {}   AR .5:   {}'.format(detbox_AP['AP .5'],detbox_AP['AR .5']))
    print('AP .75: {}   AR .75:  {}'.format(detbox_AP['AP .5'],detbox_AP['AR .5']))
    print('AP (M): {}   AR (M): {}'.format(detbox_AP['AP (M)'],detbox_AP['AR (M)']))
    print('AP (L): {}  AR (L): {}'.format(detbox_AP['AR (M)'],detbox_AP['AR (L)']))


if __name__ == '__main__':
    main(sys.argv)
