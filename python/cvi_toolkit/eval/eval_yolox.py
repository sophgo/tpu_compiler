#!/usr/bin/env python3
import onnxruntime
import numpy as np
import os
import sys
import argparse
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pymlir

COCO_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)

COCO_IDX= [
             1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
            54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
            74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
          ]

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def convert(o):
    if isinstance(o, np.int32): return float(o)
    print(type(o), o)
    raise TypeError

def get_image_id_in_path(image_path):
    stem = Path(image_path).stem
    # in val2014, name would be like COCO_val2014_000000xxxxxx.jpg
    # in val2017, name would be like 000000xxxxxx.jpg
    if (stem.rfind('_') == -1):
        id = int(stem)
    else:
        id = int(stem[stem.rfind('_') + 1:])
    return id

def get_img_id(file_name):
    ls = []
    myset = []
    annos = json.load(open(file_name, 'r'))
    for anno in annos:
        ls.append(anno['image_id'])
    myset = {}.fromkeys(ls).keys()
    return myset

def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes

def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img

def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep

def multiclass_nms_class_aware(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-aware version."""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)

def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
        )
    return dets

def multiclass_nms(boxes, scores, nms_thr, score_thr, class_agnostic=False):
    """Multiclass NMS implemented in Numpy"""
    if class_agnostic:
        nms_method = multiclass_nms_class_agnostic
    else:
        nms_method = multiclass_nms_class_aware
    return nms_method(boxes, scores, nms_thr, score_thr)

def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

def postproc(outputs, img_size, p6=False):

    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
    # batch = 1
    predictions = outputs[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    return scores, boxes_xyxy

def yolo_x_onnx_detector(img_path, input_shape, sess, score_thr, nms_thr=0.45,
                         nms_score_thr=0.1, with_p6=False, draw_image=True):
    json_dict = []
    input_shape = tuple(map(int, input_shape.split(',')))
    origin_img = cv2.imread(img_path)
    img, ratio = preproc(origin_img, input_shape)
    img = np.expand_dims(img, axis=0)
    try:
        ort_inputs = {sess.get_inputs()[0].name: img}
        output = sess.run(None, ort_inputs)[0]
    except AttributeError:
        ret = sess.run(img)
        tensors = sess.get_all_tensor()
        if 'output_Transpose_dequant' in tensors.keys():
            output = tensors['output_Transpose_dequant']
        else:
            output = tensors['output_Transpose']
    scores, boxes_xyxy = postproc(output, input_shape, p6=with_p6)
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=nms_thr, score_thr=nms_score_thr)

    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        final_boxes /= ratio
        if draw_image:
            out_path = "vis_outout"
            mkdir(out_path)
            output_path = os.path.join(out_path, os.path.split(img_path)[-1])

            origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                            conf=score_thr, class_names=COCO_CLASSES)
            cv2.imwrite(output_path, origin_img)

        boxes_xywh = xyxy2xywh(final_boxes)

        # store js info
        try:
            image_id = get_image_id_in_path(img_path)
        except ValueError:
            image_id = None
            print("Warning Make sure you are test custom image.")
        # convert to coco format
        for ind in range(final_boxes.shape[0]):
            json_dict.append({
                "image_id": image_id,
                "category_id": COCO_IDX[final_cls_inds[ind].astype(np.int32)],
                "bbox": list(boxes_xywh[ind]),
                "score": float(final_scores[ind].astype(np.float32))
            })
    return json_dict

def cal_coco_result(annotations_file, result_json_file):
    if not os.path.exists(result_json_file):
        assert ValueError("result file: {} not exist.".format(result_json_file))
    if not os.path.exists(annotations_file):
        assert ValueError("annotations_file file: {} not exist.".format(annotations_file))

    annType = ['segm', 'bbox', 'keypoints']
    annType = annType[1]
    cocoGt = COCO(annotations_file)
    imgIds = get_img_id(result_json_file)
    cocoDt = cocoGt.loadRes(result_json_file)
    imgIds = sorted(imgIds)
    imgIds = imgIds[0:5000]
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

def parse_args():
    parser = argparse.ArgumentParser(description='Eval YOLO networks.')
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="Model definition file")
    parser.add_argument("--net_input_dims", type=str, default="640,640",
                        help="'height,width' dimensions of net input tensors.")
    parser.add_argument("-i", "--input_file", type=str, default=None,
                        help="Input image for testing")
    parser.add_argument("--draw_image", action="store_true",
                        help="Draw results on image")
    parser.add_argument("--dump_blobs",
                        help="Dump all blobs into a file in npz format")
    parser.add_argument("--nms_score_thr", type=float, default=0.01,
                        help="Object confidence threshold")
    parser.add_argument("--nms_threshold", type=float, default=0.65,
                        help="NMS threshold")
    parser.add_argument("--coco_image_path", type=str, default='/work/dataset/coco/val2017',
                        help="dataset image list file")
    parser.add_argument("--coco_annotation", type=str, default='/work/dataset/coco/annotations/instances_val2017.json',
                        help="annotations file")
    parser.add_argument("--coco_result_jason_file", type=str, default="restult_yolox_s.json",
                        help="Result json file")
    parser.add_argument("--pre_result_json", type=str, default=None,
                        help="when present, use pre detected result file, skip detection")
    parser.add_argument("-s", "--score_thr", type=float, default=0.3,
                        help="Score threshould to filter the result.", )
    parser.add_argument("--with_p6", action="store_true",
                        help="Whether your model uses p6 in FPN/PAN.",)
    parser.add_argument("--count", type=int, default=-1)
    args = parser.parse_args()
    return args

def main(argv):
    args = parse_args()

    if args.pre_result_json:
        cal_coco_result(args.coco_annotation, args.pre_result_json)
        return

    result_jist = []
    result_json_file = args.coco_result_jason_file

    if args.model.endswith(".onnx"):
        sess = onnxruntime.InferenceSession(args.model)
    elif args.model.endswith(".mlir"):
        module = pymlir.module()
        module.load(args.model)
        sess = module
        print("model loaded")
    else:
        assert(0)

    if args.input_file:
        print("image stored in vis_outout.")
        yolo_x_onnx_detector(args.input_file, args.net_input_dims, sess, args.score_thr, args.nms_threshold,
                             nms_score_thr=0.1, with_p6=args.with_p6, draw_image=True)
        return

    if not os.path.exists(args.coco_image_path):
        raise ValueError ("Dataset path doesn't exist.")

    image_list = os.listdir(args.coco_image_path)
    count_num = 0
    with tqdm(total= args.count if args.count > 0 else len(image_list)) as pbar:
        for image_name in image_list:
            image_path = os.path.join(args.coco_image_path, image_name)
            jlist = yolo_x_onnx_detector(image_path, args.net_input_dims, sess, args.score_thr, args.nms_threshold,
                         nms_score_thr=args.nms_score_thr, with_p6=args.with_p6, draw_image=args.draw_image)

            result_jist.extend(jlist)
            count_num += 1
            if (count_num == args.count):
                break
            pbar.update(1)

    if os.path.exists(result_json_file):
        os.remove(result_json_file)
    with open(result_json_file, 'w') as file:
        json.dump(result_jist, file, default=convert)

    # eval coco
    cal_coco_result(args.coco_annotation, result_json_file)

if __name__ == '__main__':
    main(sys.argv)