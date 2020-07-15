#!/usr/bin/env python3

import numpy as np
import os
import sys
import argparse
import glob
import time
import cv2
import caffe
from cvi_toolkit.model import CaffeModel

PRESETS = {
    'coco': { 'classes': [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    ], 'anchors': [[0.738768, 2.42204, 4.30971, 10.246, 12.6868],
                   [0.874946, 2.65704, 7.04493, 4.59428, 11.8741]]
    },
    'voc': { 'classes': [
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
        "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
        "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"],
        'anchors':  [[1.08, 3.42, 6.63, 9.42, 16.62],
                    [1.19, 4.41, 11.38, 5.11, 10.52]]
    }
}

def check_files(args):
    if not os.path.isfile(args.model_def):
        print("cannot find the file %s", args.model_def)
        sys.exit(1)

    if not os.path.isfile(args.pretrained_model):
        print("cannot find the file %s", args.pretrained_model)
        sys.exit(1)

    if not os.path.isfile(args.input_file):
        print("cannot find the file %s", args.input_file)
        sys.exit(1)

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
    parser.add_argument("--draw_image", type=str, default='',
                        help="Draw results on image")
    parser.add_argument("--dump_blobs",
                        help="Dump all blobs into a file in npz format")
    parser.add_argument("--obj_threshold", type=float, default=0.18,
                        help="Object confidence threshold")
    parser.add_argument("--nms_threshold", type=float, default=0.5,
                        help="NMS threshold")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Set batch size")
    parser.add_argument("--dataset", type=str, default='voc')

    args = parser.parse_args()
    check_files(args)
    return args

def get_boxes(output, img_size, grid_size, num_boxes):
    """ extract bounding boxes from the last layer """

    w_img, h_img = img_size[1], img_size[0]
    boxes = np.reshape(output, (grid_size, grid_size, num_boxes, 4))

    offset = np.tile(np.arange(grid_size)[:, np.newaxis],
                     (grid_size, 1, num_boxes))

    boxes[:, :, :, 0] += offset
    boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
    boxes[:, :, :, 0:2] /= 7.0
    # the predicted size is the square root of the box size
    boxes[:, :, :, 2:4] *= boxes[:, :, :, 2:4]

    boxes[:, :, :, [0, 2]] *= w_img
    boxes[:, :, :, [1, 3]] *= h_img

    return boxes


def parse_yolo_output_v1(output, img_size, num_classes):
    """ convert the output of the last fully connected layer (Darknet v1) """

    n_coord_box = 4    # number of coordinates in each bounding box
    grid_size = 7

    sc_offset = grid_size * grid_size * num_classes

    # autodetect num_boxes
    num_boxes = int((output.shape[0] - sc_offset) /
                    (grid_size*grid_size*(n_coord_box+1)))
    box_offset = sc_offset + grid_size * grid_size * num_boxes

    class_probs = np.reshape(output[0:sc_offset], (grid_size, grid_size, num_classes))
    confidences = np.reshape(output[sc_offset:box_offset], (grid_size, grid_size, num_boxes))

    probs = np.zeros((grid_size, grid_size, num_boxes, num_classes))
    for i in range(num_boxes):
        for j in range(num_classes):
            probs[:, :, i, j] = class_probs[:, :, j] * confidences[:, :, i]

    boxes = get_boxes(output[box_offset:], img_size, grid_size, num_boxes)

    return boxes, probs


def get_candidate_objects(output, img_size, args):
    """ convert network output to bounding box predictions """
    threshold = args.obj_threshold
    iou_threshold = args.nms_threshold

    classes = PRESETS[args.dataset]['classes']

    boxes, probs = parse_yolo_output_v1(output, img_size, len(classes))

    filter_mat_probs = (probs >= threshold)
    filter_mat_boxes = np.nonzero(filter_mat_probs)[0:3]
    boxes_filtered = boxes[filter_mat_boxes]
    probs_filtered = probs[filter_mat_probs]
    classes_num_filtered = np.argmax(probs, axis=3)[filter_mat_boxes]

    idx = np.argsort(probs_filtered)[::-1]
    boxes_filtered = boxes_filtered[idx]
    probs_filtered = probs_filtered[idx]
    classes_num_filtered = classes_num_filtered[idx]

    # too many detections - exit
    if len(boxes_filtered) > 1e3:
        print("Too many detections, maybe an error? : {}".format(
            len(boxes_filtered)))
        return []

    probs_filtered = non_maxima_suppression(boxes_filtered, probs_filtered,
                                            classes_num_filtered, iou_threshold)

    filter_iou = (probs_filtered > 0.0)
    boxes_filtered = boxes_filtered[filter_iou]
    probs_filtered = probs_filtered[filter_iou]
    classes_num_filtered = classes_num_filtered[filter_iou]

    result = []
    for class_id, box, prob in zip(classes_num_filtered, boxes_filtered, probs_filtered):
        result.append([classes[class_id], box[0], box[1], box[2], box[3], prob])

    return result

def non_maxima_suppression(boxes, probs, classes_num, thr=0.2):
    """ greedily suppress low-scoring overlapped boxes """
    for i, box in enumerate(boxes):
        if probs[i] == 0:
            continue
        for j in range(i+1, len(boxes)):
            if classes_num[i] == classes_num[j] and iou(box, boxes[j]) > thr:
                probs[j] = 0.0

    return probs


def iou(box1, box2, denom="min"):
    """ compute intersection over union score """
    int_tb = min(box1[0]+0.5*box1[2], box2[0]+0.5*box2[2]) - \
             max(box1[0]-0.5*box1[2], box2[0]-0.5*box2[2])
    int_lr = min(box1[1]+0.5*box1[3], box2[1]+0.5*box2[3]) - \
             max(box1[1]-0.5*box1[3], box2[1]-0.5*box2[3])

    intersection = max(0.0, int_tb) * max(0.0, int_lr)
    area1, area2 = box1[2]*box1[3], box2[2]*box2[3]
    control_area = min(area1, area2) if denom == "min"  \
                   else area1 + area2 - intersection
    return intersection / control_area

def draw_box(img, name, box, score):
    """ draw a single bounding box on the image """
    xmin, ymin, xmax, ymax = box

    box_tag = '{} : {:.2f}'.format(name, score)
    text_x, text_y = 5, 7

    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    boxsize, _ = cv2.getTextSize(box_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (xmin, ymin-boxsize[1]-text_y),
                  (xmin+boxsize[0]+text_x, ymin), (0, 225, 0), -1)
    cv2.putText(img, box_tag, (xmin+text_x, ymin-text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

def show_results(img, results, args):
    """ draw bounding boxes on the image """
    img_width, img_height = img.shape[1], img.shape[0]
    disp_console = True
    imshow = True

    for result in results:
        box_x, box_y, box_w, box_h = [int(v) for v in result[1:5]]
        if disp_console:
            print('    class : {}, [x,y,w,h]=[{:d},{:d},{:d},{:d}], Confidence = {}'.\
                format(result[0], box_x, box_y, box_w, box_h, str(result[5])))
        xmin, xmax = max(box_x-box_w//2, 0), min(box_x+box_w//2, img_width)
        ymin, ymax = max(box_y-box_h//2, 0), min(box_y+box_h//2, img_height)

        draw_box(img, result[0], (xmin, ymin, xmax, ymax), result[5])
    cv2.imwrite(args.draw_image, img)

def main(argv):
    args = parse_args()

    caffe.set_mode_cpu()
    net_input_dims = [int(s) for s in args.net_input_dims.split(',')]
    image = caffe.io.load_image(args.input_file) # load the image using caffe.io
    transformer = caffe.io.Transformer({'data': (1, 3, net_input_dims[0], net_input_dims[1])})
    transformer.set_transpose('data', (2, 0, 1))
    image_x = transformer.preprocess('data', image)

    image_x = np.expand_dims(image_x, axis=0)
    inputs = image_x
    for i in range(1, args.batch_size):
      inputs = np.append(inputs, image_x, axis=0)

    caffemodel = CaffeModel()
    caffemodel.load_model(args.model_def, args.pretrained_model)
    caffemodel.inference(inputs)
    outputs = caffemodel.net.blobs
    all_tensor_dict = caffemodel.get_all_tensor(inputs, False)
    np.savez(args.dump_blobs, **all_tensor_dict)

    if args.draw_image:
        img_cv = cv2.imread(args.input_file)
        results = get_candidate_objects(outputs['result'].data[0], image.shape, args)
        show_results(img_cv, results, args)


if __name__ == '__main__':
    main(sys.argv)
