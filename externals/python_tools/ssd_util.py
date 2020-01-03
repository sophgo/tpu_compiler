import cv2
import numpy as np


def _softmax(x, t=-100.):
    x = x - np.max(x)
    if np.min(x) < t:
        x = x / np.min(x) * t
    exp_x = np.exp(x)
    out = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    return out


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def _process_feats(feature, net_input_dims, anchors, num_of_class, obj_threshold):
    yolo_w = net_input_dims[1]
    yolo_h = net_input_dims[0]
    grid_size = feature.shape[2]
    num_boxes_per_cell = 3

    feature = np.transpose(feature, (1, 2, 0))
    feature = np.reshape(feature, (grid_size, grid_size, num_boxes_per_cell,
                           5 + num_of_class))
    threshold_predictions = []

    anchors_tensor = np.array(anchors).reshape(1, 1, 3, 2)

    box_xy = _sigmoid(feature[..., :2])
    box_wh = np.exp(feature[..., 2:4]) * anchors_tensor

    box_confidence = _sigmoid(feature[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)
    box_class_probs = _softmax(feature[..., 5:])

    col = np.tile(np.arange(0, grid_size), grid_size).reshape(-1, grid_size)
    row = np.tile(np.arange(0, grid_size).reshape(-1, 1), grid_size)

    col = col.reshape(grid_size, grid_size, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_size, grid_size, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)

    box_xy += grid
    box_xy /= (grid_size, grid_size)
    box_wh /= (yolo_w, yolo_h)

    boxes = np.concatenate((box_xy, box_wh), axis=-1)

    box_score = box_confidence * box_class_probs
    box_classes = np.argmax(box_score, axis=-1)
    box_class_score = np.max(box_score, axis=-1)

    pos = np.where(box_class_score >= obj_threshold)

    boxes = boxes[pos]
    scores = box_class_score[pos]
    scores = np.expand_dims(scores, axis=-1)
    classes = box_classes[pos]
    classes = np.expand_dims(classes, axis=-1)
    if boxes is not None:
        threshold_predictions = np.concatenate((boxes, scores, classes), axis=-1)

    return threshold_predictions


def _iou(box1, box2):
    inter_left_x = max(box1[0], box2[0])
    inter_left_y = max(box1[1], box2[1])
    inter_right_x = min(box1[0] + box1[2], box2[0] + box2[2])
    inter_right_y = min(box1[1] + box1[3], box2[1] + box2[3])

    if box1[0] == box2[0] and box1[1] == box2[1] and box1[2] == box2[2] and box1[3] == box2[3]:
        return 1.

    inter_w = max(0, inter_right_x - inter_left_x)
    inter_h = max(0, inter_right_y - inter_left_y)

    inter_area = inter_w * inter_h

    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]

    iou = inter_area / (box1_area + box2_area - inter_area)

    return iou


def _non_maximum_suppression(predictions, nms_threshold):
    nms_predictions = list()
    nms_predictions.append(predictions[0])

    i = 1
    while i < len(predictions):
        nms_len = len(nms_predictions)
        keep = True
        j = 0
        while j < nms_len:
            current_iou = _iou(predictions[i][0], nms_predictions[j][0])
            if nms_threshold < current_iou < 1. and predictions[i][2] == nms_predictions[j][2]:
                keep = False

            j = j + 1
        if keep:
            nms_predictions.append(predictions[i])
        i = i + 1

    return nms_predictions


def _correct_boxes(predictions, image_shape, net_input_dims):
    yolo_w = net_input_dims[1]
    yolo_h = net_input_dims[0]
    image_shape = np.array((image_shape[1], image_shape[0]))
    input_shape = np.array([float(yolo_w), float(yolo_h)])
    new_shape = np.floor(image_shape * np.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    correct = []
    for prediction in predictions:
        x, y, w, h = prediction[0:4]
        box_xy = np.array([x, y])
        box_wh = np.array([w, h])
        score = prediction[4]
        cls = int(prediction[5])

        box_xy = (box_xy - offset) * scale
        box_wh = box_wh * scale

        box_xy = box_xy - box_wh / 2.
        box = np.concatenate((box_xy, box_wh), axis=-1)
        box *= np.concatenate((image_shape, image_shape), axis=-1)
        correct.append([box, score, cls])
    return correct


def _postprocess(features, image_shape, net_input_dims, obj_threshold, nms_threshold):
    total_predictions = []
    yolov3_num_of_class = 80
    yolov3_anchors = [[116, 90, 156, 198, 373, 326],
                      [30, 61, 62, 45, 59, 119],
                      [10, 13, 16, 30, 33, 23]]
    for i, feature in enumerate(features):
        threshold_predictions = _process_feats(feature, net_input_dims, yolov3_anchors[i],
                                               yolov3_num_of_class, obj_threshold)
        total_predictions.extend(threshold_predictions)

    if not total_predictions:
        return total_predictions

    correct_predictions = _correct_boxes(total_predictions, image_shape, net_input_dims)
    correct_predictions.sort(key=lambda tup: tup[1], reverse=True)

    nms_predictions = _non_maximum_suppression(correct_predictions, nms_threshold)
    return nms_predictions


def _batched_feature_generator(batched_features, batch=1):
    layer82_conv = batched_features['layer82-conv']
    layer94_conv = batched_features['layer94-conv']
    layer106_conv = batched_features['layer106-conv']

    for i in range(batch):
        yield [layer82_conv[i], layer94_conv[i], layer106_conv[i]]


def preprocess(net,bgr_img, net_input_dims):
    ssd_w = net_input_dims[1]
    ssd_h = net_input_dims[0]

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))  # row to col, (HWC -> CHW)
    transformer.set_mean('data', np.array([104, 117, 123], dtype=np.float32))
    transformer.set_raw_scale('data', 255)  # [0,1] to [0,255]
    transformer.set_channel_swap('data', (2, 1, 0))  # RGB to BGR

    image = caffe.io.load_image(image_path)  # range from 0 to 1
    newImage = transformer.preprocess('data', image)
    return newImage


def postprocess(batched_features, image_shape, net_input_dims,
                obj_threshold, nms_threshold, batch=1):
    i = 0
    batch_out = {}

    for feature in _batched_feature_generator(batched_features, batch):
        pred = _postprocess(feature, image_shape, net_input_dims,
                            obj_threshold, nms_threshold)

        if not pred:
            batch_out[i] = []
        else:
            batch_out[i] = pred

        i += 1

    return batch_out


def draw(image, predictions, label_file=None, verbose=False):
    image = np.copy(image)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # https://github.com/amikelive/coco-labels
    labelmap = []
    if (label_file):
        with open(label_file) as f:
            labelmap = f.readlines()

    for prediction in predictions:
        x, y, w, h = prediction[0]
        score = prediction[1]
        cls = prediction[2]

        x1 = max(0, np.floor(x + 0.5).astype(int))
        y1 = max(0, np.floor(y + 0.5).astype(int))

        x2 = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        y2 = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (x1,y1), (x2,y2), (255, 0, 0), 2)
        if (labelmap):
            cv2.putText(image, '{0} {1:.2f}'.format(labelmap[cls], score),
                        (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,0,255), 1, cv2.LINE_AA)

        if verbose:
            print('class: {0}, score: {1:.2f}'.format(cls, score))
            print('box coordinate x, y, w, h: {0}'.format(prediction[0]))

    return image
