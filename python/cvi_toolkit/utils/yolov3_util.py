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


def _process_feats_v2(feature, net_input_dims, anchors, num_of_class, obj_threshold):
    yolo_w = net_input_dims[1]
    yolo_h = net_input_dims[0]
    grid_h = feature.shape[1]
    grid_w = feature.shape[2]
    num_boxes_per_cell = 5

    feature = np.transpose(feature, (1, 2, 0))
    feature = np.reshape(feature, (grid_h, grid_w, num_boxes_per_cell,
                           5 + num_of_class))
    threshold_predictions = []

    anchors_tensor = np.array(anchors).reshape(1, 1, 5, 2)

    box_xy = _sigmoid(feature[..., :2])
    box_wh = np.exp(feature[..., 2:4]) * anchors_tensor

    box_confidence = _sigmoid(feature[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)
    box_class_probs = _softmax(feature[..., 5:])

    col = np.tile(np.arange(0, grid_w), grid_h).reshape(-1, grid_h)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_w)

    col = col.reshape(grid_h, grid_w, 1, 1).repeat(5, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(5, axis=-2)
    grid = np.concatenate((col, row), axis=-1)

    box_xy += grid
    box_xy /= (grid_w, grid_h)
    box_wh /= (grid_w, grid_h)

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


def _process_feats_v3(feature, net_input_dims, anchors, num_of_class, obj_threshold):
    yolo_w = net_input_dims[1]
    yolo_h = net_input_dims[0]
    grid_h = feature.shape[1]
    grid_w = feature.shape[2]
    num_boxes_per_cell = 3

    feature = np.transpose(feature, (1, 2, 0))
    feature = np.reshape(feature, (grid_h, grid_w, num_boxes_per_cell,
                           5 + num_of_class))
    threshold_predictions = []

    anchors_tensor = np.array(anchors).reshape(1, 1, 3, 2)

    box_xy = _sigmoid(feature[..., :2])
    box_wh = np.exp(feature[..., 2:4]) * anchors_tensor

    box_confidence = _sigmoid(feature[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)
    box_class_probs = _softmax(feature[..., 5:])

    col = np.tile(np.arange(0, grid_w), grid_h).reshape(-1, grid_h)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_w)

    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)

    box_xy += grid
    box_xy /= (grid_w, grid_h)
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

def _postprocess_v2(features, image_shape, net_input_dims, obj_threshold, nms_threshold):
    total_predictions = []
    yolov2_num_of_class = 80
    yolov2_anchors = [[0.57273,0.677385], [1.87446,2.06253], [3.33843,5.47434], [7.88282,3.52778], [9.77052,9.16828]]
    for _, feature in enumerate(features):
        threshold_predictions = _process_feats_v2(feature, net_input_dims, yolov2_anchors,
                                               yolov2_num_of_class, obj_threshold)
        total_predictions.extend(threshold_predictions)

    if not total_predictions:
        return total_predictions

    correct_predictions = _correct_boxes(total_predictions, image_shape, net_input_dims)
    correct_predictions.sort(key=lambda tup: tup[1], reverse=True)

    nms_predictions = _non_maximum_suppression(correct_predictions, nms_threshold)
    return nms_predictions

def _postprocess_v3(features, image_shape, net_input_dims, obj_threshold, nms_threshold, tiny=False):
    total_predictions = []
    yolov3_num_of_class = 80
    if not tiny:
        yolov3_anchors = [[116,90, 156,198, 373,326],
                        [30,61, 62,45, 59,119],
                        [10,13, 16,30, 33,23]]
    else:
        yolov3_anchors = [[81,82,  135,169,  344,319],
                        [10,14,  23,27,  37,58]]

    for i, feature in enumerate(features):
        threshold_predictions = _process_feats_v3(feature, net_input_dims, yolov3_anchors[i],
                                               yolov3_num_of_class, obj_threshold)
        total_predictions.extend(threshold_predictions)

    if not total_predictions:
        return total_predictions

    correct_predictions = _correct_boxes(total_predictions, image_shape, net_input_dims)
    correct_predictions.sort(key=lambda tup: tup[1], reverse=True)

    nms_predictions = _non_maximum_suppression(correct_predictions, nms_threshold)
    return nms_predictions

def _postprocess_v4(features, image_shape, net_input_dims, obj_threshold, nms_threshold, tiny=False):
    total_predictions = []
    yolov3_num_of_class = 80
    if not tiny:
        anchors = [
            [12, 16, 19, 36, 40, 28],
            [36, 75, 76, 55, 72, 146],
            [142, 110, 192, 243, 459, 401],
        ]
    else:
        raise ValueError('not support tiny config')

    for i, feature in enumerate(features):
        threshold_predictions = _process_feats_v3(feature, net_input_dims, anchors[i],
                                               yolov3_num_of_class, obj_threshold)
        total_predictions.extend(threshold_predictions)

    if not total_predictions:
        return total_predictions

    correct_predictions = _correct_boxes(total_predictions, image_shape, net_input_dims)
    correct_predictions.sort(key=lambda tup: tup[1], reverse=True)

    nms_predictions = _non_maximum_suppression(correct_predictions, nms_threshold)
    return nms_predictions


def _batched_feature_generator_v2(batched_features, batch=1):
    conv22 = batched_features['conv22']

    for i in range(batch):
        yield [conv22[i]]

def _batched_feature_generator_v3(batched_features, spp_net, batch=1):
    if not spp_net:
        layer82_conv = batched_features['layer82-conv']
        layer94_conv = batched_features['layer94-conv']
        layer106_conv = batched_features['layer106-conv']

        for i in range(batch):
            yield [layer82_conv[i], layer94_conv[i], layer106_conv[i]]
    else:
        layer89_conv = batched_features['layer89-conv']
        layer101_conv = batched_features['layer101-conv']
        layer113_conv = batched_features['layer113-conv']

        for i in range(batch):
            yield [layer89_conv[i], layer101_conv[i], layer113_conv[i]]

def _batched_feature_generator_v3_tiny(batched_features, batch=1):
    layer16_conv = batched_features['layer16-conv']
    layer23_conv = batched_features['layer23-conv']

    for i in range(batch):
        yield [layer16_conv[i], layer23_conv[i]]

def preprocess(bgr_img, net_input_dims, do_preprocess=True):
    yolo_w = net_input_dims[1]
    yolo_h = net_input_dims[0]

    ih = bgr_img.shape[0]
    iw = bgr_img.shape[1]

    scale = min(float(yolo_w) / iw, float(yolo_h) / ih)
    rescale_w = int(iw * scale)
    rescale_h = int(ih * scale)

    # print("yolo_h: {}, yolo_w: {}, rescale_h: {}, rescale_w: {}".format(
    #       yolo_h, yolo_w, rescale_h, rescale_w))

    resized_img = cv2.resize(bgr_img, (rescale_w, rescale_h), interpolation=cv2.INTER_LINEAR)
    new_image = np.full((yolo_h, yolo_w, 3), 0, dtype=np.float32)
    paste_w = (yolo_w - rescale_w) // 2
    paste_h = (yolo_h - rescale_h) // 2

    new_image[paste_h:paste_h + rescale_h, paste_w: paste_w + rescale_w, :] = resized_img
    new_image = np.transpose(new_image, (2, 0, 1))      # row to col, (HWC -> CHW)
    if do_preprocess:
        new_image = new_image / 255.0
        new_image[[0,1,2],:,:] = new_image[[2,1,0],:,:]
    return new_image

def postprocess_v2(batched_features, image_shape, net_input_dims,
                obj_threshold, nms_threshold, batch=1):
    i = 0
    batch_out = {}

    for feature in _batched_feature_generator_v2(batched_features, batch):
        pred = _postprocess_v2(feature, image_shape, net_input_dims,
                            obj_threshold, nms_threshold)

        if not pred:
            batch_out[i] = []
        else:
            batch_out[i] = pred

        i += 1

    return batch_out

def postprocess_v3(batched_features, image_shape, net_input_dims,
                obj_threshold, nms_threshold, spp_net, batch=1):
    i = 0
    batch_out = {}

    for feature in _batched_feature_generator_v3(batched_features, spp_net, batch):
        pred = _postprocess_v3(feature, image_shape, net_input_dims,
                            obj_threshold, nms_threshold)

        if not pred:
            batch_out[i] = []
        else:
            batch_out[i] = pred

        i += 1

    return batch_out

def postprocess_v3_tiny(batched_features, image_shape, net_input_dims,
                obj_threshold, nms_threshold, batch=1):
    i = 0
    batch_out = {}

    for feature in _batched_feature_generator_v3_tiny(batched_features, batch):
        pred = _postprocess_v3(feature, image_shape, net_input_dims,
                            obj_threshold, nms_threshold, True)

        if not pred:
            batch_out[i] = []
        else:
            batch_out[i] = pred

        i += 1

    return batch_out

def postprocess_v4(batched_features, image_shape, net_input_dims,
                obj_threshold, nms_threshold, spp_net, batch=1):
    i = 0
    batch_out = {}

    def _batched_feature_generator_v4(batched_features, batch=1):
        layer139_conv = batched_features['layer139-conv']
        layer150_conv = batched_features['layer150-conv']

        for i in range(batch):
            yield [layer139_conv[i], layer150_conv[i]]


    for feature in _batched_feature_generator_v4(batched_features, batch):
        pred = _postprocess_v4(feature, image_shape, net_input_dims,
                            obj_threshold, nms_threshold, False)

        if not pred:
            batch_out[i] = []
        else:
            batch_out[i] = pred

        i += 1

    return batch_out

def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
    # print(boxes.shape)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep)

def postprocess_v4_tiny(batched_features, image_shape, net_input_dims,
                obj_threshold, nms_threshold, batch=1):
    i = 0
    batch_out = {}

    # [batch, num, 1, 4]
    box_array = batched_features[0]
    # [batch, num, num_classes]
    confs = batched_features[1]

    num_classes = confs.shape[2]

    # [batch, num, 4]
    box_array = box_array[:, :, 0]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    bboxes_batch = []
    for i in range(box_array.shape[0]):

        argwhere = max_conf[i] > obj_threshold
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        bboxes = []
        # nms for each class
        for j in range(num_classes):

            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            keep = nms_cpu(ll_box_array, ll_max_conf, nms_threshold)

            if (keep.size > 0):
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for k in range(ll_box_array.shape[0]):
                    #bboxes.append([ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2], ll_box_array[k, 3], ll_max_conf[k], ll_max_conf[k], ll_max_id[k]])
                    width = image_shape[1]
                    height = image_shape[0]
                    bboxes.append([
                        #[int(ll_box_array[k, 0] * width),
                        #int(ll_box_array[k, 1] * height),
                        #int((ll_box_array[k, 2] ) * width),
                        #int((ll_box_array[k, 3] ) * height)],
                        [int(ll_box_array[k, 0] * width),
                        int(ll_box_array[k, 1] * height),
                        int((ll_box_array[k, 2] - ll_box_array[k, 0]) * width),
                        int((ll_box_array[k, 3] - ll_box_array[k, 1]) * height)],
                        ll_max_conf[k],
                        ll_max_id[k]])

        bboxes_batch.append(bboxes)


    return bboxes_batch

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
