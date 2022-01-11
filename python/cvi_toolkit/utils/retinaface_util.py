import os
import cv2
import scipy.io
import subprocess
import numpy as np
from tqdm import tqdm


# rewrite cython in anchors.pyx
def anchors_cython(height, width, stride, base_anchors):
    nofanchors = base_anchors.shape[0]
    all_anchors = np.zeros((height, width, nofanchors, 4), dtype=np.float32)
    h, w = np.meshgrid(np.arange(height), np.arange(width))
    sh, sw = h * stride, w * stride
    all_anchors = np.expand_dims(np.stack([sw, sh, sw, sh], axis=-1), axis=-2).repeat(nofanchors, axis=-2)
    return all_anchors + base_anchors

# cp from cpu_nms_wrapper
def nms(dets, thresh):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

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

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

# cp from rcnn.processing.bbox_transform
def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    :param boxes: [N, 4* num_classes]
    :param im_shape: tuple of 2
    :return: [N, 4* num_classes]
    """
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes

# cp from generate_anchor.py
def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2 ** np.arange(3, 6), stride=16, dense_anchor=False):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.shape[0])])
    if dense_anchor:
      assert stride%2==0
      anchors2 = anchors.copy()
      anchors2[:,:] += int(stride/2)
      anchors = np.vstack( (anchors, anchors2) )
    #print('GA',base_anchor.shape, ratio_anchors.shape, anchors.shape)
    return anchors

def anchors_plane(feat_h, feat_w, stride, base_anchor):
    return anchors_cython(feat_h, feat_w, stride, base_anchor)

def generate_anchors_fpn(dense_anchor=False, cfg = None):
    #assert(False)
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """
    # if cfg is None:
    #   from ..config import config
    #   cfg = config.RPN_ANCHOR_CFG
    RPN_FEAT_STRIDE = []
    for k in cfg:
      RPN_FEAT_STRIDE.append( int(k) )
    RPN_FEAT_STRIDE = sorted(RPN_FEAT_STRIDE, reverse=True)
    anchors = []
    for k in RPN_FEAT_STRIDE:
      v = cfg[str(k)]
      bs = v['BASE_SIZE']
      __ratios = np.array(v['RATIOS'])
      __scales = np.array(v['SCALES'])
      stride = int(k)
      #print('anchors_fpn', bs, __ratios, __scales, file=sys.stderr)
      r = generate_anchors(bs, __ratios, __scales, stride, dense_anchor)
      #print('anchors_fpn', r.shape, file=sys.stderr)
      anchors.append(r)

    return anchors

# copy form widerface_generator.py
def widerface_generator(wider_face_gt_folder, wider_face_image, preprocess=None):
    wider_face_label = os.path.join(wider_face_gt_folder, 'wider_face_val.mat')
    f = scipy.io.loadmat(wider_face_label)
    event_list = f.get('event_list')
    file_list = f.get('file_list')
    face_bbx_list = f.get('face_bbx_list')

    total_image_num = 0
    # print("event_list len {}".format(len(event_list)))
    for event_idx, event in enumerate(event_list):
        total_image_num = total_image_num + len(file_list[event_idx][0])
        # print("filelist {} len {}".format(event_idx, len(file_list[event_idx][0])))

    with tqdm(total = total_image_num) as pbar:
        for event_idx, event in enumerate(event_list):
            directory = event[0][0]
            for im_idx, im in enumerate(file_list[event_idx][0]):
                im_name = im[0][0]
                face_bbx = face_bbx_list[event_idx][0][im_idx][0]
                bboxes = []

                for i in range(face_bbx.shape[0]):
                    xmin = int(face_bbx[i][0])
                    ymin = int(face_bbx[i][1])
                    xmax = int(face_bbx[i][2]) + xmin
                    ymax = int(face_bbx[i][3]) + ymin
                    bboxes.append((xmin, ymin, xmax, ymax))

                image_path = os.path.join(wider_face_image, directory, im_name + '.jpg')
                img = cv2.imread(image_path)
                if preprocess is not None:
                    img = preprocess(img)
                pbar.update(1)
                yield img, bboxes, directory, im_name

# copy form eval_widerface.py
def detect_on_widerface(img_path, wider_face_gt_folder, result_folder_path, detect_func):
    if not os.path.isdir(result_folder_path):
        os.mkdir(result_folder_path)

    for image, _, img_type, img_name in widerface_generator(wider_face_gt_folder, img_path):
        detect_result_sub_folder = os.path.join(result_folder_path, img_type)

        if not os.path.isdir(detect_result_sub_folder):
            os.mkdir(detect_result_sub_folder)

        detect_result = os.path.join(detect_result_sub_folder, img_name + '.txt')
        with open(detect_result, 'w') as fp:
            pred = detect_func(image)
            fp.write(img_name + '\n')
            fp.write(str(pred.shape[0]) + '\n')

            for i in range(pred.shape[0]):
                face = pred[i]
                # x, y, w, h, confidence
                fp.write(str(face[0]) + " " + str(face[1]) + " " +
                         str(face[2]) + " " + str(face[3]) + " " + str(face[4]) + '\n')


def evaluation(pred_folder, model_name):
    acc_log = "acc.log"
    wider_eval_tool = os.path.join(os.environ['TPU_PYTHON_PATH'], 'cvi_toolkit', 'eval', 'wider_eval_tools')

    acc_log_path = os.path.join(wider_eval_tool, acc_log)
    folder_name = os.path.basename(pred_folder)

    cmd = 'cp -r {} {};'.format(pred_folder, wider_eval_tool)
    cmd += 'pushd {};'.format(wider_eval_tool)
    cmd += 'octave --no-window-system --eval \"wider_eval(\'{}\', \'{}\')\" 2>&1 | tee {} ;'.format(folder_name, model_name, acc_log)
    cmd += 'rm -rf {};'.format(folder_name)
    cmd += 'popd;'
    cmd += 'mv {} . ;'.format(acc_log_path)
    print("exec:{}".format(cmd))
    subprocess.call(cmd, shell=True, executable='/bin/bash')


class RetinaFace:
    def __init__(self, nms_threshold=0.4, obj_threshold=0.5):
        self.nms_threshold = nms_threshold
        self.obj_threshold = obj_threshold
        self._feat_stride_fpn = [32, 16, 8]
        _ratio = (1.,)
        self.anchor_cfg = {
            '32': {'SCALES': (32,16), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
            '16': {'SCALES': (8,4), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
            '8': {'SCALES': (2,1), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
        }

        self.fpn_keys = []
        for s in self._feat_stride_fpn:
            self.fpn_keys.append('stride%s'%s)

        self._anchors_fpn = dict(zip(self.fpn_keys, \
            generate_anchors_fpn(dense_anchor=False, cfg=self.anchor_cfg)))
        for k in self._anchors_fpn:
            v = self._anchors_fpn[k].astype(np.float32)
            self._anchors_fpn[k] = v

        self._num_anchors = dict(zip(self.fpn_keys, [anchors.shape[0] \
            for anchors in self._anchors_fpn.values()]))

        self.im_scale_w = 0
        self.im_scale_h = 0
        # self.nms = nms(self.nms_threshold)


    @staticmethod
    def bbox_pred(boxes, box_deltas):
        """
        Transform the set of class-agnostic boxes into class-specific boxes
        by applying the predicted offsets (box_deltas)
        :param boxes: !important [N 4]
        :param box_deltas: [N, 4 * num_classes]
        :return: [N 4 * num_classes]
        """
        if boxes.shape[0] == 0:
            return np.zeros((0, box_deltas.shape[1]))

        boxes = boxes.astype(np.float, copy=False)
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
        ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

        dx = box_deltas[:, 0:1]
        dy = box_deltas[:, 1:2]
        dw = box_deltas[:, 2:3]
        dh = box_deltas[:, 3:4]

        pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
        pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
        pred_w = np.exp(dw) * widths[:, np.newaxis]
        pred_h = np.exp(dh) * heights[:, np.newaxis]

        pred_boxes = np.zeros(box_deltas.shape)
        # x1
        pred_boxes[:, 0:1] = pred_ctr_x - 0.5 * (pred_w - 1.0)
        # y1
        pred_boxes[:, 1:2] = pred_ctr_y - 0.5 * (pred_h - 1.0)
        # x2
        pred_boxes[:, 2:3] = pred_ctr_x + 0.5 * (pred_w - 1.0)
        # y2
        pred_boxes[:, 3:4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

        if box_deltas.shape[1]>4:
            pred_boxes[:,4:] = box_deltas[:,4:]

        return pred_boxes

    @staticmethod
    def landmark_pred(boxes, landmark_deltas):
        if boxes.shape[0] == 0:
            return np.zeros((0, landmark_deltas.shape[1]))
        boxes = boxes.astype(np.float, copy=False)
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
        ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)
        pred = landmark_deltas.copy()
        for i in range(5):
            pred[:,i,0] = landmark_deltas[:,i,0]*widths + ctr_x
            pred[:,i,1] = landmark_deltas[:,i,1]*heights + ctr_y
        return pred

    @staticmethod
    def _softmax(x, t=-100.):
        x = x - np.max(x)
        if np.min(x) < t:
            x = x / np.min(x) * t
        exp_x = np.exp(x)
        out = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return out

    @staticmethod
    def _softmax_4d(x, axis=-1):
        x = x - np.max(x, axis=axis)
        exp_x = np.exp(x)
        out = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        return out

    def preprocess(self, img_bgr, retinaface_w, retinaface_h, do_preprocess=True):
        # FIXME - Do not resize directly
        # Imitate yolo, fill the zero to the correct aspect ratio, and then resize

        self.im_scale_w = float(retinaface_w) / float(img_bgr.shape[1])
        self.im_scale_h = float(retinaface_h) / float(img_bgr.shape[0])
        img = cv2.resize(img_bgr, None, None, fx=self.im_scale_w, fy=self.im_scale_h, interpolation=cv2.INTER_LINEAR)
        if do_preprocess:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))  # row to col, (HWC -> CHW)
        img = img.reshape((1, 3, img.shape[1], img.shape[2]))
        img = img.astype(np.float32)
        return img

    def postprocess(self, y, retinaface_x, retinaface_y, py_softmax=False, dequant=False, do_preprocess=True):
        proposals_list = []
        scores_list = []
        landmarks_list = []
        # print(y.keys())

        for idx, s in enumerate(self._feat_stride_fpn):
            stride = int(s)
            if dequant:
                if py_softmax is True:
                    cls_tensor_name = 'face_rpn_cls_score_reshape_stride{}_dequant'.format(s)
                else:
                    cls_tensor_name = 'face_rpn_cls_prob_reshape_stride{}_dequant'.format(s)

                bbox_tensor_name = 'face_rpn_bbox_pred_stride{}_dequant'.format(s)
                pts_tensor_name = 'face_rpn_landmark_pred_stride{}_dequant'.format(s)

                if do_preprocess == False:
                    bbox_tensor_name = bbox_tensor_name + '_cast'
                    pts_tensor_name = pts_tensor_name + '_cast'
            else:
                if py_softmax is True:
                    cls_tensor_name = 'face_rpn_cls_score_reshape_stride{}'.format(s)
                else:
                    cls_tensor_name = 'face_rpn_cls_prob_reshape_stride{}'.format(s)

                bbox_tensor_name = 'face_rpn_bbox_pred_stride{}'.format(s)
                pts_tensor_name = 'face_rpn_landmark_pred_stride{}'.format(s)

            score_tensor = y[cls_tensor_name]
            bbox_tensor = y[bbox_tensor_name]
            pts_tensor = y[pts_tensor_name]

            if py_softmax is True:
                # Apply face_rpn_cls_prob_stride{}
                score_tensor = self._softmax_4d(score_tensor, axis=1)
                # Apply face_rpn_cls_prob_reshape_stride{}
                score_tensor = score_tensor.reshape((1, 4, -1, score_tensor.shape[3]))

            # score of non-face vs face, take face only
            scores = score_tensor[:, self._num_anchors['stride%s'%s]:, :, :]

            height, width = bbox_tensor.shape[2], bbox_tensor.shape[3]
            A = self._num_anchors['stride%s'%s]
            K = height * width
            anchors_fpn = self._anchors_fpn['stride%s'%s]
            anchors = anchors_plane(height, width, stride, anchors_fpn)
            anchors = anchors.reshape((K * A, 4))

            scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))
            bbox_deltas = bbox_tensor.transpose((0, 2, 3, 1))
            bbox_pred_len = bbox_deltas.shape[3]//A
            bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
            bbox_deltas[:, 0::4] = bbox_deltas[:,0::4]
            bbox_deltas[:, 1::4] = bbox_deltas[:,1::4]
            bbox_deltas[:, 2::4] = bbox_deltas[:,2::4]
            bbox_deltas[:, 3::4] = bbox_deltas[:,3::4]
            proposals = self.bbox_pred(anchors, bbox_deltas)
            proposals = clip_boxes(proposals, [retinaface_x, retinaface_y])

            scores_ravel = scores.ravel()
            order = np.where(scores_ravel>=self.obj_threshold)[0]
            proposals = proposals[order, :]
            scores = scores[order]

            proposals[:, 0] /= self.im_scale_w
            proposals[:, 1] /= self.im_scale_h
            proposals[:, 2] /= self.im_scale_w
            proposals[:, 3] /= self.im_scale_h
            proposals_list.append(proposals)
            scores_list.append(scores)

            landmark_pred_len = pts_tensor.shape[1]//A
            landmark_deltas = pts_tensor.transpose((0, 2, 3, 1)).reshape((-1, 5, landmark_pred_len//5))
            landmarks = self.landmark_pred(anchors, landmark_deltas)
            landmarks = landmarks[order, :]

            landmarks[:,:,0] /= self.im_scale_w
            landmarks[:,:,1] /= self.im_scale_h
            landmarks_list.append(landmarks)

        proposals = np.vstack(proposals_list)
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        proposals = proposals[order, :]
        scores = scores[order]
        landmarks = np.vstack(landmarks_list)
        landmarks = landmarks[order].astype(np.float32, copy=False)
        pre_det = np.hstack((proposals[:,0:4], scores)).astype(np.float32, copy=False)

        keep = nms(pre_det, self.nms_threshold)
        det = np.hstack( (pre_det, proposals[:,4:]) )
        det = det[keep, :]
        landmarks = landmarks[keep]

        return det, landmarks


    def draw(self, image, faces, landmarks, verbose=False):
        image = np.copy(image)

        for i in range(faces.shape[0]):
            box = faces[i]
            box_int = faces[i].astype(int)
            landmark5 = landmarks[i]
            landmark5_int = landmark5.astype(int)

            cv2.rectangle(image, (box_int[0], box_int[1]), (box_int[2], box_int[3]), (255, 0, 0), 2)
            for l in range(landmark5_int.shape[0]):
                cv2.circle(image, (landmark5_int[l][0], landmark5_int[l][1]), 1, (0, 0, 255), 2)

            if verbose:
                print('box coordinate: {0}'.format(box))
                print('landmark coordinate: {0}'.format(landmark5))

        return image
