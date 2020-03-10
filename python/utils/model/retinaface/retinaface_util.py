import cv2
import numpy as np
from rcnn.processing.bbox_transform import clip_boxes
from rcnn.processing.generate_anchor import generate_anchors_fpn, anchors_plane
from rcnn.processing.nms import gpu_nms_wrapper, cpu_nms_wrapper
from rcnn.processing.bbox_transform import bbox_overlaps

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
        self.nms = cpu_nms_wrapper(self.nms_threshold)


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

    def preprocess(self, img_bgr, retinaface_w, retinaface_h):
        # FIXME - Do not resize directly
        # Imitate yolo, fill the zero to the correct aspect ratio, and then resize

        self.im_scale_w = float(retinaface_w) / float(img_bgr.shape[1])
        self.im_scale_h = float(retinaface_h) / float(img_bgr.shape[0])
        img = cv2.resize(img_bgr, None, None, fx=self.im_scale_w, fy=self.im_scale_h, interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))  # row to col, (HWC -> CHW)
        img = img.reshape((1, 3, img.shape[1], img.shape[2]))
        img = img.astype(np.float32)
        return img

    def postprocess(self, y, retinaface_x, retinaface_y, py_softmax=False, dequant=False):
        proposals_list = []
        scores_list = []
        landmarks_list = []
        # print(y.keys())

        for idx, s in enumerate(self._feat_stride_fpn):
            stride = int(s)
            if dequant is True:
                if py_softmax is True:
                    cls_tensor_name = 'face_rpn_cls_score_reshape_stride{}_dequant'.format(s)
                else:
                    cls_tensor_name = 'face_rpn_cls_prob_reshape_stride{}'.format(s)

                bbox_tensor_name = 'face_rpn_bbox_pred_stride{}_dequant'.format(s)
                pts_tensor_name = 'face_rpn_landmark_pred_stride{}_dequant'.format(s)
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

        keep = self.nms(pre_det)
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
