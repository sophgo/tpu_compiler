import cv2
import math
import numpy as np
from pPose_nms import pose_nms


def get_3rd_point(a, b):
    """Return vector c that perpendicular to (a - b)."""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_dir(src_point, rot_rad):
    """Rotate the point by `rot_rad` degree."""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def _center_scale_to_box(center, scale):
    pixel_std = 1.0
    w = scale[0] * pixel_std
    h = scale[1] * pixel_std
    xmin = center[0] - w * 0.5
    ymin = center[1] - h * 0.5
    xmax = xmin + w
    ymax = ymin + h
    bbox = [xmin, ymin, xmax, ymax]
    return bbox

def _box_to_center_scale(x, y, w, h, aspect_ratio=1.0, scale_mult=1.25):
    """Convert box coordinates to center and scale.
    adapted from https://github.com/Microsoft/human-pose-estimation.pytorch
    """
    pixel_std = 1
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
    if center[0] != -1:
        scale = scale * scale_mult
    return center, scale

def preprocess(bgr_img, bbox, pose_h=256, pose_w=192):
    # x, y, w, h = bbox
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]
    
    _aspect_ratio = float(pose_w) / pose_h  # w / h

    # TODO - test without roi align, crop directly
    center, scale = _box_to_center_scale(
            x, y, w, h,_aspect_ratio)

    scale = scale * 1.0
    trans = get_affine_transform(center, scale, 0, [pose_w, pose_h])
    align_img = cv2.warpAffine(bgr_img, trans, (int(pose_w), int(pose_h)), flags=cv2.INTER_LINEAR)
    align_bbox = _center_scale_to_box(center, scale)

    # TODO - get data from yolo preprocess
    rgb_align_img = cv2.cvtColor(align_img, cv2.COLOR_BGR2RGB)
    align_img = np.transpose(rgb_align_img, (2, 0, 1))  # C*H*W
    align_img = align_img / 255.0
    align_img[0, :, :] += -0.406
    align_img[1, :, :] += -0.457
    align_img[2, :, :] += -0.48
    return align_img, align_bbox

# postprocess function
def get_max_pred(heatmaps):
    num_joints = heatmaps.shape[0]
    width = heatmaps.shape[2]
    heatmaps_reshaped = heatmaps.reshape((num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 1)
    maxvals = np.max(heatmaps_reshaped, 1)

    maxvals = maxvals.reshape((num_joints, 1))
    idx = idx.reshape((num_joints, 1))

    preds = np.tile(idx, (1, 2)).astype(np.float32)

    preds[:, 0] = (preds[:, 0]) % width
    preds[:, 1] = np.floor((preds[:, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    target_coords[0:2] = affine_transform(coords[0:2], trans)
    return target_coords

def heatmap_to_coord_simple(hms, bbox):
    coords, maxvals = get_max_pred(hms)

    hm_h = hms.shape[1]
    hm_w = hms.shape[2]

    # post-processing
    for p in range(coords.shape[0]):
        hm = hms[p]
        px = int(round(float(coords[p][0])))
        py = int(round(float(coords[p][1])))
        if 1 < px < hm_w - 1 and 1 < py < hm_h - 1:
            diff = np.array((hm[py][px + 1] - hm[py][px - 1],
                             hm[py + 1][px] - hm[py - 1][px]))
            coords[p] += np.sign(diff) * .25

    preds = np.zeros_like(coords)

    # transform bbox to scale
    xmin, ymin, xmax, ymax = bbox
    w = xmax - xmin
    h = ymax - ymin
    center = np.array([xmin + w * 0.5, ymin + h * 0.5])
    scale = np.array([w, h])
    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center, scale,
                                   [hm_w, hm_h])

    return preds, maxvals


def postprocess(pose_preds, align_bbox_list, yolo_preds):
    # align_bbox_list: [[x1 y1 x2 y2], [x1 y1 x2 y2], ...]
    # yolo_preds: [[[x y w h], score, cls],[[x y w h], score, cls], [[x y w h], score, cls]]
    eval_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    pose_coords = []
    pose_scores = []
    for pred, bbox in zip(pose_preds, align_bbox_list):
        pred = np.squeeze(pred, axis=0)
        pose_coord, pose_score = heatmap_to_coord_simple(pred[eval_joints], bbox)
        pose_coords.append(np.expand_dims(pose_coord, axis=0))
        pose_scores.append(np.expand_dims(pose_score, axis=0))
    
    if len(pose_scores) == 0:
        return []

    preds_img = np.asarray(pose_coords) # [5, 1, 17, 1]
    preds_img = np.squeeze(preds_img, axis=1)
    preds_scores = np.asarray(pose_scores) # [5, 1, 17, 2]
    preds_scores = np.squeeze(preds_scores, axis=1)
    return pose_nms(yolo_preds, preds_img, preds_scores)

def draw(bgr_img, pred):
    l_pair = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (17, 11), (17, 12),  # Body
        (11, 13), (12, 14), (13, 15), (14, 16)
    ]

    p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
    line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                (77, 222, 255), (255, 156, 127),
                (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]
    img = bgr_img.copy()
    height, width = img.shape[:2]
    img = cv2.resize(img, (int(width / 2), int(height / 2)))
    for human in pred:
        part_line = {}
        kp_preds = human['keypoints']
        kp_scores = human['kp_score']
        kp_preds = np.concatenate((kp_preds, np.expand_dims((kp_preds[5, :] + kp_preds[6, :]) / 2, axis=0)), axis=0)
        kp_scores = np.concatenate((kp_scores, np.expand_dims((kp_scores[5, :] + kp_scores[6, :]) / 2, axis=0)), axis=0)

        # Draw keypoints
        for n in range(kp_scores.shape[0]):
            if kp_scores[n] <= 0.35:
                continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (int(cor_x / 2), int(cor_y / 2))
            bg = img.copy()
            cv2.circle(bg, (int(cor_x / 2), int(cor_y / 2)), 2, p_color[n], -1)
            # Now create a mask of logo and create its inverse mask also
            transparency = max(0, min(1, kp_scores[n]))
            img = cv2.addWeighted(bg, transparency, img, 1 - transparency, 0)

        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                bg = img.copy()

                X = (start_xy[0], end_xy[0])
                Y = (start_xy[1], end_xy[1])
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                stickwidth = (kp_scores[start_p] + kp_scores[end_p]) + 1
                polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(bg, polygon, line_color[i])
                # cv2.line(bg, start_xy, end_xy, line_color[i], (2 * (kp_scores[start_p] + kp_scores[end_p])) + 1)
                transparency = max(0, min(1, 0.5 * (kp_scores[start_p] + kp_scores[end_p])))
                img = cv2.addWeighted(bg, transparency, img, 1 - transparency, 0)
    
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    return img