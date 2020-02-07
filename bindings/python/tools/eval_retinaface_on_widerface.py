import argparse
import numpy as np
import os
import pymlir
from model.retinaface.retinaface_util import RetinaFace
from dataset_util.widerface.eval_widerface import detect_on_widerface, evaluation

g_wider_face_path = os.path.join(os.environ['DATASET_PATH'], 'widerface')
g_img_path = os.path.join(g_wider_face_path, 'WIDER_val/images')
g_wider_face_gt_folder = \
    os.path.join(g_wider_face_path, 'wider_face_split')

g_nms_threshold = 0.4
g_obj_threshold = 0.5
g_net_input_dims = '600,600'
g_detector = RetinaFace(g_nms_threshold, g_obj_threshold)

g_mlir_model = pymlir.module()
g_is_int8 = False


def detect(img_bgr):
    retinaface_h, retinaface_w = g_net_input_dims[0], g_net_input_dims[1]

    x = g_detector.preprocess(img_bgr, retinaface_w, retinaface_h)
    y = g_mlir_model.run(x)

    faces, landmarks = g_detector.postprocess(y, retinaface_w, retinaface_h, dequant=g_is_int8)
    ret = np.zeros(faces.shape)

    for i in range(faces.shape[0]):
        ret[i][0] = faces[i][0]
        ret[i][1] = faces[i][1]
        ret[i][2] = faces[i][2] - faces[i][0]
        ret[i][3] = faces[i][3] - faces[i][1]
        ret[i][4] = faces[i][4]

    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='',
                        help="MLIR Model file")
    parser.add_argument("--images", type=str, default=g_img_path,
                        help="dataset image folder")
    parser.add_argument("--annotation", type=str, default=g_wider_face_gt_folder,
                        help="annotation folder")
    parser.add_argument("--result", type=str, default='./result',
                        help="Result folder")
    parser.add_argument("--net_input_dims", default='600,600',
                        help="'height,width' dimensions of net input tensors.")
    parser.add_argument("--obj_threshold", type=float, default=0.005,
                        help="Object confidence threshold")
    parser.add_argument("--nms_threshold", type=float, default=0.45,
                        help="NMS threshold")
    parser.add_argument('--int8', default=False, action="store_true", help="int8 model")
    args = parser.parse_args()

    g_is_int8 = args.int8
    g_net_input_dims = [int(s) for s in args.net_input_dims.split(',')]
    g_nms_threshold = args.nms_threshold
    g_obj_threshold = args.obj_threshold

    g_mlir_model.load(args.model)
    detect_on_widerface(args.images, args.annotation, args.result, detect)
    evaluation(args.result, args.annotation)

