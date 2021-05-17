import sys
import caffe
import cv2
import os
from retinaface_util import RetinaFace

g_fp32_model_path = os.path.join(os.environ['MODEL_PATH'], 'face_detection/retinaface/caffe')
g_proto = os.path.join(g_fp32_model_path, 'R50-0000.prototxt')
g_weight_fp32 = os.path.join(g_fp32_model_path, 'R50-0000.caffemodel')
g_obj_threshold = 0.5
g_nms_threshold = 0.4
retinaface_w, retinaface_h = 150, 150


def inference_from_webcam():
    cap = cv2.VideoCapture(0)
    detector = RetinaFace()
    while True:
        ret, img_bgr = cap.read()
        x = detector.preprocess(img_bgr, retinaface_w, retinaface_h)

        net = caffe.Net(g_proto, g_weight_fp32, caffe.TEST)
        net.blobs['data'].reshape(1, 3, x.shape[2], x.shape[3])
        net.blobs['data'].data[...] = x
        y = net.forward()
        faces, landmarks = detector.postprocess(y, retinaface_w, retinaface_h)
        draw_image = detector.draw(img_bgr, faces, landmarks, True)

        cv2.imshow('face', draw_image)
        ch = cv2.waitKey(1) & 0xFF
        if ch == 27:
            break


if __name__ == '__main__':
    inference_from_webcam()
    

