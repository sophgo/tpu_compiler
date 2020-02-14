from widerface_generator import widerface_generator
import numpy as np
import os
import subprocess


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
    wider_eval_tool = os.path.join(os.environ['TPU_PYTHON_PATH'], 'dataset_util', 'widerface', 'wider_eval_tools')
    folder_name = os.path.basename(pred_folder)
    cmd = 'cp -r {} {};'.format(pred_folder, wider_eval_tool)
    cmd += 'pushd {};'.format(wider_eval_tool)
    cmd += 'octave --eval \"wider_eval(\'{}\', \'{}\')\";'.format(folder_name, model_name)
    cmd += 'rm -rf {};'.format(folder_name)
    cmd += 'popd;'
    subprocess.call(cmd, shell=True, executable='/bin/bash')


if __name__ == '__main__':
    # Following is just sample code of evaluation
    g_wider_face_path = os.path.join(os.environ['DATASET_PATH'], 'widerface')
    g_img_path = os.path.join(g_wider_face_path, 'WIDER_val/images')
    g_wider_face_gt_folder = \
        os.path.join(g_wider_face_path, 'wider_face_split')


    def face_detect(image_bgr):
        return np.zeros((1, 5))

    detect_on_widerface(g_img_path, g_wider_face_gt_folder, './result', face_detect)
    evaluation('./result', 'demo_face')
