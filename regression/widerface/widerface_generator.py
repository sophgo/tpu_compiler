import argparse
import cv2
import os
import scipy.io
from tqdm import tqdm

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


if __name__ == '__main__':
    wider_face_path = os.path.join(os.environ['DATASET_PATH'], 'widerface')
    img_path = os.path.join(wider_face_path, 'WIDER_val/images')
    wider_face_gt_folder = \
        os.path.join(wider_face_path, 'wider_face_split')

    for image, bboxes, img_type, img_name in widerface_generator(wider_face_gt_folder, img_path):
        for bbox in bboxes:
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

        cv2.imshow('widerface', image)
        print(img_type, img_name)
        ch = cv2.waitKey(0) & 0xFF
        if ch == 27:
            break
