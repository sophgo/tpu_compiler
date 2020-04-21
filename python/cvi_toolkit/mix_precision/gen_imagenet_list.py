import cv2
import numpy as np
import os

data_path = '/data/dataset/imagenet/ilsvrc12_256'
label_path = \
    os.path.join(data_path, 'imagenet_synset_to_human_label_map.txt')


def imagenet_generator(generate_count=float('inf'), preprocess=None, label_offset=0):
    count = 0
    label_names = np.loadtxt(label_path, str, delimiter='\t')

    for i in range(1, 1001):
        class_path = os.path.join(data_path, str(i))
        imgs_name = os.listdir(class_path)

        for img_name in imgs_name:
                if count >= generate_count:
                    break
                else:
                    count += 1

                img_path = os.path.join(class_path, img_name)
                x = cv2.imread(img_path)

                if preprocess is not None:
                    x = preprocess(x)

                yield x, i + label_offset, label_names[i]


if __name__ == '__main__':
    with open("imagenet_imgs.txt", "w") as out_fp:
        for i in range(1001):
            class_path = os.path.join(data_path, str(i))
            imgs_name = os.listdir(class_path)

            for img_name in imgs_name:
                img_path = os.path.join(class_path, img_name)
                out_fp.write(img_path + '\n')

