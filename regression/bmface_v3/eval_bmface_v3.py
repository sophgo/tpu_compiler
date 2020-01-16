#!/usr/bin/env python2

import argparse
import os
import random
import numpy as np
import cv2
import copy

import pymlir

MODEL_MLIR_PATH  = './bmface-v3_opt.mlir'
LFW_DATASET_PATH = '/workspace/data/dataset_zoo/lfw/bmface_preprocess/bmface_LFW/'
PAIRS_FILE_PATH  = '/workspace/data/dataset_zoo/lfw/pairs.txt'

parser = argparse.ArgumentParser(description="BMFace Evaluation on LFW Dataset.")
parser.add_argument("--model", type=str, help="The path of the mlir model.", 
                    default=MODEL_MLIR_PATH)
parser.add_argument("--dataset", type=str, help="The root directory of the LFW dataset.",
                    default=LFW_DATASET_PATH)
parser.add_argument("--pairs", type=str, help="The path of the pairs file.", 
                    default=PAIRS_FILE_PATH)
parser.add_argument("--show", type=bool, default=False)
args = parser.parse_args()


def preprocess_func(img):
# _scale and _bias is reference to bmface-v3_xxx_scale_w_xxx.caffemodel
    _scale = 0.0078125
    _bias  = np.array([-0.99609375, -0.99609375, -0.99609375], dtype=np.float32)
    img *= _scale
    img += _bias
    return np.transpose(img, (2, 0, 1))


def eval_difference(v1, v2):
    # Cosine Similarity
    v1 = v1.flatten()
    v2 = v2.flatten()
    inner_product = np.inner(v1, v2)
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    cos_theta = inner_product / (v1_norm * v2_norm)

    return 1.0 - cos_theta


def diff2score(diff):
    return 1.0 - 0.5*diff


def parse_pairs(pairs_line, counter, period=300):
    pairs_line = pairs_line.split()
    img1_name = '{name}_{num:04d}.jpg'.format(name=pairs_line[0], num=int(pairs_line[1]))
    if ((counter-1) / period) % 2 == 0:
        img2_name = '{name}_{num:04d}.jpg'.format(name=pairs_line[0], num=int(pairs_line[2]))
        label = 1
    else:
        img2_name = '{name}_{num:04d}.jpg'.format(name=pairs_line[2], num=int(pairs_line[3]))
        label = 0
    return img1_name, img2_name, label


def eval_AUC(y, pred):
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    pred_sort = np.sort(pred)[::-1]
    index = np.argsort(pred)[::-1]
    y_sort = y[index]

    tpr, fpr, thr = list(), list(), list()
    for i, item in enumerate(pred_sort):
        tpr.append(float(np.sum((y_sort[:i] == 1))) / pos)
        fpr.append(float(np.sum((y_sort[:i] == 0))) / neg)
        thr.append(item)

    tpr, fpr, thr = np.array(tpr), np.array(fpr), np.array(thr)
    auc = np.trapz(tpr, fpr)

    return auc


if __name__ == '__main__':
    # Load model
    module = pymlir.module()
    print('load module ', args.model)
    module.load(args.model)
    print('load module done')
    eval_score = list()
    
    # Prepare preprocess images set 
    preprocess_imgs = []
    for p in os.listdir(args.dataset):
        for img_name in os.listdir(os.path.join(args.dataset, p)):
            preprocess_imgs.append(img_name)
    preprocess_imgs = set(preprocess_imgs)
    # Prepare preprocess images set (END)

    f_pairs = open(args.pairs, 'r')
    f_pairs.readline()

    _line = f_pairs.readline()
    counter = 1
    while _line:
        img1_name, img2_name, label = parse_pairs(_line, counter)

        if img1_name not in preprocess_imgs or img2_name not in preprocess_imgs:
            if args.show:
                print('[{i:>4}] Skip! ( {n1}, {n2} )'.format(
                       i=counter, n1=img1_name, n2=img2_name))
            _line = f_pairs.readline()
            counter += 1
            continue

        person1 = img1_name[:-9]
        person2 = img2_name[:-9]
        img1 = cv2.imread(os.path.join(args.dataset, person1, img1_name))
        img2 = cv2.imread(os.path.join(args.dataset, person2, img2_name))
        img1 = preprocess_func(img1.astype(np.float32))
        img2 = preprocess_func(img2.astype(np.float32))

        out = module.run(img1)
        face_feature_1 = copy.copy(out)
        out = module.run(img2)
        face_feature_2 = copy.copy(out)

        feature_diff = eval_difference(face_feature_1, face_feature_2)
        _score = diff2score(feature_diff)
        eval_score.append([label, _score])

        if args.show:
            print('[{i:>4}] Diff: {d:.4f}'.format(i=counter, d=feature_diff))

        _line = f_pairs.readline()
        counter += 1

    f_pairs.close()
    eval_score = np.array(eval_score)
    #print(eval_score)
    #np.save('bmface-v3_mlir_eval_score.npy', eval_score)
    
    label = eval_score[:, 0]
    score = eval_score[:, 1]

    auc = eval_AUC(label, score)

    print('-- Evaluation Result --')
    print('Model: {_m}'.format(_m=args.model))
    print('AUC  : {_auc}'.format(_auc=auc))
