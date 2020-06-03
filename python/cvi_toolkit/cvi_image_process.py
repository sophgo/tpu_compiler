#!/usr/bin/env python3
# encoding: utf-8

import argparse
import sys, os, cv2
import numpy as np

def center_crop(img,crop_dim):
    center = np.array(img.shape[1:]) / 2.0
    crop_dim = np.array(crop_dim)
    crop = np.tile(center, (1, 2))[0] + np.concatenate([
        -(crop_dim / 2.0),
        crop_dim / 2.0
    ])
    crop = crop.astype(int)
    img = img[:, crop[0]:crop[2], crop[1]:crop[3]]
    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str)
    parser.add_argument('--resize_dims', type=str, default="256,256")
    parser.add_argument('--net_input_dims', type=str, default="224,224")
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--save', type=str)
    args = parser.parse_args()
    resize_dims = [int(dim) for dim in args.resize_dims.strip().split(',')]
    net_input_dims = [float(dim) for dim in args.net_input_dims.strip().split(',')]
    image = cv2.imread(args.image) #BGR
    x = cv2.resize(image, (resize_dims[1], resize_dims[0]))
    #HWC => CHW
    x = np.transpose(x, (2,0,1))
    #BGR => RGB
    #x = x[[2,1,0],:,:]
    #raw scale
    #x = x / 255.0
    #mean = np.array([104.01, 116.67, 112.68], dtype=np.float32)
    #mean = mean[:, np.newaxis, np.newaxis]
    #print(mean)
    #x = x - mean
    x = center_crop(x, (net_input_dims[0], net_input_dims[1]))
    x = np.expand_dims(x, axis=0)
    x = np.tile(x, (args.batch, 1, 1, 1))
    x = x.astype(np.float32)
    np.savez(args.save, x)
