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
    parser.add_argument('--yolo', type=str, default='false')
    parser.add_argument('--save', type=str)
    args = parser.parse_args()
    resize_dims = [int(dim) for dim in args.resize_dims.strip().split(',')]
    net_input_dims = [int(dim) for dim in args.net_input_dims.strip().split(',')]
    image = cv2.imread(args.image) #BGR
    x = image
    if args.yolo == 'false':
        if image.shape[0] != resize_dims[0]:
            x = cv2.resize(x, (resize_dims[1], resize_dims[0]))
        x = x.astype(np.float32)
        #HWC => CHW
        x = np.transpose(x, (2,0,1))
        #BGR => RGB
        #x = x[[2,1,0],:,:]
        #raw scale
        #x = x / 255.0
        #mean = np.array([104.01,116.67,122.68], dtype=np.float32)
        #mean = mean[:, np.newaxis, np.newaxis]
        #print(mean)
        #x = x - mean
        if x.shape[2] != net_input_dims[1]:
            x = center_crop(x, (net_input_dims[0], net_input_dims[1]))
    else:
        ih = image.shape[0]
        iw = image.shape[1]

        scale = min(float(net_input_dims[1]) / iw, float(net_input_dims[0]) / ih)
        rescale_w = int(iw * scale)
        rescale_h = int(ih * scale)

        resized_img = cv2.resize(image, (rescale_w, rescale_h), interpolation=cv2.INTER_LINEAR)
        x = np.full((net_input_dims[0], net_input_dims[1], 3), 0, dtype=np.float32)
        paste_w = (net_input_dims[1] - rescale_w) // 2
        paste_h = (net_input_dims[0] - rescale_h) // 2

        x[paste_h:paste_h + rescale_h, paste_w: paste_w + rescale_w, :] = resized_img
        x = np.transpose(x, (2, 0, 1))      # row to col, (HWC -> CHW)

    x = np.expand_dims(x, axis=0)
    x = np.tile(x, (args.batch, 1, 1, 1))
    np.savez(args.save, x)
