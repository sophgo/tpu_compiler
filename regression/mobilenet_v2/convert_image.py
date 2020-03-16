#!/usr/bin/env python
# encoding: utf-8

import argparse
import sys, os, cv2
import numpy as np

def center_crop(img,crop_dim):
  print(img.shape)
  h,w,_ = img.shape
  cropy,cropx = crop_dim
  startx = w//2-(cropx//2)
  starty = h//2-(cropy//2)
  return img[starty:starty+cropy, startx:startx+cropx, :]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str)
    parser.add_argument('--image_hw', type=int, default=224)
    parser.add_argument('--save', type=str)
    args = parser.parse_args()
    image = cv2.imread(args.image) #BGR
    image = image.astype(np.float32)
    #[0,255] => [0,1]
    image = image / 255.0
    x = cv2.resize(image, (256,256))
    x = center_crop(x, (args.image_hw, args.image_hw))
    #HWC => CHW
    x = np.transpose(x, (2,0,1))
    x = x[[2,1,0],:,:] #BGR => RGB
    x = np.expand_dims(x, axis=0)
    np.savez(args.save, data=x)
    np.save(args.save, x)

