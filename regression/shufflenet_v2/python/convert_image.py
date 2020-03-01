#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

if __name__ == '__main__':
    import argparse
    import PIL.Image
    from PIL.Image import BICUBIC
    import torchvision
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str)
    parser.add_argument('--image_hw', type=int, default=224)
    parser.add_argument('--save', type=str)
    args = parser.parse_args()
    img = PIL.Image.open(args.image).convert('RGB').resize([256,256], BICUBIC)
    img = torchvision.transforms.functional.resize(img, (args.image_hw, args.image_hw))
    img = torchvision.transforms.functional.to_tensor(img).unsqueeze(0).numpy()
    np.savez(args.save, data=img)
    np.save(args.save, img)
