#!/usr/bin/env python3

import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np

from cvi_toolkit.data.preprocess import preprocess, get_preprocess_parser
from cvi_toolkit.model.ModelFactory import ModelFactory

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from os import listdir
from os.path import splitext
import logging
from glob import glob
from PIL import Image
import torch.nn.functional as F
import skimage.io as io


parser = argparse.ArgumentParser(
    description="Classification Evaluation on ImageNet Dataset.")
parser.add_argument("--model_def", type=str,
                    help="Model definition file", default=None)
parser.add_argument("--pretrained_model", type=str,
                    help="Load weights from previously saved parameters.", default=None)
parser.add_argument("--mlir_file", type=str, help="mlir file.", default=None)
parser.add_argument("--dataset", type=str,
                    help="The root directory of the ImageNet dataset.")
parser.add_argument("--model_type", type=str,
                    help="model framework type, default: caffe", default='caffe')
parser.add_argument("--count", type=int, default=50000)
parser = get_preprocess_parser(existed_parser=parser)

args = parser.parse_args()

class BasicDataset():
  def __init__(self, imgs_dir, masks_dir):
    self.imgs_dir = imgs_dir
    self.masks_dir = masks_dir
    self.mask_size = (256,256)

    self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                if not file.startswith('.')]
    print(f'Creating dataset with {len(self.ids)} examples')

  def __len__(self):
      return len(self.ids)

  def __getitem__(self, i):
    idx = self.ids[i]
    print(self.masks_dir + idx + '.*')
    mask_file = glob(self.masks_dir + idx + '.*')
    img_file = glob(self.imgs_dir + idx + '.*')

    assert len(mask_file) == 1, \
        f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
    assert len(img_file) == 1, \
        f'Either no image or multiple images found for the ID {idx}: {img_file}'
    mask = Image.open(mask_file[0]).resize(self.mask_size) /255
    img = Image.open(img_file[0]).convert('RGB')

    return {
        'image': torch.from_numpy(np.array(img)).type(torch.FloatTensor),
        'mask': torch.from_numpy(np.array(mask)).type(torch.FloatTensor)
    }

def second(elem):
  return elem[1]

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(img, mask):
    idx_lt = mask < 0.5
    idx_gt = mask >= 0.5
    mask[idx_lt] = 0
    mask[idx_gt] = 1
    idx_lt = img < 0.5
    idx_gt = img >= 0.5
    img[idx_lt] = 0
    img[idx_gt] = 1
    return np.sum(img == mask)/img.size

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])

def labelVisualize(img, idx, color_dict,num_class = 2):
    img = np.asarray(img).reshape(256,256)
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    img_out = img_out / 255
    print(os.getcwd())
    io.imsave(os.path.join(os.getcwd(),"%d_predict.png"%idx),img_out)

if __name__ == '__main__':
  val_dir = os.path.join(args.dataset, 'val/')
  dir_img = os.path.join(val_dir, "image/")
  dir_mask = os.path.join(val_dir, "mask/")

  val_dataset = BasicDataset(dir_img, dir_mask)
  val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
  batch_size = 1

  net = ModelFactory()
  # load model
  net.load_model(args.model_type, model_file=args.model_def,
                 weight_file=args.pretrained_model, mlirfile=args.mlir_file)

  args.net_input_dims = args.net_input_dims if args.net_input_dims else \
                        net.get_input_shape()
  preprocessor = preprocess()
  # Because of Resize by PyTorch transforms, we set resize dim same with network input(don't do anything )
  # transposed already in ToTensor(),
  args.pixel_format = 'GRAYSCALE'
  preprocessor.config(**vars(args))


  # validate(val_loader, module, criterion, args)
  batch_time = AverageMeter('Time', ':6.3f')
  top1 = AverageMeter('Acc@1', ':6.2f')
  progress = ProgressMeter(
      len(val_dataset),
      [batch_time, top1],
      prefix='Test: ')

  end = time.time()
  i = 0
  for batch in val_loader:
    image = batch['image']
    mask = batch['mask']
    # compute output
    x = image[0].numpy() * 255
    x = np.transpose(x, (1,2,0))
    x = preprocessor.run(x)

    # run inference

    res = net.inference(x)
    output_shape = (res.shape[2], res.shape[3])
    output = np.reshape(res, output_shape)
    # labelVisualize(output, i, COLOR_DICT)
    # measure accuracy and record loss
    acc1 = accuracy(output, np.asarray(mask))
    print("acc1: ", acc1)
    top1.update(acc1)

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    progress.display(i + 1)
    i = i + 1


  print(top1.avg)
