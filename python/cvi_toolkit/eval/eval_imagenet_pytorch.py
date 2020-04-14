#!/usr/bin/env python3
# use pytorch for dataloader
# https://github.com/pytorch/examples/blob/master/imagenet/main.py

import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np
from PIL import Image
# import matplotlib.pyplot as plt

import pymlir

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

parser = argparse.ArgumentParser(description="Classification Evaluation on ImageNet Dataset.")
parser.add_argument("--model", type=str)
parser.add_argument("--dataset", type=str, help="The root directory of the ImageNet dataset.")
parser.add_argument("--image_resize_dims", type=str, default='256,256')
parser.add_argument("--net_input_dims", type=str, default='224,224')
parser.add_argument("--raw_scale", type=float, help="Multiply raw input image data by this scale.")
parser.add_argument("--mean", help="Per Channel image mean values")
parser.add_argument("--mean_file", type=str, help="the resized ImageNet dataset mean file.")
parser.add_argument("--input_scale", type=float,
                    help="Multiply input features by this scale.", default=1.0)
parser.add_argument("--count", type=int, default=50000)
parser.add_argument("--dump_data", type=bool, default=False)
parser.add_argument("--show", type=bool, default=False)
parser.add_argument("--loader_transforms", type=int, help="image transform ny torch loader", default=0)
args = parser.parse_args()

def second(elem):
  return elem[1]

def get_topk(a, k):
  idx = np.argpartition(-a.ravel(),k)[:k]
  # return np.column_stack(np.unravel_index(idx, a.shape))
  topk = list(zip(idx, np.take(a, idx)))
  #return topk
  topk.sort(key=second, reverse=True)
  return topk

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

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
  do_loader_transforms = args.loader_transforms
  traindir = os.path.join(args.dataset, 'train')
  valdir = os.path.join(args.dataset, 'val')
  # onedir = os.path.join(args.dataset, 'one')
  batch_size = 1

  image_resize_dims = [int(s) for s in args.image_resize_dims.split(',')]
  net_input_dims = [int(s) for s in args.net_input_dims.split(',')]
  image_resize_dims = [ max(x,y) for (x,y) in zip(image_resize_dims, net_input_dims)]

  if args.raw_scale:
    raw_scale = float(args.raw_scale)
  else:
    raw_scale = 255.0
  if args.mean:
    mean = np.array([float(s) for s in args.mean.split(',')], dtype=np.float32)
    print('mean', mean)
    mean = mean[:, np.newaxis, np.newaxis]
  else:
    if args.mean_file:
      mean = np.load(args.mean_file)
      # print('mean shape', mean.shape)
      # only need the 3D value
      mean = mean[0]
    else:
      mean = np.array([])

  input_scale = float(args.input_scale)

  # load model
  module = pymlir.module()
  print('load module ', args.model)
  module.load(args.model)
  print('load module done')

  if (do_loader_transforms):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(image_resize_dims),
            transforms.CenterCrop(net_input_dims),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=True)
  else:
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(image_resize_dims),
            transforms.CenterCrop(net_input_dims),
            transforms.ToTensor()
        ])),
        batch_size=batch_size, shuffle=True)

  # validate(val_loader, module, criterion, args)
  batch_time = AverageMeter('Time', ':6.3f')
  losses = AverageMeter('Loss', ':.4e')
  top1 = AverageMeter('Acc@1', ':6.2f')
  top5 = AverageMeter('Acc@5', ':6.2f')
  progress = ProgressMeter(
      len(val_loader),
      [batch_time, losses, top1, top5],
      prefix='Test: ')

  # define loss function (criterion) and optimizer
  criterion = nn.CrossEntropyLoss()

  end = time.time()
  for i, (images, target) in enumerate(val_loader):
    # compute output
    # output = model(images)
    if (do_loader_transforms):
      # loader do normalize already
      x = images[0].numpy() * input_scale

      # reshape 4 dim for cpu inference
      x = np.expand_dims(x, axis=0)
    else:
      # pytorch ToTensor() will do HWC to CHW, and change range to [0.0, 1.0]
      # for pytorch, seeing errors if not include ToTensor in transforms
      # change to range [0, 255]
      x = images[0].numpy() * 255
      x = x.astype('uint8')
      # print(x.shape)
      # transposed already in ToTensor()
      # x = np.transpose(x, (2, 0, 1))
      # still need the swap for caffe models
      x = x[[2,1,0], :, :]
      # print('img mean int8', np.mean(np.reshape(img_swap, (3, -1)), axis=1))
      # print('img std int8', np.std(np.reshape(img_swap, (3, -1)), axis=1))
      x = x.astype(np.float32)
      x = x * raw_scale / 255.0
      # apply mean
      if mean.size != 0:
        x -= mean
      # expand to 4-D again
      x = np.expand_dims(x, axis=0)

    if input_scale != 1.0:
      x *= input_scale

    # run inference
    # print('x.shape', x.shape)
    res = module.run(x)
    # print('res.shape', res.shape)
    assert(len(res) == 1)
    prob  = list(res.values())[0]
    prob = np.reshape(prob, (prob.shape[0], prob.shape[1]))

    if args.show is True:
      for i_th in get_topk(prob, 5):
        print(i_th)
      print(target)

      if input_scale != 1.0:
        x /= input_scale
      if mean.size != 0:
        x += mean
      x = x[0]
      # print(x.shape)
      x = x[[2,1,0], :, :]
      x = np.transpose(x, (1, 2, 0))
      # print(x.shape)
      im = Image.fromarray(np.uint8(x))
      imgplot = plt.imshow(im)
      plt.show()

    output = torch.from_numpy(prob)

    loss = criterion(output, target)

    # measure accuracy and record loss
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    losses.update(loss.item(), images.size(0))
    top1.update(acc1[0], images.size(0))
    top5.update(acc5[0], images.size(0))

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    # if i % args.print_freq == 0:
    if (i + 1) % 50 == 0:
      progress.display(i + 1)

    if (i + 1) >= int(args.count):
      break

  # TODO: this should also be done with the ProgressMeter
  print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        .format(top1=top1, top5=top5))

  print(top1.avg)
