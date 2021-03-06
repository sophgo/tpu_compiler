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

from cvi_toolkit.data.preprocess import preprocess, get_preprocess_parser
from cvi_toolkit.model.ModelFactory import ModelFactory

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

parser = argparse.ArgumentParser(
    description="Classification Evaluation on ImageNet Dataset.")
parser.add_argument("--model_def", type=str,
                    help="Model definition file", default=None)
parser.add_argument("--pretrained_model", type=str,
                    help="Load weights from previously saved parameters.", default=None)
parser.add_argument("--mlir_file", type=str, help="mlir file.", default=None)
parser.add_argument("--label_file", type=str, help="label file.", default=None)
parser.add_argument("--dataset", type=str,
                    help="The root directory of the ImageNet dataset.")
parser.add_argument("--model_type", type=str,
                    help="model framework type, default: caffe", default='caffe')
parser.add_argument("--count", type=int, default=50000)
parser = get_preprocess_parser(existed_parser=parser)

args = parser.parse_args()


def second(elem):
  return elem[1]


def get_topk(a, k):
  idx = np.argpartition(-a.ravel(), k)[:k]
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

  traindir = os.path.join(args.dataset, 'train')
  valdir = os.path.join(args.dataset, 'val')
  # onedir = os.path.join(args.dataset, 'one')
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
  preprocessor.config(**vars(args))

  val_loader = torch.utils.data.DataLoader(
      datasets.ImageFolder(valdir, transforms.Compose([
          transforms.Resize(preprocessor.resize_dims),
          transforms.CenterCrop(preprocessor.net_input_dims),
          transforms.ToTensor()
      ])), batch_size=batch_size, shuffle=True)
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

  # target translate
  label_map = list()
  if args.label_file != None:
    labels = np.loadtxt(args.label_file, str, delimiter='\t')
    sorted_labels = sorted(labels)
    label_map = [i for i in range(len(labels))]
    is_sorted = True
    for i,label in enumerate(labels):
      index = sorted_labels.index(label)
      label_map[index] = i
      if index != i and is_sorted:
        is_sorted = False
    if is_sorted:
      label_map.clear()

  end = time.time()
  for i, (images, target) in enumerate(val_loader):
    if len(label_map) != 0:
      # adjust target
      for j in range(len(target)):
        target[j] = label_map[target[j]]
    # compute output
    # output = model(images)
    # Pytorch ToTensor will make tesnor range to [0, 1]
    # recover to [0, 255]
    x = images[0].numpy() * 255
    if preprocessor.channel_order == 'bgr':
      x = x[[2, 1, 0], :, :]

    x *= preprocessor.perchannel_scale
    x -= preprocessor.perchannel_mean
    x = np.expand_dims(x, axis=0)
    # run inference
    res = net.inference({net.get_input_name()[0]: x})
    res = np.reshape(res, (res.shape[0], res.shape[1]))

    output = torch.from_numpy(res)

    # tensorflow slim mobilenet has 1001 classes, all classes will be shift one
    # the first element is empty background class
    if output.size(1) == 1001:
        output = output[:, 1:1001]
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
    if (i + 1) % 1000 == 0:
      progress.display(i + 1)

    if int(args.count) > 0 and (i + 1) >= int(args.count):
      break

  # TODO: this should also be done with the ProgressMeter
  print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        .format(top1=top1, top5=top5))

  print(top1.avg)
