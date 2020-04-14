#!/usr/bin/env python3
# use mxnet and gluoncv for dataloader

import sys
import os
import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt

import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon.data.vision import transforms
from gluoncv.data import imagenet
from collections import namedtuple
import pymlir

parser = argparse.ArgumentParser(description="Classification Evaluation on ImageNet Dataset.")
parser.add_argument("--model", type=str)
parser.add_argument("--dataset", type=str, help="The root directory of the ImageNet dataset.")
parser.add_argument("--mean", help="Per Channel image mean values")
parser.add_argument("--mean_file", type=str, help="the resized ImageNet dataset mean file.")
parser.add_argument("--input_scale", type=float,
                    help="Multiply input features by this scale.")
parser.add_argument("--count", type=int, default=50000)
parser.add_argument("--dump_data", type=bool, default=False)
parser.add_argument("--show", type=bool, default=False)
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

if __name__ == '__main__':
  do_loader_transforms = False
  batch_size = 1

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
  if args.input_scale:
    input_scale = float(args.input_scale)
  else:
    input_scale = 1.0

  # load model
  module = pymlir.module()
  print('load module ', args.model)
  module.load(args.model)
  print("load module done")

  # MX ctx for data loading
  ctx = [mx.cpu()]

  # Define evaluation metrics
  acc_top1 = mx.metric.Accuracy()
  acc_top5 = mx.metric.TopKAccuracy(5)

  # Define image transforms
  # Move ToTensor() and normalize later, for dumping purpose
  if (do_loader_transforms):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize
    ])
  else:
    transform_test = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224)
    ])

  # Load and process input
  val_data = gluon.data.DataLoader(
    imagenet.classification.ImageNet(args.dataset, train=False).transform_first(transform_test),
    batch_size=batch_size, shuffle=True)

  # Compute evaluations
  acc_top1.reset()
  acc_top5.reset()
  num_batches = int(args.count)/batch_size
  print('[0 / %d] batches done'%(num_batches))
  # Loop over batches
  for i, batch in enumerate(val_data):
    # Load batch
    data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
    label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
    if (do_loader_transforms):
       x = data[0].asnumpy() * 255
    else:
      # Do more transform
      x = data[0].asnumpy().astype('uint8')
      # do transpose HKC -> CHW
      x = np.transpose(x[0], (2, 0, 1))
      # do swap RGB -> BGR for caffe models
      x = x[[2,1,0], :, :]
      # print('img mean int8', np.mean(np.reshape(img_swap, (3, -1)), axis=1))
      # print('img std int8', np.std(np.reshape(img_swap, (3, -1)), axis=1))
      x = x.astype(np.float32)
      # apply mean
      if mean.size != 0:
        x -= mean
      # expand to 4-D again
      x = np.expand_dims(x, axis=0)

    if input_scale != 1.0:
      x *= input_scale
    #inputs = np.ascontiguousarray(img)

    # Perform forward pass
    # print('x.shape', x.shape)
    res = module.run(x)
    # print('res.shape', res.shape)
    assert(len(res) == 1)
    prob  = res.values()[0]

    if args.show is True:
      for i_th in get_topk(prob, 5):
        print(i_th)
      print(label)

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

    # Update accuracy metrics
    outputs = [mx.nd.array(prob)]
    acc_top1.update(label, outputs)
    acc_top5.update(label, outputs)
    if (i+1)%50==0:
      print('[%d / %d] batches done'%(i+1,num_batches))
    # if need to show or dump
    if args.show:
      o_np = outputs[0].asnumpy()
      l_np = label[0].asnumpy()
      #print('output')
      #print(o_np)
      print("Output Top-K", 5)
      for i_th in get_topk(o_np, 5):
        print(i_th)
      print('Lable')
      print(l_np)
    if (i+1)>=args.count:
      break

  # Print results
  _, top1 = acc_top1.get()
  _, top5 = acc_top5.get()
  print("Top-1 accuracy: {:.4f}, Top-5 accuracy: {:.4f}".format(top1, top5))
