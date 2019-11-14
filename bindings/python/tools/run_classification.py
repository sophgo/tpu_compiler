#!/usr/bin/env python2
# use mxnet for dataloader for now

import sys
import os
import numpy as np
import argparse
#import matplotlib
import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon.data.vision import transforms
from gluoncv.data import imagenet
from collections import namedtuple
import pymlir

parser = argparse.ArgumentParser(description="Classification Evaluation on ImageNet Dataset.")
parser.add_argument("--model", type=str)
parser.add_argument("--dataset", type=str, help="The root directory of the ImageNet dataset.")
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
  # MX ctx for data loading
  ctx = [mx.cpu()]

  batch_size = 1
  num_batches = int(args.count)/batch_size

  # load model
  module = pymlir.module()
  module.load(args.model)
  print("load module done")

  # Define evaluation metrics
  acc_top1 = mx.metric.Accuracy()
  acc_top5 = mx.metric.TopKAccuracy(5)

  # Define image transforms
  # Move ToTensor() and normalize later, for dumping purpose
  if (False):
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
    batch_size=batch_size, shuffle=False)

  # Compute evaluations
  acc_top1.reset()
  acc_top5.reset()
  print('[0 / %d] batches done'%(num_batches))
  # Loop over batches
  for i, batch in enumerate(val_data):
    # Load batch
    data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
    label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
    # Do more transform
    orig_img = data[0].asnumpy().astype('uint8')
    img = np.transpose(orig_img[0], (2, 0, 1))
    img_swap = img[[2,1,0], :, :]
    # print('img mean int8', np.mean(np.reshape(img_swap, (3, -1)), axis=1))
    # print('img std int8', np.std(np.reshape(img_swap, (3, -1)), axis=1))
    d = img_swap.astype(np.float32)
    mean = np.load(args.mean_file)
    # expand to 4-D again
    x = np.expand_dims(d, axis=0)
    x -= mean
    if args.input_scale is not None:
      x *= float(args.input_scale)
    # print('x mean int8', np.mean(np.reshape(x, (3, -1)), axis=1))
    # print('x std int8', np.std(np.reshape(x, (3, -1)), axis=1))
    #inputs = np.ascontiguousarray(img)

    # Perform forward pass
    if True:
      # print('x.shape', x.shape)
      res = module.run(x)
      # print('res.shape', res.shape)

    # Update accuracy metrics
    outputs = [mx.nd.array(res)]
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

