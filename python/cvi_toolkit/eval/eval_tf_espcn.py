#!/usr/bin/env python3

import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np
import tensorflow as tf
from skimage.measure import compare_ssim

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
    description="Structural similarity index")
parser.add_argument("--model_def", type=str,
                    help="Model definition file", default=None)
parser.add_argument("--pretrained_model", type=str,
                    help="Load weights from previously saved parameters.", default=None)
parser.add_argument("--mlir_file", type=str, help="mlir file.", default=None)
parser.add_argument("--dataset", type=str,
                    help="The root directory of the ImageNet dataset.")
parser.add_argument("--model_type", type=str,
                    help="model framework type, default: caffe", default='caffe')
parser.add_argument("--input_data_format", type=str,
                    help="model framework input shape order, default hwc", default='hwc')
parser.add_argument("--count", type=int, default=50000)
parser = get_preprocess_parser(existed_parser=parser)

args = parser.parse_args()

class BasicDataset():
  def __init__(self, imgs_dir):
    self.imgs_dir = imgs_dir

    self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                if not file.startswith('.')]
    print(f'Creating dataset with {len(self.ids)} examples')

  def __len__(self):
      return len(self.ids)

  def __getitem__(self, i):
    idx = self.ids[i]
    print(self.imgs_dir+ idx + '.*')
    img_file = glob(self.imgs_dir + idx + '.*')

    assert len(img_file) == 1, \
        f'Either no image or multiple images found for the ID {idx}: {img_file}'
    # https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    img = Image.open(img_file[0]).convert('L')

    return {
        'image': torch.from_numpy(np.array(img)).type(torch.FloatTensor),
        'path': img_file,
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

# copy from https://stackoverflow.com/questions/39051451/ssim-ms-ssim-for-tensorflow
def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
    weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    for l in range(level):
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        filtered_im1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        filtered_im2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
        img1 = filtered_im1
        img2 = filtered_im2

    # list to tensor of dim D+1
    mssim = tf.pack(mssim, axis=0)
    mcs = tf.pack(mcs, axis=0)

    value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                            (mssim[level-1]**weight[level-1]))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value

def accuracy(img, img_noise):
    # get h/w, 1, 256, 256, 1 -> 256,256
    img_hw = img[0, :, :, 0]
    img_noise_hw = img_noise[0, :, :, 0]
    (score, diff) = compare_ssim(img_hw, img_noise_hw, full=True)
    diff = (diff * 255).astype("uint8")
    #print("SSIM: {}".format(score))
    return score
    #return ssim(img[0], img_noise[0],
    #              data_range=img_noise.max() - img_noise.min())
    #return tf_ssim(img_noise, img)

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
  dir_img = os.path.join(args.dataset, 'val/')

  val_dataset = BasicDataset(dir_img)
  # https://github.com/pytorch/pytorch/issues/1579
  batch_size = 1
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

  net = ModelFactory()
  # load model
  net.load_model(args.model_type, model_file=args.model_def,
                 weight_file=args.pretrained_model, mlirfile=args.mlir_file)

  if args.net_input_dims:
      net_input_dims = args.net_input_dims
  else:
      # read from caffe
      net_input_dims = net.get_input_shape()

  preprocessor = preprocess()
  # Because of Resize by PyTorch transforms, we set resize dim same with network input(don't do anything )
  # transposed already in ToTensor(),
  preprocessor.config(net_input_dims=net_input_dims,
                      resize_dims=net_input_dims,
                      mean=args.mean,
                      mean_file=args.mean_file,
                      input_scale=args.input_scale,
                      raw_scale=args.raw_scale,
                      std=args.std,
                      rgb_order=args.model_channel_order,
                      data_format="nhwc",
                      bgray=True)

  image_resize_dims = [int(s) for s in args.image_resize_dims.split(',')]
  net_input_dims = [int(s) for s in args.net_input_dims.split(',')]
  image_resize_dims = [max(x, y)
                       for (x, y) in zip(image_resize_dims, net_input_dims)]
  raw_scale = args.raw_scale

  # validate(val_loader, module, criterion, args)
  batch_time = AverageMeter('Time', ':6.3f')
  top1 = AverageMeter('ssim', ':6.2f')
  progress = ProgressMeter(
      len(val_dataset),
      [batch_time, top1],
      prefix='Test: ')

  end = time.time()
  i = 0
  for batch in val_loader:
    image = batch['image']
    path = batch['path'][0][batch_size-1]
    # compute output
    #print(image.shape)
    ## gray shape should be <1, h, w>
    x = image.numpy() * 255

    # slign tf shape order: hwc
    x = np.transpose(x, (1, 2, 0))
    img = np.expand_dims(x, axis=0)
    # only one channel, input_channel_order/output_channel_order dont care
    x = preprocessor.run(path, output_channel_order=args.model_channel_order,
            input_data_format = args.input_data_format)

    # run inference
    res = net.inference(x)
    # labelVisualize(output, i, COLOR_DICT)
    # measure accuracy and record loss
    acc1 = accuracy(img, res)
    #print("acc1: ", acc1)
    top1.update(acc1)

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    progress.display(i + 1)
    i = i + 1


  print(top1.avg)
