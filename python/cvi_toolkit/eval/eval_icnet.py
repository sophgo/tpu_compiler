#!/usr/bin/env python3

from __future__ import print_function, absolute_import, division
import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np
import sys
import caffe
import argparse
import numpy as np
import scipy.misc
from os import listdir
from os.path import splitext
import logging
import glob
from PIL import Image
import torch.nn.functional as F

from cvi_toolkit.data.preprocess import preprocess, get_preprocess_parser
from cvi_toolkit.model.ModelFactory import ModelFactory

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from skimage.transform import resize


#!/usr/bin/python
#
# Cityscapes labels
#
# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py

from collections import namedtuple


#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]


#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label      = { label.name    : label for label in labels           }
# id to label object
id2label        = { label.id      : label for label in labels           }
# trainId to label object
trainId2label   = { label.trainId : label for label in reversed(labels) }
# category to list of label objects
category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]

#--------------------------------------------------------------------------------
# Assure single instance name
#--------------------------------------------------------------------------------

# returns the label name that describes a single instance (if possible)
# e.g.     input     |   output
#        ----------------------
#          car       |   car
#          cargroup  |   car
#          foo       |   None
#          foogroup  |   None
#          skygroup  |   None
def assureSingleInstanceName( name ):
    # if the name is known, it is not a group
    if name in name2label:
        return name
    # test if the name actually denotes a group
    if not name.endswith("group"):
        return None
    # remove group
    name = name[:-len("group")]
    # test if the new name exists
    if not name in name2label:
        return None
    # test if the new name denotes a label that actually has instances
    if not name2label[name].hasInstances:
        return None
    # all good then
    return name

def fast_hist(a, b, n):
    # print('saving')
    # sio.savemat('/tmp/fcn_debug/xx.mat', {'a':a, 'b':b, 'n':n})

    k = np.where((a >= 0) & (a < n))[0]
    bc = np.bincount(n * a[k].astype(int) + b[k], minlength=n**2)
    if len(bc) != n**2:
        # ignore this example if dimension mismatch
        return 0
    return bc.reshape(n, n)

def get_scores(hist):
    # Mean pixel accuracy
    acc = np.diag(hist).sum() / (hist.sum() + 1e-12)

    # Per class accuracy
    cl_acc = np.diag(hist) / (hist.sum(1) + 1e-12)

    # Per class IoU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-12)

    return acc, np.nanmean(cl_acc), np.nanmean(iu), cl_acc, iu

def get_out_scoremap(net):
    out = net.blobs['conv6_interp'].data[0].argmax(axis=0).astype(np.uint8)
    #out = net.blobs['conv6_interp'].data[0].argmax(axis=0)
    #pred = net.blobs['conv6_interp'].data[0]
    #max_idx = pred.argmax(axis=0)
    #for i, v in np.ndenumerate(max_idx):
    #    max_idx[i] = np.ravel_multi_index((v,) + i, pred.shape)
    #out = np.take(pred, max_idx).astype(np.uint8)
    return out

def feed_net(net, in_):
    """
    Load prepared input into net.
    """
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_

def segrun(net, in_):
    feed_net(net, in_)
    net.forward()
    return get_out_scoremap(net)

#--------------------------------------------------------------------------------
# Main for testing
#--------------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Classification Evaluation on CityScape Dataset.")
parser.add_argument("--model_def", type=str,
                    help="Model definition file", default=None)
parser.add_argument("--pretrained_model", type=str,
                    help="Load weights from previously saved parameters.", default=None)
parser.add_argument("--mlir_file", type=str, help="mlir file.", default=None)
parser.add_argument("--dataset", type=str,
                    help="Path to the original cityscapes dataset")
parser.add_argument("--model_type", type=str,
                    help="model framework type, default: caffe", default='caffe')
parser.add_argument("--result_dir", type=str, required=True, help="Path to the generated images to be evaluated")
parser.add_argument("--output_dir", type=str, required=True, help="Where to save the evaluation results")
#parser.add_argument("--gpu_id", type=int, default=0, help="Which gpu id to use")
parser.add_argument("--split", type=str, default='val', help="Data split to be evaluated")
parser.add_argument("--save_output_images", type=int, default=0, help="Whether to save the FCN output images")

parser = get_preprocess_parser(existed_parser=parser)
args = parser.parse_args()

class cityscapes:
    def __init__(self, data_path):
        # data_path something like /data2/cityscapes
        self.dir = data_path
        self.classes = ['road', 'sidewalk', 'building', 'wall', 'fence',
                        'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                        'sky', 'person', 'rider', 'car', 'truck',
                        'bus', 'train', 'motorcycle', 'bicycle']
        #self.mean = np.array((72.78044, 83.21195, 73.45286), dtype=np.float32)
        #self.mean = np.array((123.68, 116.779, 103.939), dtype=np.float32)
        mean = [float(x) for x in args.mean.split(",")]
        self.mean = np.array(mean, dtype=np.float32)
        # import cityscapes label helper and set up label mappings
        sys.path.insert(0, '{}/scripts/helpers/'.format(self.dir))
        self.id2trainId = {label.id: label.trainId for label in labels}  # dictionary mapping from raw IDs to train IDs
        self.trainId2color = {label.trainId: label.color for label in labels}  # dictionary mapping train IDs to colors as 3-tuples

    def get_dset(self, split):
        '''
        List images as (city, id) for the specified split

        TODO(shelhamer) generate splits from cityscapes itself, instead of
        relying on these separately made text files.
        '''
        if split == 'train':
            dataset = open('{}/ImageSets/segFine/train.txt'.format(self.dir)).read().splitlines()
        else:
            dataset = open('{}/ImageSets/segFine/val.txt'.format(self.dir)).read().splitlines()
        return [(item.split('/')[0], item.split('/')[1]) for item in dataset]

    def load_image(self, split, city, idx):
        im = Image.open('{}/leftImg8bit_sequence/{}/{}/{}_leftImg8bit.png'.format(self.dir, split, city, idx))
        return im

    def assign_trainIds(self, label):
        """
        Map the given label IDs to the train IDs appropriate for training
        Use the label mapping provided in labels.py from the cityscapes scripts
        """
        label = np.array(label, dtype=np.float32)
        if sys.version_info[0] < 3:
            for k, v in self.id2trainId.iteritems():
                label[label == k] = v
        else:
            for k, v in self.id2trainId.items():
                label[label == k] = v
        return label

    def load_label(self, split, city, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        label = Image.open('{}/{}/{}/{}_gtFine_labelIds.png'.format(self.dir, split, city, idx))
        label = self.assign_trainIds(label)  # get proper labels for eval
        label = np.array(label, dtype=np.uint8)
        label = label[np.newaxis, ...]
        return label

    def preprocess(self, im):
        """
        Preprocess loaded image (by load_image) for Caffe:
        - cast to float
        - subtract mean
        - switch channels RGB -> BGR
        - transpose to channel x height x width order
        https://github.com/hszhao/ICNet/blob/master/evaluation/eval_sub.m
        """
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:, :, ::-1]
        #in_ -= self.mean
        in_ = in_.transpose((2, 0, 1)) # chw
        # bgr to rgb
        in_ = in_[[2,1,0], :, :]
        mean = self.mean[:, np.newaxis, np.newaxis]
        in_ -= mean
        in_ = in_[[0,1,2], :, :] # bgr
        return in_

    def palette(self, label):
        '''
        Map trainIds to colors as specified in labels.py
        '''
        if label.ndim == 3:
            label= label[0]
        color = np.empty((label.shape[0], label.shape[1], 3))
        if sys.version_info[0] < 3:
            for k, v in self.trainId2color.iteritems():
                color[label == k, :] = v
        else:
            for k, v in self.trainId2color.items():
                color[label == k, :] = v
        return color

    def make_boundaries(label, thickness=None):
        """
        Input is an image label, output is a numpy array mask encoding the boundaries of the objects
        Extract pixels at the true boundary by dilation - erosion of label.
        Don't just pick the void label as it is not exclusive to the boundaries.
        """
        assert(thickness is not None)
        import skimage.morphology as skm
        void = 255
        mask = np.logical_and(label > 0, label != void)[0]
        selem = skm.disk(thickness)
        boundaries = np.logical_xor(skm.dilation(mask, selem),
                                    skm.erosion(mask, selem))
        return boundaries

    def list_label_frames(self, split):
        """
        Select labeled frames from a split for evaluation
        collected as (city, shot, idx) tuples
        """
        def file2idx(f):
            """Helper to convert file path into frame ID"""
            city, shot, frame = (os.path.basename(f).split('_')[:3])
            return "_".join([city, shot, frame])
        frames = []
        cities = [os.path.basename(f) for f in glob.glob('{}/{}/*'.format(self.dir, split))]
        for c in cities:
            files = sorted(glob.glob('{}/{}/{}/*labelIds.png'.format(self.dir, split, c)))
            nr = len(files)
            for i in files:
                idx = file2idx(i)
                path = args.result_dir + '/' + idx + '_leftImg8bit.png'
                if not os.path.exists(path):
                    files.remove(i)
                    print("{} not exist, skip".format(path))
            frames.extend([file2idx(f) for f in files])
            print("please remove it for test")
            break
        return frames

    def collect_frame_sequence(self, split, idx, length):
        """
        Collect sequence of frames preceding (and including) a labeled frame
        as a list of Images.

        Note: 19 preceding frames are provided for each labeled frame.
        """
        SEQ_LEN = length
        city, shot, frame = idx.split('_')
        frame = int(frame)
        frame_seq = []
        for i in range(frame - SEQ_LEN, frame + 1):
            frame_path = '{0}/leftImg8bit_sequence/val/{1}/{1}_{2}_{3:0>6d}_leftImg8bit.png'.format(
                self.dir, city, shot, i)
            frame_seq.append(Image.open(frame_path))
        return frame_seq

def main():
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    if args.save_output_images > 0:
        output_image_dir = args.output_dir + 'image_outputs/'
        if not os.path.isdir(output_image_dir):
            os.makedirs(output_image_dir)
    CS = cityscapes(args.dataset)
    n_cl = len(CS.classes)
    label_frames = CS.list_label_frames(args.split)
    #caffe.set_device(args.gpu_id)
    #caffe.set_mode_gpu()
    net = caffe.Net(args.model_def,
                    args.pretrained_model,
                    caffe.TEST)

    hist_perframe = np.zeros((n_cl, n_cl))
    for i, idx in enumerate(label_frames):
        if i % 10 == 0:
            print('Evaluating: %d/%d' % (i, len(label_frames)))
        city = idx.split('_')[0]
        # idx is city_shot_frame
        label = CS.load_label(args.split, city, idx)
        #print(city, label, idx, "XX")
        im_file = args.result_dir + '/' + idx + '_leftImg8bit.png'
        im = np.array(Image.open(im_file))
        # im = scipy.misc.imresize(im, (256, 256))
        #im = scipy.misc.imresize(im, (label.shape[1], label.shape[2]))
        #im = resize(im, (label.shape[1], label.shape[2]))
        image_resize_dims = [int(x) for x in args.image_resize_dims.split(",")]
        im = resize(im, (image_resize_dims[0], image_resize_dims[1]))
        out = segrun(net, CS.preprocess(im))
        hist_perframe += fast_hist(label.flatten(), out.flatten(), n_cl)
        if args.save_output_images > 0:
            label_im = CS.palette(label)
            pred_im = CS.palette(out)
            scipy.misc.imsave(output_image_dir + '/' + str(i) + '_pred.jpg', pred_im)
            scipy.misc.imsave(output_image_dir + '/' + str(i) + '_gt.jpg', label_im)
            scipy.misc.imsave(output_image_dir + '/' + str(i) + '_input.jpg', im)

    mean_pixel_acc, mean_class_acc, mean_class_iou, per_class_acc, per_class_iou = get_scores(hist_perframe)
    with open(args.output_dir + '/evaluation_results.txt', 'w') as f:
        f.write('Mean pixel accuracy: %f\n' % mean_pixel_acc)
        f.write('Mean class accuracy: %f\n' % mean_class_acc)
        f.write('Mean class IoU: %f\n' % mean_class_iou)
        f.write('************ Per class numbers below ************\n')
        for i, cl in enumerate(CS.classes):
            while len(cl) < 15:
                cl = cl + ' '
            f.write('%s: acc = %f, iou = %f\n' % (cl, per_class_acc[i], per_class_iou[i]))

if __name__ == '__main__':
    main()
