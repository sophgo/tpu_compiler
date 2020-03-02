#!/usr/bin/env python

import struct
import numpy as np
import sys
import caffe
import cv2
from PIL import Image, ImageFile
#
# please install torch first
# pip install http://download.pytorch.org/whl/torch-0.1.11.post5-cp35-cp35m-macosx_10_7_x86_64.whl
# pip install torchvision
#
#from torchvision import transforms
#if len(sys.argv) < 9:
#  print("Usage: %s N C H W filename_prefix caffe_deploy_proto caffe_model output_blob_name" % sys.argv[0])
#  print("\texample: ./randomn_image.py 1 1 512 512 test_espcn_cat_in_fp32 espcn_2x.prototxt "
#          " caffe/espcn_2x.caffemodel 'Conv2D_2'")
#  exit(-1)


if len(sys.argv) < 6:
  print("Usage: %s N C H W filename_prefix " % sys.argv[0])
  print("  output: filename_prefix.jpg, filename_prefix.bin with nchw shape, bin file will astype to float32 as input")
  exit(-1)

n = int(sys.argv[1])
c = int(sys.argv[2])
h = int(sys.argv[3])
w = int(sys.argv[4])
out_name = sys.argv[5]

#mydata = np.random.randint(0, 255, (n, c, h, w))
#np.save(out_name, mydata)
img = Image.new('RGB', (h, w), color = 'red')
if c == 1:
  img = img.convert('L')

image = out_name
img.save(image, "JPEG", quality=80, optimize=True, progressive=True)
arr = np.array(img).astype('float32')
print(arr.shape, arr.dtype, "shape")
arr.tofile(out_name + ".bin")
exit (0)


#deploy = sys.argv[6]
#caffemodel = sys.argv[7]
#output_layer_name = sys.argv[8]
#
#
#img = Image.new('RGB', (h, w), color = 'red')
#if c == 1:
#  img = img.convert('L')
#
#image = out_name + ".jpg"
#img.save(image, "JPEG", quality=80, optimize=True, progressive=True)
#
#tfms = transforms.Compose([
#    transforms.Resize((h,w)),
#    transforms.ToTensor(),
#    #transforms.Grayscale(num_output_channels=1),
#    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#    ])
#tfms_img = tfms(img).unsqueeze(0).numpy()
#tfms_img = tfms_img.astype('float32')
#print("input shape", tfms_img.shape, tfms_img.dtype)
#
#model = caffe.Net(deploy,caffemodel,caffe.TEST)
#model.blobs['data'].data[...] = tfms_img
#out = model.forward()
#neuron = model.blobs[output_layer_name].data
#print(output_layer_name, "shape", neuron.shape)
#
#print("output", out_name + ".bin")
#tfms_img.tofile(out_name + ".bin")
#
#print('output', output_layer_name, 'as caffe_ref.bin')
#neuron.tofile("caffe_ref.bin")
