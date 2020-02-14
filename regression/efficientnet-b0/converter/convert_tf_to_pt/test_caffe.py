import os
os.environ['GLOG_minloglevel'] = '2' 
import sys
import torch
import caffe
from PIL import Image
from torchvision import transforms
import numpy as np

if not len(sys.argv) == 3:
  sys.exit("too less argc : model name is needed! -> efficientnet-b0/efficientnet-b1/efficientnet-b2/efficientnet-b3")

type_name = sys.argv[1]
model_name = sys.argv[2]
deploy = "../caffemodel/" + type_name + "/" + model_name + ".prototxt"
caffemodel = "../caffemodel/" + type_name + "/" + model_name + ".caffemodel"
#deploy = "/data/models/caffe/MobileNetV3.prototxt"
#caffemodel = "/data/models/caffe/MobileNetV3.caffemodel"

tfms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
img = tfms(Image.open('test.jpg')).unsqueeze(0).numpy()
img.tofile("test_husky.bin")
npz_out = {}
npz_out['data'] = img
np.savez("efficientnet_in_fp32.npz", **npz_out)
#print(img)
model = caffe.Net(deploy,caffemodel,caffe.TEST)
print("image shape {}".format(img.shape))
model.blobs['data'].data[...] = img
print(img.shape)
out = model.forward()
de1 = model.blobs["_conv_stem_crop"].data
prob = model.blobs["_fc"].data

#print(de1.reshape(-1))

print(prob.reshape(-1))
print(prob.shape)
print(np.argmax(prob))
print(prob.reshape(-1)[250])
#print(img.shape)

with open("input_test.bin", "wb") as f:
    img.tofile(f)
with open("out.bin", "wb") as f2:
    prob.tofile(f2)
