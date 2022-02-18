#!/usr/bin/env python3

import cv2
import pyruntime

# read input
image = cv2.imread("input.png", cv2.IMREAD_GRAYSCALE)
template = cv2.imread("template.png", cv2.IMREAD_GRAYSCALE)
ih,iw = image.shape
th,tw = template.shape
oh,ow = (ih-th+1),(iw-tw+1)

# ======= by cvimodel ==============
model = pyruntime.Model("ccoeff_normed.cvimodel")
if model == None:
    raise Exception("cannot load cvimodel")

# fill data to inputs
data0 = model.inputs[0].data
data1 = model.inputs[1].data
data0[:] = image.reshape(data0.shape)
data1[:] = template.reshape(data1.shape)
# forward
model.forward()
output = model.outputs[0].data.flatten()[0]
model_loc = (int(output%ow),int(output/ow))
print("model location(x,y):{}".format(model_loc))

# ======== by opencv ==============
res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
_,_,_,max_loc = cv2.minMaxLoc(res)
print("opencv location(x,y):{}".format(max_loc))

if model_loc == max_loc:
    print("match success")
else:
    print("match failed")
