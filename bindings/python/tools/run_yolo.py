#!/usr/bin/env python2

import sys
import numpy as np
import pymlir
from yolov3_util import preprocess, postprocess

if len(sys.argv) != 3:
  print("Usage: %s model.mlir input.npy" % sys.argv[0])
  exit(-1)


module = pymlir.module()
print('load module ', sys.argv[1])
module.load(sys.argv[1])
print("load module done")
# module.dump()

x = np.load(sys.argv[2])
print('x.shape', x.shape)
res = module.run(x)
# data = module.get_all_tensor()
for key, value in res.items():
  print(str(key), value.shape)

#for item in data:
#  print(item)
#print(data['layer106-conv'].shape)
#print(data['layer82-conv'].shape)
#print(data['layer94-conv'].shape)

image_shape = [576,768]
net_input_dims = [416,416]
obj_threshold = 0.3
nms_threshold = 0.5

out_feat = {}
if ('layer82-conv' in res.keys()):
  out_feat['layer82-conv'] = res['layer82-conv']
  out_feat['layer94-conv'] = res['layer94-conv']
  out_feat['layer106-conv'] = res['layer106-conv']
elif ('layer82-conv_dequant' in res.keys()):
  out_feat['layer82-conv'] = res['layer82-conv_dequant']
  out_feat['layer94-conv'] = res['layer94-conv_dequant']
  out_feat['layer106-conv'] = res['layer106-conv_dequant']
else:
  assert(False)

batched_predictions = postprocess(out_feat, image_shape, net_input_dims,
                              obj_threshold, nms_threshold, batch=1)
# batch = 1
predictions = batched_predictions[0]
print(predictions)
