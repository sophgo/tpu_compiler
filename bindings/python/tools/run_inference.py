#!/usr/bin/python

import sys
import numpy as np
import pymlir

def second(elem):
  return elem[1]

def get_topk(a, k):
  idx = np.argpartition(-a.ravel(),k)[:k]
  # return np.column_stack(np.unravel_index(idx, a.shape))
  topk = list(zip(idx, np.take(a, idx)))
  #return topk
  topk.sort(key=second, reverse=True)
  return topk

if len(sys.argv) < 3 or len(sys.argv) > 4:
  print("Usage: %s model.mlir input.npy [K]" % sys.argv[0])
  exit(-1)

K = 0
if len(sys.argv) == 4:
  K = int(sys.argv[3])

module = pymlir.module()
module.load(sys.argv[1])
print("load module done")
# module.dump()

x = np.load(sys.argv[2])
print('x.shape', x.shape)
res = module.run(x)
print('res.shape', res.shape)

if K > 0:
  print("Show Top-K", K)
  # print(get_topk(data, K), sep='\n')
  for i in get_topk(res, K):
    print(i)
