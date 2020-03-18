#!/usr/bin/env python

import numpy as np
import sys

def second(elem):
  return elem[1]

def get_topk(a, k):
  idx = np.argpartition(-a.ravel(),k)[:k]
  # return np.column_stack(np.unravel_index(idx, a.shape))
  topk = list(zip(idx, np.take(a, idx)))
  #return topk
  topk.sort(key=second, reverse=True)
  return topk

if not 2 <= len(sys.argv) <= 4:
  print("Usage: %s filename.npz [array_name] [K]" % sys.argv[0])
  exit(-1)

K = 0
if len(sys.argv) == 4:
  K = int(sys.argv[3])

npzfile = np.load(sys.argv[1])

if len(sys.argv) == 2 or sys.argv[2] == "--list":
  print("\n".join(npzfile.files))
  exit(0)

d = npzfile[sys.argv[2]]
np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)
if K < 0:
  np.set_printoptions(threshold=sys.maxsize)
print(d)
print('shape', d.shape)
print('dtype', d.dtype)

dims = len(d.shape)
if dims == 1:
  n = 1
  c = 1
  h = 1
  w = dims
elif dims == 2:
  n = 1
  c = 1
  h = d.shape[0]
  w = d.shape[1]
elif dims == 3:
  n = 1
  c = d.shape[0]
  h = d.shape[1]
  w = d.shape[2]
elif dims == 4:
  n = d.shape[0]
  c = d.shape[1]
  h = d.shape[2]
  w = d.shape[3]
elif dims == 5:
  n = d.shape[0]
  c = d.shape[1]
  ic = d.shape[2]
  h = d.shape[3]
  w = d.shape[4]
else:
  print("invalid shape")
  exit(-1)

if (n == 1 or n == 0) and c == 3:
  # show input image mean & std
  print('max', np.amax(np.reshape(d, (3, -1)), axis=1))
  print('min', np.amin(np.reshape(d, (3, -1)), axis=1))
  print('mean', np.mean(np.reshape(d, (3, -1)), axis=1))
  print('abs mean fp32', np.mean(np.abs(np.reshape(d, (3, -1))), axis=1))
  print('std fp32', np.std(np.reshape(d, (3, -1)), axis=1))

if K > 0:
  print("Show Top-K", K)
  # print(get_topk(data, K), sep='\n')
  for i in get_topk(d, K):
    print(i)
