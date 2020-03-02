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

if len(sys.argv) < 7:
  print("Usage: %s filename.bin int8|bf16|float32 N C H W [K]" % sys.argv[0])
  exit(-1)

f = sys.argv[1]
t = sys.argv[2]
n = int(sys.argv[3])
c = int(sys.argv[4])
h = int(sys.argv[5])
w = int(sys.argv[6])
K = 0
if len(sys.argv) == 8:
  K = int(sys.argv[7])

if t == 'bf16':
  t = 'uint16'

if t != 'int8' and t != 'float32' and t!='uint16':
  print("Invalid type", t)
  exit(-1)

d = np.fromfile(f, dtype=t)
if n == 0 and c == 0 and h == 0:
  d = np.reshape(np.ravel(d), (w))
elif n == 0 and c == 0:
  d = np.reshape(np.ravel(d), (h, w))
elif n == 0:
  d = np.reshape(np.ravel(d), (c, h, w))
else:
  d = np.reshape(np.ravel(d), (n, c, h, w))
np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)
if K < 0:
  np.set_printoptions(threshold=sys.maxsize)
print(d)
print('shape', d.shape)

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
