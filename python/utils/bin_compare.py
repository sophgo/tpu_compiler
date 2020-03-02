#!/usr/bin/env python

import numpy as np
import sys
import struct
from tensor_compare import TensorCompare

def second(elem):
  return elem[1]

def get_topk(a, k):
  idx = np.argpartition(-a.ravel(),k)[:k]
  # return np.column_stack(np.unravel_index(idx, a.shape))
  topk = list(zip(idx, np.take(a, idx)))
  #return topk
  topk.sort(key=second, reverse=True)
  return topk

def bf16_to_fp32(d_bf16):
  s = d_bf16.shape
  d_bf16 = d_bf16.flatten()
  assert d_bf16.dtype == np.uint16
  d_fp32 = np.empty_like(d_bf16, dtype=np.float32)
  for i in range(len(d_bf16)):
    d_fp32[i] = struct.unpack('<f', struct.pack('<HH', 0, d_bf16[i]))[0]
  return d_fp32.reshape(s)

if len(sys.argv) < 8:
  print("Usage: %s f1.bin f2.bin int8|bf16|float32 N C H W [K] [tolerance]" % sys.argv[0])
  exit(-1)

f1 = sys.argv[1]
f2 = sys.argv[2]
t = sys.argv[3]
n = int(sys.argv[4])
c = int(sys.argv[5])
h = int(sys.argv[6])
w = int(sys.argv[7])
K = 0
if len(sys.argv) >= 9:
  K = int(sys.argv[8])
close_tolerance = 5
if len(sys.argv) == 10:
  close_tolerance = int(sys.argv[9])

if t == 'bf16':
  t = 'uint16'

if t != 'int8' and t != 'float32' and t!= 'uint16':
  print("Invalid type", t)
  exit(-1)

d1 = np.fromfile(f1, dtype=t)
d2 = np.fromfile(f2, dtype=t)
if n == 0 and c == 0 and h == 0:
  d1 = np.reshape(np.ravel(d1), (w))
  d2 = np.reshape(np.ravel(d2), (w))
elif n == 0 and c == 0:
  d1 = np.reshape(np.ravel(d1), (h, w))
  d2 = np.reshape(np.ravel(d2), (h, w))
elif n == 0:
  d1 = np.reshape(np.ravel(d1), (c, h, w))
  d2 = np.reshape(np.ravel(d2), (c, h, w))
else:
  d1 = np.reshape(np.ravel(d1), (n, c, h, w))
  d2 = np.reshape(np.ravel(d2), (n, c, h, w))
#np.set_printoptions(precision=4)
#np.set_printoptions(suppress=True)
#print(d1)
#print(d2)
#print('d1 shape', d1.shape)
#print('d2 shape', d2.shape)

if t == 'uint16':
  print("BF16 Show Top-{} in uint16".format(K))
  print(f1)
  for i in get_topk(d1, K):
    print(i)
  print(f2)
  for i in get_topk(d2, K):
    print(i)
  d1 = bf16_to_fp32(d1)
  d2 = bf16_to_fp32(d2)

if K > 0:
  print("Show Top-{}".format(K))
  print(f1)
  for i in get_topk(d1, K):
    print(i)
  print(f2)
  for i in get_topk(d2, K):
    print(i)

tc = TensorCompare(close_order_tol=close_tolerance)
result = tc.compare(d1, d2)
tc.print_result(d1, d2, "", result, 3)
print("[{}] [{}] {} [{}]".format(f1, f2, result[1],
      "PASSED" if result[0] else "FAILED"))
sys.exit(0 if result[0] else -1)
