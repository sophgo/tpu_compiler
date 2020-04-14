#!/usr/bin/env python3

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

if len(sys.argv) < 3:
  print("Usage: %s fp32.bin int8.bin [input_scale=1.0] [threshold]" % sys.argv[0])
  exit(-1)

f0 = sys.argv[1]
f1 = sys.argv[2]
scale = 1.0
if len(sys.argv) >= 4:
  scale = float(sys.argv[3])
threshold = 0.0
if len(sys.argv) >= 5:
  threshold = float(sys.argv[4])

d_fp32 = np.fromfile(f0, dtype=np.float32)

if scale != 1.0:
  d_fp32 = d_fp32 * scale

if threshold != 0.0:
  d_fp32 = d_fp32 * 128.0 / threshold

d_fp32 = np.floor(d_fp32 + 0.5)

# saturate
mask = d_fp32 > 127
d_fp32[mask] = 127
mask = d_fp32 < -128
d_fp32[mask] = -128

d_int8 = d_fp32.astype(np.int8)
d_int8.tofile(f1)
