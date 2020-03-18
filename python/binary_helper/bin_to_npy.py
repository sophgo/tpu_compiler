#!/usr/bin/env python

import numpy as np
import sys

if len(sys.argv) != 8:
  print("Usage: %s filename.bin int8|float32 N C H W output.npy" % sys.argv[0])
  exit(-1)

f = sys.argv[1]
t = sys.argv[2]
n = int(sys.argv[3])
c = int(sys.argv[4])
h = int(sys.argv[5])
w = int(sys.argv[6])
f_npy = sys.argv[7]

if t != 'int8' and t != 'float32' and t!= 'uint8':
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
print('shape', d.shape)

np.save(f_npy, d)
