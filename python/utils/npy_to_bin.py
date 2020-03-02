#!/usr/bin/env python

import numpy as np
import sys

if len(sys.argv) != 3:
  print("Usage: %s filename.npy output.bin" % sys.argv[0])
  exit(-1)

f_npy = sys.argv[1]
f_bin = sys.argv[2]

d = np.load(f_npy)
print('shape', d.shape)
print('dtype', d.dtype)
d.tofile(f_bin)
