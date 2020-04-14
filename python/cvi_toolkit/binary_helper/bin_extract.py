#!/usr/bin/env python3

import numpy as np
import sys

if not (len(sys.argv) == 6):
  print("Usage: %s from.bin to.bin int8|bf16|float32 offset size" % sys.argv[0])
  exit(-1)

f1 = sys.argv[1]
f2 = sys.argv[2]
t = sys.argv[3]
offset = int(sys.argv[4], 16)
count = int(sys.argv[5])

if t == 'bf16':
  t = 'uint16'

if t != 'int8' and t != 'float32' and t!='uint16':
  print("Invalid type", t)
  exit(-1)

d1 = np.fromfile(f1, dtype=t)
offset = offset / d1.itemsize
d2 = d1[offset:offset+count]
d2.tofile(f2)
