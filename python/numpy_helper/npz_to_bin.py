#!/usr/bin/env python

import numpy as np
import sys

if not 4 <= len(sys.argv) <= 5:
  print("Usage: %s filename.npz array_name filename.bin [int8|bf16|float32]" % sys.argv[0])
  exit(-1)

npzfile = np.load(sys.argv[1])
d = npzfile[sys.argv[2]]
print('shape', d.shape)
print('dtype', d.dtype)
if len(sys.argv) == 5:
  dtype = sys.argv[4]
  if dtype == "int8":
    d = d.astype(np.int8)
  elif dtype == "bf16" or dtype == "uint16":
    d = d.astype(np.uint16)
  elif dtype == "float32":
    d = d.astype(np.float32)
  else:
    print("{}: Invalid dtype {}".format(sys.argv[0], dtype))
    exit(-1)
d.tofile(sys.argv[3])
