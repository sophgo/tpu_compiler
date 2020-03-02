#!/usr/bin/env python

import numpy as np
import sys

if len(sys.argv) != 4:
  print("Usage: %s in.npz out.npz arr1,arr2,arr3" % sys.argv[0])
  exit(-1)

npz_in = np.load(sys.argv[1])
npz_out = {}
for s in sys.argv[3].split(','):
  print(s)
  d = npz_in[s]
  npz_out[s] = d
np.savez(sys.argv[2], **npz_out)
