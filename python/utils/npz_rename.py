#!/usr/bin/env python
import numpy as np
import sys

if len(sys.argv) != 4:
  print("Usage: {} in.npz name1 name2".format(sys.argv[0]))
  exit(-1)

npz_in = np.load(sys.argv[1])

npz_out = {}
d = npz_in[sys.argv[2]]
npz_out[sys.argv[3]] = d
np.savez(sys.argv[1], **npz_out)