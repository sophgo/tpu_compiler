#!/usr/bin/env python3

import numpy as np
import sys
import csv

if not (len(sys.argv) == 4):
  print("Usage: %s from.bin map.csv to.npz" % sys.argv[0])
  exit(-1)

f_bin = sys.argv[1]
f_map = sys.argv[2]
f_npz = sys.argv[3]

d1_int8 = np.fromfile(f_bin, dtype='int8')
d1_bf16 = np.fromfile(f_bin, dtype='uint16')
d1_fp32 = np.fromfile(f_bin, dtype='float32')
npz  = {}
with open(f_map, mode='r') as mapfile:
  reader = csv.reader(mapfile)
  # mapdict = (rows[0]:int(rows[1],0) for rows in reader}
  # mapdict = dict((rows[0],int(rows[1],0)) for rows in reader)
  for rows in reader:
    name = rows[0]
    offset = int(rows[1],0)
    t = rows[2]
    shape = (int(rows[3]),  int(rows[4]), int(rows[5]), int(rows[6]))
    count = np.prod(np.array(shape))
    print(name, t, offset, shape, count)
    if t == 'int8':
      npz[name] = d1_int8[offset:offset+count].reshape(shape)
    elif t == 'uint16':
      offset = offset / d1_bf16.itemsize
      npz[name] = d1_bf16[offset:offset+count].reshape(shape)
    elif t == 'float32':
      offset = offset / d1_fp32.itemsize
      npz[name] = d1_fp32[offset:offset+count].reshape(shape)
    else:
      print("Invalid type", t)
      exit(-1)
  np.savez(f_npz, **npz)

