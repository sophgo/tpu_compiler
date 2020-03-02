#!/usr/bin/env python

import numpy as np
import sys
import struct

def bf16_to_fp32(bf16_value):
    return struct.unpack('<f', struct.pack('<HH', 0, bf16_value))[0]


if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage: {} in.npz out.npy".format(sys.argv[0]))
    exit(-1)



  npz_in = np.load(sys.argv[1])
  npz_out = {}
  for s in npz_in:
    bf16_arr = npz_in[s]
    fp32_arr = np.empty_like(bf16_arr, dtype=np.float32)
    for x, y in np.nditer([bf16_arr, fp32_arr], op_flags=['readwrite']):
      y[...] = bf16_to_fp32(x)
  
    npz_out[s] = fp32_arr
    
  np.savez(sys.argv[2], **npz_out)


