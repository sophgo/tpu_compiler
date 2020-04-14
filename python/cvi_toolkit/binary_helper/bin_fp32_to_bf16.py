#!/usr/bin/env python3

import numpy as np
import sys
import struct

if len(sys.argv) < 3:
  print("Usage: %s fp32.bin bf16.bin [input_scale=1.0]" % sys.argv[0])
  exit(-1)

f0 = sys.argv[1]
f1 = sys.argv[2]

scale=1.0
if len(sys.argv) == 4:
    scale = float(sys.argv[3])

d_fp32 = np.fromfile(f0, dtype=np.float32)

if scale != 1.0:
  d_fp32 = d_fp32 * scale

# rounding
d_fp32 = d_fp32 * 1.001957

d_bf16 = np.empty_like(d_fp32, dtype=np.uint16)

for i in range(len(d_fp32)):
  d_bf16[i] = struct.unpack('<HH', struct.pack('<f', d_fp32[i]))[1]
  # print(i, d_fp32[i], d_bf16[i])
  # print("fp32, ", struct.unpack('<f', struct.pack('<HH', 0, d_bf16[i]))[0])

d_bf16.tofile(f1)
