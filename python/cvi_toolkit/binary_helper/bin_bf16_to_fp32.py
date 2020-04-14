#!/usr/bin/env python3

import numpy as np
import sys
import struct

if len(sys.argv) != 3:
  print("Usage: %s bf16.bin fp32.bin" % sys.argv[0])
  exit(-1)

f0 = sys.argv[1]
f1 = sys.argv[2]

d_bf16 = np.fromfile(f0, dtype=np.uint16)
d_fp32 = np.empty_like(d_bf16, dtype=np.float32)

for i in range(len(d_bf16)):
  d_fp32[i] = struct.unpack('<f', struct.pack('<HH', 0, d_bf16[i]))[0]
  # print(i, d_bf16[i], d_fp32[i])
  # print("bf16, ", struct.unpack('<HH', struct.pack('<f', d_fp32[i]))[1])

d_fp32.tofile(f1)
