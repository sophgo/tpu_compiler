from .npz_compare import npz_compare
from .npz_dump import npz_dump
import numpy as np
import sys
import struct

def bf16_to_fp32(bf16_value):
    return struct.unpack('<f', struct.pack('<HH', 0, bf16_value))[0]


def npz_rename(args):
    if len(args) < 2:
        print("Usage: {} rename in.npz name1 name2".format(sys.argv[0]))
        exit(-1)
    npz_in = np.load(args[0])
    npz_out = {}
    d = npz_in[args[1]]
    npz_out[args[2]] = d
    np.savez(args[1], **npz_out)

def npz_extract(args):
    if len(args) < 3:
        print("Usage: {} extract in.npz out.npz arr1,arr2,arr3".format(sys.argv[0]))
        exit(-1)
    npz_in = np.load(args[0])
    npz_out = {}
    for s in args[2].split(','):
        print(s)
        d = npz_in[s]
        npz_out[s] = d
    np.savez(args[1], **npz_out)

def npz_to_bin(args):
    if len(args) < 3:
        print("Usage: {} filename.npz array_name filename.bin [int8|bf16|float32]".format(sys.argv[0]))
        exit(-1)
    npzfile = np.load(args[0])
    d = npzfile[args[1]]
    print('shape', d.shape)
    print('dtype', d.dtype)
    if len(args) == 4:
        dtype = args[3]
        if dtype == "int8":
            d = d.astype(np.int8)
        elif dtype == "bf16" or dtype == "uint16":
            d = d.astype(np.uint16)
        elif dtype == "float32":
            d = d.astype(np.float32)
        else:
            print("{}: Invalid dtype {}".format(sys.argv[0], dtype))
            exit(-1)
    d.tofile(args[2])

def npz_bf16_to_fp32(args):
    if len(args) < 2:
        print("Usage: {} bf16_to_fp32 in.npz out.npy".format(sys.argv[0]))
        exit(-1)
    npz_in = np.load(args[0])
    npz_out = {}
    for s in npz_in:
        bf16_arr = npz_in[s]
        fp32_arr = np.empty_like(bf16_arr, dtype=np.float32)
    for x, y in np.nditer([bf16_arr, fp32_arr], op_flags=['readwrite']):
      if s.dtype == np.float32:
        y[...] = x
      else:
        y[...] = bf16_to_fp32(x)

    npz_out[s] = fp32_arr

    np.savez(args[1], **npz_out)