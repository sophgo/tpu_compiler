#!/usr/bin/env python3

from __future__ import division
import numpy as np
import os
import sys
import argparse
import struct
import csv
from math import fabs
from enum import IntEnum
from .tensor_compare import TensorCompare, TensorCompareStats
import threading
import multiprocessing

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def parse_args(args_list):
    parser = argparse.ArgumentParser(description='Compare two npz tensor files.')
    parser.add_argument("target_file",
                        help="Comparing target file")
    parser.add_argument("ref_file",
                        help="Comparing reference file")
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument('--discard', '-d', action='count', default=0)
    parser.add_argument('--dtype', type=str,
                        help="force dtype")
    parser.add_argument("--tolerance", type=str, default='0.99,0.99,0.90,50',
                        help="tolerance for cos/cor/euclid similarity/SQNR")
    parser.add_argument('--op_info', type=str,
                        help="A csv file op_info, including order and dequant threshold")
    parser.add_argument("--dequant", action='store_true', default=False,
                        help="Do dequantization flag, use threshold table provided in --op_info")
    parser.add_argument('--order', type=str,
                        help="A csv file containing order of the tensors, used when --op_info is not present")
    parser.add_argument("--tensor", type=str,
                        help="Compare one specific tensor by name")
    parser.add_argument("--excepts", type=str,
                        help="List of tensors except from comparing")
    parser.add_argument("--full-array", action='store_true', default=False,
                        help="Dump full array data when comparing failed")
    parser.add_argument("--stats_int8_tensor", action='store_true', default=False,
                        help="Do statistics on int8 tensor for saturate ratio and low ratio")
    parser.add_argument("--save", type=str,
                        help="Save result as a csv file")
    args = parser.parse_args(args_list)
    return args

def bf16_to_fp32(d_bf16):
  s = d_bf16.shape
  d_bf16 = d_bf16.flatten()
  assert d_bf16.dtype == np.uint16
  d_fp32 = np.empty_like(d_bf16, dtype=np.float32)
  for i in range(len(d_bf16)):
    d_fp32[i] = struct.unpack('<f', struct.pack('<HH', 0, d_bf16[i]))[0]
  return d_fp32.reshape(s)

def align_type_and_shape(d1, d2, force_dtype=np.void):
  try:
    d2 = d2.reshape(d1.shape)
  except:
    print("WARRING: Two narraies are not the same shape." + \
          " {} v.s. {}, check if continue:".format(d1.shape, d2.shape))
    # check if do-early-stride case
    if d2.shape[2] % d1.shape[2] == 0 and \
       d2.shape[3] % d1.shape[3] == 0:
      sh = int(d2.shape[2] / d1.shape[2])
      sw = int(d2.shape[3] / d1.shape[3])
      d2 = d2[:, :, ::sh, ::sw]
      print("Ignore this warning, continue")
    else:
      raise ValueError("Fatal, stop")

  t1 = d1.dtype
  t2 = d2.dtype
  # print(t1, d1.size, d1.shape)
  # print(t2, d2.size, d2.shape)
  if force_dtype != np.void:
    t = force_dtype
  else:
    if t1 == np.int8 or t2 == np.int8:
      t = np.int8
    else:
      t = np.float32
    if t1 == np.uint16:
      d1 = bf16_to_fp32(d1)
    if t2 == np.uint16:
      d2 = bf16_to_fp32(d2)
  d1 = d1.astype(t)
  d2 = d2.astype(t)
  return d1, d2

def load_op_info(op_info_file):
  ordered_names = []
  thresholds = {}
  with open(op_info_file, mode='r') as mapfile:
    print("Using op_info file %s"%(op_info_file))
    reader = csv.reader(mapfile)
    for rows in reader:
      ordered_names.append(rows[0])
      if (rows[2][:4] == "INT8"):
        thresholds[rows[0]] = float(rows[3])
      else:
        thresholds[rows[0]] = 0.0
  return ordered_names, thresholds

def load_order(order_file):
  ordered_names = []
  with open(order_file, mode='r') as mapfile:
    print("Using order file %s"%(order_file))
    reader = csv.reader(mapfile)
    for rows in reader:
      ordered_names.append(rows[0])
  return ordered_names

def dequantize(d1, threshold):
  # print("Apply dequantization with threshold {}".format(threshold))
  d1 = d1 * threshold / 127.0
  return d1

def discard_res_data(args):
  res_data = np.load(args.target_file)
  ref_data = np.load(args.ref_file)
  name = res_data.files
  length = len(name)
  trunc_data = {}
  for i in range(length-1):
    trunc_data[name[i]] = res_data[name[i]]

  box = ref_data[name[-1]].shape[2]
  trunc_data[name[-1]] = res_data[name[-1]][:,:,0:box,:]
  np.savez(args.target_file, **trunc_data)

def tensor_stats(d):
  stats = {}
  d_int8 = d.astype(np.int8)
  b_sat_pos = d_int8 == 127
  b_sat_neg = d_int8 == -128
  b_low = np.absolute(d_int8) < 8 # 16, 32, 63
  stats["sat_ratio_pos"] = len(d_int8[b_sat_pos]) / d_int8.size
  stats["sat_ratio_neg"] = len(d_int8[b_sat_neg]) / d_int8.size
  stats["low_ratio"]     = len(d_int8[b_low])     / d_int8.size
  print("    sat_ratio_pos = {:.4f}   [{}/{}]".format(stats["sat_ratio_pos"], len(d_int8[b_sat_pos]), d_int8.size))
  print("    sat_ratio_neg = {:.4f}   [{}/{}]".format(stats["sat_ratio_neg"], len(d_int8[b_sat_neg]), d_int8.size))
  print("    low_ratio     = {:.4f}   [{}/{}]".format(stats["low_ratio"], len(d_int8[b_low]), d_int8.size))

def compare_one_array(tc, npz1, npz2, name, force_dtype, thresholds, verbose, lock, dic):
  lock.acquire()
  d1 = npz1[name]
  d2 = npz2[name]
  lock.release()
  if name in thresholds and not thresholds[name] == 0.0:
    # print("Apply dequantization with threhold {}".format(thresholds[name]))
    d1 = dequantize(d1, thresholds[name])
  try:
    d1, d2 = align_type_and_shape(d1, d2, force_dtype=force_dtype)
  except:
    raise ValueError("{} in two npz file is not same shape. {} v.s. {}".format(name, d1.shape, d2.shape))
  result = tc.compare(d1, d2)
  # tc.print_result(d1, d2, name, result, verbose)
  dic[name] = result
  return result

def print_result_one_array(tc, npz1, npz2, name, force_dtype, thresholds, verbose, lock, dic):
  lock.acquire()
  d1 = npz1[name]
  d2 = npz2[name]
  lock.release()
  if name in thresholds and not thresholds[name] == 0.0:
    print("Apply dequantization with threhold {}".format(thresholds[name]))
    d1 = dequantize(d1, thresholds[name])
  d1, d2 = align_type_and_shape(d1, d2, force_dtype=force_dtype)
  tc.print_result(d1, d2, name, dic[name], verbose)

def npz_compare(args_list):
  lock = multiprocessing.Lock()
  dic = multiprocessing.Manager().dict()
  args = parse_args(args_list)
  f1 = args.target_file
  f2 = args.ref_file
  if args.discard:
    discard_res_data(args)

  np.set_printoptions(precision=6)
  np.set_printoptions(suppress=True)
  if args.full_array:
    np.set_printoptions(threshold=sys.maxsize)
  force_dtype=np.void
  if args.dtype:
    force_dtype=args.dtype
  if args.tolerance:
    tolerance = [float(s) for s in args.tolerance.split(',')]
  excepts = []
  if args.excepts:
    excepts = [str(s) for s in args.excepts.split(',')]

  ordered_names = []
  thresholds = {}
  if args.op_info:
    if args.dequant:
      ordered_names, thresholds = load_op_info(args.op_info)
    else:
      ordered_names, _ = load_op_info(args.op_info)
  else:
    if args.order:
      ordered_names = load_order(args.order)
    if args.dequant:
      print("op_info is needed for dequantization")
      sys.exit(-1)

  npz1 = np.load(f1)
  npz2 = np.load(f2)

  # Add default similarity threshold to zero
  # Elder regression script does not have SQNR
  while (len(tolerance) < 4):
    tolerance.append(float('-inf'))

  tc = TensorCompare(close_order_tol=3,
                     cosine_similarity_tol = tolerance[0],
                     correlation_similarity_tol = tolerance[1],
                     euclidean_similarity_tol = tolerance[2],
                     signal_to_quantization_noise_tol = tolerance[3])

  if args.tensor:
    print("Comparing %s ..."%(args.tensor))
    name = args.tensor
    result = compare_one_array(tc, npz1, npz2, name, force_dtype,
                               thresholds, args.verbose, lock, dic)
    print_result_one_array(tc, npz1, npz2, name, force_dtype,
                           thresholds, args.verbose, lock, dic)
    sys.exit(0 if result[0] else -1)

  common = set(npz1.files) & set(npz2.files)
  npz1_s = set(npz1.files) - common
  npz2_s = set(npz2.files) - common
  if excepts:
    common = common - set(excepts)
  if ordered_names:
    names = []
    for name in ordered_names:
        if name in common:
          names.append(name)
  else:
    names = common

  stats = TensorCompareStats()

  names_list = list(names)  # deep copy
  process_number = multiprocessing.cpu_count()

  while(len(names_list) > 0):
      # take process number names
      # take name which will do compare
      compare_process_name_list = names_list[:process_number]
      names_list = names_list[process_number:]  # remove done name
      processes = []
      for name in compare_process_name_list:
          p = multiprocessing.Process(target=compare_one_array,
                                      args=(tc, npz1, npz2, name, force_dtype, thresholds,
                                            args.verbose, lock, dic))
          processes.append(p)
          p.start()

      for j in processes:
          j.join()

  for name in names:
    stats.update(name, dic.get(name))
    print_result_one_array(tc, npz1, npz2, name, force_dtype,
                           thresholds, args.verbose, lock, dic)
    if args.stats_int8_tensor:
      d1 = npz1[name]
      tensor_stats(d1)

  stats.print_result()
  if (args.save):
    stats.save_result(args.save)
    print("Results save as {}".format(args.save))
  print("Target    {}".format(f1))
  print("Reference {}".format(f2))
  if (stats.failed == 0):
    print("npz compare PASSED.", flush=True)
    return stats
  else:
    print("npz compare FAILED.", flush=True)
    sys.exit(-1)

if __name__ == '__main__':
    npz_compare(sys.argv)
