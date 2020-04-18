#!/usr/bin/env python3

import numpy as np
import sys
import struct
# from math import fabs
from enum import IntEnum
from scipy import spatial
from math import *
from collections import OrderedDict

def second(elem):
  return elem[1]

def get_topk(a, k):
  if (a.size < k):
    return []
  idx = np.argpartition(-a.ravel(),k)[:k]
  # return np.column_stack(np.unravel_index(idx, a.shape))
  topk = list(zip(idx, np.take(a, idx)))
  #return topk
  topk.sort(key=second, reverse=True)
  return topk

class TensorCompare():
  NOT_MATCH   = "NOT_MATCH"
  EQUAL       = "EQUAL"
  NOT_EQUAL   = "NOT_EQUAL"
  CLOSE       = "CLOSE"
  SIMILAR     = "SIMILAR"
  NOT_SIMILAR = "NOT_SIMLIAR"

  def __init__(self, close_order_tol=3,
               cosine_similarity_tol = 0.99,
               correlation_similarity_tol = 0.99,
               euclidean_similarity_tol = 0.90):
    self.close_order_tol            = close_order_tol
    self.cosine_similarity_tol      = cosine_similarity_tol
    self.correlation_similarity_tol = correlation_similarity_tol
    self.euclidean_similarity_tol   = euclidean_similarity_tol
    return

  def square_rooted(self, x):
    return round(sqrt(sum([a*a for a in x])),3)

  def cosine_similarity(self, x, y):
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return round(numerator/float(denominator),3)

  def euclidean_distance(self, x, y):
    return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))

  def show_diff(self, d1, d2):
    d1f = d1.flatten()
    d2f = d2.flatten()
    print("show_diff, total len = " + str(len(d1f)))
    if d1f.dtype == np.int8:
      assert(d2f.dtype == np.int8)
      for i in range(len(d1f)):
        if (d1f[i] != d2f[i]):
          print(i, d1f[i], d2f[i])
    else:
      atol = 10**(-self.close_order_tol)
      rtol = 10**(-self.close_order_tol)
      for i in range(len(d1f)):
        if fabs(d1f[i] - d2f[i]) > (atol + rtol * fabs(d2f[i])):
          print(i, d1f[i], d2f[i])

  def compare(self, d1, d2):
    details = {}
    if (len(d1) != len(d2)):
      return (False, self.NOT_MATCH, details)

    if np.array_equal(d1, d2):
      return (True, self.EQUAL, details)

    # int8 only check equal, not close
    if d1.dtype == np.int8:
      return (False, self.NOT_EQUAL, details)

    # check allclose
    for order in range(10, 1, -1):
      if (np.allclose(d1, d2, rtol=1 * 10**(-order), atol=1e-8, equal_nan=True)):
        break
    if order >= self.close_order_tol:
      details["close_order"] = order
      return (True, self.CLOSE, details)

    # check similarity
    # cosine similarity
    # cosine_similarity_my = self.cosine_similarity(d1.flatten(), d2.flatten())
    # print("Cosine Similarity    (my): ", cosine_similarity_my)
    cosine_similarity = 1 - spatial.distance.cosine(d1.flatten(), d2.flatten())
    # print("Cosine Similarity    (sp): ", cosine_similarity)
    # correlation similarity
    correlation_similarity = 1 - spatial.distance.correlation(d1.flatten(), d2.flatten())
    #print("Correlation Similarity   : ", correlation_similarity)
    # measure euclidean similarity
    m = (d1+d2)/2
    euclidean_similarity = 1 - (self.euclidean_distance(d1.flatten(), d2.flatten()) /
                                self.square_rooted(m.flatten()))
    # print("Euclidean Similarity (my): ", euclidean_simliarity_my)

    details["cosine_similarity"]       = cosine_similarity
    details["correlation_similarity"]  = correlation_similarity
    details["euclidean_similarity"]    = euclidean_similarity
    # check similarity
    if (cosine_similarity > self.cosine_similarity_tol
        and correlation_similarity > self.correlation_similarity_tol
        and euclidean_similarity > self.euclidean_similarity_tol):
      return (True, self.SIMILAR, details)
    else:
      # Not similar
      return (False, self.NOT_SIMILAR, details)

  def print_result(self, d1, d2, name, result, verbose):
    print("[{:<32}] {:>12} [{:>6}]".format(name, result[1],
           "PASSED" if result[0] else "FAILED"))
    if (verbose > 0):
      print("    {} {} ".format(d1.shape, d1.dtype))
      if (result[1] == self.CLOSE):
        print("    close order            = {}".format(result[2]["close_order"]))
      if (result[1] == self.SIMILAR or result[1] == self.NOT_SIMILAR):
        print("    cosine_similarity      = {:.6f}".format(result[2]["cosine_similarity"]))
        print("    correlation_similarity = {:.6f}".format(result[2]["correlation_similarity"]))
        print("    euclidean_similarity   = {:.6f}".format(result[2]["euclidean_similarity"]))
    if (verbose > 1 and not result[0]):
      K = 5
      print("Target")
      for i in get_topk(d1, K):
        print(i)
      print("Reference")
      for i in get_topk(d2, K):
        print(i)
    if (verbose > 2 and not result[0]):
      self.show_diff(d1,d2)
    if (verbose > 3 and not result[0]):
      print("Target")
      print(d1)
      print("Reference")
      print(d2)

class TensorCompareStats():
  def __init__(self):
    self.passed = 0
    self.failed = 0
    self.results = OrderedDict()
    self.count = {}
    self.count[TensorCompare.NOT_MATCH] = 0
    self.count[TensorCompare.EQUAL] = 0
    self.count[TensorCompare.NOT_EQUAL] = 0
    self.count[TensorCompare.CLOSE] = 0
    self.count[TensorCompare.SIMILAR] = 0
    self.count[TensorCompare.NOT_SIMILAR] = 0
    self.min_cosine_similarity = 1.0
    self.min_correlation_similarity = 1.0
    self.min_euclidean_similarity = 1.0

  def update(self, name, result):
    self.results[name] = result
    if result[0]:
      self.passed = self.passed + 1
      assert (result[1] == TensorCompare.EQUAL
              or result[1] == TensorCompare.CLOSE
              or result[1] == TensorCompare.SIMILAR)
    else:
      self.failed = self.failed + 1
      assert (result[1] == TensorCompare.NOT_EQUAL
              or result[1] == TensorCompare.NOT_SIMILAR)
    self.count[result[1]] = self.count[result[1]] + 1
    # record min similarity
    if result[1] == TensorCompare.SIMILAR or result[1] == TensorCompare.NOT_SIMILAR:
      self.min_cosine_similarity = min(self.min_cosine_similarity, result[2]["cosine_similarity"])
      self.min_correlation_similarity = min(self.min_correlation_similarity, result[2]["correlation_similarity"])
      self.min_euclidean_similarity = min(self.min_euclidean_similarity, result[2]["euclidean_similarity"])

  def print_result(self):
    print("%d compared"%(len(self.results)))
    print("%d passed"%(self.passed))
    print("  %d equal, %d close, %d similar"
          %(self.count[TensorCompare.EQUAL],
            self.count[TensorCompare.CLOSE],
            self.count[TensorCompare.SIMILAR]))
    print("%d failed"%(self.failed))
    print("  %d not equal, %d not similar"
          %(self.count[TensorCompare.NOT_EQUAL],
            self.count[TensorCompare.NOT_SIMILAR]))
    print("min_similiarity = ({}, {}, {})".format(
            self.min_cosine_similarity,
            self.min_correlation_similarity,
            self.min_euclidean_similarity))

  def save_result(self, csv_file):
    has_similarity = lambda x: (x == TensorCompare.SIMILAR
                                or x == TensorCompare.NOT_SIMILAR)
    with open(csv_file, mode='w') as f:
      f.write("name, equal, close, close_order, similar, "
              "sim_cosn, sim_corr, sim_eucl, pass\n")
      for name, result in self.results.items():
        f.write("{}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(
          name,
          bool(result[1] == TensorCompare.EQUAL),
          bool(result[1] == TensorCompare.CLOSE),
          result[2]["close_order"] if (result[1] == TensorCompare.CLOSE) else "na.",
          bool(result[1] == TensorCompare.SIMILAR),
          result[2]["cosine_similarity"] if has_similarity(result[1]) else "na.",
          result[2]["correlation_similarity"] if has_similarity(result[1])  else "na.",
          result[2]["euclidean_similarity"] if has_similarity(result[1])  else "na.",
          bool(result[0]),
        ))
