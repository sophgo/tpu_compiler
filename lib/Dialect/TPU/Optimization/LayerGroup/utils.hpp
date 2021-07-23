#ifndef CLUSTER_UTILS_H
#define CLUSTER_UTILS_H

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Passes.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/QuantizationArithmetic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tpuc/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/Support/Debug.h>
#include "tpuc/MachineInfo.h"
#include <math.h>

namespace mlir {

#define gaddr_t uint64_t
#define laddr_t uint32_t

#define NPU_NUM (MInfo::lane_num)
#define EU_NUM (MInfo::eu_num)
#define LOCAL_MEM_SIZE (MInfo::lmem_per_lane)
#define LOCAL_BANK_SIZE (MInfo::lmem_per_lane / MInfo::lmem_bank_num)

static inline int ceiling_func(int numerator, int denominator) {
  return (numerator + denominator - 1) / denominator;
}

#define __ALIGN_MASK(x, mask) (((x) + (mask)) & ~(mask))
#define ALIGN(x, a) __ALIGN_MASK(x, (__typeof__(x))(a)-1)

#define bmerr_t int
#define BM_SUCCESS 0
#define BM_ERR_FAILURE 1
#define BM_ERR_NOT_SUPPORTED 2

#define LOG_TAB_L0 ""
#define LOG_TAB_L1 "  "
#define LOG_TAB_L2 "    "
#define LOG_TAB_L3 "      "
#define LOG_TAB_L4 "        "
#define LOG_TAB_L5 "          "

#define SMALL_TDMA_SIZE (10*1024*1024)
// layer group strategy
typedef enum {
  LG_FIT_SLICE_METHOD = 0,
  LG_MAX_SLICE_METHOD = 1,
  LG_MAX_H_SLICE
}LG_Slice_Limit;

typedef enum {
  LG_Slice_Dim_H = 0,
  LG_Slice_Dim_W = 1
}LG_Slice_Dim;

typedef enum {
  NEURON = 0,
  COEFF = 1,
  COEFF_INT8 = 2,
  BIAS_INT8 = 3
} TransportDataType;

static inline void printFunction(FuncOp *fn) {
  std::string res;
  llvm::raw_string_ostream os(res);
  fn->walk([&](Operation *op) {
    op->print(os);
    os << "\n";
  });
  llvm::errs() << res;
}

static inline bool isValidTpuOp(Operation *op) {
  return (!isa<tpu::LoadWeightOp>(op) && !isa<tpu::WeightFileOp>(op) &&
          !isa<tpu::NoneOp>(op) && op->getName().getDialect()->getNamespace() == "tpu");
}

static inline bool isValidLayerGroupOp(Operation *op) {
  if (isa<tpu::LoadWeightOp>(op) || isa<tpu::WeightFileOp>(op) || isa<tpu::QuantOp>(op) ||
      isa<ReturnOp>(op) || isa<tpu::NoneOp>(op)) {
    return false;
  } else {
    return true;
  }
}

static inline int64_t top_size(Operation *op) {
  return op->getNumResults();
}

static inline int64_t bottom_size(Operation *op) {
  return op->getNumOperands();
}

static inline llvm::StringRef top_name(Operation *op, int idx) {
  auto op_top = op->getResult(idx).getDefiningOp();
  if (op_top && isValidTpuOp(op_top)) {
    auto name = mlir::getOpName(op_top);
    return name;
  } else if (auto load_op = dyn_cast<tpu::LoadWeightOp>(op_top))
    return load_op.name();
  else
    return llvm::StringRef();
}

static inline llvm::StringRef bottom_name(Operation *op, int idx) {
  auto op_bottom = op->getOperand(idx).getDefiningOp();
  if (op_bottom && isValidTpuOp(op_bottom)) {
    auto name = mlir::getOpName(op_bottom);
    return name;
  } else if (auto load_op = dyn_cast<tpu::LoadWeightOp>(op_bottom))
    return load_op.name();
  else
    return llvm::StringRef();
}

static inline llvm::StringRef name(Operation *op) {
  auto op_name = mlir::getOpName(op);
  return op_name;
}

static inline std::vector<int64_t> input_shape(Operation *op, int idx) {
  std::vector<int64_t> shape =
      op->getOperand(idx).getType().cast<TensorType>().getShape();
  return shape;
}

static inline std::vector<int64_t> output_shape(Operation *op, int idx) {
  std::vector<int64_t> shape =
      op->getResult(idx).getType().cast<TensorType>().getShape();
  return shape;
}

void getConvParam(Operation *p, int &n, int &ic, int &ih, int &iw, int &oc,
                  int &oh, int &ow, int &g, int &kh, int &kw, int &ins_h,
                  int &ins_w, int &sh, int &sw, int &pt, int &pb, int &pl,
                  int &pr, int &dh, int &dw, bool &is_dw, bool &with_bias,
                  bool &do_relu, bool &do_ic_align, bool &fuse_leaky,
                  int &pad_value);

void getConcatParam(Operation *op, int &axis, bool &do_relu);

void getSliceParam(Operation *op, int &axis);

void getUpsampleParam(Operation *op, int &scale_h, int &scale_w);

void getPoolingParam(Operation *op, int &n, int &c, int &ih, int &iw, int &oh,
                            int &ow, int &kh, int &kw, int &sh, int &sw, int &pt, int &pb,
                            int &pl, int &pr, int &pad_value, bool &is_global, bool &do_relu,
                            bool &count_include_pad);

void getEltwiseAddParam(Operation *op, bool &do_early_stride, int &h_stride,
                               int &w_stride);

void getEltwiseReluParam(Operation * op, bool &do_relu);

void getLrnParam(Operation * op, uint32_t &local_size,
                  int &sum_rshift, int &lrn_rshift,
                  int &quant_data0, int &quant_data1,
                  float &alpha, float &k);

void getLayerNormParam(Operation *op, std::vector<int64_t> &input_shape,
                       std::vector<int> &normalized_shape, int &axis);
}
#endif
