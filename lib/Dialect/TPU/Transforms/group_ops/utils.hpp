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
#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
#include "mlir/Dialect/TPU/QuantizationArithmetic.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/Support/Debug.h>
#include "mlir/Dialect/TPU/MachineInfo.h"
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
#define BM_SUCCESS 0              // The operation was successful
#define BM_ERR_AGAIN 1            // Not ready yet
#define BM_ERR_FAILURE 2          // General failure
#define BM_ERR_TIMEOUT 3          // Timeout
#define BM_ERR_UNINITIALIZED 4    // Uninitialzed
#define BM_ERR_INVALID_ARGUMENT 5 // Arguments invalid
#define BM_ERR_NOMEM 6            // Not enough memory
#define BM_ERR_DATA 7             // Data error
#define BM_ERR_BUSY 8             // Busy
#define BM_ERR_NOT_SUPPORTED 9    // Not supported yet

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
          !isa<tpu::NoneOp>(op) && op->getName().getDialect().str() == "tpu");
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
  auto op_top = op->getResult(idx)->getDefiningOp();
  if (op_top && isValidTpuOp(op_top)) {
    auto name = mlir::getOpName(op_top);
    return name;
  } else if (auto load_op = dyn_cast<tpu::LoadWeightOp>(op_top))
    return load_op.name();
  else
    return llvm::StringRef();
}

static inline llvm::StringRef bottom_name(Operation *op, int idx) {
  auto op_bottom = op->getOperand(idx)->getDefiningOp();
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
      op->getOperand(idx)->getType().cast<TensorType>().getShape();
  return shape;
}

static inline std::vector<int64_t> output_shape(Operation *op, int idx) {
  std::vector<int64_t> shape =
      op->getResult(idx)->getType().cast<TensorType>().getShape();
  return shape;
}

void getConvParam(Operation *p, int &n, int &ic, int &ih, int &iw, int &oc,
                         int &oh, int &ow, int &g, int &kh, int &kw, int &sh, int &sw,
                         int &pt, int &pb, int &pl, int &pr, int &dh, int &dw,
                         bool &is_dw, bool &with_bias, bool &do_relu, bool &do_ic_align,
                         bool &fuse_leaky);

void getConcatParam(Operation *op, int &axis, bool &do_relu);

void getSliceParam(Operation *op, int &axis);

void getUpsampleParam(Operation *op, int &scale_h, int &scale_w);

void getPoolingParam(Operation *op, int &n, int &c, int &ih, int &iw, int &oh,
                            int &ow, int &kh, int &kw, int &sh, int &sw, int &pt, int &pb,
                            int &pl, int &pr, bool &is_global, bool &do_relu,
                            bool &count_include_pad);

void getEltwiseAddParam(Operation *op, bool &do_early_stride, int &h_stride,
                               int &w_stride);

void getEltwiseReluParam(Operation *op, bool &do_relu);
} // namespace mlir
#endif
