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
#define LOCAL_BANK_SIZE (MInfo::lmem_per_lane/MInfo::lmem_bank_num)

static inline int ceiling_func(int numerator, int denominator)
{
  return (numerator + denominator - 1) / denominator;
}

#define __ALIGN_MASK(x,mask)    (((x)+(mask))&~(mask))
#define ALIGN(x,a)              __ALIGN_MASK(x,(__typeof__(x))(a)-1)

#define bmerr_t int
#define BM_SUCCESS 0               // The operation was successful
#define BM_ERR_AGAIN 1             // Not ready yet
#define BM_ERR_FAILURE 2           // General failure
#define BM_ERR_TIMEOUT 3           // Timeout
#define BM_ERR_UNINITIALIZED 4     // Uninitialzed
#define BM_ERR_INVALID_ARGUMENT 5  // Arguments invalid
#define BM_ERR_NOMEM 6             // Not enough memory
#define BM_ERR_DATA 7              // Data error
#define BM_ERR_BUSY 8              // Busy
#define BM_ERR_NOT_SUPPORTED 9     // Not supported yet


typedef enum  {
  NEURON = 0,
  COEFF = 1,
  COEFF_INT8 = 2,
  BIAS_INT8 = 3
}TransportDataType;

static inline void printFunction(FuncOp * fn) {
  std::string res;
  llvm::raw_string_ostream os(res);
  fn->walk([&](Operation * op) {
    op->print(os);
    os << "\n";
  });
  llvm::errs() << res;
}

static inline bool isValidTpuOp(Operation *op)
{
  return (!isa<tpu::LoadWeightOp>(op) && !isa<tpu::WeightFileOp>(op) &&
          !isa<tpu::NoneOp>(op) &&
          op->getName().getDialect().str() == "tpu");
}

static inline bool isValidLayerGroupOp(Operation *op) {
  if (isa<tpu::LoadWeightOp>(op)
          || isa<tpu::WeightFileOp>(op)
          || isa<tpu::QuantOp>(op)
          || isa<ReturnOp>(op)
          || isa<tpu::NoneOp>(op)) {
            return false;
  } else {
    return true;
  }
}

static inline int64_t top_size(Operation * op) {
  return op->getNumResults();
}

static inline int64_t bottom_size(Operation * op) {
  return op->getNumOperands();
}

static inline llvm::StringRef top_name(Operation * op, int idx) {
  auto op_top = op->getResult(idx)->getDefiningOp();
  if (op_top && isValidTpuOp(op_top)) {
    auto name = mlir::getOpName(op_top);
    return name;
  }
  else if (auto load_op = dyn_cast<tpu::LoadWeightOp>(op_top))
    return load_op.name();
  else
    return llvm::StringRef();
}

static inline llvm::StringRef bottom_name(Operation * op, int idx) {
  auto op_bottom = op->getOperand(idx)->getDefiningOp();
  if (op_bottom && isValidTpuOp(op_bottom)) {
    auto name = mlir::getOpName(op_bottom);
    return name;
  }
  else if (auto load_op = dyn_cast<tpu::LoadWeightOp>(op_bottom))
    return load_op.name();
  else
    return llvm::StringRef();

}

static inline llvm::StringRef name(Operation * op) {
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


static void getConvParam( Operation *p,
                          int &n, int &ic, int &ih, int &iw,
                          int &oc, int &oh, int &ow, int &g,
                          int &kh, int &kw,
                          int &sh, int &sw, int &pt, int &pb,
                          int &pl, int &pr, int &dh, int &dw,
                          bool &is_dw, bool &with_bias,
                          bool &do_relu,
                          bool &do_ic_align,
                          bool &fuse_leaky) {
  if (auto op = dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(p)) {
    bool is_deconv = false;
    parseConvParam(op.param(), is_deconv, op.input(), op.output(), op.filter(),
                    n, ic, ih, iw, oc, oh, ow, g,
                    kh, kw, sh, sw, pt, pb, pl, pr, dh, dw, is_dw, with_bias, do_relu);
    do_ic_align = op.do_ic_alignment().hasValue() ?
                  op.do_ic_alignment().getValue() : false;
    fuse_leaky = op.fused_leaky();
  } else if (auto op = dyn_cast<tpu::TG_BF16_Conv2DOp>(p)) {
    bool is_deconv = false;
    parseConvParam(op.param(), is_deconv, op.input(), op.output(), op.filter(),
                    n, ic, ih, iw, oc, oh, ow, g,
                    kh, kw, sh, sw, pt, pb, pl, pr, dh, dw, is_dw, with_bias, do_relu);
    do_ic_align = op.do_ic_alignment().hasValue() ?
                  op.do_ic_alignment().getValue() : false;
    fuse_leaky = op.fused_leaky();
  } else if (auto op = dyn_cast<tpu::TG_INT8_PC_DeConv2DOp>(p)) {
    bool is_deconv = true;
    parseConvParam(op.param(), is_deconv, op.input(), op.output(), op.filter(),
                    n, ic, ih, iw, oc, oh, ow, g,
                    kh, kw, sh, sw, pt, pb, pl, pr, dh, dw, is_dw, with_bias, do_relu);
    do_ic_align = op.do_ic_alignment().hasValue() ?
                  op.do_ic_alignment().getValue() : false;
    fuse_leaky = op.fused_leaky();
  }else if (auto op = dyn_cast<tpu::TG_BF16_DeConv2DOp>(p)) {
    bool is_deconv = true;
    parseConvParam(op.param(), is_deconv, op.input(), op.output(), op.filter(),
                    n, ic, ih, iw, oc, oh, ow, g,
                    kh, kw, sh, sw, pt, pb, pl, pr, dh, dw, is_dw, with_bias, do_relu);
    do_ic_align = op.do_ic_alignment().hasValue() ?
                  op.do_ic_alignment().getValue() : false;
    fuse_leaky = op.fused_leaky();
  } else {
    assert(!"Only support INT8/BF16 Conv in LayerGroup");
  }
}

static void getConcatParam(Operation *op,
                           int &axis, bool &do_relu) {
  if (isa<tpu::TG_INT8_ConcatOp>(op)) {
    auto concat_op = dyn_cast<tpu::TG_INT8_ConcatOp>(op);
    axis = concat_op.axis().getLimitedValue();
    do_relu = concat_op.do_relu();
  } else if (isa<tpu::TG_BF16_ConcatOp>(op)){
    auto concat_op = dyn_cast<tpu::TG_BF16_ConcatOp>(op);
    axis = concat_op.axis().getLimitedValue();
    do_relu = concat_op.do_relu();
  } else {
    assert(!"Only support INT8/BF16 Concat in LayerGroup");
  }
}

static void getSliceParam(Operation * op,
                          int &axis) {
  if (isa<tpu::TG_INT8_SliceOp>(op)) {
    auto slice_op = dyn_cast<tpu::TG_INT8_SliceOp>(op);
    axis = slice_op.axis().getLimitedValue();
  } else if (isa<tpu::TG_BF16_SliceOp>(op)) {
    auto slice_op = dyn_cast<tpu::TG_BF16_SliceOp>(op);
    axis = slice_op.axis().getLimitedValue();
  } else {
    assert(!"Only support INT8/BF16 Slice in LayerGroup");
  }
}

static void getUpsampleParam(Operation * op,
                             int &scale) {
  if (isa<tpu::TG_INT8_UpsampleOp>(op)) {
    auto upsample_op = dyn_cast<tpu::TG_INT8_UpsampleOp>(op);
    scale = upsample_op.scale().getLimitedValue();
  } else if (isa<tpu::TG_BF16_UpsampleOp>(op)) {
    auto upsample_op = dyn_cast<tpu::TG_BF16_UpsampleOp>(op);
    scale = upsample_op.scale().getLimitedValue();
  } else {
    assert(!"Only support INT8/BF16 Upsample in LayerGroup");
  }
}

static void getPoolingParam(Operation * op,
                            int &n, int &c, int &ih, int &iw,
                            int &oh, int &ow,
                            int &kh, int &kw, int &sh, int &sw,
                            int &pt, int &pb, int &pl, int &pr,
                            bool &is_global, bool &do_relu,
                            bool &count_include_pad) {
  if (isa<tpu::TG_INT8_PoolAvg2DOp>(op)) {
    auto pooling_op = cast<tpu::TG_INT8_PoolAvg2DOp>(op);
    parsePoolParam(pooling_op.param(), pooling_op.input(),
                   pooling_op.output(),
                   n, c, ih, iw, oh, ow,
                   kh, kw, sh, sw, pt, pb, pl, pr,
                   is_global, do_relu, count_include_pad);
  } else if (isa<tpu::TG_INT8_PoolMax2DOp>(op)) {
    auto pooling_op = cast<tpu::TG_INT8_PoolMax2DOp>(op);
    parsePoolParam(pooling_op.param(), pooling_op.input(),
                   pooling_op.output(),
                   n, c, ih, iw, oh, ow,
                   kh, kw, sh, sw, pt, pb, pl, pr,
                   is_global, do_relu, count_include_pad);
  } else if (isa<tpu::TG_BF16_PoolAvg2DOp>(op)) {
    auto pooling_op = cast<tpu::TG_BF16_PoolAvg2DOp>(op);
    parsePoolParam(pooling_op.param(), pooling_op.input(),
                   pooling_op.output(),
                   n, c, ih, iw, oh, ow,
                   kh, kw, sh, sw, pt, pb, pl, pr,
                   is_global, do_relu, count_include_pad);
  } else if (isa<tpu::TG_BF16_PoolMax2DOp>(op)) {
    auto pooling_op = cast<tpu::TG_BF16_PoolMax2DOp>(op);
    parsePoolParam(pooling_op.param(), pooling_op.input(),
                   pooling_op.output(),
                   n, c, ih, iw, oh, ow,
                   kh, kw, sh, sw, pt, pb, pl, pr,
                   is_global, do_relu, count_include_pad);
  } else {
    assert(!"Only support INT8/BF16 Pooling in LayerGroup");
  }
}

static void getEltwiseAddParam(Operation * op,
                               bool &do_early_stride,
                               int &h_stride, int &w_stride) {
  if (isa<tpu::TG_INT8_EltwiseAddOp>(op)) {
    auto eltwise_op = dyn_cast<tpu::TG_INT8_EltwiseAddOp>(op);
    do_early_stride = eltwise_op.do_early_stride();
    h_stride = eltwise_op.early_stride_h().getLimitedValue();
    w_stride = eltwise_op.early_stride_w().getLimitedValue();
  } else if(isa<tpu::TG_BF16_EltwiseAddOp>(op)) {
    auto eltwise_op = dyn_cast<tpu::TG_BF16_EltwiseAddOp>(op);
    do_early_stride = eltwise_op.do_early_stride();
    h_stride = eltwise_op.early_stride_h().getLimitedValue();
    w_stride = eltwise_op.early_stride_w().getLimitedValue();
  } else {
    assert(!"Unsupport eltwise add op in Layergroup.");
  }
}

static void getEltwiseReluParam(Operation * op,
                                bool &do_relu) {
  if (auto eltwise_op = dyn_cast<tpu::TG_INT8_EltwiseAddOp>(op)) {
    do_relu = eltwise_op.do_relu();
  } else if(auto eltwise_op = dyn_cast<tpu::TG_BF16_EltwiseAddOp>(op)) {
    do_relu = eltwise_op.do_relu();
  } else if(auto eltwise_op = dyn_cast<tpu::TG_INT8_EltwiseMulOp>(op)) {
    do_relu = eltwise_op.do_relu();
  } else if(auto eltwise_op = dyn_cast<tpu::TG_BF16_EltwiseMulOp>(op)) {
    do_relu = eltwise_op.do_relu();
  } else {
    assert(!"Unsupport eltwise op in Layergroup.");
  }
}
}
#endif