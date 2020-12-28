//===- CompressWeight- Implementation of activation compression -----------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file implements the activation compression.
//
//===----------------------------------------------------------------------===//

#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Passes.h"
#include "tpuc/TPUCompressUtil.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/MachineInfo.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "tpuc/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "compress-activation"

using namespace mlir;

namespace {

struct CmprStat{
  int store;
  int load;
  Operation *prevOp;
};


static bool isTgConvOp(Operation *op) {
  if (llvm::dyn_cast<tpu::TG_INT8_PT_Conv2DOp>(op) ||
      llvm::dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(op) ||
      llvm::dyn_cast<tpu::TG_BF16_Conv2DOp>(op))
    return true;
  return false;
}

static bool isTgEltAddOp(Operation *op) {
  if (llvm::dyn_cast<tpu::TG_INT8_EltwiseAddOp>(op) ||
      llvm::dyn_cast<tpu::TG_BF16_EltwiseAddOp>(op))
    return true;
  return false;
}

static bool isLgLoadNeuronOp(Operation *op) {
  if (llvm::dyn_cast<tpu::TL_LG_LoadNeuronOp>(op))
    return true;
  return false;
}

static bool isTlConvOp(Operation *op) {
  if (llvm::dyn_cast<tpu::TL_LW_Conv2DOp>(op))
    return true;
  return false;
}

static bool isStoreCompressedOp(Operation *op) {
  if (isTgConvOp(op) || isTgEltAddOp(op))
    return true;
  return false;

}
static bool isLoadDecompressedOp(Operation *op) {
  if (isTgConvOp(op) || isTgEltAddOp(op) || isLgLoadNeuronOp(op) ||
      isTlConvOp(op))
    return true;
  return false;
}

template <typename OpTy>
static void setTgOpCompressed(OpTy convOp, PatternRewriter &rewriter,
                              int cmprNStep, int cmprOcStep, int cmprOhStep,
                              int64_t stepSize, int64_t totalSize) {
  auto op = convOp.getOperation();
  convOp.setAttr("store_compr_act",
                 Builder(op->getContext()).getBoolAttr(true));
  convOp.setAttr("store_compr_act_param",
      tpu::ActCmprParam::get(
          Builder(op->getContext()).getI32IntegerAttr(cmprNStep),
          Builder(op->getContext()).getI32IntegerAttr(cmprOcStep),
          Builder(op->getContext()).getI32IntegerAttr(cmprOhStep),
          Builder(op->getContext()).getI64IntegerAttr(stepSize),
          Builder(op->getContext()).getI64IntegerAttr(totalSize),
          rewriter.getContext()));
}

template <typename OpTy>
static void setTgOpDeCompressed(OpTy &convOp, PatternRewriter &rewriter,
                                int cmprNStep, int cmprOcStep, int cmprOhStep,
                                int64_t stepSize, int64_t totalSize) {
  auto op = convOp.getOperation();
  convOp.setAttr("load_compr_act",
                 Builder(op->getContext()).getBoolAttr(true));
  convOp.setAttr("load_compr_act_param",
      tpu::ActCmprParam::get(
          Builder(op->getContext()).getI32IntegerAttr(cmprNStep),
          Builder(op->getContext()).getI32IntegerAttr(cmprOcStep),
          Builder(op->getContext()).getI32IntegerAttr(cmprOhStep),
          Builder(op->getContext()).getI64IntegerAttr(stepSize),
          Builder(op->getContext()).getI64IntegerAttr(totalSize),
          rewriter.getContext()));
}

template <typename OpTy, typename NextOpTy>
static void setNextOpAsDecompressed(OpTy convOp, PatternRewriter &rewriter,
                                    int cmprNStep, int cmprOcStep,
                                    int cmprOhStep, int64_t stepSize,
                                    int64_t totalSize) {
  auto op = convOp.getOperation();
  for (auto &use : op->getResult(0).getUses()) {
    auto useOp = use.getOwner();
    auto nextTpuOp = llvm::dyn_cast<NextOpTy>(useOp);

    LLVM_DEBUG(llvm::dbgs()
        << "StoreTgConvCmprAct: layer ID " << getOpLayerId(op)
        << "" << getOpName(op)
        << ", " << op->getName() << ", store compressed, next op "
        << getOpName(nextTpuOp.getOperation())
        << ", " << nextTpuOp.getOperation()->getName()
        << ", load compressed\n");

    nextTpuOp.setAttr("load_compr_act",
                      Builder(op->getContext()).getBoolAttr(true));
    nextTpuOp.setAttr("load_compr_act_param",
        tpu::ActCmprParam::get(
            Builder(op->getContext()).getI32IntegerAttr(cmprNStep),
            Builder(op->getContext()).getI32IntegerAttr(cmprOcStep),
            Builder(op->getContext()).getI32IntegerAttr(cmprOhStep),
            Builder(op->getContext()).getI64IntegerAttr(stepSize),
            Builder(op->getContext()).getI64IntegerAttr(totalSize),
            rewriter.getContext()));
  }
}

static bool isLargeDmaTransfer(Operation *op) {
  auto dataTypeSize = getDataTypeSize(op->getResult(0));
  std::vector<int64_t> shapes = getTensorShape(op->getResult(0));
  auto count = std::accumulate(std::begin(shapes), std::end(shapes), 1,
                               std::multiplies<>());
  auto storeSize = count * dataTypeSize;
  if (storeSize >= 512)
    return true;

  return false;
}

template <typename OpTy>
bool isValidCompressTgConvParam(OpTy convOp) {
  // tg dw-conv not integrated with tg conv
  if (convOp.param().is_dw().getValue())
    return false;

  // Not support group conv
  if (convOp.param().group().getInt() > 1)
    return false;

  if (!convOp.tile_param().hasValue())
    return false;

  return true;
}

static bool isValidCompressTgConvOp(Operation *op) {
  // tg dw-conv not integrated with tg conv
  int ow_step = 0;
  if (auto convOp = llvm::dyn_cast<tpu::TG_INT8_PT_Conv2DOp>(op)) {
    if (!isValidCompressTgConvParam<tpu::TG_INT8_PT_Conv2DOp>(convOp))
      return false;

    ow_step = convOp.tile_param().getValue().ow_step().getInt();
  } else if (auto convOp = llvm::dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(op)) {
    if (!isValidCompressTgConvParam<tpu::TG_INT8_PC_Conv2DOp>(convOp))
      return false;

    ow_step = convOp.tile_param().getValue().ow_step().getInt();
  } else if (auto convOp = llvm::dyn_cast<tpu::TG_BF16_Conv2DOp>(op)) {
    if (!isValidCompressTgConvParam<tpu::TG_BF16_Conv2DOp>(convOp))
      return false;

    ow_step = convOp.tile_param().getValue().ow_step().getInt();
  }

  std::vector<int64_t> shapes = getTensorShape(op->getResult(0));
  int64_t n, oc, oh, ow;
  getNCHW(shapes, n, oc, oh, ow);

  // Too trivial
  if (oh == 1 || oc < (int64_t)MInfo::lane_num)
    return false;

  // Cannot compress if tiling in width
  if (ow_step < ow)
    return false;

  // Not support multi-batch
  if (n > 1)
    return false;

  return isLargeDmaTransfer(op);
}

static bool isValidDecompressTgConvOp(Operation *op) {
  std::vector<int64_t> shapes;
  int dataTypeSize = 1;
  int iw_step = 0;

  if (auto convOp = llvm::dyn_cast<tpu::TG_INT8_PT_Conv2DOp>(op)) {
    if (!isValidCompressTgConvParam<tpu::TG_INT8_PT_Conv2DOp>(convOp))
      return false;

    iw_step = convOp.tile_param().getValue().iw_step().getInt();
    dataTypeSize = getDataTypeSize(convOp.input());
    shapes = getTensorShape(convOp.input());
  } else if (auto convOp = llvm::dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(op)) {
    if (!isValidCompressTgConvParam<tpu::TG_INT8_PC_Conv2DOp>(convOp))
      return false;

    iw_step = convOp.tile_param().getValue().iw_step().getInt();
    dataTypeSize = getDataTypeSize(convOp.input());
    shapes = getTensorShape(convOp.input());
  } else if (auto convOp = llvm::dyn_cast<tpu::TG_BF16_Conv2DOp>(op)) {
    if (!isValidCompressTgConvParam<tpu::TG_BF16_Conv2DOp>(convOp))
      return false;

    iw_step = convOp.tile_param().getValue().iw_step().getInt();
    dataTypeSize = getDataTypeSize(convOp.input());
    shapes = getTensorShape(convOp.input());
  }

  if (iw_step < shapes[3])
    return false;

  auto count = std::accumulate(std::begin(shapes), std::end(shapes), 1,
                               std::multiplies<>());
  auto storeSize = count * dataTypeSize;
  if (storeSize >= 512)
    return true;

  return false;
}

static bool isValidCompressTgEltAddOp(Operation *op) {
  // Not support early stride
  if (auto tpuOp = llvm::dyn_cast<tpu::TG_INT8_EltwiseAddOp>(op)) {
    if (tpuOp.do_early_stride())
      return false;
  }else if (auto tpuOp = llvm::dyn_cast<tpu::TG_BF16_EltwiseAddOp>(op)) {
    if (tpuOp.do_early_stride())
      return false;
  }

  std::vector<int64_t> shapes = getTensorShape(op->getResult(0));
  int64_t n, c, h, w;
  getNCHW(shapes, n, c, h, w);

  // Too trivial
  if (c < (int)MInfo::lane_num)
    return false;

  // Not support multi-batch
  if (n > 1)
    return false;

  return isLargeDmaTransfer(op);
}

static bool isValidDecompressTgEltAddOp(Operation *op) {
  return isValidCompressTgEltAddOp(op);
}

static void showOpCmprStatus(Operation *op) {
  if (auto tpuOp = llvm::dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(op)) {
    if (tpuOp.store_compr_act().hasValue())
      LLVM_DEBUG(llvm::dbgs() << ", store_compr_act "
          << tpuOp.store_compr_act().getValue());
    if (tpuOp.load_compr_act().hasValue())
      LLVM_DEBUG(llvm::dbgs() << ", load_compr_act "
          << tpuOp.load_compr_act().getValue());
  } else if (auto tpuOp = llvm::dyn_cast<tpu::TG_BF16_Conv2DOp>(op)) {
    if (tpuOp.store_compr_act().hasValue())
      LLVM_DEBUG(llvm::dbgs() << ", store_compr_act "
          << tpuOp.store_compr_act().getValue());
    if (tpuOp.load_compr_act().hasValue())
      LLVM_DEBUG(llvm::dbgs() << ", load_compr_act "
          << tpuOp.load_compr_act().getValue());
  } else if (auto tpuOp = llvm::dyn_cast<tpu::TG_INT8_EltwiseAddOp>(op)) {
    if (tpuOp.store_compr_act().hasValue())
      LLVM_DEBUG(llvm::dbgs() << ", store_compr_act "
          << tpuOp.store_compr_act().getValue());
    if (tpuOp.load_compr_act().hasValue())
      LLVM_DEBUG(llvm::dbgs() << ", load_compr_act "
          << tpuOp.load_compr_act().getValue());
  } else if (auto tpuOp = llvm::dyn_cast<tpu::TG_BF16_EltwiseAddOp>(op)) {
    if (tpuOp.store_compr_act().hasValue())
      LLVM_DEBUG(llvm::dbgs() << ", store_compr_act "
          << tpuOp.store_compr_act().getValue());
    if (tpuOp.load_compr_act().hasValue())
      LLVM_DEBUG(llvm::dbgs() << ", load_compr_act "
          << tpuOp.load_compr_act().getValue());
  } else if (auto tpuOp = llvm::dyn_cast<tpu::TL_LG_LoadNeuronOp>(op)) {
    if (tpuOp.load_compr_act().hasValue())
      LLVM_DEBUG(llvm::dbgs() << ", load_compr_act "
          << tpuOp.load_compr_act().getValue());
  } else if (auto tpuOp = llvm::dyn_cast<tpu::TL_LW_Conv2DOp>(op)) {
    if (tpuOp.load_compr_act().hasValue())
      LLVM_DEBUG(llvm::dbgs() << ", load_compr_act "
          << tpuOp.load_compr_act().getValue());
  }
  LLVM_DEBUG(llvm::dbgs() << "\n");
}

static void showOpStatus(Operation *op) {

  LLVM_DEBUG(llvm::dbgs()
      << "  op " << getOpName(op)
      << ", " << op->getName()
      << ", layer ID " << getOpLayerId(op));
  showOpCmprStatus(op);

  for (auto operand : op->getOperands()) {
    auto opdOp = operand.getDefiningOp();
    if (!opdOp)
      continue;

    if (!llvm::dyn_cast<tpu::TpuOpCommonInterface>(opdOp))
      continue;

    LLVM_DEBUG(llvm::dbgs()
        << "    opd " << getOpName(opdOp)
        << ", " << opdOp->getName()
        << ", layer ID " << getOpLayerId(opdOp));
    showOpCmprStatus(opdOp);
  }

  for (auto &use : op->getResult(0).getUses()) {
    auto useOp = use.getOwner();

    if (!llvm::dyn_cast<tpu::TpuOpCommonInterface>(useOp))
      continue;

    LLVM_DEBUG(llvm::dbgs()
        << "    use " << getOpName(useOp)
        << ", " << useOp->getName()
        << ", layer ID " << getOpLayerId(useOp));
    showOpCmprStatus(useOp);
  }
}

template <typename OpTy>
class TgCompressActPattern
    : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  TgCompressActPattern(MLIRContext *ctx,
                       llvm::DenseMap<Operation *, CmprStat> &cmprMaps)
      : OpRewritePattern<OpTy>(ctx), cmprMaps(cmprMaps) {}

  LogicalResult matchAndRewrite(OpTy tpuOp,
                                PatternRewriter &rewriter) const override {
    if (tpuOp.store_compr_act().hasValue())
      return failure();

    auto op = tpuOp.getOperation();
    if (!cmprMaps[op].store)
      return failure();

    LLVM_DEBUG(llvm::dbgs()
        << "TgCompressActPattern " << getOpName(op)
        << ", " << op->getName()
        << ", layer ID " << getOpLayerId(op)
        << ", store " << cmprMaps[op].store
        << ", load " << cmprMaps[op].load
        << "\n");

    std::vector<int64_t> shapes = getTensorShape(op->getResult(0));
    int64_t n, oc, oh, ow;
    getNCHW(shapes, n, oc, oh, ow);
    int cmprNStep = 1;
    int cmprOcStep = MInfo::lane_num;
    int cmprOhStep = 1;
    int isBf16 = isBf16Tensor(op->getResult(0)) ? 1 : 0;
    int64_t stepSize = 0, totalSize = 0;
    getTiledCompressedSize(n, oc, oh, ow,
                           cmprNStep, cmprOcStep, cmprOhStep,
                           isBf16, stepSize, totalSize);

    tpuOp.setAttr("store_compr_act",
                Builder(op->getContext()).getBoolAttr(true));
    tpuOp.setAttr("store_compr_act_param",
        tpu::ActCmprParam::get(
            Builder(op->getContext()).getI32IntegerAttr(cmprNStep),
            Builder(op->getContext()).getI32IntegerAttr(cmprOcStep),
            Builder(op->getContext()).getI32IntegerAttr(cmprOhStep),
            Builder(op->getContext()).getI64IntegerAttr(stepSize),
            Builder(op->getContext()).getI64IntegerAttr(totalSize),
            rewriter.getContext()));

    showOpStatus(op);

    return success();
  }

  llvm::DenseMap<Operation *, CmprStat> &cmprMaps;
};

static bool getStoreCompressActParam(Operation *op, int &n, int &c, int &h,
                                     int64_t &step, int64_t &total) {
  if (auto tpuOp = llvm::dyn_cast<tpu::TG_INT8_PT_Conv2DOp>(op)) {
    if (tpuOp.store_compr_act_param().hasValue()) {
      parseActCompressParam(tpuOp.store_compr_act_param().getValue(), n, c, h,
                            step, total);
      return true;
    }
  } else if (auto tpuOp = llvm::dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(op)) {
    if (tpuOp.store_compr_act_param().hasValue()) {
      parseActCompressParam(tpuOp.store_compr_act_param().getValue(), n, c, h,
                            step, total);
      return true;
    }
  } else if (auto tpuOp = llvm::dyn_cast<tpu::TG_BF16_Conv2DOp>(op)) {
    if (tpuOp.store_compr_act_param().hasValue()) {
      parseActCompressParam(tpuOp.store_compr_act_param().getValue(), n, c, h,
                            step, total);
      return true;
    }
  } else if (auto tpuOp = llvm::dyn_cast<tpu::TG_INT8_EltwiseAddOp>(op)) {
    if (tpuOp.store_compr_act_param().hasValue()) {
      parseActCompressParam(tpuOp.store_compr_act_param().getValue(), n, c, h,
                            step, total);
      return true;
    }
  } else if (auto tpuOp = llvm::dyn_cast<tpu::TG_BF16_EltwiseAddOp>(op)) {
    if (tpuOp.store_compr_act_param().hasValue()) {
      parseActCompressParam(tpuOp.store_compr_act_param().getValue(), n, c, h,
                            step, total);
      return true;
    }
  }

  return false;
}

template <typename OpTy>
class TgDecompressActPattern
    : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  TgDecompressActPattern(MLIRContext *ctx,
                         llvm::DenseMap<Operation *, CmprStat> &cmprMaps)
      : OpRewritePattern<OpTy>(ctx), cmprMaps(cmprMaps) {}

  LogicalResult matchAndRewrite(OpTy tpuOp,
                                PatternRewriter &rewriter) const override {
    if (tpuOp.load_compr_act().hasValue())
      return failure();

    auto op = tpuOp.getOperation();
    if (!cmprMaps[op].load)
      return failure();

    assert(cmprMaps[op].prevOp && "Expect prev op");
    auto prevOp = cmprMaps[op].prevOp;

    LLVM_DEBUG(llvm::dbgs()
        << "TgDecompressActPattern " << getOpName(op)
        << ", " << op->getName()
        << ", layer ID " << getOpLayerId(op)
        << ", store " << cmprMaps[op].store
        << ", load " << cmprMaps[op].load
        << ", prevOp " << getOpName(prevOp)
        << ", " << prevOp->getName()
        << "\n");

    int cmpr_n = 0, cmpr_c = 0, cmpr_h = 0;
    int64_t cmpr_step = 0, cmpr_total = 0;
    getStoreCompressActParam(prevOp, cmpr_n, cmpr_c, cmpr_h, cmpr_step,
                             cmpr_total);
    tpuOp.setAttr("load_compr_act",
                  Builder(tpuOp.getContext()).getBoolAttr(true));
    tpuOp.setAttr("load_compr_act_param",
        tpu::ActCmprParam::get(
            Builder(op->getContext()).getI32IntegerAttr(cmpr_n),
            Builder(op->getContext()).getI32IntegerAttr(cmpr_c),
            Builder(op->getContext()).getI32IntegerAttr(cmpr_h),
            Builder(op->getContext()).getI64IntegerAttr(cmpr_step),
            Builder(op->getContext()).getI64IntegerAttr(cmpr_total),
            rewriter.getContext()));
    showOpStatus(tpuOp);

    return failure();
  }

  llvm::DenseMap<Operation *, CmprStat> &cmprMaps;
};

class TlLgLoadNeuronDecompressActPattern
    : public OpRewritePattern<tpu::TL_LG_LoadNeuronOp> {
public:
  using OpRewritePattern<tpu::TL_LG_LoadNeuronOp>::OpRewritePattern;

  TlLgLoadNeuronDecompressActPattern(MLIRContext *ctx,
                                     llvm::DenseMap<Operation *, CmprStat> &cmprMaps)
      : OpRewritePattern<tpu::TL_LG_LoadNeuronOp>(ctx), cmprMaps(cmprMaps) {}

  LogicalResult matchAndRewrite(tpu::TL_LG_LoadNeuronOp tpuOp,
                                PatternRewriter &rewriter) const override {

    if (tpuOp.load_compr_act().hasValue())
      return failure();

    auto op = tpuOp.getOperation();
    if (cmprMaps.find(op) == cmprMaps.end())
      return failure();
    if (!cmprMaps[op].load)
      return failure();

    assert(cmprMaps[op].prevOp && "Expect prev op");
    auto prevOp = cmprMaps[op].prevOp;

    LLVM_DEBUG(llvm::dbgs()
        << "TlLgLoadNeuronDecompressActPattern " << getOpName(op)
        << ", " << op->getName()
        << ", layer ID " << getOpLayerId(op)
        << ", store " << cmprMaps[op].store
        << ", load " << cmprMaps[op].load
        << ", prevOp " << getOpName(prevOp)
        << ", " << prevOp->getName()
        << "\n");

    int cmpr_n = 0, cmpr_c = 0, cmpr_h = 0;
    int64_t cmpr_step = 0, cmpr_total = 0;
    getStoreCompressActParam(prevOp, cmpr_n, cmpr_c, cmpr_h, cmpr_step,
                             cmpr_total);

    int eltSize = getDataTypeSize(op->getResult(0));

    // Physical offset(in byte) -> logical offset
    std::vector<int64_t> resShapes = getTensorShape(op->getResult(0));
    int64_t hOffset = tpuOp.offset().getValue() / resShapes[3] / eltSize;

    int64_t offset = cmpr_step * hOffset; // in byte
    LLVM_DEBUG(llvm::dbgs()
        << "      " << getOpName(op)
        << ", offset " << tpuOp.offset().getValue()
        << " -> " << offset
        << "(" << cmpr_step
        << " * " << hOffset
        << ")\n");

    tpuOp.removeAttr("offset");
    tpuOp.setAttr("offset",
                  Builder(op->getContext()).getI64IntegerAttr(offset));

    tpuOp.setAttr("load_compr_act",
                  Builder(tpuOp.getContext()).getBoolAttr(true));
    tpuOp.setAttr("compr_act_param",
        tpu::ActCmprParam::get(
            Builder(op->getContext()).getI32IntegerAttr(cmpr_n),
            Builder(op->getContext()).getI32IntegerAttr(cmpr_c),
            Builder(op->getContext()).getI32IntegerAttr(cmpr_h),
            Builder(op->getContext()).getI64IntegerAttr(cmpr_step),
            Builder(op->getContext()).getI64IntegerAttr(cmpr_total),
            rewriter.getContext()));

    return success();
  }

  llvm::DenseMap<Operation *, CmprStat> &cmprMaps;
};

template<typename OpTy>
class TlConvDecompressActPattern
    : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  TlConvDecompressActPattern(MLIRContext *ctx,
                             llvm::DenseMap<Operation *, CmprStat> &cmprMaps)
      : OpRewritePattern<OpTy>(ctx), cmprMaps(cmprMaps) {}

  LogicalResult matchAndRewrite(OpTy tpuOp,
                                PatternRewriter &rewriter) const override {
    if (tpuOp.load_compr_act().hasValue())
      return failure();

    auto op = tpuOp.getOperation();
    if (cmprMaps.find(op) == cmprMaps.end())
      return failure();
    if (!cmprMaps[op].load)
      return failure();

    assert(cmprMaps[op].prevOp && "Expect prev op");
    auto prevOp = cmprMaps[op].prevOp;

    LLVM_DEBUG(llvm::dbgs()
        << "TlConvDecompressActPattern " << getOpName(op)
        << ", " << op->getName()
        << ", layer ID " << getOpLayerId(op)
        << ", store " << cmprMaps[op].store
        << ", load " << cmprMaps[op].load
        << ", prevOp " << getOpName(prevOp)
        << ", " << prevOp->getName()
        << "\n");

    int cmpr_n = 0, cmpr_c = 0, cmpr_h = 0;
    int64_t cmpr_step = 0, cmpr_total = 0;
    getStoreCompressActParam(prevOp, cmpr_n, cmpr_c, cmpr_h, cmpr_step,
                             cmpr_total);

    tpuOp.setAttr("load_compr_act",
                  Builder(tpuOp.getContext()).getBoolAttr(true));
    tpuOp.setAttr("load_compr_act_param",
        tpu::ActCmprParam::get(
            Builder(op->getContext()).getI32IntegerAttr(cmpr_n),
            Builder(op->getContext()).getI32IntegerAttr(cmpr_c),
            Builder(op->getContext()).getI32IntegerAttr(cmpr_h),
            Builder(op->getContext()).getI64IntegerAttr(cmpr_step),
            Builder(op->getContext()).getI64IntegerAttr(cmpr_total),
            rewriter.getContext()));

    return success();
  }

  llvm::DenseMap<Operation *, CmprStat> &cmprMaps;
};

static bool getTiledCompressedActShapeAndSize(Operation *op,
    std::vector<std::vector<int64_t>> &storeShapes,
    std::vector<std::vector<int64_t>> &loadShapes, int isBf16,
    int &n_step, int &c_step, int &h_step, int64_t &stepSize,
    int64_t &totalSize) {

  // At least one element
  if (!storeShapes.size() || !loadShapes.size())
    return false;

  // Only support 4D tensor
  if (storeShapes[0].size() != 4)
    return false;

  // Not support different batch size
  std::vector<int64_t> outputShapes = getTensorShape(op->getResult(0));
  for (auto ss : storeShapes) {
    for (auto ls : loadShapes) {
      if ((outputShapes[0] != ss[0]) || (ss[0] != ls[0]))
        return false;
    }
  }

  int n = 0, c = 0, h = 0, w = 0;
  n = storeShapes[0][0];
  c = storeShapes[0][1];
  w = storeShapes[0][3];

  if (storeShapes[0][1] != loadShapes[0][1]) {
    // Not support different channel size
    return false;
  } else if ((storeShapes.size() == loadShapes.size()) &&
             (storeShapes.size() == 1)) {
    // one store, one load
    n_step = storeShapes[0][0];
    c_step = storeShapes[0][1];
    h_step = storeShapes[0][2];
    h = storeShapes[0][2];
  } else if ((storeShapes.size() != loadShapes.size()) &&
             (loadShapes.size() == 1)) {
    // multiple stores, one load
    n_step = storeShapes[0][0];
    c_step = storeShapes[0][1];
    h_step = storeShapes[0][2];
    h = loadShapes[0][2];
  } else {
    // multiple stores, multiple loads
    n_step = storeShapes[0][0];
    c_step = storeShapes[0][1];
    h_step = 1;

    for (auto v : storeShapes) {
      h += v[2];
    }
  }

  getTiledCompressedSize(n, c, h, w, n_step, c_step, h_step, isBf16,
                         stepSize, totalSize);

  LLVM_DEBUG(llvm::dbgs()
      << "\n  getTiledCompressedActShapeAndSize\n    "
      << "storeShapes " << storeShapes.size()
      << ", loadShapes " << loadShapes.size()
      << " shape (" << n
      << ", " << c
      << ", " << h
      << ", " << w
      << "), tile (" << n_step
      << ", " << c_step
      << ", " << h_step
      << ", " << w
      << "), stepSize " << stepSize
      << ", totalSize " << totalSize
      << "\n");

  return true;
}

//
// Almost TG and TL op do not support yet.
//
static bool isValidLoadCompressActForTlLgJoin(Operation *op, int h_step) {

  for (auto &use : op->getResult(0).getUses()) {
    auto useOp = use.getOwner();

    if (llvm::dyn_cast<tpu::TpuTLOpCodegenInterface>(useOp)) {
      // valid layer group tiu op
    } else if (llvm::dyn_cast<tpu::TL_LW_Conv2DOp>(useOp)) {
      // supported deep fusion op
    } else if (auto tpuOp = llvm::dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(useOp)) {
      std::vector<int64_t> inputShapes = getTensorShape(op->getOperand(0));
      if (inputShapes[2] != h_step) {
        return false;
      }
    } else {
      LLVM_DEBUG(llvm::dbgs() << "  user op " << getOpName(op)
          << ", " << op->getName() << " not support load_compr_act yet\n");
      return false;
    }
  }

  return true;
}

// tl_lg_store -> tl_lg_join -> tl_lg_load_neuron/tl_lw_conv2d
//  case 1: one tile
//          (1, 512, 7, 7) -> (1, 512, 7, 7)
//
//  case 2: multiple tiles of tdma store, single tile of tdma load
//          (1, 64, 19, 56)
//          (1, 64, 19, 56) -> (1, 64, 56, 56)
//          (1, 64, 18, 56)
//
//  case 3: multiple tiles of tdma store, multiple tiles of tdma load
//          (1, 64, 56, 112)    (1, 64, 35, 112)
//          (1, 64, 56, 112) -> (1, 64, 37, 112)
//                              (1, 64, 37, 112)
//
class TlLgJointCompressedActPattern
    : public OpRewritePattern<tpu::TL_LG_JoinOp> {
public:
  using OpRewritePattern<tpu::TL_LG_JoinOp>::OpRewritePattern;

  TlLgJointCompressedActPattern(MLIRContext *ctx, MInfo &mInfo)
      : OpRewritePattern<tpu::TL_LG_JoinOp>(ctx), mInfo(mInfo) {}

  LogicalResult matchAndRewrite(tpu::TL_LG_JoinOp tpuOp,
                                     PatternRewriter &rewriter) const override {
    Operation *op = tpuOp.getOperation();

    std::vector<int64_t> outputShapes = getTensorShape(op->getResult(0));
    LLVM_DEBUG(llvm::dbgs()
        << "\nTlLgJoinCmprAct: " << getOpName(op)
        << ", " << op->getName()
        << ", -> (" << outputShapes[0]
        << ", " << outputShapes[1]
        << ", " << outputShapes[2]
        << ", " << outputShapes[3]
        << ")\n");

    // Not support multi-batch yet
    if (outputShapes[0] > 1)
      return failure();

    std::vector<std::vector<int64_t>> storeShapes;
    std::vector<std::vector<int64_t>> loadShapes;
    for (auto operand : op->getOperands()) {
      auto opdOp = operand.getDefiningOp();
      std::vector<int64_t> inputShapes = getTensorShape(opdOp->getOperand(0));
      std::vector<int64_t> outputShapes = getTensorShape(opdOp->getResult(0));

      LLVM_DEBUG(llvm::dbgs()
          << "  operand " << getOpName(opdOp)
          << ", " << opdOp->getName()
          << ", (" << inputShapes[0]
          << ", " << inputShapes[1]
          << ", " << inputShapes[2]
          << ", " << inputShapes[3]
          << ") -> (" << outputShapes[0]
          << ", " << outputShapes[1]
          << ", " << outputShapes[2]
          << ", " << outputShapes[3]
          << ")\n");

      if (!llvm::dyn_cast<tpu::TL_LG_StoreOp>(opdOp))
        break;

      if (inputShapes.size() == 4)
        storeShapes.push_back(inputShapes);
    }

    // tl_lg_join -> tl_lg_load_neuron
    // tl_lg_join -> tl_lw_conv_2d
    for (auto &use : op->getResult(0).getUses()) {
      auto useOp = use.getOwner();

      std::vector<int64_t> shapes;
      if (llvm::dyn_cast<tpu::TL_LG_LoadNeuronOp>(useOp))
        shapes = getTensorShape(useOp->getResult(0));
      else if (llvm::dyn_cast<tpu::TL_LW_Conv2DOp>(useOp))
        shapes = getTensorShape(useOp->getOperand(0));
      else
        break;

      LLVM_DEBUG(llvm::dbgs()
          << "  userOp " << getOpName(useOp)
          << ", " << useOp->getName()
          << ", shape (" << shapes[0]
          << ", " << shapes[1]
          << ", " << shapes[2]
          << ", " << shapes[3]
          << ")\n");

      if (!llvm::dyn_cast<tpu::TL_LG_LoadNeuronOp>(useOp) &&
          !llvm::dyn_cast<tpu::TL_LW_Conv2DOp>(useOp))
        break;

      if (shapes.size() == 4)
        loadShapes.push_back(shapes);
    }

    int n_step, oc_step, oh_step;
    int64_t step_size, total_size;
    int isBf16 = isBf16Tensor(op->getResult(0)) ? 1 : 0;
    if (!getTiledCompressedActShapeAndSize(op, storeShapes, loadShapes, isBf16,
                                           n_step, oc_step, oh_step, step_size,
                                           total_size))
      return failure();

    if (!isValidLoadCompressActForTlLgJoin(op, oh_step))
      return failure();

    auto enableStoreCmprAct = [&](Operation *op, int64_t &offset) {
    if (auto tpuOp = llvm::dyn_cast<tpu::TL_LG_StoreOp>(op)) {
        LLVM_DEBUG(llvm::dbgs()
            << "      " << getOpName(op)
            << ", offset " << tpuOp.offset().getValue()
            << " -> " << offset << "\n");

        tpuOp.removeAttr("offset");
        tpuOp.setAttr("offset",
                      Builder(op->getContext()).getI64IntegerAttr(offset));

        auto shapes = getTensorShape(op->getOperand(0));
        offset += step_size * shapes[2] / oh_step;

        tpuOp.setAttr("store_compr_act",
                      Builder(tpuOp.getContext()).getBoolAttr(true));
        auto value =
            tpu::ActCmprParam::get(
                Builder(op->getContext()).getI32IntegerAttr(n_step),
                Builder(op->getContext()).getI32IntegerAttr(oc_step),
                Builder(op->getContext()).getI32IntegerAttr(oh_step),
                Builder(op->getContext()).getI64IntegerAttr(step_size),
                Builder(op->getContext()).getI64IntegerAttr(total_size),
                rewriter.getContext());
        tpuOp.setAttr("compr_act_param", value);
      }
    };

    auto enableLoadCmprAct = [&](Operation *op) {
      if (auto tpuOp = llvm::dyn_cast<tpu::TL_LG_LoadNeuronOp>(op)) {
        auto resultTy =
            op->getResult(0).getType().template dyn_cast<RankedTensorType>();
        auto eltType = resultTy.getElementType();
        int eltSize = llvm::divideCeil(eltType.getIntOrFloatBitWidth(), 8);

        // Physical offset(in byte) -> logical offset
        std::vector<int64_t> resShapes = getTensorShape(op->getResult(0));
        int64_t hOffset = tpuOp.offset().getValue() /
                          resShapes[3] / eltSize;

        int64_t offset = step_size * hOffset; // in byte
        LLVM_DEBUG(llvm::dbgs()
            << "      " << getOpName(op)
            << ", offset " << tpuOp.offset().getValue()
            << " -> " << offset
            << "(" << step_size
            << " * " << hOffset
            << ")\n");

        tpuOp.removeAttr("offset");
        tpuOp.setAttr("offset",
                      Builder(op->getContext()).getI64IntegerAttr(offset));

        tpuOp.setAttr("load_compr_act",
                      Builder(tpuOp.getContext()).getBoolAttr(true));
        tpuOp.setAttr("compr_act_param",
            tpu::ActCmprParam::get(
                Builder(op->getContext()).getI32IntegerAttr(n_step),
                Builder(op->getContext()).getI32IntegerAttr(oc_step),
                Builder(op->getContext()).getI32IntegerAttr(oh_step),
                Builder(op->getContext()).getI64IntegerAttr(step_size),
                Builder(op->getContext()).getI64IntegerAttr(total_size),
                rewriter.getContext()));
      } else if (auto tpuOp = llvm::dyn_cast<tpu::TL_LW_Conv2DOp>(op)) {
        tpuOp.setAttr("load_compr_act",
                      Builder(tpuOp.getContext()).getBoolAttr(true));
        tpuOp.setAttr("load_compr_act_param",
            tpu::ActCmprParam::get(
                Builder(op->getContext()).getI32IntegerAttr(n_step),
                Builder(op->getContext()).getI32IntegerAttr(oc_step),
                Builder(op->getContext()).getI32IntegerAttr(oh_step),
                Builder(op->getContext()).getI64IntegerAttr(step_size),
                Builder(op->getContext()).getI64IntegerAttr(total_size),
                rewriter.getContext()));
      } else if (auto tpuOp = llvm::dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(op)) {
        std::vector<int64_t> inputShapes = getTensorShape(op->getOperand(0));

        assert(inputShapes[2] == oh_step &&
              "tl_lg_join->tg_conv2d not support load tiled act");

        tpuOp.setAttr("load_compr_act",
                      Builder(tpuOp.getContext()).getBoolAttr(true));
      } else {
        LLVM_DEBUG(llvm::dbgs() << "      load_compr_act not set\n");
      }
    };

    // lg_tiu_op -> tl_lg_store -> tl_join
    // lg_lg_store -> tl_lw_conv2d
    //   Enable store_compr_act of tl_lg_store
    int64_t store_offset = 0;
    // int64_t load_offset = 0;
    for (auto operand : op->getOperands()) {
      auto tiuOp = operand.getDefiningOp()->getOperand(0).getDefiningOp();

      LLVM_DEBUG(llvm::dbgs()
        << "  Mark cmpr_store, tiuOp " << getOpName(tiuOp)
        << ", " << tiuOp->getName()
        << "\n");

      // Mark lg_store
      enableStoreCmprAct(operand.getDefiningOp(), store_offset);

      // Mark all user of pre tiu op
      for (auto &use : tiuOp->getResult(0).getUses()) {
        auto useOp = use.getOwner();
        LLVM_DEBUG(llvm::dbgs()
            << "    mark cmpr_load, useOp " << getOpName(useOp)
            << ", " << useOp->getName()
            << "\n");

        enableLoadCmprAct(useOp);
      }
    }

    // tl_join -> tl_lg_load_neuron
    //   Enable load_compr_act of tl_lg_load_neuron
    //
    // lg_lg_store -> tl_lw_conv2d
    //   Enable load_compr_act of tl_lw_conv2d
    //
    int index = 0;
    for (auto &use : op->getResult(0).getUses()) {
      auto useOp = use.getOwner();

      LLVM_DEBUG(llvm::dbgs()
        << "  Mark cmpr_load, useOp " << getOpName(useOp)
        << ", " << useOp->getName()
        << "\n");

      enableLoadCmprAct(useOp);
      index++;
    }

    return failure();
  }

  MInfo &mInfo;
};

struct CompressActivationPass : public mlir::PassWrapper<CompressActivationPass, FunctionPass> {
  void runOnFunction() override;

  template <typename OpTy>
  void setOpStatus(OpTy tpuOp);

  template <typename OpTy>
  void addTgConvOpToCmprMap(OpTy tpuOp);

  template <typename OpTy>
  void addTgEltAddOpToCmprMap(OpTy tpuOp);

  void addTlLdLoadNeuronOpToCmprMap(Operation *op);

  void addTlConvOpToCmprMap(Operation *op);

  template <typename OpTy>
  void checkTgOpStatus(OpTy tpuOp);

  template <typename OpTy>
  void assertTgOpLoadCompressedAct(OpTy tpuOp);

  template <typename OpTy>
  void assertTgOpLoadPlainAct(OpTy tpuOp);

  template <typename OpTy>
  void assertTgOpStoreCompressedAct(OpTy tpuOp);

  template <typename OpTy>
  void assertTgOpStorePlainAct(OpTy tpuOp);

  template <typename OpTy>
  void assertTlConvOpLoadCompressedAct(OpTy tpuOp);

  template <typename OpTy>
  void assertTlConvOpLoadPlainAct(OpTy tpuOp);

  void assertTlLgLoadNeuronOpLoadCompressedAct(tpu::TL_LG_LoadNeuronOp tpuOp);
  void assertTlLgLoadNeuronOpLoadPlainAct(tpu::TL_LG_LoadNeuronOp tpuOp);

  void addOpToCmprMaps();
  void markLoadDeCompressed();
  void showMarkCompressed();
  void analyze();

  void checkTgOpCompressStats();

  llvm::DenseMap<Operation *, CmprStat> cmprMaps;
};

} // anonymous namespace

template <typename OpTy>
void CompressActivationPass::addTgConvOpToCmprMap(OpTy tpuOp) {
  auto op = tpuOp.getOperation();

  if (isValidCompressTgConvOp(op)) {
    cmprMaps[op] = {1, 0};
  } else {
    cmprMaps[op] = {0, 0};
  }
}

template <typename OpTy>
void CompressActivationPass::addTgEltAddOpToCmprMap(OpTy tpuOp) {
  auto op = tpuOp.getOperation();

  if (isValidCompressTgEltAddOp(tpuOp)) {
    cmprMaps[op] = {1, 0, nullptr};
  } else {
    cmprMaps[op] = {0, 0, nullptr};
  }
}

void CompressActivationPass::addTlLdLoadNeuronOpToCmprMap(Operation *op) {
  cmprMaps[op] = {0, 0, nullptr};
}

void CompressActivationPass::addTlConvOpToCmprMap(Operation *op)
{
  cmprMaps[op] = {0, 0, nullptr};
}

template <typename OpTy>
void CompressActivationPass::assertTgOpLoadCompressedAct(OpTy tpuOp) {
  if (!tpuOp.load_compr_act().hasValue() ||
      !tpuOp.load_compr_act_param().hasValue()) {
    LLVM_DEBUG(llvm::dbgs() << "      " << tpuOp.getOpName()
              << ", error ! expect decompressed\n");
    showOpStatus(tpuOp.getOperation());
  }

  assert(tpuOp.load_compr_act().hasValue());
  assert(tpuOp.load_compr_act_param().hasValue());
}

template <typename OpTy>
void CompressActivationPass::assertTgOpLoadPlainAct(OpTy tpuOp) {
  if (tpuOp.load_compr_act().hasValue() ||
      tpuOp.load_compr_act_param().hasValue()) {
    LLVM_DEBUG(llvm::dbgs() << "      " << tpuOp.getOpName()
                << ", error ! expect load plain\n");
    showOpStatus(tpuOp.getOperation());
  }

  assert(!tpuOp.load_compr_act().hasValue());
  assert(!tpuOp.load_compr_act_param().hasValue());
}

template <typename OpTy>
void CompressActivationPass::assertTgOpStoreCompressedAct(OpTy tpuOp) {
  if (!tpuOp.store_compr_act().hasValue() ||
      !tpuOp.store_compr_act_param().hasValue()) {
    LLVM_DEBUG(llvm::dbgs() << "      " << tpuOp.getOpName()
                << ", error ! expect store compressed\n");
    showOpStatus(tpuOp.getOperation());
  }

  assert(tpuOp.store_compr_act().hasValue());
  assert(tpuOp.store_compr_act_param().hasValue());
}

template <typename OpTy>
void CompressActivationPass::assertTgOpStorePlainAct(OpTy tpuOp) {
  if (tpuOp.store_compr_act().hasValue() ||
      tpuOp.store_compr_act_param().hasValue()) {
    LLVM_DEBUG(llvm::dbgs() << "      " << tpuOp.getOpName()
                << ", error ! expect store plain\n");
    showOpStatus(tpuOp.getOperation());
  }

  assert(!tpuOp.store_compr_act().hasValue());
  assert(!tpuOp.store_compr_act_param().hasValue());
}

template <typename OpTy>
void CompressActivationPass::assertTlConvOpLoadCompressedAct(OpTy tpuOp) {
  if (!tpuOp.load_compr_act().hasValue() ||
      !tpuOp.load_compr_act_param().hasValue()) {
    LLVM_DEBUG(llvm::dbgs() << "      " << tpuOp.getOpName()
              << ", error ! expect decompressed\n");
    showOpStatus(tpuOp.getOperation());
  }

  assert(tpuOp.load_compr_act().hasValue());
  assert(tpuOp.load_compr_act_param().hasValue());
}

template <typename OpTy>
void CompressActivationPass::assertTlConvOpLoadPlainAct(OpTy tpuOp) {
  if (tpuOp.load_compr_act().hasValue() ||
      tpuOp.load_compr_act_param().hasValue()) {
    LLVM_DEBUG(llvm::dbgs() << "      " << tpuOp.getOpName()
                << ", error ! expect load plain\n");
    showOpStatus(tpuOp.getOperation());
  }

  assert(!tpuOp.load_compr_act().hasValue());
  assert(!tpuOp.load_compr_act_param().hasValue());
}

void CompressActivationPass::assertTlLgLoadNeuronOpLoadCompressedAct(
                                  tpu::TL_LG_LoadNeuronOp tpuOp) {
  if (!tpuOp.load_compr_act().hasValue() ||
      !tpuOp.compr_act_param().hasValue()) {
    LLVM_DEBUG(llvm::dbgs() << "      " << tpuOp.getOpName()
              << ", error ! expect decompressed\n");
    showOpStatus(tpuOp.getOperation());
  }

  assert(tpuOp.load_compr_act().hasValue());
  assert(tpuOp.compr_act_param().hasValue());
}

void CompressActivationPass::assertTlLgLoadNeuronOpLoadPlainAct(
                                tpu::TL_LG_LoadNeuronOp tpuOp) {
  if (tpuOp.load_compr_act().hasValue() ||
      tpuOp.compr_act_param().hasValue()) {
    LLVM_DEBUG(llvm::dbgs() << "      " << tpuOp.getOpName()
                << ", error ! expect load plain\n");
    showOpStatus(tpuOp.getOperation());
  }

  assert(!tpuOp.load_compr_act().hasValue());
  assert(!tpuOp.compr_act_param().hasValue());
}

template <typename OpTy>
void CompressActivationPass::checkTgOpStatus(OpTy tpuOp) {
  auto op = tpuOp.getOperation();

  if (tpuOp.load_compr_act().hasValue()) {
    assert(tpuOp.load_compr_act_param().hasValue());

    for (auto operand : op->getOperands()) {
      auto opdOp = operand.getDefiningOp();

      if (auto opdTpuOp = llvm::dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(opdOp)) {
        assertTgOpStoreCompressedAct<tpu::TG_INT8_PC_Conv2DOp>(opdTpuOp);
      } else if (auto opdTpuOp =
                 llvm::dyn_cast<tpu::TG_INT8_PT_Conv2DOp>(opdOp)) {
        assertTgOpStoreCompressedAct<tpu::TG_INT8_PT_Conv2DOp>(opdTpuOp);
      } else if (auto opdTpuOp = llvm::dyn_cast<tpu::TG_BF16_Conv2DOp>(opdOp)) {
        assertTgOpStoreCompressedAct<tpu::TG_BF16_Conv2DOp>(opdTpuOp);
      } else if (auto opdTpuOp =
                 llvm::dyn_cast<tpu::TG_INT8_EltwiseAddOp>(opdOp)) {
        assertTgOpStoreCompressedAct<tpu::TG_INT8_EltwiseAddOp>(opdTpuOp);
      } else if (auto opdTpuOp =
                 llvm::dyn_cast<tpu::TG_BF16_EltwiseAddOp>(opdOp)) {
        assertTgOpStoreCompressedAct<tpu::TG_BF16_EltwiseAddOp>(opdTpuOp);
      } else if (llvm::dyn_cast<tpu::TpuOpCommonInterface>(opdOp)) {
        llvm_unreachable("Expect supported cmpr op");
      }
    }
  } else {
    assert(!tpuOp.load_compr_act_param().hasValue());

    for (auto operand : op->getOperands()) {
      auto opdOp = operand.getDefiningOp();

      if (auto opdTpuOp = llvm::dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(opdOp)) {
        assertTgOpStorePlainAct<tpu::TG_INT8_PC_Conv2DOp>(opdTpuOp);
      } else if (auto opdTpuOp =
                 llvm::dyn_cast<tpu::TG_INT8_PT_Conv2DOp>(opdOp)) {
        assertTgOpStorePlainAct<tpu::TG_INT8_PT_Conv2DOp>(opdTpuOp);
      } else if (auto opdTpuOp = llvm::dyn_cast<tpu::TG_BF16_Conv2DOp>(opdOp)) {
        assertTgOpStorePlainAct<tpu::TG_BF16_Conv2DOp>(opdTpuOp);
      } else if (auto opdTpuOp =
                 llvm::dyn_cast<tpu::TG_INT8_EltwiseAddOp>(opdOp)) {
        assertTgOpStorePlainAct<tpu::TG_INT8_EltwiseAddOp>(opdTpuOp);
      } else if (auto opdTpuOp =
                 llvm::dyn_cast<tpu::TG_BF16_EltwiseAddOp>(opdOp)) {
        assertTgOpStorePlainAct<tpu::TG_BF16_EltwiseAddOp>(opdTpuOp);
      }
    }
  }

  if (tpuOp.store_compr_act().hasValue()) {
    assert(tpuOp.store_compr_act_param().hasValue());

    for (auto &use : op->getResult(0).getUses()) {
      auto useOp = use.getOwner();
      if (auto useTpuOp = llvm::dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(useOp)) {
        assertTgOpLoadCompressedAct<tpu::TG_INT8_PC_Conv2DOp>(useTpuOp);
      } else if (auto useTpuOp =
                 llvm::dyn_cast<tpu::TG_INT8_PT_Conv2DOp>(useOp)) {
        assertTgOpLoadCompressedAct<tpu::TG_INT8_PT_Conv2DOp>(useTpuOp);
      } else if (auto useTpuOp = llvm::dyn_cast<tpu::TG_BF16_Conv2DOp>(useOp)) {
        assertTgOpLoadCompressedAct<tpu::TG_BF16_Conv2DOp>(useTpuOp);
      } else if (auto useTpuOp =
                 llvm::dyn_cast<tpu::TG_INT8_EltwiseAddOp>(useOp)) {
        assertTgOpLoadCompressedAct<tpu::TG_INT8_EltwiseAddOp>(useTpuOp);
      } else if (auto useTpuOp =
                 llvm::dyn_cast<tpu::TG_BF16_EltwiseAddOp>(useOp)) {
        assertTgOpLoadCompressedAct<tpu::TG_BF16_EltwiseAddOp>(useTpuOp);
      } else if (auto useTpuOp =
                 llvm::dyn_cast<tpu::TL_LG_LoadNeuronOp>(useOp)) {
        assertTlLgLoadNeuronOpLoadCompressedAct(useTpuOp);
      } else if (auto useTpuOp =
                 llvm::dyn_cast<tpu::TL_LW_Conv2DOp>(useOp)) {
         assertTlConvOpLoadCompressedAct(useTpuOp);
      } else if (auto useTpuOp =
                 llvm::dyn_cast<tpu::TpuOpCommonInterface>(useOp)) {
        LLVM_DEBUG(llvm::dbgs() << "  error ! useOp " << useTpuOp.getOpName()
                   << ", " << useOp->getName() << "\n");
        llvm_unreachable("Expect supported cmpr op");
      }
    }
  } else {
    assert(!tpuOp.store_compr_act_param().hasValue());

    for (auto &use : op->getResult(0).getUses()) {
      auto useOp = use.getOwner();
      if (auto useTpuOp = llvm::dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(useOp)) {
        assertTgOpLoadPlainAct<tpu::TG_INT8_PC_Conv2DOp>(useTpuOp);
      } else if (auto useTpuOp =
                 llvm::dyn_cast<tpu::TG_INT8_PT_Conv2DOp>(useOp)) {
        assertTgOpLoadPlainAct<tpu::TG_INT8_PT_Conv2DOp>(useTpuOp);
      } else if (auto useTpuOp = llvm::dyn_cast<tpu::TG_BF16_Conv2DOp>(useOp)) {
        assertTgOpLoadPlainAct<tpu::TG_BF16_Conv2DOp>(useTpuOp);
      } else if (auto useTpuOp =
                 llvm::dyn_cast<tpu::TG_INT8_EltwiseAddOp>(useOp)) {
        assertTgOpLoadPlainAct<tpu::TG_INT8_EltwiseAddOp>(useTpuOp);
      } else if (auto useTpuOp =
                 llvm::dyn_cast<tpu::TG_BF16_EltwiseAddOp>(useOp)) {
        assertTgOpLoadPlainAct<tpu::TG_BF16_EltwiseAddOp>(useTpuOp);
      } else if (auto useTpuOp =
                 llvm::dyn_cast<tpu::TL_LG_LoadNeuronOp>(useOp)) {
        assertTlLgLoadNeuronOpLoadPlainAct(useTpuOp);
      } else if (auto useTpuOp =
                 llvm::dyn_cast<tpu::TL_LW_Conv2DOp>(useOp)) {
        assertTlConvOpLoadPlainAct<tpu::TL_LW_Conv2DOp>(useTpuOp);
      }
    }
  }
}

void CompressActivationPass::addOpToCmprMaps() {
  getFunction().walk([&](Operation *op) {
    if (auto tpuOp = llvm::dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(op)) {
      addTgConvOpToCmprMap<tpu::TG_INT8_PC_Conv2DOp>(tpuOp);
    } else if (auto tpuOp = llvm::dyn_cast<tpu::TG_INT8_PT_Conv2DOp>(op)) {
      addTgConvOpToCmprMap<tpu::TG_INT8_PT_Conv2DOp>(tpuOp);
    } else if (auto tpuOp = llvm::dyn_cast<tpu::TG_BF16_Conv2DOp>(op)) {
      addTgConvOpToCmprMap<tpu::TG_BF16_Conv2DOp>(tpuOp);
    } else if (auto tpuOp = llvm::dyn_cast<tpu::TG_INT8_EltwiseAddOp>(op)) {
      addTgEltAddOpToCmprMap<tpu::TG_INT8_EltwiseAddOp>(tpuOp);
    } else if (auto tpuOp = llvm::dyn_cast<tpu::TG_BF16_EltwiseAddOp>(op)) {
      addTgEltAddOpToCmprMap<tpu::TG_BF16_EltwiseAddOp>(tpuOp);
    } else if (auto tpuOp = llvm::dyn_cast<tpu::TL_LG_LoadNeuronOp>(op)) {
      addTlLdLoadNeuronOpToCmprMap(op);
    } else if (auto tpuOp = llvm::dyn_cast<tpu::TL_LW_Conv2DOp>(op)) {
      addTlConvOpToCmprMap(tpuOp);
    }
  });
}

void CompressActivationPass::showMarkCompressed() {
  LLVM_DEBUG(llvm::dbgs() << "\n===============\n"
             << "showMarkCompressed\n");
  for (auto m : cmprMaps) {
    auto op = m.first;
    auto status = m.second;
    auto tpuOp = llvm::dyn_cast<tpu::TpuOpCommonInterface>(op);
    LLVM_DEBUG(llvm::dbgs()
        << tpuOp.getOpName() <<  ", " << op->getName()
        << ", store " << status.store << ", load " << status.load << "\n");
    showOpStatus(m.first);
  }

  LLVM_DEBUG(llvm::dbgs() << "===============\n\n");
}

void CompressActivationPass::markLoadDeCompressed() {
  LLVM_DEBUG(llvm::dbgs() << "\n===============\n"
             << "markLoadDeCompressed\n");

  // tg conv/tg elt-add
  // Remove store compressed first
  // All user should be tg conv, tg elt-add, tl_lg_load_neuron
  // dw-conv does not support load/store compression.
  for (auto &m : cmprMaps) {
    auto op = m.first;
    if (!isStoreCompressedOp(op))
      continue;

    if (!m.second.store)
      continue;

    bool result = true;
    for (auto &use : op->getResult(0).getUses()) {
      auto useOp = use.getOwner();

      if (!isLoadDecompressedOp(useOp)) {
        result = false;
        break;
      }

      if (isTgConvOp(useOp) && !isValidDecompressTgConvOp(useOp)) {
        result = false;
        break;
      }

      if (isTgEltAddOp(useOp) && !isValidDecompressTgEltAddOp(useOp)) {
        result = false;
        break;
      }

      // Turn off tg conv + tl load neuron temporaily
      // because erfnet hang up in evb
      if (isTgConvOp(op) && isLgLoadNeuronOp(useOp)) {
        result = false;
        break;
      }

    }

    if (!result) {
      m.second.store = 0;

      auto tpuOp = llvm::dyn_cast<tpu::TpuOpCommonInterface>(op);
      LLVM_DEBUG(llvm::dbgs()
                 << "  " << tpuOp.getOpName() << ", " << op->getName()
                 << ", store plain\n");
    }
  }

  // tg elt-add
  // Remove store compressed of input op.
  // Both inputs of elt add have to be compressed.
  for (auto &m : cmprMaps) {
    auto op = m.first;
    if (!isTgEltAddOp(op))
      continue;

    int load_cmpr_cnt = 0;
    std::vector<Operation *> opdOps;
    opdOps.clear();
    for (auto operand : op->getOperands()) {
      auto opdOp = operand.getDefiningOp();
      if (!opdOp)
        continue;

      if (!isStoreCompressedOp(opdOp))
        continue;

      opdOps.push_back(opdOp);
      load_cmpr_cnt = cmprMaps[opdOp].store ?
                      load_cmpr_cnt + 1 : load_cmpr_cnt;
    }

    if (load_cmpr_cnt != 2) {
      m.second.load = 0;

      {
        auto tpuOp = llvm::dyn_cast<tpu::TpuOpCommonInterface>(op);
        LLVM_DEBUG(llvm::dbgs()
                  << "  " << tpuOp.getOpName() << ", " << op->getName()
                  << ", load plain\n");
      }

      for (auto v : opdOps) {
        cmprMaps[v].store = 0;

        auto tpuOp = llvm::dyn_cast<tpu::TpuOpCommonInterface>(v);
        LLVM_DEBUG(llvm::dbgs()
                   << "    " << tpuOp.getOpName() << ", " << v->getName()
                   << ", opd store plain\n");
      }
    } else {
      m.second.load = 1;

      {
        auto tpuOp = llvm::dyn_cast<tpu::TpuOpCommonInterface>(op);
        LLVM_DEBUG(llvm::dbgs()
                  << "  " << tpuOp.getOpName() << ", " << op->getName()
                  << ", load cmpressed\n");
      }

      for (auto v : opdOps) {
        auto tpuOp = llvm::dyn_cast<tpu::TpuOpCommonInterface>(v);
        LLVM_DEBUG(llvm::dbgs()
                   << "    " << tpuOp.getOpName() << ", " << v->getName()
                   << ", opd store compressed\n");
      }

    }
  }

  // Mark user of tg conv, tg elt-add load compressed
  for (auto &m : cmprMaps) {
    auto op = m.first;
    if (!isStoreCompressedOp(op))
      continue;

    if (!m.second.store)
      continue;

    {
      auto tpuOp = llvm::dyn_cast<tpu::TpuOpCommonInterface>(op);
      LLVM_DEBUG(llvm::dbgs()
                << "  " << tpuOp.getOpName() << ", " << op->getName()
                << ", store compressed\n");
    }

    for (auto &use : op->getResult(0).getUses()) {
      auto useOp = use.getOwner();

      if (!isLoadDecompressedOp(useOp))
        continue;

      {
        auto tpuOp = llvm::dyn_cast<tpu::TpuOpCommonInterface>(useOp);
        LLVM_DEBUG(llvm::dbgs()
                  << "    " << tpuOp.getOpName() << ", " << useOp->getName()
                  << ", make user load decompressed\n");
      }

      cmprMaps[useOp].load = 1;
      cmprMaps[useOp].prevOp = op;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "===============\n\n");
}

void CompressActivationPass::analyze() {
  addOpToCmprMaps();
  markLoadDeCompressed();
  showMarkCompressed();
}

void CompressActivationPass::checkTgOpCompressStats() {
  getFunction().walk([&](Operation *op) {
    if (auto tpuOp = llvm::dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(op)) {
      checkTgOpStatus<tpu::TG_INT8_PC_Conv2DOp>(tpuOp);
    } else if (auto tpuOp = llvm::dyn_cast<tpu::TG_INT8_PT_Conv2DOp>(op)) {
      checkTgOpStatus<tpu::TG_INT8_PT_Conv2DOp>(tpuOp);
    } else if (auto tpuOp = llvm::dyn_cast<tpu::TG_BF16_Conv2DOp>(op)) {
      checkTgOpStatus<tpu::TG_BF16_Conv2DOp>(tpuOp);
    } else if (auto tpuOp = llvm::dyn_cast<tpu::TG_INT8_EltwiseAddOp>(op)) {
      checkTgOpStatus<tpu::TG_INT8_EltwiseAddOp>(tpuOp);
    } else if (auto tpuOp = llvm::dyn_cast<tpu::TG_BF16_EltwiseAddOp>(op)) {
      checkTgOpStatus<tpu::TG_BF16_EltwiseAddOp>(tpuOp);
    }
  });
}

void CompressActivationPass::runOnFunction() {
  OwningRewritePatternList patterns;
  MInfo mInfo;
  mInfo.getChipInfo(getFunction());
  assert(MInfo::version && "refer to set-chip");

  analyze();

  // Determine whether the operation can store compressed activation.
  patterns.insert<
        TgCompressActPattern<tpu::TG_INT8_PT_Conv2DOp>,
        TgCompressActPattern<tpu::TG_INT8_PC_Conv2DOp>,
        TgCompressActPattern<tpu::TG_INT8_EltwiseAddOp>,
        TgCompressActPattern<tpu::TG_BF16_Conv2DOp>,
        TgCompressActPattern<tpu::TG_BF16_EltwiseAddOp>
      >(&getContext(), cmprMaps);
  applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));

  // Determine whether the operation can load compressed activation.
  patterns.clear();
  patterns.insert<
        TgDecompressActPattern<tpu::TG_INT8_PT_Conv2DOp>,
        TgDecompressActPattern<tpu::TG_INT8_PC_Conv2DOp>,
        TgDecompressActPattern<tpu::TG_INT8_EltwiseAddOp>,
        TgDecompressActPattern<tpu::TG_BF16_Conv2DOp>,
        TgDecompressActPattern<tpu::TG_BF16_EltwiseAddOp>,
        TlLgLoadNeuronDecompressActPattern,
        TlConvDecompressActPattern<tpu::TL_LW_Conv2DOp>
      >(&getContext(), cmprMaps);
  applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));

  checkTgOpCompressStats();

  // Determine whether the tl store operations can store tiled compressed
  // activation
  patterns.clear();
  patterns.insert<
      TlLgJointCompressedActPattern
      >(&getContext(), mInfo);
  applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
}

std::unique_ptr<mlir::Pass> mlir::createCompressActivationPass() {
  return std::make_unique<CompressActivationPass>();
}
