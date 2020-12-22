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

// Support:
//   tg conv -> tg conv
//
// TODO:
//   multi-batch
///  group conv
//   dw-conv
//   tg_conv -> tg_conv/tg_elt_add (multiple users) via template variadic
//   tg_conv -> load_neuron
//   tg_conv -> tl_lw_conv
template <typename OpTy, typename NextOpTy>
class TgConvCompressedActPattern
    : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  TgConvCompressedActPattern(MLIRContext *ctx)
      : OpRewritePattern<OpTy>(ctx) {}

  LogicalResult matchAndRewrite(OpTy convOp,
                                     PatternRewriter &rewriter) const override {
    // Already marked.
    if (convOp.store_compr_act().hasValue())
      return failure();

    // tg dw-conv not integrated with tg conv
    if (convOp.param().is_dw().getValue())
      return failure();

    // Not support group conv
    if (convOp.param().group().getInt() > 1)
      return failure();

    if (!convOp.tile_param().hasValue())
      return failure();

    auto op = convOp.getOperation();
    std::vector<int64_t> outputShapes = getTensorShape(op->getResult(0));
    int batch = outputShapes[0];
    int outputChannel = outputShapes[1];
    int outputHeight = outputShapes[2];
    int outputWidth = outputShapes[3];
    int cmprNStep = 1;
    int cmprOcStep = MInfo::lane_num;
    int cmprOhStep = 1;
    int isBf16 = isBf16Tensor(op->getResult(0)) ? 1 : 0;
    int64_t stepSize = 0, totalSize = 0;
    getTiledCompressedSize(batch, outputChannel, outputHeight, outputWidth,
                           cmprNStep, cmprOcStep, cmprOhStep,
                           isBf16, stepSize, totalSize);

    if (getOpLayerId(op) == 167) {
      LLVM_DEBUG(llvm::dbgs()
          << "StoreTgConvCmprAct: layer ID " << getOpLayerId(op)
          << "" << getOpName(op)
          << ", (" << batch << ", " << outputChannel 
          << ", " << outputHeight << ", " << outputWidth << "), "
          << ", stepSize " << stepSize << "\n");

      getTiledCompressedSize(batch, outputChannel, outputHeight, outputWidth,
                            cmprNStep, cmprOcStep, cmprOhStep,
                            isBf16, stepSize, totalSize);

    }

    // Too trivial
    if (outputHeight == 1 || outputChannel < (int)MInfo::lane_num)
      return failure();

    // Cannot compress if tiling in width
    if (convOp.tile_param().getValue().oh_step().getInt() < outputWidth)
      return failure();

    // Not support multi-batch
    if (batch > 1)
      return failure();

    // tg_conv -> tg_conv, exclude dw-conv, group conv
    for (auto &use : op->getResult(0).getUses()) {
      auto useOp = use.getOwner();
      if (auto nextConvOp = llvm::dyn_cast<NextOpTy>(useOp)) {
        if (nextConvOp.param().is_dw().getValue())
        return failure();
        if (nextConvOp.param().group().getInt() > 1)
          return failure();
      } else
        return failure();
    }

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

    for (auto &use : op->getResult(0).getUses()) {
      auto useOp = use.getOwner();
      auto nextConvOp = llvm::dyn_cast<NextOpTy>(useOp);

      LLVM_DEBUG(llvm::dbgs()
          << "StoreTgConvCmprAct: layer ID " << getOpLayerId(op)
          << "" << getOpName(op)
          << ", " << op->getName() << ", store compressed, next op "
          << getOpName(nextConvOp.getOperation())
          << ", " << nextConvOp.getOperation()->getName()
          << ", load compressed\n");

      nextConvOp.setAttr("load_compr_act",
                         Builder(op->getContext()).getBoolAttr(true));
      nextConvOp.setAttr("load_compr_act_param",
          tpu::ActCmprParam::get(
              Builder(op->getContext()).getI32IntegerAttr(cmprNStep),
              Builder(op->getContext()).getI32IntegerAttr(cmprOcStep),
              Builder(op->getContext()).getI32IntegerAttr(cmprOhStep),
              Builder(op->getContext()).getI64IntegerAttr(stepSize),
              Builder(op->getContext()).getI64IntegerAttr(totalSize),
              rewriter.getContext()));
    }

    return success();
  }
};

template<typename OpTy>
class StoreTlCompressedActPattern
    : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  StoreTlCompressedActPattern(MLIRContext *ctx)
      : OpRewritePattern<OpTy>(ctx) {}

  LogicalResult matchAndRewrite(OpTy tpuOp,
                                     PatternRewriter &rewriter) const override {

    if (!tpuOp.tl_store_flag())
      return failure();

    Operation *op = tpuOp.getOperation();

    std::vector<int64_t> inputShapes = getTensorShape(op->getOperand(0));
    std::vector<int64_t> outputShapes = getTensorShape(op->getResult(0));
    LLVM_DEBUG(llvm::dbgs()
        << "\nStoreTlCmprAct: " << getOpName(op)
        << ", " << op->getName()
        << ", (" << inputShapes[0]
        << ", " << inputShapes[1]
        << ", " << inputShapes[2]
        << ", " << inputShapes[3]
        << ") -> (" << outputShapes[0]
        << ", " << outputShapes[1]
        << ", " << outputShapes[2]
        << ", " << outputShapes[3] << ")\n  "
        << "operand " << getOpName(op->getOperand(0).getDefiningOp())
        << ", " << op->getOperand(0).getDefiningOp()->getName()
        << ", tl_store_flag " << tpuOp.tl_store_flag()
        << "\n");

    // Not support multi-batch yet
    if (outputShapes[0] > 1)
      return failure();

    for (auto &use : op->getResult(0).getUses()) {
      auto useOp = use.getOwner();
      std::vector<int64_t> inputShapes = getTensorShape(useOp->getOperand(0));
      std::vector<int64_t> outputShapes = getTensorShape(useOp->getResult(0));

      LLVM_DEBUG(llvm::dbgs()
          << "  userOp " << getOpName(useOp)
          << ", " << useOp->getName()
          << ", operand[0](" << inputShapes[0]
          << ", " << inputShapes[1]
          << ", " << inputShapes[2]
          << ", " << inputShapes[3]
          << ") -> (" << outputShapes[0]
          << ", " << outputShapes[1]
          << ", " << outputShapes[2]
          << ", " << outputShapes[3]
          << ")\n");
    }

    return failure();
  }

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
};

} // anonymous namespace

void CompressActivationPass::runOnFunction() {
  OwningRewritePatternList patterns;
  MInfo mInfo;
  mInfo.getChipInfo(getFunction());
  assert(MInfo::version && "refer to set-chip");

  // Determine whether the operation can store compressed activation.
  patterns.insert<
      TgConvCompressedActPattern<tpu::TG_INT8_PT_Conv2DOp, tpu::TG_INT8_PT_Conv2DOp>,
      TgConvCompressedActPattern<tpu::TG_INT8_PC_Conv2DOp, tpu::TG_INT8_PC_Conv2DOp>,
      TgConvCompressedActPattern<tpu::TG_BF16_Conv2DOp, tpu::TG_BF16_Conv2DOp>
      >(&getContext());
  applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));

  // Determine whether the tl store operations can store tiled compressed
  // activation
  patterns.clear();

  patterns.insert<
      TlLgJointCompressedActPattern
      >(&getContext(), mInfo);
  applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));

  patterns.clear();
  patterns.insert<
      StoreTlCompressedActPattern<tpu::TL_LW_Conv2DOp>
      >(&getContext());
  applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
}

std::unique_ptr<mlir::Pass> mlir::createCompressActivationPass() {
  return std::make_unique<CompressActivationPass>();
}
