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

#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/Dialect/TPU/TPUCompressUtil.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
#include "mlir/Dialect/TPU/MachineInfo.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "compress-activation"

using namespace mlir;

namespace {

template <typename OpTy>
uint64_t calcConv2DMemoryUsage(OpTy &op) {
  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, pt, pb, pl, pr, dh, dw;
  bool is_deconv = isa<tpu::TG_INT8_PC_DeConv2DOp>(op.getOperation());
  parseConvParam(op.param(), is_deconv, op.input(), op.output(), op.filter(),
                 n, ic, ih, iw, oc, oh, ow, g,
                 kh, kw, sh, sw, pt, pb, pl, pr, dh, dw, is_dw, with_bias, do_relu);
  uint64_t inputNeuronSizePerLane = MInfo::getSizePerLane(n, ic, ih, iw, true);
  uint64_t outputNeuronSizePerLane = MInfo::getSizePerLane(n, oc, oh, ow, true);
  uint64_t filterSizePerLane = 0;
  // filter working size *2 for double buffer
  if (g != oc) {
    if(g != 1) { // TODO, not support group convolution now.
      return MInfo::lmem_per_lane + 1;
    }
    filterSizePerLane = MInfo::getSizePerLane(ic, oc, kh, kw, false) ;
  }

  // load bias all in once
  int bias_size = with_bias ? 9 : 5;
  uint64_t biasSizePerLane = MInfo::getSizePerLane(1, oc, 1, bias_size, false);

  // total
  uint64_t totalPerLane = inputNeuronSizePerLane + outputNeuronSizePerLane +
                          filterSizePerLane + biasSizePerLane;

  return totalPerLane;
}

bool canStoreCompressedActivation(Operation *op) {
  bool storeCompressed= false;

  // Check if all user operation does not need to do tiling.
  // Only support conv->conv now.
  for (auto &use : op->getResult(0)->getUses()) {
    uint64_t totalPerLane = MInfo::lmem_per_lane + 1;
    auto useOp = use.getOwner();
    if (auto useTpuOp = dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(useOp)) {
      totalPerLane =
          calcConv2DMemoryUsage<tpu::TG_INT8_PC_Conv2DOp>(useTpuOp);
    } else {
      storeCompressed = false;
      break;
    }

    if (totalPerLane <= MInfo::lmem_per_lane)
      storeCompressed = true;
    else {
      storeCompressed = false;
      break;
    }
  }

  return storeCompressed;
}

bool canLoadCompressedActivation(Operation *op) {

  // Check if input operation store compressed activation.
  // Only support conv->conv now.
  for (auto operand : op->getOperands()) {
    auto operandOp = operand->getDefiningOp();
    if (auto operandTpuOp = dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(operandOp)) {
      if (operandTpuOp.store_compr_act().hasValue())
        return true;
    }
  }

  return false;
}

template <typename OpTy>
class StoreCompressedConvActPattern
    : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  StoreCompressedConvActPattern(MLIRContext *ctx)
      : OpRewritePattern<OpTy>(ctx) {}

  PatternMatchResult matchAndRewrite(OpTy convOp,
                                     PatternRewriter &rewriter) const override {
    // Already marked.
    if (convOp.store_compr_act().hasValue())
      return Pattern::matchFailure();

    uint64_t totalPerLane = calcConv2DMemoryUsage<OpTy>(convOp);
    if (totalPerLane > MInfo::lmem_per_lane)
      return Pattern::matchFailure();

    // A operation needs to generate the compressed activation first then
    // the user operation has to load the compressed activation.
    auto op = convOp.getOperation();
    if (!canStoreCompressedActivation(op))
      return Pattern::matchFailure();

    convOp.setAttr("store_compr_act",
                   Builder(op->getContext()).getBoolAttr(true));

    LLVM_DEBUG(llvm::dbgs()
               << "StoreCompressedConvActPattern: op "
               << convOp.name()
               << ", layer ID " << getOpLayerId(op)
               << ", store compressed activation\n");

    return Pattern::matchSuccess();
  }
};

template <typename OpTy>
class LoadCompressedConvActivationPattern
    : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LoadCompressedConvActivationPattern(MLIRContext *ctx)
      : OpRewritePattern<OpTy>(ctx) {}

  PatternMatchResult matchAndRewrite(OpTy convOp,
                                     PatternRewriter &rewriter) const override {
    // Already marked.
    if (convOp.load_compr_act().hasValue())
      return Pattern::matchFailure();

    auto op = convOp.getOperation();
    if (!canLoadCompressedActivation(op))
      return Pattern::matchFailure();

    convOp.setAttr("load_compr_act",
                   Builder(op->getContext()).getBoolAttr(true));

    LLVM_DEBUG(llvm::dbgs()
               << "LoadCompressedConvActivationPattern: op "
               << convOp.name()
               << ", layer ID " << getOpLayerId(op)
               << ", load compressed activation\n");

    return Pattern::matchSuccess();
  }
};

template<typename OpTy>
class StoreTlCompressedActPattern
    : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  StoreTlCompressedActPattern(MLIRContext *ctx)
      : OpRewritePattern<OpTy>(ctx) {}

  PatternMatchResult matchAndRewrite(OpTy tpuOp,
                                     PatternRewriter &rewriter) const override {

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
        << "operand " << getOpName(op->getOperand(0)->getDefiningOp())
        << ", " << op->getOperand(0)->getDefiningOp()->getName()
        << "\n");

    for (auto &use : op->getResult(0)->getUses()) {
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

      for (auto *operand : useOp->getOperands()) {
        std::vector<int64_t> inputShapes = getTensorShape(operand->getDefiningOp()->getOperand(0));
        std::vector<int64_t> outputShapes = getTensorShape(operand->getDefiningOp()->getResult(0));
        LLVM_DEBUG(llvm::dbgs()
            << "    operand " << operand->getDefiningOp()->getName()
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

      for (auto &nextUse : useOp->getResult(0)->getUses()) {
        auto nextUseOp = nextUse.getOwner();
        std::vector<int64_t> outputShapes = getTensorShape(nextUseOp->getResult(0));

        LLVM_DEBUG(llvm::dbgs()
            << "    nextUserOp " << getOpName(nextUseOp)
            << ", " << nextUseOp->getName()
            << ", -> (" << outputShapes[0]
            << ", " << outputShapes[1]
            << ", " << outputShapes[2]
            << ", " << outputShapes[3]
            << ")\n");

        for (auto nextUseOpOperand : nextUseOp->getOperands()) {
          std::vector<int64_t> inputShapes = getTensorShape(nextUseOpOperand);
          LLVM_DEBUG(llvm::dbgs()
              << "      operand " << nextUseOpOperand->getDefiningOp()->getName()
              << ", (" << inputShapes[0]
              << ", " << inputShapes[1]
              << ", " << inputShapes[2]
              << ", " << inputShapes[3]
              << ")\n");
        }
      }
    }

    return Pattern::matchFailure();
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

  for (auto &use : op->getResult(0)->getUses()) {
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

  PatternMatchResult matchAndRewrite(tpu::TL_LG_JoinOp tpuOp,
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

    std::vector<std::vector<int64_t>> storeShapes;
    std::vector<std::vector<int64_t>> loadShapes;
    for (auto *operand : op->getOperands()) {
      auto opdOp = operand->getDefiningOp();
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
    for (auto &use : op->getResult(0)->getUses()) {
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
      return Pattern::matchFailure();

    if (!isValidLoadCompressActForTlLgJoin(op, oh_step))
      return Pattern::matchFailure();

    auto enableStoreCmprAct = [&](Operation *op, int64_t &offset) {
    if (auto tpuOp = llvm::dyn_cast<tpu::TL_LG_StoreOp>(op)) {
        LLVM_DEBUG(llvm::dbgs()
            << "      " << getOpName(op)
            << ", offset " << tpuOp.offset().getValue().getSExtValue()
            << " -> " << offset << "\n");

        tpuOp.removeAttr("offset");
        tpuOp.setAttr("offset",
                      Builder(op->getContext()).getI64IntegerAttr(offset));

        auto shapes = getTensorShape(op->getOperand(0));
        offset += step_size * shapes[2] / oh_step;

        tpuOp.setAttr("store_compr_act",
                      Builder(tpuOp.getContext()).getBoolAttr(true));
        tpuOp.setAttr("compr_act_param",
            tpu::ActCmprParam::get(
                Builder(op->getContext()).getI32IntegerAttr(n_step),
                Builder(op->getContext()).getI32IntegerAttr(oc_step),
                Builder(op->getContext()).getI32IntegerAttr(oh_step),
                Builder(op->getContext()).getI32IntegerAttr(step_size),
                Builder(op->getContext()).getI32IntegerAttr(total_size),
                rewriter.getContext()));
      }
    };

    auto enableLoadCmprAct = [&](Operation *op) {
      if (auto tpuOp = llvm::dyn_cast<tpu::TL_LG_LoadNeuronOp>(op)) {
        std::vector<int64_t> resShapes = getTensorShape(op->getResult(0));
        int64_t hOffset = tpuOp.offset().getValue().getSExtValue() / resShapes[3];
        // assert(hOffset < resShapes[2]);
        int64_t offset = step_size * hOffset;
        LLVM_DEBUG(llvm::dbgs()
            << "      " << getOpName(op)
            << ", offset " << tpuOp.offset().getValue().getSExtValue()
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
                Builder(op->getContext()).getI32IntegerAttr(step_size),
                Builder(op->getContext()).getI32IntegerAttr(total_size),
                rewriter.getContext()));
      } else if (auto tpuOp = llvm::dyn_cast<tpu::TL_LW_Conv2DOp>(op)) {
        tpuOp.setAttr("load_compr_act",
                      Builder(tpuOp.getContext()).getBoolAttr(true));
        tpuOp.setAttr("load_compr_act_param",
            tpu::ActCmprParam::get(
                Builder(op->getContext()).getI32IntegerAttr(n_step),
                Builder(op->getContext()).getI32IntegerAttr(oc_step),
                Builder(op->getContext()).getI32IntegerAttr(oh_step),
                Builder(op->getContext()).getI32IntegerAttr(step_size),
                Builder(op->getContext()).getI32IntegerAttr(total_size),
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
    for (auto *operand : op->getOperands()) {
      auto tiuOp = operand->getDefiningOp()->getOperand(0)->getDefiningOp();

      LLVM_DEBUG(llvm::dbgs()
        << "  Mark cmpr_store, tiuOp " << getOpName(tiuOp)
        << ", " << tiuOp->getName()
        << "\n");

      // Mark lg_store
      enableStoreCmprAct(operand->getDefiningOp(), store_offset);

      // Mark all user of pre tiu op
      for (auto &use : tiuOp->getResult(0)->getUses()) {
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
    for (auto &use : op->getResult(0)->getUses()) {
      auto useOp = use.getOwner();

      LLVM_DEBUG(llvm::dbgs()
        << "  Mark cmpr_load, useOp " << getOpName(useOp)
        << ", " << useOp->getName()
        << "\n");

      enableLoadCmprAct(useOp);
      index++;
    }

    return Pattern::matchFailure();
  }

  MInfo &mInfo;
};

struct CompressActivationPass : public FunctionPass<CompressActivationPass> {
  void runOnFunction() override;
};

} // anonymous namespace

void CompressActivationPass::runOnFunction() {
  OwningRewritePatternList patterns;
  std::string getRunChipType;
  MInfo mInfo;
  get_cvichip_name(getRunChipType);
  mInfo.getChipInfo(getRunChipType.c_str());
  assert(MInfo::version && "refer to set-chip");

  // Determine whether the operation can store compressed activation.
  patterns.insert<
      StoreCompressedConvActPattern<tpu::TG_INT8_PC_Conv2DOp>
      >(&getContext());
  applyPatternsGreedily(getFunction(), patterns);

  // Determine whether the operation can load compressed activation.
  patterns.clear();

  patterns.insert<
      LoadCompressedConvActivationPattern<tpu::TG_INT8_PC_Conv2DOp>
      >(&getContext());
  applyPatternsGreedily(getFunction(), patterns);

  // Determine whether the tl store operations can store tiled compressed
  // activation
  patterns.clear();

  patterns.insert<
      TlLgJointCompressedActPattern
      >(&getContext(), mInfo);
  applyPatternsGreedily(getFunction(), patterns);
}

std::unique_ptr<OpPassBase<FuncOp>> mlir::createCompressActivationPass() {
  return std::make_unique<CompressActivationPass>();
}

static PassRegistration<CompressActivationPass>
    pass("compress-activation", "Compress activation");
