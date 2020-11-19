//===- AssignNeuronAddress.cpp - assigned neuron address ------------------===//
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
// This file assined neuron address
//
//===----------------------------------------------------------------------===//

#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUCompressUtil.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tpuc/Support/TensorFile.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"

#define DEBUG_TYPE "group_ops"

// this pass will do two kinds of optimization between two
// group
// case 1: laddr_a and laddr_b is equal
//    group 0:
//    tl_store gaddr_a, laddr_a
//    tl_join gaddr_a
//    group 1:
//    tl_load laddr_b, gaddr_a
// optimize to =======>
//    delete tl_store/tl_join/tl_load
//
//
// case 2: laddr_a and laddr_b is not equal
//    group 0:
//    tl_store gaddr_a, laddr_a
//    tl_join gaddr_a
//    group 1:
//    tl_load laddr_b, gaddr_a
// optimize to ========>
//    tl_copy laddr_b, laddr_a

// two group should be tl group and have no n/h slice
// This optimization works well for small network, for example
// efficientnet_b0:
//     310 fps ===> 346 fps
// resnet18:
//     247.5 fps ===> 251.9 fps

namespace mlir {

static tpu::TL_LG_CopyOp build_tl_copy_op(PatternRewriter &rewriter,
                                          Operation * src,
                                          Operation * dst) {
  std::vector<NamedAttribute> attrs;

  // build tl_copy instruction
  auto src_op = dyn_cast<tpu::TL_LG_StoreOp>(src);
  auto dst_op = dyn_cast<tpu::TL_LG_LoadNeuronOp>(dst);

  attrs.push_back(rewriter.getNamedAttr("name", dst_op.nameAttr()));
  attrs.push_back(rewriter.getNamedAttr("la_src", src_op.laddrAttr()));
  attrs.push_back(rewriter.getNamedAttr("la_dst", dst_op.laddrAttr()));
  attrs.push_back(rewriter.getNamedAttr("align", dst_op.alignAttr()));

  std::vector<Value> operands;
  operands.push_back(src->getOperand(0));

  // build tl_load operation
  auto op = rewriter.create<tpu::TL_LG_CopyOp>(dst->getLoc(),
          dst_op.getType(), ArrayRef<Value>{operands},
          ArrayRef<NamedAttribute>{attrs});

  return op;
}

// eliminate the redundant tmda operations
struct EliminateDeadcodePattern : public RewritePattern {
  EliminateDeadcodePattern(MLIRContext *context)
      : RewritePattern("tpu.tl_lg_join", 1, context),
        ctx_(context) {}

  LogicalResult
      matchAndRewrite(Operation *op,
                     PatternRewriter &rewriter) const override {
    Operation * tl_join = op;
    Operation * load_op;
    if (tl_join->getNumOperands() != 1)
      return failure();
    Operation * store_op = tl_join->getOperand(0).getDefiningOp();
    auto tl_store = dyn_cast<tpu::TL_LG_StoreOp>(store_op);
    if (!tl_store)
      return failure();

    if (!(tl_join->getResult(0).hasOneUse()))
      return failure();
    for (auto &use : op->getResult(0).getUses()) {
      load_op = use.getOwner();
      break;
    }
    // should be join + load, or return fail
    Operation * join_next = tl_join->getNextNode();
    if (join_next != load_op)
      return failure();

    auto tl_load = dyn_cast<tpu::TL_LG_LoadNeuronOp>(load_op);
    if (!tl_load) {
      return failure();
    }

    // check tl_store and tl_load's local memory address.
    int32_t store_laddr = tl_store.laddr().getValue();
    int32_t load_laddr = tl_load.laddr().getValue();
    // if store and load share the same laddr, delete
    // load, store and join op
    if (store_laddr == load_laddr) {
      tl_load.replaceAllUsesWith(tl_store.getOperand().getDefiningOp());
    } else {
      // set disable_parallel to previous op
      Operation * prev_op = store_op->getPrevNode();
      auto prev_tl_op = llvm::dyn_cast<tpu::TpuTLOpCodegenInterface>(prev_op);
      if (tl_store.getDisableParallel())
        prev_tl_op.setDisableParallel(true);
      // set enable_parallel to next op
      Operation * next_op = load_op->getPrevNode();
      auto next_tl_op = llvm::dyn_cast<tpu::TpuTLOpCodegenInterface>(next_op);
      if (tl_load.getEnableParallel())
        next_tl_op.setEnableParallel(true);

      // build tl copy
      auto tl_copy_op = build_tl_copy_op(rewriter, store_op, load_op);
      tl_load.replaceAllUsesWith(tl_copy_op.getResult());
    }
    rewriter.eraseOp(tl_load);
    return success();
  }

  MLIRContext * ctx_;
};


class EliminateDeadcodePass : public mlir::PassWrapper<EliminateDeadcodePass, FunctionPass> {
public:
  explicit EliminateDeadcodePass() {}

  void runOnFunction() override {
    auto fn = getFunction();
    auto *context = &getContext();

    OwningRewritePatternList patterns;
    patterns.clear();
    patterns.insert<
        EliminateDeadcodePattern
        >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));
  }
};

std::unique_ptr<mlir::Pass> createEliminateDeadcodePass() {
  return std::make_unique<EliminateDeadcodePass>();
}

static PassRegistration<EliminateDeadcodePass>
    pass("dce",
         "Dead Code Eliminate such as tdma");

}

