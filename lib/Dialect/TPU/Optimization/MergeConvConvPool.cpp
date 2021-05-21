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
// This file implements the TPU dialect OP Stats pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Passes.h"
#include "tpuc/Support/TensorFile.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "llvm/Support/raw_ostream.h"
#include "tpuc/MachineInfo.h"

#define DEBUG_TYPE "eltwise_early_stride"

using namespace mlir;

namespace {

template <typename OpTy>
struct MergeConvConvPoolOpPattern : public RewritePattern {
  MergeConvConvPoolOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  int32_t calcFilterAndBiasSize(Operation *op) const {
    auto conv_ = dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(op);
    bool is_dw, with_bias, do_relu;
    int n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw;
    int pt, pb, pl, pr, dh, dw, pad_value;
    parseConvParam(conv_.param(), false, conv_.input(), conv_.output(),
                   conv_.filter(), n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h,
                   ins_w, sh, sw, pt, pb, pl, pr, dh, dw, is_dw, with_bias,
                   do_relu, pad_value);

    uint64_t filterSizePerLane = 0;
    assert(ic < 4096);
    assert(g == 1);
    filterSizePerLane = MInfo::getSizePerLane(ic, oc, kh, kw, false);

    // load bias all in once
    int bias_size = with_bias ? 9 : 5;
    uint64_t biasSizePerLane =
        MInfo::getSizePerLane(1, oc, 1, bias_size, false);
    return filterSizePerLane + biasSizePerLane;
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    llvm::errs() << "here 1\n";
    std::vector<Operation *> ops;
    if (!op->hasOneUse()) { // conv
      return failure();
    }
    ops.push_back(op);
    llvm::errs() << "here 2\n";
    auto nextOp = getNextOp(op);
    if (!dyn_cast_or_null<tpu::TG_INT8_PC_Conv2DOp>(nextOp)) {
      return failure();
    }
    llvm::errs() << "here 3\n";
    if (!nextOp->hasOneUse()) { // conv
      return failure();
    }
    ops.push_back(nextOp);
    nextOp = getNextOp(nextOp);
    llvm::errs() << "here 4\n";
    if (!dyn_cast_or_null<tpu::TG_INT8_PoolMax2DOp>(nextOp)) {
      return failure();
    }
    ops.push_back(nextOp);
    llvm::errs() << "here 5\n";

    int32_t pin_lmem_size = 0;
    for (auto op_ : ops) {
      if (isa<tpu::TG_INT8_PC_Conv2DOp>(op_)) {
        pin_lmem_size += calcFilterAndBiasSize(op_);
        llvm::errs() << "pin_lmem_size:" << pin_lmem_size << "\n";
      }
    }
    int free_lmem_size = (int)MInfo::lmem_per_lane - pin_lmem_size;
    if (free_lmem_size <= 0) {
      return failure();
    }
    llvm::errs() << "free_lmem_size:" << free_lmem_size << "\n";

    auto last_ = ops[ops.size() - 1];
    auto shape = getTensorShape(last_->getResult(0));
    for (int ow_step = shape[3]; ow_step > 0; --ow_step) {
      for (int oh_step = shape[2]; oh_step > 0; --oh_step) {
        for (int on_step = shape[0]; on_step > 0; --on_step) {
          int cur_on = on_step;
          int cur_oh = oh_step;
          int cur_ow = ow_step;

          int max_ifmap = 0;
          int max_ofmap = 0;

          for (int i = (int)ops.size() - 1; i >= 0; --i) {
            auto op_ = ops[i];
            int cur_ih = 0;
            int cur_iw = 0;
            if (auto pool_ = dyn_cast<tpu::TG_INT8_PoolMax2DOp>(op_)) {
              Pool2DParamParser p(pool_);
              auto cur_ih = (cur_oh - 1) * p.sh + p.kh;
              auto cur_iw = (cur_ow - 1) * p.sw + p.kw;
              if (cur_ih > p.ih) {
                cur_ih = p.ih;
              }
              if (cur_iw > p.iw) {
                cur_iw = p.iw;
              }
              int ifmap = MInfo::getSizePerLane(cur_on, p.c, cur_ih, cur_iw, true);
              int ofmap = MInfo::getSizePerLane(cur_on, p.c, cur_oh, cur_ow, true);
              max_ifmap = ifmap > max_ifmap ? ifmap : max_ifmap;
              max_ofmap = ofmap > max_ofmap ? ofmap : max_ofmap;
              llvm::errs() << "poolmax cur oh:" << cur_oh << " ow:" << cur_ow << " ih:"
                           << cur_ih << " iw: " << cur_iw << "\n"
                           << " ifmap:" << ifmap << " ofmap:" << ofmap << "\n";
            } else if (auto conv_ = dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(op_)) {
              Conv2DParamParser p(conv_);
              if (p.dh > 1) {
                p.kh = p.dh * (p.kh - 1) + 1;
              }
              if (p.dw > 1) {
                p.kw = p.dw * (p.kw - 1) + 1;
              }
              auto cur_ih = (cur_oh - 1) * p.sh + p.kh;
              auto cur_iw = (cur_ow - 1) * p.sw + p.kw;
              int ifmap = MInfo::getSizePerLane(cur_on, p.ic, cur_ih, cur_iw, true);
              int ofmap = MInfo::getSizePerLane(cur_on, p.oc, cur_oh, cur_ow, true);
              max_ifmap = ifmap > max_ifmap ? ifmap : max_ifmap;
              max_ofmap = ofmap > max_ofmap ? ofmap : max_ofmap;
              llvm::errs() << "conv cur oh:" << cur_oh << " ow:" << cur_ow << " ih:"
                           << cur_ih << " iw: " << cur_iw
                           << " ifmap:" << ifmap << " ofmap:" << ofmap << "\n";
            }
            cur_oh = cur_ih;
            cur_ow = cur_iw;
          }
          llvm::errs() << "max_ifmap:" << max_ifmap
                       << " max_ofmap:" << max_ofmap << "\n";
          if (max_ifmap + max_ofmap <= free_lmem_size) {
            llvm::errs() << "on_step:" << on_step << " oh_step:" << oh_step
                         << " ow_step:" << ow_step << "\n";
            goto success;
          }
        }
      }
    }
    return failure();
  success:
    llvm::errs() << "find matched pattern\n";
    auto conv_0_ = dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(ops[0]);
    auto conv_1_ = dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(ops[1]);
    auto pool_ = dyn_cast<tpu::TG_INT8_PoolMax2DOp>(ops[2]);
    std::vector<Value> operands;
    operands.push_back(conv_0_.getOperand(0));
    operands.push_back(conv_0_.getOperand(1));
    operands.push_back(conv_0_.getOperand(2));
    operands.push_back(conv_1_.getOperand(1));
    operands.push_back(conv_1_.getOperand(2));
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("name", pool_.nameAttr()));
    attrs.push_back(rewriter.getNamedAttr("param_0", conv_0_.paramAttr()));
    attrs.push_back(rewriter.getNamedAttr("param_1", conv_1_.paramAttr()));

    rewriter.setInsertionPointAfter(pool_);
    auto newOp = rewriter.create<tpu::TG_INT8_MergeConvConvPoolOp>(
      pool_.getLoc(), pool_->getResult(0).getType(),
      ArrayRef<Value>{operands},
      ArrayRef<NamedAttribute>{attrs});
    pool_.getResult().replaceAllUsesWith(newOp.getResult());
    rewriter.eraseOp(pool_);
    rewriter.eraseOp(conv_1_);
    rewriter.eraseOp(conv_0_);
    return success();
  }
};

class MergeConvConvPoolOpPass
    : public mlir::PassWrapper<MergeConvConvPoolOpPass, FunctionPass> {
public:
  void runOnFunction() override {
    auto fn = getFunction();
    MInfo machineInfo;
    machineInfo.getChipInfo(fn);
    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.clear();
    patterns.insert<
        MergeConvConvPoolOpPattern<tpu::TG_INT8_PC_Conv2DOp>
      >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::createMergeConvConvPoolPass() {
  return std::make_unique<MergeConvConvPoolOpPass>();
}
