//===- TpuOpStats.cpp - Implementation of TPU Op Stats ---------===//
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

#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"


using namespace mlir;

namespace {

struct TpuRefactorEltAndConvPattern : public RewritePattern {
  TpuRefactorEltAndConvPattern(MLIRContext *context)
      : RewritePattern("tpu.eltwise_add", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto eltAddOp = cast<tpu::EltwiseAddOp>(op);
    llvm::errs() << eltAddOp.getOperationName() << ":" << getOpName(eltAddOp)<< "\n";

    bool isKernel1x1AndStrideBiggerThanOne = true;
    Operation *nextOp = nullptr;
    int strideH, strideW;

    for (auto &use : op->getResult(0)->getUses()) {
        nextOp = use.getOwner();
        // llvm::errs() << nextOp->getName() << "\n";
        // if (matchPattern(nextOp, m_Op<tpu::Conv2DOp>())) {
        if (auto convOp = dyn_cast<tpu::Conv2DOp>(nextOp)) {
            // auto allocOp = dyn_cast<AllocOp>(opInst)
            auto filter_type = convOp.filter()->getType().template cast<TensorType>();
            std::vector<int64_t> f_s(filter_type.getShape());
            int kh = f_s[f_s.size() - 2];
            int kw = f_s[f_s.size() - 1];
            strideH = convOp.param().stride_h().getValue().getLimitedValue();
            strideW = convOp.param().stride_w().getValue().getLimitedValue();
            // llvm::errs() << convOp.getOperationName() << ":" << getOpName(convOp)<< "\n";
            if((kh == 1) && (kw == 1) && (strideH > 1) && (strideW > 1)) {
                // llvm::errs() << "Find \n";
                //do nothing
            } else {
                isKernel1x1AndStrideBiggerThanOne = false;
            }
        } else {
            isKernel1x1AndStrideBiggerThanOne = false;
        }
    }

    if(isKernel1x1AndStrideBiggerThanOne) {
        llvm::errs() << "Refactor elt and conv" << "\n";
        for (auto &use : op->getResult(0)->getUses()) { //Refactor convOp
            nextOp = use.getOwner();
            auto convOp = dyn_cast<tpu::Conv2DOp>(nextOp);
            convOp.setAttr("param",
                tpu::ConvParam::get(
                    rewriter.getI32IntegerAttr(1),
                    rewriter.getI32IntegerAttr(1),
                    convOp.param().padding(),
                    convOp.param().dilation_h(),
                    convOp.param().dilation_w(),
                    convOp.param().group(),
                    convOp.param().is_dw(),
                    convOp.param().with_bias(),
                    convOp.param().do_relu(),
                    rewriter.getContext()));//rewrite strideH
        }
        auto shape = eltAddOp.output()->getType().cast<TensorType>().getShape();//Refactor eltOp
        int on = shape[0];
        int oc = shape[1];
        assert(shape[2] % strideH == 0);
        assert(shape[3] % strideW == 0);
        int oh = shape[2] / strideH;
        int ow = shape[3] / strideW;

        eltAddOp.setAttr("do_early_stride", rewriter.getBoolAttr(true));
        eltAddOp.setAttr("early_stride_h", rewriter.getI32IntegerAttr(strideH));
        eltAddOp.setAttr("early_stride_w", rewriter.getI32IntegerAttr(strideW));
        auto type = RankedTensorType::get({on, oc, oh, ow}, FloatType::getF32(rewriter.getContext()));
        eltAddOp.getResult()->setType(type);//rewrite inputShape
    }
    return matchFailure();
  }
};

class RefactorEltAndConvPass : public FunctionPass<RefactorEltAndConvPass> {
public:
  explicit RefactorEltAndConvPass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();
    OwningRewritePatternList patterns;
    auto *context = &getContext();

    patterns.clear();

    patterns.insert<TpuRefactorEltAndConvPattern>(context);
    applyPatternsGreedily(fn, patterns);
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createRefactorEltAndConvPass() {
  return std::make_unique<RefactorEltAndConvPass>();
}

static PassRegistration<RefactorEltAndConvPass>
    pass("eltwise-early-stride",
         "Refactor hStride of elt and conv op");
