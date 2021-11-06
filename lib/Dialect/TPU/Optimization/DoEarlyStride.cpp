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
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/Passes.h"
#include "tpuc/Support/TensorFile.h"
#include "tpuc/TPUOperationSupport.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "eltwise_early_stride"

using namespace mlir;

namespace {

template <typename OpTy, typename NextConvTy>
struct MoveConvStrideToEltwiseOpPattern : public RewritePattern {
  MoveConvStrideToEltwiseOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {

    Operation *nextOp = nullptr;
    int strideH = 0;
    int strideW = 0;
    for (auto &use : op->getResult(0).getUses()) {
      nextOp = use.getOwner();
      if (auto convOp = dyn_cast<NextConvTy>(nextOp)) {
        auto filter_type =
            convOp.filter().getType().template cast<TensorType>();
        std::vector<int64_t> f_s(filter_type.getShape());
        int kh = f_s[f_s.size() - 2];
        int kw = f_s[f_s.size() - 1];
        int sh = convOp.param().stride_h().getInt();
        int sw = convOp.param().stride_w().getInt();
        if (strideH == 0 || strideW == 0) {
          strideH = sh;
          strideW = sw;
        }
        if (strideH != sh || strideW != sw) {
          LLVM_DEBUG(llvm::errs()
                     << "stride of all successor conv2d should be same\n");
          return failure();
        }
        if (sh == 1 || sw == 1) {
          return failure();
        }
        if (kh != 1 || kw != 1) {
          return failure();
        }
      } else {
        // if one of uses is not 1x1 conv,
        // we cannot do early stride.
        return failure();
      }
    }

    auto eltOp = cast<OpTy>(op);
    auto shape = getTensorShape(eltOp.getResult());
    if (shape[2] % strideH != 0 || shape[3] % strideW != 0) {
      // padding case, stop
      return failure();
    }

    for (auto &use : op->getResult(0).getUses()) { // Refactor convOp
      nextOp = use.getOwner();
      auto convOp = dyn_cast<NextConvTy>(nextOp);
      convOp->setAttr("param",
                     tpu::ConvParam::get(
                         rewriter.getI32IntegerAttr(1), // stride_h,
                         rewriter.getI32IntegerAttr(1), // stride_w,
                         convOp.param().padding(), convOp.param().dilation_h(),
                         convOp.param().dilation_w(),
                         convOp.param().padding_t(), convOp.param().padding_b(),
                         convOp.param().padding_l(), convOp.param().padding_r(),
                         convOp.param().group(), convOp.param().is_dw(),
                         convOp.param().with_bias(), convOp.param().do_relu(),
                         convOp.param().ins(), convOp.param().pad_value(),
                         rewriter.getContext())); // rewrite strideH
    }

    int on = shape[0];
    int oc = shape[1];
    int oh = shape[2] / strideH;
    int ow = shape[3] / strideW;
    eltOp->setAttr("do_early_stride", rewriter.getBoolAttr(true));
    eltOp->setAttr("early_stride_h", rewriter.getI32IntegerAttr(strideH));
    eltOp->setAttr("early_stride_w", rewriter.getI32IntegerAttr(strideW));

    auto tensorType = eltOp.getResult().getType().template cast<RankedTensorType>();
    auto type = RankedTensorType::get({on, oc, oh, ow},
                                      tensorType.getElementType());
    eltOp.getResult().setType(type); // rewrite inputShape
    return success();
  }
};

class MoveConvStrideToEltwiseOpPass
    : public mlir::PassWrapper<MoveConvStrideToEltwiseOpPass,
                               FunctionPass> {
public:
  void runOnFunction() override {
    auto fn = getFunction();
    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.clear();

    patterns.insert<
        MoveConvStrideToEltwiseOpPattern<tpu::TG_INT8_EltwiseAddOp,
                                         tpu::TG_INT8_Conv2DOp>,
        MoveConvStrideToEltwiseOpPattern<tpu::TG_BF16_EltwiseAddOp,
                                         tpu::TG_BF16_Conv2DOp>
      >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::createMoveConvStrideToEltwiseOpPass() {
  return std::make_unique<MoveConvStrideToEltwiseOpPass>();
}
