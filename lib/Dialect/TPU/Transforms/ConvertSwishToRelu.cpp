//===- ConvertScale.cpp - convert scale -----------------------------------===//
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
// This file implements the convert swish to relu.
//
//===----------------------------------------------------------------------===//

#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tpuc/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "convert_swish_to_relu"

using namespace mlir;

namespace {

struct TpuConvertSwishToReLUPattern : public RewritePattern {
  TpuConvertSwishToReLUPattern(MLIRContext *context)
      : RewritePattern("tpu.eltwise_mul", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto eltmulOp = cast<tpu::EltwiseMulOp>(op);
    LLVM_DEBUG(llvm::errs() << eltmulOp.getOperationName() << "\n";);

    // op_name
    std::string op_name = eltmulOp.name().str();
    bool is_swish = false;
    // for EltwiseMulOp, handle swish case only, other case needs more tuning
    if (isa<tpu::Conv2DOp>(op->getOperand(0).getDefiningOp()) &&
        isa<tpu::SigmoidOp>(op->getOperand(1).getDefiningOp())) {
      is_swish = true;
      LLVM_DEBUG(llvm::errs() << "EltwiseMul Op: " << op_name << " is Swish." <<"\n";);
    }
    if (!is_swish) {
      return failure();
    }

    // parse param
    std::vector<int64_t> shape;
    int64_t input_size, n, c, h, w;
    getTensorShapeAndSize(eltmulOp.getOperand(0), shape, input_size);
    getNCHW(shape, n, c, h, w);

    std::vector<Value> operands;
    operands.push_back(eltmulOp.getOperand(0));

    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("name",
                    rewriter.getStringAttr(op_name)));

    attrs.push_back(rewriter.getNamedAttr("quant",
                                          getDefaultQuantParam(rewriter)));
    rewriter.replaceOpWithNewOp<tpu::ReluOp>(
        eltmulOp, eltmulOp.getResult().getType(), ArrayRef<Value>{operands},
        ArrayRef<NamedAttribute>{attrs});

    return success();
  }
};

class ConvertSwishToReLUPass : public mlir::PassWrapper<ConvertSwishToReLUPass, FunctionPass> {
public:
  explicit ConvertSwishToReLUPass(llvm::raw_ostream &os = llvm::errs())
      : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<TpuConvertSwishToReLUPattern>(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::createConvertSwishToReLUPass() {
  return std::make_unique<ConvertSwishToReLUPass>();
}

static PassRegistration<ConvertSwishToReLUPass>
    pass("convert-swish-to-relu", "Convert Swish Activation to Relu");
