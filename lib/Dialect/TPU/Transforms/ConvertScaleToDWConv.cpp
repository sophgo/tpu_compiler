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
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

struct TpuScaleOpPattern : public RewritePattern {
  TpuScaleOpPattern(MLIRContext *context)
      : RewritePattern("tpu.scale", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto scaleOp = cast<tpu::ScaleOp>(op);
    llvm::errs() << scaleOp.getOperationName() << "\n";
    TensorFile *wTF = getWeightTensorFile(op);

    // op_name
    std::string op_name = scaleOp.name().str();
    llvm::errs() << "Scale Op: " << op_name << "\n";

    // parse param
    std::vector<int64_t> shape;
    int64_t input_size, n, c, h, w;
    getTensorShapeAndSize(scaleOp.input(), shape, input_size);
    getNCHW(shape, n, c, h, w);

    // get tensor
    auto scale = readAndDeleteWeightTensor<float>(scaleOp.scale(), wTF);
    std::unique_ptr<std::vector<float> > bias = nullptr;
    if ( !isTensorNone(scaleOp.bias()) ) {
      bias = readAndDeleteWeightTensor<float>(scaleOp.bias(), wTF);
    }

    StringRef storageType = "FP32";
    auto filter_type = std::vector<int64_t>({c, 1, 1, 1, 1});
    addWeightTensorAndUpdateWeightOp<float>(scaleOp.scale(),
        "scale", *scale, filter_type, storageType, wTF);
    if (bias) {
      auto bias_type = std::vector<int64_t>({c});
      addWeightTensorAndUpdateWeightOp<float>(scaleOp.bias(),
          "scale", *bias, bias_type, storageType, wTF);
    }

    // replace scale with conv
    // keep the op_name because the calibration table is using this name

    std::vector<Value *> operands;
    operands.push_back(scaleOp.getOperand(0));
    operands.push_back(scaleOp.getOperand(1));
    operands.push_back(scaleOp.getOperand(2));
    auto NoneOp = rewriter.create<tpu::NoneOp>(
        rewriter.getUnknownLoc(), rewriter.getNoneType());
    operands.push_back(NoneOp.getResult());  // quant_scale
    operands.push_back(NoneOp.getResult());  // quant_zeropoint
    operands.push_back(NoneOp.getResult());  // quant_rshift
    operands.push_back(NoneOp.getResult());  // quant_multiplier

    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(op_name)));
    attrs.push_back(rewriter.getNamedAttr("param",
        tpu::ConvParam::get(
            rewriter.getI32IntegerAttr(1),
            rewriter.getI32IntegerAttr(1),
            rewriter.getStringAttr("VALID"),
            rewriter.getI32IntegerAttr(1),
            rewriter.getI32IntegerAttr(1),
            rewriter.getI32IntegerAttr(c),
            rewriter.getBoolAttr(true),
            rewriter.getBoolAttr(bias?true:false),
            rewriter.getBoolAttr(scaleOp.do_relu()),
            rewriter.getContext())));
    attrs.push_back(rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
    if (scaleOp.layer_id().hasValue()) {
      attrs.push_back(rewriter.getNamedAttr("layer_id", scaleOp.layer_idAttr()));
    }
    rewriter.replaceOpWithNewOp<tpu::Conv2DOp>(
        scaleOp, scaleOp.getResult()->getType(),
        ArrayRef<Value *>{operands}, ArrayRef<NamedAttribute>{attrs});

    return matchSuccess();
  }
};

class ConvertScaleToDWConvPass : public FunctionPass<ConvertScaleToDWConvPass> {
public:
  explicit ConvertScaleToDWConvPass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<TpuScaleOpPattern>(context);
    applyPatternsGreedily(fn, patterns);
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createConvertScaleToDWConvPass() {
  return std::make_unique<ConvertScaleToDWConvPass>();
}

static PassRegistration<ConvertScaleToDWConvPass>
    pass("convert-scale-to-dwconv",
         "Convert a scale operation to a dwconv operation");
