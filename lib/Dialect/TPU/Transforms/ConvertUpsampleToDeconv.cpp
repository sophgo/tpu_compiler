//===- ConvertUpsampleToDeconv.cpp - convert unsample to deconv -----------===//
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
// This file convert upsample op to deconv.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "upsample_to_deconv"

using namespace mlir;

namespace {

struct TpuUpsampleOpPattern : public RewritePattern {
  TpuUpsampleOpPattern(MLIRContext *context)
      : RewritePattern("tpu.upsample", 7, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto upsampleOp = cast<tpu::UpsampleOp>(op);
    assert(op->getNumOperands() == 1 && "operands num should be 1");
    LLVM_DEBUG(llvm::errs() << upsampleOp.getOperationName() << "\n";);
    TensorFile *wTF = getWeightTensorFile(op);
    Value *wFV = getWeightFileValue(op);

    // op_name
    std::string op_name =
        upsampleOp.getAttrOfType<StringAttr>("name").getValue().str();
    LLVM_DEBUG(llvm::errs() << "upsample Op: " << op_name << "\n";);

    auto input = op->getOperand(0);
    auto input_type = input->getType().cast<RankedTensorType>();
    auto input_shape = input_type.getShape();
    auto scale = upsampleOp.scale().getLimitedValue();
    if (input_shape.size() == 4 && input_shape[2] == 1 && input_shape[3] == 1) {
      if (isa<tpu::SigmoidOp>(input->getDefiningOp())) {
        // TODO(charle.hu): workaround, if convert to deconv, will fail in backend, test by ecanet50
        return matchFailure();
      }
    }
    int g = input_shape[1];
    int oc = input_shape[1] / g;
    int ic = input_shape[1] / g;
    int h = scale;
    int w = scale;

    int count = g * oc * ic * h * w;
    std::vector<float> filter(count, 1);
    std::vector<int64_t> filter_shape;
    if (g != 1) {
      filter_shape.push_back(g);
    }
    filter_shape.push_back(oc);
    filter_shape.push_back(ic);
    filter_shape.push_back(h);
    filter_shape.push_back(w);
    auto filterValue = addWeightTensorAndCreateWeightOp<float>(op, "filter",
                                 filter, filter_shape, "NONE", wTF, wFV);
    auto NoneOp = OpBuilder(op).create<tpu::NoneOp>(rewriter.getUnknownLoc(),
                                                    rewriter.getNoneType());
    std::vector<Value *> operands;
    operands.push_back(input);
    operands.push_back(filterValue);
    operands.push_back(NoneOp.getResult()); // bias
    operands.push_back(NoneOp.getResult()); // quant_scale
    operands.push_back(NoneOp.getResult()); // quant_zeropoint
    operands.push_back(NoneOp.getResult()); // quant_rshift
    operands.push_back(NoneOp.getResult()); // quant_multiplier

    bool is_dw = true;
    bool with_bias = false;
    std::vector<int64_t> kernel(2), stride(2), padding(2), dilation(2);
    kernel[0] = kernel[1] = scale;
    padding[0] = padding[1] = 0;
    dilation[0] = dilation[1] = 1;
    stride[0] = stride[1] = scale;

    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("name", upsampleOp.nameAttr()));

    if (upsampleOp.layer_id().hasValue()) {
      attrs.push_back(rewriter.getNamedAttr("layer_id",
                                            upsampleOp.layer_idAttr()));
    }
    attrs.push_back(rewriter.getNamedAttr("param",
                    tpu::ConvParam::get(rewriter.getI32IntegerAttr(stride[0]),
                                        rewriter.getI32IntegerAttr(stride[1]),
                                        rewriter.getStringAttr("VALID"),
                                        rewriter.getI32IntegerAttr(dilation[0]),
                                        rewriter.getI32IntegerAttr(dilation[1]),
                                        rewriter.getI32IntegerAttr(padding[0]),
                                        rewriter.getI32IntegerAttr(padding[0]),
                                        rewriter.getI32IntegerAttr(padding[1]),
                                        rewriter.getI32IntegerAttr(padding[1]),
                                        rewriter.getI32IntegerAttr(g),
                                        rewriter.getBoolAttr(is_dw),
                                        rewriter.getBoolAttr(with_bias),
                                        rewriter.getBoolAttr(false),
                                        rewriter.getContext())));
  attrs.push_back(rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
  rewriter.replaceOpWithNewOp<tpu::DeConv2DOp>(
        upsampleOp, upsampleOp.getResult()->getType(),
        ArrayRef<Value *>{operands}, ArrayRef<NamedAttribute>{attrs});

  return matchSuccess();
}
};

class ConvertUpsampleToDeconvPass : public FunctionPass<ConvertUpsampleToDeconvPass> {
public:
  explicit ConvertUpsampleToDeconvPass(llvm::raw_ostream &os = llvm::errs())
      : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<TpuUpsampleOpPattern>(context);
    applyPatternsGreedily(fn, patterns);
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

void tpu::UpsampleOp::getCanonicalizationPatterns(
         OwningRewritePatternList &results,
         MLIRContext *context) {
  results.insert<TpuUpsampleOpPattern>(context);
}

std::unique_ptr<OpPassBase<FuncOp>> mlir::createConvertUpsampleToDeconvPass() {
  return std::make_unique<ConvertUpsampleToDeconvPass>();
}

static PassRegistration<ConvertUpsampleToDeconvPass>
    pass("convert-upsample-to-deconv",
         "Convert a upsample operation to deconv");
