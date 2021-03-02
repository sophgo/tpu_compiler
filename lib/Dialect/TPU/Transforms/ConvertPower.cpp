//===- ConvertPower.cpp - convert
// Power----------------------------------===//
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
// This file implements the power
//
//===----------------------------------------------------------------------===//

#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tpuc/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "convert_power"

using namespace mlir;

namespace {

// if power == 1, then convert to scale
struct TpuPowerToScalePattern : public RewritePattern {
  TpuPowerToScalePattern(MLIRContext *context)
      : RewritePattern("tpu.power", 2, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto powerOp = cast<tpu::PowerOp>(op);
    LLVM_DEBUG(llvm::errs() << powerOp.getOperationName() << ":"
                            << getOpName(op) << "\n";);
    float power = powerOp.power().convertToFloat();
    float scale = powerOp.scale().convertToFloat();
    float shift = powerOp.shift().convertToFloat();
    if (power != 1.0f) {
      return failure();
    }
    if (scale == 1.0f && shift == 0.0f) {
      // remove this op
      rewriter.replaceOp(op, {op->getOperand(0)});
      return success();
    }
    TensorFile *wTF = getWeightTensorFile(op);
    Value wfV = getWeightFileValue(op);
    // op_name
    std::string op_name =
        powerOp->getAttrOfType<StringAttr>("name").getValue().str();
    std::vector<int64_t> shape;
    int64_t input_size;
    getTensorShapeAndSize(powerOp.input(), shape, input_size);
    assert(shape.size() >= 2); // if only 1 dims, need reshape
    int64_t oc = shape[1];
    std::vector<float> new_scale(oc);
    std::vector<float> new_bias(oc);
    for (int i = 0; i < oc; ++i) {
      new_scale[i] = scale;
      new_bias[i] = shift;
    }
    std::vector<std::vector<float> *> newWeights{&new_scale, &new_bias};
    std::vector<Value> newOperands;
    newOperands.push_back(op->getOperand(0));
    for (int i = 0; i < 2; ++i) {
      auto tensor_name = op_name + "_scale_" + std::to_string(i);
      auto type =
          RankedTensorType::get({oc}, FloatType::getF32(rewriter.getContext()));
      wTF->addTensor<float>(tensor_name, newWeights[i], type);
      std::vector<NamedAttribute> attrs;
      attrs.push_back(
          rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
      auto new_weight_op = rewriter.create<tpu::LoadWeightOp>(
          op->getLoc(), type, ArrayRef<Value>{wfV},
          ArrayRef<NamedAttribute>{attrs});
      newOperands.push_back(new_weight_op);
    }
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("name", rewriter.getStringAttr(op_name)));
    rewriter.replaceOpWithNewOp<tpu::ScaleOp>(
        powerOp, powerOp.getResult().getType(), ArrayRef<Value>{newOperands},
        ArrayRef<NamedAttribute>{attrs});

    return success();
  }
};

} // namespace

void tpu::PowerOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<TpuPowerToScalePattern>(context);
}
