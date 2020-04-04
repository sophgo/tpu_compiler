//===- GenPowerWeight.cpp - Implementation of dynamice generate tanh lookup table / slope ---------===//
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
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/Support/Debug.h>
#include <float.h>

#define DEBUG_TYPE "gen-power_weight"

using namespace mlir;

static std::vector<std::string> passed;
namespace {

struct TpuGenPowerWeightPattern : public RewritePattern {
  TpuGenPowerWeightPattern(MLIRContext *context)
      : RewritePattern("tpu.power", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto pwOp = dyn_cast<tpu::PowerOp>(op);
    TensorFile *wTF = getWeightTensorFile(op);
    Value *wfV = getWeightFileValue(op);

    LLVM_DEBUG(llvm::errs() << pwOp.getOperationName()
        << ", scale is " << pwOp.scale().convertToFloat() << "\n"
        << ", power is " << pwOp.power().convertToFloat() << "\n"
        << ", shift is " << pwOp.shift().convertToFloat() << "\n";);

    std::string op_name = pwOp.getAttrOfType<StringAttr>("name").getValue().str();

    // TODO: not use name as uniq id
    if (std::find(passed.begin(), passed.end(), op_name) != passed.end()) {
      LLVM_DEBUG(llvm::errs() << pwOp.name() << " gen already\n";);
      return matchFailure();
    }

    passed.push_back(op_name);


    // TODO: not duplicat code in quant16
    std::vector<std::unique_ptr<std::vector<float> > > weights(2);
    int weight_idx = 0;
    // 0 is input
    for (unsigned i = 1; i < pwOp.getNumOperands(); ++i) {
      auto weight_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
          pwOp.getOperand(i)->getDefiningOp());
      assert(weight_op);
      assert(weight_op.name().hasValue());
      auto tensor_name = weight_op.name().getValue();
      LLVM_DEBUG(llvm::errs() << "  weight[" << i << "] : " << tensor_name << "\n";);
      auto type = weight_op.getResult()->getType().cast<TensorType>();
      weights[weight_idx] = wTF->readTensor<float>(tensor_name, type);
      weight_idx++;
      // delete the tensor from the weight file
      wTF->deleteTensor<float>(tensor_name);
    }

    auto scale_type = pwOp.scale_table()->getType().cast<TensorType>();
    std::vector<int64_t> scale_shape(scale_type.getShape());
    assert(scale_shape.size() == 4);

    auto shift_type = pwOp.shift_table()->getType().cast<TensorType>();
    std::vector<int64_t> shift_shape(shift_type.getShape());
    assert(shift_shape.size() == 4);

    // weight order result from CaffeToMlirTranslate.cpp' push order
    std::vector<float> scale(weights[0]->size());
    std::vector<float> shift(weights[1]->size());

    for (uint32_t i = 0; i < scale.size(); i++) {
      scale[i] = pwOp.scale().convertToFloat();
    }

    for (uint32_t i = 0; i < shift.size(); i++) {
      shift[i] = pwOp.shift().convertToFloat();
    }

    // update op
    std::vector<Value *> newOperands;
    newOperands.push_back(pwOp.getOperand(0)); // <! 0 is input

    // add new filter and bias weight
    std::vector<std::vector<float> *> newWeights{ &scale, &shift };
    std::vector<std::vector<int64_t> > weightShapes{ scale_shape, shift_shape };

    // 2 means scale / shift
    for (int i = 0; i < 2; ++i) {
      auto tensor_name = op_name + "_gen_weight_" + std::to_string(i);
      LLVM_DEBUG(llvm::errs() << "  new_weight[" << i << "] : " << tensor_name << "\n";);

      auto type = RankedTensorType::get(weightShapes[i],
              FloatType::getF32(rewriter.getContext()));

      wTF->addTensor<float>(tensor_name, newWeights[i], type);
      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
      attrs.push_back(rewriter.getNamedAttr("storage", rewriter.getStringAttr("FP32")));

      auto new_weight_op = rewriter.create<tpu::LoadWeightOp>(op->getLoc(), type,
          ArrayRef<Value *>{wfV}, ArrayRef<NamedAttribute>{attrs});
      newOperands.push_back(new_weight_op);
    }

    // replace with the new tanh op
    auto origAttrs = pwOp.getAttrs();
    std::vector<NamedAttribute> newAttrs(origAttrs.begin(), origAttrs.end());

    rewriter.replaceOpWithNewOp<tpu::PowerOp>(
        pwOp, pwOp.getResult()->getType(),
        ArrayRef<Value *>{newOperands}, ArrayRef<NamedAttribute>{newAttrs});

    return matchSuccess();

  }
};

class GenPowerWeightPass : public FunctionPass<GenPowerWeightPass> {
public:
  explicit GenPowerWeightPass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();
    auto *context = &getContext();
    OwningRewritePatternList patterns;
    patterns.insert<TpuGenPowerWeightPattern>(context);
    applyPatternsGreedily(fn, patterns);
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createGenPowerWeightPass() {
  return std::make_unique<GenPowerWeightPass>();
}

static PassRegistration<GenPowerWeightPass>
    pass("gen-power-weight",
         "duplicate scale/shift for leverage depthwise op");
