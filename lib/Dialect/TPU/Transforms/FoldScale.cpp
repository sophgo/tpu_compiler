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

struct TpuFoldScalePattern : public RewritePattern {
  TpuFoldScalePattern(MLIRContext *context, TensorFile *weightTensorFile)
      : RewritePattern("tpu.scale", 1, context),
        weightTensorFile_(weightTensorFile) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto laterScaleOp = cast<tpu::ScaleOp>(op);
    llvm::errs() << laterScaleOp.getOperationName() << "\n";

    // match consecutive scale operations
    auto formerOp = laterScaleOp.getOperand(0)->getDefiningOp();
    if (!matchPattern(formerOp, m_Op<tpu::ScaleOp>()))
      return matchFailure();
    auto formerScaleOp = cast<tpu::ScaleOp>(formerOp);

    // op_name from the later scale
    std::string op_name = laterScaleOp.getAttrOfType<StringAttr>("name").getValue().str();
    llvm::errs() << "Scale Op: " << op_name << "\n";
    auto one_weight_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
          laterScaleOp.getOperand(1)->getDefiningOp());
    auto weightFileVar = one_weight_op.getOperand();

    // find scale and bias tensor for both later and former scale_op
    std::vector<std::unique_ptr<std::vector<float> > > laterWeights(2);
    for (int i = 0; i < 2; ++i) {
      auto weight_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
          laterScaleOp.getOperand(i + 1)->getDefiningOp());
      assert(weight_op);
      assert(weight_op.name().hasValue());
      auto tensor_name = weight_op.name().getValue();
      llvm::errs() << "  weight[" << i << "] : " << tensor_name << "\n";
      auto type = weight_op.getResult()->getType().cast<TensorType>();
      laterWeights[i] = weightTensorFile_->readTensor<float>(tensor_name, type);
      // delete the tensor from the weight file
      weightTensorFile_->deleteTensor<float>(tensor_name);
    }
    std::vector<std::unique_ptr<std::vector<float> > > formerWeights(2);
    for (int i = 0; i < 2; ++i) {
      auto weight_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
          formerScaleOp.getOperand(i + 1)->getDefiningOp());
      assert(weight_op);
      assert(weight_op.name().hasValue());
      auto tensor_name = weight_op.name().getValue();
      llvm::errs() << "  weight[" << i << "] : " << tensor_name << "\n";
      auto type = weight_op.getResult()->getType().cast<TensorType>();
      formerWeights[i] = weightTensorFile_->readTensor<float>(tensor_name, type);
      // delete the tensor from the weight file
      weightTensorFile_->deleteTensor<float>(tensor_name);
    }

    // convert tensors
    llvm::errs() << "  former scale size: " << formerWeights[0]->size() << "\n";
    llvm::errs() << "  former bias size: " << formerWeights[1]->size() << "\n";
    llvm::errs() << "  later scale size: " << laterWeights[0]->size() << "\n";
    llvm::errs() << "  later bias size: " << laterWeights[1]->size() << "\n";
    int oc = (int)formerWeights[0]->size();
    std::vector<float> new_scale(oc);
    std::vector<float> new_bias(oc);

    float *former_scale = (float *)formerWeights[0]->data();
    float *former_bias = (float *)formerWeights[1]->data();
    float *later_scale = (float *)laterWeights[0]->data();
    float *later_bias = (float *)laterWeights[1]->data();

    for (int i = 0; i < oc; ++i) {
      new_scale[i] = former_scale[i] * later_scale[i];
      new_bias[i] = former_bias[i] * later_scale[i] + later_bias[i];
    }
    std::vector<std::vector<float> *> newWeights{ &new_scale, &new_bias };

    std::vector<Value *> newOperands;
    newOperands.push_back(formerScaleOp.getOperand(0));
    // add new scale and bias ops
    for (int i = 0; i < 2; ++i) {
      auto tensor_name = op_name + "_fold_" + std::to_string(i);
      llvm::errs() << "  new_weight[" << i << "] : " << tensor_name << "\n";
      auto type = rewriter.getTensorType({oc}, FloatType::getF32(rewriter.getContext()));
      weightTensorFile_->addTensor<float>(tensor_name, newWeights[i], type);
      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
      auto new_weight_op = rewriter.create<tpu::LoadWeightOp>(loc, type,
          ArrayRef<Value *>{weightFileVar}, ArrayRef<NamedAttribute>{attrs});
      newOperands.push_back(new_weight_op);
    }

    // replace the later scale with the new scale
    // the former one will be removed automatically
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(op_name)));
    rewriter.replaceOpWithNewOp<tpu::ScaleOp>(
        laterScaleOp, formerScaleOp.getResult()->getType(),
        ArrayRef<Value *>{newOperands}, ArrayRef<NamedAttribute>{attrs});

    return matchSuccess();
  }

  TensorFile *weightTensorFile_;
};

class FoldScalePass : public FunctionPass<FoldScalePass> {
public:
  explicit FoldScalePass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    // find tensor filename
    llvm::StringRef filename;
    fn.walk<tpu::LoadFileOp>([&](tpu::LoadFileOp op) {
      filename = op.getAttrOfType<StringAttr>("filename").getValue();
      llvm::errs() << "LoadFileOp filename " << filename << "\n";
    });
    auto weightTensorFile = openTensorFile(filename);

    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<TpuFoldScalePattern>(context, weightTensorFile.get());
    applyPatternsGreedily(fn, patterns);

    std::string newName;
    weightTensorFile->keep(true, &newName);
    fn.walk<tpu::LoadFileOp>([&](tpu::LoadFileOp op) {
      OpBuilder b(fn.getBody());
      op.setAttr("filename", b.getStringAttr(newName));
      llvm::errs() << "LoadFileOp filename updated to " << newName << "\n";
    });
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

std::unique_ptr<FunctionPassBase> mlir::createFoldScalePass() {
  return std::make_unique<FoldScalePass>();
}

static PassRegistration<FoldScalePass>
    pass("fold-scale",
         "Fold two consecutive scale operations into one");
