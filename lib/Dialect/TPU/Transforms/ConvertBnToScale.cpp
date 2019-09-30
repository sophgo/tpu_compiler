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
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

#if 0
struct TpuBatchNormOpPattern : public OpRewritePattern<tpu::BatchNormOp> {
  using OpRewritePattern<tpu::BatchNormOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(tpu::BatchNormOp op,
                                     PatternRewriter &rewriter) const {
    llvm::errs() << "match " << op.getOperationName() << "\n";

    // TODO: optimize performance

    // find tensor filename
    tpu::LoadWeightOp one_weight_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
        op.getOperand(1)->getDefiningOp());
    assert (one_weight_op);
    tpu::LoadFileOp loadFileOp = llvm::dyn_cast_or_null<tpu::LoadFileOp>(
        one_weight_op.getOperand()->getDefiningOp());
    assert (loadFileOp);
    auto filename = loadFileOp.getAttrOfType<StringAttr>("filename").getValue();
    llvm::errs() << "LoadFileOp filename " << filename << "\n";
    auto weightTensorFile = openTensorFile(filename);

    // rewrite all. noted the 4th operand of bn will be removed automatically.
    rewriter.replaceOpWithNewOp<tpu::ScaleOp>(
        op, op.getResult()->getType(),
        ArrayRef<Value *>{op.getOperand(0), op.getOperand(1), op.getOperand(2)},
        ArrayRef<NamedAttribute>{});

    return matchSuccess();
    //return matchFailure();
  }
};
#endif

struct TpuBatchNormOpPattern : public RewritePattern {
  TpuBatchNormOpPattern(MLIRContext *context, TensorFile *weightTensorFile)
      : RewritePattern("tpu.batch_norm", 1, context),
        weightTensorFile_(weightTensorFile) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto bnOp = cast<tpu::BatchNormOp>(op);
    assert(op->getNumOperands() == 4);
    llvm::errs() << bnOp.getOperationName() << "\n";
    auto loc = op->getLoc();

    // op_name
    std::string op_name = bnOp.getAttrOfType<StringAttr>("name").getValue().str();
    llvm::errs() << "BatchNorm Op: " << op_name << "\n";
    auto one_weight_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
          bnOp.getOperand(1)->getDefiningOp());
    auto weightFileVar = one_weight_op.getOperand();

    // find mean, variance, scale tensor, and delete them
    std::vector<std::unique_ptr<std::vector<float> > > bnWeights(3);
    for (int i = 0; i < 3; ++i) {
      auto weight_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
          bnOp.getOperand(i + 1)->getDefiningOp());
      assert(weight_op);
      assert(weight_op.name().hasValue());
      auto tensor_name = weight_op.name().getValue();
      llvm::errs() << "  weight[" << i << "] : " << tensor_name << "\n";
      auto type = weight_op.getResult()->getType().cast<TensorType>();
      bnWeights[i] = weightTensorFile_->readTensor<float>(tensor_name, type);
      // delete the tensor from the weight file
      weightTensorFile_->deleteTensor<float>(tensor_name);
    }

    // convert tensors
    llvm::errs() << "  mean size: " << bnWeights[0]->size() << "\n";
    llvm::errs() << "  variance size: " << bnWeights[1]->size() << "\n";
    llvm::errs() << "  scale size: " << bnWeights[2]->size() << "\n";
    int oc = (int)bnWeights[0]->size();
    std::vector<float> new_scale(oc);
    std::vector<float> new_bias(oc);

    float *mean = (float *)bnWeights[0]->data();
    float *variance = (float *)bnWeights[1]->data();
    float *scale = (float *)bnWeights[2]->data();

    float eps = 1.0e-5;
    float scale_factor = 1 / scale[0];
    for (int i = 0; i < oc; ++i) {
      mean[i] = mean[i] * scale_factor;
      variance[i] = variance[i] * scale_factor;
      new_scale[i] = 1.0 / sqrt(variance[i] + eps);
      new_bias[i] = -1.0 * new_scale[i] * mean[i];
    }
    std::vector<std::vector<float> *> newWeights{ &new_scale, &new_bias };

    std::vector<Value *> newOperands;
    newOperands.push_back(bnOp.getOperand(0));
    // add new scale and bias ops
    for (int i = 0; i < 2; ++i) {
      auto tensor_name = op_name + "_to_scale_" + std::to_string(i);
      llvm::errs() << "  new_weight[" << i << "] : " << tensor_name << "\n";
      auto type = rewriter.getTensorType({oc}, FloatType::getF32(rewriter.getContext()));
      weightTensorFile_->addTensor<float>(tensor_name, newWeights[i], type);
      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
      auto new_weight_op = rewriter.create<tpu::LoadWeightOp>(loc, type,
          ArrayRef<Value *>{weightFileVar}, ArrayRef<NamedAttribute>{attrs});
      newOperands.push_back(new_weight_op);
    }

    // replace bn with scale
    // keep the op_name because the calibration table is using this name
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(op_name)));
    rewriter.replaceOpWithNewOp<tpu::ScaleOp>(
        bnOp, bnOp.getResult()->getType(),
        ArrayRef<Value *>{newOperands}, ArrayRef<NamedAttribute>{attrs});

    return matchSuccess();
  }

  TensorFile *weightTensorFile_;
};

class ConvertBnToScalePass : public FunctionPass<ConvertBnToScalePass> {
public:
  explicit ConvertBnToScalePass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    // find tensor filename
    //OpBuilder b(fn.getBody());
    llvm::StringRef filename;
    fn.walk<tpu::LoadFileOp>([&](tpu::LoadFileOp op) {
      filename = op.getAttrOfType<StringAttr>("filename").getValue();
      llvm::errs() << "LoadFileOp filename " << filename << "\n";
    });
    auto weightTensorFile = openTensorFile(filename);

    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<TpuBatchNormOpPattern>(context, weightTensorFile.get());
    applyPatternsGreedily(fn, patterns);
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

std::unique_ptr<FunctionPassBase> mlir::createConvertBnToScalePass() {
  return std::make_unique<ConvertBnToScalePass>();
}

static PassRegistration<ConvertBnToScalePass>
    pass("convert-bn-to-scale",
         "Convert a BN operation to Scale operation");
