//===- ConvertBnToScale.cpp - convert batchnorm to scale ------------------===//
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
// This file convert batchnorm op to scale.
//
//===----------------------------------------------------------------------===//

#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tpuc/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "bn_to_scale"

using namespace mlir;

namespace {

void TpuBNOpPattern(Operation *op, PatternRewriter &rewriter);

struct TpuBatchNormOpPattern : public RewritePattern {
  TpuBatchNormOpPattern(MLIRContext *context)
      : RewritePattern("tpu.batch_norm", 7, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto bnOp = cast<tpu::BatchNormOp>(op);
    if (isTensorNone(bnOp.bias())) {
      // bn case
    }
    else {
      // pspnet BN
      TpuBNOpPattern(op, rewriter);
      return success();
    }

    LLVM_DEBUG(llvm::errs() << bnOp.getOperationName() << "\n";);
    auto loc = op->getLoc();
    TensorFile *wTF = getWeightTensorFile(op);
    Value wfV = getWeightFileValue(op);

    // op_name
    std::string op_name =
        bnOp->getAttrOfType<StringAttr>("name").getValue().str();
    LLVM_DEBUG(llvm::errs() << "BatchNorm Op: " << op_name << "\n";);

    // find mean, variance, scale tensor, and delete them
    std::vector<std::unique_ptr<std::vector<float> > > bnWeights(3);
    for (int i = 0; i < 3; ++i) {
      auto weight_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
          bnOp.getOperand(i + 1).getDefiningOp());
      assert(weight_op && "weight op should be exist");
      auto tensor_name = weight_op.name();
      LLVM_DEBUG(llvm::errs() << "  weight[" << i << "] : "
                              << tensor_name << "\n";);
      auto type = weight_op.getResult().getType().cast<TensorType>();
      bnWeights[i] = wTF->readTensor<float>(tensor_name, type);
      // delete the tensor from the weight file
      wTF->deleteTensor<float>(tensor_name);
    }

    // convert tensors
    LLVM_DEBUG(llvm::errs() << "  mean size: " << bnWeights[0]->size() << "\n"
               << "  variance size: " << bnWeights[1]->size() << "\n"
               << "  scale size: " << bnWeights[2]->size() << "\n";);
    int oc = (int)bnWeights[0]->size();
    std::vector<float> new_scale(oc);
    std::vector<float> new_bias(oc);

    float *mean = (float *)bnWeights[0]->data();
    float *variance = (float *)bnWeights[1]->data();
    float *scale = (float *)bnWeights[2]->data();
    float variance_epsilon = bnOp.variance_epsilon().convertToFloat();
    float scale_factor = 1 / scale[0];
    for (int i = 0; i < oc; ++i) {
      mean[i] = mean[i] * scale_factor;
      variance[i] = variance[i] * scale_factor;
      if (fabs(variance[i]) <= variance_epsilon && fabs(mean[i]) <= 1e-8) {
        LLVM_DEBUG(llvm::errs() << "BN: var too small, i=" << i
                                << ", v=" << std::to_string(variance[i])
                                << ", m=" << std::to_string(mean[i]) << "\n";);
      }
      new_scale[i] = 1.0 / sqrt(variance[i] + variance_epsilon);
      new_bias[i] = -1.0 * new_scale[i] * mean[i];
    }
    std::vector<std::vector<float> *> newWeights{ &new_scale, &new_bias };

    std::vector<Value> newOperands;
    newOperands.push_back(bnOp.getOperand(0));
    // add new scale and bias ops
    for (int i = 0; i < 2; ++i) {
      auto tensor_name = op_name + "_to_scale_" + std::to_string(i);
      LLVM_DEBUG(llvm::errs() << "  new_weight[" << i << "] : "
                              << tensor_name << "\n";);
      auto type = RankedTensorType::get({oc},
                                    FloatType::getF32(rewriter.getContext()));
      wTF->addTensor<float>(tensor_name, newWeights[i], type);
      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr("name",
                               rewriter.getStringAttr(tensor_name)));
      auto new_weight_op = rewriter.create<tpu::LoadWeightOp>(loc, type,
          ArrayRef<Value>{wfV}, ArrayRef<NamedAttribute>{attrs});
      newOperands.push_back(new_weight_op);
    }

    // replace bn with scale
    // keep the op_name because the calibration table is using this name
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("name",
                    rewriter.getStringAttr(op_name)));
    rewriter.replaceOpWithNewOp<tpu::ScaleOp>(
        bnOp, bnOp.getResult().getType(),
        ArrayRef<Value>{newOperands}, ArrayRef<NamedAttribute>{attrs});

    return success();
  }
};

void TpuBNOpPattern(Operation *op, PatternRewriter &rewriter) {
  auto bnOp = cast<tpu::BatchNormOp>(op);
  assert(op->getNumOperands() == 5 && "operands num should be 5");
  int weightNum = op->getNumOperands() - 1; // except input
  LLVM_DEBUG(llvm::errs() << bnOp.getOperationName() << "\n";);
  auto loc = op->getLoc();
  TensorFile *wTF = getWeightTensorFile(op);
  Value wfV = getWeightFileValue(op);

  // op_name
  std::string op_name =
    bnOp->getAttrOfType<StringAttr>("name").getValue().str();
  LLVM_DEBUG(llvm::errs() << "BN Op: " << op_name << "\n";);

  // find mean, variance, scale tensor, and delete them
  std::vector<std::unique_ptr<std::vector<float> > > bnWeights(weightNum);
  for (int i = 0; i < weightNum; ++i) {
    auto weight_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
        bnOp.getOperand(i + 1).getDefiningOp());
    assert(weight_op && "weight op should be exist");
    auto tensor_name = weight_op.name();
    LLVM_DEBUG(llvm::errs() << "  weight[" << i << "] : "
        << tensor_name << "\n";);
    auto type = weight_op.getResult().getType().cast<TensorType>();
    bnWeights[i] = wTF->readTensor<float>(tensor_name, type);
    // delete the tensor from the weight file
    wTF->deleteTensor<float>(tensor_name);
  }

  // convert tensors
  LLVM_DEBUG(llvm::errs() << "  slope size: " << bnWeights[0]->size() << "\n"
      << "  bias size: " << bnWeights[1]->size() << "\n"
      << "  mean size: " << bnWeights[2]->size() << "\n"
      << "  variance size: " << bnWeights[2]->size() << "\n";);
  int oc = (int)bnWeights[0]->size();
  std::vector<float> new_scale(oc);
  std::vector<float> new_bias(oc);

  float *slope = (float *)bnWeights[0]->data();
  float *bias = (float *)bnWeights[1]->data();
  float *mean= (float *)bnWeights[2]->data();
  float *variance = (float *)bnWeights[3]->data();

  float variance_epsilon = bnOp.variance_epsilon().convertToFloat();
  for (int i = 0; i < oc; ++i) {
    float _mean = mean[i];
    float _variance = variance[i] * slope[i];
    if (fabs(_variance) <= variance_epsilon && fabs(_mean) <= 1e-8) {
      LLVM_DEBUG(llvm::errs() << "BN: var too small, i=" << i
          << ", v=" << std::to_string(_variance)
          << ", m=" << std::to_string(_mean) << "\n";);
      // set to zero
      new_scale[i] = 1.0;
      new_bias[i] = 0.0;
    } else {
      new_scale[i] = 1.0 / sqrt(variance[i] + variance_epsilon) * slope[i];
      new_bias[i] = -1.0 * new_scale[i] * mean[i] + bias[i];
    }
  }
  std::vector<std::vector<float> *> newWeights{ &new_scale, &new_bias };

  std::vector<Value> newOperands;
  newOperands.push_back(bnOp.getOperand(0));
  // add new scale and bias ops
  for (int i = 0; i < 2; ++i) {
    auto tensor_name = op_name + "_to_scale_" + std::to_string(i);
    LLVM_DEBUG(llvm::errs() << "  new_weight[" << i << "] : "
        << tensor_name << "\n";);
    auto type = RankedTensorType::get({oc},
        FloatType::getF32(rewriter.getContext()));
    wTF->addTensor<float>(tensor_name, newWeights[i], type);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("name",
          rewriter.getStringAttr(tensor_name)));
    auto new_weight_op = rewriter.create<tpu::LoadWeightOp>(loc, type,
        ArrayRef<Value>{wfV}, ArrayRef<NamedAttribute>{attrs});
    newOperands.push_back(new_weight_op);
  }

  // replace bn with scale
  // keep the op_name because the calibration table is using this name
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("name",
        rewriter.getStringAttr(op_name)));

  rewriter.replaceOpWithNewOp<tpu::ScaleOp>(
      bnOp, bnOp.getResult().getType(),
      ArrayRef<Value>{newOperands}, ArrayRef<NamedAttribute>{attrs});

}

class ConvertBnToScalePass : public mlir::PassWrapper<ConvertBnToScalePass, FunctionPass> {
public:
  explicit ConvertBnToScalePass(llvm::raw_ostream &os = llvm::errs())
      : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<TpuBatchNormOpPattern>(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

void tpu::BatchNormOp::getCanonicalizationPatterns(
         OwningRewritePatternList &results,
         MLIRContext *context) {
  results.insert<TpuBatchNormOpPattern>(context);
}

std::unique_ptr<mlir::Pass> mlir::createConvertBnToScalePass() {
  return std::make_unique<ConvertBnToScalePass>();
}

