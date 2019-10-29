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

struct TpuFuseScaleIntoConvPattern : public RewritePattern {
  TpuFuseScaleIntoConvPattern(MLIRContext *context, TensorFile *weightTensorFile)
      : RewritePattern("tpu.scale", 1, context),
        weightTensorFile_(weightTensorFile) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto scaleOp = cast<tpu::ScaleOp>(op);
    llvm::errs() << scaleOp.getOperationName() << "\n";

    // match consecutive scale operations
    auto formerOp = scaleOp.getOperand(0)->getDefiningOp();
    if (!matchPattern(formerOp, m_Op<tpu::Conv2DOp>()))
      return matchFailure();
    auto convOp = cast<tpu::Conv2DOp>(formerOp);

    // op_name from the scale
    std::string op_name = scaleOp.getAttrOfType<StringAttr>("name").getValue().str();
    llvm::errs() << "Scale Op: " << op_name << "\n";
    auto one_weight_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
          scaleOp.getOperand(1)->getDefiningOp());
    auto weightFileVar = one_weight_op.getOperand();

    // find scale and bias tensor for scale op
    std::vector<std::unique_ptr<std::vector<float> > > scaleWeights(2);
    for (int i = 0; i < 2; ++i) {
      auto weight_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
          scaleOp.getOperand(i + 1)->getDefiningOp());
      assert(weight_op);
      assert(weight_op.name().hasValue());
      auto tensor_name = weight_op.name().getValue();
      llvm::errs() << "  weight[" << i << "] : " << tensor_name << "\n";
      auto type = weight_op.getResult()->getType().cast<TensorType>();
      scaleWeights[i] = weightTensorFile_->readTensor<float>(tensor_name, type);
      // delete the tensor from the weight file
      weightTensorFile_->deleteTensor<float>(tensor_name);
    }

    // find filter and bias tensor for conv op
    std::vector<std::unique_ptr<std::vector<float> > > convWeights(2);
    for (unsigned i = 0; i < convOp.getNumOperands() - 1; ++i) {
      auto weight_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
          convOp.getOperand(i + 1)->getDefiningOp());
      assert(weight_op);
      assert(weight_op.name().hasValue());
      auto tensor_name = weight_op.name().getValue();
      llvm::errs() << "  weight[" << i << "] : " << tensor_name << "\n";
      auto type = weight_op.getResult()->getType().cast<TensorType>();
      convWeights[i] = weightTensorFile_->readTensor<float>(tensor_name, type);
      // delete the tensor from the weight file
      weightTensorFile_->deleteTensor<float>(tensor_name);
    }

    // convert tensors
    llvm::errs() << "  scale scale size: " << scaleWeights[0]->size() << "\n";
    llvm::errs() << "  scale bias size: " << scaleWeights[1]->size() << "\n";
    llvm::errs() << "  conv filter size: " << convWeights[0]->size() << "\n";
    if (convWeights[1]) {
      llvm::errs() << "  conv bias size: " << convWeights[1]->size() << "\n";
    } else {
      llvm::errs() << "  conv has no bias\n";
    }

    auto filter_type = convOp.filter()->getType().cast<TensorType>();
    std::vector<int64_t> filter_shape(filter_type.getShape());
    assert(filter_shape.size() == 4);
    int64_t filter_size = std::accumulate(std::begin(filter_shape),
        std::end(filter_shape), 1, std::multiplies<>());
    assert(filter_size == (int64_t)convWeights[0]->size());
    int64_t oc = filter_shape[0];
    int64_t inner_size = filter_shape[1] * filter_shape[2] * filter_shape[3];
    assert(oc == (int64_t)scaleWeights[0]->size());
    std::vector<float> new_filter(filter_size);
    std::vector<float> new_bias(oc);

    float *filter = (float *)convWeights[0]->data();
    float *bias = nullptr;
    if (convWeights[1]) {
      bias = (float *)convWeights[1]->data();
    }
    float *scale = (float *)scaleWeights[0]->data();
    float *scale_bias = (float *)scaleWeights[1]->data();

    for (int i = 0; i < oc; ++i) {
      for (int j = 0; j < inner_size; ++j) {
        new_filter[i * inner_size + j] = filter[i * inner_size + j] * scale[i];
      }
      new_bias[i] = scale_bias[i];
      if (bias) {
        new_bias[i] += bias[i] * scale[i];
      }
    }

    std::vector<std::vector<float> *> newWeights{ &new_filter, &new_bias };
    std::vector<std::vector<int64_t> > weightShapes{ filter_shape, std::vector<int64_t>{oc} };

    std::vector<Value *> newOperands;
    newOperands.push_back(convOp.getOperand(0));
    // add new filter and bias weight ops
    for (int i = 0; i < 2; ++i) {
      auto tensor_name = op_name + "_fuse_scale_" + std::to_string(i);
      llvm::errs() << "  new_weight[" << i << "] : " << tensor_name << "\n";
      auto type = rewriter.getTensorType(weightShapes[i], FloatType::getF32(rewriter.getContext()));
      weightTensorFile_->addTensor<float>(tensor_name, newWeights[i], type);
      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
      auto new_weight_op = rewriter.create<tpu::LoadWeightOp>(loc, type,
          ArrayRef<Value *>{weightFileVar}, ArrayRef<NamedAttribute>{attrs});
      newOperands.push_back(new_weight_op);
    }

    // replace the scale with the new conv op
    // the former conv op one will be removed automatically
    auto origAttrs = convOp.getAttrs();
    //update name with the later op name, because this name is for calibration table
    std::vector<NamedAttribute> newAttrs(origAttrs.begin(), origAttrs.end());
    for (auto &elt : newAttrs) {
      if (elt.first == "name") {
        elt.second = rewriter.getStringAttr(op_name);
      }
    }
    //attrs.set("name", rewriter.getStringAttr(op_name));
    //attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(op_name)));
    rewriter.replaceOpWithNewOp<tpu::Conv2DOp>(
        scaleOp, convOp.getResult()->getType(),
        ArrayRef<Value *>{newOperands}, ArrayRef<NamedAttribute>{newAttrs});
    return matchSuccess();
  }

  TensorFile *weightTensorFile_;
};

class FuseScaleIntoConvPass : public FunctionPass<FuseScaleIntoConvPass> {
public:
  explicit FuseScaleIntoConvPass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

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
    patterns.insert<TpuFuseScaleIntoConvPattern>(context, weightTensorFile.get());
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

std::unique_ptr<FunctionPassBase> mlir::createFuseScaleIntoConvPass() {
  return std::make_unique<FuseScaleIntoConvPass>();
}

static PassRegistration<FuseScaleIntoConvPass>
    pass("fuse-scale-into-conv",
         "Fuse scale op into conv op");
