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

#define DEBUG_TYPE "convert_scale"

using namespace mlir;

namespace {

static llvm::cl::opt<bool> clSkipMultiUsedScaleOp(
    "skip-mult-used-scale-op",
    llvm::cl::desc("Skip Multiple uses for Scale Op"),
    llvm::cl::init(false));


struct TpuFoldScalePattern : public RewritePattern {
  TpuFoldScalePattern(MLIRContext *context)
      : RewritePattern("tpu.scale", 5, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto laterScaleOp = cast<tpu::ScaleOp>(op);
    LLVM_DEBUG(llvm::errs() << laterScaleOp.getOperationName() << "\n";);
    TensorFile *wTF = getWeightTensorFile(op);
    Value *wfV = getWeightFileValue(op);

    // match consecutive scale operations
    auto formerOp = laterScaleOp.getOperand(0)->getDefiningOp();
    if (!isa<tpu::ScaleOp>(formerOp))
      return matchFailure();
    if (clSkipMultiUsedScaleOp && !formerOp->getResult(0)->hasOneUse()) {
      std::string op_name = formerOp->getAttrOfType<StringAttr>("name").getValue().str();
      LLVM_DEBUG(llvm::errs() << "Some one need to use Scale Op: " << op_name << ", not remove it\n";);
      return matchFailure();
    }

    auto formerScaleOp = cast<tpu::ScaleOp>(formerOp);

    // op_name from the later scale
    std::string op_name = laterScaleOp.getAttrOfType<StringAttr>("name").getValue().str();
    LLVM_DEBUG(llvm::errs() << "Scale Op: " << op_name << "\n";);

    // find scale and bias tensor for both later and former scale_op
    std::vector<std::unique_ptr<std::vector<float> > > laterWeights(2);
    for (int i = 0; i < 2; ++i) {
      auto weight_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
          laterScaleOp.getOperand(i + 1)->getDefiningOp());
      assert(weight_op);
      assert(weight_op.name().hasValue());
      auto tensor_name = weight_op.name().getValue();
      LLVM_DEBUG(llvm::errs() << "  weight[" << i << "] : " << tensor_name << "\n";);
      auto type = weight_op.getResult()->getType().cast<TensorType>();
      laterWeights[i] = wTF->readTensor<float>(tensor_name, type);
      // delete the tensor from the weight file
      wTF->deleteTensor<float>(tensor_name);
    }
    std::vector<std::unique_ptr<std::vector<float> > > formerWeights(2);
    for (int i = 0; i < 2; ++i) {
      auto weight_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
          formerScaleOp.getOperand(i + 1)->getDefiningOp());
      assert(weight_op);
      assert(weight_op.name().hasValue());
      auto tensor_name = weight_op.name().getValue();
      LLVM_DEBUG(llvm::errs() << "  weight[" << i << "] : " << tensor_name << "\n";);
      auto type = weight_op.getResult()->getType().cast<TensorType>();
      formerWeights[i] = wTF->readTensor<float>(tensor_name, type);
      // delete the tensor from the weight file
      wTF->deleteTensor<float>(tensor_name);
    }

    // convert tensors
    LLVM_DEBUG(
        llvm::errs() << "  former scale size: " << formerWeights[0]->size() << "\n"
                     << "  former bias size: " << formerWeights[1]->size() << "\n"
                     << "  later scale size: " << laterWeights[0]->size() << "\n"
                     << "  later bias size: " << laterWeights[1]->size() << "\n";);
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
      LLVM_DEBUG(llvm::errs() << "  new_weight[" << i << "] : " << tensor_name << "\n";);
      auto type = RankedTensorType::get({oc}, FloatType::getF32(rewriter.getContext()));
      wTF->addTensor<float>(tensor_name, newWeights[i], type);
      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
      auto new_weight_op = rewriter.create<tpu::LoadWeightOp>(loc, type,
          ArrayRef<Value *>{wfV}, ArrayRef<NamedAttribute>{attrs});
      newOperands.push_back(new_weight_op);
    }

    // replace the later scale with the new scale
    // the former one will be removed automatically
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(op_name)));
    if (laterScaleOp.layer_id().hasValue()) {
      attrs.push_back(rewriter.getNamedAttr("layer_id", laterScaleOp.layer_idAttr()));
    }
    rewriter.replaceOpWithNewOp<tpu::ScaleOp>(
        laterScaleOp, formerScaleOp.getResult()->getType(),
        ArrayRef<Value *>{newOperands}, ArrayRef<NamedAttribute>{attrs});

    return matchSuccess();
  }
};

class FoldScalePass : public FunctionPass<FoldScalePass> {
public:
  explicit FoldScalePass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<TpuFoldScalePattern>(context);
    applyPatternsGreedily(fn, patterns);
  }

private:
  llvm::raw_ostream &os;
};


struct TpuMergeScaleIntoConvPattern : public RewritePattern {
  TpuMergeScaleIntoConvPattern(MLIRContext *context)
      : RewritePattern("tpu.scale", 4, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto scaleOp = cast<tpu::ScaleOp>(op);
    LLVM_DEBUG(llvm::errs() << scaleOp.getOperationName() << "\n";);
    TensorFile *wTF = getWeightTensorFile(op);
    Value *wfV = getWeightFileValue(op);

    // match consecutive scale operations
    auto formerOp = scaleOp.getOperand(0)->getDefiningOp();
    if (!isa<tpu::Conv2DOp>(formerOp))
      return matchFailure();
    if (clSkipMultiUsedScaleOp && !formerOp->getResult(0)->hasOneUse()) {
      std::string op_name = formerOp->getAttrOfType<StringAttr>("name").getValue().str();
      LLVM_DEBUG(llvm::errs() << "Some one need to use Scale Op: " << op_name << ", not remove it\n";);
      return matchFailure();
    }

    auto convOp = cast<tpu::Conv2DOp>(formerOp);

    // op_name from the scale
    std::string op_name = scaleOp.getAttrOfType<StringAttr>("name").getValue().str();
    LLVM_DEBUG(llvm::errs() << "Scale Op: " << op_name << "\n";);

    // find scale and bias tensor for scale op
    std::vector<std::unique_ptr<std::vector<float> > > scaleWeights(2);
    for (int i = 0; i < 2; ++i) {
      auto weight_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
          scaleOp.getOperand(i + 1)->getDefiningOp());
      assert(weight_op);
      assert(weight_op.name().hasValue());
      auto tensor_name = weight_op.name().getValue();
      LLVM_DEBUG(llvm::errs() << "  weight[" << i << "] : " << tensor_name << "\n";);
      auto type = weight_op.getResult()->getType().cast<TensorType>();
      scaleWeights[i] = wTF->readTensor<float>(tensor_name, type);
      // delete the tensor from the weight file
      wTF->deleteTensor<float>(tensor_name);
    }

    // find filter and bias tensor for conv op
    assert(convOp.getNumOperands() == 7);
    std::vector<std::unique_ptr<std::vector<float> > > convWeights(2);
    for (unsigned i = 0; i < 2; ++i) {
      auto weight_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
          convOp.getOperand(i + 1)->getDefiningOp());
      if (!weight_op) {
        convWeights[i] = nullptr;
        continue;
      }
      assert(weight_op.name().hasValue());
      auto tensor_name = weight_op.name().getValue();
      LLVM_DEBUG(llvm::errs() << "  weight[" << i << "] : " << tensor_name << "\n";);
      auto type = weight_op.getResult()->getType().cast<TensorType>();
      convWeights[i] = wTF->readTensor<float>(tensor_name, type);
      // delete the tensor from the weight file
      wTF->deleteTensor<float>(tensor_name);
    }

    // convert tensors
    LLVM_DEBUG(
      llvm::errs() << "  scale scale size: " << scaleWeights[0]->size() << "\n";
      llvm::errs() << "  scale bias size: " << scaleWeights[1]->size() << "\n";
      llvm::errs() << "  conv filter size: " << convWeights[0]->size() << "\n";
      if (convWeights[1]) {
        llvm::errs() << "  conv bias size: " << convWeights[1]->size() << "\n";
      } else {
        llvm::errs() << "  conv has no bias\n";
      }
    );

    auto filter_type = convOp.filter()->getType().cast<TensorType>();
    std::vector<int64_t> filter_shape(filter_type.getShape());
    int64_t filter_size = std::accumulate(std::begin(filter_shape),
        std::end(filter_shape), 1, std::multiplies<>());
    assert(filter_size == (int64_t)convWeights[0]->size());
    int64_t oc, inner_size;
    //assert(filter_shape.size() == 4 || filter_shape.size() == 5);
    if (filter_shape.size() == 4) {
      oc = filter_shape[0];
      inner_size = filter_shape[1] * filter_shape[2] * filter_shape[3];
    } else if (filter_shape.size() == 5) {
      // g, oc/g, ic/g, kh, kw
      oc = filter_shape[0] * filter_shape[1];
      inner_size = filter_shape[2] * filter_shape[3] * filter_shape[4];
    } else {
      assert(0);
    }
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
      auto tensor_name = op_name + "_merge_scale_" + std::to_string(i);
      LLVM_DEBUG(llvm::errs() << "  new_weight[" << i << "] : " << tensor_name << "\n";);
      auto type = RankedTensorType::get(weightShapes[i], FloatType::getF32(rewriter.getContext()));
      wTF->addTensor<float>(tensor_name, newWeights[i], type);
      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
      auto new_weight_op = rewriter.create<tpu::LoadWeightOp>(loc, type,
          ArrayRef<Value *>{wfV}, ArrayRef<NamedAttribute>{attrs});
      newOperands.push_back(new_weight_op);
    }
    newOperands.push_back(convOp.getOperand(3));
    newOperands.push_back(convOp.getOperand(4));
    newOperands.push_back(convOp.getOperand(5));
    newOperands.push_back(convOp.getOperand(6));

    // replace the scale with the new conv op
    // the former conv op one will be removed automatically
    //convOp.param().setAttr("with_bias", rewriter.getBoolAttr(true));
    convOp.setAttr("param",
        tpu::ConvParam::get(
            convOp.param().stride_h(),
            convOp.param().stride_w(),
            convOp.param().padding(),
            convOp.param().dilation_h(),
            convOp.param().dilation_w(),
            convOp.param().group(),
            convOp.param().is_dw(),
            rewriter.getBoolAttr(true),
            convOp.param().do_relu(),
            rewriter.getContext()));
    auto origAttrs = convOp.getAttrs();
    //update name with the later op name, because this name is for calibration table
    std::vector<NamedAttribute> newAttrs(origAttrs.begin(), origAttrs.end());
    for (auto &elt : newAttrs) {
      if (elt.first == "name") {
        elt.second = rewriter.getStringAttr(op_name);
      }
    }
    rewriter.replaceOpWithNewOp<tpu::Conv2DOp>(
        scaleOp, convOp.getResult()->getType(),
        ArrayRef<Value *>{newOperands}, ArrayRef<NamedAttribute>{newAttrs});

    return matchSuccess();
  }
};

class MergeScaleIntoConvPass : public FunctionPass<MergeScaleIntoConvPass> {
public:
  explicit MergeScaleIntoConvPass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<TpuMergeScaleIntoConvPattern>(context);
    applyPatternsGreedily(fn, patterns);
  }

private:
  llvm::raw_ostream &os;
};

struct TpuConvertScaleToDWConvPattern : public RewritePattern {
  TpuConvertScaleToDWConvPattern(MLIRContext *context)
      : RewritePattern("tpu.scale", 3, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto scaleOp = cast<tpu::ScaleOp>(op);
    LLVM_DEBUG(llvm::errs() << scaleOp.getOperationName() << "\n";);
    TensorFile *wTF = getWeightTensorFile(op);

    // op_name
    std::string op_name = scaleOp.name().str();
    LLVM_DEBUG(llvm::errs() << "Scale Op: " << op_name << "\n";);

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
    patterns.insert<TpuConvertScaleToDWConvPattern>(context);
    applyPatternsGreedily(fn, patterns);
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

// Canonicalizer
void tpu::ScaleOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<
      TpuFoldScalePattern,
      TpuMergeScaleIntoConvPattern,
      TpuConvertScaleToDWConvPattern>(context);
}

// register passes
std::unique_ptr<OpPassBase<FuncOp>> mlir::createFoldScalePass() {
  return std::make_unique<FoldScalePass>();
}

static PassRegistration<FoldScalePass>
    pass_1("fold-scale",
         "Fold two consecutive scale operations into one");

std::unique_ptr<OpPassBase<FuncOp>> mlir::createMergeScaleIntoConvPass() {
  return std::make_unique<MergeScaleIntoConvPass>();
}

static PassRegistration<MergeScaleIntoConvPass>
    pass_2("merge-scale-into-conv",
         "Merge scale op into conv op");

std::unique_ptr<OpPassBase<FuncOp>> mlir::createConvertScaleToDWConvPass() {
  return std::make_unique<ConvertScaleToDWConvPass>();
}

static PassRegistration<ConvertScaleToDWConvPass>
    pass_3("convert-scale-to-dwconv",
         "Convert a scale operation to a dwconv operation");
