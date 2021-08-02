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
// This file implements the conversion of scale.
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

#include <sstream>

#define DEBUG_TYPE "convert_scale"

using namespace mlir;

namespace {

struct TpuRemoveScalePattern : public RewritePattern {
  TpuRemoveScalePattern(MLIRContext *context)
      : RewritePattern("tpu.scale", 6, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto castOp = cast<tpu::ScaleOp>(op);
    TensorFile *wTF = getWeightTensorFile(op);

    Operation *formerOp = op->getOperand(0).getDefiningOp();
    if (!formerOp->getResult(0).hasOneUse()) {
      return failure();
    }

    // check whether is only one scale
    auto scale_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
        castOp.getOperand(1).getDefiningOp());
    if (scale_op == nullptr) {
      return failure();
    }
    auto scale_name = scale_op.name();
    auto scale_type = scale_op.getResult().getType().cast<TensorType>();
    auto scale_weight = wTF->readTensor<float>(scale_name, scale_type);
    float scale = scale_weight->at(0);
    for (auto &data : *scale_weight) {
      if (data != scale) {
        return failure();
      }
    }

    auto bias_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
        castOp.getOperand(2).getDefiningOp());
    llvm::StringRef bias_name;
    if (bias_op != nullptr) {
      bias_name = bias_op.name();
      auto bias_type = bias_op.getResult().getType().cast<TensorType>();
      auto bias_weight = wTF->readTensor<float>(bias_name, bias_type);
      float bias = bias_weight->at(0);
      if (bias != 0.0f) {
        return failure();
      }
      for (auto &data : *bias_weight) {
        if (data != bias) {
          return failure();
        }
      }
    }

    if (scale == 1.0f) {
      wTF->deleteTensor<float>(scale_name);
      if (bias_op != nullptr) {
        wTF->deleteTensor<float>(bias_name);
      }
      rewriter.replaceOp(op, op->getOperand(0));
      return success();
    }
    do {
      if (!formerOp->getResult(0).hasOneUse()) {
        return failure();
      }
      if (isa<tpu::FullyConnectedOp>(formerOp)) {
        break;
      }
      if (isa<tpu::PermuteOp>(formerOp) || isa<tpu::ReshapeOp>(formerOp)) {
        formerOp = formerOp->getOperand(0).getDefiningOp();
        continue;
      }
      return failure();
    } while (formerOp != nullptr);

    auto fcOp = cast<tpu::FullyConnectedOp>(formerOp);
    std::vector<int64_t> shape = getTensorShape(fcOp.filter());
    auto filter = readAndDeleteWeightTensor<float>(fcOp.filter(), wTF);
    for (auto &data : *filter) {
      data *= scale;
    }
    addWeightTensorAndUpdateWeightOp<float>(fcOp.filter(), "_scale", *filter,
                                            shape, "NONE", wTF);
    if (isTensorNone(fcOp.bias()) == false) {
      shape = getTensorShape(fcOp.bias());
      auto bias = readAndDeleteWeightTensor<float>(fcOp.bias(), wTF);
      for (auto &data : *bias) {
        data *= scale;
      }
      addWeightTensorAndUpdateWeightOp<float>(fcOp.bias(), "_scale", *bias,
                                              shape, "NONE", wTF);
    }
    // fix name
    fcOp->setAttr("name", rewriter.getStringAttr(fcOp.name().str() + "_scale"));
    formerOp = op->getOperand(0).getDefiningOp();
    do {
      if (auto castOp = llvm::dyn_cast<tpu::ReshapeOp>(formerOp)) {
        castOp->setAttr("name",
                        rewriter.getStringAttr(castOp.name().str() + "_scale"));
      } else if (auto castOp = llvm::dyn_cast<tpu::PermuteOp>(formerOp)) {
        castOp->setAttr("name",
                        rewriter.getStringAttr(castOp.name().str() + "_scale"));
      } else {
        break;
      }
      formerOp = formerOp->getOperand(0).getDefiningOp();
    } while (formerOp != nullptr);

    wTF->deleteTensor<float>(scale_name);
    if (bias_op != nullptr) {
      wTF->deleteTensor<float>(bias_name);
    }
    rewriter.replaceOp(op, op->getOperand(0));
    return success();
  }
};

struct TpuFoldScalePattern : public RewritePattern {
  TpuFoldScalePattern(MLIRContext *context)
      : RewritePattern("tpu.scale", 5, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto laterScaleOp = cast<tpu::ScaleOp>(op);
    LLVM_DEBUG(llvm::errs() << laterScaleOp.getOperationName() << "\n";);
    TensorFile *wTF = getWeightTensorFile(op);
    Value wfV = getWeightFileValue(op);

    // match consecutive scale operations
    auto formerOp = laterScaleOp.getOperand(0).getDefiningOp();
    if (!isa<tpu::ScaleOp>(formerOp)) {
      return failure();
    }

    if (!formerOp->getResult(0).hasOneUse()) {
      return failure();
    }

    auto formerScaleOp = cast<tpu::ScaleOp>(formerOp);

    // op_name from the later scale
    auto nameAttr = laterScaleOp->getAttrOfType<StringAttr>("name");
    std::string op_name = nameAttr.getValue().str();
    LLVM_DEBUG(llvm::errs() << "Scale Op: " << op_name << "\n";);

    // find scale and bias tensor for both later and former scale_op
    int oc = 0;
    std::unique_ptr<std::vector<float>> laterWeight;
    std::unique_ptr<std::vector<float>> laterBias;
    do {
      auto weight_op = dyn_cast_or_null<tpu::LoadWeightOp>(laterScaleOp.getOperand(1).getDefiningOp());
      auto weight_type = weight_op.getResult().getType().cast<TensorType>();
      laterWeight = wTF->readTensor<float>(weight_op.name(), weight_type);
      wTF->deleteTensor<float>(weight_op.name());
      oc = (int)laterWeight->size();
      // bias
      auto bias_op = dyn_cast_or_null<tpu::LoadWeightOp>(laterScaleOp.getOperand(2).getDefiningOp());
      if (bias_op) {
        auto bias_type = weight_op.getResult().getType().cast<TensorType>();
        laterBias = wTF->readTensor<float>(bias_op.name(), bias_type);
        wTF->deleteTensor<float>(bias_op.name());
      } else {
        laterBias = std::make_unique<std::vector<float>>(oc, 0);
      }
    } while (0);

    std::unique_ptr<std::vector<float>> formerWeight;
    std::unique_ptr<std::vector<float>> formerBias;
    do {
      auto weight_op = dyn_cast_or_null<tpu::LoadWeightOp>(formerScaleOp.getOperand(1).getDefiningOp());
      auto weight_type = weight_op.getResult().getType().cast<TensorType>();
      formerWeight = wTF->readTensor<float>(weight_op.name(), weight_type);
      wTF->deleteTensor<float>(weight_op.name());
      // bias
      auto bias_op = dyn_cast_or_null<tpu::LoadWeightOp>(formerScaleOp.getOperand(2).getDefiningOp());
      if (bias_op) {
        auto bias_type = weight_op.getResult().getType().cast<TensorType>();
        formerBias = wTF->readTensor<float>(bias_op.name(), bias_type);
        wTF->deleteTensor<float>(bias_op.name());
      } else {
        formerBias = std::make_unique<std::vector<float>>(oc, 0);
      }
    } while (0);

    std::vector<float> new_scale(oc);
    std::vector<float> new_bias(oc);

    float *former_scale = (float *)formerWeight->data();
    float *former_bias = (float *)formerBias->data();
    float *later_scale = (float *)laterWeight->data();
    float *later_bias = (float *)laterBias->data();

    for (int i = 0; i < oc; ++i) {
      new_scale[i] = former_scale[i] * later_scale[i];
    }
    bool is_zero_bias = true;
    for (int i = 0; i < oc; ++i) {
      new_bias[i] = former_bias[i] * later_scale[i] + later_bias[i];
      if (new_bias[i] != 0) {
        is_zero_bias = false;
      }
    }

    std::vector<Value> newOperands;
    newOperands.push_back(formerScaleOp.getOperand(0));

    auto new_weight_name = op_name + "_fold_w";
    auto new_weight_type = RankedTensorType::get({oc},
                                  FloatType::getF32(rewriter.getContext()));
    wTF->addTensor<float>(new_weight_name, &new_scale, new_weight_type);
    std::vector<NamedAttribute> new_weight_attrs;
    new_weight_attrs.push_back(rewriter.getNamedAttr("name",
                    rewriter.getStringAttr(new_weight_name)));
    auto new_weight_op = rewriter.create<tpu::LoadWeightOp>(loc, new_weight_type,
        ArrayRef<Value>{wfV}, ArrayRef<NamedAttribute>{new_weight_attrs});
    newOperands.push_back(new_weight_op);

    if (is_zero_bias) {
      auto noneOp = rewriter.create<tpu::NoneOp>(loc, rewriter.getNoneType());
      newOperands.push_back(noneOp);
    } else {
      auto new_bias_name = op_name + "_fold_b";
      auto new_bias_type = RankedTensorType::get({oc},
                               FloatType::getF32(rewriter.getContext()));
      wTF->addTensor<float>(new_bias_name, &new_bias, new_bias_type);
      std::vector<NamedAttribute> new_bias_attrs;
      new_bias_attrs.push_back(rewriter.getNamedAttr("name",
                      rewriter.getStringAttr(new_bias_name)));
      auto new_bias_op = rewriter.create<tpu::LoadWeightOp>(loc, new_bias_type,
          ArrayRef<Value>{wfV}, ArrayRef<NamedAttribute>{new_bias_attrs});
      newOperands.push_back(new_bias_op);
    }

    // replace the later scale with the new scale
    // the former one will be removed automatically
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("name",
                    rewriter.getStringAttr(op_name)));
    // replace former scale op with new scale op
    rewriter.replaceOpWithNewOp<tpu::ScaleOp>(
        formerScaleOp, formerScaleOp.getResult().getType(),
        ArrayRef<Value>{newOperands}, ArrayRef<NamedAttribute>{attrs});
    // delete later scale op
    op->getResult(0).replaceAllUsesWith(op->getOperand(0));
    rewriter.eraseOp(op);

    return success();
  }
};

class FoldScalePass : public mlir::PassWrapper<FoldScalePass, FunctionPass> {
public:
  explicit FoldScalePass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<TpuFoldScalePattern>(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));
  }

private:
  llvm::raw_ostream &os;
};


struct TpuMergeScaleIntoConvPattern : public RewritePattern {
  TpuMergeScaleIntoConvPattern(MLIRContext *context)
      : RewritePattern("tpu.scale", 4, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto scaleOp = cast<tpu::ScaleOp>(op);
    LLVM_DEBUG(llvm::errs() << scaleOp.getOperationName() << "\n";);
    TensorFile *wTF = getWeightTensorFile(op);
    Value wfV = getWeightFileValue(op);

    // match consecutive scale operations
    auto formerOp = scaleOp.getOperand(0).getDefiningOp();
    if (!isa<tpu::Conv2DOp>(formerOp)) {
      return failure();
    }
    if (!formerOp->getResult(0).hasOneUse()) {
      return failure();
    }

    auto convOp = cast<tpu::Conv2DOp>(formerOp);

    // op_name from the scale
    auto nameAttr = scaleOp->getAttrOfType<StringAttr>("name");
    std::string op_name = nameAttr.getValue().str();
    LLVM_DEBUG(llvm::errs() << "Scale Op: " << op_name << "\n";);

    // find scale and bias tensor for scale op
    std::vector<std::unique_ptr<std::vector<float> > > scaleWeights(2);

    // if no bias, we make fake one
    int weight_nr = 1;

    if (auto bias = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(scaleOp.getOperand(2).getDefiningOp())) {
      // has bias
      weight_nr = 2;
    }
    else {
      std::vector<int64_t> shape;
      int64_t input_size, n, c, h, w;
      getTensorShapeAndSize(scaleOp.input(), shape, input_size);
      getNCHW(shape, n, c, h, w);

      // bias shape should be as same as channel, fill with 0 for non-bias case
      scaleWeights[1] = std::make_unique<std::vector<float>> (c, 0);
    }

    for (int i = 0; i < weight_nr; ++i) {
      auto weight_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
          scaleOp.getOperand(i + 1).getDefiningOp());
      assert(weight_op && "weight op should be exist");
      auto tensor_name = weight_op.name();
      LLVM_DEBUG(llvm::errs() << "  weight[" << i << "] : "
                              << tensor_name << "\n";);
      auto type = weight_op.getResult().getType().cast<TensorType>();
      scaleWeights[i] = wTF->readTensor<float>(tensor_name, type);
      // delete the tensor from the weight file
      wTF->deleteTensor<float>(tensor_name);
    }

    // find filter and bias tensor for conv op
    assert(convOp.getNumOperands() == 7 && "operands num should be 7");
    std::vector<std::unique_ptr<std::vector<float> > > convWeights(2);
    for (unsigned i = 0; i < 2; ++i) {
      auto weight_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
          convOp.getOperand(i + 1).getDefiningOp());
      if (!weight_op) {
        convWeights[i] = nullptr;
        continue;
      }
      auto tensor_name = weight_op.name();
      LLVM_DEBUG(llvm::errs() << "  weight[" << i << "] : "
                              << tensor_name << "\n";);
      auto type = weight_op.getResult().getType().cast<TensorType>();
      convWeights[i] = wTF->readTensor<float>(tensor_name, type);
      // delete the tensor from the weight file
      // wTF->deleteTensor<float>(tensor_name);
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

    auto filter_type = convOp.filter().getType().cast<TensorType>();
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
      llvm_unreachable("unsupported shape size");
    }
    if (oc != (int64_t)scaleWeights[0]->size()) {
      std::stringstream err_msg;
      err_msg << op_name << " oc v.s. scaleWeights (" << oc << " v.s "
              << (int64_t)scaleWeights[0]->size() << ") size not equal";
      llvm_unreachable(err_msg.str().c_str());
    }
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
    std::vector<std::vector<int64_t> > weightShapes{ filter_shape,
                                                     std::vector<int64_t>{oc} };

    std::vector<Value> newOperands;
    newOperands.push_back(convOp.getOperand(0));
    // add new filter and bias weight ops
    for (int i = 0; i < 2; ++i) {
      auto tensor_name = op_name + "_merge_scale_" + std::to_string(i);
      LLVM_DEBUG(llvm::errs() << "  new_weight[" << i << "] : "
                              << tensor_name << "\n";);
      auto type = RankedTensorType::get(weightShapes[i],
                                    FloatType::getF32(rewriter.getContext()));
      wTF->addTensor<float>(tensor_name, newWeights[i], type);
      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr("name",
                      rewriter.getStringAttr(tensor_name)));
      auto new_weight_op = rewriter.create<tpu::LoadWeightOp>(loc, type,
          ArrayRef<Value>{wfV}, ArrayRef<NamedAttribute>{attrs});
      newOperands.push_back(new_weight_op);
    }
    newOperands.push_back(convOp.getOperand(3));
    newOperands.push_back(convOp.getOperand(4));
    newOperands.push_back(convOp.getOperand(5));
    newOperands.push_back(convOp.getOperand(6));

    // replace the scale with the new conv op
    // the former conv op one will be removed automatically
    //convOp.param()->setAttr("with_bias", rewriter.getBoolAttr(true));
    convOp->setAttr("param",
        tpu::ConvParam::get(
            convOp.param().stride_h(),
            convOp.param().stride_w(),
            convOp.param().padding(),
            convOp.param().dilation_h(),
            convOp.param().dilation_w(),
            convOp.param().padding_t(),
            convOp.param().padding_b(),
            convOp.param().padding_l(),
            convOp.param().padding_r(),
            convOp.param().group(),
            convOp.param().is_dw(),
            rewriter.getBoolAttr(true),
            convOp.param().do_relu(),
            convOp.param().ins(),
            convOp.param().pad_value(),
            rewriter.getContext()));
    auto origAttrs = convOp->getAttrs();
    // update name with the later op name, because this name is for
    // calibration table
    std::vector<NamedAttribute> newAttrs(origAttrs.begin(), origAttrs.end());
    for (auto &elt : newAttrs) {
      if (elt.first == "name") {
        elt.second = rewriter.getStringAttr(op_name);
      }
    }
    // replace former conv op with new conv op
    rewriter.replaceOpWithNewOp<tpu::Conv2DOp>(
        convOp, convOp.getResult().getType(),
        ArrayRef<Value>{newOperands}, ArrayRef<NamedAttribute>{newAttrs});
    // delete later scale op
    op->getResult(0).replaceAllUsesWith(op->getOperand(0));
    rewriter.eraseOp(op);

    return success();
  }
};

class MergeScaleIntoConvPass : public mlir::PassWrapper<MergeScaleIntoConvPass, FunctionPass> {
public:
  explicit MergeScaleIntoConvPass(llvm::raw_ostream &os = llvm::errs())
     : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<TpuMergeScaleIntoConvPattern>(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));
  }

private:
  llvm::raw_ostream &os;
};

struct TpuConvertScaleToDWConvPattern : public RewritePattern {
  TpuConvertScaleToDWConvPattern(MLIRContext *context)
      : RewritePattern("tpu.scale", 3, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto scaleOp = cast<tpu::ScaleOp>(op);
    LLVM_DEBUG(llvm::errs() << scaleOp.getOperationName() << "\n";);
    TensorFile *wTF = getWeightTensorFile(op);

    // op_name
    std::string op_name = scaleOp.name().str();
    LLVM_DEBUG(llvm::errs() << "Scale Op: " << op_name << "\n";);

    // parse param
    std::vector<int64_t> shape;
    int64_t input_size, n, c, d, h, w;
    bool convertToConv3d = false;
    getTensorShapeAndSize(scaleOp.input(), shape, input_size);
    if (shape.size() == 5) {
      getNCDHW(shape, n, c, d, h, w);
      convertToConv3d = true;
    } else {
      getNCHW(shape, n, c, h, w);
    }

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

    std::vector<Value> operands;
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
    attrs.push_back(rewriter.getNamedAttr("name",
                    rewriter.getStringAttr(op_name)));
    if (!convertToConv3d) {
      attrs.push_back(rewriter.getNamedAttr("param",
          tpu::ConvParam::get(
              rewriter.getI32IntegerAttr(1),
              rewriter.getI32IntegerAttr(1),
              rewriter.getStringAttr("VALID"),
              rewriter.getI32IntegerAttr(1),
              rewriter.getI32IntegerAttr(1),
              rewriter.getI32IntegerAttr(0), // pd_t
              rewriter.getI32IntegerAttr(0), // pd_b
              rewriter.getI32IntegerAttr(0), // pd_l
              rewriter.getI32IntegerAttr(0), // pd_r
              rewriter.getI32IntegerAttr(c),
              rewriter.getBoolAttr(true),
              rewriter.getBoolAttr(bias?true:false),
              rewriter.getBoolAttr(scaleOp.do_relu()),
              rewriter.getI32ArrayAttr(ArrayRef<int32_t>({})), // [0]ins_w/[1]ins_h
              rewriter.getI32IntegerAttr(0), // pad_value
              rewriter.getContext())));
      attrs.push_back(rewriter.getNamedAttr("quant",
                                            getDefaultQuantParam(rewriter)));

      rewriter.replaceOpWithNewOp<tpu::Conv2DOp>(
          scaleOp, scaleOp.getResult().getType(),
          ArrayRef<Value>{operands}, ArrayRef<NamedAttribute>{attrs});
    } else {
      attrs.push_back(rewriter.getNamedAttr("param",
          tpu::Conv3dParam::get(
              rewriter.getI32IntegerAttr(1),
              rewriter.getI32IntegerAttr(1),
              rewriter.getI32IntegerAttr(1),
              rewriter.getStringAttr("SAME"),
              rewriter.getI32IntegerAttr(1),
              rewriter.getI32IntegerAttr(1),
              rewriter.getI32IntegerAttr(1),
              rewriter.getI32IntegerAttr(0), // pd_d0
              rewriter.getI32IntegerAttr(0), // pd_d1
              rewriter.getI32IntegerAttr(0), // pd_t
              rewriter.getI32IntegerAttr(0), // pd_b
              rewriter.getI32IntegerAttr(0), // pd_l
              rewriter.getI32IntegerAttr(0), // pd_r
              rewriter.getI32IntegerAttr(c),
              rewriter.getBoolAttr(true),
              rewriter.getBoolAttr(bias?true:false),
              rewriter.getBoolAttr(scaleOp.do_relu()),
              rewriter.getI32ArrayAttr(ArrayRef<int32_t>({})), // [0]ins_w/[1]ins_h
              rewriter.getContext())));
      attrs.push_back(rewriter.getNamedAttr("quant",
                                            getDefaultQuantParam(rewriter)));

      rewriter.replaceOpWithNewOp<tpu::Conv3DOp>(
          scaleOp, scaleOp.getResult().getType(),
          ArrayRef<Value>{operands}, ArrayRef<NamedAttribute>{attrs});
    }

    return success();
  }
};

class ConvertScaleToDWConvPass : public mlir::PassWrapper<ConvertScaleToDWConvPass, FunctionPass> {
public:
  explicit ConvertScaleToDWConvPass(llvm::raw_ostream &os = llvm::errs())
     : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<TpuConvertScaleToDWConvPattern>(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

// Canonicalizer
void tpu::ScaleOp::getCanonicalizationPatterns(
                                              OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<
      TpuRemoveScalePattern,
      TpuFoldScalePattern,
      TpuMergeScaleIntoConvPattern,
      TpuConvertScaleToDWConvPattern>(context);
}

// register passes
std::unique_ptr<mlir::Pass> mlir::createFoldScalePass() {
  return std::make_unique<FoldScalePass>();
}

std::unique_ptr<mlir::Pass> mlir::createMergeScaleIntoConvPass() {
  return std::make_unique<MergeScaleIntoConvPass>();
}

std::unique_ptr<mlir::Pass> mlir::createConvertScaleToDWConvPass() {
  return std::make_unique<ConvertScaleToDWConvPass>();
}
