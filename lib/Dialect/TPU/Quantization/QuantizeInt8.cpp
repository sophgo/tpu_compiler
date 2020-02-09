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
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/QuantizationArithmetic.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

#include <sstream>
#include <fstream>

#define DEBUG_TYPE "quantize_int8"

using namespace mlir;

static llvm::cl::OptionCategory clOptionsCategory("quantization options");

static llvm::cl::opt<bool> clQuantConvPerChannel(
    "enable-conv-per-channel",
    llvm::cl::desc("Enable per channel quantization for convolution weight"),
    llvm::cl::init(false),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<bool> clQuantConvMultiplier(
    "enable-conv-multiplier",
    llvm::cl::desc("Enable per channel multiplier quantization for convolution"),
    llvm::cl::init(false),
    llvm::cl::cat(clOptionsCategory));

namespace {

static std::unique_ptr<std::vector<float> > readAndDeleteWeightTensor(
    Value *opd, TensorFile *wTF) {
  auto weightOp = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
      opd->getDefiningOp());
  assert(weightOp);
  assert(weightOp.name().hasValue());
  auto name = weightOp.name().getValue();
  LLVM_DEBUG(llvm::errs() << "  weight : " << name << "\n";);
  auto type = weightOp.getResult()->getType().cast<TensorType>();
  auto T = wTF->readTensor<float>(name, type);
  // delete the tensor from the weight file
  wTF->deleteTensor<float>(name);
  return std::move(T);
}

static void addWeightTensorAndUpdateWeightOp(Value* opd,
    std::vector<float> &weight, std::vector<int64_t> &shape,
    std::string &storageType, Type &eltType,
    PatternRewriter &rewriter, TensorFile *wTF) {
  auto weightOp = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
      opd->getDefiningOp());
  auto name = weightOp.name().getValue().str() + "_quant_int8";
  LLVM_DEBUG(llvm::errs() << "  new_weight : " << name << "\n";);
  auto type = RankedTensorType::get(shape, eltType);
  wTF->addTensor<float>(name, &weight, type);
  weightOp.setAttr("name", rewriter.getStringAttr(name));
  weightOp.setAttr("storage", rewriter.getStringAttr(storageType));
  weightOp.getResult()->setType(type);
}

static Value* addWeightTensorAndCreateWeightOp(std::string name,
    std::vector<float> &weight, std::vector<int64_t> &shape,
    std::string &storageType, Type &eltType,
    PatternRewriter &rewriter, Location &loc,
    TensorFile *wTF, Value *wFV) {
  LLVM_DEBUG(llvm::errs() << "  new_weight[rshift] : " << name << "\n";);
  auto type = RankedTensorType::get(shape, eltType);
  wTF->addTensor<float>(name, &weight, type);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(name)));
  attrs.push_back(rewriter.getNamedAttr("storage",
      rewriter.getStringAttr(storageType)));
  return rewriter.create<tpu::LoadWeightOp>(loc, type,
      ArrayRef<Value *>{wFV}, ArrayRef<NamedAttribute>{attrs});
}

typedef enum {
  INT8_PER_LAYER   = 01,
  INT8_PER_CHANNEL = 02,
  INT8_MULTIPLER   = 03
} QUANT_INT8_TYPE_e;

struct TpuQuantConv2DOpPattern : public RewritePattern {
  TpuQuantConv2DOpPattern(MLIRContext *context, TensorFile *weightTensorFile,
      Value* weightFileVar)
      : RewritePattern("tpu.conv_2d", 1, context),
        weightTensorFile_(weightTensorFile),
        weightFileVar_(weightFileVar) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto convOp = cast<tpu::Conv2DOp>(op);
    auto loc = op->getLoc();

    if (convOp.quant() != "NONE") {
      LLVM_DEBUG(llvm::errs() << convOp.name() << " quantized already\n";);
      return matchFailure();
    }
    assert(convOp.per_channel_info_is_aggregated() == false);

    // get quant type
    QUANT_INT8_TYPE_e quant;
    if (!clQuantConvPerChannel) {
      assert(!clQuantConvMultiplier
             && "enable per channel before enable multiplier");
      quant = INT8_PER_LAYER;
    } else if (!clQuantConvMultiplier) {
      quant = INT8_PER_CHANNEL;
    } else {
      quant = INT8_MULTIPLER;
    }

    // get threshold
    float threshold_x = getPreviousOpThreshold(op);
    float threshold_y;
    if (convOp.fused_eltwise_method() == "NONE") {
      threshold_y = convOp.threshold_y().getValue().convertToFloat();
    } else {
      threshold_y =
          convOp.threshold_y_before_eltwise().getValue().convertToFloat();
    }
    LLVM_DEBUG(llvm::errs() << " > " << convOp.name()
                 << ", threshold_y = "<< std::to_string(threshold_y)
                 << ", threshold_x = " << std::to_string(threshold_x) << "\n";);

    // get filter tensor
    auto filter = readAndDeleteWeightTensor(convOp.getOperand(1),
        weightTensorFile_);
    auto filterType = convOp.filter()->getType().cast<TensorType>();
    std::vector<int64_t> filterShape(filterType.getShape());
    int64_t filterSize = std::accumulate(std::begin(filterShape),
        std::end(filterShape), 1, std::multiplies<>());
    assert(filterSize == (int64_t)filter->size());
    int64_t oc = 0;
    if (filterShape.size() == 4) {
      oc = filterShape[0];
    } else if (filterShape.size() == 5) {
      assert(convOp.group() != 1);
      // g, oc/g, ic/g, kh, kw
      oc = filterShape[0] * filterShape[1];
    } else {
      assert(0);
    }
    assert(filterSize % oc == 0);
    int64_t isz = filterSize / oc;

    // get bias tensor
    std::unique_ptr<std::vector<float> > bias = nullptr;
    std::vector<int64_t> biasShape;
    int64_t biasSize = 0;
    if (convOp.with_bias()) {
      bias = readAndDeleteWeightTensor(convOp.getOperand(2),
          weightTensorFile_);
      auto biasType = convOp.getOperand(2)->getType().cast<TensorType>();
      biasShape = std::vector<int64_t>(biasType.getShape());
      biasSize = std::accumulate(std::begin(biasShape),
          std::end(biasShape), 1, std::multiplies<>());
      assert(biasSize == oc);
      assert(biasSize == (int64_t)bias->size());
    }

    // create new tensors
    // TODO: use float to save all weights for now
    auto new_filter = std::make_unique<std::vector<float> >(filterSize);
    std::unique_ptr<std::vector<float> > new_bias = nullptr;
    if (bias) {
      new_bias = std::make_unique<std::vector<float> >(biasSize);
    }

    // create tensors for rshift and multiplier
    // TODO: use float to save all weights for now
    auto rshift_per_layer = std::make_unique<std::vector<float> >(1);
    auto rshift_per_channel = std::make_unique<std::vector<float> >(oc);
    auto multiplier_per_channel = std::make_unique<std::vector<float> >(oc);

    // quantization
    if (quant == INT8_PER_LAYER) {
      quantizeWeightInt8PerLayer(filter->data(), bias ? bias->data() : nullptr,
                                 oc, isz, threshold_y, threshold_x,
                                 new_filter->data(), bias ? new_bias->data() : nullptr,
                                 rshift_per_layer->data());

    } else if (quant == INT8_PER_CHANNEL) {
      quantizeWeightInt8PerChannel(filter->data(), bias ? bias->data() : nullptr,
                                 oc, isz, threshold_y, threshold_x,
                                 new_filter->data(), bias ? new_bias->data() : nullptr,
                                 rshift_per_channel->data());

    } else if (quant == INT8_MULTIPLER) {
      quantizeWeightInt8Multiplier(filter->data(), bias ? bias->data() : nullptr,
                                 oc, isz, threshold_y, threshold_x,
                                 new_filter->data(), bias ? new_bias->data() : nullptr,
                                 rshift_per_channel->data(),
                                 multiplier_per_channel->data());

    } else {
      assert(0);
    }

    // update op
    // TODO: use float to save all weights for now
    Type eltType = FloatType::getF32(rewriter.getContext());
    std::string storageType = "INT8";
    addWeightTensorAndUpdateWeightOp(convOp.getOperand(1), *new_filter,
        filterShape, storageType, eltType, rewriter, weightTensorFile_);
    if (bias) {
      // for per_channel quant, bias store as INT32, per layer use INT16
      storageType = (quant == INT8_PER_LAYER) ? "INT16" : "INT32";
      addWeightTensorAndUpdateWeightOp(convOp.getOperand(2), *new_bias,
          biasShape, storageType, eltType, rewriter, weightTensorFile_);
    }

    // newOperands for create a new conv Op
    std::vector<Value *> newOperands;
    newOperands.push_back(convOp.getOperand(0));
    newOperands.push_back(convOp.getOperand(1));
    if (bias) {
      newOperands.push_back(convOp.getOperand(2));
    }

    // add rshift and multiplier (if present) to weight
    if (quant == INT8_PER_LAYER) {
      auto shape = std::vector<int64_t>{1};
      std::string storageType = "NONE";
      auto new_op = addWeightTensorAndCreateWeightOp(
          convOp.name().getValue().str() + "_quant_int8_rshift",
          *rshift_per_layer, shape, storageType, eltType,
          rewriter, loc, weightTensorFile_, weightFileVar_);
      newOperands.push_back(new_op);
    } else if (quant == INT8_PER_CHANNEL) {
      auto shape = std::vector<int64_t>{oc};
      std::string storageType = "UINT32";
      auto new_op = addWeightTensorAndCreateWeightOp(
          convOp.name().getValue().str() + "_quant_int8_rshift",
          *rshift_per_channel, shape, storageType, eltType,
          rewriter, loc, weightTensorFile_, weightFileVar_);
      newOperands.push_back(new_op);
    } else if (quant == INT8_MULTIPLER) {
      auto shape = std::vector<int64_t>{oc};
      std::string storageType = "UINT32";

      auto new_op_1 = addWeightTensorAndCreateWeightOp(
          convOp.name().getValue().str() + "_quant_int8_rshift",
          *rshift_per_channel, shape, storageType, eltType,
          rewriter, loc, weightTensorFile_, weightFileVar_);
      newOperands.push_back(new_op_1);

      auto new_op_2 = addWeightTensorAndCreateWeightOp(
          convOp.name().getValue().str() + "_quant_int8_multiplier",
          *multiplier_per_channel, shape, storageType, eltType,
          rewriter, loc, weightTensorFile_, weightFileVar_);
      newOperands.push_back(new_op_2);
    } else {
      assert(0);
    }

    // if fused with eltwise, push the last operand
    if (convOp.fused_eltwise_method() != "NONE") {
      newOperands.push_back(convOp.getOperand(convOp.getNumOperands() - 1));
    }

    // set quant type
    if (quant == INT8_PER_LAYER) {
      convOp.setAttr("quant", rewriter.getStringAttr("INT8"));
    } else if (quant == INT8_PER_CHANNEL) {
      convOp.setAttr("quant", rewriter.getStringAttr("INT8_PER_CHANNEL"));
    } else if (quant == INT8_MULTIPLER) {
      convOp.setAttr("quant", rewriter.getStringAttr("INT8_MULTIPLIER"));
    }

    // replace with the new conv op
    auto origAttrs = convOp.getAttrs();
    std::vector<NamedAttribute> newAttrs(origAttrs.begin(), origAttrs.end());
    rewriter.replaceOpWithNewOp<tpu::Conv2DOp>(
        convOp, convOp.getResult()->getType(),
        ArrayRef<Value *>{newOperands}, ArrayRef<NamedAttribute>{newAttrs});

    return matchSuccess();
  }


  TensorFile *weightTensorFile_;
  Value* weightFileVar_;
};

struct TpuQuantEltwiseOpPattern : public RewritePattern {
  TpuQuantEltwiseOpPattern(MLIRContext *context, TensorFile *weightTensorFile,
                           Value *weightFileVar)
      : RewritePattern(tpu::EltwiseOp::getOperationName(), 1, context),
        weightTensorFile_(weightTensorFile), weightFileVar_(weightFileVar) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto eltOp = cast<tpu::EltwiseOp>(op);
    std::string op_name =
        eltOp.getAttrOfType<StringAttr>("name").getValue().str();
    // auto loc = op->getLoc();

    if (eltOp.quant() != "NONE") {
      LLVM_DEBUG(llvm::errs() << eltOp.name() << " quantized already\n";);
      return matchFailure();
    }

    // Support Sum first
    if (eltOp.method() != "SUM") {
      eltOp.setAttr("quant", rewriter.getStringAttr("INT8"));
      return matchSuccess();
    }

    SmallVector<Value *, 4> newOperands(op->getOperands().begin(),
                                        op->getOperands().end());
    SmallVector<NamedAttribute, 8> newAttrs(op->getAttrs().begin(),
                                            op->getAttrs().end());

    // Update quantization status
    newAttrs.push_back(
        rewriter.getNamedAttr("quant", rewriter.getStringAttr("INT8")));

    const unsigned nInputs = op->getNumOperands();

    std::vector<float> threshold_x(nInputs);
    float threshold_y;
    // determine multiplier and rshift according each threshold_x
    // scale[i] = threshold_x[i] / threshold_y
    // each scale will be implemented by hardware as
    // scale[i] = multiplier / (1 << rshift)
    // find a rshift, that put max(multiplier) into range (64, 127)
    uint32_t rshift;
    std::vector<int8_t> multiplier(nInputs);
    for (unsigned index = 0; index < nInputs; ++index) {
      // get threshold_x
      threshold_x[index] = getPreviousOpThreshold(op, index);
    }
    // get threshold_y
    threshold_y = eltOp.threshold_y().getValue().convertToFloat();
    if (eltOp.method() == "SUM") {
      // determine rshift for all inputs, and multiplier for each input
      // use max threshold_x to find rshift first
      float max_threshold_x =
          *std::max_element(std::begin(threshold_x), std::end(threshold_x));
      rshift = findRShiftAndMultiplierFromQScale(max_threshold_x / threshold_y);
      for (int index = 0; index < 2; ++index) {
        float qscale = threshold_x[index] / threshold_y;
        multiplier[index] =
            (int8_t)findMultiplierFromQScaleAndRShift(qscale, rshift);
      }

      auto multiplier_int8 = std::vector<float>(nInputs);
      for (unsigned i = 0; i < nInputs; ++i) {
        multiplier_int8[i] = (float)multiplier[i];
      }

      // Add rshift and multiplier to weight
      Type eltType = FloatType::getF32(rewriter.getContext());
      std::string storageType = "NONE";
      auto loc = op->getLoc();

      auto rshiftData = std::vector<float>(1);
      auto rshiftShape = std::vector<int64_t>{1};
      rshiftData[0] = rshift;
      auto rshiftOp = addWeightTensorAndCreateWeightOp(
          eltOp.name().getValue().str() + "_quant_int8_rshift", rshiftData,
          rshiftShape, storageType, eltType, rewriter, loc, weightTensorFile_,
          weightFileVar_);
      newOperands.push_back(rshiftOp);

      auto multiplierShape = std::vector<int64_t>{nInputs};
      auto multiplierOp = addWeightTensorAndCreateWeightOp(
          eltOp.name().getValue().str() + "_quant_int8_multiplier",
          multiplier_int8, multiplierShape, storageType, eltType, rewriter, loc,
          weightTensorFile_, weightFileVar_);
      newOperands.push_back(multiplierOp);
    }

    // Replace with new eltwise op
    rewriter.replaceOpWithNewOp<tpu::EltwiseOp>(
        eltOp, eltOp.getResult()->getType(), ArrayRef<Value *>{newOperands},
        ArrayRef<NamedAttribute>{newAttrs});

    return matchSuccess();
  }

  TensorFile *weightTensorFile_;
  Value *weightFileVar_;
};

struct TpuQuantFullyConnectedOpPattern : public RewritePattern {
  TpuQuantFullyConnectedOpPattern(MLIRContext *context,
      TensorFile *weightTensorFile, Value* weightFileVar)
      : RewritePattern("tpu.fully_connected", 1, context),
        weightTensorFile_(weightTensorFile),
        weightFileVar_(weightFileVar) {}

  PatternMatchResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    auto fcOp = cast<tpu::FullyConnectedOp>(op);
    std::string op_name = fcOp.name().getValue().str();
    auto loc = op->getLoc();

    if (fcOp.quant() != "NONE") {
      LLVM_DEBUG(llvm::errs() << fcOp.name() << " quantized already\n";);
      return matchFailure();
    }

    float threshold_x = getPreviousOpThreshold(op);
    float threshold_y = fcOp.threshold_y().getValue().convertToFloat();
    LLVM_DEBUG(llvm::errs() << " > " << op_name
                 << ", threshold_y = " << std::to_string(threshold_y)
                 << ", threshold_x = " << std::to_string(threshold_x) << "\n";);

    // find filter and bias tensor
    std::vector<std::unique_ptr<std::vector<float> > > weights(2);
    for (unsigned i = 0; i < fcOp.getNumOperands() - 1; ++i) {
      auto weight_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
          fcOp.getOperand(i + 1)->getDefiningOp());
      assert(weight_op);
      assert(weight_op.name().hasValue());
      auto tensor_name = weight_op.name().getValue();
      LLVM_DEBUG(llvm::errs() << "  weight[" << i << "] : " << tensor_name << "\n";);
      auto type = weight_op.getResult()->getType().cast<TensorType>();
      weights[i] = weightTensorFile_->readTensor<float>(tensor_name, type);
      // delete the tensor from the weight file
       weightTensorFile_->deleteTensor<float>(tensor_name);
    }
    float *filter = (float *)weights[0]->data();
    float *bias = nullptr;
    if (weights[1]) {
      bias = (float *)weights[1]->data();
    }

    // create new tensors for quantized fliter and bias
    auto filter_type = fcOp.filter()->getType().cast<TensorType>();
    std::vector<int64_t> filter_shape(filter_type.getShape());
    assert(filter_shape.size() == 2);
    int64_t filter_size = std::accumulate(std::begin(filter_shape),
        std::end(filter_shape), 1, std::multiplies<>());
    assert(filter_size == (int64_t)weights[0]->size());
    int64_t n = filter_shape[0];
    //std::vector<int8_t> new_filter(filter_size);
    //std::vector<int8_t> new_bias(oc);
    // TODO: use float for now, need to change to int8
    std::vector<float> new_filter(filter_size);
    std::vector<float> new_bias(n);

    // quantization
    // TODO: use only float in weight file for now
    std::vector<float> rshift(1);
    // find the max fabs weight value
    float max_filter_abs = fabs(filter[0]);
    for (int i = 0; i < filter_size; ++i) {
      if ( fabs(filter[i]) > max_filter_abs ) {
          max_filter_abs = fabs(filter[i]);
      }
    }
    LLVM_DEBUG(llvm::errs() << "  max filter : " << max_filter_abs << "\n";);

    // find rshift
    // Q(W) = W * (threshold_x / threshold_y) * (1 << rshift)
    // find a rshift put the Q(max_filter_abs) in range (64, 127)
    assert(threshold_x);
    rshift[0] = (float)findRShiftForFilter(max_filter_abs, threshold_y, threshold_x);
    LLVM_DEBUG(llvm::errs() << "  rshift : " << rshift[0] << "\n";);

    // quantize weight
    for (int i = 0; i < filter_size; ++i) {
      new_filter[i] = (float)quantizeFilterRShift(filter[i], threshold_y,
                                 threshold_x, (uint32_t)rshift[0]);
    }
    if (bias) {
      for (int i = 0; i < n; ++i) {
        new_bias[i] = (float)quantizeBiasRShiftI16(bias[i], threshold_y,
                                 (uint32_t)rshift[0]);
      }
    }

    // update op
    std::vector<Value *> newOperands;
    newOperands.push_back(fcOp.getOperand(0));

    // add new filter and bias weight
    std::vector<std::vector<float> *> newWeights{ &new_filter, &new_bias };
    std::vector<std::vector<int64_t> > weightShapes{ filter_shape, std::vector<int64_t>{n} };
    for (int i = 0; i < 2; ++i) {
      if (!bias && i == 1)
        continue;
      auto tensor_name = op_name + "_quant_int8_" + std::to_string(i);
      LLVM_DEBUG(llvm::errs() << "  new_weight[" << i << "] : " << tensor_name << "\n";);
      auto type = RankedTensorType::get(weightShapes[i], FloatType::getF32(rewriter.getContext()));
      weightTensorFile_->addTensor<float>(tensor_name, newWeights[i], type);
      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
      if (i == 0) {
        // filter store as INT8
        attrs.push_back(rewriter.getNamedAttr("storage", rewriter.getStringAttr("INT8")));
      } else if (i == 1) {
        // bias store as INT16
        attrs.push_back(rewriter.getNamedAttr("storage", rewriter.getStringAttr("INT16")));
      }
      auto new_weight_op = rewriter.create<tpu::LoadWeightOp>(op->getLoc(), type,
          ArrayRef<Value *>{weightFileVar_}, ArrayRef<NamedAttribute>{attrs});
      newOperands.push_back(new_weight_op);
    }

    // add rshift to weight
    auto tensor_name = op_name + "_quant_int8_rshift";
    LLVM_DEBUG(llvm::errs() << "  new_weight[rshift] : " << tensor_name << "\n";);
    // TODO: use only float in weight file for now
    //auto type = RankedTensorType::get(std::vector<int64_t>{1},
    //    IntegerType::get(32, rewriter.getContext()));
    //weightTensorFile_->addTensor<uint32_t>(tensor_name, &rshift, type);
    auto type = RankedTensorType::get(std::vector<int64_t>{1},
        FloatType::getF32(rewriter.getContext()));
    weightTensorFile_->addTensor<float>(tensor_name, &rshift, type);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
    attrs.push_back(rewriter.getNamedAttr("storage", rewriter.getStringAttr("NONE")));
    auto new_weight_op = rewriter.create<tpu::LoadWeightOp>(loc, type,
        ArrayRef<Value *>{weightFileVar_}, ArrayRef<NamedAttribute>{attrs});
    newOperands.push_back(new_weight_op);

    // replace with the new fc op
    auto origAttrs = fcOp.getAttrs();
    std::vector<NamedAttribute> newAttrs(origAttrs.begin(), origAttrs.end());
    newAttrs.push_back(rewriter.getNamedAttr("quant", rewriter.getStringAttr("INT8")));
    rewriter.replaceOpWithNewOp<tpu::FullyConnectedOp>(
        fcOp, fcOp.getResult()->getType(),
        ArrayRef<Value *>{newOperands}, ArrayRef<NamedAttribute>{newAttrs});

    return matchSuccess();
  }

  TensorFile *weightTensorFile_;
  Value* weightFileVar_;
};

template<typename TensorTyOp>
struct TpuQuantDefaultPattern : public RewritePattern {
  TpuQuantDefaultPattern(MLIRContext *context, TensorFile *weightTensorFile,
      Value* weightFileVar)
      : RewritePattern(TensorTyOp::getOperationName(), 1, context),
        weightTensorFile_(weightTensorFile),
        weightFileVar_(weightFileVar) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto castOp = cast<TensorTyOp>(op);
    if (castOp.quant() != "NONE") {
      LLVM_DEBUG(llvm::errs() << castOp.name() << " quantized already\n";);
      return matchFailure();
    }
    castOp.setAttr("quant", rewriter.getStringAttr("INT8"));

    return matchSuccess();
  }

  TensorFile *weightTensorFile_;
  Value* weightFileVar_;
};

struct TpuQuantReluOpPattern : public RewritePattern {
  TpuQuantReluOpPattern(MLIRContext *context, TensorFile *weightTensorFile,
      Value* weightFileVar)
      : RewritePattern("tpu.relu", 1, context),
        weightTensorFile_(weightTensorFile),
        weightFileVar_(weightFileVar) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto reluOp = cast<tpu::ReluOp>(op);
    if (reluOp.quant() != "NONE") {
      LLVM_DEBUG(llvm::errs() << reluOp.name() << " quantized already\n";);
      return matchFailure();
    }

    reluOp.setAttr("quant", rewriter.getStringAttr("INT8"));

    float negative_slope = reluOp.negative_slope().convertToFloat();
    LLVM_DEBUG(llvm::errs() << "  negative_slope: "
                            << std::to_string(negative_slope) << "\n";);
    if (negative_slope < 1e-7) {
      // this is relu rather than leaky relu, no quantization needed
      return matchSuccess();
    }

    // create tensors for rshift and multiplier
    // TODO: use float to save all weights for now
    auto rshift_pos = std::vector<float>(1);
    auto multiplier_pos = std::vector<float>(1);
    auto rshift_neg = std::vector<float>(1);
    auto multiplier_neg = std::vector<float>(1);

    // quantization
    float threshold_x = getPreviousOpThreshold(op);
    float threshold_y = getOpThreshold(op);
    double qscale_pos = threshold_x / threshold_y;
    if (fabs(threshold_x - threshold_y) < 1e-5 * std::min(threshold_x, threshold_y)) {
      // no positive scale
      rshift_pos[0] = 0;
      multiplier_pos[0] = 0;
      LLVM_DEBUG(llvm::errs() << "  Positive: no_scale\n";);
    } else {
      uint32_t uint_multiplier_pos;
      rshift_pos[0] = (float)findRShiftAndMultiplierFromQScale(qscale_pos,
          &uint_multiplier_pos, false);
      multiplier_pos[0] = (float)uint_multiplier_pos;
      LLVM_DEBUG(llvm::errs() << "  Positive: ";);
      LLVM_DEBUG(llvm::errs() << "  [multiplier : rshift] = ["
                              << std::to_string(multiplier_pos[0]) << " : "
                              << std::to_string(rshift_pos[0]) << "]\n";);
    }
    // negative
    float qscale_neg = fabs(qscale_pos * negative_slope);
    uint32_t uint_multiplier_neg = 0;
    rshift_neg[0] = (float)findRShiftAndMultiplierFromQScale(qscale_neg,
        &uint_multiplier_neg, false);
    multiplier_neg[0] = (float)uint_multiplier_neg;
    LLVM_DEBUG(llvm::errs() << "  Negative: ";);
    LLVM_DEBUG(llvm::errs() << "  [multiplier : rshift] = ["
                            << std::to_string(multiplier_neg[0]) << " : "
                            << std::to_string(rshift_neg[0]) << "]\n";);

    // newOperands for create a new conv Op
    auto loc = op->getLoc();
    std::vector<Value *> newOperands;
    newOperands.push_back(reluOp.getOperand(0));

    auto shape = std::vector<int64_t>{1};
    Type eltType = FloatType::getF32(rewriter.getContext());
    std::string storageType = "INT8";

    auto new_op_1 = addWeightTensorAndCreateWeightOp(
        reluOp.name().getValue().str() + "_quant_int8_rshift_pos",
        rshift_pos, shape, storageType, eltType,
        rewriter, loc, weightTensorFile_, weightFileVar_);
    newOperands.push_back(new_op_1);

    auto new_op_2 = addWeightTensorAndCreateWeightOp(
        reluOp.name().getValue().str() + "_quant_int8_multiplier_pos",
        multiplier_pos, shape, storageType, eltType,
        rewriter, loc, weightTensorFile_, weightFileVar_);
    newOperands.push_back(new_op_2);

    auto new_op_3 = addWeightTensorAndCreateWeightOp(
        reluOp.name().getValue().str() + "_quant_int8_rshift_neg",
        rshift_neg, shape, storageType, eltType,
        rewriter, loc, weightTensorFile_, weightFileVar_);
    newOperands.push_back(new_op_3);

    auto new_op_4 = addWeightTensorAndCreateWeightOp(
        reluOp.name().getValue().str() + "_quant_int8_multiplier_neg",
        multiplier_neg, shape, storageType, eltType,
        rewriter, loc, weightTensorFile_, weightFileVar_);
    newOperands.push_back(new_op_4);


    // replace with the new conv op
    auto origAttrs = reluOp.getAttrs();
    std::vector<NamedAttribute> newAttrs(origAttrs.begin(), origAttrs.end());
    rewriter.replaceOpWithNewOp<tpu::ReluOp>(
        reluOp, reluOp.getResult()->getType(),
        ArrayRef<Value *>{newOperands}, ArrayRef<NamedAttribute>{newAttrs});
    return matchSuccess();
  }

  TensorFile *weightTensorFile_;
  Value* weightFileVar_;
};

struct TpuQuantPReluOpPattern : public RewritePattern {
  TpuQuantPReluOpPattern(MLIRContext *context, TensorFile *weightTensorFile,
                           Value *weightFileVar)
      : RewritePattern("tpu.prelu", 1, context),
        weightTensorFile_(weightTensorFile), weightFileVar_(weightFileVar) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                  PatternRewriter &rewriter) const override {

    auto preluOp = cast<tpu::PReluOp>(op);
    std::string op_name =
        preluOp.getAttrOfType<StringAttr>("name").getValue().str();
    auto loc = op->getLoc();

    if (preluOp.quant() != "NONE") {
      LLVM_DEBUG(llvm::errs() << preluOp.name() << " quantized already\n";);
      return matchFailure();
    }

    // get threshold
    float threshold_x = getPreviousOpThreshold(op);
    float threshold_y = getOpThreshold(op);;

    LLVM_DEBUG(llvm::errs() << " > " << preluOp.name()
                 << ", threshold_y = "<< std::to_string(threshold_y)
                 << ", threshold_x = " << std::to_string(threshold_x) << "\n";);

    // find negative slope tensor
    auto weight_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
          preluOp.getOperand(1)->getDefiningOp());
    assert(weight_op);
    assert(weight_op.name().hasValue());
    auto tensor_name = weight_op.name().getValue();
    LLVM_DEBUG(llvm::errs() << "  weight[" << 1 << "] : " << tensor_name << "\n";);
    auto type = weight_op.getResult()->getType().cast<TensorType>();

    auto weight = weightTensorFile_->readTensor<float>(tensor_name, type);
    float *negative_slope = (float *)weight->data();

    auto slopeType = preluOp.negative_slope()->getType().cast<TensorType>();
    std::vector<int64_t> slopeShape(slopeType.getShape());

    assert(slopeShape.size() >= 2);
    int64_t slopeSize = std::accumulate(std::begin(slopeShape),
        std::end(slopeShape), 1, std::multiplies<>());
    int64_t c = slopeShape[1]; // c means the channel number
    // assert(slopeSize == (int64_t)negative_slope->size());
    assert(slopeSize == c);

    // create new tensors
    // TODO: use float to save all weights for now
    auto new_negative_slope = std::vector<float>(slopeSize);

    // create tensors for rshift and multiplier
    // TODO: use float to save all weights for now
    auto rshift_pos = std::vector<float>(1);
    auto multiplier_pos = std::vector<float>(1);
    auto rshift_neg = std::vector<float>(1);
    auto multiplier_neg = std::vector<float>(1);

    // quantization
    double qscale_pos = threshold_x / threshold_y;
    // positive
    uint32_t uint_multiplier_pos;
    LLVM_DEBUG(llvm::errs() << "  Positive: ";);
    rshift_pos[0] = (float)findRShiftAndMultiplierFromQScale(qscale_pos, &uint_multiplier_pos, false);
    multiplier_pos[0] = (float)uint_multiplier_pos;
    LLVM_DEBUG(llvm::errs() << "  [multiplier : rshift] = ["
                  << std::to_string(multiplier_pos[0]) << " : "
                  << std::to_string(rshift_pos[0]) << "]\n";);
    // negative
    LLVM_DEBUG(llvm::errs() << "  Negative:";);

    /* I8 Multiplier version */
    float max_abs_negative_qscale = fabs(qscale_pos * negative_slope[0]);
    for (int i = 1; i < c; ++i) {
      if (fabs(qscale_pos * negative_slope[i]) > max_abs_negative_qscale){
        max_abs_negative_qscale = fabs(qscale_pos * negative_slope[i]) > max_abs_negative_qscale;
      }
    }
    uint32_t uint_multiplier_neg = 0;
    float rshift_tmp = (float)findRShiftAndMultiplierFromQScale(fabs(max_abs_negative_qscale),
                                                                &uint_multiplier_neg, false);
    rshift_neg[0] = rshift_tmp;
    LLVM_DEBUG(llvm::errs() << "  [multiplier : rshift] = ["
                            << std::to_string(uint_multiplier_neg) << " : "
                            << std::to_string(rshift_neg[0]) << "]\n";);

    for (int i = 0; i < c; ++i) {
      new_negative_slope[i] = (float)quantizeFilterRShift(negative_slope[i], threshold_y,
                                                          threshold_x, rshift_neg[0]);
    }

    /// backend not support INT32 Multiplier now so use INT8 way now.
    // else if (quant == INT8_MULTIPLER) {
    //   /* I32 Multiplier version */
    //   float max_slope_abs = fabs(negative_slope[0]);
    //   for (int i = 1; i < c; ++i) {
    //     if ( fabs(negative_slope[i]) > max_slope_abs) {
    //         max_slope_abs = fabs(negative_slope[i]);
    //     }
    //   }
    //   LLVM_DEBUG(llvm::errs() << "  max slope : " << max_slope_abs << "\n";);

    //   double qscale_neg = findQScaleForFilter(max_slope_abs, threshold_y, threshold_x);
    //   uint32_t uint_multiplier_neg;
    //   rshift_neg[0] = (float)findRShiftAndMultiplierFromQScale(qscale_neg, &uint_multiplier_neg, false);
    //   multiplier_neg[0] = (float)uint_multiplier_neg;
    //   LLVM_DEBUG(llvm::errs() << "  [multiplier : rshift] = ["
    //                  << std::to_string(multiplier_neg[0]) << " : "
    //                  << std::to_string(rshift_neg[0]) << "]\n";);

    //   for (int i = 0; i < c; ++i) {
    //     new_negative_slope[i] =
    //         (float)quantizeFilterRShiftAndMultiplier(negative_slope[i],
    //             threshold_y, threshold_x, (uint32_t)rshift_neg[0], uint_multiplier_neg, false);
    //     new_negative_slope[i] *= multiplier_neg[0];
    //   }
    // }

    // update op
    // TODO: use float to save all weights for now
    Type eltType = FloatType::getF32(rewriter.getContext());
    std::string storageType = "INT8";
    addWeightTensorAndUpdateWeightOp(preluOp.getOperand(1), new_negative_slope,
        slopeShape, storageType, eltType, rewriter, weightTensorFile_);

    // newOperands for create a new conv Op
    std::vector<Value *> newOperands;
    newOperands.push_back(preluOp.getOperand(0));
    newOperands.push_back(preluOp.getOperand(1));

    auto shape = std::vector<int64_t>{1};
    std::string storageType_r = "INT8";
    std::string storageType_m = "INT8";
    /// backend not support INT32 Multiplier now so use INT8 way now.
    // storageType_m = "INT32";

    auto new_op_1 = addWeightTensorAndCreateWeightOp(
      preluOp.name().getValue().str() + "_quant_int8_rshift_pos",
      rshift_pos, shape, storageType_r, eltType,
      rewriter, loc, weightTensorFile_, weightFileVar_);
    newOperands.push_back(new_op_1);

    auto new_op_2 = addWeightTensorAndCreateWeightOp(
        preluOp.name().getValue().str() + "_quant_int8_multiplier_pos",
        multiplier_pos, shape, storageType_m, eltType,
        rewriter, loc, weightTensorFile_, weightFileVar_);
    newOperands.push_back(new_op_2);

    auto new_op_3 = addWeightTensorAndCreateWeightOp(
        preluOp.name().getValue().str() + "_quant_int8_rshift_neg",
        rshift_neg, shape, storageType_r, eltType,
        rewriter, loc, weightTensorFile_, weightFileVar_);
    newOperands.push_back(new_op_3);

#if 0
    auto new_op_4 = addWeightTensorAndCreateWeightOp(
        preluOp.name().getValue().str() + "_quant_int8_multiplier_neg",
        multiplier_neg, shape, storageType_m, eltType,
        rewriter, loc, weightTensorFile_, weightFileVar_);
    newOperands.push_back(new_op_4);
#endif

    // set quant type
    preluOp.setAttr("quant", rewriter.getStringAttr("INT8"));

    // replace with the new conv op
    auto origAttrs = preluOp.getAttrs();
    std::vector<NamedAttribute> newAttrs(origAttrs.begin(), origAttrs.end());
    rewriter.replaceOpWithNewOp<tpu::PReluOp>(
        preluOp, preluOp.getResult()->getType(),
        ArrayRef<Value *>{newOperands}, ArrayRef<NamedAttribute>{newAttrs});

    return matchSuccess();
  }

  TensorFile *weightTensorFile_;
  Value *weightFileVar_;
};

struct TpuQuantScaleOpPattern : public RewritePattern {
  TpuQuantScaleOpPattern(MLIRContext *context, TensorFile *weightTensorFile,
                           Value *weightFileVar)
      : RewritePattern("tpu.scale", 1, context),
        weightTensorFile_(weightTensorFile), weightFileVar_(weightFileVar) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto scaleOp = cast<tpu::ScaleOp>(op);
    auto loc = op->getLoc();
    std::string op_name =
        scaleOp.getAttrOfType<StringAttr>("name").getValue().str();

    if (scaleOp.quant() != "NONE") {
      LLVM_DEBUG(llvm::errs() << scaleOp.name() << " quantized already\n";);
      return matchFailure();
    }

    // check if second input is load weight op
    auto weight_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
            scaleOp.getOperand(1)->getDefiningOp());

    if (weight_op){

      // get quant type
      QUANT_INT8_TYPE_e quant;
      if (!clQuantConvPerChannel) {
        quant = INT8_PER_LAYER;
      } else {
        quant = INT8_PER_CHANNEL;
      }
      float threshold_y = scaleOp.threshold_y().getValue().convertToFloat();
      float threshold_x = getPreviousOpThreshold(op);
      // find scale and bias tensor
      std::vector<std::unique_ptr<std::vector<float>>> weights(2);
      for (unsigned i = 0; i < scaleOp.getNumOperands() - 1; ++i) {
        auto weight_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
            scaleOp.getOperand(i + 1)->getDefiningOp());
        assert(weight_op);
        assert(weight_op.name().hasValue());
        auto tensor_name = weight_op.name().getValue();
        LLVM_DEBUG(llvm::errs()
                       << "  weight[" << i << "] : " << tensor_name << "\n";);
        auto type = weight_op.getResult()->getType().cast<TensorType>();
        weights[i] = weightTensorFile_->readTensor<float>(tensor_name, type);
        // delete the tensor from the weight file
        weightTensorFile_->deleteTensor<float>(tensor_name);
      }

      // get scale tensor
      float *scale = (float *)weights[0]->data();
      float *bias = nullptr;
      if (weights[1]) {
        bias = (float *)weights[1]->data();
      }

      // create new tensors for quantized scale and bias
      auto scale_type = scaleOp.scale()->getType().cast<TensorType>();
      std::vector<int64_t> scale_shape(scale_type.getShape());
      std::vector<int64_t> bias_shape;
      int64_t scale_size =
          std::accumulate(std::begin(scale_shape), std::end(scale_shape), 1,
                          std::multiplies<>());

      assert(scale_size == (int64_t)weights[0]->size());
      int64_t oc;
      if(scale_shape.size() == 4){
        oc = scale_shape[1];
      } else if (scale_shape.size() == 1){
        oc = scale_shape[0];
      } else if (scale_shape.size() == 2){
        oc = scale_shape[1];
      }

      // TODO: use float for now, need to change to int8
      auto new_scale = std::make_unique<std::vector<float>>(scale_size);
      std::unique_ptr<std::vector<float>> new_bias = nullptr;
      if (bias) {
        new_bias = std::make_unique<std::vector<float>>(oc);
        auto biasType = scaleOp.getOperand(2)->getType().cast<TensorType>();
        bias_shape = biasType.getShape();
      }
      assert(scale_size % oc == 0);
      int64_t isz = scale_size / oc;

      // create tensors for rshift and multiplier
      // TODO: use float to save all weights for now
      auto rshift_per_layer = std::make_unique<std::vector<float>>(1);
      auto rshift_per_channel = std::make_unique<std::vector<float>>(oc);
      auto multiplier_per_layer = std::make_unique<std::vector<float>>(1);
      auto multiplier_per_channel = std::make_unique<std::vector<float>>(oc);

      // quantization
      if (quant == INT8_PER_LAYER) {
        quantizeWeightInt8PerLayerMultiplier(
            scale, bias ? bias : nullptr, oc, isz, threshold_y, threshold_x,
            new_scale->data(), bias ? new_bias->data() : nullptr,
            rshift_per_layer->data(), multiplier_per_layer->data());

      } else if (quant == INT8_PER_CHANNEL) {
        quantizeWeightInt8Multiplier(scale, bias ? bias : nullptr,
                                   oc, isz, threshold_y, threshold_x,
                                   new_scale->data(), bias ? new_bias->data() : nullptr,
                                   rshift_per_channel->data(),
                                   multiplier_per_channel->data());
      } else {
        assert(0);
      }

      // update op
      // TODO: use float to save all weights for now
      Type eltType = FloatType::getF32(rewriter.getContext());
      std::string storageType = "INT8";
      addWeightTensorAndUpdateWeightOp(scaleOp.getOperand(1), *new_scale,
                                       scale_shape, storageType, eltType,
                                       rewriter, weightTensorFile_);
      if (bias) {
        // for per_channel quant, bias store as INT32, per layer use INT16
        storageType = (quant == INT8_PER_LAYER) ? "INT16" : "INT32";
        addWeightTensorAndUpdateWeightOp(scaleOp.getOperand(2), *new_bias,
                                         bias_shape, storageType, eltType,
                                         rewriter, weightTensorFile_);
      }

      // newOperands for create a new conv Op
      std::vector<Value *> newOperands;
      newOperands.push_back(scaleOp.getOperand(0));
      newOperands.push_back(scaleOp.getOperand(1));
      if (bias) {
        newOperands.push_back(scaleOp.getOperand(2));
      }
      // add rshift and multiplier (if present) to weight
      if (quant == INT8_PER_LAYER) {
        auto shape = std::vector<int64_t>{1};
        std::string storageType = "UINT32";
        auto new_op_1 = addWeightTensorAndCreateWeightOp(
            scaleOp.name().getValue().str() + "_quant_int8_rshift",
            *rshift_per_layer, shape, storageType, eltType, rewriter, loc,
            weightTensorFile_, weightFileVar_);
        newOperands.push_back(new_op_1);
        auto new_op_2 = addWeightTensorAndCreateWeightOp(
            scaleOp.name().getValue().str() + "_quant_int8_multiplier",
            *multiplier_per_layer, shape, storageType, eltType, rewriter, loc,
            weightTensorFile_, weightFileVar_);
        newOperands.push_back(new_op_2);

      } else if (quant == INT8_PER_CHANNEL) {
        auto shape = std::vector<int64_t>{oc};
        std::string storageType = "UINT32";

        auto new_op_1 = addWeightTensorAndCreateWeightOp(
            scaleOp.name().getValue().str() + "_quant_int8_rshift",
            *rshift_per_channel, shape, storageType, eltType, rewriter, loc,
            weightTensorFile_, weightFileVar_);
        newOperands.push_back(new_op_1);

        auto new_op_2 = addWeightTensorAndCreateWeightOp(
            scaleOp.name().getValue().str() + "_quant_int8_multiplier",
            *multiplier_per_channel, shape, storageType, eltType, rewriter, loc,
            weightTensorFile_, weightFileVar_);
        newOperands.push_back(new_op_2);
      } else {
        assert(0);
      }

      // set quant type
      if (quant == INT8_PER_LAYER) {
        scaleOp.setAttr("quant", rewriter.getStringAttr("INT8"));
      } else if (quant == INT8_PER_CHANNEL) {
        scaleOp.setAttr("quant", rewriter.getStringAttr("INT8_PER_CHANNEL"));
      }
      // replace with the new scale op
      auto origAttrs = scaleOp.getAttrs();
      std::vector<NamedAttribute> newAttrs(origAttrs.begin(), origAttrs.end());

      rewriter.replaceOpWithNewOp<tpu::ScaleOp>(
          scaleOp, scaleOp.getResult()->getType(),
          ArrayRef<Value *>{newOperands}, ArrayRef<NamedAttribute>{newAttrs});
    } else {
      // 1880v2 calculate scale i32 need to pack rshift and multiplier
      // create weight for backend to get rshift and multiplier address

      std::vector<float> threshold_x(2);
      float threshold_y;
      threshold_y = scaleOp.threshold_y().getValue().convertToFloat();
      for (int index = 0; index < 2; ++index) {
        threshold_x[index] = getPreviousOpThreshold(op, index);
      }
      float threshold_prod = std::accumulate(
          threshold_x.begin(), threshold_x.end(), 1.0, std::multiplies<>());
      uint32_t multiplier_prod = 0;
      float qscale = threshold_prod / threshold_y / 127.0;
      float rshift = (float)findRShiftAndMultiplierFromQScale(
          qscale, &multiplier_prod, true, 255);

      auto scale_type = scaleOp.scale()->getType().cast<TensorType>();
      std::vector<int64_t> scale_shape(scale_type.getShape());
      int64_t oc;
      if (scale_shape.size() == 2) {
        oc = scale_shape[1];
      } else {
        //TODO: maybe has other shape size
        assert(0 && "scale from input shape should be 2");
      }
      auto rshift_perchannel = std::make_unique<std::vector<float>>(oc, rshift);
      auto multiplier_perchannel =
          std::make_unique<std::vector<float>>(oc, multiplier_prod);

      Type eltType = FloatType::getF32(rewriter.getContext());
      std::string storageType = "UINT32";
      auto shape = std::vector<int64_t>{oc};

      // newOperands for create a new conv Op
      std::vector<Value *> newOperands;
      newOperands.push_back(scaleOp.getOperand(0));
      newOperands.push_back(scaleOp.getOperand(1));
      auto new_op_1 = addWeightTensorAndCreateWeightOp(
          scaleOp.name().getValue().str() + "_quant_int8_rshift",
          *rshift_perchannel, shape, storageType, eltType, rewriter, loc,
          weightTensorFile_, weightFileVar_);
      newOperands.push_back(new_op_1);

      auto new_op_2 = addWeightTensorAndCreateWeightOp(
          scaleOp.name().getValue().str() + "_quant_int8_multiplier",
          *multiplier_perchannel, shape, storageType, eltType, rewriter, loc,
          weightTensorFile_, weightFileVar_);
      newOperands.push_back(new_op_2);

      scaleOp.setAttr("quant", rewriter.getStringAttr("INT8"));

      // replace with the new scale op
      auto origAttrs = scaleOp.getAttrs();
      std::vector<NamedAttribute> newAttrs(origAttrs.begin(), origAttrs.end());

      rewriter.replaceOpWithNewOp<tpu::ScaleOp>(
          scaleOp, scaleOp.getResult()->getType(),
          ArrayRef<Value *>{newOperands}, ArrayRef<NamedAttribute>{newAttrs});
    }
    return matchSuccess();
  }

  TensorFile *weightTensorFile_;
  Value *weightFileVar_;
};

struct TpuQuantPowerOpPattern : public RewritePattern {
    TpuQuantPowerOpPattern(MLIRContext *context, TensorFile *weightTensorFile,
      Value* weightFileVar)
      :RewritePattern("tpu.power", 1, context),
        weightTensorFile_(weightTensorFile),
        weightFileVar_(weightFileVar) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const {

    auto powerOp = cast<tpu::PowerOp>(op);

    if(powerOp.has_table() == true){
      LLVM_DEBUG(llvm::errs() << powerOp.name() << " gen already\n";);
      return matchFailure();
    }

    std::string op_name = powerOp.getAttrOfType<StringAttr>("name").getValue().str();
    auto result_var = powerOp.getResult();

    llvm::ArrayRef<int64_t> input_shape = result_var->getType().dyn_cast<mlir::TensorType>().getShape();
    assert(input_shape.size() == 4);
    auto size = input_shape[1];// get channel number

    // get quant type
    QUANT_INT8_TYPE_e quant;
    if (!clQuantConvPerChannel) {
      assert(!clQuantConvMultiplier
             && "enable per channel before enable multiplier");
      quant = INT8_PER_LAYER;
    } else if (!clQuantConvMultiplier) {
      quant = INT8_PER_CHANNEL;
    } else {
      quant = INT8_MULTIPLER;
    }

    //assign scale and shift tensor
    std::vector<float> scale_weight(size);
    std::vector<float> shift_weight(size);

    float threshold_y,threshold_x,qscale,rshift;
    uint32_t multiplier;

    threshold_y = powerOp.threshold_y().getValue().convertToFloat();
    threshold_x = getPreviousOpThreshold(powerOp);

    qscale = (threshold_x*threshold_x) /(127.0*threshold_y);

    float scale = powerOp.scale().convertToFloat();
    float shift = powerOp.shift().convertToFloat();

    if (quant == INT8_PER_LAYER|| quant == INT8_PER_CHANNEL) {
      rshift = findRShiftAndMultiplierFromQScale(qscale);
      multiplier = findMultiplierFromQScaleAndRShift(qscale, rshift);
    }else if(quant == INT8_MULTIPLER){
      rshift = (float)findRShiftAndMultiplierFromQScale(qscale, &multiplier, true,255);
    }

    if (quant == INT8_PER_LAYER|| quant == INT8_PER_CHANNEL) {
      scale = scale*(threshold_y/threshold_x)*multiplier;
      shift = shift*(threshold_y/127.0)*multiplier;
      scale = (float)applyRShiftAndSaturateInt8(scale, (uint32_t)rshift);
      shift = (float)applyRShiftAndSaturateInt8(shift, (uint32_t)rshift);
    }else if(quant == INT8_MULTIPLER){
      scale = scale*(threshold_y/threshold_x);
      shift = shift*(threshold_y/127.0);
      scale = (float)applyMultiplierAndRShiftAndSaturateInt8(scale,rshift,  multiplier);
      shift = (float)applyMultiplierAndRShiftAndSaturateInt8(shift,rshift,  multiplier);
    }

    for (uint32_t i = 0; i < scale_weight.size(); i++) {
      scale_weight[i] = scale;
    }

    for (uint32_t i = 0; i < shift_weight.size(); i++) {
      shift_weight[i] = shift;
    }

    // update op
    std::vector<Value *> newOperands;
    newOperands.push_back(powerOp.getOperand(0));
    auto type = RankedTensorType::get(input_shape,FloatType::getF32(rewriter.getContext()));

    //add scale operand
    auto tensor_name = op_name + "_gen_scale";
    weightTensorFile_->addTensor<float>(tensor_name, scale_weight.data(), type);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
    attrs.push_back(rewriter.getNamedAttr("storage", rewriter.getStringAttr("INT8")));
    auto new_scale_op = rewriter.create<tpu::LoadWeightOp>(op->getLoc(), type,
        ArrayRef<Value *>{weightFileVar_}, ArrayRef<NamedAttribute>{attrs});
    newOperands.push_back(new_scale_op);


    //add scale operand
    tensor_name = op_name + "_gen_shift";
    weightTensorFile_->addTensor<float>(tensor_name, shift_weight.data(), type);
    std::vector<NamedAttribute> attrs_shift;
    attrs_shift.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
    attrs_shift.push_back(rewriter.getNamedAttr("storage", rewriter.getStringAttr("INT8")));
    auto new_shift_op = rewriter.create<tpu::LoadWeightOp>(op->getLoc(), type,
        ArrayRef<Value *>{weightFileVar_}, ArrayRef<NamedAttribute>{attrs_shift});
    newOperands.push_back(new_shift_op);

    powerOp.setAttr("has_table", rewriter.getBoolAttr("true"));

          // set quant type
      if (quant == INT8_PER_LAYER) {
        powerOp.setAttr("quant", rewriter.getStringAttr("INT8"));
      } else if (quant == INT8_PER_CHANNEL) {
        powerOp.setAttr("quant", rewriter.getStringAttr("INT8_PER_CHANNEL"));
      } else if (quant == INT8_MULTIPLER) {
        powerOp.setAttr("quant", rewriter.getStringAttr("INT8_MULTIPLIER"));
      }

    rewriter.replaceOpWithNewOp<tpu::PowerOp>(
        powerOp, powerOp.getResult()->getType(),
        ArrayRef<Value *>{newOperands}, ArrayRef<NamedAttribute>{powerOp.getAttrs()});

    return matchSuccess();
}
  TensorFile *weightTensorFile_;
  Value* weightFileVar_;
};

template<typename T>
static void addQuantOpAfterOp(PatternRewriter &rewriter, T &op) {
  auto loc = op.getLoc();
  float threshold_y = op.threshold_y().getValue().convertToFloat();
  std::string op_name = op.template getAttrOfType<StringAttr>("name").getValue().str();

  auto *inst = op.getOperation();
  OpBuilder builder(inst);
  auto clonedOp = cast<T>(builder.clone(*inst));

  auto type = op.getResult()->getType();
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(op_name + "_quant")));
  attrs.push_back(rewriter.getNamedAttr("threshold", rewriter.getF32FloatAttr(threshold_y)));
  attrs.push_back(rewriter.getNamedAttr("quant", rewriter.getStringAttr("INT8")));

  auto quantOp = rewriter.create<tpu::QuantizationOp>(loc, type,
      ArrayRef<Value *>{clonedOp.getResult()}, ArrayRef<NamedAttribute>{attrs});
  rewriter.replaceOp(op, {quantOp});
}

template<typename T>
static void addDequantOpBeforeOp(PatternRewriter &rewriter, T &op) {
  auto loc = op.getLoc();

  for (size_t i = 0; i < op.getOperation()->getNumOperands(); ++i) {
      float threshold_x = getPreviousOpThreshold(op, i);
      std::string op_name = getPreviousOpName(op, i).str();
      auto type = op.getOperation()->getOperand(i)->getType();
      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(op_name + "_dequant")));
      attrs.push_back(rewriter.getNamedAttr("threshold", rewriter.getF32FloatAttr(threshold_x)));
      attrs.push_back(rewriter.getNamedAttr("quant", rewriter.getStringAttr("INT8")));
      auto dequantOp = rewriter.create<tpu::DequantizationOp>(loc, type,
        ArrayRef<Value *>{op.getOperation()->getOperand(i)}, ArrayRef<NamedAttribute>{attrs});
      op.getOperation()->setOperand(i, dequantOp);
    }
  }

// insert Quant Op after input Op
struct TpuAddQuantAfterInputOpPattern : public OpRewritePattern<tpu::InputOp> {
  using OpRewritePattern<tpu::InputOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(tpu::InputOp op,
                                     PatternRewriter &rewriter) const {
    for (auto &use : op.getResult()->getUses()) {
      Operation *operandOp = use.getOwner();
      if (auto cast_op = llvm::dyn_cast_or_null<tpu::QuantizationOp>(operandOp)) {
        LLVM_DEBUG(llvm::errs() << op.name() << " quantized already\n";);
        return matchFailure();
      }
    }

    LLVM_DEBUG(llvm::errs() << op.name() << " add quantization op after Input\n";);
    addQuantOpAfterOp<tpu::InputOp>(rewriter, op);

    return matchSuccess();
  }
};

// insert Dequant Op before return Op
struct TpuAddDeQuantBeforeReturnOpPattern : public OpRewritePattern<ReturnOp> {
  using OpRewritePattern<ReturnOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(ReturnOp op,
                                     PatternRewriter &rewriter) const {
    auto formerOp = op.getOperand(0)->getDefiningOp();
    if (matchPattern(formerOp, m_Op<tpu::DequantizationOp>())) {
      LLVM_DEBUG(llvm::errs() << "return dequantized already\n";);
      return matchFailure();
    }
    if (matchPattern(formerOp, m_Op<tpu::DetectionOutputOp>())) {
      LLVM_DEBUG(llvm::errs() << "DetectionOutputOp is cpu output layer,no need dequant\n";);
      return matchFailure();
    }

    LLVM_DEBUG(llvm::errs() << " add dequantization op defore Return\n";);
    addDequantOpBeforeOp<ReturnOp>(rewriter, op);

    return matchSuccess();
  }
};

// insert Dequant Op before DetectionOuput Op
struct TpuAddDequantBeforeDetectionOutputOpPattern : public OpRewritePattern<tpu::DetectionOutputOp> {
  using OpRewritePattern<tpu::DetectionOutputOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(tpu::DetectionOutputOp op,
                                     PatternRewriter &rewriter) const {
    auto formerOp = op.getOperand(0)->getDefiningOp();
    if (matchPattern(formerOp, m_Op<tpu::DequantizationOp>())) {
      LLVM_DEBUG(llvm::errs() << "return dequantized already\n";);
      return matchFailure();
    }

  auto loc = op.getLoc();

  for (size_t i = 0; i < op.getOperation()->getNumOperands(); ++i) {

    formerOp = op.getOperand(i)->getDefiningOp();
    if (!matchPattern(formerOp, m_Op<tpu::LoadWeightOp>())&&!matchPattern(formerOp, m_Op<tpu::ReshapeOp>())) {
        float threshold_x = getPreviousOpThreshold(op, i);
        std::string op_name = getPreviousOpName(op, i).str();
        auto type = op.getOperation()->getOperand(i)->getType();
        std::vector<NamedAttribute> attrs;
        attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(op_name + "_dequant")));
        attrs.push_back(rewriter.getNamedAttr("threshold", rewriter.getF32FloatAttr(threshold_x)));
        attrs.push_back(rewriter.getNamedAttr("quant", rewriter.getStringAttr("INT8")));
        auto dequantOp = rewriter.create<tpu::DequantizationOp>(loc, type,
          ArrayRef<Value *>{op.getOperation()->getOperand(i)}, ArrayRef<NamedAttribute>{attrs});
        op.getOperation()->setOperand(i, dequantOp);
      }
    }

    return matchSuccess();
  }
};

struct TpuAddQuantAndDequantForSoftmaxOpPattern : public OpRewritePattern<tpu::SoftmaxOp> {
  using OpRewritePattern<tpu::SoftmaxOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(tpu::SoftmaxOp op,
                                     PatternRewriter &rewriter) const {
    auto formerOp = op.getOperand()->getDefiningOp();
    if (matchPattern(formerOp, m_Op<tpu::DequantizationOp>())) {
      LLVM_DEBUG(llvm::errs() << op.name() << " insert quant and dequant already\n";);
      return matchFailure();
    }

    LLVM_DEBUG(llvm::errs() << op.name() << " insert quant and dequant\n";);
    addDequantOpBeforeOp<tpu::SoftmaxOp>(rewriter, op);
    addQuantOpAfterOp<tpu::SoftmaxOp>(rewriter, op);

    return matchSuccess();
  }
};


struct TpuSimplifyQuantDequantPattern : public OpRewritePattern<tpu::DequantizationOp> {
  using OpRewritePattern<tpu::DequantizationOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(tpu::DequantizationOp op,
                                     PatternRewriter &rewriter) const {
    auto formerOp = op.getOperand()->getDefiningOp();
    if (!matchPattern(formerOp, m_Op<tpu::QuantizationOp>())) {
      LLVM_DEBUG(llvm::errs() << op.name() << " simplified quant and dequant already\n";);
      return matchFailure();
    }

    LLVM_DEBUG(llvm::errs() << " simplify quant and dequant\n";);
    rewriter.replaceOp(op, formerOp->getOperand(0));

    return matchSuccess();
  }
};

struct TpuRemoveQuantBeforeReshapOpPattern : public OpRewritePattern<tpu::ReshapeOp> {
  using OpRewritePattern<tpu::ReshapeOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(tpu::ReshapeOp op,
                                     PatternRewriter &rewriter) const {
    auto formerOp = op.getOperand()->getDefiningOp();
    if (!matchPattern(formerOp, m_Op<tpu::QuantizationOp>())) {
      LLVM_DEBUG(llvm::errs() << op.name() << "reshape op is not after softmax op keep use int8\n";);
      return matchFailure();
    }

    //remove quant op to use float32 output of softmax
    rewriter.replaceOp(formerOp, formerOp->getOperand(0));

    llvm::errs() << "Use this reshape op as cpu layer\n";
    //use reshape as cpu layer
    op.setAttr("quant", rewriter.getStringAttr("NONE"));

    return matchSuccess();
  }
};


class QuantizeInt8Pass : public FunctionPass<QuantizeInt8Pass> {
public:
  explicit QuantizeInt8Pass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    // find tensorFile and Value
    llvm::StringRef filename;
    Value* weightFileVar;
    fn.walk([&](tpu::LoadFileOp op) {
      filename = op.filename();
      LLVM_DEBUG(llvm::errs() << "LoadFileOp filename " << filename << "\n";);
      weightFileVar = op.getResult();
    });
    auto weightTensorFile = openTensorFile(filename);

    auto *context = &getContext();

    OwningRewritePatternList patterns_w;
    //concat cpu layer test
    patterns_w
        .insert<TpuQuantDefaultPattern<tpu::ConcatOp>,
                TpuQuantConv2DOpPattern,
                TpuQuantDefaultPattern<tpu::CropOp>,
                TpuQuantDefaultPattern<tpu::DivOp>,
                TpuQuantEltwiseOpPattern,
                TpuQuantFullyConnectedOpPattern,
                TpuQuantDefaultPattern<tpu::PermuteOp>,
                TpuQuantDefaultPattern<tpu::Pool2DOp>,
                TpuQuantPowerOpPattern,
                TpuQuantPReluOpPattern,
                TpuQuantReluOpPattern,
                TpuQuantDefaultPattern<tpu::ReshapeOp> ,
                TpuQuantScaleOpPattern,
                TpuQuantDefaultPattern<tpu::SigmoidOp>,
                TpuQuantDefaultPattern<tpu::SliceOp>,
                TpuQuantDefaultPattern<tpu::SqrtOp>
               >(
            context, weightTensorFile.get(), weightFileVar);

    applyPatternsGreedily(fn, patterns_w);

    OwningRewritePatternList patterns_q;

    // add Quant after Input
    patterns_q.insert<TpuAddQuantAfterInputOpPattern>(context);
    // add Dequant before Result
    patterns_q.insert<TpuAddDeQuantBeforeReturnOpPattern>(context);

    // add Quant and Dequant before and after any cpu layer
    patterns_q.insert<TpuAddQuantAndDequantForSoftmaxOpPattern>(context);

    // remove Quant op before reshape (this is for ssd softmax + flatten case)
    patterns_q.insert<TpuRemoveQuantBeforeReshapOpPattern>(context);
    // add Dequant before DetectionOuputOp which is CPU layer but also output layer
    patterns_q.insert<TpuAddDequantBeforeDetectionOutputOpPattern>(context);

    applyPatternsGreedily(fn, patterns_q);

    OwningRewritePatternList patterns_s;
    // Fold and remove consecutive Dequant and Quant
    patterns_s.insert<TpuSimplifyQuantDequantPattern>(context);
    applyPatternsGreedily(fn, patterns_s);

    std::string newName;
    weightTensorFile->keep(true, &newName);
    fn.walk([&](tpu::LoadFileOp op) {
      OpBuilder opBuilder(context);
      op.setAttr("filename", opBuilder.getStringAttr(newName));
      llvm::errs() << "LoadFileOp filename updated to " << newName << "\n";
    });
  }

private:

  llvm::raw_ostream &os;
};

} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createQuantizeInt8Pass() {
  return std::make_unique<QuantizeInt8Pass>();
}

static PassRegistration<QuantizeInt8Pass>
    pass("quant-int8",
         "Quantization to int8");
