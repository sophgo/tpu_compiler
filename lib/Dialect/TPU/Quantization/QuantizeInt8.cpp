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
#include <math.h>

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
    auto multiplier_per_channel = std::make_unique<std::vector<float> >(oc);;

    // quantization
    if (quant == INT8_PER_LAYER) {
      doWeightQuantizeInt8PerLayer(filter->data(), bias?bias->data():nullptr,
          new_filter->data(), bias?new_bias->data():nullptr,
          rshift_per_layer->data(), threshold_y, threshold_x, isz, oc);

    } else if (quant == INT8_PER_CHANNEL) {
      doWeightQuantizeInt8PerChannel(filter->data(), bias?bias->data():nullptr,
          new_filter->data(), bias?new_bias->data():nullptr,
          rshift_per_channel->data(), threshold_y, threshold_x, isz, oc);

    } else if (quant == INT8_MULTIPLER) {
      doWeightQuantizeInt8Multiplier(filter->data(), bias?bias->data():nullptr,
          new_filter->data(), bias?new_bias->data():nullptr,
          rshift_per_channel->data(), multiplier_per_channel->data(),
          threshold_y, threshold_x, isz, oc);
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

  void doWeightQuantizeInt8PerLayer(
      float *filter, float *bias, float *new_filter, float *new_bias,
      float *rshift_per_layer, float threshold_y, float threshold_x,
      int isz, int oc) const {
    int filter_size = isz * oc;
    // find the max fabs weight value
    float max_filter_abs = fabs(filter[0]);
    for (int i = 0; i < filter_size; ++i) {
      if ( fabs(filter[i]) > max_filter_abs ) {
        max_filter_abs = fabs(filter[i]);
      }
    }
    LLVM_DEBUG(llvm::errs() << "  max_filter : " << std::to_string(max_filter_abs) << "\n";);
    // find rshift
    rshift_per_layer[0] = (float)findRShift(max_filter_abs,
        threshold_y, threshold_x);
    LLVM_DEBUG(llvm::errs() << "  rshift : " << rshift_per_layer[0] << "\n";);
    // quantize weight
    for (int i = 0; i < filter_size; ++i) {
      new_filter[i] = (float)quantizeFilterRShift(filter[i],
          threshold_y, threshold_x, rshift_per_layer[0]);
    }
    if (bias) {
      for (int i = 0; i < oc; ++i) {
        new_bias[i] = (float)quantizeBiasRShiftI16(bias[i],
            threshold_y, rshift_per_layer[0]);
      }
    }
  }

  void doWeightQuantizeInt8PerChannel(
      float *filter, float *bias, float *new_filter, float *new_bias,
      float *rshift_per_channel, float threshold_y, float threshold_x,
      int isz, int oc) const {
    // find the max fabs weight value for each channel
    auto max_filter_abs = std::vector<float>(oc);
    for (int i = 0; i < oc; ++i) {
      max_filter_abs[i] = fabs(filter[isz * i]);
      for (int j = 0; j < isz; ++j) {
        if ( fabs(filter[isz * i + j]) > max_filter_abs[i] ) {
          max_filter_abs[i] = fabs(filter[isz * i + j]);
        }
      }
      LLVM_DEBUG(llvm::errs() << "  max_filter[" << i << "] : "
                   << std::to_string(max_filter_abs[i]) << "\n";);
    }

    // find rshift
    for (int i = 0; i < oc; ++i) {
      rshift_per_channel[i] = (float)findRShift(max_filter_abs[i],
          threshold_y, threshold_x);
      LLVM_DEBUG(llvm::errs() << "  rshift_per_channel[" << i << "] : "
                   << rshift_per_channel[i] << "\n";);
    }
    // quantize weight
    for (int i = 0; i < oc; ++i) {
      for (int j = 0; j < isz; ++j) {
        new_filter[isz * i + j] = (float)quantizeFilterRShift(
            filter[isz * i + j], threshold_y, threshold_x,
            rshift_per_channel[i]);
      }
    }
    if (bias) {
      for (int i = 0; i < oc; ++i) {
        new_bias[i] = (float)quantizeBiasRShiftI32(bias[i],
            threshold_y, rshift_per_channel[i]);
      }
    }
  }

  void doWeightQuantizeInt8Multiplier(
      float *filter, float *bias, float *new_filter, float *new_bias,
      float *rshift_per_channel, float *multiplier_per_channel,
      float threshold_y, float threshold_x, int isz, int oc) const {
    // find the max fabs weight value for each channel
    auto max_filter_abs = std::vector<float>(oc);
    for (int i = 0; i < oc; ++i) {
      max_filter_abs[i] = fabs(filter[isz * i]);
      for (int j = 0; j < isz; ++j) {
        if ( fabs(filter[isz * i + j]) > max_filter_abs[i] ) {
          max_filter_abs[i] = fabs(filter[isz * i + j]);
        }
      }
      LLVM_DEBUG(llvm::errs() << "  max_filter[" << i << "] : "
                   << std::to_string(max_filter_abs[i]) << "\n";);
    }

    // find qscale
    for (int i = 0; i < oc; ++i) {
      float qscale = findQScale(max_filter_abs[i], threshold_y, threshold_x);
      uint32_t multiplier;
      rshift_per_channel[i] = (float)findRShiftAndMultiplierFromQScale(
              qscale, &multiplier, true);
      multiplier_per_channel[i] = (float)multiplier;
      LLVM_DEBUG(llvm::errs() << "  [multiplier : rshift][" << i << "] = ["
                   << std::to_string(multiplier_per_channel[i]) << " : "
                   << std::to_string(rshift_per_channel[i]) << "]\n";);
    }
    // quantize weight
    for (int i = 0; i < oc; ++i) {
      for (int j = 0; j < isz; ++j) {
        new_filter[isz * i + j] = (float)quantizeFilterRShiftAndMultiplier(
            filter[isz * i + j], threshold_y, threshold_x,
            rshift_per_channel[i], multiplier_per_channel[i], true);
      }
    }
    if (bias) {
      for (int i = 0; i < oc; ++i) {
        new_bias[i] = (float)quantizeBiasRShiftAndMultiplier(bias[i],
            threshold_y, rshift_per_channel[i], multiplier_per_channel[i],
            true);
      }
    }
  }

  TensorFile *weightTensorFile_;
  Value* weightFileVar_;
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
    rshift[0] = (float)findRShift(max_filter_abs, threshold_y, threshold_x);
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

struct TpuQuantPool2DOpPattern : public RewritePattern {
  TpuQuantPool2DOpPattern(MLIRContext *context, TensorFile *weightTensorFile,
      Value* weightFileVar)
      : RewritePattern("tpu.pool_2d", 1, context),
        weightTensorFile_(weightTensorFile),
        weightFileVar_(weightFileVar) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto poolOp = cast<tpu::Pool2DOp>(op);
    std::string op_name = poolOp.getAttrOfType<StringAttr>("name").getValue().str();
    //auto loc = op->getLoc();

    if (poolOp.quant() != "NONE") {
      LLVM_DEBUG(llvm::errs() << poolOp.name() << " quantized already\n";);
      return matchFailure();
    }
    poolOp.setAttr("quant", rewriter.getStringAttr("INT8"));

    return matchSuccess();
  }

  TensorFile *weightTensorFile_;
  Value* weightFileVar_;
};

struct TpuQuantEltwiseOpPattern : public RewritePattern {
  TpuQuantEltwiseOpPattern(MLIRContext *context, TensorFile *weightTensorFile,
      Value* weightFileVar)
      : RewritePattern("tpu.eltwise", 1, context),
        weightTensorFile_(weightTensorFile),
        weightFileVar_(weightFileVar) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto eltOp = cast<tpu::EltwiseOp>(op);
    std::string op_name = eltOp.getAttrOfType<StringAttr>("name").getValue().str();
    //auto loc = op->getLoc();

    if (eltOp.quant() != "NONE") {
      LLVM_DEBUG(llvm::errs() << eltOp.name() << " quantized already\n";);
      return matchFailure();
    }
    eltOp.setAttr("quant", rewriter.getStringAttr("INT8"));

    return matchSuccess();
  }

  TensorFile *weightTensorFile_;
  Value* weightFileVar_;
};

struct TpuQuantSigmoidOpPattern : public RewritePattern {
  TpuQuantSigmoidOpPattern(MLIRContext *context, TensorFile *weightTensorFile,
                           Value *weightFileVar)
      : RewritePattern("tpu.sigmoid", 1, context),
        weightTensorFile_(weightTensorFile), weightFileVar_(weightFileVar) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto sigOp = cast<tpu::SigmoidOp>(op);
    std::string op_name =
        sigOp.getAttrOfType<StringAttr>("name").getValue().str();
    // auto loc = op->getLoc();

    if (sigOp.quant() != "NONE") {
      LLVM_DEBUG(llvm::errs() << sigOp.name() << " quantized already\n";);
      return matchFailure();
    }
    sigOp.setAttr("quant", rewriter.getStringAttr("INT8"));

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
    std::string op_name =
        scaleOp.getAttrOfType<StringAttr>("name").getValue().str();

    if (scaleOp.quant() != "NONE") {
      LLVM_DEBUG(llvm::errs() << scaleOp.name() << " quantized already\n";);
      return matchFailure();
    }
    
    // check if scale second is load weight op
    auto weight_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
            scaleOp.getOperand(1)->getDefiningOp());
    if (weight_op){
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
      float *scale = (float *)weights[0]->data();
      float *bias = nullptr;
      if (weights[1]) {
        bias = (float *)weights[1]->data();
      }
     
      // create new tensors for quantized scale and bias
      auto scale_type = scaleOp.scale()->getType().cast<TensorType>();
      std::vector<int64_t> scale_shape(scale_type.getShape());

      int64_t scale_size =
          std::accumulate(std::begin(scale_shape), std::end(scale_shape), 1,
                          std::multiplies<>());
      assert(scale_size == (int64_t)weights[0]->size());
      int64_t n = scale_shape[0];
      // TODO: use float for now, need to change to int8
      std::vector<float> new_scale(scale_size);
      std::vector<float> new_bias(n);
      std::cout << "new_scale size: " << scale_size << std::endl;
      std::cout << "bias size: " << n << std::endl;

      // TODO: use only float in weight file for now
      std::vector<float> rshift(1);
      // find the max fabs weight value
      float max_scale_abs = fabs(scale[0]);
      for (int i = 0; i < scale_size; ++i) {
        if (fabs(scale[i]) > max_scale_abs) {
          max_scale_abs = fabs(scale[i]);
        }
      }
      LLVM_DEBUG(llvm::errs() << "  max scale : " << max_scale_abs << "\n";);

      // find rshift
      // Q(W) = W * (threshold_x / threshold_y) * (1 << rshift)
      // find a rshift put the Q(max_scale_abs) in range (64, 127)
      assert(threshold_x);
      rshift[0] = (float)findRShift(max_scale_abs, threshold_y, threshold_x);
      LLVM_DEBUG(llvm::errs() << "  rshift : " << rshift[0] << "\n";);

      // quantize weight
      for (int i = 0; i < scale_size; ++i) {
        new_scale[i] = (float)quantizeFilterRShift(
            scale[i], threshold_y, threshold_x, (uint32_t)rshift[0]);
      }
      if (bias) {
        for (int i = 0; i < n; ++i) {
          new_bias[i] = (float)quantizeBiasRShiftI16(bias[i], threshold_y,
                                                     (uint32_t)rshift[0]);
        }
      }
      // update op
      std::vector<Value *> newOperands;
      newOperands.push_back(scaleOp.getOperand(0));

      // add new scale and bias weight
      std::vector<std::vector<float> *> newWeights{&new_scale, &new_bias};
      std::vector<std::vector<int64_t>> weightShapes{scale_shape,
                                                     std::vector<int64_t>{n}};
      for (int i = 0; i < 2; ++i) {
        if (!bias && i == 1)
          continue;
        auto tensor_name = op_name + "_quant_int8_" + std::to_string(i);
        LLVM_DEBUG(llvm::errs() << "  new_weight[" << i << "] : " << tensor_name
                                << "\n";);
        auto type = RankedTensorType::get(
            weightShapes[i], FloatType::getF32(rewriter.getContext()));
        weightTensorFile_->addTensor<float>(tensor_name, newWeights[i], type);
        std::vector<NamedAttribute> attrs;
        attrs.push_back(
            rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
        if (i == 0) {
          // scale store as INT8
          attrs.push_back(
              rewriter.getNamedAttr("storage", rewriter.getStringAttr("INT8")));
        } else if (i == 1) {
          // bias store as INT16
          attrs.push_back(rewriter.getNamedAttr(
              "storage", rewriter.getStringAttr("INT16")));
        }
        auto new_weight_op = rewriter.create<tpu::LoadWeightOp>(
            op->getLoc(), type, ArrayRef<Value *>{weightFileVar_},
            ArrayRef<NamedAttribute>{attrs});
        newOperands.push_back(new_weight_op);
      }
      // replace with the new scale op
      auto origAttrs = scaleOp.getAttrs();
      std::vector<NamedAttribute> newAttrs(origAttrs.begin(), origAttrs.end());
      newAttrs.push_back(
          rewriter.getNamedAttr("quant", rewriter.getStringAttr("INT8")));
      rewriter.replaceOpWithNewOp<tpu::ScaleOp>(
          scaleOp, scaleOp.getResult()->getType(), ArrayRef<Value *>{newOperands},
          ArrayRef<NamedAttribute>{newAttrs});
    }else{
      scaleOp.setAttr("quant", rewriter.getStringAttr("INT8"));
    }
    return matchSuccess();
  }

  TensorFile *weightTensorFile_;
  Value *weightFileVar_;
};

template<typename T>
static void addQuantOpAfterOp(PatternRewriter &rewriter,
    T &op, float threshold, std::string op_name) {
  auto loc = op.getLoc();

  auto *inst = op.getOperation();
  OpBuilder builder(inst);
  auto clonedOp = cast<T>(builder.clone(*inst));

  auto type = op.getResult()->getType();
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(op_name)));
  attrs.push_back(rewriter.getNamedAttr("threshold", rewriter.getF32FloatAttr(threshold)));
  attrs.push_back(rewriter.getNamedAttr("quant", rewriter.getStringAttr("INT8")));

  auto quantOp = rewriter.create<tpu::QuantizationOp>(loc, type,
      ArrayRef<Value *>{clonedOp.getResult()}, ArrayRef<NamedAttribute>{attrs});
  rewriter.replaceOp(op, {quantOp});
}

template<typename T>
static void addDequantOpBeforeOp(PatternRewriter &rewriter,
    T &op, float threshold, std::string op_name) {
  auto loc = op.getLoc();

  auto type = op.getOperation()->getOperand(0)->getType();
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(op_name)));
  attrs.push_back(rewriter.getNamedAttr("threshold", rewriter.getF32FloatAttr(threshold)));
  attrs.push_back(rewriter.getNamedAttr("quant", rewriter.getStringAttr("INT8")));
  auto dequantOp = rewriter.create<tpu::DequantizationOp>(loc, type,
      ArrayRef<Value *>{op.getOperation()->getOperand(0)}, ArrayRef<NamedAttribute>{attrs});
  op.getOperation()->setOperand(0, dequantOp);
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
    float threshold_y = op.threshold_y().getValue().convertToFloat();
    std::string op_name = op.getAttrOfType<StringAttr>("name").getValue().str();
    addQuantOpAfterOp<tpu::InputOp>(rewriter, op, threshold_y, op_name + "_quant");

    return matchSuccess();
  }
};

// insert Dequant Op before return Op
struct TpuAddQuantBeforeReturnOpPattern : public OpRewritePattern<ReturnOp> {
  using OpRewritePattern<ReturnOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(ReturnOp op,
                                     PatternRewriter &rewriter) const {
    auto formerOp = op.getOperand(0)->getDefiningOp();
    if (matchPattern(formerOp, m_Op<tpu::DequantizationOp>())) {
      LLVM_DEBUG(llvm::errs() << "return dequantized already\n";);
      return matchFailure();
    }

    LLVM_DEBUG(llvm::errs() << " add dequantization op defore Return\n";);
    float threshold_x = getPreviousOpThreshold(op);
    addDequantOpBeforeOp<ReturnOp>(rewriter, op, threshold_x, "return");

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
    std::string op_name = op.getAttrOfType<StringAttr>("name").getValue().str();

    float threshold_x = getPreviousOpThreshold(op);
    addDequantOpBeforeOp<tpu::SoftmaxOp>(rewriter, op, threshold_x, op_name + "_quant");

    float threshold_y = op.threshold_y().getValue().convertToFloat();
    addQuantOpAfterOp<tpu::SoftmaxOp>(rewriter, op, threshold_y, op_name + "_dequant");

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
    patterns_w.insert<TpuQuantConv2DOpPattern>(context,
        weightTensorFile.get(), weightFileVar);
    patterns_w.insert<TpuQuantFullyConnectedOpPattern>(context,
        weightTensorFile.get(), weightFileVar);
    patterns_w.insert<TpuQuantPool2DOpPattern>(context,
        weightTensorFile.get(), weightFileVar);
    patterns_w.insert<TpuQuantEltwiseOpPattern>(context,
        weightTensorFile.get(), weightFileVar);
    patterns_w.insert<TpuQuantSigmoidOpPattern>(context, 
        weightTensorFile.get(), weightFileVar);
    patterns_w.insert<TpuQuantScaleOpPattern>(context, 
        weightTensorFile.get(), weightFileVar);
    applyPatternsGreedily(fn, patterns_w);

    OwningRewritePatternList patterns_q;
    // add Quant after Input
    patterns_q.insert<TpuAddQuantAfterInputOpPattern>(context);
    // add Dequant before Result
    patterns_q.insert<TpuAddQuantBeforeReturnOpPattern>(context);
    // add Quant and Dequant before and after any cpu layer
    patterns_q.insert<TpuAddQuantAndDequantForSoftmaxOpPattern>(context);
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
