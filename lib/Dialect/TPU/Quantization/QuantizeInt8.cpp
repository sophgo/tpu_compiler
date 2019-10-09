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
#include "mlir/Dialect/TPU/QuantizationUtils.h"
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

static inline uint32_t findRShift(float max_weight, float threshold_y,
    float threshold_x) {
  // Q(W) = W * (threshold_x / threshold_y) * (1 << rshift)
  // find a rshift put the Q(max_filter_abs) in range (64, 127)
  assert(threshold_y > 0);
  float a = max_weight * threshold_x / threshold_y;
  assert(a < 128);
  for (uint32_t rshift = 0; rshift < 32; ++rshift) {
    if ( (a * (1 << rshift)) >= 64 )
      return rshift;
  }
  assert(false);
  return 31;
}

static inline int8_t quantizeFilterRShift(float w, float threshold_y,
    float threshold_x, uint32_t rshift) {
  float factor = (threshold_x / threshold_y) * (1 << rshift);
  float q_f = w * factor;
  #if 0
  // away_from_zero
  int q_i = (q_f >= 0) ? (int)ceil(q_f) : (int)floor(q_f);
  #else
  int q_i = (int)roundf(q_f);
  #endif
  if ( (q_i > 127) || (q_i < -128) ) {
    llvm::errs() << "  element exceeds limits [-128, 127] : "
                 << w << " -> " << q_i << "\n";
  }
  //assert( (q_i <= 127) && (q_i >= -128) );
  if ( q_i > 127 )
    q_i = 127;
  if ( q_i < -128 )
    q_i = -128;

  return (int8_t)q_i;
}

static inline int16_t quantizeBiasRShiftI16(float w, float threshold_y,
    uint32_t rshift) {
  float factor = (128.0f / threshold_y) * (1 << rshift);
  int q = w * factor;
  if ( (q > 65535) || (q < -65536) ) {
    llvm::errs() << "  element exceeds limits [-65536, 65535] : "
                 << std::to_string(w) << " -> " << std::to_string(q)
                 << ", rshift = " << rshift << "\n";
  }
  assert( (q <= 65535) && (q >= -65536) );
  if ( q > 65535 )
    q = 65535;
  if ( q < -65536 )
    q = -65536;
  return (int16_t)q;
}

static inline int32_t quantizeBiasRShiftI32(float w, float threshold_y,
    uint32_t rshift) {
  float factor = (128.0f / threshold_y) * (1 << rshift);
  int32_t q = (int32_t)(w * factor);
  return (int32_t)q;
}

static inline float findMultiplier(float max_weight, float threshold_y,
    float threshold_x) {
  // Q(W) = W * (threshold_x / threshold_y) * M
  // find a M put the Q(max_filter_abs) = 127
  assert(threshold_y > 0);
  float m = 127.0f * threshold_y / (max_weight * threshold_x);
  return m;
}

static inline int8_t quantizeFilterMultiplier(float w, float threshold_y,
    float threshold_x, float multiplier) {
  float factor = (threshold_x / threshold_y) * multiplier;
  float q_f = w * factor;
  #if 0
  // away_from_zero
  int q_i = (q_f >= 0) ? (int)ceil(q_f) : (int)floor(q_f);
  #else
  int q_i = (int)roundf(q_f);
  #endif
  if ( (q_i > 127) || (q_i < -128) ) {
    llvm::errs() << "  element exceeds limits [-128, 127] : "
                 << w << " -> " << q_i << "\n";
  }
  //assert( (q_i <= 127) && (q_i >= -128) );
  if ( q_i > 127 )
    q_i = 127;
  if ( q_i < -128 )
    q_i = -128;

  return (int8_t)q_i;
}

static inline int32_t quantizeBiasMultiplier(float w, float threshold_y,
    float multiplier) {
  float factor = (128.0f / threshold_y) * multiplier;
  int32_t q = (int32_t)(w * factor);
  return (int32_t)q;
}

namespace {

struct TpuQuantConv2DOpPattern : public RewritePattern {
  TpuQuantConv2DOpPattern(MLIRContext *context, TensorFile *weightTensorFile,
      Value* weightFileVar)
      : RewritePattern("tpu.conv_2d", 1, context),
        weightTensorFile_(weightTensorFile),
        weightFileVar_(weightFileVar) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto convOp = cast<tpu::Conv2DOp>(op);
    std::string op_name = convOp.getAttrOfType<StringAttr>("name").getValue().str();
    auto loc = op->getLoc();

    if (convOp.quant() != "NONE") {
      llvm::errs() << convOp.name() << " quantized already\n";
      return matchFailure();
    }

    float threshold_y, threshold_x;
    auto status = getPreviousOpThreshold(op, &threshold_x);
    assert(succeeded(status));
    threshold_y = convOp.threshold_y().getValue().convertToFloat();
    llvm::errs() << " > " << op_name << ", threshold_y = " << std::to_string(threshold_y)
        << ", threshold_x = "
        << (succeeded(status) ? std::to_string(threshold_x) : std::string("not present"))
        << "\n";

    // find filter and bias tensor
    std::vector<std::unique_ptr<std::vector<float> > > weights(2);
    for (unsigned i = 0; i < convOp.getNumOperands() - 1; ++i) {
      auto weight_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
          convOp.getOperand(i + 1)->getDefiningOp());
      assert(weight_op);
      assert(weight_op.name().hasValue());
      auto tensor_name = weight_op.name().getValue();
      llvm::errs() << "  weight[" << i << "] : " << tensor_name << "\n";
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
    auto filter_type = convOp.filter()->getType().cast<TensorType>();
    std::vector<int64_t> filter_shape(filter_type.getShape());
    assert(filter_shape.size() == 4);
    int64_t filter_size = std::accumulate(std::begin(filter_shape),
        std::end(filter_shape), 1, std::multiplies<>());
    assert(filter_size == (int64_t)weights[0]->size());
    int64_t oc = filter_shape[0];
    int64_t inner_size = filter_shape[1] * filter_shape[2] * filter_shape[3];
    //std::vector<int8_t> new_filter(filter_size);
    //std::vector<int8_t> new_bias(oc);
    // TODO: use float for now, need to change to int8
    std::vector<float> new_filter(filter_size);
    std::vector<float> new_bias(oc);

    // quantization
    // TODO: use only float in weight file for now
    std::vector<float> rshift(1);
    std::vector<float> rshift_per_channel(oc);
    std::vector<float> multiplier(oc);
    if (!clQuantConvPerChannel) {
      assert(!clQuantConvMultiplier && "enable per channel before enable multiplier");
      // find the max fabs weight value
      float max_filter_abs = fabs(filter[0]);
      for (int i = 0; i < filter_size; ++i) {
        if ( fabs(filter[i]) > max_filter_abs ) {
          max_filter_abs = fabs(filter[i]);
        }
      }
      llvm::errs() << "  max filter : " << max_filter_abs << "\n";

      // find rshift
      // Q(W) = W * (threshold_x / threshold_y) * (1 << rshift)
      // find a rshift put the Q(max_filter_abs) in range (64, 127)
      assert(threshold_x);
      rshift[0] = (float)findRShift(max_filter_abs, threshold_y, threshold_x);
      llvm::errs() << "  rshift : " << rshift[0] << "\n";

      // quantize weight
      for (int i = 0; i < filter_size; ++i) {
        new_filter[i] = (float)quantizeFilterRShift(filter[i], threshold_y,
                                   threshold_x, (uint32_t)rshift[0]);
      }
      if (bias) {
        for (int i = 0; i < oc; ++i) {
          new_bias[i] = (float)quantizeBiasRShiftI16(bias[i], threshold_y,
                                   (uint32_t)rshift[0]);
        }
      }
    } else {
      // find the max fabs weight value for each channel
      std::vector<float> max_filter_abs(oc);
      for (int i = 0; i < oc; ++i) {
        max_filter_abs[i] = fabs(filter[inner_size * i]);
        for (int j = 0; j < inner_size; ++j) {
          if ( fabs(filter[inner_size * i + j]) > max_filter_abs[i] ) {
            max_filter_abs[i] = fabs(filter[inner_size * i + j]);
          }
        }
      }
      for (int i = 0; i < oc; ++i) {
        llvm::errs() << "  max filter[" << i << "] : " << max_filter_abs[i] << "\n";
      }

      if (!clQuantConvMultiplier) {
        // find rshift
        // Q(W) = W * (threshold_x / threshold_y) * (1 << rshift)
        // find a rshift put the Q(max_filter_abs) in range (64, 127)
        assert(threshold_x);
        for (int i = 0; i < oc; ++i) {
          rshift_per_channel[i] = (float)findRShift(max_filter_abs[i],
                                                    threshold_y, threshold_x);
          llvm::errs() << "  rshift_per_channel[" << i << "] : "
              << (uint32_t)rshift_per_channel[i] << "\n";
        }

        // quantize weight
        for (int i = 0; i < oc; ++i) {
          for (int j = 0; j < inner_size; ++j) {
            new_filter[inner_size * i + j] =
                (float)quantizeFilterRShift(filter[inner_size * i + j], threshold_y,
                                    threshold_x, (uint32_t)rshift_per_channel[i]);
          }
        }
        if (bias) {
          for (int i = 0; i < oc; ++i) {
            new_bias[i] = (float)quantizeBiasRShiftI32(bias[i], threshold_y,
                                    (uint32_t)rshift_per_channel[i]);
          }
        }
      } else {
        // find muliplier
        assert(threshold_x);
        for (int i = 0; i < oc; ++i) {
          multiplier[i] = findMultiplier(max_filter_abs[i], threshold_y, threshold_x);
          llvm::errs() << "  multiplier[" << i << "] : "
                       << std::to_string(multiplier[i]) << "\n";
        }

        // quantize weight
        for (int i = 0; i < oc; ++i) {
          for (int j = 0; j < inner_size; ++j) {
            new_filter[inner_size * i + j] =
                (float)quantizeFilterMultiplier(filter[inner_size * i + j],
                                                threshold_y, threshold_x,
                                                multiplier[i]);
          }
        }
        if (bias) {
          for (int i = 0; i < oc; ++i) {
            new_bias[i] = (float)quantizeBiasMultiplier(bias[i], threshold_y,
                                                        multiplier[i]);
          }
        }
      }
    }

    // update op
    std::vector<Value *> newOperands;
    newOperands.push_back(convOp.getOperand(0));

    // add new filter and bias weight
    std::vector<std::vector<float> *> newWeights{ &new_filter, &new_bias };
    std::vector<std::vector<int64_t> > weightShapes{ filter_shape, std::vector<int64_t>{oc} };
    for (int i = 0; i < 2; ++i) {
      if (!bias && i == 1)
        continue;
      auto tensor_name = op_name + "_quant_int8_" + std::to_string(i);
      llvm::errs() << "  new_weight[" << i << "] : " << tensor_name << "\n";
      auto type = rewriter.getTensorType(weightShapes[i], FloatType::getF32(rewriter.getContext()));
      weightTensorFile_->addTensor<float>(tensor_name, newWeights[i], type);
      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
      auto new_weight_op = rewriter.create<tpu::LoadWeightOp>(op->getLoc(), type,
          ArrayRef<Value *>{weightFileVar_}, ArrayRef<NamedAttribute>{attrs});
      newOperands.push_back(new_weight_op);
    }

    // add rshift to weight
    if (!clQuantConvPerChannel) {
      auto tensor_name = op_name + "_quant_int8_rshift";
      llvm::errs() << "  new_weight[rshift] : " << tensor_name << "\n";
      auto type = rewriter.getTensorType(std::vector<int64_t>{1},
          IntegerType::get(32, rewriter.getContext()));
      // TODO: use only float in weight file for now
      //weightTensorFile_->addTensor<uint32_t>(tensor_name, &rshift, type);
      weightTensorFile_->addTensor<float>(tensor_name, &rshift, type);
      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
      auto new_weight_op = rewriter.create<tpu::LoadWeightOp>(loc, type,
          ArrayRef<Value *>{weightFileVar_}, ArrayRef<NamedAttribute>{attrs});
      newOperands.push_back(new_weight_op);
    } else {
      auto type = rewriter.getTensorType(std::vector<int64_t>{oc},
          IntegerType::get(32, rewriter.getContext()));
      std::vector<NamedAttribute> attrs;
      if (!clQuantConvMultiplier) {
        auto tensor_name = op_name + "_quant_int8_rshift";
        llvm::errs() << "  new_weight[rshift] : " << tensor_name << "\n";
        // TODO: use only float in weight file for now
        //weightTensorFile_->addTensor<uint32_t>(tensor_name, &rshift, type);
        weightTensorFile_->addTensor<float>(tensor_name, &rshift_per_channel, type);
        attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
      } else {
        auto tensor_name = op_name + "_quant_int8_multiplier";
        llvm::errs() << "  new_weight[multiplier] : " << tensor_name << "\n";
        weightTensorFile_->addTensor<float>(tensor_name, &multiplier, type);
        attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
      }
      auto new_weight_op = rewriter.create<tpu::LoadWeightOp>(loc, type,
          ArrayRef<Value *>{weightFileVar_}, ArrayRef<NamedAttribute>{attrs});
      newOperands.push_back(new_weight_op);
    }

    // replace with the new conv op
    auto origAttrs = convOp.getAttrs();
    std::vector<NamedAttribute> newAttrs(origAttrs.begin(), origAttrs.end());
    if (!clQuantConvPerChannel) {
      newAttrs.push_back(rewriter.getNamedAttr("quant", rewriter.getStringAttr("INT8")));
    } else {
      if (!clQuantConvMultiplier) {
        newAttrs.push_back(rewriter.getNamedAttr("quant", rewriter.getStringAttr("INT8_PER_CHANNEL")));
      } else {
        newAttrs.push_back(rewriter.getNamedAttr("quant", rewriter.getStringAttr("INT8_MULTIPLIER")));
      }
    }
    rewriter.replaceOpWithNewOp<tpu::Conv2DOp>(
        convOp, convOp.getResult()->getType(),
        ArrayRef<Value *>{newOperands}, ArrayRef<NamedAttribute>{newAttrs});

    return matchSuccess();
  }

  TensorFile *weightTensorFile_;
  Value* weightFileVar_;
};

struct TpuQuantFullyConnectedOpPattern : public RewritePattern {
  TpuQuantFullyConnectedOpPattern(MLIRContext *context, TensorFile *weightTensorFile,
      Value* weightFileVar)
      : RewritePattern("tpu.fully_connected", 1, context),
        weightTensorFile_(weightTensorFile),
        weightFileVar_(weightFileVar) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto fcOp = cast<tpu::FullyConnectedOp>(op);
    std::string op_name = fcOp.getAttrOfType<StringAttr>("name").getValue().str();
    auto loc = op->getLoc();

    if (fcOp.quant() != "NONE") {
      llvm::errs() << fcOp.name() << " quantized already\n";
      return matchFailure();
    }

    float threshold_y, threshold_x;
    auto status = getPreviousOpThreshold(op, &threshold_x);
    assert(succeeded(status));
    threshold_y = fcOp.threshold_y().getValue().convertToFloat();
    llvm::errs() << " > " << op_name << ", threshold_y = " << std::to_string(threshold_y)
        << ", threshold_x = "
        << (succeeded(status) ? std::to_string(threshold_x) : std::string("not present"))
        << "\n";

    // find filter and bias tensor
    std::vector<std::unique_ptr<std::vector<float> > > weights(2);
    for (unsigned i = 0; i < fcOp.getNumOperands() - 1; ++i) {
      auto weight_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
          fcOp.getOperand(i + 1)->getDefiningOp());
      assert(weight_op);
      assert(weight_op.name().hasValue());
      auto tensor_name = weight_op.name().getValue();
      llvm::errs() << "  weight[" << i << "] : " << tensor_name << "\n";
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
    llvm::errs() << "  max filter : " << max_filter_abs << "\n";

    // find rshift
    // Q(W) = W * (threshold_x / threshold_y) * (1 << rshift)
    // find a rshift put the Q(max_filter_abs) in range (64, 127)
    assert(threshold_x);
    rshift[0] = (float)findRShift(max_filter_abs, threshold_y, threshold_x);
    llvm::errs() << "  rshift : " << rshift[0] << "\n";

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
      llvm::errs() << "  new_weight[" << i << "] : " << tensor_name << "\n";
      auto type = rewriter.getTensorType(weightShapes[i], FloatType::getF32(rewriter.getContext()));
      weightTensorFile_->addTensor<float>(tensor_name, newWeights[i], type);
      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
      auto new_weight_op = rewriter.create<tpu::LoadWeightOp>(op->getLoc(), type,
          ArrayRef<Value *>{weightFileVar_}, ArrayRef<NamedAttribute>{attrs});
      newOperands.push_back(new_weight_op);
    }

    // add rshift to weight
    auto tensor_name = op_name + "_quant_int8_rshift";
    llvm::errs() << "  new_weight[rshift] : " << tensor_name << "\n";
    auto type = rewriter.getTensorType(std::vector<int64_t>{1},
        IntegerType::get(32, rewriter.getContext()));
    // TODO: use only float in weight file for now
    //weightTensorFile_->addTensor<uint32_t>(tensor_name, &rshift, type);
    weightTensorFile_->addTensor<float>(tensor_name, &rshift, type);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
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
      llvm::errs() << poolOp.name() << " quantized already\n";
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
      llvm::errs() << eltOp.name() << " quantized already\n";
      return matchFailure();
    }

    // replace with the new op
    //auto origAttrs = eltOp.getAttrs();
    //std::vector<NamedAttribute> newAttrs(origAttrs.begin(), origAttrs.end());
    //newAttrs.push_back(rewriter.getNamedAttr("quant", rewriter.getStringAttr("INT8")));
    //rewriter.replaceOpWithNewOp<tpu::FullyConnectedOp>(
    //    fcOp, fcOp.getResult()->getType(),
    //    ArrayRef<Value *>{newOperands}, ArrayRef<NamedAttribute>{newAttrs});

    eltOp.setAttr("quant", rewriter.getStringAttr("INT8"));

    return matchSuccess();
  }

  TensorFile *weightTensorFile_;
  Value* weightFileVar_;
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
        llvm::errs() << op.name() << " quantized already\n";
        return matchFailure();
      }
    }

    llvm::errs() << op.name() << " add quantization op after Input\n";
    float threshold_y = op.threshold_y().getValue().convertToFloat();
    std::string op_name = op.getAttrOfType<StringAttr>("name").getValue().str();
    addQuantOpAfterOp<tpu::InputOp>(rewriter, op, threshold_y, op_name);

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
      llvm::errs() << "return dequantized already\n";
      return matchFailure();
    }

    llvm::errs() << " add dequantization op defore Return\n";
    float threshold_x;
    auto status = getPreviousOpThreshold(op, &threshold_x);
    assert(succeeded(status));
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
      llvm::errs() << op.name() << " insert quant and dequant already\n";
      return matchFailure();
    }

    llvm::errs() << op.name() << " insert quant and dequant\n";
    std::string op_name = op.getAttrOfType<StringAttr>("name").getValue().str();
    float threshold_x;
    auto status = getPreviousOpThreshold(op, &threshold_x);
    assert(succeeded(status));
    addDequantOpBeforeOp<tpu::SoftmaxOp>(rewriter, op, threshold_x, op_name);

    float threshold_y = op.threshold_y().getValue().convertToFloat();
    addQuantOpAfterOp<tpu::SoftmaxOp>(rewriter, op, threshold_y, op_name);

    return matchSuccess();
  }
};

struct TpuSimplifyQuantDequantPattern : public OpRewritePattern<tpu::DequantizationOp> {
  using OpRewritePattern<tpu::DequantizationOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(tpu::DequantizationOp op,
                                     PatternRewriter &rewriter) const {
    auto formerOp = op.getOperand()->getDefiningOp();
    if (!matchPattern(formerOp, m_Op<tpu::QuantizationOp>())) {
      llvm::errs() << op.name() << " simplified quant and dequant already\n";
      return matchFailure();
    }

    llvm::errs() << " simplify quant and dequant\n";
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
    fn.walk<tpu::LoadFileOp>([&](tpu::LoadFileOp op) {
      filename = op.getAttrOfType<StringAttr>("filename").getValue();
      llvm::errs() << "LoadFileOp filename " << filename << "\n";
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

    weightTensorFile->keep();
  }

private:

  llvm::raw_ostream &os;
};

} // namespace

std::unique_ptr<FunctionPassBase> mlir::createQuantizeInt8Pass() {
  return std::make_unique<QuantizeInt8Pass>();
}

static PassRegistration<QuantizeInt8Pass>
    pass("quant-int8",
         "Quantization to int8");
