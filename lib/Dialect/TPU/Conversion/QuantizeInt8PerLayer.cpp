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

#include <sstream>
#include <fstream>
using namespace mlir;

static llvm::cl::OptionCategory clOptionsCategory("quantization options");

static llvm::cl::opt<bool> clQuantConvPerChannel(
    "enable-conv-per-channel",
    llvm::cl::desc("Enable per channel quantization for convolution weight"),
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

static inline int8_t quantizeFilter(float w, float threshold_y,
    float threshold_x, uint32_t rshift) {
  float factor = (threshold_x / threshold_y) * (1 << rshift);
  int q = w * factor;
  assert( (q <= 127) && (q >= -128) );
  return (int8_t)q;
}

static inline int16_t quantizeBiasI16(float w, float threshold_y,
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

static inline int32_t quantizeBiasI32(float w, float threshold_y,
    uint32_t rshift) {
  float factor = (128.0f / threshold_y) * (1 << rshift);
  int32_t q = (int32_t)(w * factor);
  return (int32_t)q;
}

static LogicalResult getPreviousOpThreshold(Operation *op, float *threshold) {
  if (op->getNumOperands() == 0) {
    return failure();
  }
  auto formerOp = op->getOperand(0)->getDefiningOp();
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::InputOp>(formerOp)) {
    *threshold = cast_op.threshold_y().getValue().convertToFloat();
    return success();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::Conv2DOp>(formerOp)) {
    *threshold = cast_op.threshold_y().getValue().convertToFloat();
    return success();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::FullyConnectedOp>(formerOp)) {
    *threshold = cast_op.threshold_y().getValue().convertToFloat();
    return success();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::Pool2DOp>(formerOp)) {
    *threshold = cast_op.threshold_y().getValue().convertToFloat();
    return success();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::BatchNormOp>(formerOp)) {
    *threshold = cast_op.threshold_y().getValue().convertToFloat();
    return success();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::ScaleOp>(formerOp)) {
    *threshold = cast_op.threshold_y().getValue().convertToFloat();
    return success();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::ReluOp>(formerOp)) {
    *threshold = cast_op.threshold_y().getValue().convertToFloat();
    return success();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::EltwiseOp>(formerOp)) {
    *threshold = cast_op.threshold_y().getValue().convertToFloat();
    return success();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::SoftmaxOp>(formerOp)) {
    *threshold = cast_op.threshold_y().getValue().convertToFloat();
    return success();
  }
  return failure();
}

template<typename T>
static LogicalResult getOpThreshold(T &op, float *threshold_y, float *threshold_x) {
  assert(op.name().hasValue());
  std::string op_name = op.name().getValue().str();
  *threshold_y = op.threshold_y().getValue().convertToFloat();
  auto status = getPreviousOpThreshold(op, threshold_x);
  llvm::errs() << " > " << op_name << ", threshold_y = " << std::to_string(*threshold_y)
      << ", threshold_x = "
      << (succeeded(status) ? std::to_string(*threshold_x) : std::string("not present"))
      << "\n";
  return status;
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
    auto loc = op->getLoc();

    if (convOp.quant() != "NONE") {
      llvm::errs() << convOp.name() << " quantized already\n";
      return matchFailure();
    }

    float threshold_y, threshold_x;
    assert(succeeded(getOpThreshold<tpu::Conv2DOp>(convOp,
                     &threshold_y, &threshold_x)));

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
    if (!clQuantConvPerChannel) {
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
        new_filter[i] = (float)quantizeFilter(filter[i], threshold_y, threshold_x, (uint32_t)rshift[0]);
      }
      if (bias) {
        for (int i = 0; i < oc; ++i) {
          new_bias[i] = (float)quantizeBiasI16(bias[i], threshold_y, (uint32_t)rshift[0]);
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

      // find rshift
      // Q(W) = W * (threshold_x / threshold_y) * (1 << rshift)
      // find a rshift put the Q(max_filter_abs) in range (64, 127)
      assert(threshold_x);
      for (int i = 0; i < oc; ++i) {
        rshift_per_channel[i] = (float)findRShift(max_filter_abs[i], threshold_y, threshold_x);
        llvm::errs() << "  rshift_per_channel[" << i << "] : "
            << (uint32_t)rshift_per_channel[i] << "\n";
      }

      // quantize weight
      for (int i = 0; i < oc; ++i) {
        for (int j = 0; j < inner_size; ++j) {
          new_filter[inner_size * i + j] =
              (float)quantizeFilter(filter[inner_size * i + j], threshold_y, threshold_x,
                                    (uint32_t)rshift_per_channel[i]);
        }
      }
      if (bias) {
        for (int i = 0; i < oc; ++i) {
          new_bias[i] = (float)quantizeBiasI32(bias[i], threshold_y, (uint32_t)rshift_per_channel[i]);
        }
      }
    }

    // update op
    std::vector<Value *> newOperands;
    newOperands.push_back(convOp.getOperand(0));

    // add new filter and bias weight
    std::vector<std::vector<float> *> newWeights{ &new_filter, &new_bias };
    std::vector<std::vector<int64_t> > weightShapes{ filter_shape, std::vector<int64_t>{oc} };
    std::string op_name = convOp.getAttrOfType<StringAttr>("name").getValue().str();
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
    if (!clQuantConvPerChannel) {
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
      // TODO: use only float in weight file for now
      //weightTensorFile_->addTensor<uint32_t>(tensor_name, &rshift, type);
      weightTensorFile_->addTensor<float>(tensor_name, &rshift_per_channel, type);
      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
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
      newAttrs.push_back(rewriter.getNamedAttr("quant", rewriter.getStringAttr("INT8_PER_CHANNEL")));
    }
    rewriter.replaceOpWithNewOp<tpu::Conv2DOp>(
        convOp, convOp.getResult()->getType(),
        ArrayRef<Value *>{newOperands}, ArrayRef<NamedAttribute>{newAttrs});

    return matchSuccess();
  }

  TensorFile *weightTensorFile_;
  Value* weightFileVar_;
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

    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<TpuQuantConv2DOpPattern>(context, weightTensorFile.get(), weightFileVar);
    applyPatternsGreedily(fn, patterns);
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
