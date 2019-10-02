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

namespace {

class QuantizeInt8PerLayerPass : public FunctionPass<QuantizeInt8PerLayerPass> {
public:
  explicit QuantizeInt8PerLayerPass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    auto *context = &getContext();
    Builder builder(context);
    // assign the threshold_y to each op
    fn.walk([&](Operation *op) {
      os << op->getName() << "\n";

      QuantizeLayer<tpu::InputOp>(builder, op);
      QuantizeLayer<tpu::Conv2DOp>(builder, op);
      QuantizeLayer<tpu::FullyConnectedOp>(builder, op);
      QuantizeLayer<tpu::AveragePool2DOp>(builder, op);
      QuantizeLayer<tpu::MaxPool2DOp>(builder, op);
      QuantizeLayer<tpu::BatchNormOp>(builder, op);
      QuantizeLayer<tpu::ScaleOp>(builder, op);
      QuantizeLayer<tpu::ReluOp>(builder, op);
      QuantizeLayer<tpu::EltwiseOp>(builder, op);
      QuantizeLayer<tpu::SoftmaxOp>(builder, op);
    });
  }

private:
  template<typename T>
  void QuantizeLayer(Builder &builder, Operation *op) {
    auto cast_op = llvm::dyn_cast_or_null<T>(op);
    if (cast_op) {
      assert(cast_op.name().hasValue());
      std::string op_name = cast_op.name().getValue().str();
      float threshold = cast_op.threshold_y().getValue().convertToFloat();
      os << " > " << op_name << ", " << threshold << "\n";
    }
  }

  llvm::raw_ostream &os;
};

} // namespace


static llvm::cl::OptionCategory clOptionsCategory("quantization options");

static llvm::cl::opt<bool> clQuantConvPerChannel(
    "enable-conv-per-channel",
    llvm::cl::desc("Enable per channel quantization for convolution weight"),
    llvm::cl::init(false),
    llvm::cl::cat(clOptionsCategory));

std::unique_ptr<FunctionPassBase> mlir::createQuantizeInt8PerLayerPass() {
  return std::make_unique<QuantizeInt8PerLayerPass>();
}

static PassRegistration<QuantizeInt8PerLayerPass>
    pass("quant-int8",
         "Quantization to int8");
