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
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

namespace {
template<typename T>
struct TpuQuantizationOpPattern : public RewritePattern {
  TpuQuantizationOpPattern(MLIRContext *context, StringRef opName,
      uint64_t *pos, llvm::raw_ostream &map_os, size_t alignment)
      : RewritePattern(opName, 1, context),
        pos_(pos),
        map_os_(map_os),
        alignment_(alignment) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto castOp = cast<T>(op);
    if (castOp.offset().hasValue()) {
      // assigned already
      return matchFailure();
    }

    auto curPos = *pos_;
    auto type = castOp.getResult()->getType().template cast<TensorType>();
    std::vector<int64_t> shape = type.getShape();
    auto count = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto size = count * sizeof(int8_t);
    // pad to alignment
    if (size % alignment_) {
      size = size + alignment_ - (size % alignment_);
    }
    auto newPos = curPos + size;

    llvm::errs() << llvm::format("[%-36s][%8d] : [ ",
                                 castOp.name().getValue().str().c_str(),
                                 size)
                 << llvm::format_hex(curPos, 10) << " --> "
                 << llvm::format_hex(newPos, 10) << " ]\n";
    // expand to dims=4
    while (shape.size() < 4)
      shape.insert(shape.begin(), 1);
    map_os_ << castOp.name() << "," << llvm::format_hex(curPos, 10) << ",int8,"
            << shape[0] << "," << shape[1] << ","
            << shape[2] << "," << shape[3] << "\n";

    castOp.setAttr("offset", rewriter.getI64IntegerAttr(curPos));
    *pos_ = newPos;

    return matchSuccess();
  }

  uint64_t *pos_;
  llvm::raw_ostream &map_os_;
  size_t alignment_;
};

static llvm::cl::opt<size_t> clNeuronAlignment(
    "tpu-neuron-address-align",
    llvm::cl::desc("Specify the alignment for neuron"),
    llvm::cl::init(16));

static llvm::cl::opt<std::string> clNeuronMapFilename(
    "tpu-neuron-map-filename",
    llvm::cl::desc("record neuron offset with its name into a csv map file"),
    llvm::cl::init("-"));

class AssignNeuronAddressPass : public FunctionPass<AssignNeuronAddressPass> {
public:
  explicit AssignNeuronAddressPass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    // create a map file
    std::unique_ptr<llvm::ToolOutputFile> neuronMapFile = nullptr;
    if (clNeuronMapFilename != "-") {
      std::string errorMessage;
      neuronMapFile = openOutputFile(clNeuronMapFilename, &errorMessage);
      if (!neuronMapFile) {
        llvm::errs() << errorMessage << "\n";
        exit(1);
      }
    }

    // TODO: to add mutex to pretect pos for thread safe
    uint64_t pos = 0;
    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<TpuQuantizationOpPattern<tpu::QuantizationOp> >(
        context, "tpu.quantization", &pos, neuronMapFile->os(), clNeuronAlignment);
    // assigne quantizationOp first, as input has to be at address 0
    // TODO: remove this constrain, input should be able to be any address
    applyPatternsGreedily(fn, patterns);
    patterns.clear();
    patterns.insert<TpuQuantizationOpPattern<tpu::Conv2DOp> >(
        context, "tpu.conv_2d", &pos, neuronMapFile->os(), clNeuronAlignment);
    patterns.insert<TpuQuantizationOpPattern<tpu::FullyConnectedOp> >(
        context, "tpu.fully_connected", &pos, neuronMapFile->os(), clNeuronAlignment);
    patterns.insert<TpuQuantizationOpPattern<tpu::Pool2DOp> >(
        context, "tpu.pool_2d", &pos, neuronMapFile->os(), clNeuronAlignment);
    patterns.insert<TpuQuantizationOpPattern<tpu::ReluOp> >(
        context, "tpu.relu", &pos, neuronMapFile->os(), clNeuronAlignment);
    patterns.insert<TpuQuantizationOpPattern<tpu::EltwiseOp> >(
        context, "tpu.eltwise", &pos, neuronMapFile->os(), clNeuronAlignment);
    applyPatternsGreedily(fn, patterns);

    if (neuronMapFile) {
      neuronMapFile->keep();
    }
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

std::unique_ptr<FunctionPassBase> mlir::createAssignNeuronAddressPass() {
  return std::make_unique<AssignNeuronAddressPass>();
}

static PassRegistration<AssignNeuronAddressPass>
    pass("assign-neuron-address",
         "Assign address to each neuron");