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

template<typename OpTy>
struct AssignGAddrTGInt8Pattern : public RewritePattern {
  AssignGAddrTGInt8Pattern(MLIRContext *context,
      uint64_t *pos, llvm::raw_ostream &map_os, size_t alignment)
      : RewritePattern(OpTy::getOperationName(), 1, context),
        pos_(pos),
        map_os_(map_os),
        alignment_(alignment) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto castOp = cast<OpTy>(op);
    if (castOp.gaddr().hasValue()) {
      // assigned already
      return matchFailure();
    }

    auto curPos = *pos_;
    auto type = castOp.getResult()->getType().template cast<TensorType>();
    std::vector<int64_t> shape = type.getShape();
    auto count = std::accumulate(std::begin(shape), std::end(shape),
                                 1, std::multiplies<>());
    size_t size;
    std::string dtype;
    size = count * sizeof(int8_t);
    dtype = "int8";

    // pad to alignment
    if (size % alignment_) {
      size = size + alignment_ - (size % alignment_);
    }
    auto newPos = curPos + size;

    llvm::errs() << llvm::format("[%-36s][%8d] : [ ",
                                 getOpName(op).str().c_str(), size)
                 << llvm::format_hex(curPos, 10) << " --> "
                 << llvm::format_hex(newPos, 10) << " ]\n";
    // expand to dims=4
    while (shape.size() < 4)
      shape.insert(shape.begin(), 1);
    map_os_ << getOpName(op) << "," << llvm::format_hex(curPos, 10) << ","
            << dtype << ","
            << shape[0] << "," << shape[1] << ","
            << shape[2] << "," << shape[3] << "\n";

    setOpAddress(op, curPos);
    *pos_ = newPos;

    return matchSuccess();
  }

  uint64_t *pos_;
  llvm::raw_ostream &map_os_;
  size_t alignment_;
};


template<typename OpTy>
struct AssignGAddrTGBf16Pattern : public RewritePattern {
  AssignGAddrTGBf16Pattern(MLIRContext *context,
      uint64_t *pos, llvm::raw_ostream &map_os, size_t alignment)
      : RewritePattern(OpTy::getOperationName(), 1, context),
        pos_(pos),
        map_os_(map_os),
        alignment_(alignment) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto castOp = cast<OpTy>(op);
    if (castOp.gaddr().hasValue()) {
      // assigned already
      return matchFailure();
    }

    auto curPos = *pos_;
    auto type = castOp.getResult()->getType().template cast<TensorType>();
    std::vector<int64_t> shape = type.getShape();
    auto count = std::accumulate(std::begin(shape), std::end(shape),
                                 1, std::multiplies<>());
    size_t size;
    std::string dtype;
    size = count * sizeof(uint16_t);
    dtype = "uint16";

    // pad to alignment
    if (size % alignment_) {
      size = size + alignment_ - (size % alignment_);
    }
    auto newPos = curPos + size;

    llvm::errs() << llvm::format("[%-36s][%8d] : [ ",
                                 getOpName(op).str().c_str(), size)
                 << llvm::format_hex(curPos, 10) << " --> "
                 << llvm::format_hex(newPos, 10) << " ]\n";
    // expand to dims=4
    while (shape.size() < 4)
      shape.insert(shape.begin(), 1);
    map_os_ << getOpName(op) << "," << llvm::format_hex(curPos, 10) << ","
            << dtype << ","
            << shape[0] << "," << shape[1] << ","
            << shape[2] << "," << shape[3] << "\n";

    setOpAddress(op, curPos);
    *pos_ = newPos;

    return matchSuccess();
  }

  uint64_t *pos_;
  llvm::raw_ostream &map_os_;
  size_t alignment_;
};



// to be removed
template<typename OpTy>
struct TpuQuantizationOpPattern : public RewritePattern {
  TpuQuantizationOpPattern(MLIRContext *context,
      uint64_t *pos, llvm::raw_ostream &map_os, size_t alignment)
      : RewritePattern(OpTy::getOperationName(), 1, context),
        pos_(pos),
        map_os_(map_os),
        alignment_(alignment) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto castOp = cast<OpTy>(op);
    if (castOp.offset().hasValue()) {
      // assigned already
      return matchFailure();
    }

    auto curPos = *pos_;
    auto type = castOp.getResult()->getType().template cast<TensorType>();
    std::vector<int64_t> shape = type.getShape();
    auto count = std::accumulate(std::begin(shape), std::end(shape),
                                 1, std::multiplies<>());
    size_t size;
    std::string dtype;
    if (castOp.quant() == "INT8" || castOp.quant() == "INT8_PER_CHANNEL"
        || castOp.quant() == "INT8_MULTIPLIER") {
      size = count * sizeof(int8_t);
      dtype = "int8";
    } else if (castOp.quant() == "BF16") {
      size = count * sizeof(uint16_t);
      dtype = "uint16";
    } else {
      assert(0);
    }
    // pad to alignment
    if (size % alignment_) {
      size = size + alignment_ - (size % alignment_);
    }
    auto newPos = curPos + size;

    llvm::errs() << llvm::format("[%-36s][%8d] : [ ",
                                 getOpName(op).str().c_str(), size)
                 << llvm::format_hex(curPos, 10) << " --> "
                 << llvm::format_hex(newPos, 10) << " ]\n";
    // expand to dims=4
    while (shape.size() < 4)
      shape.insert(shape.begin(), 1);
    map_os_ << castOp.name() << "," << llvm::format_hex(curPos, 10) << ","
            << dtype << ","
            << shape[0] << "," << shape[1] << ","
            << shape[2] << "," << shape[3] << "\n";

    //castOp.setAttr("offset", rewriter.getI64IntegerAttr(curPos));
    setOpAddress(op, curPos);
    *pos_ = newPos;

    return matchSuccess();
  }

  uint64_t *pos_;
  llvm::raw_ostream &map_os_;
  size_t alignment_;
};

struct TpuSliceAddressPattern : public RewritePattern {
  TpuSliceAddressPattern(MLIRContext *context)
      : RewritePattern("tpu.slice", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto castOp = cast<tpu::SliceOp>(op);
    if (castOp.offset().hasValue()) {
      // assigned already
      return matchFailure();
    }

    auto curPos = getPreviousOpAddress(castOp);
    int32_t count = castOp.input_offset().getValue().getLimitedValue();

    size_t size;
    if (castOp.quant() == "INT8" || castOp.quant() == "INT8_PER_CHANNEL"
        || castOp.quant() == "INT8_MULTIPLIER") {
      size = count * sizeof(int8_t);
    } else if (castOp.quant() == "BF16") {
      size = count * sizeof(uint16_t);
    } else {
      assert(0);
    }

    castOp.setAttr("offset", rewriter.getI64IntegerAttr(curPos + size));

    return matchSuccess();
  }
};

struct TpuReshapeAddressPattern : public RewritePattern {
  TpuReshapeAddressPattern(MLIRContext *context)
      : RewritePattern("tpu.reshape", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto castOp = cast<tpu::ReshapeOp>(op);
    if (castOp.offset().hasValue()) {
      // assigned already
      return matchFailure();
    }

    auto prevPos = getPreviousOpAddress(castOp);
    castOp.setAttr("offset", rewriter.getI64IntegerAttr(prevPos));

    return matchSuccess();
  }
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

    // assigne quantizationOp first, as input has to be at address 0
    // TODO: remove this constrain, input should be able to be any address
    patterns.insert<TpuQuantizationOpPattern<tpu::QuantizationOp>>(
        context, &pos, neuronMapFile->os(), clNeuronAlignment);
    applyPatternsGreedily(fn, patterns);
    patterns.clear();

    // assigne gaddr for TG Ops
    patterns.insert<
          // tg int8 ops
          AssignGAddrTGInt8Pattern<tpu::TG_INT8_ConcatOp>,
          AssignGAddrTGInt8Pattern<tpu::TG_INT8_PT_Conv2DOp>,
          AssignGAddrTGInt8Pattern<tpu::TG_INT8_PC_Conv2DOp>,
          AssignGAddrTGInt8Pattern<tpu::TG_INT8_EltwiseAddOp>,
          AssignGAddrTGInt8Pattern<tpu::TG_INT8_EltwiseMaxOp>,
          AssignGAddrTGInt8Pattern<tpu::TG_INT8_EltwiseMulOp>,
          AssignGAddrTGInt8Pattern<tpu::TG_INT8_LeakyReluOp>,
          AssignGAddrTGInt8Pattern<tpu::TG_INT8_PoolAvg2DOp>,
          AssignGAddrTGInt8Pattern<tpu::TG_INT8_PoolMax2DOp>,

          // tg bf16 ops
          AssignGAddrTGBf16Pattern<tpu::TG_BF16_ConcatOp>,
          AssignGAddrTGBf16Pattern<tpu::TG_BF16_Conv2DOp>,
          AssignGAddrTGBf16Pattern<tpu::TG_BF16_EltwiseAddOp>,
          AssignGAddrTGBf16Pattern<tpu::TG_BF16_EltwiseMaxOp>,
          AssignGAddrTGBf16Pattern<tpu::TG_BF16_EltwiseMulOp>,
          AssignGAddrTGBf16Pattern<tpu::TG_BF16_LeakyReluOp>,
          AssignGAddrTGBf16Pattern<tpu::TG_BF16_PoolAvg2DOp>,
          AssignGAddrTGBf16Pattern<tpu::TG_BF16_PoolMax2DOp>
        >(context, &pos, neuronMapFile->os(), clNeuronAlignment);
    applyPatternsGreedily(fn, patterns);
    patterns.clear();

    patterns.insert<
          TpuQuantizationOpPattern<tpu::DeConv2DOp>,
          TpuQuantizationOpPattern<tpu::DivOp>,
          TpuQuantizationOpPattern<tpu::CropOp>,
          TpuQuantizationOpPattern<tpu::FullyConnectedOp>,
          TpuQuantizationOpPattern<tpu::PermuteOp>,
          TpuQuantizationOpPattern<tpu::PowerOp>,
          TpuQuantizationOpPattern<tpu::PReluOp>,
          TpuQuantizationOpPattern<tpu::SigmoidOp>,
          TpuQuantizationOpPattern<tpu::ScaleOp>,
          TpuQuantizationOpPattern<tpu::SqrtOp>,
          TpuQuantizationOpPattern<tpu::TanHOp>
        >(context, &pos, neuronMapFile->os(), clNeuronAlignment);
    applyPatternsGreedily(fn, patterns);
    patterns.clear();

    patterns.insert<TpuSliceAddressPattern, TpuReshapeAddressPattern>(context);
    applyPatternsGreedily(fn, patterns);

    if (neuronMapFile) {
      neuronMapFile->keep();
    }
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createAssignNeuronAddressPass() {
  return std::make_unique<AssignNeuronAddressPass>();
}

static PassRegistration<AssignNeuronAddressPass>
    pass("assign-neuron-address",
         "Assign address to each neuron");
