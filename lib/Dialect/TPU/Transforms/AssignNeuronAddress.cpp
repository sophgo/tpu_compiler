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

template<typename OpTy>
struct TpuReshapeAddressPattern : public RewritePattern {
  TpuReshapeAddressPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto castOp = cast<OpTy>(op);
    if (castOp.gaddr().hasValue()) {
      // assigned already
      return matchFailure();
    }

    auto prevPos = getPreviousOpAddress(op);
    setOpAddress(op, prevPos);

    return matchSuccess();
  }
};

template<typename OpTy>
struct TpuSliceAddressPattern : public RewritePattern {
  TpuSliceAddressPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto castOp = cast<OpTy>(op);
    if (castOp.gaddr().hasValue()) {
      // assigned already
      return matchFailure();
    }

    // trying to avoid doing copy
    // however, since we didn't pass stride info to backend API yet
    // this is only working for batch_size = 1
    std::vector<int64_t> input_shape = getTensorShape(castOp.input());
    assert(input_shape[0] == 1);

    auto curPos = getPreviousOpAddress(castOp);
    int axis  = castOp.axis().getLimitedValue();
    int offset  = castOp.offset().getLimitedValue();

    assert(axis == 1);
    int64_t isz = 1;
    for (unsigned i = axis + 1; i < input_shape.size(); i++) {
      isz *= input_shape[i];
    }
    int64_t count = offset * isz;

    size_t offset_bytes;
    if (isa<tpu::TG_INT8_SliceOp>(op)) {
      offset_bytes = count * sizeof(int8_t);
    } else if (isa<tpu::TG_BF16_SliceOp>(op)) {
      offset_bytes = count * sizeof(uint16_t);
    } else {
      assert(0);
    }

    setOpAddress(op, curPos + offset_bytes);

    return matchSuccess();
  }
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

    setOpAddress(op, curPos);
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

    // assigne quantizationOp first, as input has to be at address 0
    // TODO: remove this constrain, input should be able to be any address
    patterns.insert<TpuQuantizationOpPattern<tpu::QuantizationOp>>(
        context, &pos, neuronMapFile->os(), clNeuronAlignment);
    applyPatternsGreedily(fn, patterns);

    // assigne gaddr for TG Ops
    patterns.clear();
    patterns.insert<
        // tg int8 ops
        AssignGAddrTGInt8Pattern<tpu::TG_INT8_BroadcastMulOp>,
        AssignGAddrTGInt8Pattern<tpu::TG_INT8_ConcatOp>,
        AssignGAddrTGInt8Pattern<tpu::TG_INT8_PT_Conv2DOp>,
        AssignGAddrTGInt8Pattern<tpu::TG_INT8_PC_Conv2DOp>,
        AssignGAddrTGInt8Pattern<tpu::TG_INT8_CropOp>,
        AssignGAddrTGInt8Pattern<tpu::TG_INT8_PT_DeConv2DOp>,
        AssignGAddrTGInt8Pattern<tpu::TG_INT8_PC_DeConv2DOp>,
        AssignGAddrTGInt8Pattern<tpu::TG_INT8_DivOp>,
        AssignGAddrTGInt8Pattern<tpu::TG_INT8_EltwiseAddOp>,
        AssignGAddrTGInt8Pattern<tpu::TG_INT8_EltwiseMaxOp>,
        AssignGAddrTGInt8Pattern<tpu::TG_INT8_EltwiseMulOp>,
        AssignGAddrTGInt8Pattern<tpu::TG_INT8_FullyConnectedOp>,
        AssignGAddrTGInt8Pattern<tpu::TG_INT8_LeakyReluOp>,
        AssignGAddrTGInt8Pattern<tpu::TG_INT8_PermuteOp>,
        AssignGAddrTGInt8Pattern<tpu::TG_INT8_PoolAvg2DOp>,
        AssignGAddrTGInt8Pattern<tpu::TG_INT8_PoolMax2DOp>,
        AssignGAddrTGInt8Pattern<tpu::TG_INT8_ShuffleChannelOp>,
        AssignGAddrTGInt8Pattern<tpu::TG_INT8_PReluOp>,
        AssignGAddrTGInt8Pattern<tpu::TG_INT8_SigmoidOp>,
        AssignGAddrTGInt8Pattern<tpu::TG_INT8_SqrtOp>,
        AssignGAddrTGInt8Pattern<tpu::TG_INT8_UpsampleOp>,

        // tg bf16 ops
        AssignGAddrTGInt8Pattern<tpu::TG_BF16_BroadcastMulOp>,
        AssignGAddrTGBf16Pattern<tpu::TG_BF16_ConcatOp>,
        AssignGAddrTGBf16Pattern<tpu::TG_BF16_Conv2DOp>,
        AssignGAddrTGBf16Pattern<tpu::TG_BF16_CropOp>,
        AssignGAddrTGBf16Pattern<tpu::TG_BF16_DeConv2DOp>,
        AssignGAddrTGBf16Pattern<tpu::TG_BF16_DivOp>,
        AssignGAddrTGBf16Pattern<tpu::TG_BF16_EltwiseAddOp>,
        AssignGAddrTGBf16Pattern<tpu::TG_BF16_EltwiseMaxOp>,
        AssignGAddrTGBf16Pattern<tpu::TG_BF16_EltwiseMulOp>,
        AssignGAddrTGBf16Pattern<tpu::TG_BF16_FullyConnectedOp>,
        AssignGAddrTGBf16Pattern<tpu::TG_BF16_LeakyReluOp>,
        AssignGAddrTGBf16Pattern<tpu::TG_BF16_PermuteOp>,
        AssignGAddrTGBf16Pattern<tpu::TG_BF16_PoolAvg2DOp>,
        AssignGAddrTGBf16Pattern<tpu::TG_BF16_PoolMax2DOp>,
        AssignGAddrTGBf16Pattern<tpu::TG_BF16_PReluOp>,
        AssignGAddrTGBf16Pattern<tpu::TG_BF16_ShuffleChannelOp>,
        AssignGAddrTGBf16Pattern<tpu::TG_BF16_SigmoidOp>,
        AssignGAddrTGBf16Pattern<tpu::TG_BF16_SqrtOp>,
        AssignGAddrTGBf16Pattern<tpu::TG_BF16_UpsampleOp>

        >(context, &pos, neuronMapFile->os(), clNeuronAlignment);
    applyPatternsGreedily(fn, patterns);

    // to be removed
    patterns.clear();
    patterns.insert<
        TpuQuantizationOpPattern<tpu::PowerOp>,
        TpuQuantizationOpPattern<tpu::TanHOp>
        >(context, &pos, neuronMapFile->os(), clNeuronAlignment);
    applyPatternsGreedily(fn, patterns);

    // no copy address assignment
    patterns.clear();
    patterns.insert<
        TpuSliceAddressPattern<tpu::TG_INT8_SliceOp>,
        TpuSliceAddressPattern<tpu::TG_BF16_SliceOp>,
        TpuReshapeAddressPattern<tpu::TG_INT8_ReshapeOp>,
        TpuReshapeAddressPattern<tpu::TG_BF16_ReshapeOp>
        >(context);
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
