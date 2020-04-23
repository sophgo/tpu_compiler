//===- AssignNeuronAddress.cpp - assigned neuron address ------------------===//
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
// This file assined neuron address
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

#define DEBUG_TYPE "assign_neuron_address"

using namespace mlir;

namespace {

template<typename OpTy>
struct AssignGAddrPattern : public RewritePattern {
  AssignGAddrPattern(MLIRContext *context,
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
    auto elementType = type.getElementType();
    auto count = std::accumulate(std::begin(shape), std::end(shape),
                                 1, std::multiplies<>());
    uint32_t dtype_size;
    std::string dtype;
    if (elementType.isF32()) {
      dtype_size = sizeof(float);
      dtype = "fp32";
    } else if (elementType.isInteger(8)) {
      dtype_size = sizeof(int8_t);
      dtype = "int8";
    } else if (elementType.isBF16()) {
      dtype_size = sizeof(uint16_t);
      dtype = "uint16";
    } else {
      llvm_unreachable("unsupported data type");
    }

    size_t size = count * dtype_size;

    // pad to alignment
    if (size % alignment_) {
      size = size + alignment_ - (size % alignment_);
    }
    auto newPos = curPos + size;

    LLVM_DEBUG(llvm::errs() << llvm::format("[%-36s][%8d] : [ ",
                                 getOpName(op).str().c_str(), size)
                 << llvm::format_hex(curPos, 10) << " --> "
                 << llvm::format_hex(newPos, 10) << " ]\n";);
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

template <typename OpTy> struct TpuSliceAddressPattern : public RewritePattern {
  TpuSliceAddressPattern(MLIRContext *context, uint64_t *pos,
                         llvm::raw_ostream &map_os, size_t alignment)
      : RewritePattern(OpTy::getOperationName(), 1, context), pos_(pos),
        map_os_(map_os), alignment_(alignment) {}

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
    auto curPos = getPreviousOpAddress(castOp);
    int axis = castOp.axis().getLimitedValue();
    int offset = castOp.offset().getLimitedValue();

    assert((axis == 1) && "axis should be 1");
    size_t dtype_bytes;
    std::string dtype;
    if (isa<tpu::TG_INT8_SliceOp>(op)) {
      dtype_bytes = 1;
      dtype = "int8";
    } else if (isa<tpu::TG_BF16_SliceOp>(op)) {
      dtype_bytes = 2;
      dtype = "uint16";
    } else {
      llvm_unreachable("unhandled op");
    }

    std::vector<int64_t> input_shape = getTensorShape(castOp.input());
    if (input_shape[0] == 1) {
      // if batch is 1, then no copy
      int64_t isz = 1;
      for (unsigned i = axis + 1; i < input_shape.size(); i++) {
        isz *= input_shape[i];
      }
      size_t offset_bytes = offset * isz * dtype_bytes;
      setOpAddress(op, curPos + offset_bytes);

    } else {
      auto curPos = *pos_;
      std::vector<int64_t> shape = getTensorShape(castOp.getResult());
      auto count = std::accumulate(std::begin(shape), std::end(shape), 1,
                                   std::multiplies<>());
      size_t size = count * dtype_bytes;

      // pad to alignment
      if (size % alignment_) {
        size = size + alignment_ - (size % alignment_);
      }
      auto newPos = curPos + size;

      LLVM_DEBUG(llvm::errs() << llvm::format("[%-36s][%8d] : [ ",
                                   getOpName(op).str().c_str(), size)
                   << llvm::format_hex(curPos, 10) << " --> "
                   << llvm::format_hex(newPos, 10) << " ]\n";);
      // expand to dims=4
      while (shape.size() < 4)
        shape.insert(shape.begin(), 1);
      map_os_ << getOpName(op) << "," << llvm::format_hex(curPos, 10) << ","
              << dtype << "," << shape[0] << "," << shape[1] << "," << shape[2]
              << "," << shape[3] << "\n";

      setOpAddress(op, curPos);
      *pos_ = newPos;
    }
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
  explicit AssignNeuronAddressPass() {}

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

    // assigne gaddr for Ops
    patterns.clear();
    patterns.insert<
        // tg int8 ops
        AssignGAddrPattern<tpu::TG_INT8_BroadcastMulOp>,
        AssignGAddrPattern<tpu::TG_INT8_ConcatOp>,
        AssignGAddrPattern<tpu::TG_INT8_PT_Conv2DOp>,
        AssignGAddrPattern<tpu::TG_INT8_PC_Conv2DOp>,
        AssignGAddrPattern<tpu::TG_INT8_CropOp>,
        AssignGAddrPattern<tpu::TG_INT8_PT_DeConv2DOp>,
        AssignGAddrPattern<tpu::TG_INT8_PC_DeConv2DOp>,
        AssignGAddrPattern<tpu::TG_INT8_EltwiseAddOp>,
        AssignGAddrPattern<tpu::TG_INT8_EltwiseMaxOp>,
        AssignGAddrPattern<tpu::TG_INT8_EltwiseMulOp>,
        AssignGAddrPattern<tpu::TG_INT8_FullyConnectedOp>,
        AssignGAddrPattern<tpu::TG_INT8_LeakyReluOp>,
        AssignGAddrPattern<tpu::TG_INT8_LutOp>,
        AssignGAddrPattern<tpu::TG_INT8_LrnOp>,
        AssignGAddrPattern<tpu::TG_INT8_PermuteOp>,
        AssignGAddrPattern<tpu::TG_INT8_PoolAvg2DOp>,
        AssignGAddrPattern<tpu::TG_INT8_PoolMax2DOp>,
        AssignGAddrPattern<tpu::TG_INT8_ShuffleChannelOp>,
        AssignGAddrPattern<tpu::TG_INT8_SwapChannelOp>,
        AssignGAddrPattern<tpu::TG_INT8_PixelShuffleOp>,
        AssignGAddrPattern<tpu::TG_INT8_ClipOp>,
        AssignGAddrPattern<tpu::TG_INT8_PReluOp>,
        AssignGAddrPattern<tpu::TG_INT8_ReluOp>,
        AssignGAddrPattern<tpu::TG_INT8_UpsampleOp>,

        // tg bf16 ops
        AssignGAddrPattern<tpu::TG_BF16_BroadcastMulOp>,
        AssignGAddrPattern<tpu::TG_BF16_ConcatOp>,
        AssignGAddrPattern<tpu::TG_BF16_Conv2DOp>,
        AssignGAddrPattern<tpu::TG_BF16_CropOp>,
        AssignGAddrPattern<tpu::TG_BF16_DeConv2DOp>,
        AssignGAddrPattern<tpu::TG_BF16_EltwiseAddOp>,
        AssignGAddrPattern<tpu::TG_BF16_EltwiseMaxOp>,
        AssignGAddrPattern<tpu::TG_BF16_EltwiseMulOp>,
        AssignGAddrPattern<tpu::TG_BF16_FullyConnectedOp>,
        AssignGAddrPattern<tpu::TG_BF16_LeakyReluOp>,
        AssignGAddrPattern<tpu::TG_BF16_LutOp>,
        AssignGAddrPattern<tpu::TG_BF16_LrnOp>,
        AssignGAddrPattern<tpu::TG_BF16_PermuteOp>,
        AssignGAddrPattern<tpu::TG_BF16_PoolAvg2DOp>,
        AssignGAddrPattern<tpu::TG_BF16_PoolMax2DOp>,
        AssignGAddrPattern<tpu::TG_BF16_PReluOp>,
        AssignGAddrPattern<tpu::TG_BF16_ReluOp>,
        AssignGAddrPattern<tpu::TG_BF16_ClipOp>,
        AssignGAddrPattern<tpu::TG_BF16_ShuffleChannelOp>,
        AssignGAddrPattern<tpu::TG_BF16_ClipOp>,
        AssignGAddrPattern<tpu::TG_BF16_SwapChannelOp>,
        AssignGAddrPattern<tpu::TG_BF16_PixelShuffleOp>,
        AssignGAddrPattern<tpu::TG_BF16_UpsampleOp>,

        // fp32 cpu ops
        AssignGAddrPattern<tpu::QuantOp>

        >(context, &pos, neuronMapFile->os(), clNeuronAlignment);
    applyPatternsGreedily(fn, patterns);

    // no copy address assignment
    patterns.clear();
    patterns.insert<
        TpuSliceAddressPattern<tpu::TG_INT8_SliceOp>,
        TpuSliceAddressPattern<tpu::TG_BF16_SliceOp>
        >(context, &pos, neuronMapFile->os(), clNeuronAlignment);
    applyPatternsGreedily(fn, patterns);

    if (neuronMapFile) {
      neuronMapFile->keep();
    }
  }
};

} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createAssignNeuronAddressPass() {
  return std::make_unique<AssignNeuronAddressPass>();
}

static PassRegistration<AssignNeuronAddressPass>
    pass("assign-neuron-address",
         "Assign address to each neuron");
