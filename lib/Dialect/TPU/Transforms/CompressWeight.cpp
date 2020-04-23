//===- CompressWeight- Implementation of weight compression --------------===//
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
// This file implements the weight compression.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/Dialect/TPU/TPUCompressUtil.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"

#include "../DeepFusion/MachineInfo.h"

#define DEBUG_TYPE "compress-weight"

using namespace mlir;

namespace {

static llvm::cl::opt<std::string> clCompressedWeightMapFileName(
    "tpu-compressed-weight-map-filename",
    llvm::cl::desc("record neuron offset with its name into a csv map file"),
    llvm::cl::init("-"));

struct CompressInfo {
  llvm::StringRef name;
  uint64_t size;
  uint64_t compressedSize;
};

template <typename TensorTyOp>
class CompressConvolutionWeightPattern : public OpRewritePattern<TensorTyOp> {
public:
  using OpRewritePattern<TensorTyOp>::OpRewritePattern;

  CompressConvolutionWeightPattern(MLIRContext *ctx,
      std::vector<CompressInfo> &compressedWeigtInfos)
      : OpRewritePattern<TensorTyOp>(ctx),
        compressedWeigtInfos_(compressedWeigtInfos) {}

  PatternMatchResult matchAndRewrite(TensorTyOp convOp,
                                     PatternRewriter &rewriter) const override {

    // Already compressed.
    if (convOp.compressed_weight().hasValue())
      return Pattern::matchFailure();

    // Not support group convolution and depthwise convolution
    if (convOp.param().group().getValue().getLimitedValue() > 1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "CompressWeight: layer ID " << convOp.layer_id()
                 << ", " << convOp.name()
                 << ", groups " << convOp.param().group().getValue().getLimitedValue()
                 << ", not support group convolution\n");
      return Pattern::matchFailure();
    }

    auto filterTy =
        convOp.filter()->getType().template dyn_cast<RankedTensorType>();
    auto filterElementType = filterTy.getElementType();
    assert(filterElementType.isIntOrFloat() && "Expect result is int or float");
    if (!filterElementType.isIntOrFloat())
      return Pattern::matchFailure();

    // Not support bfloat16 yet.
    if (filterElementType.isBF16()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "CompressWeight: layer ID " << convOp.layer_id()
                 << ", " << convOp.name()
                 << ", not support bfloat16\n");
      return Pattern::matchFailure();
    }

    uint64_t filterDataTypeSize = filterElementType.getIntOrFloatBitWidth() / 8;

    // Only support int8 or bf16
    if (filterDataTypeSize != 1 && !filterElementType.isBF16()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "CompressWeight: layer ID " << convOp.layer_id()
                 << ", " << convOp.name()
                 << ", only support int8 or bf16\n");
      return Pattern::matchFailure();
    }

    // Same filter shape but fill compressed data
    TensorFile *wTF = getWeightTensorFile(convOp.getOperation());
    auto filter = readAndDeleteWeightTensor<int8_t>(convOp.filter(), wTF);
    int64_t filterSize;
    std::vector<int64_t> filterShape;
    getTensorShapeAndSize(convOp.filter(), filterShape, filterSize);
    assert(filterSize == (int64_t)filter->size() &&
           "filter size should be equal");

    LLVM_DEBUG(llvm::dbgs()
               << "CompressWeight: layer ID " << convOp.layer_id()
               << ", " << convOp.name() << "\n"
               << "  filter(" << filterShape[0]
               << ", " << filterShape[1]
               << ", " << filterShape[2]
               << ", " << filterShape[3]
               << "), filterDataTypeSize " << filterDataTypeSize
               << "\n");

    auto newFilter = std::make_unique<std::vector<int8_t> >(filterSize);
    std::memset(newFilter->data(), 0, filterSize);

    // Filter layout in dilact (oc, ic, kh, kw)
    // But the storage layout is already changed to (1, oc, kh*kw, ic)
    int oc = filterShape[0];
    int ic = filterShape[1];
    int kh = filterShape[2];
    int kw = filterShape[3];

    // Split output channel in unit of lane number
    bool canCompress = true;
    int oc_step = MInfo::lane_num;
    auto buffer =
        std::make_unique<std::vector<uint8_t> >(oc_step * kh * kw * ic);

    int totalSize = 0;
    int totalCompressedSize = 0;
    for (int oc_pos = 0; oc_pos < oc; oc_pos += oc_step) {
      int cur_oc = std::min(oc - oc_pos, oc_step);
      int step_size = cur_oc * kh * kw * ic;
      int pos = oc_pos * kh * kw * ic;

      // H/W constrain: must align 16B
      if (pos % 16) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  [oc_pos=" << oc_pos << "] cur_oc " << cur_oc
                   << ", step_size " << step_size
                   << ", pos " << pos
                   << ", not aligned, SKIP\n");
        canCompress = false;
        break;
      }

      // Compressing uses uint8 data type.
      // Create unit8 tiled buffer for it.
      auto plainData = std::make_unique<std::vector<uint8_t> >(step_size);
      std::memcpy(plainData->data(), filter->data() + pos, step_size);

      // Calculate compress parametar first.
      CompressCommandInfo cmdInfo;
      std::memset(&cmdInfo, 0, sizeof(cmdInfo));
      cmdInfo.signedness = 1; // int8
      cmdInfo.is_bfloat16 = 0;
      getCompressParameter(plainData->data(), step_size, cmdInfo.signedness,
                          cmdInfo.is_bfloat16, &cmdInfo);

      // Create Compress data.
      int requiredSize = getCompressedDataSize(step_size, /*dataType*/0);
      auto compressedData =
          std::make_unique<std::vector<uint8_t> >(requiredSize);
      int compressedSize = 0;
      compressInt8Data(plainData->data(), step_size, compressedData->data(),
                       &compressedSize, &cmdInfo);

      // Compress size must be less than tiled size.
      if (compressedSize > step_size) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  [oc_pos=" << oc_pos << "] cur_oc " << cur_oc
                   << ", step_size " << step_size
                   << ", compressedSize " << compressedSize
                   << ", SKIP\n");
        canCompress = false;
        break;
      } else {
        totalSize += step_size;
        totalCompressedSize += compressedSize;
      }

      // Fill compressed data.
      std::memcpy(newFilter->data() + pos, compressedData->data(),
                  compressedSize);
    }

    if (filterDataTypeSize == 1) {
    } else if (filterElementType.isBF16()) {
      assert(filterDataTypeSize == 2 && "Expect data type size of bf16 is 2");
    }

    if (canCompress) {
      addWeightTensorAndUpdateWeightOp<int8_t>(convOp.filter(),
          "z", *newFilter, filterShape, "INT8", wTF);

      assert(memcmp(filter->data(), newFilter->data(), filterSize)
             && "Expect compressed content");

      convOp.setAttr("tiled_oc_step", rewriter.getI32IntegerAttr(oc_step));
      convOp.setAttr("compressed_weight", rewriter.getBoolAttr(true));

      struct CompressInfo info;
      info.name = convOp.name();
      info.size = totalSize;
      info.compressedSize = totalCompressedSize;
      compressedWeigtInfos_.push_back(info);

      LLVM_DEBUG(llvm::dbgs()
                << "  compressedWeigtInfos size "
                << compressedWeigtInfos_.size() << "\n");

      return Pattern::matchSuccess();
    } else  {
      addWeightTensorAndUpdateWeightOp<int8_t>(convOp.filter(),
          "", *filter, filterShape, "INT8", wTF);

      return Pattern::matchFailure();
    }
  }

  std::vector<struct CompressInfo> &compressedWeigtInfos_;
};

struct CompressWeightPass : public FunctionPass<CompressWeightPass> {
  void runOnFunction() override;

  void generateReport(std::vector<struct CompressInfo> &compressedWeigtInfos);
};
} // anonymous namespace

void CompressWeightPass::generateReport(
    std::vector<struct CompressInfo> &compressedWeigtInfos) {

  // Create a map file if compressed weight existed.
  if (clCompressedWeightMapFileName == "-" || !compressedWeigtInfos.size())
    return;

  std::string errorMessage;
  std::unique_ptr<llvm::ToolOutputFile> outputFile =
      openOutputFile(clCompressedWeightMapFileName, &errorMessage);
  if (!outputFile) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  auto &outputOS = outputFile->os();
  uint64_t totalSize = 0;
  uint64_t totalCompressedSize = 0;
  outputOS << "name, totalSize, totalCompressedSize, reduced, ratio\n";
  for (auto info : compressedWeigtInfos) {
    totalSize += info.size;
    totalCompressedSize += info.compressedSize;
    outputOS << info.name << ", " << info.size
             << ", " << info.compressedSize
             << ", " << info.size - info.compressedSize
             << ", "
             << int(info.compressedSize * 1.0 / info.size * 100.0)
             << "%\n";
  }

  outputOS << "totalSize " << totalSize
           << ", totalCompressedSize " << totalCompressedSize
           << ", reduced " << totalSize - totalCompressedSize
           << ", ratio " << int(totalCompressedSize * 1.0 / totalSize * 100.0)
           << "%\n";
  outputFile->keep();
}

void CompressWeightPass::runOnFunction() {
  std::vector<struct CompressInfo> compressedWeigtInfos;

  // Compress convolution weight
  OwningRewritePatternList patterns;
  patterns.insert<
      CompressConvolutionWeightPattern<tpu::TL_LW_Conv2DOp>
      >(&getContext(), compressedWeigtInfos);
  applyPatternsGreedily(getFunction(), patterns);

  // Remove offset in load weight first.
  // Then run assign-weight-address pass to generate the compressed weight.
  getFunction().walk([&](Operation *op) {
    if (auto loadWeightOp = dyn_cast<tpu::LoadWeightOp>(op)) {
      loadWeightOp.removeAttr("offset");
    }
  });

  generateReport(compressedWeigtInfos);
}

std::unique_ptr<OpPassBase<FuncOp>> mlir::createCompressWeightPass() {
  return std::make_unique<CompressWeightPass>();
}

static PassRegistration<CompressWeightPass>
    pass("compress-weight", "Compress weight");
