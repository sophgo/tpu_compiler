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

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/MachineInfo.h"
#include "tpuc/Passes.h"
#include "tpuc/Support/TensorFile.h"
#include "tpuc/TPUCompressUtil.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

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

template <typename TensorTyOp, typename DataType>
bool tryCompressConvWeight(TensorTyOp convOp, PatternRewriter &rewriter,
                           int oc_step,
                           std::vector<CompressInfo> &compressInfos) {
  bool isBf16Flt = isBf16Tensor(convOp.filter());
  int fltEltSize = getDataTypeSize(convOp.filter());
  assert(fltEltSize == sizeof(DataType) && "Expect correct data type");

  // Same filter shape but fill compressed data
  TensorFile *wTF = getWeightTensorFile(convOp.getOperation());
  auto filter = readAndDeleteWeightTensor<DataType>(convOp.filter(), wTF);
  int64_t filterSize;
  std::vector<int64_t> filterShape;
  getTensorShapeAndSize(convOp.filter(), filterShape, filterSize);
  assert(filterSize == (int64_t)filter->size() &&
         "filter size should be equal");

  LLVM_DEBUG(llvm::dbgs() << "CompressWeight: layer ID " << getOpLayerId(convOp)
                          << ", " << convOp.name() << "\n"
                          << "  filter(" << filterShape[0] << ", "
                          << filterShape[1] << ", " << filterShape[2] << ", "
                          << filterShape[3] << "), fltEltSize " << fltEltSize
                          << "\n");

  auto newFilter = std::make_unique<std::vector<DataType>>(filterSize);
  std::memset(newFilter->data(), 0, filterSize * fltEltSize);

  int oc = filterShape[0];
  int ic = filterShape[1];
  int kh = filterShape[2];
  int kw = filterShape[3];

  bool canCompress = true;
  int totalSize = 0;
  int totalCompressedSize = 0;

  // Allocate max buffer
  int maxPlainSize = oc_step * kh * kw * ic * fltEltSize;
  auto plainData = std::make_unique<std::vector<uint8_t>>(maxPlainSize);

  int maxComprSize = getCompressedDataSize(maxPlainSize, isBf16Flt ? 1 : 0);
  auto compressedData = std::make_unique<std::vector<uint8_t>>(maxComprSize);

  for (int oc_pos = 0; oc_pos < oc; oc_pos += oc_step) {
    int cur_oc = std::min(oc - oc_pos, oc_step);
    int stepSize = cur_oc * kh * kw * ic * fltEltSize;
    int pos = oc_pos * kh * kw * ic * fltEltSize;

    // H/W constraint: must align 16B
    if (pos % 16) {
      canCompress = false;
      break;
    }

    std::memcpy(plainData->data(), filter->data() + pos / fltEltSize, stepSize);

    // Calculate compress parameter first.
    CompressCommandInfo cmdInfo;
    std::memset(&cmdInfo, 0, sizeof(cmdInfo));
    cmdInfo.signedness = isBf16Flt ? 0 : 1;
    cmdInfo.is_bfloat16 = isBf16Flt ? 1 : 0;
    cmdInfo.bias0 = isBf16Flt ? 127 : 0;
    getCompressParameter(plainData->data(), stepSize, cmdInfo.signedness,
                         cmdInfo.is_bfloat16, &cmdInfo);

    int compressedSize = maxComprSize;
    if (isBf16Flt)
      compressBf16Data(plainData->data(), stepSize, compressedData->data(),
                       &compressedSize, &cmdInfo);
    else
      compressInt8Data(plainData->data(), stepSize, compressedData->data(),
                       &compressedSize, &cmdInfo);

    // Compress size must be less than tiled size.
    LLVM_DEBUG(llvm::dbgs()
               << "  [oc_pos=" << oc_pos << "] cur_oc " << cur_oc
               << ", stepSize " << stepSize << ", compressedSize "
               << compressedSize << ", pos " << pos << ", totalSize "
               << totalSize << ", totalCompressedSize " << totalCompressedSize
               << ", filterSize " << filterSize * fltEltSize << "\n");

    if (compressedSize > stepSize) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  [oc_pos=" << oc_pos << "] cur_oc " << cur_oc
                 << ", stepSize " << stepSize << ", compressedSize "
                 << compressedSize << ", SKIP\n");
      canCompress = false;
      break;
    } else {
      totalSize += stepSize;
      totalCompressedSize += compressedSize;
    }

    // Fill compressed data.
    std::memcpy(newFilter->data() + pos / fltEltSize, compressedData->data(),
                compressedSize);
  }

  if (canCompress) {
    addWeightTensorAndUpdateWeightOp<DataType>(
        convOp.filter(), "z", *newFilter, filterShape,
        isBf16Flt ? "BF16" : "INT8", wTF);

    assert(memcmp(filter->data(), newFilter->data(), filterSize) &&
           "Expect compressed content");

    convOp->setAttr("tiled_oc_step", rewriter.getI32IntegerAttr(oc_step));
    convOp->setAttr("compressed_weight", rewriter.getBoolAttr(true));

    // set compressed flag on TL_LoadCoeffOp for layer group
    if (auto load_op =
            dyn_cast<tpu::TL_LG_LoadCoeffOp>(convOp.filter().getDefiningOp())) {
      load_op->setAttr("compressed_weight", rewriter.getBoolAttr(true));
    }

    if (auto load_op =
            dyn_cast<tpu::LoadWeightOp>(convOp.filter().getDefiningOp())) {
      load_op->setAttr("compressed", rewriter.getBoolAttr(true));
    }

    struct CompressInfo info;
    info.name = convOp.name();
    info.size = totalSize;
    info.compressedSize = totalCompressedSize;
    compressInfos.push_back(info);

    LLVM_DEBUG(llvm::dbgs()
               << "  compressInfos size " << compressInfos.size() << "\n");
  } else {
    addWeightTensorAndUpdateWeightOp<DataType>(
        convOp.filter(), "", *filter, filterShape, isBf16Flt ? "BF16" : "INT8",
        wTF);
  }

  return canCompress;
}

template <typename TensorTyOp, typename DataType>
class TlLgConvCompressedWightPattern : public OpRewritePattern<TensorTyOp> {
public:
  using OpRewritePattern<TensorTyOp>::OpRewritePattern;

  TlLgConvCompressedWightPattern(MLIRContext *ctx,
                                 std::vector<CompressInfo> &compressInfos)
      : OpRewritePattern<TensorTyOp>(ctx), compressInfos_(compressInfos) {}

  LogicalResult matchAndRewrite(TensorTyOp convOp,
                                PatternRewriter &rewriter) const override {

    // Already compressed.
    if (convOp.compressed_weight().hasValue())
      return failure();

    // for layer group, several conv may refer to one load coeff op
    // no need to compress every time.
    if (auto load_op =
            dyn_cast<tpu::TL_LG_LoadCoeffOp>(convOp.filter().getDefiningOp())) {
      if (load_op.compressed_weight().hasValue() &&
          load_op.compressed_weight().getValue()) {
        convOp->setAttr("compressed_weight", rewriter.getBoolAttr(true));
        return failure();
      }
    }

    if (auto load_op =
            dyn_cast<tpu::LoadWeightOp>(convOp.filter().getDefiningOp())) {
      if (load_op.compressed()) {
        convOp->setAttr("compressed_weight", rewriter.getBoolAttr(true));
        return failure();
      }
    }

    // Split output channel in unit of lane number for deep fusion
    int oc_step = MInfo::lane_num;
    // not split for lg
    if ((TensorTyOp::getOperationName() == "tpu.tl_lg_int8_conv_2d") ||
        (TensorTyOp::getOperationName() == "tpu.tl_lg_bf16_conv_2d")) {
      std::vector<int64_t> filterShape = getTensorShape(convOp.filter());
      oc_step = filterShape[0];
    }

    bool canCompress = tryCompressConvWeight<TensorTyOp, DataType>(
        convOp, rewriter, oc_step, compressInfos_);
    if (canCompress)
      return success();

    return failure();
  }

  std::vector<struct CompressInfo> &compressInfos_;
};

template <typename TensorTyOp, typename DataType>
class TgConvCompressedWeightPattern : public OpRewritePattern<TensorTyOp> {
public:
  using OpRewritePattern<TensorTyOp>::OpRewritePattern;

  TgConvCompressedWeightPattern(MLIRContext *ctx,
                                std::vector<CompressInfo> &compressInfos)
      : OpRewritePattern<TensorTyOp>(ctx), compressInfos_(compressInfos) {}

  LogicalResult matchAndRewrite(TensorTyOp convOp,
                                PatternRewriter &rewriter) const override {


    // Not support group convolution and depthwise convolution
    if (convOp.param().group().getInt() > 1)
      return failure();

    convOp->setAttr("do_compress", rewriter.getBoolAttr(true));
    return success();
  }

  std::vector<struct CompressInfo> &compressInfos_;
};

template <typename T>
static void stridedMatrixMemcpy(T *dstPtr, T *srcPtr, int srcStride, int H,
                                int W) {
  for (int i = 0; i < H; ++i) {
    for (int j = 0; j < W; ++j) {
      dstPtr[i * W + j] = srcPtr[i * srcStride + j];
    }
  }
}

struct CompressWeightPass
    : public mlir::PassWrapper<CompressWeightPass, FunctionPass> {
  void runOnFunction() override;

  void generateReport(std::vector<struct CompressInfo> &compressInfos);
};
} // anonymous namespace

void CompressWeightPass::generateReport(
    std::vector<struct CompressInfo> &compressInfos) {

  // Create a map file if compressed weight existed.
  if (clCompressedWeightMapFileName == "-" || !compressInfos.size())
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
  for (auto info : compressInfos) {
    totalSize += info.size;
    totalCompressedSize += info.compressedSize;
    outputOS << info.name << ", " << info.size << ", " << info.compressedSize
             << ", " << info.size - info.compressedSize << ", "
             << int(info.compressedSize * 1.0 / info.size * 100.0) << "%\n";
  }

  outputOS << "totalSize " << totalSize << ", totalCompressedSize "
           << totalCompressedSize << ", reduced "
           << totalSize - totalCompressedSize << ", ratio "
           << int(totalCompressedSize * 1.0 / totalSize * 100.0) << "%\n";
  outputFile->keep();
}

void CompressWeightPass::runOnFunction() {
  std::vector<struct CompressInfo> compressInfos;
  MInfo::getChipInfo(getFunction());

  // Remove compressed weight attributes first.
  getFunction().walk([&](Operation *op) {
    llvm::TypeSwitch<Operation *>(op)
        .Case<tpu::TG_INT8_Conv2DOp, tpu::TG_BF16_Conv2DOp,
              tpu::TL_LG_INT8_Conv2DOp, tpu::TL_LG_BF16_Conv2DOp,
              tpu::TL_LG_LoadCoeffOp, tpu::LoadWeightOp>(
            [&](auto tpuOp) { tpuOp->removeAttr("compressed_weight"); })
        .Case<tpu::TG_INT8_FullyConnectedOp, tpu::TG_BF16_FullyConnectedOp>(
            [&](auto tpuOp) {
              tpuOp->setAttr("do_compress",
                             Builder(op->getContext()).getBoolAttr(true));
            });
  });

  // Compress convolution weight
  OwningRewritePatternList patterns;
  patterns.insert<
      TgConvCompressedWeightPattern<tpu::TG_INT8_Conv2DOp, int8_t>,
      TgConvCompressedWeightPattern<tpu::TG_BF16_Conv2DOp, uint16_t>,
      TlLgConvCompressedWightPattern<tpu::TL_LG_INT8_Conv2DOp, int8_t>,
      TlLgConvCompressedWightPattern<tpu::TL_LG_BF16_Conv2DOp, uint16_t>>(
      &getContext(), compressInfos);
  applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));

  // Remove offset in load weight first.
  // Then run assign-weight-address pass to generate the compressed weight.
  getFunction().walk([&](Operation *op) {
    if (auto loadWeightOp = dyn_cast<tpu::LoadWeightOp>(op)) {
      loadWeightOp->removeAttr("offset");
    }
  });

  generateReport(compressInfos);
}

std::unique_ptr<mlir::Pass> mlir::createCompressWeightPass() {
  return std::make_unique<CompressWeightPass>();
}
