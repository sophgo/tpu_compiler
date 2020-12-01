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

#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Passes.h"
#include "tpuc/TPUCompressUtil.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/MachineInfo.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "tpuc/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"

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
class CompressConvolutionWeightPattern : public OpRewritePattern<TensorTyOp> {
public:
  using OpRewritePattern<TensorTyOp>::OpRewritePattern;

  CompressConvolutionWeightPattern(MLIRContext *ctx,
                                   std::vector<CompressInfo> &compressInfos)
      : OpRewritePattern<TensorTyOp>(ctx), compressInfos_(compressInfos) {}

  LogicalResult matchAndRewrite(TensorTyOp convOp,
                                     PatternRewriter &rewriter) const override {

    // Already compressed.
    if (convOp.compressed_weight().hasValue())
      return failure();

    auto op = convOp.getOperation();
    // for layer group, several conv may refer to one load coeff op
    // no need to compress every time.
    if (auto load_op = dyn_cast<tpu::TL_LG_LoadCoeffOp>
                                  (convOp.filter().getDefiningOp())) {
      if (load_op.compressed_weight().hasValue() &&
          load_op.compressed_weight().getValue()) {
        convOp.setAttr("compressed_weight", rewriter.getBoolAttr(true));
        return failure();
      }
    }

    if (auto load_op = dyn_cast<tpu::LoadWeightOp>
                                  (convOp.filter().getDefiningOp())) {
      if (load_op.compressed()) {
        convOp.setAttr("compressed_weight", rewriter.getBoolAttr(true));
        return failure();
      }
    }

    // Not support group convolution and depthwise convolution
    if (convOp.param().group().getInt() > 1) {
      LLVM_DEBUG(llvm::dbgs()
          << "CompressWeight: layer ID " << getOpLayerId(op)
          << ", " << convOp.name()
          << ", groups " << convOp.param().group().getInt()
          << ", not support group convolution\n");
      return failure();
    }

    auto filterTy =
        convOp.filter().getType().template dyn_cast<RankedTensorType>();
    auto fltElemType = filterTy.getElementType();
    assert(fltElemType.isIntOrFloat() && "Expect result is int or float");
    if (!fltElemType.isIntOrFloat())
      return failure();

    uint64_t fltEltSize = llvm::divideCeil(fltElemType.getIntOrFloatBitWidth(),
                                           8);

    if (fltElemType.isBF16())
      assert(fltEltSize == sizeof(DataType) &&
             "Expect data type size of bf16 is 2");

    // Only support int8 or bf16
    if (fltEltSize != 1 && !fltElemType.isBF16()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "CompressWeight: layer ID " << getOpLayerId(op)
                 << ", " << convOp.name()
                 << ", only support int8 or bf16\n");
      return failure();
    }

    // Same filter shape but fill compressed data
    TensorFile *wTF = getWeightTensorFile(convOp.getOperation());
    auto filter = readAndDeleteWeightTensor<DataType>(convOp.filter(), wTF);
    int64_t filterSize;
    std::vector<int64_t> filterShape;
    getTensorShapeAndSize(convOp.filter(), filterShape, filterSize);
    assert(filterSize == (int64_t)filter->size() &&
           "filter size should be equal");

    LLVM_DEBUG(llvm::dbgs()
        << "CompressWeight: layer ID " << getOpLayerId(op)
        << ", " << convOp.name() << "\n"
        << "  filter(" << filterShape[0]
        << ", " << filterShape[1]
        << ", " << filterShape[2]
        << ", " << filterShape[3]
        << "), fltEltSize " << fltEltSize
        << "\n");

    auto newFilter = std::make_unique<std::vector<DataType> >(filterSize);
    std::memset(newFilter->data(), 0, filterSize * sizeof(DataType));

    // Filter layout in dialact (oc, ic, kh, kw)
    // But the storage layout is already changed to (1, oc, kh*kw, ic)
    int oc = filterShape[0];
    int ic = filterShape[1];
    int kh = filterShape[2];
    int kw = filterShape[3];

    bool canCompress = true;
    // Split output channel in unit of lane number for df only
    int oc_step = MInfo::lane_num;
    // not split for lg
    if ((TensorTyOp::getOperationName() == "tpu.tl_lg_int8_conv_2d") ||
        (TensorTyOp::getOperationName() == "tpu.tl_lg_bf16_conv_2d"))
      oc_step = oc;

    auto buffer =
        std::make_unique<std::vector<uint8_t> >(oc_step * kh * kw * ic *
                                                sizeof(DataType));

    int totalSize = 0;
    int totalCompressedSize = 0;
    for (int oc_pos = 0; oc_pos < oc; oc_pos += oc_step) {
      int cur_oc = std::min(oc - oc_pos, oc_step);
      int step_size = cur_oc * kh * kw * ic * sizeof(DataType);
      int pos = oc_pos * kh * kw * ic * sizeof(DataType);

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

      auto plainData = std::make_unique<std::vector<uint8_t> >(step_size);
      std::memcpy(plainData->data(), filter->data() + pos, step_size);

      // Calculate compress parameter first.
      CompressCommandInfo cmdInfo;
      std::memset(&cmdInfo, 0, sizeof(cmdInfo));
      cmdInfo.signedness = fltElemType.isBF16() ? 0 : 1;
      cmdInfo.is_bfloat16 = fltElemType.isBF16() ? 1 : 0;
      cmdInfo.bias0 = fltElemType.isBF16() ? 127 : 0;
      getCompressParameter(plainData->data(), step_size, cmdInfo.signedness,
                           cmdInfo.is_bfloat16, &cmdInfo);

      // Create Compress data.
      int requiredSize = getCompressedDataSize(step_size,
                                               fltElemType.isBF16() ? 1 : 0);
      auto compressedData =
          std::make_unique<std::vector<uint8_t> >(requiredSize);
      int compressedSize = 0;

      if (fltElemType.isBF16())
        compressBf16Data(plainData->data(), step_size, compressedData->data(),
                         &compressedSize, &cmdInfo);
      else
        compressInt8Data(plainData->data(), step_size, compressedData->data(),
                         &compressedSize, &cmdInfo);

      // Compress size must be less than tiled size.
      LLVM_DEBUG(llvm::dbgs()
          << "  [oc_pos=" << oc_pos << "] cur_oc " << cur_oc
          << ", step_size " << step_size
          << ", compressedSize " << compressedSize << "\n");

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

    if (canCompress) {
      addWeightTensorAndUpdateWeightOp<DataType>(convOp.filter(),
          "z", *newFilter, filterShape,
          fltElemType.isBF16() ? "BF16" : "INT8", wTF);

      assert(memcmp(filter->data(), newFilter->data(), filterSize)
             && "Expect compressed content");

      convOp.setAttr("tiled_oc_step", rewriter.getI32IntegerAttr(oc_step));
      convOp.setAttr("compressed_weight", rewriter.getBoolAttr(true));

      // set compressed flag on TL_LoadCoeffOp for layer group
      if (auto load_op = dyn_cast<tpu::TL_LG_LoadCoeffOp>
                                  (convOp.filter().getDefiningOp())) {
        load_op.setAttr("compressed_weight", rewriter.getBoolAttr(true));
      }

      if (auto load_op = dyn_cast<tpu::LoadWeightOp>
                                  (convOp.filter().getDefiningOp())) {
        load_op.setAttr("compressed", rewriter.getBoolAttr(true));
      }

      struct CompressInfo info;
      info.name = convOp.name();
      info.size = totalSize;
      info.compressedSize = totalCompressedSize;
      compressInfos_.push_back(info);

      LLVM_DEBUG(llvm::dbgs()
          << "  compressInfos size "
          << compressInfos_.size() << "\n");

      return success();
    } else  {
      addWeightTensorAndUpdateWeightOp<DataType>(convOp.filter(),
          "", *filter, filterShape,
          fltElemType.isBF16() ? "BF16" : "INT8", wTF);

      return failure();
    }
  }

  std::vector<struct CompressInfo> &compressInfos_;
};

template<typename T>
static void stridedMatrixMemcpy(T *dstPtr, T *srcPtr, int srcStride,
                                int H, int W) {
  for (int i = 0; i < H; ++i) {
    for (int j = 0; j < W; ++j) {
      dstPtr[i * W + j] = srcPtr[i * srcStride + j];;
    }
  }
}

template <typename TensorTyOp, typename DataType>
class CompressFcWeightPattern
    : public OpRewritePattern<TensorTyOp> {
public:
  using OpRewritePattern<TensorTyOp>::OpRewritePattern;

  CompressFcWeightPattern(MLIRContext *ctx,
                          std::vector<CompressInfo> &compressInfos)
      : OpRewritePattern<TensorTyOp>(ctx), compressInfos_(compressInfos) {}

  LogicalResult matchAndRewrite(TensorTyOp fcOp,
                                     PatternRewriter &rewriter) const override {

    // Step size must generated first
    if (!fcOp.tile_param().hasValue())
      return failure();

    if (fcOp.compressed_weight().hasValue())
      return failure();

    std::vector<int32_t> tileValues;
    arrayAttrToVector(fcOp.tile_param().getValue().tile_step(), tileValues);

    std::vector<int> n_poss;
    std::vector<int> k_poss;
    std::vector<int> n_sizes;
    std::vector<int> k_sizes;
    arrayAttrToVector(fcOp.tile_param().getValue().n_poss(), n_poss);
    arrayAttrToVector(fcOp.tile_param().getValue().k_poss(), k_poss);
    arrayAttrToVector(fcOp.tile_param().getValue().n_sizes(), n_sizes);
    arrayAttrToVector(fcOp.tile_param().getValue().k_sizes(), k_sizes);

    assert((n_poss.size() == k_poss.size()) &&
           (k_poss.size() == n_sizes.size()) &&
           (n_sizes.size() == k_sizes.size()));

    auto filterTy =
        fcOp.filter().getType().template dyn_cast<RankedTensorType>();
    auto fltElemType = filterTy.getElementType();

    // Same filter shape but fill compressed data
    TensorFile *wTF = getWeightTensorFile(fcOp.getOperation());
    auto filter = readAndDeleteWeightTensor<DataType>(fcOp.filter(), wTF);
    int64_t filterSize;
    std::vector<int64_t> filterShape;
    getTensorShapeAndSize(fcOp.filter(), filterShape, filterSize);

    // In dialect, weight shape is NxK !
    // weight already transposed (NxK -> KxN), but shape not updated !
    int N = filterShape[0];
    int K = filterShape[1];

    LLVM_DEBUG(llvm::dbgs()
        << "CompressFcWeightPattern: layer ID "
        << mlir::getOpLayerId(fcOp.getOperation())
        << ", " << fcOp.name() << "\n  "
        << "weight shape (K=" << K
        << ", N=" << N << ")\n  "
        << "tileM " << tileValues[0]
        << ", tileK " << tileValues[1]
        << ", tileN " << tileValues[2]
        << "\n");

    auto newFilter = std::make_unique<std::vector<DataType>>(filterSize);
    std::memset(newFilter->data(), 0, filterSize * sizeof(DataType));

    //
    // Weight (K, N)
    //
    // -------------
    // | A00 | A01 |              A00
    // ------|------ K    =>      A10
    // | A10 | A11 |              A01
    // -------------              A11
    //       N

    bool canCompress = true;
    int dstOffset = 0;
    std::vector<int> compr_weight_poss;
    std::vector<int> compr_weight_sizes;
    for (unsigned i = 0; i < n_poss.size(); ++i) {
      int n_pos = n_poss[i];
      int k_pos = k_poss[i];
      int n_size = n_sizes[i];
      int k_size = k_sizes[i];
      int srcOffset = k_pos * N + n_pos;

      assert((n_pos < N) && (n_size <= N) && "Expect valid n pos, size");
      assert((k_pos < K) && (k_size <= K) && "Expect valid k pos, size");

      int step_size = n_size * k_size * sizeof(DataType);
      auto plainData = std::make_unique<std::vector<uint8_t> >(step_size);
      stridedMatrixMemcpy<DataType>((DataType *)plainData->data(),
                                    filter->data() + srcOffset,
                                    N, k_size, n_size);

      // Calculate compress parameter first.
      CompressCommandInfo cmdInfo;
      std::memset(&cmdInfo, 0, sizeof(cmdInfo));
      cmdInfo.signedness = fltElemType.isBF16() ? 0 : 1;
      cmdInfo.is_bfloat16 = fltElemType.isBF16() ? 1 : 0;
      cmdInfo.bias0 = fltElemType.isBF16() ? 127 : 0;
      getCompressParameter(plainData->data(), step_size, cmdInfo.signedness,
                           cmdInfo.is_bfloat16, &cmdInfo);

      // Create Compress data.
      int requiredSize = getCompressedDataSize(step_size,
                                               fltElemType.isBF16() ? 1 : 0);
      auto compressedData =
          std::make_unique<std::vector<uint8_t> >(requiredSize);
      int compressedSize = 0;

      if (fltElemType.isBF16())
        compressBf16Data(plainData->data(), step_size, compressedData->data(),
                         &compressedSize, &cmdInfo);
      else
        compressInt8Data(plainData->data(), step_size, compressedData->data(),
                         &compressedSize, &cmdInfo);

      if ((dstOffset + compressedSize) > (K * N * (int)sizeof(DataType))) {
        LLVM_DEBUG(llvm::dbgs()
            << "      compressed size exceed, dstOffset " << dstOffset
            << " + " << compressedSize
            << " > " << (K * N)
            << "\n");
        canCompress = false;
        break;
      }

      // Fill compressed data.
      std::memcpy(newFilter->data() + dstOffset / sizeof(DataType),
                  compressedData->data(), compressedSize);

      compr_weight_poss.push_back(dstOffset);
      compr_weight_sizes.push_back(compressedSize);

      dstOffset += compressedSize;
    }

    if (canCompress) {
      addWeightTensorAndUpdateWeightOp<DataType>(fcOp.filter(),
          "z", *newFilter, filterShape,
          fltElemType.isBF16() ? "BF16" : "INT8", wTF);

      fcOp.setAttr("compressed_weight", rewriter.getBoolAttr(true));
      fcOp.setAttr("compr_weight_poss",
                  rewriter.getI32ArrayAttr(compr_weight_poss));
      fcOp.setAttr("compr_weight_sizes",
                  rewriter.getI32ArrayAttr(compr_weight_sizes));

      struct CompressInfo info;
      info.name = fcOp.name();
      info.size = K * N * sizeof(DataType);
      info.compressedSize = dstOffset;
      compressInfos_.push_back(info);

      LLVM_DEBUG(llvm::dbgs()
          << "  compressInfos entries " << compressInfos_.size() << "\n");

      return success();
    } else {
      addWeightTensorAndUpdateWeightOp<DataType>(fcOp.filter(),
          "", *filter, filterShape,
          fltElemType.isBF16() ? "BF16" : "INT8", wTF);

      return failure();
    }
  }

  std::vector<struct CompressInfo> &compressInfos_;
};

struct CompressWeightPass : public mlir::PassWrapper<CompressWeightPass, FunctionPass> {
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
  std::vector<struct CompressInfo> compressInfos;
  MInfo::getChipInfo(getFunction());

  // Compress convolution weight
  OwningRewritePatternList patterns;
  patterns.insert<
      CompressConvolutionWeightPattern<tpu::TL_LW_Conv2DOp, int8_t>,
      CompressConvolutionWeightPattern<tpu::TL_LG_INT8_Conv2DOp, int8_t>,
      CompressConvolutionWeightPattern<tpu::TL_LG_BF16_Conv2DOp, uint16_t>
      >(&getContext(), compressInfos);
  applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));

  patterns.clear();
  patterns.insert<
      CompressFcWeightPattern<tpu::TG_INT8_FullyConnectedOp, int8_t>,
      CompressFcWeightPattern<tpu::TG_BF16_FullyConnectedOp, uint16_t>
      >(&getContext(), compressInfos);
  applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));

  // Remove offset in load weight first.
  // Then run assign-weight-address pass to generate the compressed weight.
  getFunction().walk([&](Operation *op) {
    if (auto loadWeightOp = dyn_cast<tpu::LoadWeightOp>(op)) {
      loadWeightOp.removeAttr("offset");
    }
  });

  generateReport(compressInfos);
}

std::unique_ptr<mlir::Pass> mlir::createCompressWeightPass() {
  return std::make_unique<CompressWeightPass>();
}
