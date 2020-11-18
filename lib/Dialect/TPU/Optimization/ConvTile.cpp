//===- ConvTiling.cpp - Implementation of convolution tiling --------------===//
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
// This file implements the convolution tiling.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/Dialect/TPU/MachineInfo.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TensorFile.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/ADT/StringRef.h"

#define DEBUG_TYPE "conv-tile"

namespace mlir {
namespace tpu {

class ConvolutionBaseModel {
public:
  ConvolutionBaseModel(Operation *op, const MInfo mInfo)
      : op(op), mInfo(mInfo) {
    dataTypeSize = getDataTypeSize(op->getResult(0));
  }

  struct TileInfo {
    int n_step;
    int oc_step;
    int ic_step;
    int oh_step;
    int ow_step;
    int ih_step;
    int iw_step;
    bool use_double_buffer;
    bool favor_dma;
  };

  int getLmSizePerLane(TileInfo &tileInfo);
  TileInfo getTileSizes(bool use_double_buffer, bool favor_dma);
  bool checkDmaPolicy(TileInfo &tileInfo);
  bool isNoTile(TileInfo &tileInfo);

  int input_n = {0};
  int input_c = {0};
  int input_h = {0};
  int input_w = {0};
  int groups = {1};
  int output_c = {0};
  int output_h = {0};
  int output_w = {0};
  int kh = {1};
  int kw = {1};
  int dilation_h = {1};
  int dilation_w = {1};
  int pad_top = {0};
  int pad_bottom = {0};
  int pad_left = {0};
  int pad_right = {0};
  int pad_value = {0};
  int insert_h = {0};
  int insert_w = {0};
  int stride_h = {1};
  int stride_w = {1};
  bool with_bias = {false};
  bool do_relu = {false};
  bool do_chl_quan = {false};
  bool is_dw = {false};
  bool do_ic_alignment = {false};
  bool do_leaky_relu = {false};

  Operation *op = {nullptr};
  const MInfo &mInfo;
  int dataTypeSize = {1};
};

bool ConvolutionBaseModel::isNoTile(TileInfo &tileInfo) {
  if (tileInfo.n_step == input_n && tileInfo.oc_step == output_c &&
      tileInfo.oh_step == output_h && tileInfo.ow_step == output_w &&
      tileInfo.ic_step == input_c)
  return true;

  return false;
}

int ConvolutionBaseModel::getLmSizePerLane(TileInfo &tileInfo) {
  int n_step = tileInfo.n_step;
  int oc_step = tileInfo.oc_step;
  int oh_step = tileInfo.oh_step;
  int ow_step = tileInfo.ow_step;
  int ih_step = tileInfo.ih_step;
  int iw_step = tileInfo.iw_step;
  int ic_step = tileInfo.ic_step;
  bool useDoubleBuffer = tileInfo.use_double_buffer;

  assert((n_step > 0) && (oc_step > 0) && (oh_step > 0) && (ow_step > 0) &&
         (ic_step > 0) && (ih_step > 0) && "Expect positive tile values");
  assert((n_step <= input_n) && (oc_step <= output_c) &&
         (oh_step <= output_h) && (ow_step <= output_w) &&
         (ih_step <= input_h) && (iw_step <= input_w) && (ic_step <= input_c) &&
         "Expect valid tile range");

  // Weight shape(1, 1, tiledOc, Kh*Kw, Ic) in local memory, not EU aligned
  int ic_step_4_weight = is_dw ? 1 : ic_step;
  ic_step_4_weight = do_ic_alignment ?
                     llvm::alignTo(ic_step, 2) : ic_step_4_weight;

  uint64_t weightSize = mInfo.getSizePerLane(1, oc_step, kh * kw * dataTypeSize,
                                             ic_step_4_weight,
                                             /*eu_align=*/false);

  // Input shape (n, ic, ih, iw) in local memory, EU aligned
  uint64_t inputSize = mInfo.getSizePerLane(n_step, ic_step, ih_step,
                                            iw_step * dataTypeSize,
                                            /*eu_align=*/true);

  // Output shape (n, oc, oh, ow) in local memory, EU aligned
  uint64_t outputSize = mInfo.getSizePerLane(n_step, oc_step, oh_step,
                                            ow_step * dataTypeSize,
                                             /*eu_align=*/true);

  // Bias shape in local memory, not EU aligned
  //   Per-channel: (1, 1, tiledOc, 1, [9/5])
  //   Per-tensor:  (2, 1, tiledOc, 1, 1)
  uint64_t biasSize = 0;
  if (do_chl_quan) {
    int unitSize = with_bias ? 9 : 5;
    biasSize = mInfo.getSizePerLane(1, oc_step, 1, unitSize,
                                    /*eu_align=*/false);
  } else if (with_bias) {
    biasSize = mInfo.getSizePerLane(2, oc_step * dataTypeSize, 1, 1,
                                    /*eu_align=*/false);
  }

  // Leaky relu needs tl_neg, tl_relu.
  // TIU intermediate buffer, not need double buffer.
  uint64_t extraSize = 0;
  if (do_leaky_relu)
    extraSize = 2 * outputSize;

  uint64_t bufferMultiplier = useDoubleBuffer ? 2 : 1;
  uint64_t totalSize = (inputSize + outputSize + weightSize + biasSize) *
                       bufferMultiplier + extraSize;

  if (totalSize <= mInfo.lmem_per_lane) {
    LLVM_DEBUG(llvm::dbgs()
        << "  ConvolutionBaseModel::getLmSizePerLane\n    "
        << "Tile (n_step=" << n_step
        << ", oc_step=" << oc_step
        << ", oh_step=" << oh_step
        << ", ow_step=" << ow_step
        << ", ih_step=" << ih_step
        << ", iw_step=" << iw_step
        << ", ic_step=" << ic_step << ")\n    "
        << "inputSize " << inputSize
        << ", outputSize " << outputSize
        << ", weightSize " << weightSize
        << ", biasSize " << biasSize
        << "(do_chl_quan " << do_chl_quan
        << ", with_bias " << with_bias << ")"
        << ", totalSize " << totalSize << "\n    "
        << "tiled input shape (" << n_step
        << ", " << ic_step
        << ", " << ih_step
        << ", " << iw_step << ")\n    "
        << "tiled weight shape (" << oc_step
        << ", " << ic_step_4_weight
        << ", " << kh
        << ", " << kw << ")\n    "
        << "tiled output shape (" << n_step
        << ", " << oc_step
        << ", " << oh_step
        << ", " << ow_step << ")\n    "
        << "use_double_buffer " << tileInfo.use_double_buffer
        << ", favor_dma " << tileInfo.favor_dma << "\n");
  }

  return totalSize;
}

// I try to maximize the local memory utilization,
// but it causes large write latency, especially in cross-layer.
// However TDMA engine can handle small data transfer efficiently.
//
// E.g. Resnet50 scale2b_branch2c in DDR3 platform.
//   (1, 96, 56, 56) tiu 19471, store 31056, 77 fps
//   (1, 32, 56, 56) tiu 6535, store 10376, 84 fps
//
// The load/store reorder may be useful in intra-layer and
// inter-layer.
//
// The next-generation chip will do DMA store once intermediate
// result is generated.
//
// The following is temporary solution.
// I decrease the output channel size to trigger frequent DMA store.
// So local memory is wasted.
bool ConvolutionBaseModel::checkDmaPolicy(TileInfo &tileInfo) {
  if (!tileInfo.favor_dma)
    return true;

  // DMA efficiency: OH * OW >= 256B
  const int dma_min_size = 256;
  int ofmap_plane_size = tileInfo.oh_step * tileInfo.ow_step;

  if ((tileInfo.oc_step > (int)mInfo.lane_num) &&
      (ofmap_plane_size > (1 * dma_min_size))) {
    return false;
  }
  if ((tileInfo.oc_step > (2 * (int)mInfo.lane_num)) &&
      (ofmap_plane_size < dma_min_size)) {
    // even oh*ow is smaller, use at most 2xlanes_num
    return false;
  }

  return true;
}

ConvolutionBaseModel::TileInfo ConvolutionBaseModel::getTileSizes(
    bool use_double_buffer, bool favor_dma) {
  ConvolutionBaseModel::TileInfo tileInfo = {0};

  int oc = output_c / groups;
  int ic = input_c / groups;
  if (is_dw) {
    oc = output_c;
    ic = input_c;
  }

  int max_n_step = std::min(input_n, mInfo.MAX_TIU_BATCH);
  int max_oc_step = std::min(oc, mInfo.MAX_TIU_CHANNEL);
  int num_oc_step = llvm::divideCeil(max_oc_step, mInfo.lane_num);
  int max_oh_step = std::min(output_h, mInfo.MAX_TIU_HEIGHT);
  int max_ow_step = std::min(output_w, mInfo.MAX_TIU_WIDTH);
  int max_ic_step = std::min(ic, mInfo.MAX_TIU_CHANNEL);

  int kh_extent = dilation_h * (kh - 1) + 1;
  int kw_extent = dilation_w * (kw - 1) + 1;

  // Split output width
  for (int ow_step = max_ow_step; ow_step > 0; --ow_step) {
    int iw_step = std::min((ow_step - 1) * stride_w + kw_extent, input_w);

    if ((iw_step == input_w) && (stride_w > 1)) {
      // For better DMA transfer efficiency, use whole width.
      //   E.g.
      //     ifmap (1, 512, 28, 28), kernel (1, 1), stride 2
      //
      //     input (27, 27) needed, but (27, 28) is better
      iw_step = std::min(iw_step + stride_w - 1, input_w);
    }

    // Split output height
    for (int oh_step = max_oh_step; oh_step > 0; --oh_step) {
      // When the width tiling is used, there is no need to do height tiling.
      if (ow_step < max_ow_step)
        oh_step = 1;

      int ih_step = std::min((oh_step - 1) * stride_w + kh_extent, input_h);

      // Split output channel
      for (int oc_i = 0; oc_i < num_oc_step; ++oc_i) {
        // Downward, align lanes
        //   E.g. oc = 48, oc_step: 48, 32
        int oc_step = std::min((num_oc_step - oc_i) * (int)mInfo.lane_num, oc);

        for (int n_step = max_n_step; n_step > 0; --n_step) {
            tileInfo = {0};
            tileInfo.n_step = n_step;
            tileInfo.oc_step = oc_step;
            tileInfo.oh_step = oh_step;
            tileInfo.ow_step = ow_step;
            tileInfo.ih_step = ih_step;
            tileInfo.iw_step = iw_step;
            tileInfo.ic_step = is_dw ? oc_step : max_ic_step;
            tileInfo.use_double_buffer = use_double_buffer;
            tileInfo.favor_dma = favor_dma;

            uint64_t needed = (uint64_t)getLmSizePerLane(tileInfo);
            if (needed <= mInfo.lmem_per_lane && checkDmaPolicy(tileInfo))
              return tileInfo;
        }
      }
    }
  }

  tileInfo = {0};
  return tileInfo;
}

template<typename OpTy>
class ConvolutionModel : public ConvolutionBaseModel {
public:
  ConvolutionModel(OpTy tpuOp, const MInfo &mInfo)
      : ConvolutionBaseModel {tpuOp.getOperation(), mInfo} {
    parseConvParam(tpuOp.param(), false, tpuOp.input(), tpuOp.output(),
                   tpuOp.filter(), input_n, input_c, input_h, input_w,
                   output_c, output_h, output_w, groups, kh, kw, stride_h,
                   stride_w, pad_top, pad_bottom, pad_left, pad_right,
                   dilation_h, dilation_w, is_dw, with_bias, do_relu, pad_value);

    do_ic_alignment = tpuOp.do_ic_alignment().hasValue()
                         ? tpuOp.do_ic_alignment().getValue() : false;

    if (llvm::dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(tpuOp.getOperation()))
      do_chl_quan = true;

    do_leaky_relu = tpuOp.do_leaky_relu();
  }
};

template<typename OpTy>
class convertConvTilePattern : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  convertConvTilePattern(MLIRContext *ctx, MInfo &mInfo)
    : OpRewritePattern<OpTy>(ctx), mInfo(mInfo) {}

  PatternMatchResult matchAndRewrite(OpTy tpuOp,
                                     PatternRewriter &rewriter) const override {

    // Already configured
    if (tpuOp.tile_param().hasValue())
      return Pattern::matchFailure();

    LLVM_DEBUG(llvm::dbgs()
        << "convertConvTilePattern: layer ID "
        << getOpLayerId(tpuOp.getOperation())
        << ", " << tpuOp.name() << "\n");

    auto convModel(std::make_unique<ConvolutionModel<OpTy>>(tpuOp, mInfo));

    // Evaluate no-tile, then double buffer
    ConvolutionBaseModel::TileInfo tileInfo = convModel->getTileSizes(false,
                                                                      false);
    if (!convModel->isNoTile(tileInfo))
      tileInfo = convModel->getTileSizes(true, true);

    if (!tileInfo.n_step)
      return Pattern::matchFailure();

    tpuOp.setAttr("tile_param",
        tpu::ConvTileParam::get(
            rewriter.getI32IntegerAttr(tileInfo.n_step),
            rewriter.getI32IntegerAttr(tileInfo.oc_step),
            rewriter.getI32IntegerAttr(tileInfo.oh_step),
            rewriter.getI32IntegerAttr(tileInfo.ow_step),
            rewriter.getI32IntegerAttr(tileInfo.ih_step),
            rewriter.getI32IntegerAttr(tileInfo.iw_step),
            rewriter.getI32IntegerAttr(tileInfo.ic_step),
            rewriter.getBoolAttr(tileInfo.use_double_buffer),
            rewriter.getContext()));
    return Pattern::matchSuccess();
  }

  MInfo &mInfo;
};

struct ConvTilePass : public FunctionPass<ConvTilePass> {
  void runOnFunction() override;
};

void ConvTilePass::runOnFunction() {
  MInfo Machineinfo;
  Machineinfo.getChipInfo(getFunction());
  assert(MInfo::version && "refer to set-chip");

  getFunction().walk([&](Operation *op) {
    if (auto tpuOp = dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(op)) {
      tpuOp.removeAttr("tile_param");
    }
  });

  OwningRewritePatternList patterns;
  patterns.insert<
      convertConvTilePattern<tpu::TG_INT8_PC_Conv2DOp>,
      convertConvTilePattern<tpu::TG_INT8_PT_Conv2DOp>,
      convertConvTilePattern<tpu::TG_BF16_Conv2DOp>
      >(&getContext(), Machineinfo);
  applyPatternsGreedily(getFunction(), patterns);
}

void PopulateConvTilePatterns(
    MLIRContext *context, OwningRewritePatternList *patterns, MInfo &mInfo) {
  patterns->insert<
      convertConvTilePattern<tpu::TG_INT8_PC_Conv2DOp>,
      convertConvTilePattern<tpu::TG_INT8_PT_Conv2DOp>,
      convertConvTilePattern<tpu::TG_BF16_Conv2DOp>
      >(context, mInfo);
}

std::unique_ptr<OpPassBase<FuncOp>> createConvTilePass() {
  return std::make_unique<tpu::ConvTilePass>();
}

} // namespace tpu
} // namespace mlir
