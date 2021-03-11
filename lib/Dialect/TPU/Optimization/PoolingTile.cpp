//===- PoolingTile.cpp - Implementation of Pooling tiling -----------------===//
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
// This file implements the tiling of pooling.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/MachineInfo.h"
#include "tpuc/Passes.h"
#include "tpuc/Support/TensorFile.h"
#include "tpuc/TPUOperationSupport.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "pooling-tile"

namespace mlir {
namespace tpu {

class PoolingBaseModel {
public:
  PoolingBaseModel(Operation *op, const MInfo mInfo) : op(op), mInfo(mInfo) {}

  struct TileInfo {
    int oc_step;
    int oh_step;
    int ow_step;
    int ih_step;
    int iw_step;
  };

  TileInfo getTileSize();

  int n = {1};
  int c = {1};
  int h = {1};
  int w = {1};
  int oh = {1};
  int ow = {1};
  int pad_t = {0};
  int pad_b = {0};
  int pad_l = {0};
  int pad_r = {0};
  int kh = {1};
  int kw = {1};
  int stride_h = {1};
  int stride_w = {1};

  Operation *op = {nullptr};
  const MInfo &mInfo;
};

PoolingBaseModel::TileInfo PoolingBaseModel::getTileSize() {
  TileInfo tileInfo = {};
  int32_t step_c, step_oh, step_ow;

  // determin the shape of tile.
  for (step_ow = stride_w > 15 ? 1 : ow; step_ow > 0; step_ow--) {
    for (step_oh = stride_h > 15 ? 1 : oh; step_oh > 0; step_oh--) {
      for (step_c = std::min(c, mInfo.MAX_TIU_CHANNEL); step_c > 0;
           step_c -= mInfo.lane_num) {
        if (step_c != c) {
          step_c = llvm::alignTo(step_c, mInfo.lane_num);
        }
        auto step_ih = (step_oh - 1) * stride_h + kh;
        auto step_iw = (step_ow - 1) * stride_w + kw;
        if (step_ih > h) {
          step_ih = h;
        }
        if (step_iw > w) {
          step_iw = w;
        }

        uint64_t inputSize = mInfo.getSizePerLane(1, step_c, step_ih, step_iw,
                                                  /*eu_align=*/true);
        uint64_t outputSize = mInfo.getSizePerLane(1, step_c, step_oh, step_ow,
                                                   /*eu_align=*/true);
        auto totalSize = 2 * (inputSize + outputSize);
        if (totalSize <= mInfo.lmem_per_lane) {
          tileInfo.oc_step = step_c;
          tileInfo.oh_step = step_oh;
          tileInfo.ow_step = step_ow;
          tileInfo.ih_step = step_ih;
          tileInfo.iw_step = step_iw;
        }
      }
    }

    return tileInfo;
  }

  llvm_unreachable("Pooling expect valid tile");
}

template <typename OpTy>
class PoolingModel : public PoolingBaseModel {
public:
  PoolingModel(OpTy tpuOp, const MInfo mInfo)
      : PoolingBaseModel{tpuOp.getOperation(), mInfo} {
    bool is_global, do_relu, count_include_pad;
    int pad_value;
    parsePoolParam(tpuOp.param(), tpuOp.input(), tpuOp.output(), n, c, h, w, oh,
                   ow, kh, kw, stride_h, stride_w, pad_t, pad_b, pad_l, pad_r,
                   pad_value, is_global, do_relu, count_include_pad);

    c *= n; // fuse n and c
    n = 1;
  }
};

template <typename OpTy>
class convertPoolingTilePattern : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  convertPoolingTilePattern(MLIRContext *ctx, MInfo &mInfo)
      : OpRewritePattern<OpTy>(ctx), mInfo(mInfo) {}

  LogicalResult matchAndRewrite(OpTy tpuOp,
                                PatternRewriter &rewriter) const override {
    // Already configured
    if (tpuOp.tile_param().hasValue())
      return failure();

    LLVM_DEBUG(llvm::dbgs() << "convertPoolingTilePattern: layer ID "
                            << getOpLayerId(tpuOp.getOperation()) << ", "
                            << tpuOp.name() << "\n");

    auto poolModel(std::make_unique<PoolingModel<OpTy>>(tpuOp, mInfo));

    PoolingBaseModel::TileInfo tileInfo = poolModel->getTileSize();

    tpuOp->setAttr(
        "tile_param",
        tpu::PoolTileParam::get(rewriter.getI32IntegerAttr(tileInfo.oc_step),
                                rewriter.getI32IntegerAttr(tileInfo.oh_step),
                                rewriter.getI32IntegerAttr(tileInfo.ow_step),
                                rewriter.getI32IntegerAttr(tileInfo.ih_step),
                                rewriter.getI32IntegerAttr(tileInfo.iw_step),
                                rewriter.getContext()));
    return success();
  }

  MInfo &mInfo;
};

struct PoolingTilePass
    : public mlir::PassWrapper<PoolingTilePass, FunctionPass> {
  void runOnFunction() override;
};

void PoolingTilePass::runOnFunction() {
  MInfo mInfo;
  mInfo.getChipInfo(getFunction());
  assert(MInfo::version && "refer to set-chip");

  getFunction().walk([&](Operation *op) {
    if (auto tpuOp = dyn_cast<tpu::TG_INT8_PoolMax2DOp>(op)) {
      tpuOp->removeAttr("tile_param");
    }
  });

  OwningRewritePatternList patterns;
  patterns.insert<convertPoolingTilePattern<tpu::TG_INT8_PoolMax2DOp>>(
      &getContext(), mInfo);
  applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
}

void PopulatePoolingTilePatterns(MLIRContext *context,
                                 OwningRewritePatternList *patterns,
                                 MInfo &mInfo) {
  patterns->insert<convertPoolingTilePattern<tpu::TG_INT8_PoolMax2DOp>>(context,
                                                                        mInfo);
}

std::unique_ptr<mlir::Pass> createPoolingTilePass() {
  return std::make_unique<tpu::PoolingTilePass>();
}

} // namespace tpu
} // namespace mlir
