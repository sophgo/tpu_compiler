//===- FullyConnectedTile.cpp - Implementation of FC tiling ---------------===//
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
// This file implements the tiling of matrix multiplication.
//
//===----------------------------------------------------------------------===//

#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/Passes.h"
#include "tpuc/MachineInfo.h"
#include "tpuc/Support/TensorFile.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "fully-connected-tile"

namespace mlir {
namespace tpu {

class FullyConnectedModel {
public:
  FullyConnectedModel(const MInfo &mInfo, int M, int K, int N, bool hasB,
                      int dataTypeSize)
      : mInfo(mInfo), M(M), K(K), N(N), hasB(hasB), dataTypeSize(dataTypeSize) {
  }

  struct TileInfo {
    int m_step;
    int n_step;
    int k_step;
  };

  TileInfo getTileSizes();
  int getLmSizePerLane(int tileM, int tileK, int tileN, bool do_parallel = false);
  void getTilePoss(TileInfo tileInfo, std::vector<int> &n_poss,
                   std::vector<int> &k_poss, std::vector<int> &n_sizes,
                   std::vector<int> &k_sizes);

  const MInfo &mInfo;
  int M = {0};
  int K = {0};
  int N = {0};
  bool hasB = {false};
  int dataTypeSize = {1};
};

// Y(M, N) = L(M, K) * R(K, N) + B(1, N)
int FullyConnectedModel::getLmSizePerLane(int tileM, int tileK, int tileN, bool do_parallel) {
  const int npuNum = mInfo.lane_num;
  const int euNum = mInfo.eu_num / dataTypeSize; // bf16: 1/2 eu num
  int blob_L = 1, blob_R = 1, blob_B = 1, blob_Y = 1;
  if (do_parallel && !(tileM == M && tileK == K && tileN == N)) {
    blob_L = (K != tileK ? 2 : 1);
    blob_R = 2;
    blob_B = (N != tileN) ? 2 : 1;
    blob_Y = (K == tileK ? 2 : (N + tileN - 1) / tileN);
  }
  int tileLSize =
      tileM * euNum * llvm::divideCeil(tileK, euNum * npuNum) * dataTypeSize;

  int tileRSize =
      tileK * euNum * llvm::divideCeil(tileN, euNum * npuNum) * dataTypeSize;

  // int8: 4*int8(32bit), bf16: 2*bf16
  int biasN = (dataTypeSize == 2) ? 2 : 4;

  int tileBSize = hasB
                      ? (biasN * euNum *
                         llvm::divideCeil(tileN, euNum * npuNum) * dataTypeSize)
                      : 0;

  // Partial sum 32bit, 2x(bf16), 4x(int8)
  int tileYSize =
      tileM * euNum * llvm::divideCeil(tileN, euNum * npuNum) * dataTypeSize;
  if (tileK < K)
    tileYSize *= 4 / dataTypeSize;

  int totalSize = blob_L * tileLSize + blob_R * tileRSize + blob_Y * tileYSize +
                  blob_B * tileBSize;

  if (totalSize <= (int)mInfo.lmem_per_lane) {
    LLVM_DEBUG(llvm::dbgs()
               << "    FullyConnectedModel::getLmSizePerLane:\n      "
               << "M " << M << ", K " << K << ", N " << N << "\n      "
               << "tileL(" << tileM << ", " << tileK << ")\n      "
               << "tileR(" << tileK << ", " << tileN << ")\n      "
               << "tileY(" << tileM << ", " << tileN << ")\n      "
               << "tileYSize " << tileYSize << ", tileLSize " << tileLSize
               << ", tileRSize " << tileRSize << ", tileBSize " << tileBSize
               << "\n");
  }

  return totalSize;
}

FullyConnectedModel::TileInfo FullyConnectedModel::getTileSizes() {
  int max_tiu = (4095 - 32);
  int maxM = std::min(M, max_tiu); // TIU 12bit
  int maxK = std::min(K, max_tiu); // TIU 12bit
  int maxN = std::min(N, max_tiu);
  // 1/2 EU in bf16
  int totalEuNum = mInfo.lane_num * (mInfo.eu_num / dataTypeSize);
  bool do_parallel = (maxM == M);
  // try parallel first
  for (int tileM = maxM; tileM > 0;) {
    for (int tileK = maxK; tileK > 0; tileK--) {
      for (int tileN = maxN; tileN > 0;) {
        int needed = getLmSizePerLane(tileM, tileK, tileN, do_parallel);
        if (needed <= (int)mInfo.lmem_per_lane) {
          return {tileM, tileN, tileK};
        }
        if (tileN % totalEuNum) {
          tileN -= (tileN % totalEuNum);
        } else {
          tileN -= totalEuNum;
        }
      }
    }
    if (do_parallel) {
      do_parallel = false;
    } else {
      tileM--;
    }
  }
  llvm_unreachable("FC tiling failed");
  return {0, 0, 0};
}

void FullyConnectedModel::getTilePoss(TileInfo tileInfo,
                                      std::vector<int> &n_poss,
                                      std::vector<int> &k_poss,
                                      std::vector<int> &n_sizes,
                                      std::vector<int> &k_sizes) {
  int k_step = tileInfo.k_step;
  int n_step = tileInfo.n_step;

  // Each tiled_R(weight) is only loaded once.
  // tiled_L(input) reload is reload once tiled_weight moves right.
  //
  // for each tiled N
  for (int n_pos = 0; n_pos < N; n_pos += n_step) {
    int n_size = std::min(n_step, N - n_pos);

    // for each tiled K
    for (int k_pos = 0; k_pos < K; k_pos += k_step) {
      // Y(M, N) = L(M, K) * R(K, N) + B(1, N)
      // tiled_Y(M, tiled_K) = tiled_L(M, tiled_K) * tiled_R(tiled_K, tiled_N) +
      //                       tiled_B(1, tiled_N)
      //
      // L = [L0, L1, ... Lk-1]
      // R = [R0,0,   R0,1,   ..., R0,n-1
      //      R1,0,
      //
      //      Rk-1,0, Rk-1,1, ..., Rk-1,n-1]
      // B = [B0, B1, ... Bn-1]
      //
      // tiled_y,i += L0 * R0,i + L1 * R1,i + ... + Ln-1 * Rk-1,i + Bi
      int k_size = std::min(k_step, K - k_pos);

      n_poss.push_back(n_pos);
      k_poss.push_back(k_pos);
      n_sizes.push_back(n_size);
      k_sizes.push_back(k_size);
    }
  }
}

template <typename OpTy>
class convertFullyConnectedTilePattern : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  convertFullyConnectedTilePattern(MLIRContext *ctx, MInfo &mInfo)
      : OpRewritePattern<OpTy>(ctx), mInfo(mInfo) {}

  LogicalResult matchAndRewrite(OpTy tpuOp,
                                     PatternRewriter &rewriter) const override {

    // Already configured
    if (tpuOp.tile_param().hasValue())
      return failure();

    auto op = tpuOp.getOperation();

    LLVM_DEBUG(llvm::dbgs()
               << "convertFullyConnectedTilePattern: layer ID "
               << mlir::getOpLayerId(op) << ", " << tpuOp.name() << "\n");

    int batch, m, k, n;
    parseFullyConnectedParam(tpuOp.input(), tpuOp.filter(), tpuOp.output(),
                             batch, m, k, n);
    if (batch > 1) {
      return failure();
    }

    bool hasBias = isTensorNone(tpuOp.bias()) ? false : true;

    int dataTypeSize = getDataTypeSize(op->getResult(0));
    auto fcModel(std::make_unique<FullyConnectedModel>(mInfo, m, k, n,
                                                       hasBias, dataTypeSize));

    FullyConnectedModel::TileInfo tileInfo = fcModel->getTileSizes();
    if (!tileInfo.m_step || !tileInfo.k_step || !tileInfo.n_step)
      return failure();

    SmallVector<int32_t, 4> tileValues = {tileInfo.m_step, tileInfo.k_step,
                                          tileInfo.n_step};

    std::vector<int> n_poss;
    std::vector<int> k_poss;
    std::vector<int> n_sizes;
    std::vector<int> k_sizes;
    fcModel->getTilePoss(tileInfo, n_poss, k_poss, n_sizes, k_sizes);
    tpuOp->setAttr("tile_param",
                  tpu::FcTileParam::get(rewriter.getI32ArrayAttr(tileValues),
                                        rewriter.getI32ArrayAttr(n_poss),
                                        rewriter.getI32ArrayAttr(k_poss),
                                        rewriter.getI32ArrayAttr(n_sizes),
                                        rewriter.getI32ArrayAttr(k_sizes),
                                        rewriter.getContext()));

    return success();
  }

  MInfo &mInfo;
};

struct FullyConnectedTilePass : public mlir::PassWrapper<FullyConnectedTilePass, FunctionPass> {
  void runOnFunction() override;
};

void FullyConnectedTilePass::runOnFunction() {
  MInfo machineInfo;
  machineInfo.getChipInfo(getFunction());
  assert(MInfo::version && "refer to set-chip");

  getFunction().walk([&](Operation *op) {
    if (auto tpuOp = dyn_cast<tpu::TG_INT8_FullyConnectedOp>(op)) {
      tpuOp->removeAttr("tile_step");
    } else if (auto tpuOp = dyn_cast<tpu::TG_BF16_FullyConnectedOp>(op)) {
      tpuOp->removeAttr("tile_step");
    }
  });

  OwningRewritePatternList patterns;
  patterns
      .insert<convertFullyConnectedTilePattern<tpu::TG_INT8_FullyConnectedOp>,
              convertFullyConnectedTilePattern<tpu::TG_BF16_FullyConnectedOp>>(
          &getContext(), machineInfo);
  applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
}

void PopulateFullyConnectedTilePatterns(MLIRContext *context,
                                        OwningRewritePatternList *patterns,
                                        MInfo &mInfo) {
  patterns
      ->insert<convertFullyConnectedTilePattern<tpu::TG_INT8_FullyConnectedOp>,
               convertFullyConnectedTilePattern<tpu::TG_BF16_FullyConnectedOp>>(
          context, mInfo);
}

std::unique_ptr<mlir::Pass> createFullyConnectedTilePass() {
  return std::make_unique<tpu::FullyConnectedTilePass>();
}

} // namespace tpu
} // namespace mlir
