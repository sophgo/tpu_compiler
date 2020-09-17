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

#define DEBUG_TYPE "fully-connected-tile"

using namespace mlir;

namespace {

class FullyConnectedModel {
public:
  FullyConnectedModel(const MInfo &mInfo, int M, int K, int N, bool hasB,
                      int dataTypeSize)
      : mInfo(mInfo), M(M), K(K), N(N), hasB(hasB), dataTypeSize(dataTypeSize)
      {}

  struct TileInfo {
    int m_step;
    int n_step;
    int k_step;
  };

  TileInfo getTileSizes();
  int getLmSizePerLane(int tileM, int tileK, int tileN);
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
int FullyConnectedModel::getLmSizePerLane(int tileM, int tileK, int tileN) {
  const int npuNum = mInfo.lane_num;
  const int euNum = mInfo.eu_num / dataTypeSize;  // bf16: 1/2 eu num

  int tileLSize = tileM * euNum * llvm::divideCeil(tileK, euNum * npuNum) *
                  dataTypeSize;

  int tileRSize = tileK * euNum * llvm::divideCeil(tileN, euNum * npuNum) *
                  dataTypeSize;

  int tileBSize = hasB ?
      (2 * euNum * llvm::divideCeil(tileN, euNum * npuNum) * dataTypeSize) : 0;

  // Partial sum 32bit, 2x(bf16), 4x(int8)
  int tileYSize = tileM * euNum * llvm::divideCeil(tileN, euNum * npuNum) *
                  dataTypeSize;
  if (tileK < K)
    tileYSize *= 4 / dataTypeSize;

  int totalSize = tileLSize + tileRSize + tileYSize + tileBSize;

  if (totalSize <= (int)mInfo.lmem_per_lane) {
    LLVM_DEBUG(llvm::dbgs()
        << "    FullyConnectedModel::getLmSizePerLane:\n      "
        << "M " << M
        << ", K " << K
        << ", N " << N << "\n      "
        << "tileL(" << tileM
        << ", " << tileK << ")\n      "
        << "tileR(" << tileK
        << ", " << tileN << ")\n      "
        << "tileY(" << tileM
        << ", " << tileN << ")\n      "
        << "tileYSize " << tileYSize
        << ", tileLSize " << tileLSize
        << ", tileRSize " << tileRSize
        << ", tileBSize " << tileBSize << "\n");
  }

  return totalSize;
}

FullyConnectedModel::TileInfo FullyConnectedModel::getTileSizes() {
  int tileM = M;

  // 1/2 EU in bf16
  int totalEuNum = (dataTypeSize == 2) ? mInfo.eu_num / 2 : mInfo.eu_num;
  totalEuNum *= mInfo.lane_num;
  int tileN = (N >= totalEuNum) ? totalEuNum : N;

  int maxK = (1 << 12) - 1; // TIU 12 bit
  int tileK = (K > maxK) ? maxK : K;

  TileInfo tileInfo = {0};
  for (; tileK > 0; --tileK) {
    int needed = getLmSizePerLane(tileM, tileK, tileN);
    if (needed <= (int)mInfo.lmem_per_lane) {
      tileInfo.m_step = tileM;
      tileInfo.k_step = tileK;
      tileInfo.n_step = tileN;
      break;
    }
  }

  return tileInfo;
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

template<typename OpTy>
class convertFullyConnectedTilePattern : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  convertFullyConnectedTilePattern(MLIRContext *ctx, MInfo &mInfo)
      : OpRewritePattern<OpTy>(ctx), mInfo(mInfo) {}

  PatternMatchResult matchAndRewrite(OpTy tpuOp,
                                     PatternRewriter &rewriter) const override {

    // Already configured
    if (tpuOp.tile_param().hasValue())
      return Pattern::matchFailure();

    auto op = tpuOp.getOperation();

    LLVM_DEBUG(llvm::dbgs()
        << "convertFullyConnectedTilePattern: layer ID "
        << mlir::getOpLayerId(op)
        << ", " << tpuOp.name() << "\n");

    int m, k, n;
    parseFullyConnectedParam(tpuOp.input(), tpuOp.output(), tpuOp.filter(), m,
                             k, n);

    bool hasBias = isTensorNone(tpuOp.bias()) ? false : true;

    auto retType =
        op->getResult(0)->getType().template dyn_cast<RankedTensorType>();
    auto elementType = retType.getElementType();
    int dataTypeSize = elementType.getIntOrFloatBitWidth() / 8;

    auto fcModel(std::make_unique<FullyConnectedModel>(mInfo, m, k, n,
                                                       hasBias, dataTypeSize));

    FullyConnectedModel::TileInfo tileInfo = fcModel->getTileSizes();
    if (!tileInfo.m_step || !tileInfo.k_step || !tileInfo.n_step)
      return Pattern::matchFailure();

    SmallVector<int32_t, 4> tileValues = {
        tileInfo.m_step, tileInfo.k_step, tileInfo.n_step};

    std::vector<int> n_poss;
    std::vector<int> k_poss;
    std::vector<int> n_sizes;
    std::vector<int> k_sizes;
    fcModel->getTilePoss(tileInfo, n_poss, k_poss, n_sizes, k_sizes);
    tpuOp.setAttr("tile_param",
                  tpu::FcTileParam::get(rewriter.getI32ArrayAttr(tileValues),
                                        rewriter.getI32ArrayAttr(n_poss),
                                        rewriter.getI32ArrayAttr(k_poss),
                                        rewriter.getI32ArrayAttr(n_sizes),
                                        rewriter.getI32ArrayAttr(k_sizes),
                                        rewriter.getContext()));

    return Pattern::matchSuccess();
  }

  MInfo &mInfo;
};

struct FullyConnectedTilePass : public FunctionPass<FullyConnectedTilePass> {
  void runOnFunction() override;
};

} // anonymous namespace

void FullyConnectedTilePass::runOnFunction() {
  std::string getRunChipType;
  MInfo machineInfo;
  get_cvichip_name(getRunChipType);
  machineInfo.getChipInfo(getRunChipType.c_str());
  assert(MInfo::version && "refer to set-chip");

  getFunction().walk([&](Operation *op) {
    if (auto tpuOp = dyn_cast<tpu::TG_INT8_FullyConnectedOp>(op)) {
      tpuOp.removeAttr("tile_step");
    } else if (auto tpuOp = dyn_cast<tpu::TG_BF16_FullyConnectedOp>(op)) {
      tpuOp.removeAttr("tile_step");
    }
  });

  OwningRewritePatternList patterns;
  patterns.insert<
      convertFullyConnectedTilePattern<tpu::TG_INT8_FullyConnectedOp>,
      convertFullyConnectedTilePattern<tpu::TG_BF16_FullyConnectedOp>
      >(&getContext(), machineInfo);
  applyPatternsGreedily(getFunction(), patterns);
}

std::unique_ptr<OpPassBase<FuncOp>> mlir::createFullyConnectedTilePass() {
  return std::make_unique<FullyConnectedTilePass>();
}

static PassRegistration<FullyConnectedTilePass>
  pass("fully-connected-tile", "Fully connected tile");
