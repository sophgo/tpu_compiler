//===- TgOpTile.cpp - Implementation of TG Op tiling ----------------------===//
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
// This file implements the tiling of TG operation.
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

#include "TgOpTile.h"

#define DEBUG_TYPE "tg-op-tile"

using namespace mlir;

namespace {
struct TgOpTilePass : public FunctionPass<TgOpTilePass> {
  void runOnFunction() override {
    std::string getRunChipType;
    MInfo mInfo;
    get_cvichip_name(getRunChipType);

    if (!getRunChipType.size())
      return;

    mInfo.getChipInfo(getRunChipType.c_str());
    assert(MInfo::version && "refer to set-chip");

    getFunction().walk([&](Operation *op) {
      if (auto tpuOp = dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(op)) {
        tpuOp.removeAttr("tile_param");
      } else if (auto tpuOp = dyn_cast<tpu::TG_INT8_FullyConnectedOp>(op)) {
        tpuOp.removeAttr("tile_step");
      } else if (auto tpuOp = dyn_cast<tpu::TG_BF16_FullyConnectedOp>(op)) {
        tpuOp.removeAttr("tile_step");
      }
    });

    OwningRewritePatternList patterns;
    tpu::PopulateConvTilePatterns(&getContext(), &patterns, mInfo);
    tpu::PopulateFullyConnectedTilePatterns(&getContext(), &patterns, mInfo);
    applyPatternsGreedily(getFunction(), patterns);
  }
};

} // anonymous namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createTgOpTilePass() {
  return std::make_unique<TgOpTilePass>();
}

static PassRegistration<TgOpTilePass>
    pass("tg-op-tile",
         "TG Op tiling");
