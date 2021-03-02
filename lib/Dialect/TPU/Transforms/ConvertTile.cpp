//===- ConvertTile.cpp - convert
// Tile----------------------------------===//
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
// This file implements the tile
//
//===----------------------------------------------------------------------===//

#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tpuc/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "convert_tile"

using namespace mlir;

namespace {

static bool getAxisAndTile(const std::vector<int32_t> &resp, int &axis,
                           int &tile) {
  if (resp.size() != 4) {
    return false;
  }
  int32_t total =
      std::accumulate(resp.begin(), resp.end(), 1, std::multiplies<int32_t>());
  for (int i = 0; i < 4; i++) {
    if (resp[i] == total) {
      axis = i;
      tile = total;
      return true;
    }
  }
  return false;
}

// if tile == 1, then convert to scale
struct TpuTileToUpsamplePattern : public RewritePattern {
  TpuTileToUpsamplePattern(MLIRContext *context)
      : RewritePattern("tpu.tile", 8, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto tileOp = cast<tpu::TileOp>(op);
    std::vector<int32_t> resp1;
    arrayAttrToVector(tileOp.resp().getValue(), resp1);
    std::vector<int32_t> ones(4, 1);
    if (resp1 == ones) {
      // if resp = [1,1,1,1], remove this op
      rewriter.replaceOp(op, {op->getOperand(0)});
      return success();
    }
    std::vector<int64_t> shape1;
    int64_t input_size1;
    getTensorShapeAndSize(tileOp.input(), shape1, input_size1);
    if (shape1.size() != 4) {
      return failure();
    }
    int axis1 = 0, tile1 = 0;
    if (false == getAxisAndTile(resp1, axis1, tile1) || axis1 < 2) {
      return failure();
    }

    auto formerOp = tileOp.getOperand(0).getDefiningOp();
    if (false == isa<tpu::TileOp>(formerOp)) {
      return failure();
    }
    auto tileOp2 = cast<tpu::TileOp>(formerOp);
    std::vector<int32_t> resp2;
    arrayAttrToVector(tileOp2.resp().getValue(), resp2);

    std::vector<int64_t> shape2;
    int64_t input_size2;
    getTensorShapeAndSize(tileOp2.input(), shape2, input_size2);
    if (shape2.size() != 4 || shape2[2] != 1 || shape2[3] != 1) {
      return failure();
    }
    int axis2 = 0, tile2 = 0;
    if (false == getAxisAndTile(resp2, axis2, tile2) || axis2 < 2) {
      return failure();
    }
    if (tile2 != tile1 || axis2 < 2 || axis2 == axis1) {
      return failure();
    }
    // remove this op
    std::string op_name =
        tileOp->getAttrOfType<StringAttr>("name").getValue().str();

    std::vector<Value> newOperands;
    newOperands.push_back(tileOp2.getOperand(0));
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("name", rewriter.getStringAttr(op_name)));
    attrs.push_back(
        rewriter.getNamedAttr("scale_h", rewriter.getI32IntegerAttr(tile1)));
    attrs.push_back(
        rewriter.getNamedAttr("scale_w", rewriter.getI32IntegerAttr(tile1)));
    attrs.push_back(
        rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
    auto upsampleOp = rewriter.create<tpu::UpsampleOp>(
        op->getLoc(), tileOp.getResult().getType(),
        ArrayRef<Value>{newOperands}, ArrayRef<NamedAttribute>{attrs});
    rewriter.replaceOp(tileOp, {upsampleOp.getResult()});
    rewriter.replaceOp(tileOp2, {upsampleOp.getResult()});
    return success();
  }
};

} // namespace

void tpu::TileOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  // directly use backend tile op
  // results.insert<TpuTileToUpsamplePattern>(context);
}

void tpu::TileInterpOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  // do nothing
}
