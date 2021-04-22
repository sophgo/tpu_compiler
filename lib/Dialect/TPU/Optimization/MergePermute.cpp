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
// This file implements the TPU dialect OP Stats pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Passes.h"
#include "tpuc/Support/TensorFile.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "llvm/Support/raw_ostream.h"
#include "tpuc/MachineInfo.h"

#define DEBUG_TYPE "eltwise_early_stride"

using namespace mlir;

namespace {

template <typename OpTy>
struct MergePermuteOpPattern : public RewritePattern {
  MergePermuteOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  bool checkPattern(Operation *op) const {
    // reshape
    auto inst_1 = dyn_cast<tpu::ReshapeOp>(op);

    // reshape
    auto inst_2 = getNextOp(inst_1);
    if (!(inst_2 && isa<tpu::ReshapeOp>(inst_2)))
      return false;

    // permute
    auto inst_3 = getNextOp(inst_2);
    if (!(inst_3 && isa<tpu::PermuteOp>(inst_3)))
      return false;

    // pad
    auto inst_4 = getNextOp(inst_3);
    if (!(inst_4 && isa<tpu::PadOp>(inst_4)))
      return false;

    // reshape
    auto inst_5 = getNextOp(inst_4);
    if (!(inst_5 && isa<tpu::ReshapeOp>(inst_5)))
      return false;

    // relu
    auto inst_6 = getNextOp(inst_5);
    if (!(inst_6 && isa<tpu::ReluOp>(inst_6)))
      return false;

    // reshape
    auto inst_7 = getNextOp(inst_6);
    if (!(inst_7 && isa<tpu::ReshapeOp>(inst_7)))
      return false;

    // permute
    auto inst_8 = getNextOp(inst_7);
    if (!(inst_8 && isa<tpu::PermuteOp>(inst_8)))
      return false;

    // reshape
    auto inst_9 = getNextOp(inst_8);
    if (!(inst_9 && isa<tpu::ReshapeOp>(inst_9)))
      return false;

    return true;
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {

    // check for pattern:
    // reshape + reshape + permute + pad + reshape + relu + reshape + permute + reshape
    // convert to
    // pad + relu + reshape

    // reshape
    auto inst_1 = dyn_cast<tpu::ReshapeOp>(op);

    // reshape
    auto inst_2 = getNextOp(inst_1);
    if (!(inst_2 && isa<tpu::ReshapeOp>(inst_2)))
      return failure();

    // permute
    auto inst_3 = getNextOp(inst_2);
    if (!(inst_3 && isa<tpu::PermuteOp>(inst_3)))
      return failure();

    // pad
    auto inst_4 = getNextOp(inst_3);
    if (!(inst_4 && isa<tpu::PadOp>(inst_4)))
      return failure();

    // reshape
    auto inst_5 = getNextOp(inst_4);
    if (!(inst_5 && isa<tpu::ReshapeOp>(inst_5)))
      return failure();

    // relu
    auto inst_6 = getNextOp(inst_5);
    if (!(inst_6 && isa<tpu::ReluOp>(inst_6)))
      return failure();

    // reshape
    auto inst_7 = getNextOp(inst_6);
    if (!(inst_7 && isa<tpu::ReshapeOp>(inst_7)))
      return failure();

    // permute
    auto inst_8 = getNextOp(inst_7);
    if (!(inst_8 && isa<tpu::PermuteOp>(inst_8)))
      return failure();

    // reshape
    auto inst_9 = getNextOp(inst_8);
    if (!(inst_9 && isa<tpu::ReshapeOp>(inst_9)))
      return failure();

    //
    rewriter.setInsertionPointAfter(inst_9);
    auto pad_inst_4 = dyn_cast_or_null<tpu::PadOp>(inst_4);
    auto relu_inst_6 = dyn_cast_or_null<tpu::ReluOp>(inst_6);
    auto reshape_inst_9 = dyn_cast_or_null<tpu::ReshapeOp>(inst_9);

    std::vector<int32_t> pads;
    arrayAttrToVector(pad_inst_4.pads().getValue(), pads);
    auto const_val = pad_inst_4.const_val().convertToFloat();
    // generate pad
    SmallVector<Attribute, 8> padsAttr;
    for (unsigned int i = 0; i < 8; i++) {
      int v = 0;
      if (i == 3)
        v = pads[2];
      if (i == 7)
        v = pads[6];
      auto padAttr = rewriter.getI32IntegerAttr(v);
      padsAttr.push_back(padAttr);
    }

    std::vector<int64_t> inst_1_shape = getTensorShape(inst_1->getOperand(0));
    auto inst_1_type = inst_1->getResult(0).getType().cast<RankedTensorType>();
    std::vector<Value> operands;
    operands.push_back(inst_1.getOperand());
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("name", pad_inst_4.nameAttr()));
    attrs.push_back(
        rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
    attrs.push_back(rewriter.getNamedAttr("pads",
                           rewriter.getArrayAttr(padsAttr)));
    attrs.push_back(rewriter.getNamedAttr("const_val",
                           rewriter.getF32FloatAttr(const_val)));
    RankedTensorType output_type = RankedTensorType::get(
                          {inst_1_shape[0], inst_1_shape[1],
                           inst_1_shape[2], inst_1_shape[3] + pads[2] + pads[6]},
                           inst_1_type.getElementType());

    auto newPadOp = rewriter.create<tpu::PadOp>(
          inst_9->getLoc(), output_type,
          ArrayRef<Value>{operands},
          ArrayRef<NamedAttribute>{attrs});

    // generate relu
    std::vector<Value> ops_relu;
    ops_relu.push_back(newPadOp.getResult());
    std::vector<NamedAttribute> attrs_relu;
    attrs_relu.push_back(rewriter.getNamedAttr("name", relu_inst_6.nameAttr()));
    attrs_relu.push_back(
        rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
    auto newReluOp = rewriter.create<tpu::ReluOp>(
          inst_9->getLoc(), output_type,
          ArrayRef<Value>{ops_relu},
          ArrayRef<NamedAttribute>{attrs_relu});

    // generate reshape
    std::vector<Value> ops_reshape;
    ops_reshape.push_back(newReluOp.getResult());
    std::vector<NamedAttribute> attrs_reshape;
    attrs_reshape.push_back(rewriter.getNamedAttr("name", reshape_inst_9.nameAttr()));
    auto newReshapeOp = rewriter.create<tpu::ReshapeOp>(
          inst_9->getLoc(), inst_9->getResult(0).getType(),
          ArrayRef<Value>{ops_reshape},
          ArrayRef<NamedAttribute>{attrs_reshape});

    inst_9->getResult(0).replaceAllUsesWith(newReshapeOp.getResult());

    // erase 9 insts
    rewriter.eraseOp(inst_9);
    rewriter.eraseOp(inst_8);
    rewriter.eraseOp(inst_7);
    rewriter.eraseOp(inst_6);
    rewriter.eraseOp(inst_5);
    rewriter.eraseOp(inst_4);
    rewriter.eraseOp(inst_3);
    rewriter.eraseOp(inst_2);
    rewriter.eraseOp(inst_1);

    return success();
  }
};

class MergePermuteOpPass
    : public mlir::PassWrapper<MergePermuteOpPass, FunctionPass> {
public:
  void runOnFunction() override {
    auto fn = getFunction();
    MInfo machineInfo;
    machineInfo.getChipInfo(fn);
    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.clear();
    patterns.insert<
        MergePermuteOpPattern<tpu::ReshapeOp>
      >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::createMergePermutePass() {
  return std::make_unique<MergePermuteOpPass>();
}
