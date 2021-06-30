//===- ConvertSwapChannel.cpp - convert Conv2D----------------------------------===//
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
// This file implements the conversion of Conv2D.
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
#include "llvm/Support/FormatVariadic.h"
#include <math.h> // ceilf/floorf
#include "tpuc/SimpleAnalysis.h"

#define DEBUG_TYPE "convert_conv"

using namespace mlir;

namespace {

struct TpuMergeSwapChannelToConv2DPattern : public RewritePattern {
  TpuMergeSwapChannelToConv2DPattern(MLIRContext *context)
      : RewritePattern("tpu.swap_channel", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto swapOp = cast<tpu::SwapChannelOp>(op);
    std::vector<int32_t> order;
    arrayAttrToVector(swapOp.channel_order(), order);
    std::vector<Operation *> targetOps;
    // check if all use is conv2d
    for (auto &use : op->getResult(0).getUses()) {
      auto nextOp = use.getOwner();
      if (auto castOp = dyn_cast_or_null<tpu::CropOp>(nextOp)) {
        auto shape = getTensorShape(castOp.getResult());
        if (shape[1] != (int)order.size()) {
          return failure();
        }
        auto latterOp = getNextOp(nextOp);
        if (!latterOp) {
          return failure();
        }
        if (!isa<tpu::Conv2DOp>(latterOp)) {
          return failure();
        }
        targetOps.push_back(latterOp);
      } else if (isa<tpu::Conv2DOp>(nextOp)) {
        targetOps.push_back(nextOp);
      } else {
        return failure();
      }
    }

    for (auto target : targetOps) {
      auto convOp = dyn_cast<tpu::Conv2DOp>(target);
      auto is_dw = convOp.param().is_dw().getValue();
      auto group = (int)convOp.param().group().getInt();
      if (is_dw || group > 1) {
        return failure();
      }
    }

    for (auto target : targetOps) {
      swapChannelOfConv2dWeight(target, order, rewriter);
    }

    swapOp.replaceAllUsesWith(op->getOperand(0));
    swapOp.erase();

    return success();
  }

  void swapChannelOfConv2dWeight(Operation *op, std::vector<int32_t> &order,
                                 PatternRewriter &rewriter) const {
    llvm::errs() << "swapChannelOfConv2dWeight:";
    op->getResult(0).dump();
    TensorFile *wTF = getWeightTensorFile(op);
    Value wfV = getWeightFileValue(op);
    auto convOp = dyn_cast<tpu::Conv2DOp>(op);
    // find filter and bias tensor for conv op
    assert(convOp.getNumOperands() == 7 && "Conv2D op should have 7 operands");
    std::unique_ptr<std::vector<float>> convWeights;

    auto filter_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
        convOp.getOperand(1).getDefiningOp());
    assert(filter_op);

    auto filter_name = filter_op.name();
    auto filter_type = filter_op.getResult().getType().cast<TensorType>();
    convWeights = wTF->readTensor<float>(filter_name, filter_type);
    // delete the tensor from the weight file
    wTF->deleteTensor<float>(filter_name);

    std::vector<int64_t> filter_shape(filter_type.getShape());
    int64_t filter_size =
        std::accumulate(std::begin(filter_shape), std::end(filter_shape), 1,
                        std::multiplies<>());
    assert(filter_size == (int64_t)convWeights->size() &&
           "filter size should be equal");
    int64_t oc, ic, frame_size;
    int64_t index = filter_shape.size();
    assert((index == 4 || index == 5) && "filter shape size should be 4 or 5");
    frame_size = filter_shape[index - 1] * filter_shape[index - 2];
    ic = filter_shape[index - 3];
    oc = filter_shape[index - 4];
    if (index == 5) {
      oc *= filter_shape[index - 5];
    }
    std::vector<float> new_filter(filter_size);
    float *filter = (float *)convWeights->data();
    for (int i = 0; i < oc; ++i) {
      for (int j = 0; j < ic; ++j) {
        assert(order[j] < ic);
        float *in = filter + i * ic * frame_size + order[j] * frame_size;
        float *out =
            (float *)new_filter.data() + i * ic * frame_size + j * frame_size;
        memcpy(out, in, frame_size * sizeof(float));
      }
    }
    std::string tensor_name = filter_name.str() + "_swap_channel";
    wTF->addTensor<float>(tensor_name, &new_filter, filter_type);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr(
        "name", rewriter.getStringAttr(tensor_name)));
    attrs.push_back(rewriter.getNamedAttr(
        "storage", rewriter.getStringAttr(filter_op.storage().str())));
    auto new_filter_op = rewriter.create<tpu::LoadWeightOp>(
        filter_op->getLoc(), filter_type, ArrayRef<Value>{wfV},
        ArrayRef<NamedAttribute>{attrs});

    rewriter.replaceOp(filter_op, {new_filter_op});
  }
};

} // namespace

void tpu::SwapChannelOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<TpuMergeSwapChannelToConv2DPattern>(context);
}
