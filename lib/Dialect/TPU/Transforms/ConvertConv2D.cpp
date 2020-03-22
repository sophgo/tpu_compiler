//===- TpuOpStats.cpp - Implementation of TPU Op Stats ---------===//
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

#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

struct TpuMergeSwapChannelToConv2DPattern : public RewritePattern {
  TpuMergeSwapChannelToConv2DPattern(MLIRContext *context)
      : RewritePattern("tpu.conv_2d", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto convOp = cast<tpu::Conv2DOp>(op);
    llvm::errs() << convOp.getOperationName() << ":" << getOpName(convOp)
                 << "\n";

    // match SwapChannel Op that is following conv2d
    auto formerOp = op->getOperand(0)->getDefiningOp();

    if (!isa<tpu::SwapChannelOp>(formerOp)) {
      return matchFailure();
    }

    if (convOp.param().group().getValue().getLimitedValue() != 1) {
      return matchFailure();
    }

    auto swapOp = cast<tpu::SwapChannelOp>(formerOp);
    std::vector<int32_t> order;
    arrayAttrToVector(swapOp.channel_order().getValue(), order);

    auto inputOp = formerOp->getOperand(0);
    auto loc = op->getLoc();
    TensorFile *wTF = getWeightTensorFile(op);
    Value *wfV = getWeightFileValue(op);

    // find filter and bias tensor for conv op
    assert(convOp.getNumOperands() == 7);
    std::unique_ptr<std::vector<float>> convWeights;
    auto filter_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
        convOp.getOperand(1)->getDefiningOp());
    if (!filter_op) {
      return matchFailure();
    }
    assert(filter_op.name().hasValue());
    auto filter_name = filter_op.name().getValue();
    auto filter_type = filter_op.getResult()->getType().cast<TensorType>();
    convWeights = wTF->readTensor<float>(filter_name, filter_type);
    // delete the tensor from the weight file
    wTF->deleteTensor<float>(filter_name);

    std::vector<int64_t> filter_shape(filter_type.getShape());
    int64_t filter_size =
        std::accumulate(std::begin(filter_shape), std::end(filter_shape), 1,
                        std::multiplies<>());
    assert(filter_size == (int64_t)convWeights->size());
    int64_t oc, ic, frame_size;
    int64_t index = filter_shape.size();
    assert(index == 4 || index == 5);
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
    attrs.push_back(
        rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
    auto new_filter_op = rewriter.create<tpu::LoadWeightOp>(
        loc, filter_type, ArrayRef<Value *>{wfV},
        ArrayRef<NamedAttribute>{attrs});
    convOp.setOperand(0, inputOp);
    convOp.setOperand(1, new_filter_op.getResult());

    // remove the swap channel Op
    rewriter.replaceOp(formerOp, {convOp});
    return matchSuccess();
  }
};

} // namespace

void tpu::Conv2DOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<TpuMergeSwapChannelToConv2DPattern>(context);
}

void tpu::DeConv2DOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {}
