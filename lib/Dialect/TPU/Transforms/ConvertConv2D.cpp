//===- ConvertConv2D.cpp - convert Conv2D----------------------------------===//
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

#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "convert_conv"

using namespace mlir;

namespace {

struct TpuMergeSwapChannelToConv2DPattern : public RewritePattern {
  TpuMergeSwapChannelToConv2DPattern(MLIRContext *context)
      : RewritePattern("tpu.conv_2d", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto convOp = cast<tpu::Conv2DOp>(op);
    LLVM_DEBUG(llvm::errs() << convOp.getOperationName() << ":"
                            << getOpName(op)<< "\n";);

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
    assert(convOp.getNumOperands() == 7 && "Conv2D op should have 7 operands");
    std::unique_ptr<std::vector<float>> convWeights;
    auto filter_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
        convOp.getOperand(1)->getDefiningOp());
    if (!filter_op) {
      return matchFailure();
    }
    auto filter_name = filter_op.name();
    auto filter_type = filter_op.getResult()->getType().cast<TensorType>();
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

struct TpuConvertDilationWeightPattern : public RewritePattern {
  TpuConvertDilationWeightPattern(MLIRContext *context)
      : RewritePattern("tpu.conv_2d", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto convOp = cast<tpu::Conv2DOp>(op);
    LLVM_DEBUG(llvm::errs() << convOp.getOperationName() << ":"
                            << getOpName(op)<< "\n";);

    auto dh = convOp.param().dilation_h().getValue().getLimitedValue();
    auto dw = convOp.param().dilation_w().getValue().getLimitedValue();
    const int DILATION_H_MAX = 15;
    const int DILATION_W_MAX = 15;
    if (dh <= DILATION_H_MAX && dw <= DILATION_W_MAX)
      return matchFailure();

    TensorFile *wTF = getWeightTensorFile(op);
    auto filter = readAndDeleteWeightTensor<float>(convOp.filter(), wTF);
    std::vector<int64_t> filterShape;
    filterShape = getTensorShape(convOp.filter());

    int64_t oc = 0;
    int64_t ic = 0;
    int64_t kh = 0;
    int64_t kw = 0;
    if (filterShape.size() == 4) {
      oc = filterShape[0];
      ic = filterShape[1];
      kh = filterShape[2];
      kw = filterShape[3];
    } else if (filterShape.size() == 5) {
      // g, oc/g, ic/g, kh, kw
      oc = filterShape[0] * filterShape[1];
      ic = filterShape[2];
      kh = filterShape[3];
      kw = filterShape[4];
    } else {
      assert(0);
    }

    int insertNumH = 0;
    int insertNumW = 0;
    int newDilationH = dh;
    int newDilationW = dw;
    while(1) {
      insertNumH++;
      newDilationH = (dh - 1 - insertNumH) / (insertNumH + 1) + 1;
      if (((dh - 1 - insertNumH) % (insertNumH + 1) == 0) &&
         newDilationH < DILATION_H_MAX)
        break;
    }

    while(1) {
      insertNumW++;
      newDilationW = (dw - 1 - insertNumW) / (insertNumW + 1) + 1;
      if (((dw - 1 - insertNumW) % (insertNumW + 1) == 0) &&
         newDilationW < DILATION_W_MAX)
        break;
    }

    int k_ext_h = (insertNumH + 1) * (kh - 1) + 1;
    int k_ext_w = (insertNumW + 1) * (kw - 1) + 1;
    filterShape[2] = k_ext_h;
    filterShape[3] = k_ext_w;
    auto filterSize = oc * ic * k_ext_h * k_ext_w;
    std::vector<float> newFilter(filterSize, 0);
    for (int i = 0; i < oc * ic; i++) {
      for (int j = 0; j < kh; j++) {
        for (int k = 0; k < kw; k++) {
          auto old_offset = i * kh * kw + j * kw + k;
          auto new_offset = i * k_ext_h * k_ext_w +
                            j * (insertNumW + 1) * k_ext_w +
                            k * (insertNumH + 1);
          newFilter[new_offset] = filter->data()[old_offset];
        }
      }
    }
    // update op
    addWeightTensorAndUpdateWeightOp<float>(convOp.getOperand(1),
        "dilation", newFilter, filterShape, "INT8", wTF);

    // rewrite pad
    convOp.setAttr("param",
           tpu::ConvParam::get(
                convOp.param().stride_h(),
                convOp.param().stride_w(),
                convOp.param().padding(),
                rewriter.getI32IntegerAttr(newDilationH),
                rewriter.getI32IntegerAttr(newDilationW),
                convOp.param().padding_t(),
                convOp.param().padding_b(),
                convOp.param().padding_l(),
                convOp.param().padding_r(),
                convOp.param().group(),
                convOp.param().is_dw(),
                convOp.param().with_bias(),
                convOp.param().do_relu(),
                rewriter.getContext()));

    return matchSuccess();
  }
};


struct TpuSplitConv2DPattern : public RewritePattern {
  TpuSplitConv2DPattern(MLIRContext *context)
      : RewritePattern("tpu.conv_2d", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto convOp = cast<tpu::Conv2DOp>(op);
    LLVM_DEBUG(llvm::errs() << convOp.getOperationName() << ":"
                            << getOpName(op)<< "\n";);
    auto param = convOp.param();
    auto pt = param.padding_t().getValue().getLimitedValue();
    auto pb = param.padding_b().getValue().getLimitedValue();
    auto pl = param.padding_l().getValue().getLimitedValue();
    auto pr = param.padding_r().getValue().getLimitedValue();

    const int PAD_H_MAX = 15;
    const int PAD_W_MAX = 15;
    if (pt <= PAD_H_MAX && pb <= PAD_H_MAX &&
        pl <= PAD_W_MAX && pr <= PAD_W_MAX)
      return matchFailure();

    std::vector<NamedAttribute> attrs;
    std::vector<Value *> operands;
    operands.push_back(op->getOperand(0));

    auto name = convOp.name().str();
    name = name + "_" + "pad";
    attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(name)));

    SmallVector<Attribute, 8> padsAttr;
    int pad_h_begin = pt;
    int pad_w_begin = pl;

    int pad_h_end = pb;
    int pad_w_end = pr;

    auto padAttr = rewriter.getI32IntegerAttr(0);
    padsAttr.push_back(padAttr); // pad_n_begin;
    padsAttr.push_back(padAttr); // pad_c_begin;

    padAttr = rewriter.getI32IntegerAttr(pad_h_begin);
    padsAttr.push_back(padAttr);

    padAttr = rewriter.getI32IntegerAttr(pad_w_begin);
    padsAttr.push_back(padAttr);

    padAttr = rewriter.getI32IntegerAttr(0);
    padsAttr.push_back(padAttr); // pad_n_end;
    padsAttr.push_back(padAttr); // pad_c_end;

    padAttr = rewriter.getI32IntegerAttr(pad_h_end);
    padsAttr.push_back(padAttr);

    padAttr = rewriter.getI32IntegerAttr(pad_w_end);
    padsAttr.push_back(padAttr);


    attrs.push_back(rewriter.getNamedAttr("pads",
                                      rewriter.getArrayAttr(padsAttr)));
    attrs.push_back(rewriter.getNamedAttr("const_val",
                                      rewriter.getF32FloatAttr(0)));
    attrs.push_back(rewriter.getNamedAttr("quant",
                                      getDefaultQuantParam(rewriter)));

    auto input_type = convOp.input()->getType().dyn_cast<TensorType>();
    auto input_shape = input_type.getShape();
    int64_t output_h = input_shape[2] + pt + pb;
    int64_t output_w = input_shape[3] + pl + pr;

    auto output_type = RankedTensorType::get(
                          {input_shape[0], input_shape[1],
                           output_h, output_w},
                           input_type.getElementType());

    rewriter.setInsertionPoint(op);
    auto padOp = rewriter.create<tpu::PadOp>(op->getLoc(),
                           ArrayRef<mlir::Type>{output_type}, operands, attrs);
    // rewrite pad
    convOp.setAttr("param",
           tpu::ConvParam::get(
                convOp.param().stride_h(),
                convOp.param().stride_w(),
                convOp.param().padding(),
                convOp.param().dilation_h(),
                convOp.param().dilation_w(),
                rewriter.getI32IntegerAttr(0),
                rewriter.getI32IntegerAttr(0),
                rewriter.getI32IntegerAttr(0),
                rewriter.getI32IntegerAttr(0),
                convOp.param().group(),
                convOp.param().is_dw(),
                convOp.param().with_bias(),
                convOp.param().do_relu(),
                rewriter.getContext()));
    op->setOperand(0, padOp.getResult());
    return matchSuccess();
  }
};
} // namespace

void tpu::Conv2DOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<TpuMergeSwapChannelToConv2DPattern,
                 TpuSplitConv2DPattern,
                 TpuConvertDilationWeightPattern >(context);
}

void tpu::DeConv2DOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {}


