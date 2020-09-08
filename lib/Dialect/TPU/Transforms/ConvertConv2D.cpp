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
#include "llvm/Support/FormatVariadic.h"
#include <math.h> // ceilf/floorf
#include "mlir/Dialect/TPU/SimpleAnalysis.h"

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
    attrs.push_back(rewriter.getNamedAttr(
        "storage", rewriter.getStringAttr(filter_op.storage().str())));
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
                convOp.param().ins(),
                rewriter.getContext()));

    return matchSuccess();
  }
};

template <typename OpTy>
std::pair<std::vector<Value *>, std::vector<NamedAttribute> > getTwiceWDeConv(
    OpTy castOp,
    Operation *op,
    PatternRewriter &rewriter) {

  // input
  std::vector<int64_t> shape;
  int64_t input_size, in, ic, ih, iw;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, in, ic, ih, iw);

  // output
  std::vector<int64_t> output_shape;
  int64_t output_size;
  getTensorShapeAndSize(op->getResult(0), output_shape, output_size);

  // get weight
  TensorFile *wTF = getWeightTensorFile(op);
  Value *wfV = getWeightFileValue(op);

  // construct deconv
  auto input = op->getOperand(0);
  auto input_type = input->getType().cast<RankedTensorType>();
  auto input_shape = input_type.getShape();
  int g = input_shape[1];
  int oc = input_shape[1] / g;
  ic = input_shape[1] / g;
  int h = 1;
  int w = iw * 2; // ONLY extend h

  int count = g * oc * ic * h * w;
  std::vector<float> filter(count, 1);
  std::vector<int64_t> filter_shape;
  if (g != 1) {
    filter_shape.push_back(g);
  }

  filter_shape.push_back(oc);
  filter_shape.push_back(ic);
  filter_shape.push_back(h);
  filter_shape.push_back(w);

  auto filterValue = addWeightTensorAndCreateWeightOp<float>(op, "filter",
      filter, filter_shape, "NONE", wTF, wfV);

  auto NoneOp = OpBuilder(op).create<tpu::NoneOp>(rewriter.getUnknownLoc(),
      rewriter.getNoneType());

  std::vector<Value *> operands;
  operands.push_back(input);
  operands.push_back(filterValue);
  operands.push_back(NoneOp.getResult()); // bias
  operands.push_back(NoneOp.getResult()); // quant_scale
  operands.push_back(NoneOp.getResult()); // quant_zeropoint
  operands.push_back(NoneOp.getResult()); // quant_rshift
  operands.push_back(NoneOp.getResult()); // quant_multiplier

  bool is_dw = true;
  bool with_bias = false;
  std::vector<int64_t> kernel(2), stride(2), padding(2), dilation(2);
  kernel[0] = h;
  kernel[1] = w;
  padding[0] = padding[1] = 0;
  dilation[0] = dilation[1] = 1;
  stride[0] = 1;
  stride[1] = 2; // set w to twice

  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("name", castOp.nameAttr()));
  attrs.push_back(rewriter.getNamedAttr("param",
        tpu::ConvParam::get(rewriter.getI32IntegerAttr(stride[0]),
          rewriter.getI32IntegerAttr(stride[1]),
          rewriter.getStringAttr("VALID"),
          rewriter.getI32IntegerAttr(dilation[0]),
          rewriter.getI32IntegerAttr(dilation[1]),
          rewriter.getI32IntegerAttr(padding[0]),
          rewriter.getI32IntegerAttr(padding[0]),
          rewriter.getI32IntegerAttr(padding[1]),
          rewriter.getI32IntegerAttr(padding[1]),
          rewriter.getI32IntegerAttr(g),
          rewriter.getBoolAttr(is_dw),
          rewriter.getBoolAttr(with_bias),
          rewriter.getBoolAttr(false),
          rewriter.getI32ArrayAttr(ArrayRef<int32_t>({})), // [0]ins_w/[1]ins_h
          rewriter.getContext())));
  attrs.push_back(rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));

  return std::make_pair(operands, attrs);
}

void getInterpHWScale(
    const int height1, const int width1,
    const int height2, const int width2,
    float* rheight, float* rwidth) {

  if (height1 == 1) {
    *rheight = height2;
  }
  else {
    *rheight = static_cast<float>(height2 - 1) / (height1 - 1);
  }

  if (width1 == 1) {
    *rwidth = width2;
  }
  else {
    *rwidth = static_cast<float>(width2 - 1) / (width1 - 1);
  }
}

template <typename OpTy>
std::pair<std::vector<Value *>, std::vector<NamedAttribute> > getTileInterp(
    OpTy castOp,
    Operation *op,
    PatternRewriter &rewriter) {

  // input
  std::vector<int64_t> shape;
  int64_t input_size, in, ic, ih, iw;
  getTensorShapeAndSize(castOp.getOperand(0), shape, input_size);
  getNCHW(shape, in, ic, ih, iw);

  // output
  std::vector<int64_t> output_shape;
  int64_t output_size, oh, ow;
  getTensorShapeAndSize(op->getResult(0), output_shape, output_size);
  // x2 means each output determinded by 4 point, e.g.:
  // deconv should copy w axis twise and it should be as below:
  // e.g.:
  // 0 1 2 deconv  0 0 1 1 2 2 w-scale is 2  01 01 12 12 22
  // 3 4 5 ----->  3 3 4 4 5 5 ------------> 34 34 45 45 55
  oh = output_shape[2] * 2;
  ow = output_shape[3] * 2;

  // construct tile_interp
  auto input = castOp.getResult();

  auto NoneOp = OpBuilder(op).create<tpu::NoneOp>(rewriter.getUnknownLoc(),
      rewriter.getNoneType());

  std::vector<Value *> operands;
  operands.push_back(input);
  operands.push_back(NoneOp.getResult()); // bias
  operands.push_back(NoneOp.getResult()); // quant_scale
  operands.push_back(NoneOp.getResult()); // quant_zeropoint
  operands.push_back(NoneOp.getResult()); // quant_rshift
  operands.push_back(NoneOp.getResult()); // quant_multiplier

  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("name", castOp.nameAttr()));

  // assign scale info
  SmallVector<Attribute, 2> padsAttr; // h,w

  float rwidth = 0.f;
  float rheight = 0.f;
  getInterpHWScale(ih, iw, oh, ow, &rheight, &rwidth);

  assert((ceilf(rwidth) == rwidth && floorf(rwidth) == rwidth) && "rwidth should be integer");
  assert((ceilf(rheight) == rheight && floorf(rheight) == rheight) && "rheight should be integer");

  padsAttr.push_back(rewriter.getI32IntegerAttr((int)rwidth));
  padsAttr.push_back(rewriter.getI32IntegerAttr((int)rheight));

  attrs.push_back(rewriter.getNamedAttr("resp",
                            rewriter.getArrayAttr(padsAttr)));
  attrs.push_back(rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));

  return std::make_pair(operands, attrs);
}

/**
 * \brief init weight of interp
 * \data2 weight for interp, the size SHOULD be height2 * width2 * 2 * 2
 * \is_x which means get x or y weight
 */
void fillInterpWeightfilter(float *data2, const int height1, const int width1,
    const int height2, const int width2, bool is_x) {

  const float rheight = (height2 > 1) ? static_cast<float>(height1 - 1) / (height2 - 1) : 0.f;
  const float rwidth = (width2 > 1) ? static_cast<float>(width1 - 1) / (width2 - 1) : 0.f;
  int _width = width2 * 2;
  for (int h2 = 0; h2 < height2; ++h2) {
    const float h1r = rheight * h2;
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const float h1lambda = h1r - h1;
    const float h0lambda = float(1.) - h1lambda;
    for (int w2 = 0; w2 < width2; ++w2) {
      const float w1r = rwidth * w2;
      const int w1 = w1r;
      const int w1p = (w1 < width1 - 1) ? 1 : 0;
      const float w1lambda = w1r - w1;
      const float w0lambda = float(1.) - w1lambda;
      if (is_x) {
        int _h2 = h2 * 2;
        int _w2 = w2 * 2;
        float _w0lambda = w0lambda;
        float _w1lambda = w1lambda;
        if (!w1p) {
          // boundry, swap it
          _w0lambda = w1lambda;
          _w1lambda = w0lambda;
        }
        data2[_h2 * _width+ _w2] = _w0lambda;
        data2[(_h2+1) * _width+ _w2] = _w0lambda;
        data2[_h2 * _width+ _w2+1] = _w1lambda;
        data2[(_h2+1) * _width+ _w2+1] = _w1lambda;
        //printf("set to (%d,%d,%d,%d) -> (%p,%p,%p,%p)\n",
        //    _h2 * _width + _w2,   (1+_h2) * _width + _w2,
        //    _h2 * _width + _w2+1, (_h2+1) * _width + _w2+1,
        //    &data2[_h2 * _width+ _w2], &data2[(_h2+1) * _width+ _w2],
        //    &data2[_h2 * _width+ _w2+1], &data2[(_h2+1) * _width+ _w2+1]);
      }
      else {
        int _h2 = h2 * 2;
        int _w2 = w2;
        _width = width2;
        float _h0lambda = h0lambda;
        float _h1lambda = h1lambda;
        if (!h1p) {
          // boundry, swap it
          _h0lambda = h1lambda;
          _h1lambda = h0lambda;
        }
        data2[_h2 * _width+ _w2] = _h0lambda;
        data2[(_h2+1) * _width+ _w2] = _h1lambda;
      }
    }
  }
}

template <typename OpTy>
std::pair<std::vector<Value *>, std::vector<NamedAttribute> > getHWaxisWeight(
    OpTy castOp,
    int is_x,
    Operation *op,
    PatternRewriter &rewriter) {

  // input
  std::vector<int64_t> shape;
  int64_t input_size, in, ic, ih, iw;
  getTensorShapeAndSize(castOp.getOperand(0), shape, input_size);
  getNCHW(shape, in, ic, ih, iw);

  // output
  std::vector<int64_t> output_shape;
  int64_t output_size, oh, ow;
  getTensorShapeAndSize(op->getResult(0), output_shape, output_size);
  oh = output_shape[2];
  ow = output_shape[3];

  // get weight
  TensorFile *wTF = getWeightTensorFile(op);
  Value *wfV = getWeightFileValue(op);

  // calcuate scale
  float rwidth = 0.f;
  float rheight = 0.f;
  getInterpHWScale(ih, iw, oh, ow, &rheight, &rwidth);

  // init weight size, 2*2 means all output determined by 4 point
  // n/c set 1 to broadcast it
  in = 1;
  ic = 1;
  int filterH = ih * 2;
  int filterW = iw * 2;
  if (!is_x) {
    filterW = iw;
  }
  int64_t filterSize = in * ic * rheight * rwidth * filterH * filterW;
  std::vector<int64_t> filterShape;
  std::vector<float> filter(filterSize, 0);
  filterShape.push_back(in);
  filterShape.push_back(ic);
  filterShape.push_back(filterH);
  filterShape.push_back(filterW);

  fillInterpWeightfilter(filter.data(), ih, iw, oh, ow, is_x);

  // dynamic add weight as input
  auto filterValue = addWeightTensorAndCreateWeightOp<float>(op, "filter",
      filter, filterShape, "NONE", wTF, wfV);

  // construct eltwise
  auto input = castOp.getResult();

  auto NoneOp = OpBuilder(op).create<tpu::NoneOp>(rewriter.getUnknownLoc(),
      rewriter.getNoneType());

  std::vector<Value *> operands;
  operands.push_back(input);
  operands.push_back(filterValue);
  operands.push_back(NoneOp.getResult()); // quant_scale
  operands.push_back(NoneOp.getResult()); // quant_zeropoint
  operands.push_back(NoneOp.getResult()); // quant_rshift
  operands.push_back(NoneOp.getResult()); // quant_multiplier

  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("name", castOp.nameAttr()));
  attrs.push_back(rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));

  return std::make_pair(operands, attrs);
}

template <typename OpTy>
std::pair<std::vector<Value *>, std::vector<NamedAttribute> > getConv(
    OpTy castOp,
    int is_x,
    Operation *op,
    PatternRewriter &rewriter) {

  // input
  std::vector<int64_t> shape;
  int64_t input_size, in, ic, ih, iw;
  getTensorShapeAndSize(castOp.getOperand(0), shape, input_size);
  getNCHW(shape, in, ic, ih, iw);

  // output
  std::vector<int64_t> output_shape;
  int kh = 1;
  int kw = 2;
  int stride_h = 2;
  int stride_w = 2;
  if (is_x == 0) {
    kh = 2;
    kw = 1;
    stride_w = 1;
    stride_h = 2;
  }

  output_shape.push_back(in);
  output_shape.push_back(ic);
  output_shape.push_back(ih);
  output_shape.push_back(iw);

  // get weight
  TensorFile *wTF = getWeightTensorFile(op);
  Value *wfV = getWeightFileValue(op);

  int64_t filterSize = in * ic * kh * kw;
  std::vector<int64_t> filterShape;
  std::vector<float> filter(filterSize, 1);
  int g = ic;
  filterShape.push_back(g); // g
  filterShape.push_back(ic); // oc
  filterShape.push_back(ic);
  filterShape.push_back(kh);
  filterShape.push_back(kw);

  // dynamic add weight as input
  auto filterValue = addWeightTensorAndCreateWeightOp<float>(op, "filter",
      filter, filterShape, "NONE", wTF, wfV);

  // construct eltwise
  auto input = castOp.getResult();

  auto NoneOp = OpBuilder(op).create<tpu::NoneOp>(rewriter.getUnknownLoc(),
      rewriter.getNoneType());

  std::vector<Value *> operands;
  operands.push_back(input);
  operands.push_back(filterValue);
  operands.push_back(NoneOp.getResult()); // bias
  operands.push_back(NoneOp.getResult()); // quant_scale
  operands.push_back(NoneOp.getResult()); // quant_zeropoint
  operands.push_back(NoneOp.getResult()); // quant_rshift
  operands.push_back(NoneOp.getResult()); // quant_multiplier

  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("name", castOp.nameAttr()));

  std::vector<int64_t> dilation(2);
  dilation[0] = dilation[1] = 1;
  int padding = 0;
  int is_dw = 1;
  int with_bias = 0;

  attrs.push_back(rewriter.getNamedAttr("param",
        tpu::ConvParam::get(rewriter.getI32IntegerAttr(stride_h),
          rewriter.getI32IntegerAttr(stride_w),
          rewriter.getStringAttr("VALID"),
          rewriter.getI32IntegerAttr(dilation[0]),
          rewriter.getI32IntegerAttr(dilation[1]),
          rewriter.getI32IntegerAttr(padding),
          rewriter.getI32IntegerAttr(padding),
          rewriter.getI32IntegerAttr(padding),
          rewriter.getI32IntegerAttr(padding),
          rewriter.getI32IntegerAttr(g),
          rewriter.getBoolAttr(is_dw),
          rewriter.getBoolAttr(with_bias),
          rewriter.getBoolAttr(false),
          rewriter.getI32ArrayAttr(ArrayRef<int32_t>({})), // [0]ins_w/[1]ins_h
          rewriter.getContext())));
  attrs.push_back(rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));

  return std::make_pair(operands, attrs);
}

// \floatDividend if gived, it should be find one divisor that the range should be in
// floatDividend < x < 2 * floatDividend
// e.g: getDivisors(32, 5) should be 4 * 8, 5< 8 < 10
std::pair<std::vector<std::pair<int, int> >, int> getDivisors(int n, int floatDividend = 0) {
  std::vector<std::pair<int, int> > divisors;
  int insertMax = 14;
  auto div = [&](int n) mutable -> int {
    // FIXME: depends by hw, 14 is the max size of insert number
    if (n < insertMax) {
      return n; // no need to slice
    }

    for (int i=sqrt(n); i > 1; i--) {
      if (n%i == 0 && i < insertMax)
      {
        return i;
      }
    }
    return 0;
  };

  int maxFloatDividend = 0;
  if (floatDividend) {
    // check possible divisors's range between \floatDividend<x<2*\floatDividend
    int found = 0;
    int i;
    int floatDividendStart = std::min(2 * floatDividend - 1, insertMax);
    for (i = floatDividendStart; i > floatDividend; i--) {
      float is_disivible = n / (float)i;
      if ((ceilf(is_disivible) == is_disivible && floorf(is_disivible) == is_disivible)) {
        found = 1;
        break;
      }
    }

    if (found) {
      n = n / i;
      maxFloatDividend = i;
    }
    else {
      return std::make_pair(divisors, maxFloatDividend);
    }
  }

  while (n != 1) {
    int d = div(n);
    if (!d) {
      divisors.clear();
      break;
    }

    divisors.push_back(std::make_pair(d, 1));
    n = n / d;
  }

  return std::make_pair(divisors, maxFloatDividend);
}

struct TpuMergeInterpToConv2DPattern : public RewritePattern {
  TpuMergeInterpToConv2DPattern(MLIRContext *context)
      : RewritePattern("tpu.interp", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto _interpOp = cast<tpu::InterpOp>(op);
    std::string op_name = _interpOp.name().str();
    LLVM_DEBUG(llvm::errs() << _interpOp.getOperationName() << ":"
                            << getOpName(op)<< "\n";);

    // parse param
    // input
    std::vector<int64_t> shape;
    int64_t input_size, in, ic, ih, iw;
    getTensorShapeAndSize(op->getOperand(0), shape, input_size);
    getNCHW(shape, in, ic, ih, iw);

    // output
    auto resultT = std::make_unique<std::vector<float> >(input_size);
    std::vector<int64_t> output_shape;
    int64_t output_size, oh, ow;
    getTensorShapeAndSize(op->getResult(0), output_shape, output_size);
    oh = output_shape[2];
    ow = output_shape[3];

    if (oh == ih && ow == iw) {
      // no need to do interp, just delete it
      rewriter.eraseOp(op);
      return matchSuccess();
    }


    // replace with conv under \is_shrink is true,
    // otherwise, leverage deconv
    bool is_shrink = true;
    int oc = ic;
    int filter_size = 0;
    int stride_h = 0, stride_w = 0;
    int dilation_h = 1, dilation_w = 1;
    float filter_val = 0;
    std::vector<int64_t> filter_shape;

    auto pad_beg_ = _interpOp.pad_beg().getLimitedValue();
    auto pad_end_ = _interpOp.pad_end().getLimitedValue();
    assert(!pad_beg_ && !pad_end_ && "not support pad_begin/pad_end yet");

    auto shrink_factor = _interpOp.shrink_factor().getLimitedValue();
    auto zoom_factor = _interpOp.zoom_factor().getLimitedValue();
    if (shrink_factor && !zoom_factor) {
      const int shrink_factor = shrink_factor;
      assert(shrink_factor >= 1 && "Shrink factor must be positive");
      filter_size = shrink_factor * shrink_factor;
      filter_val = 1 / (float)filter_size;
      filter_shape = {oc, ic, shrink_factor, shrink_factor};
      stride_h = shrink_factor;
      stride_w = shrink_factor;
    } else if (zoom_factor &&
        !shrink_factor) {
      assert(zoom_factor >= 1 && "Zoom factor must be positive");
    } else if (_interpOp.height().getLimitedValue() && _interpOp.width().getLimitedValue()) {
      if (oh > ih && ow > iw) {
        // zoom
        is_shrink = false;
      }
      else if (oh < ih && ow < iw) {
        // shrink
      }
      else {
        std::string errorMsg = std::string(__func__) + " failed, Op " +
          op->getName().getStringRef().str() +
          " not support ih/iw" + std::to_string(ih) + "/" + std::to_string(iw) +
          ", oh/ow:" + std::to_string(oh) + "/" + std::to_string(ow) + "\n";
        llvm_unreachable(errorMsg.c_str());
      }
    } else if (zoom_factor &&
        shrink_factor) {

      std::string errorMsg = std::string(__func__) + " failed, Op " +
        op->getName().getStringRef().str() +
        " not support yet zoom_factor:" + std::to_string(zoom_factor) +
        "/shrink_factor:" + std::to_string(shrink_factor) + "\n";

      if (zoom_factor > 1 && shrink_factor > 1) {
        // TODO: shrink and zoom

        //height_out_ = (height_in_eff_ - 1) / shrink_factor + 1;
        //width_out_ = (width_in_eff_ - 1) / shrink_factor + 1;
        //height_out_ = height_out_ + (height_out_ - 1) * (zoom_factor - 1);
        //width_out_ = width_out_ + (width_out_ - 1) * (zoom_factor - 1);
        llvm_unreachable(errorMsg.c_str());
      }

      if (zoom_factor == 1) {
        assert(shrink_factor >= 1 && "Shrink factor must be positive");
        filter_size = shrink_factor * shrink_factor;
        //filter_val = 1 / (float)filter_size;
        filter_val = 1; // nearnest
        filter_shape = {oc, ic, (int64_t)shrink_factor, (int64_t)shrink_factor};
        stride_h = shrink_factor;
        stride_w = shrink_factor;
      }
      else if (shrink_factor == 1) {
        // zoom
        is_shrink = false;
        int height_in_eff_ = ih + pad_beg_ + pad_end_;
        int width_in_eff_ = iw + pad_beg_ + pad_end_;
        int height_out_ = height_in_eff_ + (height_in_eff_ - 1) * (zoom_factor - 1);
        int width_out_ = width_in_eff_ + (width_in_eff_ - 1) * (zoom_factor - 1);
        assert(height_out_ == oh && width_out_ == ow &&
            "oh/ow not equal with frontend's");
      }
      else {
        llvm_unreachable(errorMsg.c_str());
      }
    }

    TensorFile *wTF = getWeightTensorFile(op);
    Value *wfV = getWeightFileValue(op);
    auto loc = op->getLoc();

    if (is_shrink) {
      // replace with conv
      int inner_size = filter_size * ic;
      std::vector<float> new_filter(inner_size * oc, 0);
      for (int i = 0; i < oc; ++i) {
        // only fill one channel by oc
        std::fill(new_filter.begin() + i * inner_size + i * filter_size,
            new_filter.begin() + i * inner_size + i * filter_size + 1,
            filter_val); //nearnest
        //std::fill(new_filter.begin() + i * inner_size + i * filter_size,
        //    new_filter.begin() + i * inner_size + (i+1) * filter_size,
        //    filter_val);
      }

      std::vector<std::vector<float> *> newWeights{ &new_filter };
      std::vector<std::vector<int64_t> > weightShapes{ filter_shape };

      std::vector<Value *> newOperands;
      newOperands.push_back(_interpOp.getOperand(0));

      // add new filter and no bias ops
      for (int i = 0; i < 1; ++i) {
        auto tensor_name = op_name + "_conv_" + std::to_string(i);
        LLVM_DEBUG(llvm::errs() << "  new_weight[" << i << "] : "
            << tensor_name << "\n";);
        auto type = RankedTensorType::get(weightShapes[i],
            FloatType::getF32(rewriter.getContext()));
        wTF->addTensor<float>(tensor_name, newWeights[i], type);
        std::vector<NamedAttribute> attrs;
        attrs.push_back(rewriter.getNamedAttr("name",
              rewriter.getStringAttr(tensor_name)));
        auto new_weight_op = rewriter.create<tpu::LoadWeightOp>(loc, type,
            ArrayRef<Value *>{wfV}, ArrayRef<NamedAttribute>{attrs});
        newOperands.push_back(new_weight_op);
      }

      auto NoneOp = rewriter.create<tpu::NoneOp>(
          rewriter.getUnknownLoc(), rewriter.getNoneType());

      newOperands.push_back(NoneOp.getResult());  // bias
      newOperands.push_back(NoneOp.getResult());  // quant_scale
      newOperands.push_back(NoneOp.getResult());  // quant_zeropoint
      newOperands.push_back(NoneOp.getResult());  // quant_rshift
      newOperands.push_back(NoneOp.getResult());  // quant_multiplier

      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr("name",
            rewriter.getStringAttr(op_name)));
      attrs.push_back(rewriter.getNamedAttr("param",
            tpu::ConvParam::get(
              rewriter.getI32IntegerAttr(stride_h),
              rewriter.getI32IntegerAttr(stride_w),
              rewriter.getStringAttr("VALID"), // convOp.param().padding
              rewriter.getI32IntegerAttr(dilation_h),
              rewriter.getI32IntegerAttr(dilation_w),
              rewriter.getI32IntegerAttr(0), //convOp.param().padding_t(),
              rewriter.getI32IntegerAttr(1), //convOp.param().padding_b(),
              rewriter.getI32IntegerAttr(0), //convOp.param().padding_l(),
              rewriter.getI32IntegerAttr(1), //convOp.param().padding_r(),
              rewriter.getI32IntegerAttr(1), //convOp.param().group(),
              rewriter.getBoolAttr(false), //convOp.param().is_dw(),
              rewriter.getBoolAttr(false), //bias
              rewriter.getBoolAttr(false), //convOp.param().do_relu(),
              rewriter.getI32ArrayAttr(ArrayRef<int32_t>({})), // [0]ins_w/[1]ins_h
              rewriter.getContext())));

      attrs.push_back(rewriter.getNamedAttr("quant",
            getDefaultQuantParam(rewriter)));

      rewriter.replaceOpWithNewOp<tpu::Conv2DOp>(
          _interpOp, _interpOp.getResult()->getType(),
          ArrayRef<Value *>{newOperands}, ArrayRef<NamedAttribute>{attrs});
    }
    else {
      int kh = -1;
      int kw = -1;

      // calcuate scale
      float rwidth = 0.f;
      float rheight = 0.f;
      int rwidthInt = 0;
      int rheightInt = 0;
      // keep Dividend / Divisor for later non-divisable
      std::vector<std::pair<int, int> > maxInsertWAtOnce;
      std::vector<std::pair<int, int> > maxInsertHAtOnce;
      // seperate Dividend, Divisor as scale to deal with float case
      // scale[0] as h, scale[1] for w
      // pair is Dividend / Divisor
      SmallVector<std::pair<int, int>, 2> scale = {{0,0}, {0,0}};

      getInterpHWScale(ih, iw, oh, ow, &rheight, &rwidth);
      int floatDividend = 0;
      int maxFloatDividend = 0;

      // deal with non divisable case
      // h
      if ((ceilf(rheight) == rheight && floorf(rheight) == rheight)) {
        // integer case
        rheightInt = int(rheight);
        std::tie(maxInsertHAtOnce, maxFloatDividend) = getDivisors(rheightInt);
      }
      else {
        // 2047 / 63 = 89 * 23 / 7 * 9
        // float case: e.g: 6->33 = 6 * (2/5)
        floatDividend = ih - 1;
        std::tie(maxInsertHAtOnce, maxFloatDividend) = getDivisors(oh - 1, floatDividend);
        if (!maxInsertHAtOnce.size()) {
          // TODO: seperate all divisor
          maxInsertHAtOnce.push_back(std::make_pair(oh, ih - 1));
        }
        scale[0] = (std::make_pair(maxFloatDividend, floatDividend));
      }

      // w
      if ((ceilf(rwidth) == rwidth && floorf(rwidth) == rwidth)) {
        // integer case
        rwidthInt = int(rwidth);
        std::tie(maxInsertWAtOnce, maxFloatDividend) = getDivisors(rwidthInt);
      }
      else {
        // float case: e.g: 6->33 = 6 * (2/5)
        // we seleprate integer part and float part
        // 6->33 = 32 / 5 = 4 * (8/5) = 4x . 8/5
        // 8/5 which means we insert (8-1) and stride = 5
        // NOTICE: float part SHOULD BE 1<x<2
        floatDividend = iw - 1;
        std::tie(maxInsertWAtOnce, maxFloatDividend) = getDivisors(ow - 1, floatDividend);
        if (!maxInsertWAtOnce.size()) {
          // TODO: seperate all divisor
          maxInsertWAtOnce.push_back(std::make_pair(ow, iw - 1));
        }
        scale[1] = (std::make_pair(maxFloatDividend, floatDividend));
      }

      if (!maxInsertHAtOnce.size() && !maxInsertWAtOnce.size()) {
        // TODO: verify it under interpreter
        // zoom, leverage by deconv
        // leverage following steps
        // 0. deconv scale set to 2
        // 1. tile
        // 2. eltwise x + depthwise with kernel shape <1x2> to get R1/R2
        // 3. eltwise y + depthwise with kernel shape <2x1> to get P

        std::vector<Value *> operands;
        std::vector<NamedAttribute> attrs;
        std::tie(operands, attrs) = getTwiceWDeConv(_interpOp, op, rewriter);

        // deconv, we extend w-axis for tile
        // [[1,2], [3,4]] => [[1,1,2,2], [3,3,4,4]]
        auto deconv = rewriter.create<tpu::DeConv2DOp>(loc,
            _interpOp.getOperand(0)->getType(),
            ArrayRef<Value *>{operands},
            ArrayRef<NamedAttribute>{attrs});

        // tile, we tile by scale, e.g
        // 0 1 2  scale 2 with w   01 01 12 12 22
        // 3 4 5  -------------->  34 34 45 45 55
        //                         34 34 45 45 55
        //                         34 34 45 45 55
        std::tie(operands, attrs) = getTileInterp(deconv, op, rewriter);
        auto tile_interp = rewriter.create<tpu::TileInterpOp>(loc,
            deconv.getOperand(0)->getType(),
            ArrayRef<Value *>{operands},
            ArrayRef<NamedAttribute>{attrs});

        // eltwise x + depthwise with kernel shape <1x2> to get R1/R2
        // bilinear intepolation
        //
        // y2 Q12---R2---Q22
        //    |          |
        //    |          |
        // y1 Q11---R1---Q21
        //   x1     x     x2
        //
        // 0 1 2 scale 2 in w  |01| 01 12 12 22   |01|
        // 3 4 5 ------------> |34| 34 45 45 55 ->|34| determine (0,0) of output
        //                      34  34 45 45 55
        //                      34  34 45 45 55
        //
        // R1 = f(x1,y1) + (x - x1)/(x2 - x1)*(f(x2,y1) - f(x1,y1))
        //    = (1 - (x - x1)/(x2 - x1) * f(x1,y1) + (x - x1)/(x2 - x1) * f(x2,y1)
        //    = Wx1y1 * f(x1,y1) + Wx2y1 * f(x2,y1)
        // R2 = f(x1,y2) + (x - x1)/(x2 - x1)*(f(x2,y2) - f(x1,y2))
        //    = Wx1y2 * f(x1,y2) + Wx2y2 * f(x2,y2)
        //
        // offline calulate Wx1y1/Wx2y1/Wx1y2/Wx2y2
        bool is_x = true;
        std::tie(operands, attrs) = getHWaxisWeight(tile_interp, is_x, op, rewriter);
        auto eltMulWaxis = rewriter.create<tpu::EltwiseMulOp>(loc,
            tile_interp.getOperand(0)->getType(),
            ArrayRef<Value *>{operands},
            ArrayRef<NamedAttribute>{attrs});

        std::tie(operands, attrs) = getConv(eltMulWaxis, is_x, op, rewriter);
        auto depthWithConvW = rewriter.create<tpu::Conv2DOp>(loc,
            tile_interp.getOperand(0)->getType(),
            ArrayRef<Value *>{operands},
            ArrayRef<NamedAttribute>{attrs});
        (void)depthWithConvW; // why defined but not used?
        // y accumulate
        is_x = false;
        std::tie(operands, attrs) = getHWaxisWeight(tile_interp, is_x, op, rewriter);
        auto eltMulHaxis = rewriter.create<tpu::EltwiseMulOp>(loc,
            tile_interp.getOperand(0)->getType(),
            ArrayRef<Value *>{operands},
            ArrayRef<NamedAttribute>{attrs});

        std::tie(operands, attrs) = getConv(eltMulHaxis, is_x, op, rewriter);
        auto depthWithConvH = rewriter.create<tpu::Conv2DOp>(loc,
            tile_interp.getOperand(0)->getType(),
            ArrayRef<Value *>{operands},
            ArrayRef<NamedAttribute>{attrs});

        rewriter.replaceOp(op, {depthWithConvH});
      }
      else {

        // construct conv with insert/padding
        auto input = op->getOperand(0);
        auto input_type = input->getType().cast<RankedTensorType>();
        auto input_shape = input_type.getShape();
        int g = input_shape[1]; // g == ic for depthwise

        auto NoneOp = OpBuilder(op).create<tpu::NoneOp>(rewriter.getUnknownLoc(),
            rewriter.getNoneType());

        int _ic = ic;
        int _oc = oc;
        int _ih = ih;
        int _iw = iw;
        int is1x1Input = ih == 1 && ih == iw;
        int loop = std::max(maxInsertHAtOnce.size(), maxInsertWAtOnce.size());

        auto calc_dilute_hw = [&](int h, int ins_h, int ins_h_l, int pad_h_b, int pad_h_t) mutable -> int {
          return (h - 1) * (ins_h + 1) + ins_h_l +
            1 + pad_h_t + pad_h_b;
        };

        auto calc_output_hw = [&](int hw, int khw, int stride) mutable -> int {
          return (hw - khw)/stride + 1;
        };

        auto createConvAttr = [&](
          std::vector<int64_t> kernel, std::vector<int64_t> stride,
          std::vector<int64_t> dilation, std::vector<int64_t> padding,
          int g, bool is_dw, bool with_bias, std::vector<int32_t> ins) mutable ->
          std::vector<NamedAttribute> {
            std::vector<NamedAttribute> attrs;
            attrs.push_back(rewriter.getNamedAttr("param",
                  tpu::ConvParam::get(rewriter.getI32IntegerAttr(stride[0]),
                    rewriter.getI32IntegerAttr(stride[1]),
                    rewriter.getStringAttr("VALID"),
                    rewriter.getI32IntegerAttr(dilation[0]),
                    rewriter.getI32IntegerAttr(dilation[1]),
                    rewriter.getI32IntegerAttr(padding[0]), // top
                    rewriter.getI32IntegerAttr(padding[1]), // bottom
                    rewriter.getI32IntegerAttr(padding[2]),
                    rewriter.getI32IntegerAttr(padding[3]),
                    rewriter.getI32IntegerAttr(g),
                    rewriter.getBoolAttr(is_dw),
                    rewriter.getBoolAttr(with_bias),
                    rewriter.getBoolAttr(false),
                    rewriter.getI32ArrayAttr(ArrayRef<int32_t>({ins})), // [0]ins_w/[1]ins_h
                    rewriter.getContext())));
            return attrs;
          };

        auto createConv2D = [&](Value* input, int d, bool isNonDivisible = false) mutable ->
          std::tuple<std::vector<Value *>, std::vector<NamedAttribute>, RankedTensorType > {

          if (_ih == 1 || _iw == 1) {
            assert(_iw == _ih && "not support asymmetrical under _ih = 1 or _iw = 1");
          }

          rheightInt = 1;
          rwidthInt = 1;
          std::vector<int64_t> kernel(2), stride(2), dilation(2), padding(4);
          std::vector<int32_t> ins(2), _ins(2);

          // TODO: support d not integer case, e.g: d = 1.3
          stride[0] = 1; // sh
          stride[1] = 1; // sw

          if (isNonDivisible) {
            if (scale[0].first) {
              std::tie(rheightInt, stride[0]) = scale[0];
            }

            if (scale[1].first) {
              std::tie(rwidthInt, stride[1]) = scale[1];
            }
          }
          else {
            int divisor, dividend;
            if (d < (int)maxInsertHAtOnce.size()) { // star with 0
              std::tie(dividend, divisor) = maxInsertHAtOnce[d];
              float rheight = dividend / (float)divisor;
              if ((ceilf(rheight) == rheight && floorf(rheight) == rheight)) {
                rheightInt = rheight;//divisible
              }
              else {
                stride[0] = divisor; // sh
                rheightInt = dividend - 1; // hw ins_w
              }
            }

            if (d < (int)maxInsertWAtOnce.size()) { // star with 0
              std::tie(dividend, divisor) = maxInsertWAtOnce[d];
              float rwidth = dividend / (float)divisor;
              if ((ceilf(rwidth) == rwidth && floorf(rwidth) == rwidth)) {
                rwidthInt = rwidth; //divisible
              }
              else {
                stride[1] = divisor;
                rwidthInt = dividend - 1; // hw ins_w
              }
            }
          }

          // init parameter
          kh = (rheightInt - 1) * 2 + 1;
          kw = (rwidthInt - 1) * 2 + 1;
          bool is_dw = true;
          bool with_bias = false;
          kernel[0] = kh;
          kernel[1] = kw;
          dilation[0] = dilation[1] = 1;

          ins[0] = rwidthInt - 1; // hw ins_w
          ins[1] = rheightInt - 1; // hw ins_h
          _ins = ins;

          padding[0] = padding[1] = rheightInt - 1; // padding top/bottom
          padding[2] = padding[3] = rwidthInt - 1; // padding left/right

          // depthwise case
          _oc = 1;
          _ic = 1;

          if (is1x1Input) {
            kh = rheightInt;
            kw = rwidthInt;
            stride[0] = kh;
            stride[1] = kw;
            padding[2] = padding[3] = padding[0] = padding[1] = 0;

            ins.clear();
          }

          // init filter
          int count = g * _ic * _oc * kh * kw; // depthewise, ic == oc
          std::vector<float> filter(count, 1);
          std::vector<int64_t> filter_shape;

          if (is1x1Input) {
            // default fill to 1
          }
          else {
            // fill filter from corner
#if 1
            for (int i = 0; i < kh / 2 + 1; i++) {
              for (int j = 0; j < kw / 2 + 1; j++) {
                float f = (i + 1) * (j + 1) / float(rheightInt * rwidthInt);
                filter.data()[i * kw + j] = f;
                filter.data()[i * kw + (kw-1)-j] = f;
                filter.data()[(kh-1-i) * kw + j] = f;
                filter.data()[(kh-1-i) * kw + (kw-1)-j] = f;
              }
            }
#else
            for (int i = 0; i < kh / 2; i++) {
              int idx = 1;
              for (int j = 0; j < kw / 2; j++) {
                filter.data()[i * kw + j] = ((i + 1) * idx) / float(rwidthInt * rheightInt);
                filter.data()[i * kw + (kw-1)-j] = filter.data()[i * kw + j];
                filter.data()[(kh-1-i) * kw + j] = filter.data()[i * kw + j];
                filter.data()[(kh-1-i) * kw + (kw-1)-j] = filter.data()[i * kw + j];
                idx++;
              }
            }

            // fill center with vertical
            int idx = 1;
            for (int i = 0; i < kh / 2; i++) {
              for (int j = kw / 2; j < kw / 2 + 1; j++) {
                // kernel should the same
                filter.data()[i * kw + j] = (idx) / float(rheightInt);
                filter.data()[(kh-1-i) * kw + j] = filter.data()[i * kw + j];
              }
              idx++;
            }

            // fill center with horizontal
            idx = 1;
            for (int i = kh / 2; i < kh / 2 + 1; i++) {
              for (int j = 0; j < kw / 2 + 1; j++) {
                // kernel should the same
                filter.data()[i * kw + j] = (idx) / float(rwidthInt);
                filter.data()[i * kw + (kw-1)-j] = filter.data()[i * kw + j];
                idx++;
              }
            }

            // fill center
            filter.data()[(kh / 2) * kw + kw / 2] = 1;
#endif
            // duplicate to ic oc g
            int j = _ic * _oc * g;
            int khw = kh * kw;
            for (int i = 1; i < j; i++) {
              std::copy(filter.data(), filter.data() + khw,
                  filter.data() + i * khw);
            }
          }

          if (g != 1) {
            filter_shape.push_back(g);
          }

          filter_shape.push_back(_oc);
          filter_shape.push_back(_ic);
          filter_shape.push_back(kh);
          filter_shape.push_back(kw);

          // prepare filter
          auto filterValue = addWeightTensorAndCreateWeightOp<float>(op, std::to_string(d) + "_filter",
              filter, filter_shape, "NONE", wTF, wfV);

          // it could Dilated in activation in hw once `ins` set
          // the final output should be input->Dilated(ins_w/ins_h)->conv
          std::vector<int64_t> top_dim(2); // [0] is h, [1] is w
          _oc = ic; //depthwise case
          std::string prefix = llvm::formatv("_{0}_", std::to_string(d)).str();

          if (!is1x1Input) {
            // to check memory usage per lane
            // create fake op for check
            std::vector<Value *> operands;
            operands.push_back(input);
            operands.push_back(filterValue);
            operands.push_back(NoneOp.getResult()); // bias
            operands.push_back(NoneOp.getResult()); // quant_scale
            operands.push_back(NoneOp.getResult()); // quant_zeropoint
            operands.push_back(NoneOp.getResult()); // quant_rshift
            operands.push_back(NoneOp.getResult()); // quant_multiplier

            std::vector<NamedAttribute> attrs =
              createConvAttr(kernel, stride, dilation, padding, g, is_dw, with_bias, ins);
            attrs.push_back(rewriter.getNamedAttr("name",
                rewriter.getStringAttr("fakeop")));

            int ih_ext = calc_dilute_hw(_ih, ins[1], 0, padding[0], padding[1]);
            int iw_ext = calc_dilute_hw(_iw, ins[0], 0, padding[2], padding[3]);
            top_dim[0] = calc_output_hw(ih_ext, kh, stride[0]); // oh
            top_dim[1] = calc_output_hw(iw_ext, kw, stride[1]); // ow
            RankedTensorType dilateOutput = RankedTensorType::get(
                {in, _oc, top_dim[0], top_dim[1]},
                input_type.getElementType());

            auto fakeOp = rewriter.create<tpu::Conv2DOp>(loc,
                dilateOutput,
                ArrayRef<Value *>{operands},
                ArrayRef<NamedAttribute>{attrs});

            // FIXME: no need init every travel
            std::string getRunChipType = "cv183x";
            MInfo Machineinfo;
            //get_cvichip_name(getRunChipType); FIXME: get chip info
            Machineinfo.getChipInfo(getRunChipType.c_str());
            uint64_t totalPerLane = SimpleConv2DMemoryUsageAnalysis(fakeOp, NULL);

            // depthwse with ins SHOULD not slice h/w
            // just slice ic, <n, ic, ih, iw> -> <n, 1, ih, iw>
            int chunkPerLane = (ic + MInfo::lane_num) / MInfo::lane_num;
            // TODO: slice h/w under ins case
            bool isInsInConv = false;
            if (!isInsInConv || totalPerLane / chunkPerLane > MInfo::lmem_per_lane) {
              // if lmem not enough
              LLVM_DEBUG(llvm::errs() << _interpOp.nameAttr().getValue() <<
                  ", lmem not enough, dynamic add dilate op\n");

              // create dilateOp if need
              top_dim[0] = calc_dilute_hw(_ih, ins[1], 0, 0, 0);
              top_dim[1] = calc_dilute_hw(_iw, ins[0], 0, 0, 0);

              // init output
              RankedTensorType output = RankedTensorType::get(
                  {in, _oc, top_dim[0], top_dim[1]},
                  input_type.getElementType());

              // init input
              operands.clear();
              operands.push_back(input);

              // init attr
              std::vector<NamedAttribute> attrs;
              attrs.push_back(rewriter.getNamedAttr("ins",
                  rewriter.getI32ArrayAttr(ArrayRef<int32_t>({ins}))));// [0]ins_w/[1]ins_h
              attrs.push_back(rewriter.getNamedAttr("name",
                    rewriter.getStringAttr(prefix + "_dilate_" + _interpOp.nameAttr().getValue().str())));
              attrs.push_back(rewriter.getNamedAttr("fill_constant",
                    rewriter.getI32IntegerAttr(0))); // default insert 0
              attrs.push_back(
                  rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));

              auto dilateOp = rewriter.create<tpu::DilateOp>(loc,
                  output,
                  ArrayRef<Value *>{operands},
                  ArrayRef<NamedAttribute>{attrs});
              input = dilateOp.getResult();
              ins = {0, 0}; // no dilate in conv
            }
          }

          // prepare output operands
          std::vector<Value *> operands;
          operands.push_back(input);
          operands.push_back(filterValue);
          operands.push_back(NoneOp.getResult()); // bias
          operands.push_back(NoneOp.getResult()); // quant_scale
          operands.push_back(NoneOp.getResult()); // quant_zeropoint
          operands.push_back(NoneOp.getResult()); // quant_rshift
          operands.push_back(NoneOp.getResult()); // quant_multiplier


          // prepare attr
#if 0
          std::vector<NamedAttribute> attrs;
          attrs.push_back(rewriter.getNamedAttr("param",
                tpu::ConvParam::get(rewriter.getI32IntegerAttr(stride[0]),
                  rewriter.getI32IntegerAttr(stride[1]),
                  rewriter.getStringAttr("VALID"),
                  rewriter.getI32IntegerAttr(dilation[0]),
                  rewriter.getI32IntegerAttr(dilation[1]),
                  rewriter.getI32IntegerAttr(padding[0]), // top
                  rewriter.getI32IntegerAttr(padding[1]), // bottom
                  rewriter.getI32IntegerAttr(padding[2]),
                  rewriter.getI32IntegerAttr(padding[3]),
                  rewriter.getI32IntegerAttr(g),
                  rewriter.getBoolAttr(is_dw),
                  rewriter.getBoolAttr(with_bias),
                  rewriter.getBoolAttr(false),
                  rewriter.getI32ArrayAttr(ArrayRef<int32_t>({ins})), // [0]ins_w/[1]ins_h
                  rewriter.getContext())));
#else
            std::vector<NamedAttribute> attrs =
              createConvAttr(kernel, stride, dilation, padding, g, is_dw, with_bias, ins);
#endif

          if (loop - 1 == d) {
            // last one replace the interp name for compare
            prefix = "";
          }

          attrs.push_back(rewriter.getNamedAttr("name",
                rewriter.getStringAttr(prefix + _interpOp.nameAttr().getValue().str())));
          attrs.push_back(
              rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));

          // prepare output shape
          if (is1x1Input) {
            // upsample
            top_dim[0] = _ih * stride[0];
            top_dim[1] = _iw * stride[1];
          } else {
            int ih_ext = calc_dilute_hw(_ih, _ins[1], 0, padding[0], padding[1]);
            int iw_ext = calc_dilute_hw(_iw, _ins[0], 0, padding[2], padding[3]);
            top_dim[0] = calc_output_hw(ih_ext, kh, stride[0]); // oh
            top_dim[1] = calc_output_hw(iw_ext, kw, stride[1]); // ow
          }

          auto input_type = input->getType().cast<RankedTensorType>();
          RankedTensorType output = RankedTensorType::get(
              {in, _oc, top_dim[0], top_dim[1]},
              input_type.getElementType());

          return std::make_tuple(operands, attrs, output);
        };

        // recursive add

        tpu::DeConv2DOp deconv2d;
        tpu::Conv2DOp conv2d;
        int d_start = -1;
        if (scale[0].first != 0 || scale[1].first != 0) {
          d_start = loop;
          loop++; // put later for accuracy
        }

        for (int d = 0; d < loop; d++) {
          std::vector<Value *> operands;
          std::vector<NamedAttribute> attrs;
          RankedTensorType output;
          std::tie(operands, attrs, output) = createConv2D(input, d, d == d_start);

          if (is1x1Input) {
            // just deconv(upsample) it
            deconv2d = rewriter.create<tpu::DeConv2DOp>(loc,
                output,
                ArrayRef<Value *>{operands},
                ArrayRef<NamedAttribute>{attrs});
            input = deconv2d.getResult();
          }
          else {
            conv2d = rewriter.create<tpu::Conv2DOp>(loc,
                output,
                ArrayRef<Value *>{operands},
                ArrayRef<NamedAttribute>{attrs});
            input = conv2d.getResult();
          }

          // intpu as previous output
          auto input_type = input->getType().cast<RankedTensorType>();
          auto input_shape = input_type.getShape();
          _ih = input_shape[2]; // next input's shape
          _iw = input_shape[3];
        }


        {
          // interp's output SHOULD BE EQ with conv's
          std::vector<int64_t> conv_output_shape;
          int64_t conv_on, conv_oc, conv_oh, conv_ow;
          int64_t conv_output_size;
          getTensorShapeAndSize(input, conv_output_shape, conv_output_size);
          getNCHW(conv_output_shape, conv_on, conv_oc, conv_oh, conv_ow);
          assert((conv_on == in &&
                conv_oc == oc &&
                conv_oh == oh &&
                conv_ow == ow) && "Transformsed conv shape SHOULD be equal with interp");
        }

        if (is1x1Input) {
          rewriter.replaceOp(_interpOp, {deconv2d});
        }
        else {
          rewriter.replaceOp(_interpOp, {conv2d});
        }
      }
    }

    return matchSuccess();
  }
};

class ConvertInterpToConvDeconvPass : public FunctionPass<ConvertInterpToConvDeconvPass> {
public:
  explicit ConvertInterpToConvDeconvPass(llvm::raw_ostream &os = llvm::errs())
      : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<TpuMergeInterpToConv2DPattern>(context);
    applyPatternsGreedily(fn, patterns);
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

void tpu::Conv2DOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<TpuMergeSwapChannelToConv2DPattern,
                 TpuConvertDilationWeightPattern >(context);
}

void tpu::DeConv2DOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {}

void tpu::InterpOp::getCanonicalizationPatterns(
         OwningRewritePatternList &results,
         MLIRContext *context) {
  results.insert<TpuMergeInterpToConv2DPattern>(context);
}

static PassRegistration<ConvertInterpToConvDeconvPass>
    pass("convert-intep-to-conv",
         "Convert a interp operation to conv/decon");
