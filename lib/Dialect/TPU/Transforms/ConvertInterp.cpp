//===- ConvertInterp.cpp - convert Conv2D----------------------------------===//
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

// \floatDividend if gived, it should be find one divisor that the range should be in
// floatDividend < x < 2 * floatDividend
// e.g: getDivisors(32, 5) should be 4 * 8, 5< 8 < 10
std::pair<std::vector<std::pair<int, int> >, int> getDivisors(int n, int floatDividend = 0,
    bool isInsInConv = true) {
  std::vector<std::pair<int, int> > divisors;
  int insertMax = 14;
  if (!isInsInConv) {
    insertMax = n; // directly use dilate op
  }
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

  LogicalResult matchAndRewrite(Operation *op,
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
    std::string ctd = _interpOp.coordinate_transformation_mode().str();
    if (ctd != "align_corners"){
      return failure();
    }
    if (oh == ih && ow == iw) {
      // no need to do interp, just delete it
      rewriter.eraseOp(op);
      return success();
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

    auto pad_beg_ = _interpOp.pad_beg();
    auto pad_end_ = _interpOp.pad_end();
    assert(!pad_beg_ && !pad_end_ && "not support pad_begin/pad_end yet");

    auto shrink_factor = _interpOp.shrink_factor();
    auto zoom_factor = _interpOp.zoom_factor();
    if (shrink_factor && !zoom_factor) {
      assert(shrink_factor >= 1 && "Shrink factor must be positive");
      filter_size = shrink_factor * shrink_factor;
      filter_val = 1 / (float)filter_size;
      filter_shape = {oc, ic, shrink_factor, shrink_factor};
      stride_h = shrink_factor;
      stride_w = shrink_factor;
    } else if (zoom_factor &&
        !shrink_factor) {
      assert(zoom_factor >= 1 && "Zoom factor must be positive");
    } else if (_interpOp.height() && _interpOp.width()) {
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
    Value wfV = getWeightFileValue(op);
    auto loc = op->getLoc();

    // calcuate scale
    float rwidth = 0.f;
    float rheight = 0.f;
    int rwidthInt = 0;
    int rheightInt = 0;

    if (is_shrink) {
      getInterpHWScale(oh, ow, ih, iw, &rheight, &rwidth);
      if ((ceilf(rheight) == rheight && floorf(rheight) == rheight)
          && ((ceilf(rwidth) == rwidth && floorf(rwidth) == rwidth))) {
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

        std::vector<Value> newOperands;
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
              ArrayRef<Value>{wfV}, ArrayRef<NamedAttribute>{attrs});
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
                rewriter.getI32IntegerAttr(shrink_factor),
                rewriter.getI32IntegerAttr(shrink_factor),
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
                rewriter.getI32IntegerAttr(0), //pad_value
                rewriter.getContext())));

        attrs.push_back(rewriter.getNamedAttr("quant",
              getDefaultQuantParam(rewriter)));

        rewriter.replaceOpWithNewOp<tpu::Conv2DOp>(
            _interpOp, _interpOp.getResult().getType(),
            ArrayRef<Value>{newOperands}, ArrayRef<NamedAttribute>{attrs});
      }
      else {
        LLVM_DEBUG(llvm::errs() << " not support shrink from oh/ow("
            << oh << "/" << ow << ") to ih/iw("
            << ih << "/" << iw << ") " << op_name << "\n";);
        return failure();
      }
    }
    else {
      int kh = -1;
      int kw = -1;

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

      // TODO: slice h/w under ins case
      bool isInsInConv = false;

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
          // TODO: merge info into scale
          std::vector<std::pair<int, int> > ohDivisors;
          std::vector<std::pair<int, int> > ihDivisors;
          std::tie(ohDivisors, maxFloatDividend) = getDivisors(oh - 1, 0, isInsInConv);
          std::tie(ihDivisors, maxFloatDividend) = getDivisors(ih - 1, 0, isInsInConv);
          if (!ohDivisors.size()) {
            ohDivisors.push_back(std::make_pair(oh - 1, 1));
          }
          if (!ihDivisors.size()) {
            ihDivisors.push_back(std::make_pair(ih - 1, 1));
          }
          auto maxCount = std::max(ohDivisors.size(), ihDivisors.size());
          for (int i = 0; i < (int)maxCount; i++) {
            int ohDivisor = i < (int)ohDivisors.size() ? ohDivisors[i].first : 1;
            int ihDivisor = i < (int)ihDivisors.size() ? ihDivisors[i].first : 1;
            maxInsertHAtOnce.push_back(std::make_pair(ohDivisor, ihDivisor));
          }
        }
        else {
          scale[0] = (std::make_pair(maxFloatDividend, floatDividend));
        }
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
          std::vector<std::pair<int, int> > owDivisors;
          std::vector<std::pair<int, int> > iwDivisors;
          std::tie(owDivisors, maxFloatDividend) = getDivisors(ow - 1, 0, isInsInConv);
          std::tie(iwDivisors, maxFloatDividend) = getDivisors(iw - 1, 0, isInsInConv);
          if (!owDivisors.size()) {
            owDivisors.push_back(std::make_pair(ow - 1, 1));
          }
          if (!iwDivisors.size()) {
            iwDivisors.push_back(std::make_pair(iw - 1, 1));
          }
          auto maxCount = std::max(owDivisors.size(), iwDivisors.size());
          for (int i = 0; i < (int)maxCount; i++) {
            int owDivisor = i < (int)owDivisors.size() ? owDivisors[i].first : 1;
            int iwDivisor = i < (int)iwDivisors.size() ? iwDivisors[i].first : 1;
            maxInsertWAtOnce.push_back(std::make_pair(owDivisor, iwDivisor));
          }
        }
        else {
          scale[1] = (std::make_pair(maxFloatDividend, floatDividend));
        }
      }

      // construct conv with insert/padding
      auto input = op->getOperand(0);
      auto input_type = input.getType().cast<RankedTensorType>();
      auto input_shape = input_type.getShape();
      int g = input_shape[1]; // g == ic for depthwise

      auto NoneOp = OpBuilder(op).create<tpu::NoneOp>(rewriter.getUnknownLoc(),
          rewriter.getNoneType());

      int _ic = ic;
      int _oc = oc;
      int _ih = ih;
      int _iw = iw;
      // NOTICE: 1x1 ALWAYS fill the same value
      int is1x1Input = ih == 1 && ih == iw;
      if (is1x1Input && (!maxInsertHAtOnce.size() || !maxInsertWAtOnce.size())) {
        // deeplabv3_mobilenetv2 case
        // 1x1->46x80 case that 46 seperate 2x23 and the limitation of dilate
        // is 15, replace with upsample case(tiu copy)
        std::vector<NamedAttribute> attrs;
        std::vector<Value> operands;
        attrs.push_back(
            rewriter.getNamedAttr("name", _interpOp.nameAttr()));
        attrs.push_back(
            rewriter.getNamedAttr("scale_h", rewriter.getI32IntegerAttr(oh)));
        attrs.push_back(
            rewriter.getNamedAttr("scale_w", rewriter.getI32IntegerAttr(ow)));
        attrs.push_back(
            rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
        operands.push_back(input);
        operands.push_back(NoneOp.getResult());
        auto upsample = rewriter.create<tpu::UpsampleOp>(
            op->getLoc(), op->getResult(0).getType().cast<RankedTensorType>(), operands, attrs);
        rewriter.replaceOp(op, {upsample});
        return success();
      }

      // check for hw spec, ins/stride range is 0-15 in 1835
      for (auto h_ins_stride : maxInsertHAtOnce) {
        int stride, ins;
        std::tie(ins, stride) = h_ins_stride;
        if (ins > 15 || stride > 15) {
          LLVM_DEBUG(llvm::errs() << "h-params over hardware limitation, leverage cpu,"
              << "ins/stride is:" << ins << "/" << stride << "\n";);
          return failure();
        }
      }

      for (auto w_ins_stride : maxInsertWAtOnce) {
        int stride, ins;
        std::tie(ins, stride) = w_ins_stride;
        if (ins > 15 || stride > 15) {
          LLVM_DEBUG(llvm::errs() << "w-params over hardware limitation, leverage cpu,"
              << "ins/stride is:" << ins << "/" << stride << "\n";);
          return failure();
        }
      }

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
                tpu::ConvParam::get(
                  rewriter.getI32IntegerAttr(kernel[0]),
                  rewriter.getI32IntegerAttr(kernel[1]),
                  rewriter.getI32IntegerAttr(stride[0]),
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
                  rewriter.getI32IntegerAttr(0), //pad_value
                  rewriter.getContext())));
          return attrs;
        };

      auto createConv2D = [&](Value input, int d, bool isNonDivisible = false) mutable ->
        std::tuple<std::vector<Value>, std::vector<NamedAttribute>, RankedTensorType > {

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
                rheightInt = dividend; // hw ins_w
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
                rwidthInt = dividend; // hw ins_w
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
            for (int i = 0; i < kh / 2 + 1; i++) {
              for (int j = 0; j < kw / 2 + 1; j++) {
                float f = (i + 1) * (j + 1) / float(rheightInt * rwidthInt);
                filter.data()[i * kw + j] = f;
                filter.data()[i * kw + (kw-1)-j] = f;
                filter.data()[(kh-1-i) * kw + j] = f;
                filter.data()[(kh-1-i) * kw + (kw-1)-j] = f;
              }
            }

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
            std::vector<Value> operands;
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
                ArrayRef<Value>{operands},
                ArrayRef<NamedAttribute>{attrs});

            // FIXME: no need init every travel
            MInfo::getChipInfo("cv183x");
            assert(MInfo::version && "refer to set-chip");
            uint64_t totalPerLane = SimpleConv2DMemoryUsageAnalysis(fakeOp, NULL);

            // depthwse with ins SHOULD not slice h/w
            // just slice ic, <n, ic, ih, iw> -> <n, 1, ih, iw>
            int chunkPerLane = (ic + MInfo::lane_num) / MInfo::lane_num;
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
                  ArrayRef<Value>{operands},
                  ArrayRef<NamedAttribute>{attrs});
              input = dilateOp.getResult();
              ins = {0, 0}; // no dilate in conv
            }
          }

          // prepare output operands
          std::vector<Value> operands;
          operands.push_back(input);
          operands.push_back(filterValue);
          operands.push_back(NoneOp.getResult()); // bias
          operands.push_back(NoneOp.getResult()); // quant_scale
          operands.push_back(NoneOp.getResult()); // quant_zeropoint
          operands.push_back(NoneOp.getResult()); // quant_rshift
          operands.push_back(NoneOp.getResult()); // quant_multiplier


          // prepare attr
          std::vector<NamedAttribute> attrs =
            createConvAttr(kernel, stride, dilation, padding, g, is_dw, with_bias, ins);

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

          auto input_type = input.getType().cast<RankedTensorType>();
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
        std::vector<Value> operands;
        std::vector<NamedAttribute> attrs;
        RankedTensorType output;
        std::tie(operands, attrs, output) = createConv2D(input, d, d == d_start);

        if (is1x1Input) {
          // just deconv(upsample) it
          deconv2d = rewriter.create<tpu::DeConv2DOp>(loc,
              output,
              ArrayRef<Value>{operands},
              ArrayRef<NamedAttribute>{attrs});
          input = deconv2d.getResult();
        }
        else {
          conv2d = rewriter.create<tpu::Conv2DOp>(loc,
              output,
              ArrayRef<Value>{operands},
              ArrayRef<NamedAttribute>{attrs});
          input = conv2d.getResult();
        }

        // intpu as previous output
        auto input_type = input.getType().cast<RankedTensorType>();
        auto input_shape = input_type.getShape();
        _ih = input_shape[2]; // next input's shape
        _iw = input_shape[3];

        int hardwareHWMax = 4095 - 32;

        std::vector<int64_t> curr_output_shape;
        int64_t curr_output_size;
        getTensorShapeAndSize(operands[0], curr_output_shape, curr_output_size);
        if (curr_output_shape[2] > hardwareHWMax || curr_output_shape[3] > hardwareHWMax) {
          LLVM_DEBUG(llvm::errs() << "hw over hardware limitation, leverage cpu, hw is:"
              << curr_output_shape[2] << "/" << curr_output_shape[3] << "\n";);
          return failure();
        }
      }

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

      if (is1x1Input) {
        rewriter.replaceOp(_interpOp, {deconv2d});
      }
      else {
        rewriter.replaceOp(_interpOp, {conv2d});
      }
    }

    return success();
  }
};

class ConvertInterpToConvDeconvPass : public mlir::PassWrapper<ConvertInterpToConvDeconvPass, FunctionPass> {
public:
  explicit ConvertInterpToConvDeconvPass(llvm::raw_ostream &os = llvm::errs())
      : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<TpuMergeInterpToConv2DPattern>(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

void tpu::InterpOp::getCanonicalizationPatterns(
         OwningRewritePatternList &results,
         MLIRContext *context) {
  results.insert<TpuMergeInterpToConv2DPattern>(context);
}
