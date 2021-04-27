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

template <typename OpTy>
struct MergeConvPadReluPattern : public RewritePattern {
  MergeConvPadReluPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {

    auto convOp = dyn_cast_or_null<tpu::Conv2DOp>(op);
    if (!convOp)
      return failure();

    auto nextOp = getNextOp(convOp);
    if (!(nextOp && isa<tpu::PadOp>(nextOp)))
      return failure();

    auto nextnextOp = getNextOp(nextOp);
    if (!(nextnextOp && isa<tpu::ReluOp>(nextnextOp)))
      return failure();

    auto padOp = dyn_cast_or_null<tpu::PadOp>(nextOp);
    auto reluOp = dyn_cast_or_null<tpu::ReluOp>(nextnextOp);

    bool is_dw, with_bias, do_relu;
    int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw;
    int pt, pb, pl, pr, dh, dw, pad_value;
    parseConvParam(convOp.param(), false, convOp.input(), convOp.output(),
                   convOp.filter(), n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw,
                   pt, pb, pl, pr, dh, dw, is_dw, with_bias, do_relu,
                   pad_value);


    std::vector<int32_t> pads;
    arrayAttrToVector(padOp.pads().getValue(), pads);

    if (!((pt == 0) && (pb == 0)))
      return failure();

    // check if conv with pad from padOp can make ih == oh, iw == ow
    int cal_oh = ( ih + pads[2] + pads[6] - kh) / sh + 1;
    int cal_ow = ( iw + pads[3] + pads[7] - kw) / sw + 1;
    // if success, we can merge pad with conv
    if (!(cal_oh == ih && cal_ow == iw))
      return failure();

    std::vector<NamedAttribute> attrs;
    std::vector<Value> operands;
    for(uint i = 0; i < convOp.getNumOperands(); i++)
      operands.push_back(convOp.getOperand(i));

    attrs.push_back(rewriter.getNamedAttr("name",
                  convOp.nameAttr()));
    attrs.push_back(rewriter.getNamedAttr("param",
    tpu::ConvParam::get(
        rewriter.getI32IntegerAttr(sh),
        rewriter.getI32IntegerAttr(sw),
        rewriter.getStringAttr("VALID"),
        rewriter.getI32IntegerAttr(dh),
        rewriter.getI32IntegerAttr(dw),
        rewriter.getI32IntegerAttr(pads[2]), // pd_t
        rewriter.getI32IntegerAttr(pads[6]), // pd_b
        rewriter.getI32IntegerAttr(pads[3]), // pd_l
        rewriter.getI32IntegerAttr(pads[7]), // pd_r
        rewriter.getI32IntegerAttr(g),
        rewriter.getBoolAttr(is_dw),
        rewriter.getBoolAttr(with_bias),
        rewriter.getBoolAttr(true),
        rewriter.getI32ArrayAttr(ArrayRef<int32_t>({})), // [0]ins_w/[1]ins_h
        rewriter.getI32IntegerAttr(0), //pad_value
        rewriter.getContext())));
    attrs.push_back(
        rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));

    auto newConvOp = rewriter.create<tpu::Conv2DOp>(
                      reluOp.getLoc(), reluOp->getResult(0).getType(),
                      ArrayRef<Value>{operands},
                      ArrayRef<NamedAttribute>{attrs});

    reluOp.getResult().replaceAllUsesWith(newConvOp.getResult());
    rewriter.eraseOp(reluOp);
    rewriter.eraseOp(padOp);
    rewriter.eraseOp(convOp);

    return success();
  }
};

template <typename OpTy>
struct MergeConvReluPattern : public RewritePattern {
  MergeConvReluPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {

    auto convOp = dyn_cast_or_null<tpu::Conv2DOp>(op);
    if (!convOp)
      return failure();

    auto nextOp = getNextOp(convOp);
    if (!(nextOp && isa<tpu::PadOp>(nextOp)))
      return failure();

    auto nextnextOp = getNextOp(nextOp);
    if (!(nextnextOp && isa<tpu::ReluOp>(nextnextOp)))
      return failure();

    auto padOp = dyn_cast_or_null<tpu::PadOp>(nextOp);
    auto reluOp = dyn_cast_or_null<tpu::ReluOp>(nextnextOp);

    float const_val = padOp.const_val().convertToFloat();
    if (const_val < 0.0)
      return failure();

    // set relu for conv
    convOp->setAttr("param",
          tpu::ConvParam::get(
              convOp.param().stride_h(),
              convOp.param().stride_w(),
              convOp.param().padding(),
              convOp.param().dilation_h(),
              convOp.param().dilation_w(),
              convOp.param().padding_t(),
              convOp.param().padding_b(),
              convOp.param().padding_l(),
              convOp.param().padding_r(),
              convOp.param().group(),
              convOp.param().is_dw(),
              convOp.param().with_bias(),
              rewriter.getBoolAttr(true),
              convOp.param().ins(),
              convOp.param().pad_value(),
              rewriter.getContext()));

    reluOp.getResult().replaceAllUsesWith(padOp.getResult());
    rewriter.eraseOp(reluOp);

    return success();
  }
};

template <typename OpTy>
struct SwitchConvPadHWPattern : public RewritePattern {
  SwitchConvPadHWPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {


    // reshape
    auto inst_0 = op->getOperand(0).getDefiningOp();
    if (!(inst_0 && isa<tpu::ReshapeOp>(inst_0)))
      return failure();

    // conv
    auto inst_1 = op;

    // pad
    auto inst_2 = getNextOp(inst_1);
    if (!(inst_2 && isa<tpu::PadOp>(inst_2)))
      return failure();

    // reshape
    auto inst_3 = getNextOp(inst_2);
    if (!(inst_3 && isa<tpu::ReshapeOp>(inst_3)))
      return failure();

    // conv
    auto inst_4 = getNextOp(inst_3);
    if (!(inst_4 && isa<tpu::Conv2DOp>(inst_4)))
      return failure();

    // reshape
    auto inst_5 = getNextOp(inst_4);
    if (!(inst_5 && isa<tpu::ReshapeOp>(inst_5)))
      return failure();

    // conv
    auto inst_6 = getNextOp(inst_5);
    if (!(inst_6 && isa<tpu::Conv2DOp>(inst_6)))
      return failure();

    // pad
    auto inst_7 = getNextOp(inst_6);
    if (!(inst_7 && isa<tpu::PadOp>(inst_7)))
      return failure();

    // reshape
    auto inst_8 = getNextOp(inst_7);
    if (!(inst_8 && isa<tpu::ReshapeOp>(inst_8)))
      return failure();

    // conv
    auto inst_9 = getNextOp(inst_8);
    if (!(inst_9 && isa<tpu::Conv2DOp>(inst_9)))
      return failure();

    auto firstReshapeOp = dyn_cast_or_null<tpu::ReshapeOp>(inst_0);
    auto conv_0 = dyn_cast_or_null<tpu::Conv2DOp>(inst_1);
    auto conv_1 = dyn_cast_or_null<tpu::Conv2DOp>(inst_4);
    auto conv_2 = dyn_cast_or_null<tpu::Conv2DOp>(inst_6);
    auto conv_3 = dyn_cast_or_null<tpu::Conv2DOp>(inst_9);
    auto pad_0 = dyn_cast_or_null<tpu::PadOp>(inst_2);
    auto pad_1 = dyn_cast_or_null<tpu::PadOp>(inst_7);

    std::vector<int64_t> conv0_opd0 = getTensorShape(inst_1->getOperand(0));
    std::vector<int64_t> conv1_opd0 = getTensorShape(inst_4->getOperand(0));
    std::vector<int64_t> conv2_opd0 = getTensorShape(inst_6->getOperand(0));
    std::vector<int64_t> conv3_opd0 = getTensorShape(inst_9->getOperand(0));
    // search for (n, c, 1, w) case
    if(!(conv0_opd0.size() == 4 && conv0_opd0[2] == 1))
      return failure();
    // search for (n, c, 1, w) case
    if(!(conv2_opd0.size() == 4 && conv2_opd0[2] == 1))
      return failure();
    // search for (n, c, w) case
    if(!(conv1_opd0.size() == 3))
      return failure();
    // search for (n, c, w) case
    if(!(conv3_opd0.size() == 3))
      return failure();

    rewriter.setInsertionPointAfter(inst_9);
    std::vector<int64_t> shape_0 = getTensorShape(inst_0->getResult(0));
    auto type_0 = inst_0->getResult(0).getType().cast<RankedTensorType>();
    RankedTensorType out_type_0 = RankedTensorType::get(
                          {shape_0[0], shape_0[1],
                           shape_0[3], shape_0[2]},
                           type_0.getElementType());

    // 1. add reshape ->(n, c, 1, w) to (n, c, w, 1)
    std::vector<NamedAttribute> attrs_0;
    std::vector<Value> operands_0;
    operands_0.push_back(inst_0->getOperand(0));
    attrs_0.push_back(rewriter.getNamedAttr("name", firstReshapeOp.nameAttr()));
    auto new_inst_0 = rewriter.create<tpu::ReshapeOp>(
                      inst_0->getLoc(), out_type_0,
                      ArrayRef<Value>{operands_0},
                      ArrayRef<NamedAttribute>{attrs_0});

    // 2 add conv0
    //   convert conv(n, c, 1, w) to conv(n, c, w, 1)
    // 2.1 first convert weight shape
    TensorFile *wTF = getWeightTensorFile(op);
    std::vector<int64_t> shape_conv_0_w = getTensorShape(conv_0.getOperand(1));
    if (!(shape_conv_0_w.size() == 4 && shape_conv_0_w[2] == 1))
      return failure();
    shape_conv_0_w[2] = shape_conv_0_w[3];
    shape_conv_0_w[3] = 1;
    auto filter = readAndDeleteWeightTensor<float>(conv_0.filter(), wTF);
    addWeightTensorAndUpdateWeightOp<float>(conv_0.getOperand(1),
      "", *filter, shape_conv_0_w, "NONE", wTF);

    // 2.2 convert conv, switch output h/w and param h/w
    std::vector<NamedAttribute> attrs_1;
    std::vector<Value> operands_1;
    operands_1.push_back(new_inst_0.getResult());
    for(uint i = 1; i < conv_0.getNumOperands(); i++)
      operands_1.push_back(conv_0.getOperand(i));
    attrs_1.push_back(rewriter.getNamedAttr("name",
                  conv_0.nameAttr()));
    // switch h/w and dilation_h/w
    attrs_1.push_back(rewriter.getNamedAttr("param",
        tpu::ConvParam::get(
              conv_0.param().stride_w(),
              conv_0.param().stride_h(),
              conv_0.param().padding(),
              conv_0.param().dilation_w(),
              conv_0.param().dilation_h(),
              conv_0.param().padding_t(),
              conv_0.param().padding_b(),
              conv_0.param().padding_l(),
              conv_0.param().padding_r(),
              conv_0.param().group(),
              conv_0.param().is_dw(),
              conv_0.param().with_bias(),
              conv_0.param().do_relu(),
              conv_0.param().ins(),
              conv_0.param().pad_value(),
              rewriter.getContext())));
    attrs_1.push_back(
        rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));

    std::vector<int64_t> shape_conv_0 = getTensorShape(conv_0.getResult());
    RankedTensorType out_type_conv_0 = RankedTensorType::get(
                          {shape_conv_0[0], shape_conv_0[1],
                           shape_conv_0[3], shape_conv_0[2]},
                           type_0.getElementType());

    auto new_inst_1 = rewriter.create<tpu::Conv2DOp>(
                      conv_0.getLoc(), out_type_conv_0,
                      ArrayRef<Value>{operands_1},
                      ArrayRef<NamedAttribute>{attrs_1});

    // 3 add pad0
    std::vector<int32_t> pads;
    std::vector<Value> operands_2;
    std::vector<NamedAttribute> attrs_2;
    SmallVector<Attribute, 8> padsAttr;
    std::vector<int64_t> shape_pad_0 = getTensorShape(new_inst_1.getResult());
    arrayAttrToVector(pad_0.pads().getValue(), pads);
    auto const_val = pad_0.const_val().convertToFloat();
    for (unsigned int i = 0; i < 8; i++) {
      int v = pads[i];
      if (i == 2)
        v = pads[3];
      else if (i == 3)
        v = pads[2];
      else if (i == 6)
        v = pads[7];
      else if (i == 7)
        v = pads[6];
      auto padAttr = rewriter.getI32IntegerAttr(v);
      padsAttr.push_back(padAttr);
    }

    operands_2.push_back(new_inst_1.getResult());
    attrs_2.push_back(rewriter.getNamedAttr("name", pad_0.nameAttr()));
    attrs_2.push_back(
        rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
    attrs_2.push_back(rewriter.getNamedAttr("pads",
                           rewriter.getArrayAttr(padsAttr)));
    attrs_2.push_back(rewriter.getNamedAttr("const_val",
                           rewriter.getF32FloatAttr(const_val)));
    RankedTensorType output_type_pad = RankedTensorType::get(
                          {shape_pad_0[0], shape_pad_0[1],
                           shape_pad_0[2] + pads[3] + pads[7],
                           shape_pad_0[3] + pads[2] + pads[6]},
                           type_0.getElementType());

    auto new_inst_2 = rewriter.create<tpu::PadOp>(
          pad_0.getLoc(), output_type_pad,
          ArrayRef<Value>{operands_2},
          ArrayRef<NamedAttribute>{attrs_2});

    // 4 add conv1
    std::vector<NamedAttribute> attrs_3;
    std::vector<Value> operands_3;

    operands_3.push_back(new_inst_2.getResult());
    for(uint i = 1; i < conv_1.getNumOperands(); i++)
      operands_3.push_back(conv_1.getOperand(i));
    attrs_3.push_back(rewriter.getNamedAttr("name",
                  conv_1.nameAttr()));
    attrs_3.push_back(rewriter.getNamedAttr("param",
        tpu::ConvParam::get(
              conv_1.param().stride_h(),
              conv_1.param().stride_w(),
              conv_1.param().padding(),
              conv_1.param().dilation_h(),
              conv_1.param().dilation_w(),
              conv_1.param().padding_t(),
              conv_1.param().padding_b(),
              conv_1.param().padding_l(),
              conv_1.param().padding_r(),
              conv_1.param().group(),
              conv_1.param().is_dw(),
              conv_1.param().with_bias(),
              conv_1.param().do_relu(),
              conv_1.param().ins(),
              conv_1.param().pad_value(),
              rewriter.getContext())));
    attrs_3.push_back(
        rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));

    std::vector<int64_t> shape_conv_1 = getTensorShape(conv_1.getResult());
    if (shape_conv_1.size() != 3)
      return failure();
    RankedTensorType out_type_conv_1 = RankedTensorType::get(
                          {shape_conv_1[0], shape_conv_1[1],
                           shape_conv_1[2], 1},
                           type_0.getElementType());

    auto new_inst_3 = rewriter.create<tpu::Conv2DOp>(
                      conv_1.getLoc(), out_type_conv_1,
                      ArrayRef<Value>{operands_3},
                      ArrayRef<NamedAttribute>{attrs_3});

    // 5 add conv2
    auto filter_conv2 = readAndDeleteWeightTensor<float>(conv_2.filter(), wTF);
    addWeightTensorAndUpdateWeightOp<float>(conv_2.getOperand(1),
      "", *filter_conv2, shape_conv_0_w, "NONE", wTF);

    std::vector<NamedAttribute> attrs_4;
    std::vector<Value> operands_4;
    operands_4.push_back(new_inst_3.getResult());
    for(uint i = 1; i < conv_2.getNumOperands(); i++)
      operands_4.push_back(conv_2.getOperand(i));
    attrs_4.push_back(rewriter.getNamedAttr("name",
                  conv_2.nameAttr()));
    attrs_4.push_back(rewriter.getNamedAttr("param",
        tpu::ConvParam::get(
              conv_2.param().stride_w(),
              conv_2.param().stride_h(),
              conv_2.param().padding(),
              conv_2.param().dilation_w(),
              conv_2.param().dilation_h(),
              conv_2.param().padding_t(),
              conv_2.param().padding_b(),
              conv_2.param().padding_l(),
              conv_2.param().padding_r(),
              conv_2.param().group(),
              conv_2.param().is_dw(),
              conv_2.param().with_bias(),
              conv_2.param().do_relu(),
              conv_2.param().ins(),
              conv_2.param().pad_value(),
              rewriter.getContext())));
    attrs_4.push_back(
        rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));

    std::vector<int64_t> shape_conv_2 = getTensorShape(conv_2.getResult());
    RankedTensorType out_type_conv_2 = RankedTensorType::get(
                          {shape_conv_2[0], shape_conv_2[1],
                           shape_conv_2[3], shape_conv_2[2]},
                           type_0.getElementType());

    auto new_inst_4 = rewriter.create<tpu::Conv2DOp>(
                      conv_2.getLoc(), out_type_conv_2,
                      ArrayRef<Value>{operands_4},
                      ArrayRef<NamedAttribute>{attrs_4});

    // 6 add pad1
    std::vector<Value> operands_5;
    std::vector<NamedAttribute> attrs_5;
    SmallVector<Attribute, 8> padsAttr_1;
    std::vector<int64_t> shape_pad_1 = getTensorShape(new_inst_4.getResult());
    arrayAttrToVector(pad_1.pads().getValue(), pads);
    const_val = pad_1.const_val().convertToFloat();
    for (unsigned int i = 0; i < 8; i++) {
      int v = pads[i];
      if (i == 2)
        v = pads[3];
      else if (i == 3)
        v = pads[2];
      else if (i == 6)
        v = pads[7];
      else if (i == 7)
        v = pads[6];
      auto padAttr = rewriter.getI32IntegerAttr(v);
      padsAttr_1.push_back(padAttr);
    }

    operands_5.push_back(new_inst_4.getResult());
    attrs_5.push_back(rewriter.getNamedAttr("name", pad_1.nameAttr()));
    attrs_5.push_back(
        rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
    attrs_5.push_back(rewriter.getNamedAttr("pads",
                           rewriter.getArrayAttr(padsAttr_1)));
    attrs_5.push_back(rewriter.getNamedAttr("const_val",
                           rewriter.getF32FloatAttr(const_val)));
    RankedTensorType output_type_pad_1 = RankedTensorType::get(
                          {shape_pad_1[0], shape_pad_1[1],
                           shape_pad_1[2] + pads[3] + pads[7],
                           shape_pad_1[3] + pads[2] + pads[6]},
                           type_0.getElementType());

    auto new_inst_5 = rewriter.create<tpu::PadOp>(
          pad_1.getLoc(), output_type_pad_1,
          ArrayRef<Value>{operands_5},
          ArrayRef<NamedAttribute>{attrs_5});

    // 7 add conv2
    std::vector<NamedAttribute> attrs_6;
    std::vector<Value> operands_6;

    operands_6.push_back(new_inst_5.getResult());
    for(uint i = 1; i < conv_3.getNumOperands(); i++)
      operands_6.push_back(conv_3.getOperand(i));
    attrs_6.push_back(rewriter.getNamedAttr("name",
                  conv_3.nameAttr()));
    attrs_6.push_back(rewriter.getNamedAttr("param",
        tpu::ConvParam::get(
              conv_3.param().stride_h(),
              conv_3.param().stride_w(),
              conv_3.param().padding(),
              conv_3.param().dilation_h(),
              conv_3.param().dilation_w(),
              conv_3.param().padding_t(),
              conv_3.param().padding_b(),
              conv_3.param().padding_l(),
              conv_3.param().padding_r(),
              conv_3.param().group(),
              conv_3.param().is_dw(),
              conv_3.param().with_bias(),
              conv_3.param().do_relu(),
              conv_3.param().ins(),
              conv_3.param().pad_value(),
              rewriter.getContext())));
    attrs_6.push_back(
        rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));

    std::vector<int64_t> shape_conv_3 = getTensorShape(conv_3.getResult());
    if (shape_conv_3.size() != 3)
      return failure();
    RankedTensorType out_type_conv_3 = RankedTensorType::get(
                          {shape_conv_3[0], shape_conv_3[1],
                           shape_conv_3[2], 1},
                           type_0.getElementType());

    auto new_inst_6 = rewriter.create<tpu::Conv2DOp>(
                      conv_3.getLoc(), out_type_conv_3,
                      ArrayRef<Value>{operands_6},
                      ArrayRef<NamedAttribute>{attrs_6});

    // erase orignal ops and update dependency
    inst_9->getResult(0).replaceAllUsesWith(new_inst_6.getResult());

    rewriter.eraseOp(inst_9);
    rewriter.eraseOp(inst_8);
    rewriter.eraseOp(inst_7);
    rewriter.eraseOp(inst_6);
    rewriter.eraseOp(inst_5);
    rewriter.eraseOp(inst_4);
    rewriter.eraseOp(inst_3);
    rewriter.eraseOp(inst_2);
    rewriter.eraseOp(inst_1);
    rewriter.eraseOp(inst_0);
    return success();
  }
};

template <typename OpTy>
struct SwitchConv2DHWPattern : public RewritePattern {
  SwitchConv2DHWPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto conv_op = dyn_cast_or_null<tpu::Conv2DOp>(op);
    if (!conv_op)
      return failure();
    auto prev_op = op->getOperand(0).getDefiningOp();

    std::vector<int64_t> shape_opd0 = getTensorShape(conv_op->getOperand(0));
    // search for (n, c, 1, w) case
    if(!(shape_opd0.size() == 4 && shape_opd0[2] == 1 && shape_opd0[3] != 1))
      return failure();

    // generate reshape op
    rewriter.setInsertionPointAfter(conv_op);
    std::vector<int64_t> shape_0 = getTensorShape(prev_op->getResult(0));
    auto type_0 = prev_op->getResult(0).getType().cast<RankedTensorType>();
    RankedTensorType out_type_0 = RankedTensorType::get(
                          {shape_0[0], shape_0[1],
                           shape_0[3], shape_0[2]},
                           type_0.getElementType());

    // 1. add reshape ->(n, c, 1, w) to (n, c, w, 1)
    std::vector<NamedAttribute> attrs_0;
    std::vector<Value> operands_0;
    std::string reshape_inst_0_name = conv_op.nameAttr().getValue().str() + "_reshape";
    operands_0.push_back(conv_op.getOperand(0));
    attrs_0.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(reshape_inst_0_name)));
    auto reshape_inst_0 = rewriter.create<tpu::ReshapeOp>(
                      conv_op.getLoc(), out_type_0,
                      ArrayRef<Value>{operands_0},
                      ArrayRef<NamedAttribute>{attrs_0});

    // generate conv op, update weight op
    TensorFile *wTF = getWeightTensorFile(op);
    std::vector<int64_t> shape_filter = getTensorShape(conv_op.getOperand(1));
    if (!(shape_filter.size() == 4 && shape_filter[2] == 1))
      return failure();
    shape_filter[2] = shape_filter[3];
    shape_filter[3] = 1;
    auto filter = readAndDeleteWeightTensor<float>(conv_op.filter(), wTF);
    addWeightTensorAndUpdateWeightOp<float>(conv_op.getOperand(1),
      "", *filter, shape_filter, "NONE", wTF);

    // convert conv, switch output h/w and param h/w
    std::vector<NamedAttribute> conv_attrs;
    std::vector<Value> conv_opds;
    conv_opds.push_back(reshape_inst_0.getResult());
    for(uint i = 1; i < conv_op.getNumOperands(); i++)
      conv_opds.push_back(conv_op.getOperand(i));
    conv_attrs.push_back(rewriter.getNamedAttr("name",
                  conv_op.nameAttr()));
    // switch h/w and dilation_h/w
    conv_attrs.push_back(rewriter.getNamedAttr("param",
        tpu::ConvParam::get(
              conv_op.param().stride_w(),
              conv_op.param().stride_h(),
              conv_op.param().padding(),
              conv_op.param().dilation_w(),
              conv_op.param().dilation_h(),
              conv_op.param().padding_t(),
              conv_op.param().padding_b(),
              conv_op.param().padding_l(),
              conv_op.param().padding_r(),
              conv_op.param().group(),
              conv_op.param().is_dw(),
              conv_op.param().with_bias(),
              conv_op.param().do_relu(),
              conv_op.param().ins(),
              conv_op.param().pad_value(),
              rewriter.getContext())));
    conv_attrs.push_back(
        rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));

    std::vector<int64_t> shape_conv_0 = getTensorShape(conv_op.getResult());
    RankedTensorType out_type_conv_0 = RankedTensorType::get(
                          {shape_conv_0[0], shape_conv_0[1],
                           shape_conv_0[3], shape_conv_0[2]},
                           type_0.getElementType());

    auto new_conv = rewriter.create<tpu::Conv2DOp>(
                      conv_op.getLoc(), out_type_conv_0,
                      ArrayRef<Value>{conv_opds},
                      ArrayRef<NamedAttribute>{conv_attrs});

    // reshape back
    std::vector<NamedAttribute> attrs_1;
    std::vector<Value> operands_1;
    std::string reshape_inst_1_name = conv_op.nameAttr().getValue().str() + "_reshape_back";
    operands_1.push_back(new_conv.getResult());
    attrs_1.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(reshape_inst_1_name)));
    auto reshape_inst_1 = rewriter.create<tpu::ReshapeOp>(
                      conv_op.getLoc(), conv_op.getResult().getType(),
                      ArrayRef<Value>{operands_1},
                      ArrayRef<NamedAttribute>{attrs_1});

    // update dependency
    conv_op.getResult().replaceAllUsesWith(reshape_inst_1.getResult());
    // erase ops
    rewriter.eraseOp(conv_op);

    return success();
  }
};

template <typename OpTy>
struct SwitchPadHWPattern : public RewritePattern {
  SwitchPadHWPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto pad_op = dyn_cast_or_null<tpu::PadOp>(op);
    if (!pad_op)
      return failure();
    auto prev_op = op->getOperand(0).getDefiningOp();

    std::vector<int64_t> shape_opd0 = getTensorShape(pad_op->getOperand(0));
    // search for (n, c, 1, w) case
    if(!(shape_opd0.size() == 4 && shape_opd0[2] == 1 && shape_opd0[3] != 1))
      return failure();

    // generate reshape op
    rewriter.setInsertionPointAfter(pad_op);
    std::vector<int64_t> shape_0 = getTensorShape(prev_op->getResult(0));
    auto type_0 = prev_op->getResult(0).getType().cast<RankedTensorType>();
    RankedTensorType out_type_0 = RankedTensorType::get(
                          {shape_0[0], shape_0[1],
                           shape_0[3], shape_0[2]},
                           type_0.getElementType());

    // 1. add reshape ->(n, c, 1, w) to (n, c, w, 1)
    std::vector<NamedAttribute> attrs_0;
    std::vector<Value> operands_0;
    std::string reshape_inst_0_name = pad_op.nameAttr().getValue().str() + "_reshape";
    operands_0.push_back(pad_op.getOperand());
    attrs_0.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(reshape_inst_0_name)));
    auto reshape_inst_0 = rewriter.create<tpu::ReshapeOp>(
                      pad_op.getLoc(), out_type_0,
                      ArrayRef<Value>{operands_0},
                      ArrayRef<NamedAttribute>{attrs_0});


    // pad
    std::vector<int32_t> pads;
    std::vector<Value> pad_opds;
    std::vector<NamedAttribute> pad_attrs;
    SmallVector<Attribute, 8> padsAttr;
    std::vector<int64_t> shape_pad_0 = getTensorShape(reshape_inst_0.getResult());
    arrayAttrToVector(pad_op.pads().getValue(), pads);
    auto const_val = pad_op.const_val().convertToFloat();
    for (unsigned int i = 0; i < 8; i++) {
      int v = pads[i];
      if (i == 2)
        v = pads[3];
      else if (i == 3)
        v = pads[2];
      else if (i == 6)
        v = pads[7];
      else if (i == 7)
        v = pads[6];
      auto padAttr = rewriter.getI32IntegerAttr(v);
      padsAttr.push_back(padAttr);
    }

    pad_opds.push_back(reshape_inst_0.getResult());
    pad_attrs.push_back(rewriter.getNamedAttr("name", pad_op.nameAttr()));
    pad_attrs.push_back(
        rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
    pad_attrs.push_back(rewriter.getNamedAttr("pads",
                           rewriter.getArrayAttr(padsAttr)));
    pad_attrs.push_back(rewriter.getNamedAttr("const_val",
                           rewriter.getF32FloatAttr(const_val)));
    RankedTensorType output_type_pad = RankedTensorType::get(
                          {shape_pad_0[0], shape_pad_0[1],
                           shape_pad_0[2] + pads[3] + pads[7],
                           shape_pad_0[3] + pads[2] + pads[6]},
                           type_0.getElementType());

    auto new_pad_op = rewriter.create<tpu::PadOp>(
          pad_op.getLoc(), output_type_pad,
          ArrayRef<Value>{pad_opds},
          ArrayRef<NamedAttribute>{pad_attrs});

    // reshape back
    std::vector<NamedAttribute> attrs_1;
    std::vector<Value> operands_1;
    std::string reshape_inst_1_name = pad_op.nameAttr().getValue().str() + "_reshape";
    operands_1.push_back(new_pad_op.getResult());
    attrs_1.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(reshape_inst_1_name)));
    auto reshape_inst_1 = rewriter.create<tpu::ReshapeOp>(
                      pad_op.getLoc(), pad_op.getResult().getType(),
                      ArrayRef<Value>{operands_1},
                      ArrayRef<NamedAttribute>{attrs_1});

    // update dependency
    pad_op.getResult().replaceAllUsesWith(reshape_inst_1.getResult());
    // erase ops
    rewriter.eraseOp(pad_op);

    return success();
  }
};


template <typename OpTy>
struct ExtendGroupConvShapePattern : public RewritePattern {
  ExtendGroupConvShapePattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto conv_op = dyn_cast_or_null<tpu::Conv2DOp>(op);

    auto opd0 = conv_op.getOperand(0);
    auto opd1 = conv_op.getOperand(1);
    auto op_result = conv_op.getResult();

    std::vector<int64_t> input_shape = getTensorShape(opd0);
    std::vector<int64_t> filter_shape = getTensorShape(opd1);
    std::vector<int64_t> result_shape = getTensorShape(op_result);

    // input_shape size is 3 and filter_shape size is 5
    // and is group conv
    if (!((input_shape.size() == 3) && (filter_shape.size() == 5) &&
        (input_shape[1] == filter_shape[0]) &&
        (result_shape[1] == input_shape[1])))
      return failure();
    if (!(filter_shape[3] == 1 && filter_shape[4] == 1))
      return failure();

    // generate reshape op
    rewriter.setInsertionPointAfter(conv_op);
    auto type_0 = conv_op->getOperand(0).getType().cast<RankedTensorType>();
    RankedTensorType out_type_0 = RankedTensorType::get(
                          {input_shape[0], input_shape[1],
                           input_shape[2], 1},
                           type_0.getElementType());

    // 1. add reshape ->(n, c, w) to (n, c, w, 1)
    std::vector<NamedAttribute> attrs_0;
    std::vector<Value> operands_0;
    std::string reshape_inst_0_name = conv_op.nameAttr().getValue().str() + "_reshape";
    operands_0.push_back(conv_op.getOperand(0));
    attrs_0.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(reshape_inst_0_name)));
    auto reshape_inst_0 = rewriter.create<tpu::ReshapeOp>(
                      conv_op.getLoc(), out_type_0,
                      ArrayRef<Value>{operands_0},
                      ArrayRef<NamedAttribute>{attrs_0});

    // set conv
    std::vector<NamedAttribute> conv_attrs;
    std::vector<Value> conv_opds;
    conv_opds.push_back(reshape_inst_0.getResult());
    for(uint i = 1; i < conv_op.getNumOperands(); i++)
      conv_opds.push_back(conv_op.getOperand(i));
    conv_attrs.push_back(rewriter.getNamedAttr("name",
                  conv_op.nameAttr()));
    // switch h/w and dilation_h/w
    conv_attrs.push_back(rewriter.getNamedAttr("param",
        tpu::ConvParam::get(
              conv_op.param().stride_h(),
              conv_op.param().stride_w(),
              conv_op.param().padding(),
              conv_op.param().dilation_h(),
              conv_op.param().dilation_w(),
              conv_op.param().padding_t(),
              conv_op.param().padding_b(),
              conv_op.param().padding_l(),
              conv_op.param().padding_r(),
              conv_op.param().group(),
              conv_op.param().is_dw(),
              conv_op.param().with_bias(),
              conv_op.param().do_relu(),
              conv_op.param().ins(),
              conv_op.param().pad_value(),
              rewriter.getContext())));
    conv_attrs.push_back(
        rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));

    RankedTensorType out_type_conv_0 = RankedTensorType::get(
                          {result_shape[0], result_shape[1],
                           result_shape[2], 1},
                           type_0.getElementType());

    auto new_conv = rewriter.create<tpu::Conv2DOp>(
                      conv_op.getLoc(), out_type_conv_0,
                      ArrayRef<Value>{conv_opds},
                      ArrayRef<NamedAttribute>{conv_attrs});

    // reshape back
    std::vector<NamedAttribute> attrs_1;
    std::vector<Value> operands_1;
    std::string reshape_inst_1_name = conv_op.nameAttr().getValue().str() + "_reshape_back";
    operands_1.push_back(new_conv.getResult());
    attrs_1.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(reshape_inst_1_name)));
    auto reshape_inst_1 = rewriter.create<tpu::ReshapeOp>(
                      conv_op.getLoc(), conv_op.getResult().getType(),
                      ArrayRef<Value>{operands_1},
                      ArrayRef<NamedAttribute>{attrs_1});

    conv_op.getResult().replaceAllUsesWith(reshape_inst_1.getResult());
    rewriter.eraseOp(conv_op);
    return success();
  }
};

template <typename OpTy>
struct cleanReshapePattern : public RewritePattern {
  cleanReshapePattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto first_op = dyn_cast_or_null<tpu::ReshapeOp>(op);

    auto next_op = getNextOp(first_op);
    auto last_op = next_op;

    while(next_op) {
      if (isa<tpu::ReshapeOp>(next_op)) {
        last_op = next_op;
        next_op = getNextOp(next_op);
      } else {
        break;
      }
    }

    // not reshape + reshape pattern
    if (!isa<tpu::ReshapeOp>(last_op))
      return failure();

    std::vector<int64_t> input_shape = getTensorShape(first_op->getOperand(0));
    std::vector<int64_t> output_shape = getTensorShape(last_op->getResult(0));

    if (input_shape.size() != output_shape.size())
      return failure();

    for(uint i = 0; i < input_shape.size(); i++) {
      if (input_shape[i] != output_shape[i])
        return failure();
    }

    // erase all reshape ops
    auto input_operand = first_op->getOperand(0).getDefiningOp();
    last_op->getResult(0).replaceAllUsesWith(input_operand->getResult(0));
    rewriter.eraseOp(last_op);

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
        MergePermuteOpPattern<tpu::ReshapeOp>,
        MergeConvReluPattern<tpu::Conv2DOp>
      >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));

    patterns.clear();
    patterns.insert<
        SwitchConvPadHWPattern<tpu::Conv2DOp>
      >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));

    /*
    // The following optimization has the same effect as
    // SwitchConvPadHWPattern, but more common
    patterns.clear();
    patterns.insert<
        SwitchConv2DHWPattern<tpu::Conv2DOp>,
        SwitchPadHWPattern<tpu::PadOp>,
        ExtendGroupConvShapePattern<tpu::Conv2DOp>
      >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));

    patterns.clear();
    patterns.insert<
        cleanReshapePattern<tpu::ReshapeOp>
      >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));
    */
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::createMergePermutePass() {
  return std::make_unique<MergePermuteOpPass>();
}
