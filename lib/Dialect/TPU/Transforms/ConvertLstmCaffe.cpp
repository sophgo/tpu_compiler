//===- ConvertLstmCaffe.cpp - convert
// LstmCaffe----------------------------------===//
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
// This file implements to unroll lstm_caffe
//
//===----------------------------------------------------------------------===//

#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/Passes.h"
#include "tpuc/Support/TensorFile.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "convert_lstm_caffe"

using namespace mlir;

namespace {

// caffe lstm unroll
struct TpuLstmCaffeUnrollPattern : public RewritePattern {
  TpuLstmCaffeUnrollPattern(MLIRContext *context)
      : RewritePattern("tpu.lstm_caffe", 8, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto lstmOp = cast<tpu::LstmCaffeOp>(op);
    LLVM_DEBUG(llvm::errs() << lstmOp.getOperationName() << ":" << getOpName(op)
                            << "\n";);
    auto inputOp = lstmOp.getOperand(0);
    auto weight_x = lstmOp.getOperand(1);
    auto bias = lstmOp.getOperand(2);
    auto weight_h = lstmOp.getOperand(3);
    auto input_shape = getTensorShape(inputOp);
    auto x_shape = getTensorShape(weight_x);
    auto h_shape = getTensorShape(weight_h);
    std::vector<int64_t> output_shape = getTensorShape(op->getResult(0));
    int64_t seq_length = output_shape[0];
    int64_t batch_size = output_shape[1];
    int64_t num_output = output_shape[2];
    std::string op_name = lstmOp.name().str();
    TensorFile *wTF = getWeightTensorFile(op);
    auto wFV = getWeightFileValue(op);
    auto element_type =
        inputOp.getType().cast<RankedTensorType>().getElementType();
    auto NoneOp = OpBuilder(op).create<tpu::NoneOp>(rewriter.getUnknownLoc(),
                                                    rewriter.getNoneType());
    RankedTensorType output_type;
    // x_transform = input do inner product
    std::vector<Value> operands;
    std::vector<NamedAttribute> attrs;

    std::string new_name = op_name + "_x_transform";
    attrs.clear();
    operands.clear();
    attrs.push_back(
        rewriter.getNamedAttr("name", rewriter.getStringAttr(new_name)));
    attrs.push_back(
        rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
    operands.push_back(inputOp);
    operands.push_back(weight_x);
    operands.push_back(bias);
    operands.push_back(NoneOp);
    operands.push_back(NoneOp);
    operands.push_back(NoneOp);
    operands.push_back(NoneOp);
    output_type = RankedTensorType::get({seq_length, batch_size, x_shape[0]},
                                        element_type);
    auto x_transform_op = rewriter.create<tpu::FullyConnectedOp>(
        op->getLoc(), ArrayRef<mlir::Type>{output_type}, operands, attrs);

    // time step
    Value h_pre_op = NoneOp;
    Value c_pre_op = NoneOp;
    std::vector<Value> concat_operands;
    std::vector<NamedAttribute> concat_attrs;

    auto filter_h = readAndDeleteWeightTensor<float>(weight_h, wTF);
    for (int i = 0; i < seq_length; i++) {
      // slice op
      auto seq_name = op_name + "_seq_" + std::to_string(i);
      operands.clear();
      attrs.clear();
      new_name = seq_name + "_slice";
      attrs.push_back(
          rewriter.getNamedAttr("name", rewriter.getStringAttr(new_name)));
      attrs.push_back(
          rewriter.getNamedAttr("axis", rewriter.getI32IntegerAttr(0)));
      attrs.push_back(
          rewriter.getNamedAttr("offset", rewriter.getI32IntegerAttr(i)));
      attrs.push_back(
          rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
      operands.push_back(x_transform_op);
      output_type = RankedTensorType::get({1, batch_size, x_shape[0]}, element_type);
      auto x_slice_op = rewriter.create<tpu::SliceOp>(
          op->getLoc(), ArrayRef<mlir::Type>{output_type}, operands, attrs);

      Value x_op = x_slice_op;
      if (i > 0) {
        // h_i interproduct
        auto weight_h_i = addWeightTensorAndCreateWeightOp<float>(
            op, seq_name + "_h_filter", *filter_h, h_shape, "NONE", wTF, wFV);
        operands.clear();
        attrs.clear();
        new_name = seq_name + "_h_gemm";
        attrs.push_back(
            rewriter.getNamedAttr("name", rewriter.getStringAttr(new_name)));
        attrs.push_back(
            rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
        operands.push_back(h_pre_op);
        operands.push_back(weight_h_i);
        operands.push_back(NoneOp);
        operands.push_back(NoneOp);
        operands.push_back(NoneOp);
        operands.push_back(NoneOp);
        operands.push_back(NoneOp);
        output_type =
            RankedTensorType::get({1, batch_size, h_shape[0]}, element_type);
        auto h_gemm_op = rewriter.create<tpu::FullyConnectedOp>(
            op->getLoc(), ArrayRef<mlir::Type>{output_type}, operands, attrs);
        // sum op
        operands.clear();
        attrs.clear();
        new_name = seq_name + "_gate_sum";
        attrs.push_back(
            rewriter.getNamedAttr("name", rewriter.getStringAttr(new_name)));
        attrs.push_back(
            rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
        operands.push_back(h_gemm_op);
        operands.push_back(x_slice_op);
        operands.push_back(NoneOp);
        operands.push_back(NoneOp);
        operands.push_back(NoneOp);
        operands.push_back(NoneOp);
        x_op = rewriter.create<tpu::EltwiseAddOp>(
            op->getLoc(), ArrayRef<mlir::Type>{output_type}, operands, attrs);
      }
      // gate_sum split to X0,X1,X2,X3
      Value x_array[4];
      int64_t hidden_dim = h_shape[0] / 4;
      output_type =
          RankedTensorType::get({1, batch_size, hidden_dim}, element_type);
      for (int x = 0; x < 4; x++) {
        operands.clear();
        attrs.clear();
        new_name = seq_name + "_x_" + std::to_string(x);
        attrs.push_back(
            rewriter.getNamedAttr("name", rewriter.getStringAttr(new_name)));
        attrs.push_back(
            rewriter.getNamedAttr("axis", rewriter.getI32IntegerAttr(2)));
        attrs.push_back(rewriter.getNamedAttr(
            "offset", rewriter.getI32IntegerAttr(x * hidden_dim)));
        attrs.push_back(
            rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
        // operands.push_back(x_transform_reshape_op);
        operands.push_back(x_op);
        x_array[x] = rewriter.create<tpu::SliceOp>(
            op->getLoc(), ArrayRef<mlir::Type>{output_type}, operands, attrs);
      }
      // i_op = sigmoid(x0)
      operands.clear();
      attrs.clear();
      new_name = seq_name + "_x_0_i";
      attrs.push_back(
          rewriter.getNamedAttr("name", rewriter.getStringAttr(new_name)));
      attrs.push_back(
          rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
      operands.push_back(x_array[0]);
      operands.push_back(NoneOp);
      operands.push_back(NoneOp);
      auto i_op = rewriter.create<tpu::SigmoidOp>(
          op->getLoc(), ArrayRef<mlir::Type>{output_type}, operands, attrs);
      // f_op = sigmoid(x1)
      Value f_op = NoneOp;
      if (i > 0) {
        operands.clear();
        attrs.clear();
        new_name = seq_name + "_x_1_f";
        attrs.push_back(
            rewriter.getNamedAttr("name", rewriter.getStringAttr(new_name)));
        attrs.push_back(
            rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
        operands.push_back(x_array[1]);
        operands.push_back(NoneOp);
        operands.push_back(NoneOp);
        f_op = rewriter.create<tpu::SigmoidOp>(
            op->getLoc(), ArrayRef<mlir::Type>{output_type}, operands, attrs);
      }
      // o_op = sigmoid(x2)
      operands.clear();
      attrs.clear();
      new_name = seq_name + "_x_2_o";
      attrs.push_back(
          rewriter.getNamedAttr("name", rewriter.getStringAttr(new_name)));
      attrs.push_back(
          rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
      operands.push_back(x_array[2]);
      operands.push_back(NoneOp);
      operands.push_back(NoneOp);
      auto o_op = rewriter.create<tpu::SigmoidOp>(
          op->getLoc(), ArrayRef<mlir::Type>{output_type}, operands, attrs);
      // g_op = tanh(x3)
      operands.clear();
      attrs.clear();
      new_name = seq_name + "_x_3_g";
      attrs.push_back(
          rewriter.getNamedAttr("name", rewriter.getStringAttr(new_name)));
      attrs.push_back(
          rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
      operands.push_back(x_array[3]);
      operands.push_back(NoneOp);
      operands.push_back(NoneOp);
      auto g_op = rewriter.create<tpu::TanHOp>(
          op->getLoc(), ArrayRef<mlir::Type>{output_type}, operands, attrs);
      // i_g_op = i_op * g_op
      operands.clear();
      attrs.clear();
      new_name = seq_name + "_i_g";
      attrs.push_back(
          rewriter.getNamedAttr("name", rewriter.getStringAttr(new_name)));
      attrs.push_back(
          rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
      operands.push_back(i_op);
      operands.push_back(g_op);
      operands.push_back(NoneOp);
      operands.push_back(NoneOp);
      operands.push_back(NoneOp);
      operands.push_back(NoneOp);
      auto i_g_op = rewriter.create<tpu::EltwiseMulOp>(
          op->getLoc(), ArrayRef<mlir::Type>{output_type}, operands, attrs);
      // c_op = f_op * c_pre_op + i_op * g_op
      Value c_op = i_g_op;
      if (i > 0) {
        // f_c_op = f_op * c_pre_op
        operands.clear();
        attrs.clear();
        new_name = seq_name + "_f_c_pre";
        attrs.push_back(
            rewriter.getNamedAttr("name", rewriter.getStringAttr(new_name)));
        attrs.push_back(
            rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
        operands.push_back(f_op);
        operands.push_back(c_pre_op);
        operands.push_back(NoneOp);
        operands.push_back(NoneOp);
        operands.push_back(NoneOp);
        operands.push_back(NoneOp);
        auto f_c_op = rewriter.create<tpu::EltwiseMulOp>(
            op->getLoc(), ArrayRef<mlir::Type>{output_type}, operands, attrs);
        // c_op = f_c_op + i_g_op
        operands.clear();
        attrs.clear();
        new_name = seq_name + "_c";
        attrs.push_back(
            rewriter.getNamedAttr("name", rewriter.getStringAttr(new_name)));
        attrs.push_back(
            rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
        operands.push_back(f_c_op);
        operands.push_back(i_g_op);
        operands.push_back(NoneOp);
        operands.push_back(NoneOp);
        operands.push_back(NoneOp);
        operands.push_back(NoneOp);
        c_op = rewriter.create<tpu::EltwiseAddOp>(
            op->getLoc(), ArrayRef<mlir::Type>{output_type}, operands, attrs);
      }
      // tanh_c_op = tanh(c_op)
      operands.clear();
      attrs.clear();
      new_name = seq_name + "_tanh_c";
      attrs.push_back(
          rewriter.getNamedAttr("name", rewriter.getStringAttr(new_name)));
      attrs.push_back(
          rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
      operands.push_back(c_op);
      operands.push_back(NoneOp);
      operands.push_back(NoneOp);
      auto tanh_c_op = rewriter.create<tpu::TanHOp>(
          op->getLoc(), ArrayRef<mlir::Type>{output_type}, operands, attrs);
      // h_op = o_op * tanh_c_op
      operands.clear();
      attrs.clear();
      new_name = seq_name + "_h";
      attrs.push_back(
          rewriter.getNamedAttr("name", rewriter.getStringAttr(new_name)));
      attrs.push_back(
          rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
      operands.push_back(o_op);
      operands.push_back(tanh_c_op);
      operands.push_back(NoneOp);
      operands.push_back(NoneOp);
      operands.push_back(NoneOp);
      operands.push_back(NoneOp);
      h_pre_op = rewriter.create<tpu::EltwiseMulOp>(
          op->getLoc(), ArrayRef<mlir::Type>{output_type}, operands, attrs);
      c_pre_op = c_op;
      concat_operands.push_back(h_pre_op);
    }
    concat_operands.push_back(NoneOp);
    concat_operands.push_back(NoneOp);
    concat_operands.push_back(NoneOp);
    concat_operands.push_back(NoneOp);
    auto concat_name = op_name + "_concat";
    concat_attrs.push_back(
        rewriter.getNamedAttr("name", rewriter.getStringAttr(concat_name)));
    concat_attrs.push_back(
        rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
    concat_attrs.push_back(
        rewriter.getNamedAttr("axis", rewriter.getI32IntegerAttr(0)));
    output_type =
        RankedTensorType::get({seq_length, batch_size, num_output}, element_type);
    auto concat_op = rewriter.create<tpu::ConcatOp>(
        op->getLoc(), ArrayRef<mlir::Type>{output_type}, concat_operands,
        concat_attrs);
    rewriter.replaceOp(op, {concat_op});
    return success();
  }
};

} // namespace

void tpu::LstmCaffeOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<TpuLstmCaffeUnrollPattern>(context);
}
