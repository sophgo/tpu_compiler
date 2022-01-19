//===- ConvertPermute.cpp - convert
// Permute----------------------------------===//
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
// This file implements the permute to reshape
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Passes.h"
#include "tpuc/Support/TensorFile.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "convert_match_template"

using namespace mlir;

namespace {

struct TpuMatchTemplatePattern : public RewritePattern {
  TpuMatchTemplatePattern(MLIRContext *context)
      : RewritePattern("tpu.match_template", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto castOp = cast<tpu::MatchTemplateOp>(op);

    auto i_s = getTensorShape(castOp.input());
    auto m_s = getTensorShape(castOp.match());
    assert(i_s.size() == 2);
    assert(m_s.size() == 2);
    assert(i_s[0] >= m_s[0]);
    assert(i_s[1] >= m_s[1]);
    auto mode = castOp.mode().str();
    std::vector<Value> operands;
    std::vector<NamedAttribute> attrs;
    auto eltType =
        castOp.getResult().getType().cast<RankedTensorType>().getElementType();
    auto op_name = castOp.name().str();
    if (mode == "TM_CCOEFF_NORMED") {
      auto input_name = getOpName(castOp.input().getDefiningOp()).str();
      std::vector<int> c_shape(4);
      c_shape[0] = (int)(i_s[0] - m_s[0] + 1);
      c_shape[1] = (int)(i_s[1] - m_s[1] + 1), c_shape[2] = (int)(m_s[0]);
      c_shape[3] = (int)(m_s[1]);
      std::vector<int> i_stride(4);
      i_stride[3] = 1;
      i_stride[2] = m_s[1];
      i_stride[1] = 1;
      i_stride[0] = m_s[1];
      std::vector<int> o_stride(4);
      o_stride[3] = 1;
      o_stride[2] = c_shape[3] * o_stride[3];
      o_stride[1] = c_shape[2] * o_stride[2];
      o_stride[0] = c_shape[1] * o_stride[1];
      auto NoneOp = OpBuilder(op).create<tpu::NoneOp>(rewriter.getUnknownLoc(),
                                                      rewriter.getNoneType());
      // expand
      auto name = input_name + "_expand";
      operands.push_back(castOp.input());
      attrs.push_back(
          rewriter.getNamedAttr("name", rewriter.getStringAttr(name)));
      attrs.push_back(
          rewriter.getNamedAttr("shape", rewriter.getI32ArrayAttr(c_shape)));
      attrs.push_back(rewriter.getNamedAttr(
          "input_stride", rewriter.getI32ArrayAttr(i_stride)));
      attrs.push_back(rewriter.getNamedAttr(
          "output_stride", rewriter.getI32ArrayAttr(o_stride)));
      attrs.push_back(
          rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
      auto out_type = RankedTensorType::get(
          {c_shape[0]*c_shape[1], c_shape[2] * c_shape[3]}, eltType);
      auto copy_op =
          rewriter.create<tpu::CopyOp>(op->getLoc(), out_type, operands, attrs);
      // reshape match
      auto match_name = getOpName(castOp.match().getDefiningOp()).str();
      operands.clear();
      attrs.clear();
      name = match_name + "_reshape";
      operands.push_back(castOp.match());
      attrs.push_back(
          rewriter.getNamedAttr("name", rewriter.getStringAttr(name)));
      out_type = RankedTensorType::get({m_s[1] * m_s[0], 1}, eltType);
      auto match_shape_op = rewriter.create<tpu::ReshapeOp>(
          op->getLoc(), out_type, operands, attrs);
      // matmul
      operands.clear();
      attrs.clear();
      name = op_name + "_matmul";
      operands.push_back(copy_op.getResult());
      operands.push_back(match_shape_op.getResult());
      operands.push_back(NoneOp.getResult());
      operands.push_back(NoneOp.getResult());
      operands.push_back(NoneOp.getResult());
      operands.push_back(NoneOp.getResult());
      attrs.push_back(
          rewriter.getNamedAttr("name", rewriter.getStringAttr(name)));
      attrs.push_back(
          rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
      out_type = RankedTensorType::get({c_shape[0]*c_shape[1], 1}, eltType);
      auto matmul_op = rewriter.create<tpu::MatMulOp>(op->getLoc(), out_type,
                                                      operands, attrs);
      // reshape matmul
      operands.clear();
      attrs.clear();
      name = op_name + "_reshape2";
      operands.push_back(matmul_op.getResult());
      attrs.push_back(
          rewriter.getNamedAttr("name", rewriter.getStringAttr(name)));
      out_type = RankedTensorType::get({c_shape[0]*c_shape[1]}, eltType);
      auto shape2_op = rewriter.create<tpu::ReshapeOp>(
          op->getLoc(), out_type, operands, attrs);
      // input reduce_l2
      operands.clear();
      attrs.clear();
      name = input_name + "_l2";
      operands.push_back(copy_op.getResult());
      operands.push_back(NoneOp.getResult());
      operands.push_back(NoneOp.getResult());
      attrs.push_back(
          rewriter.getNamedAttr("name", rewriter.getStringAttr(name)));
      std::vector<int> axes(1, 1);
      attrs.push_back(
          rewriter.getNamedAttr("axes", rewriter.getI32ArrayAttr(axes)));
      attrs.push_back(
          rewriter.getNamedAttr("coeff", rewriter.getF32FloatAttr(-0.5f)));
      attrs.push_back(
          rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
      out_type = RankedTensorType::get({c_shape[0]*c_shape[1]}, eltType);
      auto input_l2_op = rewriter.create<tpu::ReduceL2Op>(
          op->getLoc(), out_type, operands, attrs);

      // mul
      operands.clear();
      attrs.clear();
      name = op_name + "_mul";
      operands.push_back(shape2_op.getResult());
      operands.push_back(input_l2_op.getResult());
      operands.push_back(NoneOp.getResult());
      operands.push_back(NoneOp.getResult());
      operands.push_back(NoneOp.getResult());
      operands.push_back(NoneOp.getResult());
      attrs.push_back(
          rewriter.getNamedAttr("name", rewriter.getStringAttr(name)));
      attrs.push_back(
          rewriter.getNamedAttr("quant", getDefaultQuantParam(rewriter)));
      out_type = RankedTensorType::get({c_shape[0]*c_shape[1]}, eltType);
      auto mul_op = rewriter.create<tpu::EltwiseMulOp>(op->getLoc(), out_type,
                                                       operands, attrs);
      rewriter.replaceOp(op, {mul_op.getResult()});

      return success();
    } else {
      llvm::errs() << "matchTemplate(" << mode << ") not support now\n";
      llvm_unreachable("matchTemplate pattern failed");
    }
    return success();
  }
};

} // namespace

void tpu::MatchTemplateOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<TpuMatchTemplatePattern>(context);
}
