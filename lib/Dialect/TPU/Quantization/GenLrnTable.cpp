//===- GenSqrtTable.cpp - Implementation of dynamice generate tanh lookup table
/// slope ---------===//
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
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TensorFile.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/Support/Debug.h>
#include "mlir/Dialect/TPU/QuantizationArithmetic.h"

#include <float.h>

using namespace mlir;

namespace {

#define EXP_START -62
#define EXP_END 63
#define CHANNEL 32
#define NPU_NUM 32
#define TABLE_H_BF16 32
#define TABLE_W_BF16 8
#define TABLE_H_INT8 16
#define TABLE_W_INT8 16
#define TABLE_HW_INT8 (TABLE_H_INT8 * TABLE_W_INT8)
#define TABLE_HW_BF16 (TABLE_H_BF16 * TABLE_W_BF16)
#define TBL_SHAPE_INT8 (TABLE_HW_INT8 * NPU_NUM)
#define TBL_SHAPE_BF16 (TABLE_HW_BF16 * NPU_NUM)

static void quantize_fraction(float numerator, float denominator,
                              int &right_shift_width,
                              int &numerator_quantized) {
  float denominator_ceiling = 255.0 / numerator * denominator;
  right_shift_width = 0;
  numerator_quantized = 0;
  float denominator_quantized = 1.0;
  while ((denominator_quantized * 2) < denominator_ceiling) {
    right_shift_width += 1;
    denominator_quantized = (float)(1 << right_shift_width);
  }
  float scale = denominator_quantized / denominator;
  numerator_quantized = (int)(numerator * scale + 0.5);
}

struct TpuGenLrnTablePattern : public RewritePattern {
  TpuGenLrnTablePattern(MLIRContext *context)
      : RewritePattern("tpu.lrn", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    TensorFile *wTF = getWeightTensorFile(op);
    Value *wfV = getWeightFileValue(op);
    auto lrnOp = cast<tpu::LrnOp>(op);
    std::vector<std::unique_ptr<std::vector<float>>> weights(1);

    std::string op_name =
        lrnOp.getAttrOfType<StringAttr>("name").getValue().str();

    assert(lrnOp.getOpQuant() == "INT8");
    uint32_t local_size = lrnOp.local_size().getLimitedValue();
    float alpha = lrnOp.alpha().convertToFloat();
    float beta = lrnOp.beta().convertToFloat();
    float k = lrnOp.k().convertToFloat();
    auto sq_table_op = lrnOp.getOperand(1)->getDefiningOp();
    if (isa<tpu::NoneOp>(sq_table_op) == false) {
      return matchFailure();
    }
    auto lrnThreeOp = lrnOp.getOperand(3)->getDefiningOp();
    auto lrnTwoOp = lrnThreeOp->getOperand(0)->getDefiningOp();
    auto lrnOneOp = lrnTwoOp->getOperand(0)->getDefiningOp();

    float sq_thy = getOpThreshold(lrnOneOp);
    float sumsq_thy = getOpThreshold(lrnTwoOp);
    float lrn_factor_thy = getOpThreshold(lrnThreeOp);
    float threshold_x = getPreviousOpThreshold(op);
    float threshold_y = getOpThreshold(op);

    // sq table
    std::vector<float> sq_table(TBL_SHAPE_INT8);

    for (int idx = 0; idx < TABLE_HW_INT8; ++idx) {
      float lut_input = threshold_x / 128.0 * idx;
      float lut_output = std::pow(lut_input, 2) * 256.0 / sq_thy;
      lut_output = lut_output * alpha / local_size;
      lut_output = std::floor(lut_output + 0.5);
      if (lut_output > 255.0) {
        lut_output = 255.0;
      }
      for (int n = 0; n < NPU_NUM; n++) {
        sq_table[n * TABLE_HW_INT8 + idx] = lut_output;
      }
    }
    int numerator_quantized0, sum_right_shift_width;
    quantize_fraction(sq_thy, sumsq_thy, sum_right_shift_width,
                      numerator_quantized0);
    lrnOp.setAttr("sum_right_shift_width",
                  rewriter.getI32IntegerAttr(sum_right_shift_width));
    lrnOp.setAttr("quant_data0",
                  rewriter.getI32IntegerAttr(numerator_quantized0));

    // power table
    std::vector<float> power_table(TBL_SHAPE_INT8);

    for (int idx = 0; idx < TABLE_HW_INT8; ++idx) {
      float lut_input = (float)idx / (256.0 / sumsq_thy);
      float lut_output = std::pow(lut_input + k, -beta);
      lut_output = lut_output * (256.0 / lrn_factor_thy);
      lut_output = std::floor(lut_output + 0.5);
      if (lut_output > 255.0) {
        lut_output = 255.0;
      }
      for (int n = 0; n < NPU_NUM; n++) {
        power_table[n * TABLE_HW_INT8 + idx] = lut_output;
      }
    }
    int numerator_quantized1, lrn_right_shift_width;
    quantize_fraction(threshold_x * lrn_factor_thy, threshold_y * 256.0,
                      lrn_right_shift_width, numerator_quantized1);
    lrnOp.setAttr("lrn_right_shift_width",
                  rewriter.getI32IntegerAttr(lrn_right_shift_width));
    lrnOp.setAttr("quant_data1",
                  rewriter.getI32IntegerAttr(numerator_quantized1));
    // update op params
    std::vector<int64_t> weightShape{1, NPU_NUM, TABLE_H_INT8, TABLE_W_INT8};
    auto type = RankedTensorType::get(weightShape,
                                      FloatType::getF32(rewriter.getContext()));

    // remove operand3
    lrnOp.setOperand(3, lrnOp.getOperand(1));
    // sq weight
    auto tensor_name = op_name + "_sq_gen_weight";

    wTF->addTensor<float>(tensor_name, sq_table.data(), type);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
    attrs.push_back(
        rewriter.getNamedAttr("storage", rewriter.getStringAttr("UINT8")));
    auto sq_weight_op = rewriter.create<tpu::LoadWeightOp>(
        op->getLoc(), type, ArrayRef<Value *>{wfV},
        ArrayRef<NamedAttribute>{attrs});
    lrnOp.setOperand(1, sq_weight_op);

    // power weight
    auto tensor_name2 = op_name + "_power_gen_weight";
    wTF->addTensor<float>(tensor_name2, power_table.data(), type);
    std::vector<NamedAttribute> attrs2;
    attrs2.push_back(
        rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name2)));
    attrs2.push_back(
        rewriter.getNamedAttr("storage", rewriter.getStringAttr("UINT8")));
    auto power_weight_op = rewriter.create<tpu::LoadWeightOp>(
        op->getLoc(), type, ArrayRef<Value *>{wfV},
        ArrayRef<NamedAttribute>{attrs2});
    lrnOp.setOperand(2, power_weight_op);

    // remove lrn one/two/three op
    rewriter.replaceOp(lrnThreeOp, {lrnOp});
    rewriter.replaceOp(lrnTwoOp, {lrnOp});
    rewriter.replaceOp(lrnOneOp, {lrnOp});

    return matchSuccess();
  }
};

class GenLrnTablePass : public FunctionPass<GenLrnTablePass> {
public:
  explicit GenLrnTablePass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();
    auto *context = &getContext();
    OwningRewritePatternList patterns;
    patterns.insert<TpuGenLrnTablePattern>(context);
    applyPatternsGreedily(fn, patterns);
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createGenLrnTablePass() {
  return std::make_unique<GenLrnTablePass>();
}

static PassRegistration<GenLrnTablePass> pass("gen-lrn-table",
                                              "generate lrn look up table");
