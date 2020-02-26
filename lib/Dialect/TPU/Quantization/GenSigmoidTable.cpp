//===- GenSigmoidTable.cpp - Implementation of dynamice generate tanh lookup table / slope ---------===//
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
#include "mlir/Dialect/TPU/QuantizationArithmetic.h"
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

#include <float.h>

#define DEBUG_TYPE "gen-sigmoid-table"

extern const int BF16_TABLE_START = -8;
extern const int BF16_TABLE_END = 8;
using std::vector;

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 0.5 * tanh(0.5 * x) + 0.5;
}

static void gen_bf16_sigmoid_table(int start, int end, int table_hw, float *simgoid_table){
  int half = table_hw / 2;
  int table_idx = 0;
  int range = abs(end - start);
  float interval = (float)range / (float)table_hw;
  double x_value;
  double y_value;

  // Set idx [0 , 127] fp32 and bf16 data
  for (int i = 0; i < half; i++) {
    x_value = i * interval;
    y_value = sigmoid(x_value);
    simgoid_table[table_idx] = y_value;
    table_idx++;
  }
  // set idx 128 fp32 and bf16 data
  simgoid_table[table_idx] = sigmoid(start);

  ++table_idx;
  // set idx 129 to 256, 2's complment
  for (int i = 1; i < half; i++) {
    x_value = start + i * interval;
    y_value = sigmoid(x_value);
    simgoid_table[table_idx] = y_value;
    table_idx++;
  }
}

static void gen_bf16_sigmoid_slope_table(int start, int end, int table_hw, float *sigmoid_table,
                         float *sigmoid_slope_table) {
  int range = abs(end - start);
  float interval = (float)range / (float)table_hw;
  int half = table_hw / 2;
  for (int i = 0; i < table_hw; ++i){
    double x0 = sigmoid_table[i];
    double x1 = sigmoid_table[i + 1];
    double delta = 1.0;
    if(i == half - 1){
      x1 = sigmoid(end);
    } else if (i == half){
      // idx = 128, x0 is -128, x1 is -129
      x1 = sigmoid(start - interval);
      delta = -1.0;
    } else if (i > half){
      x0 = sigmoid_table[i];
      x1 = sigmoid_table[i - 1];
      delta = -1.0;
    }
    float slope = (x1 - x0) / delta;
    sigmoid_slope_table[i] = slope;
  }
}

using namespace mlir;

namespace {

struct TpuGenSigmoidTablePattern : public RewritePattern {
  TpuGenSigmoidTablePattern(MLIRContext *context)
      : RewritePattern("tpu.sigmoid", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    TensorFile *wTF = getWeightTensorFile(op);
    Value *wfV = getWeightFileValue(op);
    auto sigOp = cast<tpu::SigmoidOp>(op);
    std::vector<std::unique_ptr<std::vector<float>>> weights(1);

    std::string op_name =
        sigOp.getAttrOfType<StringAttr>("name").getValue().str();

    if (sigOp.has_table() == true) {
      llvm::errs() << sigOp.name() << " gen already\n";
      return matchFailure();
    }
    int npu_num = 32; //<! 1880v2 hardcode

    //<! 1880v2 hw config
    int table_h;
    int table_w;
    int table_hw;

    int tbl_shape;
    vector<float> y0_table;
    vector<float> y0_slope_table; // use in bf16

    if (sigOp.getOpQuant() == "INT8") {
      //<! 1880v2 hw int8 config
      table_h = 16;
      table_w = 16;
      table_hw = table_h * table_w;

      tbl_shape = npu_num * table_hw;
      y0_table.resize(tbl_shape);

      float threshold_x = getPreviousOpThreshold(op);
      float threshold_y = getOpThreshold(op);

      // input: 0~127, -128~ -1, Y=1/(1+EXP(-X*thx/128)) * 128/thy
      // output:0~127, negative is invalid
      for (int n = 0; n < npu_num; n++) {
        for (int idx = 0; idx < table_hw; ++idx) {
          char lutInput = static_cast<char>(idx);
          float index = -lutInput * threshold_x / 127.0;
          float lutOutput = 1.0 / (1 + std::exp(index)) * 127.0 / threshold_y;
          int lutOutputI32 = std::floor(lutOutput + 0.5);
          lutOutputI32 = (lutOutputI32 > 127)
                             ? 127
                             : (lutOutputI32 < -128) ? -128 : lutOutputI32;
          y0_table[n * table_hw + idx] = lutOutputI32;
        }
      }
    } else if (sigOp.getOpQuant() == "BF16") {
      assert(0 && "wait for refactor");
    //   //<! 1880v2 hw bf16 config
    //   table_h = 32;
    //   table_w = 8;
    //   table_hw = table_h * table_w;
    //   tbl_shape = npu_num * table_hw;
    //   y0_table.resize(tbl_shape);
    //   y0_slope_table.resize(tbl_shape);
    //   vector<float> y0_fp32_table(table_hw);
    //   vector<float> y0_fp32_slope_table(table_hw);
    //   vector<uint16_t> y0_bf16_table(table_hw);
    //   vector<uint16_t> y0_bf16_slope_table(table_hw);


    //   gen_bf16_sigmoid_table(BF16_TABLE_START, BF16_TABLE_END, table_hw,
    //                          y0_fp32_table.data());

    //   gen_bf16_sigmoid_slope_table(BF16_TABLE_START, BF16_TABLE_END, table_hw,
    //                                y0_fp32_table.data(),
    //                                y0_fp32_slope_table.data());

    //   // convert fp32 to bf16
    //   FloatToBFloat16(y0_fp32_table.data(),
    //                   y0_bf16_table.data(), table_hw);
    //   FloatToBFloat16(y0_fp32_slope_table.data(),
    //                   y0_bf16_slope_table.data(), table_hw);

    //   // copy bf16 data to float table
    //   for (int i = 0; i < npu_num; ++i){
    //     std::copy(y0_bf16_table.data(), y0_bf16_table.data() + table_hw,
    //               y0_table.data() + i * table_hw);
    //     std::copy(y0_bf16_slope_table.data(),
    //               y0_bf16_slope_table.data() + table_hw,
    //               y0_slope_table.data() + i * table_hw);
    //   }
    //   for (int i = 0; i < table_hw; ++i){
    //     LLVM_DEBUG(llvm::errs() << llvm::format(
    //                    "sigmoid lut table [%d] = bf16 %f float %f\n", i,
    //                    y0_table.at(i), y0_fp32_table.at(i)));
    //   }
    //   for (int i = 0; i < table_hw; ++i) {
    //     LLVM_DEBUG(llvm::errs() << llvm::format(
    //                    "sigmoid slope table [%d] = bf16 %f float %f\n", i,
    //                    y0_slope_table.at(i), y0_fp32_slope_table.at(i)));
    //   }
    // } else {
    //   llvm::errs() << " op name: " << sigOp.name()
    //                << ",quant_type: " << sigOp.quant() << "\n";
    //   assert(0 && "not support sigmoid type");
    }

    // update op
    std::vector<Value *> newOperands;
    newOperands.push_back(op->getOperand(0));
    if (sigOp.getOpQuant() == "INT8") {
      // add new filter and bias weight
      vector<float> newWeights = y0_table;
      vector<int64_t> weightShape{1, npu_num, table_h, table_w};

      auto tensor_name = op_name + "_gen_weight";
      llvm::errs() << "  new_weight: " << tensor_name << "\n";

      auto type = RankedTensorType::get(weightShape,
              FloatType::getF32(rewriter.getContext()));

      wTF->addTensor<float>(tensor_name, newWeights.data(), type);
      std::vector<NamedAttribute> attrs;
      attrs.push_back(
          rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
      attrs.push_back(
          rewriter.getNamedAttr("storage", rewriter.getStringAttr("UINT8")));

      auto new_weight_op = rewriter.create<tpu::LoadWeightOp>(
          op->getLoc(), type, ArrayRef<Value *>{wfV},
          ArrayRef<NamedAttribute>{attrs});
      newOperands.push_back(new_weight_op);

    } else if (sigOp.getOpQuant() == "BF16") {
      vector<vector<float>> newWeights = {y0_table, y0_slope_table};
      vector<int64_t> weightShapes = {1, npu_num, table_h, table_w};
      for (int i = 0; i < 2; ++i) {
        auto tensor_name = op_name + "_gen_weight_" + std::to_string(i);
        llvm::errs() << "  new_weight[" << i << "] : " << tensor_name << "\n";

        auto type = RankedTensorType::get(
            weightShapes, FloatType::getF32(rewriter.getContext()));

        wTF->addTensor<float>(tensor_name, newWeights.at(i).data(), type);
        vector<NamedAttribute> attrs;
        attrs.push_back(
            rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
        attrs.push_back(
            rewriter.getNamedAttr("storage", rewriter.getStringAttr("UINT16")));

        auto new_weight_op = rewriter.create<tpu::LoadWeightOp>(
            op->getLoc(), type, ArrayRef<Value *>{wfV},
            ArrayRef<NamedAttribute>{attrs});
        newOperands.push_back(new_weight_op);
      }
    } else {
      llvm::errs() << "type: " << sigOp.quant()
                   << " is not support.";
      assert(false);
    }
    sigOp.setAttr("has_table", rewriter.getBoolAttr("true"));
    rewriter.replaceOpWithNewOp<tpu::SigmoidOp>(
        sigOp, sigOp.getResult()->getType(), ArrayRef<Value *>{newOperands},
        ArrayRef<NamedAttribute>{sigOp.getAttrs()});
    return matchSuccess();
  }
};

class GenSigmoidTablePass : public FunctionPass<GenSigmoidTablePass> {
public:
  explicit GenSigmoidTablePass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();
    auto *context = &getContext();
    OwningRewritePatternList patterns;
    patterns.insert<TpuGenSigmoidTablePattern>(context);
    applyPatternsGreedily(fn, patterns);
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createGenSigmoidTablePass() {
  return std::make_unique<GenSigmoidTablePass>();
}

static PassRegistration<GenSigmoidTablePass>
    pass("gen-sigmoid-table",
         "generate sigmoid look up table, y0");
