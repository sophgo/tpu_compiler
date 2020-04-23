//===- GenReciprocalTable.cpp - Implementation of dynamice generate tanh lookup table / slope ---------===//
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
#include <bmkernel/bm_kernel.h>
#include <bmkernel/bm_kernel_legacy.h>
#include <bmkernel/bm1880v2/bmkernel_1880v2.h>
#include <bmkernel/bm1880v2/1880v2_fp_convert.h>
#define DEBUG_TYPE "gen-Reciprocal-table"

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
#define TABLE_HW_INT8 (TABLE_H_INT8*TABLE_W_INT8)
#define TBL_SHAPE_INT8 (TABLE_HW_INT8*NPU_NUM)
#define TABLE_HW_BF16 (TABLE_H_BF16*TABLE_W_BF16)
#define TBL_SHAPE_BF16 (TABLE_HW_BF16*NPU_NUM)

// <! gen reciprocal f(x) = 1/x
static double _gen_reciprocal(int base, int p) {
  // y = x ^ -1
  double f = (double) (pow(base, -1 * p));
  return f;
}


void bf16_gen_reciprocal(uint16_t *table_data) {

  int exp_start = EXP_START;
  int half = TABLE_HW_BF16/2;
  int table_hw = TABLE_HW_BF16;
  uint64_t idx = 0;

  // prepare channel 0
  double s = 0.0;
  // 0^-1 is invalid, use positive/negtive max value: 0x7F7F / 0xFF7F
  table_data[idx] = 0x7F80; //<! convert to 0x7F7F

  idx++;

  // > 0, exp from 0 -62 -61 ..  62  63
  for (int i = 0; i < half - 1; i++) {
    int shift = (exp_start + i);
    bool is_odd = (shift % 2);
    float exp = shift;
    if (is_odd) {
      exp = exp - 1;
    }

    double s = _gen_reciprocal(2, exp);
    //FloatToBFloat16((float*)&s,&table_data[idx],(size_t)1);
    table_data[idx] = convert_fp32_bf16(s);
    idx++;
  }

  s = _gen_reciprocal(2, -0);
  //table_data[idx] = convert_fp32_bf16(s);
  FloatToBFloat16((float*)&s,&table_data[idx],(size_t)1);
  table_data[idx] = 0x7F80; //<! convert to 0x7F7F

  idx++;

  // < 0, exp from 0 -62 -61 ..  62  63
  for (int i = 0; i < half - 1; i++) {
    int shift = (exp_start + i);
    bool is_odd = (shift % 2);
    float exp = shift;
    if (is_odd) {
      exp = exp - 1;
    }

    double s = -1 * _gen_reciprocal(-2, exp);
    //table_data[idx] = convert_fp32_bf16(s);
    //FloatToBFloat16((float*)&s,&table_data[idx],(size_t)1);
    table_data[idx] = convert_fp32_bf16(s);
    idx++;
  }

  // duplicate channel #1 to #31
  //TODO: tensor copy
  for (uint32_t i = 1; i < NPU_NUM; i++) {
    memcpy(&table_data[i * table_hw], &table_data[0], sizeof(uint16_t) * table_hw);
  }
}

void bf16_gen_reciprocal_mantissa(uint16_t* table_mantissa) {


  int half = TABLE_HW_BF16/2;
  int table_hw = TABLE_HW_BF16;

  int idx = 0;
  double d;
  for (int i = 0; i < half; i++) {
    d = 1 + i * 1 / 128.0;
    d = (double) pow(d, -1);
    //FloatToBFloat16((float*)&d,&table_mantissa[128+idx],(size_t)1);
    table_mantissa[128+idx] = convert_fp32_bf16(d);
    //13=2^3x1.625=(2^2)x(2^1x1.625)
    d = 2 * (1 + i * 1 / 128.0);
    d = (double) pow(d, -1);
    //FloatToBFloat16((float*)&d,&table_mantissa[idx],(size_t)1);
    table_mantissa[idx] = convert_fp32_bf16(d);
    idx++;
  }


  // duplicate channel #1 to #31
  //TODO: tensor copy
  for (int i = 1; i < NPU_NUM; i++) {
    memcpy(&table_mantissa[table_hw * i], &table_mantissa[0], sizeof(uint16_t) * table_hw);
  }
}

struct TpuGenReciprocalTablePattern : public RewritePattern {
  TpuGenReciprocalTablePattern(MLIRContext *context)
      : RewritePattern("tpu.reciprocal", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    TensorFile *wTF = getWeightTensorFile(op);
    Value *wfV = getWeightFileValue(op);

    auto reciprocalOp = cast<tpu::ReciprocalOp>(op);
    std::vector<std::unique_ptr<std::vector<float> > > weights(1);

    std::string op_name = reciprocalOp.getAttrOfType<StringAttr>("name").getValue().str();
    if(reciprocalOp.has_table() == true){
      LLVM_DEBUG(llvm::errs() << reciprocalOp.name() << " gen already\n";);
      return matchFailure();
    }
    std::vector<float> y0_table(TBL_SHAPE_INT8);

    std::vector<uint16_t> table_data_lut_bf16(TBL_SHAPE_BF16);
    std::vector<uint16_t> table_data_mantissa_lut_bf16(TBL_SHAPE_BF16);

    std::vector<float> table_data_lut(TBL_SHAPE_BF16);
    std::vector<float> table_data_mantissa_lut(TBL_SHAPE_BF16);

  if (reciprocalOp.getOpQuant() == "INT8") {

    float threshold_x = getPreviousOpThreshold(op);
    float threshold_y = getOpThreshold(op);


    for (int n = 0; n < NPU_NUM; n++) {
      for (int idx = 0; idx < TABLE_HW_INT8; ++idx) {
          char lutInput = static_cast<char>(idx);
          float index = lutInput * threshold_x / 127.0;
          float lutOutput = 1.0 /(index) * 127.0 / threshold_y;
          int lutOutputI32 = std::floor(lutOutput + 0.5);
          lutOutputI32 = (lutOutputI32 > 127)
                             ? 127
                             : (lutOutputI32 < -128) ? -128 : lutOutputI32;

        y0_table[n * TABLE_HW_INT8 + idx] = lutOutputI32;
      }
    }
  }else if(reciprocalOp.getOpQuant() == "BF16"){
    LLVM_DEBUG(llvm::errs() << " op name: " << reciprocalOp.name()
                            << "gen BF16 sqrt table." << "\n");
    bf16_gen_reciprocal(table_data_lut_bf16.data());
    LLVM_DEBUG(llvm::errs() << " op name: " << reciprocalOp.name()
                            << "gen BF16 sqrt mantissa table." << "\n");

    bf16_gen_reciprocal_mantissa(table_data_mantissa_lut_bf16.data());

    std::copy(table_data_lut_bf16.data(), table_data_lut_bf16.data() + TBL_SHAPE_BF16,
              table_data_lut.data() );
    std::copy(table_data_mantissa_lut_bf16.data(),
              table_data_mantissa_lut_bf16.data() + TBL_SHAPE_BF16,
              table_data_mantissa_lut.data());

  }else{
    assert(0&&"not support");
  }

    // update op
    std::vector<Value *> newOperands;
    newOperands.push_back(op->getOperand(0));

  if (reciprocalOp.getOpQuant() == "INT8") {

    // add new filter and bias weight
    std::vector<float> newWeights = y0_table ;
    std::vector<int64_t> weightShape{1, NPU_NUM, TABLE_H_INT8, TABLE_W_INT8};

    auto tensor_name = op_name + "_gen_weight";
    LLVM_DEBUG(llvm::errs() << "  new_weight: " << tensor_name << "\n";);

    auto type = RankedTensorType::get(weightShape,
            FloatType::getF32(rewriter.getContext()));

    wTF->addTensor<float>(tensor_name, newWeights.data(), type);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
    attrs.push_back(
        rewriter.getNamedAttr("storage", rewriter.getStringAttr("UINT8")));
    auto new_weight_op = rewriter.create<tpu::LoadWeightOp>(op->getLoc(), type,
        ArrayRef<Value *>{wfV}, ArrayRef<NamedAttribute>{attrs});
    newOperands.push_back(new_weight_op);

    reciprocalOp.setAttr("has_table", rewriter.getBoolAttr("true"));
  }else if(reciprocalOp.getOpQuant() == "BF16"){

    std::vector<std::vector<float>> newWeights = {table_data_lut, table_data_mantissa_lut};
    std::vector<int64_t> weightShapes = {1, NPU_NUM, TABLE_H_BF16, TABLE_W_BF16};

    for (int i = 0; i < 2; ++i) {
      auto tensor_name = op_name + "_gen_weight_" + std::to_string(i);
      LLVM_DEBUG(llvm::errs() << "  new_weight[" << i << "] : " << tensor_name << "\n");

      auto type = RankedTensorType::get(
          weightShapes, FloatType::getF32(rewriter.getContext()));

      wTF->addTensor<float>(tensor_name, newWeights.at(i).data(), type);
      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
      attrs.push_back(rewriter.getNamedAttr("storage", rewriter.getStringAttr("BF16")));
      reciprocalOp.setAttr("has_table", rewriter.getBoolAttr("true"));

      auto new_weight_op = rewriter.create<tpu::LoadWeightOp>(
          op->getLoc(), type, ArrayRef<Value *>{wfV},
          ArrayRef<NamedAttribute>{attrs});
      newOperands.push_back(new_weight_op);
    }

  }else {
    assert(0&&"not support");
  }

  rewriter.replaceOpWithNewOp<tpu::ReciprocalOp>(
        reciprocalOp, reciprocalOp.getResult()->getType(),
        ArrayRef<Value *>{newOperands}, ArrayRef<NamedAttribute>{reciprocalOp.getAttrs()});

    return matchSuccess();
  }
};
class GenReciprocalTablePass : public FunctionPass<GenReciprocalTablePass> {
public:
  explicit GenReciprocalTablePass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();
    auto *context = &getContext();
    OwningRewritePatternList patterns;
    patterns.insert<TpuGenReciprocalTablePattern>(context);
    applyPatternsGreedily(fn, patterns);
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createGenReciprocalTablePass() {
  return std::make_unique<GenReciprocalTablePass>();
}

static PassRegistration<GenReciprocalTablePass>
    pass("gen-reciprocal-table",
         "generate reciprocal look up table, y0");
