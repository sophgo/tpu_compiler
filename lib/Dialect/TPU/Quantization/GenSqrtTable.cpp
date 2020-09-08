//===- GenSqrtTable.cpp - Implementation of dynamice generate tanh lookup table / slope ---------===//
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
#define DEBUG_TYPE "gen-sqrt-table"

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
#define TABLE_HW_BF16 (TABLE_H_BF16*TABLE_W_BF16)
#define TBL_SHAPE_INT8 (TABLE_HW_INT8*NPU_NUM)
#define TBL_SHAPE_BF16 (TABLE_HW_BF16*NPU_NUM)

// <! gen invert sqrt
static double _gen_sqrt(int base, int p) {
  // y = x ^ 0.5
  double f = (double) (pow(base, p * 0.5));
  return f;
}

static void gen_sqrt(uint16_t *table_data, uint64_t table_size) {
  //<! 32*8 table, duplicate `channel` times;

  int half = table_size / CHANNEL / 2;
  uint64_t idx = 0;
  assert(table_size);
  assert(half == 128);

  // prepare channel 0
  float s = 0.0;
  FloatToBFloat16(&s,&table_data[idx],(size_t)1); // 0^0.5 = 0
  table_data[idx] = convert_fp32_bf16(s);

  idx++;

  // > 0, exp from 0 -62 -61 ..  62  63
  for (int i = 0; i < half; i++) {
    //float exp = round((exp_start + i) / 2) * 2;
    int shift = (EXP_START + i);
    bool is_odd = (shift % 2);
    float exp = shift;
    if (is_odd) {
      exp = exp - 1;
    }

    double s = _gen_sqrt(2, exp);
    //table_data[idx] = convert_fp32_bf16(s);
    //FloatToBFloat16((float*)&s,&table_data[idx],(size_t)1);
    table_data[idx] = convert_fp32_bf16(s);
    idx++;
  }

  //std::vector<float> table_data_dump();

  for (uint32_t i = 1; i < CHANNEL; i++) {
    memcpy(&table_data[i * TABLE_HW_BF16], &table_data[0], sizeof(uint16_t) * TABLE_HW_BF16);
  }
}

static void gen_sqrt_mantissa(uint16_t* table_mantissa, uint64_t table_size) {

  uint32_t half = table_size / CHANNEL / 2;
  assert(half == 128);

  int idx = 0;
  double d;
  for (uint32_t i = 0; i < half; i++) {
    d = 1 + i * 1 / 128.0;
    d = (double) pow(d, 0.5);
   // FloatToBFloat16((float*)&d,&table_mantissa[idx+128],(size_t)1);
    table_mantissa[128+idx] = convert_fp32_bf16(d);
    LLVM_DEBUG(llvm::errs() <<","<< "table_mantissa["<<idx+128<<"] = " <<table_mantissa[128+idx];);

    //13=2^3x1.625=(2^2)x(2^1x1.625)
    d = 2 * (1 + i * 1 / 128.0);

    d = (double) pow(d, 0.5);
    //FloatToBFloat16((float*)&d,&table_mantissa[idx],(size_t)1);
    //table_mantissa[idx] = convert_fp32_bf16(d);
    table_mantissa[idx] = convert_fp32_bf16(d);
    LLVM_DEBUG(llvm::errs() <<","<< "table_mantissa["<<idx<<"] = " <<table_mantissa[idx];);
    idx++;
  }


  // duplicate channel #1 to #31
  //TODO: tensor copy
  for (int i = 1; i < CHANNEL; i++) {
    memcpy(&table_mantissa[TABLE_HW_BF16 * i], &table_mantissa[0], sizeof(uint16_t) * TABLE_HW_BF16);
  }
}

struct TpuGenSqrtTablePattern : public RewritePattern {
  TpuGenSqrtTablePattern(MLIRContext *context)
      : RewritePattern("tpu.sqrt", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    TensorFile *wTF = getWeightTensorFile(op);
    Value *wfV = getWeightFileValue(op);
    auto sqrtOp = cast<tpu::SqrtOp>(op);
    std::vector<std::unique_ptr<std::vector<float> > > weights(1);

    std::string op_name = sqrtOp.getAttrOfType<StringAttr>("name").getValue().str();

    if(sqrtOp.has_table() == true){
      LLVM_DEBUG(llvm::errs() << sqrtOp.name() << " gen already\n";);
      return matchFailure();
    }

    std::vector<float> y0_table(TBL_SHAPE_INT8);

/*    std::vector<uint16_t> table_data(table_hw);
    std::vector<uint16_t> table_data_mantissa(table_hw);
*/
    std::vector<uint16_t> table_data_lut_bf16(TBL_SHAPE_BF16);
    std::vector<uint16_t> table_data_mantissa_lut_bf16(TBL_SHAPE_BF16);

    std::vector<float> table_data_lut(TBL_SHAPE_BF16);
    std::vector<float> table_data_mantissa_lut(TBL_SHAPE_BF16);

  if (sqrtOp.getOpQuant() == "INT8") {
    float threshold_x = getPreviousOpThreshold(op);
    float threshold_y = getOpThreshold(op);
    for (int n = 0; n < NPU_NUM; n++) {
      for (int idx = 0; idx < TABLE_HW_INT8; ++idx) {
        char lutInput = static_cast<char>(idx);
        float index = lutInput * threshold_x / 127.0;
        float lutOutput = pow(index,0.5) * 127.0 / threshold_y;
        int lutOutputI32 = std::floor(lutOutput + 0.5);
        lutOutputI32 = (lutOutputI32 > 127)
                           ? 127
                           : (lutOutputI32 < -128) ? -128 : lutOutputI32;
        y0_table[n * TABLE_HW_INT8 + idx] = lutOutputI32;
      }
    }
  }else if(sqrtOp.getOpQuant() == "BF16"){
    LLVM_DEBUG(llvm::errs() << " op name: " << sqrtOp.name()
                            << "gen BF16 sqrt table." << "\n");
    gen_sqrt(table_data_lut_bf16.data(), TBL_SHAPE_BF16);
    LLVM_DEBUG(llvm::errs() << " op name: " << sqrtOp.name()
                            << "gen BF16 sqrt mantissa table." << "\n");
    gen_sqrt_mantissa(table_data_mantissa_lut_bf16.data(), TBL_SHAPE_BF16);

    std::copy(table_data_lut_bf16.data(), table_data_lut_bf16.data() + TBL_SHAPE_BF16,
              table_data_lut.data() );
    std::copy(table_data_mantissa_lut_bf16.data(),
              table_data_mantissa_lut_bf16.data() + TBL_SHAPE_BF16,
              table_data_mantissa_lut.data());
  }else {
      llvm::errs() << " op name: " << sqrtOp.name()
                   << ",quant_type: " << sqrtOp.quant() << "\n";
      assert(0 && "not support sqrt type");
  }


  std::vector<Value *> newOperands;
  newOperands.push_back(op->getOperand(0));

  // update op
  if (sqrtOp.getOpQuant() == "INT8") {

    std::vector<float> newWeights = y0_table ;
    std::vector<int64_t> weightShape{1, NPU_NUM, TABLE_H_INT8, TABLE_W_INT8};

    auto tensor_name = op_name + "_gen_weight";
    LLVM_DEBUG(llvm::errs() << "  new_weight: " << tensor_name << "\n";);

    auto type = RankedTensorType::get(weightShape,
            FloatType::getF32(rewriter.getContext()));


    wTF->addTensor<float>(tensor_name, newWeights.data(), type);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
    attrs.push_back(rewriter.getNamedAttr("storage", rewriter.getStringAttr("UINT8")));
    auto new_weight_op = rewriter.create<tpu::LoadWeightOp>(op->getLoc(), type,
        ArrayRef<Value *>{wfV}, ArrayRef<NamedAttribute>{attrs});
    newOperands.push_back(new_weight_op);

    sqrtOp.setAttr("has_table", rewriter.getBoolAttr("true"));

  }else if(sqrtOp.getOpQuant() == "BF16"){

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
      sqrtOp.setAttr("has_table", rewriter.getBoolAttr("true"));
      attrs.push_back(rewriter.getNamedAttr("storage",rewriter.getStringAttr("BF16")));
      auto new_weight_op = rewriter.create<tpu::LoadWeightOp>(
          op->getLoc(), type, ArrayRef<Value *>{wfV},
          ArrayRef<NamedAttribute>{attrs});

      newOperands.push_back(new_weight_op);

    }

  }else {
      llvm::errs() << "type: " << sqrtOp.getOpQuant().str()
                   << " is not support.";
      assert(false);
  }

    rewriter.replaceOpWithNewOp<tpu::SqrtOp>(
        sqrtOp, sqrtOp.getResult()->getType(),
        ArrayRef<Value *>{newOperands}, ArrayRef<NamedAttribute>{sqrtOp.getAttrs()});


    return matchSuccess();
  }
};

class GenSqrtTablePass : public FunctionPass<GenSqrtTablePass> {
public:
  explicit GenSqrtTablePass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();
    auto *context = &getContext();
    OwningRewritePatternList patterns;
    patterns.insert<TpuGenSqrtTablePattern>(context);
    applyPatternsGreedily(fn, patterns);
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createGenSqrtTablePass() {
  return std::make_unique<GenSqrtTablePass>();
}

static PassRegistration<GenSqrtTablePass>
    pass("gen-sqrt-table",
         "generate sqrt look up table, y0");
