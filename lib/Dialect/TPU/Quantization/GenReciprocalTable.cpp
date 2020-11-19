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
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/Passes.h"
#include "tpuc/MachineInfo.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "tpuc/Support/TensorFile.h"
#include "tpuc/TPUOperationSupport.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/Support/Debug.h>
#include "tpuc/QuantizationArithmetic.h"
#include "tpuc/NativeCpuImplementation.h"

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
#define TABLE_H_BF16 32
#define TABLE_W_BF16 8
#define TABLE_H_INT8 16
#define TABLE_W_INT8 16
#define TABLE_HW_INT8 (TABLE_H_INT8*TABLE_W_INT8)
#define TBL_SHAPE_INT8 (TABLE_HW_INT8*MInfo::lane_num)
#define TABLE_HW_BF16 (TABLE_H_BF16*TABLE_W_BF16)
#define TBL_SHAPE_BF16 (TABLE_HW_BF16*MInfo::lane_num)

struct TpuGenReciprocalTablePattern : public RewritePattern {
  TpuGenReciprocalTablePattern(MLIRContext *context)
      : RewritePattern("tpu.reciprocal", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    TensorFile *wTF = getWeightTensorFile(op);
    Value wfV = getWeightFileValue(op);

    auto reciprocalOp = cast<tpu::ReciprocalOp>(op);
    std::vector<std::unique_ptr<std::vector<float> > > weights(1);

    std::string op_name = reciprocalOp.getAttrOfType<StringAttr>("name").getValue().str();
    if(reciprocalOp.has_table() == true){
      LLVM_DEBUG(llvm::errs() << reciprocalOp.name() << " gen already\n";);
      return failure();
    }
    std::vector<float> y0_table(TBL_SHAPE_INT8);

    std::vector<uint16_t> table_data_lut_bf16(TBL_SHAPE_BF16);
    std::vector<uint16_t> table_data_mantissa_lut_bf16(TBL_SHAPE_BF16);

    std::vector<float> table_data_lut(TBL_SHAPE_BF16);
    std::vector<float> table_data_mantissa_lut(TBL_SHAPE_BF16);

  if (reciprocalOp.getOpQuant() == "INT8") {

    float threshold_x = getPreviousOpThreshold(op);
    float threshold_y = getOpThreshold(op);


    for (uint32_t n = 0; n < MInfo::lane_num; n++) {
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
    bf16_gen_reciprocal(EXP_START, EXP_END, TABLE_HW_BF16, table_data_lut_bf16.data());
    LLVM_DEBUG(llvm::errs() << " op name: " << reciprocalOp.name()
                            << "gen BF16 sqrt mantissa table." << "\n");

    bf16_gen_reciprocal_mantissa(EXP_START, EXP_END, TABLE_HW_BF16, table_data_mantissa_lut_bf16.data());

    for (uint32_t i = 1; i < MInfo::lane_num; i++) {
      memcpy(table_data_mantissa_lut_bf16.data() + i * TABLE_HW_BF16, table_data_mantissa_lut_bf16.data(), sizeof(uint16_t) * TABLE_HW_BF16);
      memcpy(table_data_lut_bf16.data() + i * TABLE_HW_BF16, table_data_lut_bf16.data(), sizeof(uint16_t) * TABLE_HW_BF16);
    }

    std::copy(table_data_lut_bf16.data(), table_data_lut_bf16.data() + TBL_SHAPE_BF16,
              table_data_lut.data() );
    std::copy(table_data_mantissa_lut_bf16.data(),
              table_data_mantissa_lut_bf16.data() + TBL_SHAPE_BF16,
              table_data_mantissa_lut.data());

  }else{
    assert(0&&"not support");
  }

    // update op
    std::vector<Value> newOperands;
    newOperands.push_back(op->getOperand(0));

  if (reciprocalOp.getOpQuant() == "INT8") {

    // add new filter and bias weight
    std::vector<float> newWeights = y0_table ;
    std::vector<int64_t> weightShape{1, MInfo::lane_num, TABLE_H_INT8, TABLE_W_INT8};

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
        ArrayRef<Value>{wfV}, ArrayRef<NamedAttribute>{attrs});
    newOperands.push_back(new_weight_op);

    reciprocalOp.setAttr("has_table", rewriter.getBoolAttr("true"));
  }else if(reciprocalOp.getOpQuant() == "BF16"){

    std::vector<std::vector<float>> newWeights = {table_data_lut, table_data_mantissa_lut};
    std::vector<int64_t> weightShapes = {1, MInfo::lane_num, TABLE_H_BF16, TABLE_W_BF16};

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
          op->getLoc(), type, ArrayRef<Value>{wfV},
          ArrayRef<NamedAttribute>{attrs});
      newOperands.push_back(new_weight_op);
    }

  }else {
    assert(0&&"not support");
  }

  rewriter.replaceOpWithNewOp<tpu::ReciprocalOp>(
        reciprocalOp, reciprocalOp.getResult().getType(),
        ArrayRef<Value>{newOperands}, ArrayRef<NamedAttribute>{reciprocalOp.getAttrs()});

    return success();
  }
};
class GenReciprocalTablePass : public mlir::PassWrapper<GenReciprocalTablePass, FunctionPass> {
public:
  explicit GenReciprocalTablePass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();
    auto *context = &getContext();
    OwningRewritePatternList patterns;
    patterns.insert<TpuGenReciprocalTablePattern>(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::createGenReciprocalTablePass() {
  return std::make_unique<GenReciprocalTablePass>();
}

static PassRegistration<GenReciprocalTablePass>
    pass("gen-reciprocal-table",
         "generate reciprocal look up table, y0");
