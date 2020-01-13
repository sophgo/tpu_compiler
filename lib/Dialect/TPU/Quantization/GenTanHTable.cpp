//===- GenTanHTable.cpp - Implementation of dynamice generate tanh lookup table / slope ---------===//
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
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/Support/Debug.h>
#include <float.h>

// FIXME: not declare here
#include <bmkernel/bm_kernel.h>
#include <bmkernel/bm_kernel_legacy.h>
#include <bmkernel/bm1880v2/bmkernel_1880v2.h>
#include <bmkernel/bm1880v2/1880v2_fp_convert.h>

#define DEBUG_TYPE "gen-tanh-table"
#define ENABLE_DBG (1)

using namespace mlir;

static double _gen_tanh(float x) {
  return tanh(x);
}

static void gen_tanh(int channel, int range_start, int range_end, float scale, int table_hw, float* tanh_hw) {
  // prepare channel 0
  // x [0, 127]
  // we re-scale [-8, 8] into 256
  u16 *table_data = new u16[table_hw];
  int half = table_hw / 2;
  u64 idx = 0;

  for (int i = 0; i < half; i++) {
    float _idx = idx / scale;
    double s = _gen_tanh(_idx);
    tanh_hw[idx] = s;
    table_data[idx] = convert_fp32_bf16((float)s);

    LLVM_DEBUG(
        llvm::errs()
        << llvm::format(
          "t [%lu] is %f[%d], 0x%x fp is %f d is %.8lf, input is %f\n",
          idx, convert_bf16_fp32(table_data[idx]), i,
          table_data[idx], (float)s, s, _idx
          )
        );

    idx++;
  }

  // x = -128
  double s = _gen_tanh(range_start);
  tanh_hw[idx] = s;
  table_data[idx] = convert_fp32_bf16((double)s);

  LLVM_DEBUG(
      llvm::errs()
      << llvm::format(
        "t [%lu] is %f[%d], 0x%x fp is %f d is %.8lf input is %d\n",
        idx, convert_bf16_fp32(table_data[idx]), -128,
        table_data[idx], (float)s, s, range_start
        )
      );
  idx++;

  // x [-128~-1], 2's complement
  for (int i = 1; i < half; i++) {
    float _idx = (i) / scale;
    double s = _gen_tanh(range_start + _idx);
    tanh_hw[idx] = s;
    table_data[idx] = convert_fp32_bf16((double)s);

    LLVM_DEBUG(
        llvm::errs()
        << llvm::format(
          "t [%lu] is %f[%d], 0x%x fp is %f d is %.8lf input is %f\n",
          idx, convert_bf16_fp32(table_data[idx]), 
          -127 + i, table_data[idx], (float)s, s, range_start + _idx
          )
        );

    idx++;
  }

  // duplicate channel #1 to #31

  //TODO: tensor copy
  for (int i = 1; i < channel; i++) {
    memcpy(&tanh_hw[i * table_hw], &tanh_hw[0], sizeof(float) * table_hw);
  }
}

static void gen_tanh_slope(int channel, int range_start, int range_end, float scale, int table_hw, float* tanh_hw, float* table_slope_f) {
  int half = table_hw / 2;
  u16 *table_slope = new u16[table_hw];

  for (int i = 0; i < table_hw; i++) {
    double x0 = tanh_hw[i];
    double x1 = tanh_hw[i+1];
    double delta = 1.0;
    if (i == half - 1) {
      //<! slope[127] means f(127)~f(128)
      double f = _gen_tanh(range_end);
      //u16 bf16 = convert_fp32_bf16(f);
      //x1 = convert_bf16_fp32(bf16);
      x1 = f;
    }
    else if (i == half) {
      // 128 index mean x1 is -129 and x0 is -128
      x1 = _gen_tanh(range_start - 1/scale);
      delta = -1.0;
    }
    else if (i > half) {
      x0 = tanh_hw[i];
      x1 = tanh_hw[i-1];
      delta = -1.0;
    }
    double s = (x1 - x0) / delta; // x1 already scale up
    table_slope[i] = convert_fp32_bf16((float)s);
    table_slope_f[i] = s;

    LLVM_DEBUG(
        llvm::errs()
        << llvm::format(
          "slope table [%u] = (bf16 %f double %.8lf float %f), 0x%x, %.8lf - %.8lf(%.8lf)\n",
          i, convert_bf16_fp32(table_slope[i]), s, (float)s, table_slope[i], x1, x0, x1-x0
          )
        );
  }

  // duplicate channel #1 to #31

  //TODO: tensor copy
  for (int i = 1; i < channel; i++) {
    memcpy(&table_slope_f[table_hw * i], &table_slope_f[0], sizeof(float) * table_hw);
  }
}

using namespace mlir;

namespace {

struct TpuGenTanHTablePattern : public RewritePattern {
  TpuGenTanHTablePattern(MLIRContext *context, TensorFile *weightTensorFile,
      Value* weightFileVar)
      : RewritePattern("tpu.tanh", 1, context),
        weightTensorFile_(weightTensorFile),
        weightFileVar_(weightFileVar) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto tanhOp = cast<tpu::TanHOp>(op);
    llvm::errs() << tanhOp.getOperationName() <<
      ", scale is " << tanhOp.scale().convertToFloat() << "\n";

    if (tanhOp.scale().convertToFloat() != -1) {
      llvm::errs() << tanhOp.name() << " gen already\n";
      return matchFailure();
    }

    std::string op_name = tanhOp.getAttrOfType<StringAttr>("name").getValue().str();
    auto formerOp = op->getOperand(0)->getDefiningOp();
    auto threshold_y_attr = formerOp->getAttrOfType<FloatAttr>("threshold_y").getValue().convertToFloat();

    // MUST import calibration
    assert(threshold_y_attr);
    // FIXME: not hard code here
    u32 channel = 32; //<! 1880v2 hardcode

    //<! 1880v2 hw config
    u32 table_h = 32;
    u32 table_w = 8;
    u32 table_hw = table_h * table_w;

    // NOTICE: activation ragne from -8 ~ +8 and slice to 256, dequantize to -127 ~ 127
    int range_start = (-1.0) * fabs(threshold_y_attr);
    int range_end = fabs(threshold_y_attr);
    float scale = table_hw / (1.0 * abs(range_start - range_end)); // 256 / 16 = 16

    llvm::errs() << "tabh Op: " << op_name << ", threshold_y_attr "
      << threshold_y_attr
      << ", scale is " << scale << "\n";

    // TODO: not duplicat code in quant16
    std::vector<std::unique_ptr<std::vector<float> > > weights(2);
    int weight_idx = 0;
    // 0 is input
    for (unsigned i = 1; i < tanhOp.getNumOperands(); ++i) {
      auto weight_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
          tanhOp.getOperand(i)->getDefiningOp());
      assert(weight_op);
      assert(weight_op.name().hasValue());
      auto tensor_name = weight_op.name().getValue();
      llvm::errs() << "  weight[" << i << "] : " << tensor_name << "\n";
      auto type = weight_op.getResult()->getType().cast<TensorType>();
      weights[weight_idx] = weightTensorFile_->readTensor<float>(tensor_name, type);
      weight_idx++;
      // delete the tensor from the weight file
      weightTensorFile_->deleteTensor<float>(tensor_name);
    }

    int tbl_shape = channel * table_hw;
    auto y0_table_type = tanhOp.y0_table()->getType().cast<TensorType>();
    std::vector<int64_t> y0_table_shape(y0_table_type.getShape());
    assert(y0_table_shape.size() == 4);

    std::vector<float> y0_table(tbl_shape);
    std::vector<float> scale_table(tbl_shape);

    assert(weights[0]->size() == (u32)tbl_shape);
    assert(weights[1]->size() == (u32)tbl_shape);

    // TODO: using double type for more accuracy
    gen_tanh(channel, range_start, range_end, scale, table_hw, y0_table.data());
    gen_tanh_slope(channel, range_start, range_end, scale, table_hw, y0_table.data(), scale_table.data());

    // update op
    std::vector<Value *> newOperands;
    newOperands.push_back(tanhOp.getOperand(0)); // <! 0 is input

    // add new filter and bias weight
    std::vector<std::vector<float> *> newWeights{ &y0_table, &scale_table };
    std::vector<std::vector<int64_t> > weightShapes{ y0_table_shape, y0_table_shape};

    // 2 means y0_table / scale
    for (int i = 0; i < 2; ++i) {
      auto tensor_name = op_name + "_gen_weight_" + std::to_string(i);
      llvm::errs() << "  new_weight[" << i << "] : " << tensor_name << "\n";

      auto type = RankedTensorType::get(weightShapes[i],
              FloatType::getF32(rewriter.getContext()));

      weightTensorFile_->addTensor<float>(tensor_name, newWeights[i], type);
      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
      attrs.push_back(rewriter.getNamedAttr("storage", rewriter.getStringAttr("FP32")));

      auto new_weight_op = rewriter.create<tpu::LoadWeightOp>(op->getLoc(), type,
          ArrayRef<Value *>{weightFileVar_}, ArrayRef<NamedAttribute>{attrs});
      newOperands.push_back(new_weight_op);
    }

    // replace with the new tanh op
    auto origAttrs = tanhOp.getAttrs();
    std::vector<NamedAttribute> newAttrs(origAttrs.begin(), origAttrs.end());
    newAttrs.push_back(rewriter.getNamedAttr("scale", rewriter.getF32FloatAttr(scale)));
    
    rewriter.replaceOpWithNewOp<tpu::TanHOp>(
        tanhOp, tanhOp.getResult()->getType(),
        ArrayRef<Value *>{newOperands}, ArrayRef<NamedAttribute>{newAttrs});

    return matchSuccess();

  }

  TensorFile *weightTensorFile_;
  Value* weightFileVar_;
};

class GenTanHTablePass : public FunctionPass<GenTanHTablePass> {
public:
  explicit GenTanHTablePass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    // find tensor filename
    llvm::StringRef filename;
    Value* weightFileVar;
    fn.walk([&](tpu::LoadFileOp op) {
      filename = op.filename();
      llvm::errs() << "LoadFileOp filename " << filename << "\n";
      weightFileVar = op.getResult();
    });
    auto weightTensorFile = openTensorFile(filename);

    auto *context = &getContext();

    OwningRewritePatternList patterns;
    patterns.insert<TpuGenTanHTablePattern>(context, weightTensorFile.get(), weightFileVar);
    applyPatternsGreedily(fn, patterns);

    std::string newName;
    weightTensorFile->keep(true, &newName);
    fn.walk([&](tpu::LoadFileOp op) {
      OpBuilder opBuilder(context);
      op.setAttr("filename", opBuilder.getStringAttr(newName));
      llvm::errs() << "LoadFileOp filename updated to " << newName << "\n";
    });
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createGenTanHTablePass() {
  return std::make_unique<GenTanHTablePass>();
}

static PassRegistration<GenTanHTablePass>
    pass("gen-tanh-table",
         "generate tanh look up table, y0 and slop");
