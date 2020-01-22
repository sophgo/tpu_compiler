//===- GenDivTable.cpp - Implementation of dynamice generate tanh lookup table / slope ---------===//
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
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/Support/Debug.h>

#include <float.h>

#define DEBUG_TYPE "gen-Div-table"

using namespace mlir;

namespace {

struct TpuGenDivTablePattern : public RewritePattern {
  TpuGenDivTablePattern(MLIRContext *context, TensorFile *weightTensorFile,
      Value* weightFileVar)
      : RewritePattern("tpu.div", 1, context),
        weightTensorFile_(weightTensorFile),
        weightFileVar_(weightFileVar) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {

    auto DivOp = cast<tpu::DivOp>(op);
    std::vector<std::unique_ptr<std::vector<float> > > weights(1);

    std::string op_name = DivOp.getAttrOfType<StringAttr>("name").getValue().str();

    if(DivOp.has_table() == true){
      LLVM_DEBUG(llvm::errs() << DivOp.name() << " gen already\n";);
      return matchFailure();
    }
  
    int npu_num = 32; //<! 1880v2 hardcode

    //<! 1880v2 hw config
    int table_h = 16;
    int table_w = 16;
    int table_hw = table_h * table_w;

    int tbl_shape = npu_num * table_hw;
    std::vector<float> y0_table(tbl_shape);


    float threshold_x = getPreviousOpThreshold(op);
    float threshold_y = getOpThreshold(op);


    for (int n = 0; n < npu_num; n++) {
      for (int idx = 0; idx < table_hw; ++idx) {
          char lutInput = static_cast<char>(idx);
          float index = lutInput * threshold_x / 127.0;
          float lutOutput = 1.0 /(index) * 127.0 / threshold_y;
          int lutOutputI32 = std::floor(lutOutput + 0.5);
          lutOutputI32 = (lutOutputI32 > 127)
                             ? 127
                             : (lutOutputI32 < -128) ? -128 : lutOutputI32;

        y0_table[n * table_hw + idx] = lutOutputI32;
      }
    }

    // update op
    std::vector<Value *> newOperands;
    newOperands.push_back(op->getOperand(0));

    // add new filter and bias weight
    std::vector<float> newWeights = y0_table ;
    std::vector<int64_t> weightShape{1, npu_num, table_h, table_w};

    auto tensor_name = op_name + "_gen_weight";
    LLVM_DEBUG(llvm::errs() << "  new_weight: " << tensor_name << "\n";);

    auto type = RankedTensorType::get(weightShape,
            FloatType::getF32(rewriter.getContext()));
   
    weightTensorFile_->addTensor<float>(tensor_name, newWeights.data(), type);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(tensor_name)));
    attrs.push_back(
        rewriter.getNamedAttr("storage", rewriter.getStringAttr("INT8")));
    auto new_weight_op = rewriter.create<tpu::LoadWeightOp>(op->getLoc(), type,
        ArrayRef<Value *>{weightFileVar_}, ArrayRef<NamedAttribute>{attrs});
    newOperands.push_back(new_weight_op);

    DivOp.setAttr("has_table", rewriter.getBoolAttr("true"));
    rewriter.replaceOpWithNewOp<tpu::DivOp>(
        DivOp, DivOp.getResult()->getType(),
        ArrayRef<Value *>{newOperands}, ArrayRef<NamedAttribute>{DivOp.getAttrs()});

    return matchSuccess();
  }

  TensorFile *weightTensorFile_;
  Value* weightFileVar_;
};

class GenDivTablePass : public FunctionPass<GenDivTablePass> {
public:
  explicit GenDivTablePass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    // find tensor filename
    llvm::StringRef filename;
    Value* weightFileVar;
    fn.walk([&](tpu::LoadFileOp op) {
      filename = op.filename();
      LLVM_DEBUG(llvm::errs() << "LoadFileOp filename " << filename << "\n";);
      weightFileVar = op.getResult();
    });
    auto weightTensorFile = openTensorFile(filename);

    auto *context = &getContext();

    OwningRewritePatternList patterns;
    patterns.insert<TpuGenDivTablePattern>(context, weightTensorFile.get(), weightFileVar);
    applyPatternsGreedily(fn, patterns);

    std::string newName;
    weightTensorFile->keep(true, &newName);
    fn.walk([&](tpu::LoadFileOp op) {
      OpBuilder opBuilder(context);
      op.setAttr("filename", opBuilder.getStringAttr(newName));
      LLVM_DEBUG(llvm::errs() << "LoadFileOp filename updated to " << newName << "\n";);
    });
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createGenDivTablePass() {
  return std::make_unique<GenDivTablePass>();
}

static PassRegistration<GenDivTablePass>
    pass("gen-div-table",
         "generate div look up table, y0");