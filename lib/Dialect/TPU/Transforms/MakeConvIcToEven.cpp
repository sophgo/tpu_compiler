//===- TpuOpStats.cpp - Implementation of TPU Op Stats ---------===//
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
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/TPU/TPUTensorSupport.h"

using namespace mlir;

namespace {

template <typename OpTy>
struct TpuRefactorOddIcConvPattern : public RewritePattern {
  TpuRefactorOddIcConvPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto convOp = cast<OpTy>(op);
    llvm::errs() << convOp.getOperationName() << ":" << getOpName(convOp)<< "\n";

    // auto shape = convOp.input()->getType().cast<TensorType>().getShape();//Refactor convOp
    int64_t inputSize;
    std::vector<int64_t> shape;
    getTensorShapeAndSize(convOp.input(), shape, inputSize);

    // auto filter_type = convOp.filter()->getType().template cast<TensorType>();
    int64_t filterSize;
    std::vector<int64_t> filterShape;
    getTensorShapeAndSize(convOp.filter(), filterShape, filterSize);
    std::vector<int64_t> newFilterShape(filterShape);
    if (filterShape.size() != 4) {
      return matchFailure();
    }
    newFilterShape[1] +=1;
    int in = shape[0];
    int ic = shape[1];
    int ih = shape[2];
    int iw = shape[3];
    int kn = filterShape[0];
    int kc = filterShape[1];
    int kh = filterShape[filterShape.size() - 2];
    int kw = filterShape[filterShape.size() - 1];
    if(kc % 2 != 0) {
      int new_ic = kc + 1;
      //Fill filter data to zero
      TensorFile *wTF = getWeightTensorFile(op);
      auto filter = readAndDeleteWeightTensor<int8_t>(convOp.filter(), wTF);
      int64_t filterSize;
      getTensorShapeAndSize(convOp.filter(), filterShape, filterSize);
      assert(filterSize == (int64_t)filter->size());

      int64_t newFilterSize = kn * new_ic * kh * kw;
      auto new_filter = std::make_shared<std::vector<int8_t> >(newFilterSize);
      int8_t* filterData = filter->data();
      int8_t* newFilter_data = new_filter->data();
      llvm::errs() << "Filter shape:(" << kn << ", " << kc << ", " << kh << ", " <<  kw << ")\n";
      for(int n_counter = 0;  n_counter < kn; n_counter++)
        for(int h_counter = 0;  h_counter < kh; h_counter++)
          for(int w_counter = 0;  w_counter < kw; w_counter++)
            for(int c_counter = 0;  c_counter < new_ic; c_counter++){
            int index_old = c_counter  +
                            w_counter * kc +
                            h_counter * kc * kw +
                            n_counter * kh * kc * kw;
            int index_new = c_counter +
                            w_counter * new_ic +
                            h_counter * new_ic * kw +
                            n_counter * kh * new_ic * kw;
            if(c_counter == kc){
              newFilter_data[index_new] = 0;
            } else {
              newFilter_data[index_new] = filterData[index_old];
            }
          }
      addWeightTensorAndUpdateWeightOp<int8_t>(convOp.getOperand(1),
          "", *new_filter, newFilterShape, "INT8", wTF);

      //Reshpae mlir file
      // input ic will remain the same, pad ic on weight and change shape for weight
      // auto type = RankedTensorType::get({in, new_ic, ih, iw}, IntegerType::get(8, rewriter.getContext()));
      // convOp.input()->setType(type);//rewrite inputShape
      auto filterType = RankedTensorType::get({kn, new_ic, kh, kw}, IntegerType::get(8, (rewriter.getContext())));
      convOp.filter()->setType(filterType);//rewrite inputShape
      convOp.setAttr("do_ic_alignment", rewriter.getBoolAttr(true));
      return matchSuccess();
    }

    return matchFailure();
  }
};

class RefactorOddIcConvPass : public FunctionPass<RefactorOddIcConvPass> {
public:
  explicit RefactorOddIcConvPass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();
    OwningRewritePatternList patterns;
    auto *context = &getContext();

    patterns.clear();

    patterns.insert<TpuRefactorOddIcConvPattern<tpu::TG_INT8_PC_Conv2DOp>,
                    TpuRefactorOddIcConvPattern<tpu::TG_INT8_PT_Conv2DOp>>(context);
    applyPatternsGreedily(fn, patterns);
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createRefactorOddIcConvPass() {
  return std::make_unique<RefactorOddIcConvPass>();
}

static PassRegistration<RefactorOddIcConvPass>
    pass("conv-ic-alignment",
         "Enable padding odd ic to even to enable double conv for conv layer");
