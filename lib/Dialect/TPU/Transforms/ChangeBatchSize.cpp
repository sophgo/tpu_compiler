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
// This file lower generic op to tg op.
//
//===----------------------------------------------------------------------===//

#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Passes.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/QuantizationArithmetic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tpuc/Support/TensorFile.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <sstream>
#include <fstream>
#include <math.h>

#define DEBUG_TYPE "convert_to_tg"

llvm::cl::opt<int32_t>
    clNewBatchSize("batch-size",
                llvm::cl::desc("new batch size to all Ops"),
                llvm::cl::init(1));


namespace mlir {

class ChangeBatchSizePass : public mlir::PassWrapper<ChangeBatchSizePass,
                                                     FunctionPass> {
public:
  void runOnFunction() override {
    auto fn = getFunction();
    Builder builder(&getContext());

    fn.walk([&](Operation *op) {
      if (isa<tpu::LoadWeightOp>(op) ||
          isa<tpu::WeightFileOp>(op) ||
          isa<tpu::NoneOp>(op)) {
        return;
      } else if (isa<tpu::InputOp>(op)) {
        auto argument = op->getOperand(0);
        auto oldType = argument.getType().cast<TensorType>();
        auto newType = updateTypeWithNewBatchSize(oldType);
        argument.setType(newType);
      }
      for (int i = 0; i < (int)op->getNumResults(); i++) {
        auto value = op->getResult(0);
        auto type = value.getType().cast<TensorType>();
        auto newType = updateTypeWithNewBatchSize(type);
        value.setType(newType);
      }
    });

    // update signature of function
    std::vector<mlir::Type> arguments;
    std::vector<mlir::Type> returns;
    Block &entryBlock = fn.front();
    auto returnOp = dyn_cast<ReturnOp>(entryBlock.back()).getOperation();
    for (uint32_t i = 0; i < entryBlock.getNumArguments(); ++i) {
      arguments.push_back(entryBlock.getArgument(i).getType());
    }
    for (uint32_t i = 0; i < returnOp->getNumOperands(); ++i) {
      returns.push_back(returnOp->getOperand(i).getType());
    }
    auto fnType = builder.getFunctionType(
          llvm::ArrayRef<mlir::Type>{arguments},
          llvm::ArrayRef<mlir::Type>{returns});
    fn.setType(fnType);
  }

private:
  Type updateTypeWithNewBatchSize(TensorType &oldType) {
    std::vector<int64_t> shape = oldType.getShape();
    if (shape.size() < 4) {
      shape.insert(shape.begin(), clNewBatchSize);
    } else {
      shape[0] *= clNewBatchSize;
    }
    auto eltType = oldType.getElementType();
    return RankedTensorType::get(shape, eltType);
  }
};

std::unique_ptr<mlir::Pass> createChangeBatchSizePass() {
  return std::make_unique<ChangeBatchSizePass>();
}

} // namespace mlir
