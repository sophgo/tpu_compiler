//===- AssignLayerId - Implementation of Layer id assignment --------------===//
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
// This file implements the TPU layer id pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TensorFile.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"

#define DEBUG_TYPE "layer_id"

using namespace mlir;

namespace {
class AssignLayerIdPass : public FunctionPass<AssignLayerIdPass> {
public:
  explicit AssignLayerIdPass() {}

  void runOnFunction() override {
    auto fn = getFunction();
    auto *context = &getContext();
    Builder builder(context);

    uint32_t layer_id = 0;
    fn.walk([&](Operation *op) {
      if (op->getName().getDialect().str() != "tpu"
          || isa<tpu::WeightFileOp>(op)
          || isa<tpu::LoadWeightOp>(op)
          || isa<tpu::NoneOp>(op)) {
        // no need to assign
      } else if ( !failed(setOpLayerId(op, layer_id)) ) {
        LLVM_DEBUG(llvm::errs() << " layer_id: " << llvm::format("%04d", layer_id)
                                << " -> " << mlir::getOpName(op)
                                << " : " << op->getName() << "\n";);
        layer_id ++;
      } else {
        llvm::errs() << "addLayerIdAttr didn't handle " << op->getName() << "\n";
        assert(false);
      }
    });
  }
};

} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createAssignLayerIdPass() {
  return std::make_unique<AssignLayerIdPass>();
}

static PassRegistration<AssignLayerIdPass>
    pass("assign-layer-id",
         "Assign layer id to each tpu op");
