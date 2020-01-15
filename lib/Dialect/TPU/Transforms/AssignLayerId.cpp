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
      int processed = 0;
      processed += addLayerIdAttr<tpu::BatchNormOp>(builder, layer_id, op);
      processed += addLayerIdAttr<tpu::ConcatOp>(builder, layer_id, op);
      processed += addLayerIdAttr<tpu::Conv2DOp>(builder, layer_id, op);
      processed += addLayerIdAttr<tpu::CropOp>(builder, layer_id, op);
      processed += addLayerIdAttr<tpu::EltwiseOp>(builder, layer_id, op);
      processed += addLayerIdAttr<tpu::FullyConnectedOp>(builder, layer_id, op);
      processed += addLayerIdAttr<tpu::InputOp>(builder, layer_id, op);
      processed += addLayerIdAttr<tpu::Pool2DOp>(builder, layer_id, op);
      processed += addLayerIdAttr<tpu::PowerOp>(builder, layer_id, op);
      processed += addLayerIdAttr<tpu::PReluOp>(builder, layer_id, op);
      processed += addLayerIdAttr<tpu::ReluOp>(builder, layer_id, op);
      processed += addLayerIdAttr<tpu::ReshapeOp>(builder, layer_id, op);
      processed += addLayerIdAttr<tpu::ScaleOp>(builder, layer_id, op);
      processed += addLayerIdAttr<tpu::SigmoidOp>(builder, layer_id, op);
      processed += addLayerIdAttr<tpu::SliceOp>(builder, layer_id, op);
      processed += addLayerIdAttr<tpu::SoftmaxOp>(builder, layer_id, op);
      processed += addLayerIdAttr<tpu::TanHOp>(builder, layer_id, op);
      processed += addLayerIdAttr<tpu::UpsampleOp>(builder, layer_id, op);
      if (op->getName().getDialect().str() != "tpu"
          || isa<tpu::QuantizationOp>(op)
          || isa<tpu::DequantizationOp>(op)
          || isa<tpu::LoadWeightOp>(op)
          || isa<tpu::LoadFileOp>(op)) {
        processed = 1;
      }
      if (!processed) {
        llvm::errs() << "addLayerIdAttr didn't handle " << op->getName() << "\n";
        assert(false);
      }
    });
  }

private:
  template<typename T>
  int addLayerIdAttr(Builder &builder, uint32_t &layer_id, Operation *op) {
      auto cast_op = llvm::dyn_cast_or_null<T>(op);
      if (cast_op) {
        std::string op_name = cast_op.name().getValue().str();
        llvm::errs() << op_name << " -> layer_id: " << layer_id << "\n";
        cast_op.setAttr("layer_id", builder.getI32IntegerAttr(layer_id));
        layer_id++;
        return 1;
      }
      return 0;
  }
};

} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createAssignLayerIdPass() {
  return std::make_unique<AssignLayerIdPass>();
}

static PassRegistration<AssignLayerIdPass>
    pass("assign-layer-id",
         "Assign layer id to each tpu op");
