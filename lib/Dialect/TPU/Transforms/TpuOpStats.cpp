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
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/StandardOps/Ops.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

class PrintTpuOpStatsPass : public ModulePass<PrintTpuOpStatsPass> {
public:
  explicit PrintTpuOpStatsPass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnModule() override {
    ModuleManager moduleManager(getModule());

    os << "Modules:\n";
    os << "-----------------------\n";
    for (auto &module : getModule()) {
      module.walk([&](Operation *op) {
        FuncOp funcOp = llvm::dyn_cast_or_null<FuncOp>(op);
        if (funcOp) {
          os << op->getName() << " : " << funcOp.getName() << "\n";
          FunctionType type = funcOp.getType();
          //type.print(os);
          type.dump();
          os << "\n";
        } else {
          os << " > " << op->getName() << "\n";
        }
      });
    }
    os << "\n";

    os << "Funcs:\n";
    os << "-----------------------\n";
    for (auto func : getModule().getOps<FuncOp>()) {
      os << func.getName() << "\n";
      func.walk([&](Operation *op) {
        os << " > " << op->getName() << "\n";
      });
    }
    os << "\n";

    os << "Module walk Conv2DOp:\n";
    os << "-----------------------\n";
    for (auto &module : getModule()) {
      module.walk<mlir::tpu::Conv2DOp>([&](mlir::tpu::Conv2DOp op) {
        os << " > " << op.getOperationName() << "\n";
      });
    }
    os << "\n";

    os << "Funcs walk Conv2DOp:\n";
    os << "-----------------------\n";
    for (auto func : getModule().getOps<FuncOp>()) {
      os << func.getName() << "\n";
      func.walk<mlir::tpu::Conv2DOp>([&](mlir::tpu::Conv2DOp op) {
        os << " > " << op.getOperationName() << "\n";
      });
    }
    os << "\n";
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

ModulePassBase *mlir::createPrintTpuOpStatsPass() {
  return new PrintTpuOpStatsPass();
}

static PassRegistration<PrintTpuOpStatsPass>
    pass("print-tpu-op-stats",
         "Print statistics of TPU operations.");
