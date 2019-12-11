//===- TpuOpPrint.cpp - Implementation of TPU Op Print ---------===//
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
// This file implements the TPU dialect OP pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

class PrintTpuOpPass : public ModulePass<PrintTpuOpPass> {
public:
  explicit PrintTpuOpPass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnModule() override {
    mlir::ModuleOp module = getModule();
    //mlir::SymbolTable moduleSymTable(module);

    os << "Modules:\n";
    os << "-----------------------\n";
    //auto mainFn = moduleSymTable.lookup<mlir::FuncOp>("main");
    for (mlir::FuncOp func :
         llvm::make_early_inc_range(module.getOps<mlir::FuncOp>())) {
      os << func.getName() << "\n";
      FunctionType type = func.getType();
      //type.print(os);
      type.dump();
      os << "\n";
    }
    os << "\n";

    os << "Funcs:\n";
    os << "-----------------------\n";
    for (auto func : module.getOps<FuncOp>()) {
      os << func.getName() << "\n";
      func.walk([&](Operation *op) {
        os << " > " << op->getName() << "\n";
      });
    }
    os << "\n";

    os << "Module walk Conv2DOp:\n";
    os << "-----------------------\n";
    module.walk([&](mlir::tpu::Conv2DOp op) {
      os << " > " << op.getOperationName() << "\n";
      //op.dump();
      //os << "\n";
    });
    os << "\n";

    os << "Funcs walk Conv2DOp:\n";
    os << "-----------------------\n";
    for (auto func : module.getOps<FuncOp>()) {
      os << func.getName() << "\n";
      func.walk([&](mlir::tpu::Conv2DOp op) {
        os << " > " << op.getOperationName() << "\n";
      });
      func.walk([&](mlir::tpu::FullyConnectedOp op) {
        os << " > " << op.getOperationName() << "\n";
      });
    }
    os << "\n";
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

std::unique_ptr<OpPassBase<ModuleOp>> mlir::createPrintTpuOpPass() {
  return std::make_unique<PrintTpuOpPass>();
}

static PassRegistration<PrintTpuOpPass>
    pass("print-tpu-op",
         "Print TPU operations.");
