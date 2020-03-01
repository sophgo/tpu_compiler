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
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

namespace {

static llvm::cl::opt<std::string> clTpuOpInfoFilename(
    "tpu-op-info-filename",
    llvm::cl::desc("dump tpu op info"),
    llvm::cl::init("-"));

class PrintTpuOpPass : public ModulePass<PrintTpuOpPass> {
public:
  explicit PrintTpuOpPass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnModule() override {
    mlir::ModuleOp module = getModule();
    //mlir::SymbolTable moduleSymTable(module);

#if 0
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
#endif

    std::unique_ptr<llvm::ToolOutputFile> file = nullptr;
    if (clTpuOpInfoFilename != "-") {
      std::string errorMessage;
      file = openOutputFile(clTpuOpInfoFilename, &errorMessage);
      if (!file) {
        llvm::errs() << errorMessage << "\n";
        exit(1);
      }
      file->keep();
      llvm::raw_ostream &file_os = file->os();

      for (auto func : module.getOps<FuncOp>()) {
        func.walk([&](Operation *op) {
          if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpCommonInterface>(op)) {
            std::string op_name = mlir::getOpName(op).str();
            file_os << op_name;
            file_os << "," << getOpLayerId(op);
            if (auto quantOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(op)) {
              file_os << "," << getOpQuant(op);
              file_os << "," << std::to_string(getOpThreshold(op));
            } else {
              file_os << "," << "NONE";
              file_os << "," << 0;
            }
            file_os << "\n";
          } else if (op->getName().getDialect().str() != "tpu"
                || isa<tpu::WeightFileOp>(op)
                || isa<tpu::LoadWeightOp>(op)
                || isa<tpu::NoneOp>(op)) {
          } else {
            llvm::errs() << "printTpuOpInfo didn't handle " << op->getName() << "\n";
            assert(false);
          }
        });
      }
    }
  }

private:
  template<typename T>
  int printTpuOpInfo(Operation *op, llvm::raw_ostream &file_os) {
      auto cast_op = llvm::dyn_cast_or_null<T>(op);
      if (cast_op) {
        std::string op_name = mlir::getOpName(op).str();
        file_os << op_name;
        if (cast_op.layer_id().hasValue())
          file_os << "," << cast_op.layer_id().getValue().getLimitedValue();
        else
          file_os << "," << "-1";
        file_os << "," << getOpQuant(op);
        file_os << "," << getOpThreshold(op);
        file_os << "\n";
        return 1;
      }
      return 0;
  }

  llvm::raw_ostream &os;
};

} // namespace

std::unique_ptr<OpPassBase<ModuleOp>> mlir::createPrintTpuOpPass() {
  return std::make_unique<PrintTpuOpPass>();
}

static PassRegistration<PrintTpuOpPass>
    pass("print-tpu-op-info",
         "Print TPU operation information.");
