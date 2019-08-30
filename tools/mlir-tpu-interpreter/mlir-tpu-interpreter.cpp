//===- mlir-tpu-interpreter.cpp - MLIR TPU Dialect Interpreter Driver---------------------===//
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
// Main entry point to a command line utility that executes an MLIR file on the
// TPU by interpreting MLIR to certain function calls and executing later.
//
//===----------------------------------------------------------------------===//


#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/Dialect/TPU/Interpreter.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));

static OwningModuleRef parseMLIRInput(StringRef inputFilename,
                                      MLIRContext *context) {
  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  return OwningModuleRef(parseSourceFile(sourceMgr, context));
}

namespace mlir {

int TpuInterpreterMain(
    int argc, char **argv,
    llvm::function_ref<LogicalResult(mlir::ModuleOp)> mlirTransformer) {

  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR TPU interpreter driver\n");

  MLIRContext context;
  auto m = parseMLIRInput(inputFilename, &context);
  if (!m) {
    llvm::errs() << "could not parse the input IR\n";
    return 1;
  }

  if (mlirTransformer)
    if (failed(mlirTransformer(m.get())))
      return EXIT_FAILURE;

  std::vector<float> input(1*3*224*224);
  std::vector<float> output(1*1000);
  std::fill (std::begin(input), std::end(input), 1.0f);
  std::vector<std::vector<float> *> inputs({&input});
  std::vector<std::vector<float> *> outputs({&output});

  if (failed(runTpuModule(m.get(), inputs, outputs)))
    return EXIT_FAILURE;

  int exitCode = EXIT_SUCCESS;
  return exitCode;
}

} // namespace mlir

static LogicalResult runMLIRPasses(ModuleOp m) {
  // As we gradually lower, the IR is inconsistent between passes. So do not
  // verify inbetween.
  PassManager pm(/*verifyPasses=*/false);

  pm.addPass(createPrintTpuOpStatsPass());

  if (failed(pm.run(m)))
    return failure();

  if (failed(m.verify()))
    return failure();

  return success();
}

// TODO: merge with JitRunnerMain()
int main(int argc, char **argv) {
  return mlir::TpuInterpreterMain(argc, argv, &runMLIRPasses);
}
