//===- tpuc-interpreter.cpp - MLIR TPU Dialect Interpreter Driver---------------------===//
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


#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Passes.h"
#include "tpuc/Interpreter.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));

static llvm::cl::opt<std::string> inputTensorFilename("tensor-in",
    llvm::cl::desc("Input Tensor Filename"),
    llvm::cl::init("-"));

static llvm::cl::opt<std::string> outputTensorFilename("tensor-out",
    llvm::cl::desc("Output Tensor Filename"),
    llvm::cl::init("-"));

static llvm::cl::opt<std::string> dumpAllTensorFilename(
    "dump-all-tensor",
    llvm::cl::desc("dump all tensor into a npz file"),
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

int main(int argc, char **argv) {
  llvm::errs() << argv[0] << " version: " << MLIR_VERSION << "\n";
  llvm::InitLLVM y(argc, argv);
  
  MLIRContext context;
  auto &registry = context.getDialectRegistry();
  registry.insert<tpu::TPUDialect,
                  mlir::StandardOpsDialect>();

  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR TPU interpreter driver\n");

  auto m = parseMLIRInput(inputFilename, &context);
  if (!m) {
    llvm::errs() << "could not parse the input IR\n";
    return 1;
  }
  
  auto inputTF = openTensorFile(inputTensorFilename);
  std::vector<std::vector<float> *> input_tensors;
  std::vector<std::vector<int64_t> > input_shapes;
  if (failed(inputTF->readAllTensors(input_tensors, input_shapes))) {
    llvm::errs() << "cound not read input tensor\n";
    return EXIT_FAILURE;
  }
  //// support one input only for now
  //assert(input_tensors.size() == 1);
  //assert(input_shapes.size() == 1);
  int64_t shape_size = 0;
  for (auto s : input_shapes) {
    shape_size += std::accumulate(s.begin(), s.end(), 1, std::multiplies<int64_t>());
  }

  std::vector<float> input_vec(shape_size);

  // flatten shapes/tensors to one array
  shape_size = 0;
  for (uint64_t i = 0; i < input_shapes.size(); i++) {
    int64_t size = std::accumulate(input_shapes[i].begin(), input_shapes[i].end(), 1, std::multiplies<int64_t>());
    std::memcpy(input_vec.data() + shape_size, input_tensors[i]->data(), size * sizeof(float));
    shape_size += size;
  }

  std::map<std::string, std::vector<float> > results;
  std::map<std::string, std::vector<int64_t> > shapeMap;
  std::map<std::string, std::vector<float> > allTensorMap;

  // its find for give input_shapes[0] cuz input shape is recorded in .mlir file
  if (failed(runTpuModule(m.get(), "", input_shapes[0], input_vec,
                          &results, &shapeMap, &allTensorMap)))
    return EXIT_FAILURE;

  if (outputTensorFilename != "-") {
    auto outputTF = openOutputTensorFile(outputTensorFilename);
    for (auto it = results.begin(); it != results.end(); it++ ) {
      auto shape = shapeMap[it->first];
      outputTF->addTensor(it->first, &it->second, shape);
    }
    outputTF->keep();
  }

  if (dumpAllTensorFilename != "-") {
    // dump all values
    auto allTensorTF = openOutputTensorFile(dumpAllTensorFilename);
    for (auto it = allTensorMap.begin(); it != allTensorMap.end(); it++ ) {
      auto shape = shapeMap[it->first];
      allTensorTF->addTensor(it->first, &it->second, shape);
    }
    allTensorTF->keep();
  }

  int exitCode = EXIT_SUCCESS;
  return exitCode;
}
