//===- tpuc-interpreter.cpp - MLIR TPU Dialect Interpreter
// Driver---------------------===//
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

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"
#include "tpuc/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include <unordered_map>

using namespace mlir;

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    inputTensorFilename("tensor-in", llvm::cl::desc("Input Tensor Filename"),
                        llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    outputTensorFilename("tensor-out", llvm::cl::desc("Output Tensor Filename"),
                         llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    dumpAllTensorFilename("dump-all-tensor",
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

  DialectRegistry registry;
  registry.insert<tpu::TPUDialect, mlir::StandardOpsDialect>();
  MLIRContext context(registry);

  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "MLIR TPU interpreter driver\n");

  auto module = parseMLIRInput(inputFilename, &context);
  if (!module) {
    llvm::errs() << "could not parse the input IR\n";
    return EXIT_FAILURE;
  }

  auto inputTensorFile = openTensorFile(inputTensorFilename);
  std::vector<std::vector<float> *> input_tensors;
  std::vector<std::vector<int64_t>> input_shapes;
  if (failed(inputTensorFile->readAllTensors(input_tensors, input_shapes))) {
    llvm::errs() << "cound not read input tensor\n";
    return EXIT_FAILURE;
  }

  int64_t shape_size = 0;
  for (auto s : input_shapes) {
    shape_size +=
        std::accumulate(s.begin(), s.end(), 1, std::multiplies<int64_t>());
  }

  std::vector<float> input_data(shape_size);

  // merge mutli input array to one array
  shape_size = 0;
  for (uint64_t i = 0; i < input_shapes.size(); i++) {
    int64_t size =
        std::accumulate(input_shapes[i].begin(), input_shapes[i].end(), 1,
                        std::multiplies<int64_t>());
    std::memcpy(input_data.data() + shape_size, input_tensors[i]->data(),
                size * sizeof(float));
    shape_size += size;
  }

  std::map<std::string, std::vector<float>> results;
  std::map<std::string, std::vector<int64_t>> shapeMap;
  std::map<std::string, std::vector<float>> allTensorMap;

  // if mlir with mutli inputs, we will use mlir to check mutli input,
  // set null vector
  std::vector<int64_t> input_shape =
      input_shapes.size() == 1 ? input_shapes[0] : std::vector<int64_t>();

  auto interpreter_ = std::make_unique<ModuleInterpreter>(module.get());
  interpreter_->allocate_tensors();
  auto input_details = interpreter_->get_input_details();
  if (input_details.size() != input_tensors.size()) {
    llvm::errs() << "Input number not same, needed is " << input_details.size()
                 << ", get " << input_tensors.size() << "\n";
    llvm_unreachable("please check input npz");
  }
  for (size_t i = 0; i < input_tensors.size(); i++) {
    if (input_tensors[i]->size() != input_details[i].second) {
      llvm::errs() << "input tensor size not same, needed is "
                   << input_details[i].second << ", get "
                   << input_tensors[i]->size() << "\n";
      llvm_unreachable("please check input npz");
    }
    std::vector<float> data(input_tensors[i]->begin(), input_tensors[i]->end());
    interpreter_->set_tensor(input_details[i].first, data);
  }
  interpreter_->invoke();
  if (outputTensorFilename != "-") {
    auto outputTensorFile = openOutputTensorFile(outputTensorFilename);
    auto output_details = interpreter_->get_output_details();
    for (auto &output_name : output_details) {
      std::vector<float> output_data = interpreter_->get_tensor(output_name);
      std::vector<int64_t> shape = interpreter_->get_tensor_shape(output_name);
      (void)outputTensorFile->addTensor(output_name, output_data.data(), shape);
    }
    outputTensorFile->keep();
  }

  if (dumpAllTensorFilename != "-") {
    // dump all values
    auto allTensorTensorFile = openOutputTensorFile(dumpAllTensorFilename);
    auto all_tensor_names = interpreter_->get_all_tensor_name();
    for (auto &tensor_name : all_tensor_names) {
      std::vector<float> output_data = interpreter_->get_tensor(tensor_name);
      std::vector<int64_t> shape = interpreter_->get_tensor_shape(tensor_name);
      (void)allTensorTensorFile->addTensor(tensor_name, output_data.data(), shape);
    }
    allTensorTensorFile->keep();
  }

  for (auto &it : input_tensors) {
    it->clear();
    it->shrink_to_fit();
    delete (it);
  }
  input_tensors.clear();
  input_tensors.shrink_to_fit();

  interpreter_.reset();
  return 0;
}
