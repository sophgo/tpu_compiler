//===- ModuleInterpreter.h - Interpreter ------------------------------*- C++ -*-===//
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
// This header file defines prototypes that expose interpreter constructors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TPU_MODULEINTERPRETER_H_
#define MLIR_DIALECT_TPU_MODULEINTERPRETER_H_

#include "mlir/IR/Module.h"

#include <iostream>
#include <fstream>

namespace mlir {

class ModuleOp;

// Implementation class for module interpreter.
class ModuleInterpreter {
public:
  template <typename T = ModuleInterpreter>
  static LogicalResult runModule(ModuleOp m,
      std::vector<std::vector<float> *> &inputs,
      std::vector<std::vector<float> *> &outputs) {

    T interpreter(m, inputs, outputs);
    if (failed(interpreter.runFunctions()))
      return failure();

    return success();
  }

protected:
  // Interpret the given MLIR module expressed in MLIR TPU IR dialect
  explicit ModuleInterpreter(ModuleOp module,
      std::vector<std::vector<float> *> &inputs,
      std::vector<std::vector<float> *> &outputs)
      : mlirModule(module), inputs(inputs), outputs(outputs) {}
  virtual ~ModuleInterpreter() {}

  virtual LogicalResult runOperation(Operation &op);

private:
  LogicalResult runFunctions();
  LogicalResult runOneFunction(FuncOp func);
  LogicalResult runBlock(Block &bb);

  // Original and translated module.
  ModuleOp mlirModule;
  std::vector<std::vector<float> *> &inputs;
  std::vector<std::vector<float> *> &outputs;

  // weight file input stream
  std::unique_ptr<std::ifstream> weight_is;

protected:
  llvm::DenseMap<Value *, std::unique_ptr<std::vector<float> > > valueMapping;
};

} // namespace mlir

#endif // MLIR_DIALECT_TPU_MODULEINTERPRETER_H_
