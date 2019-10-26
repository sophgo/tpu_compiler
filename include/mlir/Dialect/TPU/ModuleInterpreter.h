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
#include "mlir/Support/TensorFile.h"

#include <iostream>
#include <fstream>

namespace mlir {

class ModuleOp;

// Implementation class for module interpreter.
class ModuleInterpreter {
  typedef std::map<Value *, std::shared_ptr<std::vector<float> > > value_map_t;

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
      : mlirModule(module), inputs_(inputs), outputs_(outputs) {}
  virtual ~ModuleInterpreter() {}

  virtual LogicalResult runOperation(Operation &op);

private:
  LogicalResult runFunctions();
  LogicalResult runOneFunction(FuncOp func);
  LogicalResult runBlock(Block &bb);

  std::vector<std::shared_ptr<std::vector<float> > >
      getOperandTensors(Operation &opInst, value_map_t &valueMapping);

  // Original and translated module.
  ModuleOp mlirModule;
  std::vector<std::vector<float> *> &inputs_;
  std::vector<std::vector<float> *> &outputs_;

  // weight file input stream
  std::unique_ptr<std::ifstream> weight_is;
  std::unique_ptr<TensorFile> weight_file;

protected:
  value_map_t valueMapping;
};

} // namespace mlir

#endif // MLIR_DIALECT_TPU_MODULEINTERPRETER_H_
