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

#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Function.h"
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
  // Interpret the given MLIR module expressed in MLIR TPU IR dialect
  explicit ModuleInterpreter(ModuleOp module)
      : mlirModule(module) {

    for (FuncOp func : module.getOps<FuncOp>()) {
      // collect inputsList
      for (auto arg : func.getArguments()) {
        inputsList.push_back(arg);
      }
      // collect resultsList
      for (Block &bb : func.getBlocks()) {
        for (auto &op : bb) {
          if (isa<ReturnOp>(op)) {
            for (auto opd: op.getOperands()) {
              resultsList.push_back(opd);
            }
          } else if (isa<tpu::LoadFileOp>(op)) {
            auto loadFileOp = dyn_cast<tpu::LoadFileOp>(op);
            auto filename = loadFileOp.getAttrOfType<StringAttr>("filename").getValue();
            auto filename_tensorfile = llvm::sys::path::stem(filename).str() + ".npz";
            weightFile_ = openInputTensorFile(filename_tensorfile);
          }
        }
      }
    }
  }
  virtual ~ModuleInterpreter() {}

  template <typename T = ModuleInterpreter>
  static LogicalResult runModule(ModuleInterpreter *interpreter,
      std::vector<int64_t> input_shape, std::vector<float> &input_vec,
      std::map<std::string, std::vector<float> > *results,
      std::map<std::string, std::vector<float> > *allTensorMap = nullptr) {
    return interpreter->doRun(input_shape, input_vec, results, allTensorMap);
  }

  template <typename T = ModuleInterpreter>
  static LogicalResult runModule(ModuleOp m,
      std::vector<int64_t> input_shape, std::vector<float> &input_vec,
      std::map<std::string, std::vector<float> > *results,
      std::map<std::string, std::vector<float> > *allTensorMap = nullptr) {

    T interpreter(m);

    return interpreter.doRun(input_shape, input_vec, results, allTensorMap);
  }

protected:
  virtual LogicalResult runOperation(Operation &op);

private:
  LogicalResult runFunctions();
  LogicalResult runOneFunction(FuncOp func);
  LogicalResult runBlock(Block &bb);
  LogicalResult doRun(std::vector<int64_t> input_shape, std::vector<float> &input_vec,
                      std::map<std::string, std::vector<float> > *results,
                      std::map<std::string, std::vector<float> > *allTensorMap = nullptr);

  std::vector<std::shared_ptr<std::vector<float> > >
      getOperandTensors(Operation &opInst, value_map_t &valueMapping);

  std::vector<Value *> getInputsList() { return inputsList; }
  std::vector<Value *> getResultsList() { return resultsList; }

  void updateValue(Value *v, std::vector<float> &vec) {
    // deep copy
    valueMapping[v] = std::make_shared<std::vector<float> >(vec);
  }

  value_map_t getResults() {
    value_map_t results;
    for (auto res : getResultsList()) {
      results[res] = valueMapping[res];
    }
    return results;
  }

  value_map_t getValueMap() { return valueMapping; }
  void reset(void) { valueMapping.clear(); }

  // Original and translated module.
  ModuleOp mlirModule;

  // weight file input stream
  std::unique_ptr<TensorFile> weightFile_;

protected:
  value_map_t valueMapping;
  std::vector<Value *> resultsList;
  std::vector<Value *> inputsList;
};

} // namespace mlir

#endif // MLIR_DIALECT_TPU_MODULEINTERPRETER_H_
