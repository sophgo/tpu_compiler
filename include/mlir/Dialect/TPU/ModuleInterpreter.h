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
using namespace std;

namespace mlir {

class ModuleOp;

// Implementation class for module interpreter.
class ModuleInterpreter {
  typedef std::map<Value *, std::shared_ptr<std::vector<float> > > value_map_t;

public:
  template <typename T = ModuleInterpreter>
  static LogicalResult runModule(ModuleOp m,
      std::vector<int64_t> input_shape, std::vector<float> &input_vec,
      std::map<std::string, std::vector<float> > *results,
      std::map<std::string, std::vector<float> > *allTensorMap = nullptr) {

    T interpreter(m);

    // set inputs
    auto inputs = interpreter.getInputsList();
    assert(inputs.size() == 1);
    std::vector<int64_t> shape = inputs[0]->getType().template cast<TensorType>().getShape();

    assert(input_shape == shape);
    assert((int64_t)input_vec.size() == std::accumulate(shape.begin(), shape.end(), 1,
                                                        std::multiplies<int64_t>()));
    interpreter.updateValue(inputs[0], input_vec);

    // inference
    if (failed(interpreter.runFunctions()))
      return failure();

    // get results
    assert(results);
    value_map_t resultsMap = interpreter.getResults();
    for (auto it = resultsMap.begin(); it != resultsMap.end(); it++) {
      auto op = it->first->getDefiningOp();
      assert(op);
      auto vec = it->second.get();
      assert(vec);
      // deep copy
      (*results)[getOpName(op).str()] = *vec;
    }

    // get all tensor data if needed
    if (allTensorMap) {
      value_map_t valueMap = interpreter.getValueMap();
      for (auto it = valueMap.begin(); it != valueMap.end(); it++) {
        auto op = it->first->getDefiningOp();
        if (!op) {
          //it->first->dump();
          continue;
        }
        if (auto loadWeightOp = dyn_cast<tpu::LoadWeightOp>(op)) {
          continue;
        }
        auto vec = it->second.get();
        assert(vec);
        // deep copy
        (*allTensorMap)[getOpName(op).str()] = *vec;
      }
    }

    return success();
  }

protected:
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
          }
        }
      }
    }
  }
  virtual ~ModuleInterpreter() {}

  virtual LogicalResult runOperation(Operation &op);

private:
  LogicalResult runFunctions();
  LogicalResult runOneFunction(FuncOp func);
  LogicalResult runBlock(Block &bb);

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
