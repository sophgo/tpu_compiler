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

#include "mlir/Dialect/TPU/QuantizationArithmetic.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Support/TensorFile.h"

#include "llvm/Support/Debug.h"

#include <iostream>
#include <fstream>

#define DEBUG_TYPE "interpreter"

namespace mlir {

class ModuleOp;
typedef DenseMap<Value *, std::shared_ptr<std::vector<float> > > value_map_t;

// Implementation class for module interpreter.
class ModuleInterpreter {

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
          } else if (isa<tpu::WeightFileOp>(op)) {
            auto weightFileOp = dyn_cast<tpu::WeightFileOp>(op);
            weightFile_ = weightFileOp.get();
          } else if (isa<tpu::LoadWeightOp>(op)) {
            auto loadWeightOp = dyn_cast<tpu::LoadWeightOp>(op);
            LLVM_DEBUG(llvm::errs() << "LoadWeightOp" << "\n";);

            auto result = loadWeightOp.getResult();
            LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
            assert(loadWeightOp.name().hasValue());
            auto tensor_name = loadWeightOp.name().getValue();
            LLVM_DEBUG(llvm::errs() << "  tensor_name " << tensor_name << "\n";);

            auto type = result->getType().cast<TensorType>();
            std::unique_ptr<std::vector<float> > tensor= nullptr;
            if (type.getElementType().isF32()) {
              tensor = std::move(weightFile_->readTensor<float>(tensor_name, type));
            } else if (type.getElementType().isInteger(8)) {
              // TODO: we still save int8 weight as fp32 for now
              assert(0);
            } else if (type.getElementType().isBF16()) {
              auto tensor_bf16 = weightFile_->readTensor<bfloat16>(tensor_name, type);

              // TODO: convert bf16 to fp32 here for now
              // as valueMapping is hardcoded as std::vector<float>
              // TODO: more generic valueMapping
              tensor = std::move(std::make_unique<std::vector<float> >(tensor_bf16->size()));
              BFloat16ToFloat(tensor_bf16->data(), tensor->data(), tensor_bf16->size());
            } else {
              assert(0);
            }

            valueMapping[result] = std::move(tensor);
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
  TensorFile *weightFile_;

protected:
  value_map_t valueMapping;
  std::vector<Value *> resultsList;
  std::vector<Value *> inputsList;
};

} // namespace mlir

#endif // MLIR_DIALECT_TPU_MODULEINTERPRETER_H_
