//===- ModuleInterpreter.h - Interpreter ------------------------------*- C++
//-*-===//
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

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "tpuc/Interpreter/core.h"
#include "tpuc/MachineInfo.h"
#include "tpuc/QuantizationArithmetic.h"
#include "tpuc/Support/TensorFile.h"
#include "tpuc/TPUOperationSupport.h"

#include "llvm/Support/Debug.h"

#include <fstream>
#include <iostream>
#include <mutex>

#define DEBUG_TYPE "interpreter"

namespace mlir {

class ModuleOp;
using value_map_t = DenseMap<Value, std::shared_ptr<std::vector<float>>>;
using op_kernel_list = std::vector<std::shared_ptr<OpKernel>>;

enum class DeviceMode { CPU, GPU };
// Implementation class for module interpreter.
class ModuleInterpreter {

public:
  // Interpret the given MLIR module expressed in MLIR TPU IR dialect
  explicit ModuleInterpreter(ModuleOp module)
      : mlirModule(module), weightFile_(nullptr) {

    for (FuncOp func : module.getOps<FuncOp>()) {
      // collect inputsList
      for (auto arg : func.getArguments()) {
        inputsList.push_back(arg);
      }
      // collect resultsList
      for (Block &bb : func.getBlocks()) {
        for (auto &op : bb) {
          if (isa<tpu::InputOp>(op)) {
            auto inputOp = dyn_cast<tpu::InputOp>(op);
            if (inputOp.preprocess().hasValue()) {
              data_format = inputOp.preprocess()
                                .getValue()
                                .data_format()
                                .getValue()
                                .str();
            }
          } else if (isa<ReturnOp>(op)) {
            for (auto opd : op.getOperands()) {
              resultsList.push_back(opd);
            }
          } else if (isa<tpu::WeightFileOp>(op)) {
            auto weightFileOp = dyn_cast<tpu::WeightFileOp>(op);
            weightFile_ = weightFileOp.get();
          } else if (isa<tpu::LoadWeightOp>(op)) {
            auto loadWeightOp = dyn_cast<tpu::LoadWeightOp>(op);
            LLVM_DEBUG(llvm::errs() << "LoadWeightOp"
                                    << "\n";);

            auto result = loadWeightOp.getResult();
            LLVM_DEBUG(llvm::errs() << "  result "; result.getType().dump();
                       llvm::errs() << "\n";);
            auto tensor_name = loadWeightOp.name();
            LLVM_DEBUG(llvm::errs()
                           << "  tensor_name " << tensor_name << "\n";);

            auto type = result.getType().cast<TensorType>();
            std::unique_ptr<std::vector<float>> tensor = nullptr;
            if (type.getElementType().isF32()) {
              tensor =
                  std::move(weightFile_->readTensor<float>(tensor_name, type));
            } else if (type.getElementType().isInteger(8)) {
              // TODO: we still save int8 weight as fp32 for now
              assert(0);
            } else if (type.getElementType().isBF16()) {
              auto tensor_bf16 =
                  weightFile_->readTensor<bfloat16>(tensor_name, type);

              // TODO: convert bf16 to fp32 here for now
              // as valueMapping is hardcoded as std::vector<float>
              // TODO: more generic valueMapping
              tensor = std::move(
                  std::make_unique<std::vector<float>>(tensor_bf16->size()));
              BFloat16ToFloat(tensor_bf16->data(), tensor->data(),
                              tensor_bf16->size());
            } else {
              assert(0);
            }

            valueMapping[result] = std::move(tensor);
          }
        }
      }
    }
  }
  virtual ~ModuleInterpreter() {
    if (weightFile_) {
      delete weightFile_;
    }
  }
  LogicalResult
  doRun(std::vector<int64_t> input_shape, std::vector<float> &input_vec,
        std::map<std::string, std::vector<float>> *results,
        std::map<std::string, std::vector<float>> *allTensorMap = nullptr);

  void getShape(std::map<std::string, std::vector<int64_t>> *shapeMap);

  static std::string &getCustomOpPluginFile() { return customOpPluginFile_; }

  static void setCustomOpPluginFile(std::string &file) {
    customOpPluginFile_ = file;
  }

  std::vector<std::pair<std::string, size_t>> get_input_details() {
    return input_details;
  }

  std::vector<std::string> get_output_details() { return output_details; }

  // v2
  void allocate_tensors();
  void prepareOperation(Operation &op);
  void invoke();
  void invoke(std::string name);
  std::vector<std::pair<std::string, std::string>> get_tensor_info();
  bool set_tensor(std::string name, const std::vector<float> &data);

  std::vector<float> get_tensor(std::string name);
  std::vector<int64_t> get_tensor_shape(std::string name);
  void dump(std::string name);

  std::vector<std::pair<std::string, size_t>> input_details;
  std::vector<std::string> output_details;

  std::vector<std::string> get_all_tensor_name() {
    std::lock_guard<std::mutex> lock(invoke_lock);
    std::vector<std::string> ret;
    for (auto &op : oplist) {
      ret.push_back(op->get_name());
    }
    return ret;
  };

protected:
  virtual LogicalResult runOperation(Operation &op);

private:
  LogicalResult runFunctions();
  LogicalResult runOneFunction(FuncOp func);
  LogicalResult runBlock(Block &bb);

  std::vector<Value> getInputsList() { return inputsList; }
  std::vector<Value> getResultsList() { return resultsList; }

  void updateValue(Value v, std::vector<float> &vec) {
    // deep copy
    valueMapping[v] = std::make_shared<std::vector<float>>(vec);
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
  DeviceMode device = DeviceMode::CPU;
  static std::string customOpPluginFile_;

protected:
  std::string data_format;
  value_map_t valueMapping;
  std::vector<Value> resultsList;
  std::vector<Value> inputsList;
  op_kernel_list oplist;
  std::mutex invoke_lock;
};

std::vector<std::shared_ptr<std::vector<float>>>
getOperandTensors(Operation *op, const value_map_t &valueMapping);

LogicalResult
runTpuModule(ModuleOp m, std::string pluginFile,
             std::vector<int64_t> input_shape, std::vector<float> &input_data,
             std::map<std::string, std::vector<float>> *results,
             std::map<std::string, std::vector<int64_t>> *shapeMap,
             std::map<std::string, std::vector<float>> *allTensorMap);
}; // namespace mlir

#endif // MLIR_DIALECT_TPU_MODULEINTERPRETER_H_
