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
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "tpuc/Interpreter/core.h"
#include "tpuc/MachineInfo.h"
#include "tpuc/QuantizationArithmetic.h"
#include "tpuc/Support/TensorFile.h"
#include "tpuc/TPUOperationSupport.h"

#include "llvm/Support/Debug.h"

#include <fstream>
#include <iostream>
#include <mutex>
#include <omp.h>
#include <unordered_map>

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
      : mlirModule(module), weightFile_(nullptr) {}
  virtual ~ModuleInterpreter() {
    if (weightFile_) {
      delete weightFile_;
    }
  }

  Value findWeightValue(std::string name) {
    for (FuncOp func : mlirModule.getOps<FuncOp>()) {
      for (Block &bb : func.getBlocks()) {
        for (auto &op : bb) {
          if (isa<tpu::LoadWeightOp>(op)) {
            auto loadWeightOp = dyn_cast<tpu::LoadWeightOp>(op);
            std::string weight_name = loadWeightOp.name().str();
            if (weight_name == name) {
              return loadWeightOp.getResult();
            }
          }
        }
      }
    }
    return Value(nullptr);
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
  std::unordered_map<std::string,
                     std::pair<std::vector<float>, std::vector<int64_t>>>
  getWeightData() {
    return this->weight_data_list;
  }

  void setWeightData(std::string name, std::vector<float> data);

  void reset(void) {
    valueMapping.clear();
    oplist.clear();
    input_details.clear();
    output_details.clear();
    weight_data_list.clear();
  }

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

  // Original and translated module.
  ModuleOp mlirModule;

  // weight file input stream
  TensorFile *weightFile_;
  DeviceMode device = DeviceMode::CPU;
  static std::string customOpPluginFile_;

protected:
  // not use anymore
  std::vector<Value> resultsList;
  std::vector<Value> inputsList;


  value_map_t valueMapping;
  op_kernel_list oplist;
  std::unordered_map<std::string,
                     std::pair<std::vector<float>, std::vector<int64_t>>>
      weight_data_list;
  std::unordered_map<std::string, std::vector<float>> change_weight_table;
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
