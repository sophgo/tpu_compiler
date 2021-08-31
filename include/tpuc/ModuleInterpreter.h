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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "tpuc/Interpreter/cpukernel.h"
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

using op_kernel_list = std::vector<std::shared_ptr<CPUOpKernel>>;

enum class DeviceMode { CPU, GPU };
// Implementation class for module interpreter.
class ModuleInterpreter {

public:
  // Interpret the given MLIR module expressed in MLIR TPU IR dialect
  explicit ModuleInterpreter(ModuleOp module)
      : mlirModule(module), weightFile_(nullptr) {}
  virtual ~ModuleInterpreter() {}


  std::vector<std::pair<std::string, size_t>> get_input_details() {
    return input_details;
  }

  std::vector<std::string> get_output_details() { return output_details; }

  // v2
  void allocate_tensors();
  void prepareOperation(Operation &op);
  void invoke();
  void invoke(std::string name);
  void invoke_to(const std::string &name);

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
  }

  void reset(void) {
    valueMapping.clear();
    oplist.clear();
    input_details.clear();
    output_details.clear();
  }

private:
  std::vector<Value> getInputsList() { return inputsList; }
  std::vector<Value> getResultsList() { return resultsList; }

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
  weight_map_t weightMapping;
  op_kernel_list oplist;
  std::mutex invoke_lock;
};

} // namespace mlir

#endif // MLIR_DIALECT_TPU_MODULEINTERPRETER_H_
