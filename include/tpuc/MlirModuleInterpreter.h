#ifndef MLIR_MODULE_INTERPRETER_H_
#define MLIR_MODULE_INTERPRETER_H_

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
#include <list>
#include <mutex>
#include <omp.h>
#include <unordered_map>

#define DEBUG_TYPE "interpreter"

namespace mlir {

class ModuleOp;

using activations_map_t = std::map<std::string, std::shared_ptr<TensorData>>;
using kernel_map_t = std::map<std::string, std::shared_ptr<CPUOpKernel>>;
using kernel_list_t = std::list<std::shared_ptr<CPUOpKernel>>;

class MlirModuleInterpreter {
public:
  MlirModuleInterpreter(int32_t batch = 1) {
    CPUOpKernel::max_batch_size = batch;
  }
  ~MlirModuleInterpreter() {}

  void loadModule(OwningModuleRef &module_op, std::string target_op);
  void loadModule(OwningModuleRef &module_op) { loadModule(module_op, ""); }
  void fastInvokeAllBatch(std::string name, int32_t batch);
  void invokeTo(std::string name, int32_t bidx);
  void invoke(int32_t bidx) { invokeTo("", bidx); }
  void setTensor(std::string &name, const std::vector<float> &data,
                 int32_t bidx);
  std::shared_ptr<std::vector<float>> getTensor(std::string &name,
                                                int32_t bidx);
  std::vector<int64_t> getTensorShape(std::string &name);
  std::string getDataType(std::string &name);
  static void updateWeightMap(OwningModuleRef &module_op);
  std::vector<std::string> &outputDetails() { return output_details; }
  void dump(std::string &name);
  std::vector<std::string> getAllTensorName() {
    std::vector<std::string> ret;
    for (auto &kv : activationMapping) {
      ret.push_back(kv.first);
    }
    return ret;
  }
  static std::string &getCustomOpPluginFile() {
    return customOpPluginFile_;
  }
  static void setCustomOpPluginFile(std::string &file) {
    customOpPluginFile_ = file;
  }

private:
  void updateKernelList(FuncOp &op, std::string &target_op);
  bool isKernelDirty(std::shared_ptr<CPUOpKernel> &krnl, Operation *op);

public:
  std::vector<std::pair<std::string, size_t>> input_details;
  std::vector<std::string> output_details;
  static weight_map_t weightMapping;
  activations_map_t activationMapping;
  value_map_t valueMapping;
  static std::string chipname;

private:
  kernel_map_t kernel_map_;
  kernel_list_t kernel_list_;
  static std::string customOpPluginFile_;
};

} // namespace mlir

#endif // MLIR_MODULE_INTERPRETER_H_
