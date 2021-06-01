#ifndef MLIR_MODULE_INTERPRETER_H_
#define MLIR_MODULE_INTERPRETER_H_

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "tpuc/Interpreter/cpukernel.h"
#include "tpuc/MachineInfo.h"
#include "tpuc/QuantizationArithmetic.h"
#include "tpuc/Support/TensorFile.h"
#include "tpuc/TPUOperationSupport.h"

#include "llvm/Support/Debug.h"

#include <fstream>
#include <iostream>
#include <mutex>
#include <list>
#include <omp.h>
#include <unordered_map>

#define DEBUG_TYPE "interpreter"

namespace mlir {

class ModuleOp;

using weight_map_t = std::map<std::string, std::shared_ptr<std::vector<float>>>;
using activations_map_t = std::map<std::string, std::shared_ptr<std::vector<float>>>;
using value_map_t = DenseMap<Value, std::shared_ptr<std::vector<float>>>;
using kernel_map_t = std::map<std::string, std::shared_ptr<CPUOpKernel>>;
using kernel_list_t = std::list<std::shared_ptr<CPUOpKernel>>;

class MlirModuleInterpreter {
public:
  MlirModuleInterpreter() {}
  ~MlirModuleInterpreter() {}

  void loadModule(OwningModuleRef &module_op);
  void invokeTo(std::string name);
  void setTensor(std::string &name, const std::vector<float> &data);
  std::shared_ptr<std::vector<float>> getTensor(std::string &name);
  std::vector<int64_t> getTensorShape(std::string &name);
  std::string getDataType(std::string &name);
  static void updateWeightMap(OwningModuleRef &module_op);
  std::vector<std::string>& outputDetails() {
    return output_details;
  }

private:
  void updateKernelList(FuncOp &op);
  bool isKernelDirty(std::shared_ptr<CPUOpKernel> &krnl,
                     Operation *op);

public:
  std::vector<std::pair<std::string, size_t>> input_details;
  std::vector<std::string> output_details;
  static weight_map_t weightMapping;
  activations_map_t activationMapping;
  value_map_t valueMapping;

private:
  kernel_map_t kernel_map_;
  kernel_list_t kernel_list_;
};

} // namespace mlir

#endif // MLIR_MODULE_INTERPRETER_H_
