
#ifndef INTERPRETER_CPU_CORE_H
#define INTERPRETER_CPU_CORE_H
#include "tpuc/Interpreter/core.h"
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/TPUOperationSupport.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

#include <memory>
#include <vector>

namespace mlir {

template <typename T>
class CPUOpKernel : public OpKernel {

public:
  CPUOpKernel(Operation &op, value_map_t &valueMapping, bool hasOpds = true) {
    auto type = op.getResult(0).getType().cast<TensorType>();
    this->shape = type.getShape();
    this->name = getOpName(&op).str();
    this->op_type = op.getName().getStringRef().str();
    set_datatype(getOpQuant(&op).str());
    if (hasOpds) {
      assignOperandTensors(op, valueMapping);
    }
    assignResultTensor(op, valueMapping);
  }

  CPUOpKernel() = delete;

  virtual void set_tensor(const std::vector<float> &data) {
    llvm_unreachable("NOT support set_tensor");
  }

  std::vector<float> get_tensor() {
    // deep copy
    std::vector<float> ret(resTensor->begin(), resTensor->end());
    return ret;
  }

protected:

  void assignOperandTensors(Operation &op, const value_map_t &valueMapping) {
    for (auto opd : op.getOperands()) {
      if (isTensorNone(opd)) {
        opdTensors.push_back(nullptr);
        continue;
      }
      auto it = valueMapping.find(opd);
      if (it == valueMapping.end()) {
        llvm::errs() << "not find: " << opd.getDefiningOp()->getName() << "\n";
        llvm_unreachable("value mapping false");
      }
      opdTensors.emplace_back(it->second);
    }
  }

  void assignResultTensor(Operation &op, value_map_t &valueMapping) {
    auto result = op.getResult(0);
    auto size = getTensorSize(result);
    resTensor = std::make_shared<std::vector<float>>(size);
    valueMapping[result] = resTensor;
  }

protected:
  std::vector<SyncedData> opdTensors;
  SyncedData resTensor;
};

class InputOpKernel : public CPUOpKernel<InputOpKernel> {
public:
  static constexpr const char *OpName = "CPUInputOp";
  InputOpKernel(Operation &op, value_map_t &valueMapping,
                std::vector<std::pair<std::string, size_t>> &input_details);
  void invoke() override {
      // input op no need to invoke, skip
  };
  void set_tensor(const std::vector<float> &data) override;

private:
  SyncedData data;
};

}; // namespace mlir

#endif