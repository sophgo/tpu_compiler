
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

class CPUOpKernel : public OpKernel {

public:
  CPUOpKernel(Operation &op, value_map_t &valueMapping,
              bool hasOpds = true) {
    auto type = op.getResult(0).getType().cast<TensorType>();
    this->shape = type.getShape();
    this->name = getOpName(&op).str();
    this->op_type = op.getName().getStringRef().str();
    set_datatype(getOpQuant(&op).str());
    if (hasOpds) {
      assignOperandTensors(op, valueMapping);
    }
    assignResultTensor(op, valueMapping);
    signature = generateSignature(op);
    this->op = op.getResult(0).getDefiningOp();
  }

  CPUOpKernel() = delete;

  virtual ~CPUOpKernel() {}

  virtual void set_tensor(const std::vector<float> &data) {
    llvm_unreachable("NOT support set_tensor");
  }

  std::vector<float> get_tensor() {
    // deep copy
    std::vector<float> ret(resTensor->begin(), resTensor->end());
    return ret;
  }

  static std::string generateSignature(Operation &op) {
    std::string signature;
    std::string s;
    llvm::raw_string_ostream os(s);
    op.print(os);
    auto str = os.str();
    for (int i = 0; i < (int)str.size(); i++) {
      if (str[i] == ')') {
        signature = str.substr(i + 1);
        break;
      }
    }
    return signature;
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

public:
  Operation *op;
  std::string signature;
  bool dirty = true;

protected:
  std::vector<SyncedData> opdTensors;
  SyncedData resTensor;
};

class InputOpKernel : public CPUOpKernel {
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