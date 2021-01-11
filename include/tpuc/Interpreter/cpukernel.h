
#ifndef INTERPRETER_CPU_CORE_H
#define INTERPRETER_CPU_CORE_H
#include "tpuc/Interpreter/core.h"

#include "mlir/IR/Module.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <vector>

namespace mlir {
using value_map_t = DenseMap<Value, std::shared_ptr<std::vector<float>>>;

template <typename T>
class CPUOpKernel : public OpKernel {

public:
  void set_tensor(const std::vector<float> &data) {
    static_cast<T *>(this)->set_tensor(data);
  };
  std::vector<float> get_tensor() {
    return static_cast<T *>(this)->get_tensor();
  };
  void invoke() override { static_cast<T *>(this)->invoke(); };
};

class InputOpKernel : public CPUOpKernel<InputOpKernel> {
public:
  static constexpr const char *OpName = "CPUInputOp";

  InputOpKernel(SyncedData data, SyncedDataShape shape, int index = 0)
      : data(data), index(index){};

  InputOpKernel(Operation &op, value_map_t &valueMapping);
  void invoke() override{
      // input op no need to invoke, skip
  };
  void set_tensor(const std::vector<float> &data) override;
  std::vector<float> get_tensor() override;
  void dump() override;

private:
  SyncedData data;
  int index; // index
};

class LoadWeightOpKernel : public CPUOpKernel<LoadWeightOpKernel> {
public:
  static constexpr const char *OpName = "LoadWeightOp";

  LoadWeightOpKernel(SyncedData weight_data, SyncedDataShape shape)
      : weight_data(weight_data){};
  void invoke() override {
    llvm::errs() << "load wegiht op no need to invoke, skip\n";
  };
  void dump() override{};

private:
  SyncedData weight_data;
};

}; // namespace mlir

#endif