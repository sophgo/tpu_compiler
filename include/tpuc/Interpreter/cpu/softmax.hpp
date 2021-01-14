#ifndef INTERPRETER_CPU_SOFTMAX_H
#define INTERPRETER_CPU_SOFTMAX_H

#include "tpuc/Interpreter/cpukernel.h"

#include <memory>
namespace mlir {

class SoftmaxOpKernel : public CPUOpKernel<SoftmaxOpKernel> {
public:
  static constexpr const char *OpName = "CPUSoftmaxOp";

  SoftmaxOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
  void set_tensor(const std::vector<float> &data) override;
  std::vector<float> get_tensor() override;
  void dump() override{};

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;
  SyncedDataShape output_shape;
};
} // namespace mlir

#endif
