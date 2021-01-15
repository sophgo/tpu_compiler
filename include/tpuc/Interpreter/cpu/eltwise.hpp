#ifndef INTERPRETER_CPU_ELTWISE_H
#define INTERPRETER_CPU_ELTWISE_H

#include "tpuc/Interpreter/cpukernel.h"

#include <memory>
namespace mlir {

class EltwiseAddOpKernel : public CPUOpKernel<EltwiseAddOpKernel> {
public:
  static constexpr const char *OpName = "CPUEltwiseAddOp";

  EltwiseAddOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
  void set_tensor(const std::vector<float> &data) override;
  std::vector<float> get_tensor() override;
  void dump() override;

private:
  void fp32_invoke();
  void i8_invoke();

private:
  std::vector<SyncedData> inputs_data;
  SyncedData output_data;
  std::vector<SyncedDataShape> inputs_shape;
  SyncedDataShape output_shape;

  // int8
  std::vector<float> rshift;
  std::vector<float> multiplier;
  int output_offset = 0;

  // param
  bool do_relu;
};
} // namespace mlir

#endif