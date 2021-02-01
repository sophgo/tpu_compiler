#ifndef INTERPRETER_CPU_CONCAT_H
#define INTERPRETER_CPU_CONCAT_H

#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {
class ConcatOpKernel : public CPUOpKernel<ConcatOpKernel> {
public:
  static constexpr const char *OpName = "CPUConcatOp";

  ConcatOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
  void set_tensor(const std::vector<float> &data) override;
  std::vector<float> get_tensor() override;
  void dump() override;

private:
  std::vector<SyncedData> inputs_data;
  SyncedData output_data;
  std::vector<SyncedDataShape> inputs_shape;

  bool do_relu;
  // int8
  float rshift;
  std::vector<float> multiplier;

  int axis;
  size_t input_number;
  bool need_quant = false;
};
} // namespace mlir
#endif