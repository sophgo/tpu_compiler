#ifndef INTERPRETER_CPU_PAD_H
#define INTERPRETER_CPU_PAD_H

#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {
void pad_constant(float *input, float *output,
                  std::vector<int64_t> &input_shape, std::vector<int> &pads,
                  float const_val);
class PadOpKernel : public CPUOpKernel<PadOpKernel> {
public:
  static constexpr const char *OpName = "CPUPadOp";

  PadOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
  void set_tensor(const std::vector<float> &data) override;
  std::vector<float> get_tensor() override;
  void dump() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;

  // param
  int const_val;
  std::vector<int32_t> pads;
};
} // namespace mlir
#endif