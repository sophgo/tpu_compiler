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

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;

  // param
  int const_val;
  std::vector<int32_t> pads;
  std::string mode = "constant";
};
} // namespace mlir
#endif
