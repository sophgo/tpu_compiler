#ifndef INTERPRETER_CPU_CONCAT_H
#define INTERPRETER_CPU_CONCAT_H

#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {
class ConcatOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUConcatOp";

  ConcatOpKernel(Operation &op, value_map_t &valueMapping,
                 weight_map_t &weightMapping);

  void invoke() override;

private:
  std::vector<SyncedData> inputs_data;
  SyncedData output_data;
  std::vector<SyncedDataShape> inputs_shape;

  bool do_relu;
  // int8
  std::vector<float> rshift;
  std::vector<float> multiplier;

  int axis;
  size_t input_number;
  bool need_quant = false;
};
} // namespace mlir
#endif