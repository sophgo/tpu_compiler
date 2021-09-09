#ifndef INTERPRETER_CPU_WHERE_H
#define INTERPRETER_CPU_WHERE_H

#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {
class WhereOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUWhereOp";

  WhereOpKernel(Operation &op, value_map_t &valueMapping,
              weight_map_t &weightMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData condition_data;
  SyncedData x_data = NULL;
  SyncedData output_data;

  SyncedDataShape input_shape;
  SyncedDataShape condition_shape;
  SyncedDataShape x_shape;
  SyncedDataShape output_shape;
  float fill_constant;

  // int8
  std::vector<float> rshift;
  std::vector<float> multiplier;
  bool need_quant = false;
};
} // namespace mlir
#endif
