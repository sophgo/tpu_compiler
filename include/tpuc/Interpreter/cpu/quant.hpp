#ifndef INTERPRETER_CPU_QUANT_H
#define INTERPRETER_CPU_QUANT_H

#include "tpuc/Interpreter/cpukernel.h"

#include <memory>
namespace mlir {

class QuantOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUQuantOp";

  QuantOpKernel(Operation &op, value_map_t &valueMapping,
                weight_map_t &weightMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;
  std::string from;
  std::string to;
  float scale;
  int zero_point = 0;
  bool useTpuQuant;
};

} // namespace mlir

#endif
