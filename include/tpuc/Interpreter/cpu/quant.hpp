#ifndef INTERPRETER_CPU_QUANT_H
#define INTERPRETER_CPU_QUANT_H

#include "tpuc/Interpreter/cpukernel.h"

#include <memory>
namespace mlir {

class QuantOpKernel : public CPUOpKernel<QuantOpKernel> {
public:
  static constexpr const char *OpName = "CPUQuantOp";

  QuantOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
  void set_tensor(const std::vector<float> &data) override;
  std::vector<float> get_tensor() override;
  void dump();

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;
  SyncedDataShape output_shape;
  std::string from;
  std::string to;
  float scale;
  int zero_point = 0;
};
} // namespace mlir

#endif