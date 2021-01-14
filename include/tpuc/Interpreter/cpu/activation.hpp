#ifndef INTERPRETER_CPU_ACTIVATION_H
#define INTERPRETER_CPU_ACTIVATION_H

#include "tpuc/Interpreter/cpukernel.h"

#include <memory>
namespace mlir {

class ReluOpKernel : public CPUOpKernel<ReluOpKernel> {
public:
  static constexpr const char *OpName = "CPUReluOp";

  ReluOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
  void set_tensor(const std::vector<float> &data) override;
  std::vector<float> get_tensor() override;
  void dump() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;
  SyncedDataShape output_shape;
};
} // namespace mlir

#endif
