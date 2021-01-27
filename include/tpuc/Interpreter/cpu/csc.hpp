#ifndef INTERPRETER_CPU_CSC_H
#define INTERPRETER_CPU_CSC_H

#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {

class CscOpKernel : public CPUOpKernel<CscOpKernel> {
public:
  static constexpr const char *OpName = "CPUCscOp";

  CscOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
  void set_tensor(const std::vector<float> &data) override;
  std::vector<float> get_tensor() override;
  void dump() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;

  // param
  std::string pixel_format;
  int aligned;
};
} // namespace mlir
#endif