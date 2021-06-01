#ifndef INTERPRETER_CPU_CSC_H
#define INTERPRETER_CPU_CSC_H

#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {

class CscOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUCscOp";

  CscOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

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