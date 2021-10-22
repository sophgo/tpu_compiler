#ifndef INTERPRETER_CPU_CONVFC_H
#define INTERPRETER_CPU_CONVFC_H

#include "tpuc/Interpreter/cpukernel.h"

#include <memory>
namespace mlir {

class ConvFcOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUConvFcOp";
  ConvFcOpKernel(Operation &op, value_map_t &valueMapping,
                    weight_map_t &weightMapping);
  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedData filter_data;
  float qscale;
  int M,K,N;
};
} // namespace mlir

#endif
