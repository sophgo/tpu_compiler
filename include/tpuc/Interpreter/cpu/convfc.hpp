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
  SyncedData quant_scale;
  SyncedData quant_zeropoint;
  int M, K, N;
  bool mix_bf16;
};
} // namespace mlir

#endif
