#ifndef INTERPRETER_CPU_STD_H
#define INTERPRETER_CPU_STD_H

#include "tpuc/Interpreter/cpukernel.h"

#include <memory>
namespace mlir {

class StdOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUStdOp";
  StdOpKernel(Operation &op, value_map_t &valueMapping,
                    weight_map_t &weightMapping);
  void invoke() override;

private:
  void std_fp32(float *src, float *dst, int size);
  void std_bf16(float *src, float *dst, int size);

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedData lut;
  SyncedData mantissa_lut;
  SyncedDataShape input_shape;
  // param
  int32_t std_size;
  int32_t outer_size;
  int32_t start_dim;
  bool unbiased;
};
} // namespace mlir

#endif
