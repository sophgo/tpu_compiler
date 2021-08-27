#ifndef INTERPRETER_CPU_SOFTMAX_H
#define INTERPRETER_CPU_SOFTMAX_H

#include "tpuc/Interpreter/cpukernel.h"

#include <memory>
namespace mlir {

class SoftmaxOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUSoftmaxOp";

  SoftmaxOpKernel(Operation &op, value_map_t &valueMapping,
                  weight_map_t &weightMapping, bool cpu = false);
  virtual ~SoftmaxOpKernel() {
    if (max_arr)
      delete[] max_arr;
    if (sum_arr)
      delete[] sum_arr;
    if (ex_arr)
      delete[] ex_arr;
  }
  void invoke() override;

private:
  void invoke_fp32();
  void invoke_bf16();

  SyncedData input_data;
  SyncedData exp_table;
  SyncedData exp_slope_table;
  SyncedData reciprocal_table;
  SyncedData reciprocal_mantissa_table;
  SyncedData output_data;
  SyncedDataShape input_shape;
  int channel;
  int outer_dim;
  int inner_dim;

  float *max_arr = nullptr;
  float *sum_arr = nullptr;
  float *ex_arr = nullptr;

  // param
  int axis;
};

} // namespace mlir

#endif
