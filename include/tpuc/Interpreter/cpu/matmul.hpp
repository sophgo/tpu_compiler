#ifndef INTERPRETER_CPU_MATMUL_H
#define INTERPRETER_CPU_MATMUL_H

#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {

class MatMulOpKernel : public CPUOpKernel<MatMulOpKernel> {
public:
  static constexpr const char *OpName = "CPUMatMulOp";

  MatMulOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  SyncedData left_data;
  SyncedData right_data;
  SyncedData output_data;
  // param
  bool do_relu;
  bool l_trans;
  bool r_trans;
  bool o_trans;
  int batch_high;
  int batch_low;
  int M;
  int K;
  int N;
  // int8
  float rshift;
  float multiplier;
};
} // namespace mlir
#endif