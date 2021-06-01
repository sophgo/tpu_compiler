#ifndef INTERPRETER_CPU_LRN_H
#define INTERPRETER_CPU_LRN_H

#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {

class LrnOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPULrnOp";

  LrnOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData scale_data;

  SyncedData output_data;
  SyncedDataShape input_shape;

  // param
  float alpha;
  float beta;
  int local_size;
  float k;
  // int8
  SyncedData sqr_lut_data;
  SyncedData power_lut_data;
  int sum_rshift;
  int lrn_rshift;
  int quant_data0;
  int quant_data1;
};

class LrnOneOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPULrnOp";

  LrnOneOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;

  // param
  float alpha;
  int local_size;
};

class LrnTwoOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPULrnOp";

  LrnTwoOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;

  // param
  int local_size;
};

class LrnThreeOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPULrnOp";

  LrnThreeOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;

  // param
  float beta;
  float k;
};

} // namespace mlir

#endif