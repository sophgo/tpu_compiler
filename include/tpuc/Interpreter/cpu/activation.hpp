#ifndef INTERPRETER_CPU_ACTIVATION_H
#define INTERPRETER_CPU_ACTIVATION_H

#include "tpuc/Interpreter/cpukernel.h"

#include <memory>
namespace mlir {

void relu(float *src, float *dst, size_t size);
void leaky_relu(float *src, float *dst, size_t size, float negative_slope);
int omp_schedule(int count);
float BF16(float data);

class AbsOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUAbsOpOp";

  AbsOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;
};
class ExpOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUExpOpOp";

  ExpOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;

  SyncedData y0_table_op;
  SyncedData slope_table;

  // bf16
  SyncedData y0_bf16_table_op;
  SyncedData y0_bf16_slope_table;
  int bf16_min_range;
  int bf16_max_range;
};
class MishOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUMishOp";

  MishOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;

  float mish_threshold;

  SyncedData y0_table_op;
  SyncedData slope_table;

  // bf16
  SyncedData y0_bf16_table_op;
  SyncedData y0_bf16_slope_table;
  int bf16_min_range;
  int bf16_max_range;
};

class LeakyReluOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPULeakyReluOp";

  LeakyReluOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;

  std::vector<float> slope_data;

  SyncedData rshift_postive;
  SyncedData rshift_negative;
  SyncedData multiplier_postive;
  SyncedData multiplier_negative;
  float negative_slope;

  // asymmetric
  bool is_asymmetric = false;
  int input_offset = 0;
  int output_offset = 0;
};
class ReluOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUReluOp";

  ReluOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;
};

class PReluOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUPReluOp";

  PReluOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;

  SyncedData slope_data;

  SyncedData rshift_postive;
  SyncedData rshift_negative;
  SyncedData multiplier_postive;
};

class ReshapeOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUReshapeOp";

  ReshapeOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;
};

class SigmoidOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUSigmoidOp";

  SigmoidOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;

  SyncedData y0_table_op;
  SyncedData slope_table;

  // bf16
  SyncedData y0_bf16_table_op;
  SyncedData y0_bf16_slope_table;
  int bf16_min_range;
  int bf16_max_range;
};

class ReciprocalOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUReciprocalOpOp";

  ReciprocalOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;

  SyncedData y0_table_op;
  SyncedData slope_table;

  // bf16
  SyncedData y0_bf16_table_op;
  SyncedData y0_bf16_slope_table;
  int bf16_min_range;
  int bf16_max_range;
};

class SoftPlusOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUSoftPlusOpOp";

  SoftPlusOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;

  SyncedData y0_table_op;
  SyncedData slope_table;

  float threshold;
  // bf16
  SyncedData y0_bf16_table_op;
  SyncedData y0_bf16_slope_table;
  int bf16_min_range;
  int bf16_max_range;
};

class SqrtOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUSqrtOpOp";

  SqrtOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;

  SyncedData y0_table_op;
  SyncedData slope_table;

  // bf16
  SyncedData y0_bf16_table_op;
  SyncedData y0_bf16_slope_table;
  int bf16_min_range;
  int bf16_max_range;
};

class SquareOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUSquareOpOp";

  SquareOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;
};

class TanHOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUTanHOpOp";

  TanHOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;

  SyncedData y0_table_op;
  SyncedData slope_table;

  // bf16
  SyncedData y0_bf16_table_op;
  SyncedData y0_bf16_slope_table;
  int bf16_min_range;
  int bf16_max_range;
};
} // namespace mlir

#endif
