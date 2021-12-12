#ifndef INTERPRETER_CPU_ACTIVATION_H
#define INTERPRETER_CPU_ACTIVATION_H

#include "tpuc/Interpreter/cpukernel.h"

#include <memory>
namespace mlir {
class AbsOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUAbsOpOp";

  AbsOpKernel(Operation &op, value_map_t &valueMapping,
              weight_map_t &weightMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;
};
class ExpOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUExpOpOp";

  ExpOpKernel(Operation &op, value_map_t &valueMapping,
              weight_map_t &weightMapping);

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
  float scale;
  float bias;
};
class MishOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUMishOp";

  MishOpKernel(Operation &op, value_map_t &valueMapping,
               weight_map_t &weightMapping);

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

class LeakyReluOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPULeakyReluOp";

  LeakyReluOpKernel(Operation &op, value_map_t &valueMapping,
                    weight_map_t &weightMapping);

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
};
class ReluOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUReluOp";

  ReluOpKernel(Operation &op, value_map_t &valueMapping,
               weight_map_t &weightMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;
};

class PReluOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUPReluOp";

  PReluOpKernel(Operation &op, value_map_t &valueMapping,
                weight_map_t &weightMapping);

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

  ReshapeOpKernel(Operation &op, value_map_t &valueMapping,
                  weight_map_t &weightMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;
};

class SigmoidOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUSigmoidOp";

  SigmoidOpKernel(Operation &op, value_map_t &valueMapping,
                  weight_map_t &weightMapping);

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
  float scale;
  float bias;
};

class SwishOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUSwishOp";

  SwishOpKernel(Operation &op, value_map_t &valueMapping,
                weight_map_t &weightMapping);

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

class EluOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUEluOp";

  EluOpKernel(Operation &op, value_map_t &valueMapping,
              weight_map_t &weightMapping);

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

  SoftPlusOpKernel(Operation &op, value_map_t &valueMapping,
                   weight_map_t &weightMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;

  SyncedData y0_table_op;
  SyncedData slope_table;

  float scale;
  float bias;
  // bf16
  SyncedData y0_bf16_table_op;
  SyncedData y0_bf16_slope_table;
};

class PowOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUPowOpOp";

  PowOpKernel(Operation &op, value_map_t &valueMapping,
               weight_map_t &weightMapping);

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
  float coeff;
};

class TanHOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUTanHOpOp";

  TanHOpKernel(Operation &op, value_map_t &valueMapping,
               weight_map_t &weightMapping);

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

class LogOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPULogOpOp";

  LogOpKernel(Operation &op, value_map_t &valueMapping,
              weight_map_t &weightMapping);

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
