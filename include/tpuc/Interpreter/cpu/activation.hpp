#ifndef INTERPRETER_CPU_ACTIVATION_H
#define INTERPRETER_CPU_ACTIVATION_H

#include "tpuc/Interpreter/cpukernel.h"

#include <memory>
namespace mlir {

void relu(float *data, size_t size);

class ReluOpKernel : public CPUOpKernel<ReluOpKernel> {
public:
  static constexpr const char *OpName = "CPUReluOp";

  ReluOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
  void set_tensor(const std::vector<float> &data) override;
  std::vector<float> get_tensor() override;
  void dump() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;
};

class PReluOpKernel : public CPUOpKernel<PReluOpKernel> {
public:
  static constexpr const char *OpName = "CPUPReluOp";

  PReluOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
  void set_tensor(const std::vector<float> &data) override;
  std::vector<float> get_tensor() override;
  void dump() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;

  std::vector<float> slope_data;

  std::vector<float> rshift_postive;
  std::vector<float> rshift_negative;
  std::vector<float> multiplier_postive;
};

class ReshapeOpKernel : public CPUOpKernel<ReshapeOpKernel> {
public:
  static constexpr const char *OpName = "CPUReshapeOp";

  ReshapeOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
  void set_tensor(const std::vector<float> &data) override;
  std::vector<float> get_tensor() override;
  void dump() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;
};

class SigmoidOpKernel : public CPUOpKernel<SigmoidOpKernel> {
public:
  static constexpr const char *OpName = "CPUSigmoidOp";

  SigmoidOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
  void set_tensor(const std::vector<float> &data) override;
  std::vector<float> get_tensor() override;
  void dump() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;

  std::vector<float> y0_table_op;
  std::vector<float> slope_table;

  // bf16
  std::vector<float> y0_bf16_table_op;
  std::vector<float> y0_bf16_slope_table;
  int bf16_min_range;
  int bf16_max_range;
};
} // namespace mlir

#endif
