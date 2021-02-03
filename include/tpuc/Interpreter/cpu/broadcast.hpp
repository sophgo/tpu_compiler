#ifndef INTERPRETER_CPU_BROADCAST_H
#define INTERPRETER_CPU_BROADCAST_H

#include "tpuc/Interpreter/cpukernel.h"

#include <memory>
namespace mlir {

class BroadcastAddOpKernel : public CPUOpKernel<BroadcastAddOpKernel> {
public:
  static constexpr const char *OpName = "CPUBroadcastAddOp";

  BroadcastAddOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
  void set_tensor(const std::vector<float> &data) override;
  std::vector<float> get_tensor() override;
  void dump() override;
  void fp32_invoke();
  void i8_invoke();

private:
  std::vector<SyncedData> inputs_data;
  SyncedData output_data;
  std::vector<SyncedDataShape> inputs_shape;

  // param
  bool do_relu;
  float rshift;
  std::vector<float> multiplier;
};
class BroadcastMulOpKernel : public CPUOpKernel<BroadcastMulOpKernel> {
public:
  static constexpr const char *OpName = "CPUBroadcastMulOp";

  BroadcastMulOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
  void set_tensor(const std::vector<float> &data) override;
  std::vector<float> get_tensor() override;
  void dump() override;

private:
  SyncedData input_data;
  SyncedData scale;
  SyncedData output_data;
  SyncedDataShape input_shape;
  SyncedDataShape scale_shape;
  // param
  bool do_relu;
  float rshift;
  float mutliplier;
};
class BroadcastSubOpKernel : public CPUOpKernel<BroadcastSubOpKernel> {
public:
  static constexpr const char *OpName = "CPUBroadcastSubOp";

  BroadcastSubOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
  void set_tensor(const std::vector<float> &data) override;
  std::vector<float> get_tensor() override;
  void dump() override;
  void fp32_invoke();
  void i8_invoke();

private:
  std::vector<SyncedData> inputs_data;
  SyncedData output_data;
  std::vector<SyncedDataShape> inputs_shape;

  // param
  bool do_relu;
  float rshift;
  std::vector<float> multiplier;
};
} // namespace mlir

#endif