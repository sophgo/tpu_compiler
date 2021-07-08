#ifndef INTERPRETER_CPU_BROADCAST_H
#define INTERPRETER_CPU_BROADCAST_H

#include "tpuc/Interpreter/cpukernel.h"

#include <memory>
namespace mlir {

class BroadcastAddOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUBroadcastAddOp";

  BroadcastAddOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
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
class BroadcastMulOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUBroadcastMulOp";

  BroadcastMulOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData scale;
  SyncedData output_data;
  SyncedDataShape input_shape;
  SyncedDataShape scale_shape;
  // param
  bool do_relu;
  float rshift;
  float multiplier;
};
class BroadcastSubOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUBroadcastSubOp";

  BroadcastSubOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
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