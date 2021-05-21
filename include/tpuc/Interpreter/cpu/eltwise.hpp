#ifndef INTERPRETER_CPU_ELTWISE_H
#define INTERPRETER_CPU_ELTWISE_H

#include "tpuc/Interpreter/cpukernel.h"

#include <memory>
namespace mlir {

class EltwiseAddOpKernel : public CPUOpKernel<EltwiseAddOpKernel> {
public:
  static constexpr const char *OpName = "CPUEltwiseAddOp";

  EltwiseAddOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  void fp32_invoke();
  void i8_invoke();

private:
  std::vector<SyncedData> inputs_data;
  SyncedData output_data;
  std::vector<SyncedDataShape> inputs_shape;

  // int8
  std::vector<float> rshift;
  std::vector<float> multiplier;

  bool do_quant;
  // asymmetric
  bool is_asymmetric;
  std::vector<float> inputs_offset;
  int output_offset = 0;

  // param
  bool do_relu;
  std::vector<float> coeff;
};

class EltwiseMaxOpKernel : public CPUOpKernel<EltwiseMaxOpKernel> {
public:
  static constexpr const char *OpName = "CPUEltwiseMaxOp";

  EltwiseMaxOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  void fp32_invoke();
  void i8_invoke();

private:
  std::vector<SyncedData> inputs_data;
  SyncedData output_data;
  std::vector<SyncedDataShape> inputs_shape;

  // int8
  std::vector<float> rshift;
  std::vector<float> multiplier;

  bool do_quant;

  // param
  bool do_relu;
};

class EltwiseMinOpKernel : public CPUOpKernel<EltwiseMinOpKernel> {
public:
  static constexpr const char *OpName = "CPUEltwiseMinOp";

  EltwiseMinOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  void fp32_invoke();
  void i8_invoke();

private:
  std::vector<SyncedData> inputs_data;
  SyncedData output_data;
  std::vector<SyncedDataShape> inputs_shape;

  // int8
  std::vector<float> rshift;
  std::vector<float> multiplier;

  bool do_quant;

  // param
  bool do_relu;
};

class EltwiseMulOpKernel : public CPUOpKernel<EltwiseMulOpKernel> {
public:
  static constexpr const char *OpName = "CPUEltwiseMulOp";

  EltwiseMulOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  void fp32_invoke();
  void i8_invoke();

private:
  std::vector<SyncedData> inputs_data;
  SyncedData output_data;
  std::vector<SyncedDataShape> inputs_shape;

  // int8
  std::vector<float> rshift;
  std::vector<float> multiplier;
  bool do_quant;
  // asymmetric
  bool is_asymmetric = false;
  std::vector<float> inputs_offset;
  int output_offset = 0;
  // param
  bool do_relu;
};
} // namespace mlir

#endif