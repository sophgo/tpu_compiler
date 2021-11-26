#ifndef INTERPRETER_CPU_ELTWISE_H
#define INTERPRETER_CPU_ELTWISE_H

#include "tpuc/Interpreter/cpukernel.h"

#include <memory>
namespace mlir {

class EltwiseAddOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUEltwiseAddOp";

  EltwiseAddOpKernel(Operation &op, value_map_t &valueMapping,
                     weight_map_t &weightMapping);

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
  std::vector<float> coeff;
};

class EltwiseMaxOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUEltwiseMaxOp";

  EltwiseMaxOpKernel(Operation &op, value_map_t &valueMapping,
                     weight_map_t &weightMapping);

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

class EltwiseMinOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUEltwiseMinOp";

  EltwiseMinOpKernel(Operation &op, value_map_t &valueMapping,
                     weight_map_t &weightMapping);

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

class EltwiseMulOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUEltwiseMulOp";

  EltwiseMulOpKernel(Operation &op, value_map_t &valueMapping,
                     weight_map_t &weightMapping);

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

class EltwiseConstOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUEltwiseConstOp";

  EltwiseConstOpKernel(Operation &op, value_map_t &valueMapping,
                     weight_map_t &weightMapping);

  void invoke() override;

private:
  void fp32_invoke();
  void i8_invoke();

private:
  SyncedData input_data;
  SyncedData output_data;

  // int8
  std::vector<float> rshift;
  std::vector<float> multiplier;
  bool do_quant;
  // param
  bool do_relu;
  float const_val;
  int8_t op_mode;
  int8_t coeff;
};
} // namespace mlir

#endif