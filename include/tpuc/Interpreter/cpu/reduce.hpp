#ifndef INTERPRETER_CPU_REDUCE_H
#define INTERPRETER_CPU_REDUCE_H

#include "tpuc/Interpreter/cpukernel.h"

#include <memory>

namespace mlir {

class ReduceL2OpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUReduceL2Op";

  ReduceL2OpKernel(Operation &op, value_map_t &valueMapping,
                   weight_map_t &weightMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;

  // param
  std::vector<int> axes;

  // int8
  int rshift;
  int multiplier;
};

class ReduceMinOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUReduceMinOp";

  ReduceMinOpKernel(Operation &op, value_map_t &valueMapping,
                    weight_map_t &weightMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;

  // param
  std::vector<int> axes;

  // int8
  int rshift;
  int multiplier;
};

class ReduceMaxOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUReduceMaxOp";

  ReduceMaxOpKernel(Operation &op, value_map_t &valueMapping,
                    weight_map_t &weightMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;

  // param
  std::vector<int> axes;

  // int8
  int rshift;
  int multiplier;
};

class ReduceMeanOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUReduceMeanOp";

  ReduceMeanOpKernel(Operation &op, value_map_t &valueMapping,
                     weight_map_t &weightMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;

  // param
  std::vector<int> axes;

  // int8
  int rshift;
  int multiplier;
};

} // namespace mlir

#endif
