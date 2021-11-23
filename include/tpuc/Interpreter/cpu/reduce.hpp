#ifndef INTERPRETER_CPU_REDUCE_H
#define INTERPRETER_CPU_REDUCE_H

#include "tpuc/Interpreter/cpukernel.h"

#include <memory>

namespace mlir {

typedef enum reduce_type {
  REDUCE_MEAN,
  REDUCE_MAX,
  REDUCE_MIN,
  REDUCE_SUM,
  REDUCE_L2,
} reduce_type_t;

class ReduceOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUReduceOp";

  ReduceOpKernel(Operation &op, value_map_t &valueMapping,
                 weight_map_t &weightMapping);
  void invoke() override;

protected:
  SyncedData input_data;
  SyncedData output_data;
  SyncedData lut;
  SyncedData mantissa_lut;
  SyncedDataShape input_shape;

  // param
  std::vector<int> axes;
  int outer_dims;
  int inner_dims;
  int axis_dims;

  // int8
  int rshift;
  int multiplier;
  reduce_type_t type;
};

class ReduceL2OpKernel : public ReduceOpKernel {
public:
  ReduceL2OpKernel(Operation &op, value_map_t &valueMapping,
                   weight_map_t &weightMapping)
      : ReduceOpKernel(op, valueMapping, weightMapping) {
    type = REDUCE_L2;
  }
};

class ReduceMinOpKernel : public ReduceOpKernel {
public:
  ReduceMinOpKernel(Operation &op, value_map_t &valueMapping,
                    weight_map_t &weightMapping)
      : ReduceOpKernel(op, valueMapping, weightMapping) {
    type = REDUCE_MIN;
  }
};

class ReduceMaxOpKernel : public ReduceOpKernel {
public:
  ReduceMaxOpKernel(Operation &op, value_map_t &valueMapping,
                    weight_map_t &weightMapping)
      : ReduceOpKernel(op, valueMapping, weightMapping) {
    type = REDUCE_MAX;
  }
};

class ReduceSumOpKernel : public ReduceOpKernel {
public:
  ReduceSumOpKernel(Operation &op, value_map_t &valueMapping,
                    weight_map_t &weightMapping)
      : ReduceOpKernel(op, valueMapping, weightMapping) {
    type = REDUCE_SUM;
  }
};

class ReduceMeanOpKernel : public ReduceOpKernel {
public:
  ReduceMeanOpKernel(Operation &op, value_map_t &valueMapping,
                     weight_map_t &weightMapping)
      : ReduceOpKernel(op, valueMapping, weightMapping) {
    type = REDUCE_MEAN;
  }
};

} // namespace mlir

#endif
