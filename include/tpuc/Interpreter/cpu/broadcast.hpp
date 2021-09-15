#ifndef INTERPRETER_CPU_BROADCAST_H
#define INTERPRETER_CPU_BROADCAST_H

#include "tpuc/Interpreter/cpukernel.h"

#include <memory>
namespace mlir {

class BroadcastOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUBroadcastOp";

  BroadcastOpKernel(Operation &op, value_map_t &valueMapping,
                    weight_map_t &weightMapping);

  virtual void invoke() = 0;

protected:
  std::vector<SyncedData> inputs_data;
  SyncedData output_data;
  int n0, c0, h0, w0;
  int n1, c1, h1, w1;
  int on, oc, oh, ow;

  // param
  bool do_relu;
  float rshift;
  std::vector<float> multiplier;
};

class BroadcastAddOpKernel : public BroadcastOpKernel {
public:
  static constexpr const char *OpName = "CPUBroadcastAddOp";

  BroadcastAddOpKernel(Operation &op, value_map_t &valueMapping,
                       weight_map_t &weightMapping)
      : BroadcastOpKernel(op, valueMapping, weightMapping) {}

  void invoke() override;
};

class BroadcastMulOpKernel : public BroadcastOpKernel {
public:
  static constexpr const char *OpName = "CPUBroadcastMulOp";

  BroadcastMulOpKernel(Operation &op, value_map_t &valueMapping,
                       weight_map_t &weightMapping)
      : BroadcastOpKernel(op, valueMapping, weightMapping) {}

  void invoke() override;
};
class BroadcastSubOpKernel : public BroadcastOpKernel {
public:
  static constexpr const char *OpName = "CPUBroadcastSubOp";

  BroadcastSubOpKernel(Operation &op, value_map_t &valueMapping,
                       weight_map_t &weightMapping)
      : BroadcastOpKernel(op, valueMapping, weightMapping) {}

  void invoke() override;
};
} // namespace mlir

#endif