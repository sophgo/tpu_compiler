#ifndef INTERPRETER_CPU_UPSAMPLE_H
#define INTERPRETER_CPU_UPSAMPLE_H

#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {
class UpsampleOpKernel : public CPUOpKernel<UpsampleOpKernel> {
public:
  static constexpr const char *OpName = "CPUUpsampleOp";

  UpsampleOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
  void set_tensor(const std::vector<float> &data) override;
  std::vector<float> get_tensor() override;
  void dump() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;

  // param
  int scale_h;
  int scale_w;
};
} // namespace mlir
#endif