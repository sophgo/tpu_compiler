#ifndef INTERPRETER_CPU_SCALE_LUT_H
#define INTERPRETER_CPU_SCALE_LUT_H

#include "tpuc/Interpreter/cpukernel.h"

#include <memory>
namespace mlir {

class ScaleLutOpKernel : public CPUOpKernel<ScaleLutOpKernel> {
public:
  static constexpr const char *OpName = "CPUScaleLutOp";

  ScaleLutOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
  void set_tensor(const std::vector<float> &data) override;
  std::vector<float> get_tensor() override;
  void dump() override;

private:
  SyncedData input_data;
  SyncedData table;
  SyncedData output_data;
  SyncedDataShape input_shape;
  SyncedDataShape table_shape;
  // param
};
} // namespace mlir

#endif