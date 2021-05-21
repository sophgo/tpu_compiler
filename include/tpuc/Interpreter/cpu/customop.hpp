#ifndef INTERPRETER_CPU_CUSTOMOP_H
#define INTERPRETER_CPU_CUSTOMOP_H

#include "mkldnn.hpp"
#include "tpuc/Interpreter/cpukernel.h"
#include "tpuc/CustomOpPlugin.h"
#include "tpuc/CustomOpParam.h"
#include <memory>
namespace mlir {

class CustomOpKernel : public CPUOpKernel<CustomOpKernel> {
public:
  static constexpr const char *OpName = "CPUCustomOp";

  CustomOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  void fp32_invoke();
  void i8_invoke();

private:
  std::vector<SyncedData> inputs_data;
  SyncedData output_data;
  std::vector<SyncedDataShape> inputs_shape;

  std::string operation_name;
  cvi::OpParam param;
  cvi::CustomOpPlugin *plugin;
};
} // namespace mlir

#endif
