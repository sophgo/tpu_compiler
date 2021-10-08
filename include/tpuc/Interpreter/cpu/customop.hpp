#ifndef INTERPRETER_CPU_CUSTOMOP_H
#define INTERPRETER_CPU_CUSTOMOP_H

#include "mkldnn.hpp"
#include "tpuc/CustomOpParam.h"
#include "tpuc/CustomOpPlugin.h"
#include "tpuc/Interpreter/cpukernel.h"
#include <memory>
namespace mlir {

class CustomOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUCustomOp";

  CustomOpKernel(Operation &op, value_map_t &valueMapping,
                 weight_map_t &weightMapping);

  void invoke() override;

private:
  void fp32_invoke();
  void i8_invoke();

private:
  std::vector<std::shared_ptr<std::vector<float>>> inputs_data;
  std::shared_ptr<std::vector<float>> output_data;
  std::vector<std::vector<int64_t>> inputs_shape;

  std::string operation_name;
  cvi::OpParam param;
  cvi::CustomOpPlugin *plugin;
};
} // namespace mlir

#endif
