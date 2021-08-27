#include "tpuc/Interpreter/cpu/customop.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/MlirModuleInterpreter.h"

namespace mlir {

CustomOpKernel::CustomOpKernel(Operation &op, value_map_t &valueMapping,
                               weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  auto customOp = cast<tpu::CustomOp>(op);
  this->operation_name = customOp.operation_name().str();
  convertAttributesToOpParam(customOp.param(), param);

  auto &pluginFile = MlirModuleInterpreter::getCustomOpPluginFile();
  this->plugin = cvi::CustomOpPlugin::load(pluginFile);
  assert(this->plugin);

  // get tensors
  inputs_data.resize(this->opdTensors.size());
  inputs_shape.resize(this->opdTensors.size());
  for (size_t i = 0; i < this->opdTensors.size(); i++) {
    inputs_data[i] = this->opdTensors[i];
    inputs_shape[i] = getTensorShape(op.getOperand(i));
  }
  output_data = this->resTensor;
}

void CustomOpKernel::invoke() {
  if (datatype == DataType::FP32) {
    plugin->fp32Interpret(operation_name.c_str(), param, inputs_data,
                          inputs_shape, output_data, shape);
  } else if (datatype == DataType::INT8) {
    plugin->int8Interpret(operation_name.c_str(), param, inputs_data,
                          inputs_shape, output_data, shape);
  } else if (datatype == DataType::BF16) {
    plugin->bf16Interpret(operation_name.c_str(), param, inputs_data,
                          inputs_shape, output_data, shape);
  } else {
    llvm_unreachable("unsupported type");
  }
}

} // namespace mlir