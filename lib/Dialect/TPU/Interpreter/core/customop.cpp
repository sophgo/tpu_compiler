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
  inputs_data.resize(this->opdTensors.size());
  inputs_shape.resize(this->opdTensors.size());
  for (size_t i = 0; i < this->opdTensors.size(); i++) {
    inputs_shape[i] = getTensorShape(op.getOperand(i));
  }
}

void CustomOpKernel::invoke() {
  for (size_t i = 0; i < this->opdTensors.size(); i++) {
    inputs_data[i] = std::make_shared<std::vector<float>>(
        opdTensors[i]->begin(), opdTensors[i]->end());
  }
  output_data = std::make_shared<std::vector<float>>(resTensor->size());
  plugin->fp32Interpret(operation_name.c_str(), param, inputs_data,
                        inputs_shape, output_data, shape);
  if (datatype == DataType::BF16) {
    BF16(output_data->data(), output_data->data(), output_data->size());
  }
  std::copy(output_data->begin(), output_data->end(), resTensor->begin());
}

} // namespace mlir