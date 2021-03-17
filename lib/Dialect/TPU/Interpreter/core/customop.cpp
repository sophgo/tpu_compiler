#include "tpuc/Interpreter/cpu/customop.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {

CustomOpKernel::CustomOpKernel(Operation &op, value_map_t &valueMapping) {
  auto customOp = cast<tpu::CustomOp>(op);
  assert(customOp);
  LLVM_DEBUG(llvm::outs() << " CustomOp op: [" << customOp.name() << "]\n";);

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = customOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  LLVM_DEBUG(llvm::outs() << "    =>required memory size: [" << size << "]\n";);
  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();
  this->name = customOp.name().str();
  this->op_type = op.getName().getStringRef().str();
  this->operation_name = customOp.operation_name().str();
  set_datatype(getOpQuant(&op).str());

  convertAttributesToOpParam(customOp.param(), param);

  auto& pluginFile = ModuleInterpreter::getCustomOpPluginFile();
  this->plugin = cvi::CustomOpPlugin::load(pluginFile);
  assert(this->plugin);

  // get tensors
  inputs_data.resize(opTensors.size());
  inputs_shape.resize(opTensors.size());
  for (size_t i = 0; i < opTensors.size(); i++) {
    inputs_data[i] = opTensors[i];
    inputs_shape[i] = getTensorShape(op.getOperand(i));
  }
  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}

void CustomOpKernel::set_tensor(const std::vector<float> &data) {
  llvm_unreachable("TODO");
}

std::vector<float> CustomOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void CustomOpKernel::invoke() {
  if (datatype == DataType::FP32) {
    plugin->fp32Interpret(
        operation_name.c_str(), param, inputs_data,
        inputs_shape, output_data, shape);
  } else if (datatype == DataType::INT8) {
    plugin->int8Interpret(
        operation_name.c_str(), param, inputs_data,
        inputs_shape, output_data, shape);
  } else if (datatype == DataType::BF16) {
    plugin->bf16Interpret(
        operation_name.c_str(), param, inputs_data,
        inputs_shape, output_data, shape);
  } else {
    llvm_unreachable("unsupported type");
  }
}

void CustomOpKernel::dump() { OpKernel::dump(); }
} // namespace mlir