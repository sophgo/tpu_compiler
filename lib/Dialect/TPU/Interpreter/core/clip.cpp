#include "tpuc/Interpreter/cpu/clip.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/MlirModuleInterpreter.h"

namespace mlir {

ClipOpKernel::ClipOpKernel(Operation &op, value_map_t &valueMapping,
                           weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  auto clipOp = cast<tpu::ClipOp>(op);
  auto input_type = clipOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();
  this->name = clipOp.name().str();
  this->max = clipOp.max().convertToFloat();
  this->min = clipOp.min().convertToFloat();

  if (datatype == DataType::INT8) {
    auto quant_rshift = this->opdTensors[3];
    auto quant_multiplier = this->opdTensors[4];

    this->rshift = quant_rshift->at(0);
    this->multiplier = quant_multiplier->at(0);
  }
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void ClipOpKernel::invoke() {
  for (size_t i = 0; i < output_data->size(); i++) {
    output_data->at(i) = std::max(min, std::min(input_data->at(i), max));
  }

  if (datatype == DataType::INT8) {
    for (size_t i = 0; i < output_data->size(); ++i) {
      output_data->at(i) = (float)applyMultiplierAndRShiftAndSaturateInt8(
          output_data->at(i), (uint32_t)rshift, (uint32_t)multiplier, false);
    }
  }
}

} // namespace mlir