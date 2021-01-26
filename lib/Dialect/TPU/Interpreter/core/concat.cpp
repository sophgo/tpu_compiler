#include "tpuc/Interpreter/cpu/concat.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {

ConcatOpKernel::ConcatOpKernel(Operation &op, value_map_t &valueMapping) {
  auto concatOp = cast<tpu::ConcatOp>(op);
  assert(concatOp);
  llvm::outs() << " ConcatOp op: [" << concatOp.name() << "]\n";

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = concatOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  llvm::outs() << "    =>required memory size: [" << size << "]\n";

  this->shape = getTensorShape(result);

  this->name = concatOp.name().str();
  this->axis = concatOp.axis();
  this->input_number = concatOp.getNumInputs();

  this->op_type = op.getName().getStringRef().str();
  set_datatype(getOpQuant(&op).str());

  if (datatype == DataType::INT8) {
    auto quant_rshift = opTensors[input_number + 2];
    auto quant_multiplier = opTensors[input_number + 3];
    if (quant_rshift != nullptr && quant_multiplier != nullptr) {
      need_quant = true;
      rshift = quant_rshift->at(0);
      multiplier.assign(quant_multiplier->begin(), quant_multiplier->end());
    }
  }
  // get tensors
  inputs_data.resize(input_number);
  inputs_shape.resize(input_number);
  for (size_t i = 0; i < input_number; i++) {
    inputs_data[i] = opTensors[i];
    inputs_shape[i] = getTensorShape(op.getOperand(i));
  }
  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}
void ConcatOpKernel::set_tensor(const std::vector<float> &data) {
  llvm_unreachable("TODO");
};

std::vector<float> ConcatOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void ConcatOpKernel::invoke() {
  std::vector<int64_t> output_shape = this->shape;

  for (uint32_t i = output_shape.size(); i < 4; i++) {
    output_shape.push_back(1); // append to 4 dim
  }

  int global_offset = 0;
  int num_concats = 1;
  std::vector<int64_t> input_shape = inputs_shape[0];
  for (int idx = 0; idx < axis; idx++) {
    num_concats *= input_shape[idx];
  }

  int concat_output_size = 1;
  for (int idx = axis; idx < 4; idx++) {
    concat_output_size *= output_shape[idx];
  }

  for (uint32_t i = 0; i < input_number; i++) {
    float *input = (float *)inputs_data[i]->data();
    std::vector<int64_t> input_shape = inputs_shape[i];
    for (uint32_t idx = input_shape.size(); idx < 4; idx++) {
      input_shape.push_back(1);
    }

    int concat_input_size = 1;
    for (int idx = axis; idx < 4; idx++) {
      concat_input_size *= input_shape[idx];
    }

    for (int num_idx = 0; num_idx < num_concats; num_idx++) {
      auto inputT = std::make_unique<std::vector<float>>(concat_input_size);
      inputT.get()->assign(&input[num_idx * concat_input_size],
                           &input[(num_idx + 1) * concat_input_size]);

      if (need_quant) {
        for (int idx = 0; idx < concat_input_size; ++idx) {
          inputT->at(idx) = (float)applyMultiplierAndRShiftAndSaturateInt8(
              inputT->at(idx), rshift, (uint32_t)multiplier.at(i), false);
        }
      }

      auto offset = global_offset + concat_output_size * num_idx;
      std::memcpy(output_data->data() + offset, inputT.get()->data(),
                  concat_input_size * sizeof(float));
    }
    global_offset += concat_input_size;
  }
}

void ConcatOpKernel::dump() {
  OpKernel::dump();
  llvm::outs() << "\tConcat Axis: " << axis << "\n";
}
} // namespace mlir