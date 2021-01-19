#include "tpuc/Interpreter/cpu/activation.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"

// FIXME: this head should not be here,
//  using interpreter own float convert is better
#include "bmkernel/bm1880v2/1880v2_fp_convert.h"

#include <cmath>

namespace mlir {

static void hardwareLookUpTable(float *input, float *output, int n, int c,
                                int h, int w,
                                const std::vector<float> &y0_bf16_table,
                                const std::vector<float> &y0_bf16_slope_table,
                                const int bf16_table_start,
                                const int bf16_table_end) {

  int shape_size = n * c * h * w;

  // interger index range
  // from 16(-8~8)->256(lut index size)
  float scale = 256 / (bf16_table_end - bf16_table_start);

  // rounding
  scale = convert_bf16_fp32(convert_fp32_bf16(scale));

  for (int i = 0; i < shape_size; ++i) {
    float reOffset_input = convert_bf16_fp32(convert_fp32_bf16(input[i]));
    float rescale_input =
        convert_bf16_fp32(convert_fp32_bf16(reOffset_input)) * scale;
    uint16_t rescale_input_bf16 = convert_fp32_bf16(rescale_input);
    // get interger part
    int rescale_input_i8 =
        _convert_bf16_s8(rescale_input_bf16, /*int8_rnd_mode=*/1);

    // get delta x (x - x0)
    float delta_x = rescale_input - rescale_input_i8;

    // get slope
    uint16_t slope = y0_bf16_slope_table[rescale_input_i8 & 0xff];

    // base y0 = f(x0)
    uint16_t base = y0_bf16_table[rescale_input_i8 & 0xff];

    // result = y0 + delta * slope
    float r = convert_bf16_fp32(base) + delta_x * convert_bf16_fp32(slope);
    output[i] = convert_bf16_fp32(convert_fp32_bf16(r));
  }
}

void relu(float *data, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    data[i] = data[i] > 0 ? data[i] : 0;
  }
}

ReluOpKernel::ReluOpKernel(Operation &op, value_map_t &valueMapping) {
  auto reluOp = cast<tpu::ReluOp>(op);
  assert(reluOp);
  llvm::outs() << " Relu op: [" << reluOp.name() << "]\n";

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = reluOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  llvm::outs() << "    =>required memory size: [" << size << "]\n";
  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

  this->name = reluOp.name().str();
  this->op_type = op.getName().getStringRef().str();
  set_datatype(getOpQuant(&op).str());
  // get tensors
  input_data = opTensors[0];
  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}
void ReluOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " Relu op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> ReluOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void ReluOpKernel::invoke() {
  for (size_t i = 0; i < input_data->size(); ++i) {
    output_data->at(i) = input_data->at(i) > 0 ? input_data->at(i) : 0;
  }
}

void ReluOpKernel::dump() { OpKernel::dump(); }

SigmoidOpKernel::SigmoidOpKernel(Operation &op, value_map_t &valueMapping) {
  auto sigmoidOp = cast<tpu::SigmoidOp>(op);
  assert(sigmoidOp);
  llvm::outs() << " Sigmoid op: [" << sigmoidOp.name() << "]\n";

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = sigmoidOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  llvm::outs() << "    =>required memory size: [" << size << "]\n";
  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

  this->name = sigmoidOp.name().str();
  this->op_type = op.getName().getStringRef().str();
  set_datatype(getOpQuant(&op).str());
  if (datatype == DataType::INT8) {
    y0_table_op.assign(opTensors[1]->begin(), opTensors[1]->end());
    slope_table.assign(opTensors[2]->begin(), opTensors[2]->end());
  } else if (datatype == DataType::BF16) {
    y0_bf16_table_op.assign(opTensors[1]->begin(), opTensors[1]->end());
    y0_bf16_slope_table.assign(opTensors[2]->begin(), opTensors[2]->end());
    bf16_min_range = sigmoidOp.min_range().convertToFloat();
    bf16_max_range = sigmoidOp.max_range().convertToFloat();
  }

  // get tensors
  input_data = opTensors[0];
  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}
void SigmoidOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " Sigmoid op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> SigmoidOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void SigmoidOpKernel::invoke() {
  int n = shape[0];
  int c = shape[1];
  int h = shape[2];
  int w = shape[3];
  if (datatype == DataType::INT8) {
    for (size_t i = 0; i < output_data->size(); ++i) {
      output_data->at(i) = y0_table_op.at((unsigned char)input_data->at(i));
    }
  } else if (datatype == DataType::BF16) {
    hardwareLookUpTable(input_data->data(), output_data->data(), n, c, h, w,
                        y0_bf16_table_op, y0_bf16_slope_table, bf16_min_range,
                        bf16_max_range);
  } else {
    for (size_t i = 0; i < output_data->size(); ++i) {
      output_data->at(i) = 0.5 * tanh(0.5 * input_data->at(i)) + 0.5;
    }
  }
}

void SigmoidOpKernel::dump() {
  OpKernel::dump();

  if (this->datatype == DataType::BF16) {
    llvm::outs() << "\tBf16 Range: " << bf16_min_range << " ~ "
                 << bf16_max_range << "\n";
  }
}
} // namespace mlir