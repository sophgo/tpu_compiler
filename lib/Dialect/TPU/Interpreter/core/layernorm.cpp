#include "tpuc/Interpreter/cpu/layernorm.hpp"
#include "bmkernel/bm1880v2/1880v2_fp_convert.h"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"
#include "tpuc/NativeCpuImplementation.h"

namespace mlir {
static inline float BF16(float data) {
  return convert_bf16_fp32(convert_fp32_bf16(data));
}

LayerNormOpKernel::LayerNormOpKernel(Operation &op, value_map_t &valueMapping) {
  auto lnOp = cast<tpu::LayerNormOp>(op);
  assert(lnOp);
  LLVM_DEBUG(llvm::outs() << " LayerNorm op: [" << lnOp.name() << "]\n";);

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = lnOp.getResult();
  auto size = getTensorSize(result);
  LLVM_DEBUG(llvm::outs() << "    =>required memory size: [" << size << "]\n";);
  auto resultTensor = std::make_shared<std::vector<float>>(size);

  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

  this->name = lnOp.name().str();
  this->op_type = op.getName().getStringRef().str();
  set_datatype(getOpQuant(&op).str());
  // get tensors
  input_data = opTensors[0];
  eps = lnOp.eps().convertToFloat();
  arrayAttrToVector(lnOp.normalized_shape(), this->normalized_shape);
  output_data = resultTensor;
  normalized_size = 1;
  batch_size = 1;
  size_t nm_dims = normalized_shape.size();
  size_t input_dims = shape.size();
  assert(nm_dims <= input_dims);
  size_t index = 0;
  for (; index < input_dims - nm_dims; index++) {
    batch_size *= shape[index];
  }
  for (size_t i = 0; index < input_dims; index++, i++) {
    assert(normalized_shape[i] == shape[index]);
    normalized_size *= normalized_shape[i];
  }
  if (datatype == DataType::BF16) {
    lut.assign(opTensors[1]->begin(), opTensors[1]->end());
    mantissa_lut.assign(opTensors[2]->begin(), opTensors[2]->end());
  }
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}

void LayerNormOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " LayerNorm op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> LayerNormOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void LayerNormOpKernel::normalize_fp32(float *src, float *dst, int size) {
  double sum = std::accumulate(src, src + size, 0.0f);
  double mean = sum / size;
  double var = eps;
  for (int i = 0; i < size; i++) {
    var += std::pow(src[i] - mean, 2);
  }
  double div = std::sqrt(var);
  for (int i = 0; i < size; i++) {
    dst[i] = (src[i] - mean) / div;
  }
}

void LayerNormOpKernel::normalize_bf16(float *src, float *dst, int size) {
  float sum = BF16(std::accumulate(src, src + size, 0.0f));
  float mean = BF16(sum / size);
  float var = 0;
  float data;
  for (int i = 0; i < size; i++) {
    data = BF16(src[i] - mean);
    var += std::pow(data, 2);
  }
  var = BF16(var) + BF16(eps);
  bf16_lut_mantissa(&var, &var, 1, lut, mantissa_lut);
  for (int i = 0; i < size; i++) {
    data = BF16(src[i] - mean);
    dst[i] = BF16(data * var);
  }
}

void LayerNormOpKernel::invoke() {
  switch (datatype) {
  case DataType::FP32:
    for (int i = 0; i < batch_size; i++) {
      normalize_fp32(input_data->data() + i * normalized_size,
                     output_data->data() + i * normalized_size,
                     normalized_size);
    }
    return;
  case DataType::BF16:
    for (int i = 0; i < batch_size; i++) {
      normalize_bf16(input_data->data() + i * normalized_size,
                     output_data->data() + i * normalized_size,
                     normalized_size);
    }
    return;
  default:
    break;
  }
  llvm_unreachable("unsupport datatype");
}

void LayerNormOpKernel::dump() { OpKernel::dump(); }
} // namespace mlir