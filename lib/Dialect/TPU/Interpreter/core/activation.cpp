#include "tpuc/Interpreter/cpu/activation.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"
#include "tpuc/NativeCpuImplementation.h"

// FIXME: this head should not be here,
//  using interpreter own float convert is better
#include "bmkernel/bm1880v2/1880v2_fp_convert.h"

#include <cmath>

namespace mlir {
int omp_schedule(int count) {
  return (count + omp_get_num_threads() - 1) / omp_get_num_threads();
}

float BF16(float data) { return convert_bf16_fp32(convert_fp32_bf16(data)); }

void relu(float *src, float *dst, size_t size) {
#pragma omp parallel for schedule(static, omp_schedule(size))
  for (size_t i = 0; i < size; ++i) {
    dst[i] = src[i] > 0 ? src[i] : 0;
  }
}

void leaky_relu(float *src, float *dst, size_t size, float negative_slope) {
#pragma omp parallel for schedule(static, omp_schedule(size))
  for (size_t i = 0; i < size; ++i) {
    dst[i] = src[i] > 0 ? src[i] : src[i] * negative_slope;
  }
};

inline float tanh_activate(float x) { return (2 / (1 + expf(-2 * x)) - 1); }

float softplus_activate(float x, float threshold) {
  if (x > threshold)
    return x; // too large
  else if (x < -threshold)
    return expf(x); // too small
  return logf(expf(x) + 1);
}

float mish_caffe_tanh_part(float x_val, float mish_threshold) {
  return tanh_activate(softplus_activate(x_val, mish_threshold));
}

float mish_caffe(float x_val, float mish_threshold) {
  return x_val * mish_caffe_tanh_part(x_val, mish_threshold);
}

AbsOpKernel::AbsOpKernel(Operation &op, value_map_t &valueMapping) {
  auto absOp = cast<tpu::AbsOp>(op);
  assert(absOp);
  LLVM_DEBUG(llvm::outs() << " Abs op: [" << absOp.name() << "]\n";);

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = absOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  LLVM_DEBUG(llvm::outs() << "    =>required memory size: [" << size << "]\n";);
  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

  this->name = absOp.name().str();
  this->op_type = op.getName().getStringRef().str();
  set_datatype(getOpQuant(&op).str());

  // get tensors
  input_data = opTensors[0];
  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}
void AbsOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " Abs op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> AbsOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void AbsOpKernel::invoke() {
  size_t output_size = output_data->size();
#pragma omp parallel for schedule(static, omp_schedule(output_size))
  for (size_t i = 0; i < output_size; ++i) {
    output_data->at(i) = std::fabs(input_data->at(i));
  }
}

void AbsOpKernel::dump() { OpKernel::dump(); }

ExpOpKernel::ExpOpKernel(Operation &op, value_map_t &valueMapping) {
  auto expOp = cast<tpu::ExpOp>(op);
  assert(expOp);
  LLVM_DEBUG(llvm::outs() << " Exp op: [" << expOp.name() << "]\n";);

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = expOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  LLVM_DEBUG(llvm::outs() << "    =>required memory size: [" << size << "]\n";);
  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

  this->name = expOp.name().str();
  this->op_type = op.getName().getStringRef().str();
  set_datatype(getOpQuant(&op).str());

  if (datatype == DataType::INT8) {
    y0_table_op = opTensors[1];
    slope_table = opTensors[2];
  } else if (datatype == DataType::BF16) {
    y0_bf16_table_op = opTensors[1];
    y0_bf16_slope_table = opTensors[2];
    bf16_min_range = expOp.min_range().convertToFloat();
    bf16_max_range = expOp.max_range().convertToFloat();
  }

  // get tensors
  input_data = opTensors[0];
  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}
void ExpOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " Exp op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> ExpOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void ExpOpKernel::invoke() {
  size_t output_size = output_data->size();
  if (datatype == DataType::INT8) {
#pragma omp parallel for schedule(static, omp_schedule(output_size))
    for (size_t i = 0; i < output_size; ++i) {
      output_data->at(i) = y0_table_op->at((unsigned char)input_data->at(i));
    }
  } else if (datatype == DataType::BF16) {
    bf16_lut_slope(input_data->data(), output_data->data(), output_data->size(),
                   *y0_bf16_table_op, *y0_bf16_slope_table, bf16_min_range,
                   bf16_max_range);
  } else {
#pragma omp parallel for schedule(static, omp_schedule(output_size))
    for (size_t i = 0; i < output_size; ++i) {
      output_data->at(i) = std::exp(input_data->at(i));
    }
  }
}

void ExpOpKernel::dump() {
  OpKernel::dump();

  if (this->datatype == DataType::BF16) {
    llvm::outs() << "\tBf16 Range: " << bf16_min_range << " ~ "
                 << bf16_max_range << "\n";
  }
}

MishOpKernel::MishOpKernel(Operation &op, value_map_t &valueMapping) {
  auto mishOp = cast<tpu::MishOp>(op);
  assert(mishOp);
  LLVM_DEBUG(llvm::outs() << " Mish op: [" << mishOp.name() << "]\n";);

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = mishOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  LLVM_DEBUG(llvm::outs() << "    =>required memory size: [" << size << "]\n";);
  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

  this->name = mishOp.name().str();
  this->op_type = op.getName().getStringRef().str();
  this->mish_threshold = mishOp.mish_threshold().convertToFloat();

  set_datatype(getOpQuant(&op).str());

  if (datatype == DataType::INT8) {
    y0_table_op = opTensors[1];
    slope_table = opTensors[2];
  } else if (datatype == DataType::BF16) {
    y0_bf16_table_op = opTensors[1];
    y0_bf16_slope_table = opTensors[2];
    bf16_min_range = mishOp.min_range().convertToFloat();
    bf16_max_range = mishOp.max_range().convertToFloat();
  }

  // get tensors
  input_data = opTensors[0];
  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}
void MishOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " Mish op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> MishOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void MishOpKernel::invoke() {
  int size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());

  if (datatype == DataType::INT8) {
#pragma omp parallel for schedule(static, omp_schedule(size))
    for (size_t i = 0; i < output_data->size(); ++i) {
      output_data->at(i) = y0_table_op->at((unsigned char)input_data->at(i));
    }
  } else if (datatype == DataType::BF16) {
    bf16_lut_slope(input_data->data(), output_data->data(), size,
                   *y0_bf16_table_op, *y0_bf16_slope_table, bf16_min_range,
                   bf16_max_range);
  } else {
#pragma omp parallel for schedule(static, omp_schedule(size))
    for (size_t i = 0; i < output_data->size(); ++i) {
      output_data->at(i) = mish_caffe(input_data->at(i), mish_threshold);
    }
  }
}

void MishOpKernel::dump() { OpKernel::dump(); }

LeakyReluOpKernel::LeakyReluOpKernel(Operation &op, value_map_t &valueMapping) {
  auto leaky_reluOp = cast<tpu::LeakyReluOp>(op);
  assert(leaky_reluOp);
  LLVM_DEBUG(llvm::outs() << " LeakyRelu op: [" << leaky_reluOp.name()
                          << "]\n";);

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = leaky_reluOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  LLVM_DEBUG(llvm::outs() << "    =>required memory size: [" << size << "]\n";);
  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

  this->negative_slope = leaky_reluOp.negative_slope().convertToFloat();

  this->name = leaky_reluOp.name().str();
  this->op_type = op.getName().getStringRef().str();
  set_datatype(getOpQuant(&op).str());
  this->is_asymmetric = isOpQuantAsymmetric(&op);
  if (is_asymmetric) {
    this->input_offset = -getPreviousOpZeroPoint(&op);
    this->output_offset = getOpZeroPoint(&op);
  }
  // get tensors
  input_data = opTensors[0];
  output_data = resultTensor;

  if (datatype == DataType::INT8) {
    assert(opTensors[5]);
    assert(opTensors[6]);
    assert(opTensors[7]);
    assert(opTensors[8]);
    rshift_postive = opTensors[5];
    multiplier_postive = opTensors[6];
    rshift_negative = opTensors[7];
    multiplier_negative = opTensors[8];
  }
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}
void LeakyReluOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " LeakyRelu op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> LeakyReluOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void LeakyReluOpKernel::invoke() {
  int n = shape[0];
  int c = shape[1];
  int h = shape[2];
  int w = shape[3];
  int size = n * c * h * w;
  if (datatype == DataType::FP32) {
    leaky_relu(input_data->data(), output_data->data(), size, negative_slope);
  } else if (datatype == DataType::INT8) {
    if (is_asymmetric) {
      // use copy to change input point,
      // in case that modify original pointer
      std::vector<float> input_copy(input_data->begin(), input_data->end());
      size_t copy_size = input_copy.size();
#pragma omp parallel for schedule(static, omp_schedule(copy_size))
      for (size_t i = 0; i < copy_size; i++) {
        input_copy.at(i) += input_offset;
      }
      bool do_pos_scale = (multiplier_postive->at(0) != 0.0) ? true : false;
#pragma omp parallel for schedule(static, omp_schedule(size))
      for (int i = 0; i < size; ++i) {
        if (input_copy.at(i) > 0) {
          if (do_pos_scale) {
            output_data->at(i) = (float)applyMultiplierAndRShiftAndSaturateInt8(
                input_copy.at(i), (uint32_t)rshift_postive->at(0),
                multiplier_postive->at(0), false, output_offset);
          } else {
            output_data->at(i) = input_copy.at(i);
          }
        } else {
          output_data->at(i) = (float)applyMultiplierAndRShiftAndSaturateInt8(
              input_copy.at(i), (uint32_t)rshift_negative->at(0),
              multiplier_negative->at(0), false, output_offset);
        }
      }
    } else {
      bool do_pos_scale = (multiplier_postive->at(0) != 0.0) ? true : false;

      for (int i = 0; i < size; ++i) {
        if (input_data->at(i) > 0) {
          if (do_pos_scale) {
            output_data->at(i) = (float)applyMultiplierAndRShiftAndSaturateInt8(
                input_data->at(i), (uint32_t)rshift_postive->at(0),
                multiplier_postive->at(0), false);
          } else {
            output_data->at(i) = input_data->at(i);
          }
        } else {
          output_data->at(i) = (float)applyMultiplierAndRShiftAndSaturateInt8(
              input_data->at(i), (uint32_t)rshift_negative->at(0),
              multiplier_negative->at(0), false);
        }
      }
    }
  } else if (datatype == DataType::BF16) {
    leaky_relu(input_data->data(), output_data->data(), size, negative_slope);
    clean16bitmantissa(output_data->data(), output_data->data(),
                       output_data->size());
  }
}

void LeakyReluOpKernel::dump() { OpKernel::dump(); }

ReluOpKernel::ReluOpKernel(Operation &op, value_map_t &valueMapping) {
  auto reluOp = cast<tpu::ReluOp>(op);
  assert(reluOp);
  LLVM_DEBUG(llvm::outs() << " Relu op: [" << reluOp.name() << "]\n";);

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = reluOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  LLVM_DEBUG(llvm::outs() << "    =>required memory size: [" << size << "]\n";);
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
  relu(input_data->data(), output_data->data(), output_data->size());
}

void ReluOpKernel::dump() { OpKernel::dump(); }

PReluOpKernel::PReluOpKernel(Operation &op, value_map_t &valueMapping) {
  auto preluOp = cast<tpu::PReluOp>(op);
  assert(preluOp);
  LLVM_DEBUG(llvm::outs() << " PRelu op: [" << preluOp.name() << "]\n";);

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = preluOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  LLVM_DEBUG(llvm::outs() << "    =>required memory size: [" << size << "]\n";);
  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

  this->slope_data = opTensors[1];

  this->name = preluOp.name().str();
  this->op_type = op.getName().getStringRef().str();
  set_datatype(getOpQuant(&op).str());

  // get tensors
  input_data = opTensors[0];
  output_data = resultTensor;

  if (datatype == DataType::INT8) {
    assert(opTensors[6]);
    assert(opTensors[7]);
    assert(opTensors[8]);
    this->rshift_postive = opTensors[6];
    this->multiplier_postive = opTensors[7];
    this->rshift_negative = opTensors[8];
  }
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}
void PReluOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " PRelu op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> PReluOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void PReluOpKernel::invoke() {
  int n = shape[0];
  int c = shape[1];
  int h = shape[2];
  int w = shape[3];
  int size = n * c * h * w;
  for (int batch = 0; batch < n; ++batch) {
    for (int channel = 0; channel < c; ++channel) {
      int index = batch * c * w * h + channel * w * h;
      int planner = w * h;
#pragma omp parallel for schedule(static, omp_schedule(planner))
      for (int i = 0; i < planner; ++i) {
        if (input_data->at(index + i) > 0) {
          output_data->at(index + i) = input_data->at(index + i);
        } else {
          output_data->at(index + i) =
              slope_data->at(channel) * input_data->at(index + i);
        }
      }
    }
  }
  if (datatype == DataType::INT8) {
#pragma omp parallel for schedule(static, omp_schedule(size))
    for (int i = 0; i < size; ++i) {
      if (input_data->at(i) > 0) {
        output_data->at(i) = (float)applyMultiplierAndRShiftAndSaturateInt8(
            output_data->at(i), (uint32_t)rshift_postive->at(0),
            multiplier_postive->at(0), false);
      } else {
        output_data->at(i) = (float)applyRShiftAndSaturateInt8(
            output_data->at(i), (uint32_t)rshift_negative->at(0));
      }
    }
  } else if (datatype == DataType::BF16) {
    clean16bitmantissa(output_data->data(), output_data->data(),
                       output_data->size());
  }
}

void PReluOpKernel::dump() { OpKernel::dump(); }

ReciprocalOpKernel::ReciprocalOpKernel(Operation &op,
                                       value_map_t &valueMapping) {
  auto rOp = cast<tpu::ReciprocalOp>(op);
  assert(rOp);
  LLVM_DEBUG(llvm::outs() << " Reciprocal op: [" << rOp.name() << "]\n";);

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = rOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  LLVM_DEBUG(llvm::outs() << "    =>required memory size: [" << size << "]\n";);
  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

  this->name = rOp.name().str();
  this->op_type = op.getName().getStringRef().str();
  set_datatype(getOpQuant(&op).str());
  if (datatype == DataType::INT8) {
    y0_table_op = opTensors[1];
    slope_table = opTensors[2];
  } else if (datatype == DataType::BF16) {
    y0_bf16_table_op = opTensors[1];
    y0_bf16_slope_table = opTensors[2];
    bf16_min_range = rOp.min_range().convertToFloat();
    bf16_max_range = rOp.max_range().convertToFloat();
  }

  // get tensors
  input_data = opTensors[0];
  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}
void ReciprocalOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " Reciprocal op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> ReciprocalOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void ReciprocalOpKernel::invoke() {
  size_t output_size = output_data->size();
  if (datatype == DataType::INT8) {
#pragma omp parallel for schedule(static, omp_schedule(output_size))
    for (size_t i = 0; i < output_size; ++i) {
      output_data->at(i) = y0_table_op->at((unsigned char)input_data->at(i));
    }
  } else {
#pragma omp parallel for schedule(static, omp_schedule(output_size))
    for (size_t i = 0; i < output_size; ++i) {
      output_data->at(i) = 1.0 / input_data->at(i);
    }
  }
}

void ReciprocalOpKernel::dump() {
  OpKernel::dump();

  if (this->datatype == DataType::BF16) {
    llvm::outs() << "\tBf16 Range: " << bf16_min_range << " ~ "
                 << bf16_max_range << "\n";
  }
}

ReshapeOpKernel::ReshapeOpKernel(Operation &op, value_map_t &valueMapping) {
  auto reshapeOp = cast<tpu::ReshapeOp>(op);
  assert(reshapeOp);
  LLVM_DEBUG(llvm::outs() << " PRelu op: [" << reshapeOp.name() << "]\n";);

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = reshapeOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  LLVM_DEBUG(llvm::outs() << "    =>required memory size: [" << size << "]\n";);
  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

  this->name = reshapeOp.name().str();
  this->op_type = op.getName().getStringRef().str();
  set_datatype(getOpQuant(&op).str());

  // get tensors
  input_data = opTensors[0];
  output_data = opTensors[0];

  // record mapping table for next op connecting
  valueMapping[result] = std::move(opTensors[0]);
}
void ReshapeOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " PRelu op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> ReshapeOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void ReshapeOpKernel::invoke() {
  // reshape op no need to invoke, skip
}

void ReshapeOpKernel::dump() { OpKernel::dump(); }

SigmoidOpKernel::SigmoidOpKernel(Operation &op, value_map_t &valueMapping) {
  auto sigmoidOp = cast<tpu::SigmoidOp>(op);
  assert(sigmoidOp);
  LLVM_DEBUG(llvm::outs() << " Sigmoid op: [" << sigmoidOp.name() << "]\n";);

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = sigmoidOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  LLVM_DEBUG(llvm::outs() << "    =>required memory size: [" << size << "]\n";);
  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

  this->name = sigmoidOp.name().str();
  this->op_type = op.getName().getStringRef().str();
  set_datatype(getOpQuant(&op).str());
  if (datatype == DataType::INT8) {
    y0_table_op = opTensors[1];
    slope_table = opTensors[2];
  } else if (datatype == DataType::BF16) {
    y0_bf16_table_op = opTensors[1];
    y0_bf16_slope_table = opTensors[2];
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
  size_t output_size = output_data->size();
  if (datatype == DataType::INT8) {
#pragma omp parallel for schedule(static, omp_schedule(output_size))
    for (size_t i = 0; i < output_size; ++i) {
      output_data->at(i) = y0_table_op->at((unsigned char)input_data->at(i));
    }
  } else if (datatype == DataType::BF16) {
    bf16_lut_slope(input_data->data(), output_data->data(), output_size,
                   *y0_bf16_table_op, *y0_bf16_slope_table, bf16_min_range,
                   bf16_max_range);
  } else {
#pragma omp parallel for schedule(static, omp_schedule(output_size))
    for (size_t i = 0; i < output_size; ++i) {
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

SoftPlusOpKernel::SoftPlusOpKernel(Operation &op, value_map_t &valueMapping) {
  auto spOp = cast<tpu::SoftPlusOp>(op);
  assert(spOp);
  LLVM_DEBUG(llvm::outs() << " SoftPlus op: [" << spOp.name() << "]\n";);

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = spOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  LLVM_DEBUG(llvm::outs() << "    =>required memory size: [" << size << "]\n";);
  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();
  this->threshold = spOp.threshold().convertToFloat();
  this->name = spOp.name().str();
  this->op_type = op.getName().getStringRef().str();
  set_datatype(getOpQuant(&op).str());
  if (datatype == DataType::INT8) {
    y0_table_op = opTensors[1];
    slope_table = opTensors[2];
  } else if (datatype == DataType::BF16) {
    y0_bf16_table_op = opTensors[1];
    y0_bf16_slope_table = opTensors[2];
    bf16_min_range = spOp.min_range().convertToFloat();
    bf16_max_range = spOp.max_range().convertToFloat();
  }

  // get tensors
  input_data = opTensors[0];
  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}
void SoftPlusOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " SoftPlus op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> SoftPlusOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void SoftPlusOpKernel::invoke() {
  size_t output_size = output_data->size();
  if (datatype == DataType::INT8) {
#pragma omp parallel for schedule(static, omp_schedule(output_size))
    for (size_t i = 0; i < output_size; ++i) {
      output_data->at(i) = y0_table_op->at((unsigned char)input_data->at(i));
    }
  } else if (datatype == DataType::BF16) {
    bf16_lut_slope(input_data->data(), output_data->data(), output_data->size(),
                   *y0_bf16_table_op, *y0_bf16_slope_table, bf16_min_range,
                   bf16_max_range);
  } else {
#pragma omp parallel for schedule(static, omp_schedule(output_size))
    for (size_t i = 0; i < output_data->size(); ++i) {
      output_data->at(i) = softplus_activate(input_data->at(i), threshold);
    }
  }
}

void SoftPlusOpKernel::dump() {
  OpKernel::dump();

  if (this->datatype == DataType::BF16) {
    llvm::outs() << "\tBf16 Range: " << bf16_min_range << " ~ "
                 << bf16_max_range << "\n";
  }
}

SqrtOpKernel::SqrtOpKernel(Operation &op, value_map_t &valueMapping) {
  auto sqrtOp = cast<tpu::SqrtOp>(op);
  assert(sqrtOp);
  LLVM_DEBUG(llvm::outs() << " Sqrt op: [" << sqrtOp.name() << "]\n";);

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = sqrtOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  LLVM_DEBUG(llvm::outs() << "    =>required memory size: [" << size << "]\n";);
  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

  this->name = sqrtOp.name().str();
  this->op_type = op.getName().getStringRef().str();
  set_datatype(getOpQuant(&op).str());
  if (datatype == DataType::INT8) {
    y0_table_op = opTensors[1];
    slope_table = opTensors[2];
  } else if (datatype == DataType::BF16) {
    y0_bf16_table_op = opTensors[1];
    y0_bf16_slope_table = opTensors[2];
    bf16_min_range = sqrtOp.min_range().convertToFloat();
    bf16_max_range = sqrtOp.max_range().convertToFloat();
  }

  // get tensors
  input_data = opTensors[0];
  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}
void SqrtOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " Sqrt op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> SqrtOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void SqrtOpKernel::invoke() {
  size_t output_size = output_data->size();
  if (datatype == DataType::INT8) {
#pragma omp parallel for schedule(static, omp_schedule(output_size))
    for (size_t i = 0; i < output_size; ++i) {
      output_data->at(i) = y0_table_op->at((unsigned char)input_data->at(i));
    }
  } else {
#pragma omp parallel for schedule(static, omp_schedule(output_size))
    for (size_t i = 0; i < output_size; ++i) {
      output_data->at(i) = pow(input_data->at(i), 0.5);
    }
  }
}

void SqrtOpKernel::dump() {
  OpKernel::dump();

  if (this->datatype == DataType::BF16) {
    llvm::outs() << "\tBf16 Range: " << bf16_min_range << " ~ "
                 << bf16_max_range << "\n";
  }
}

SquareOpKernel::SquareOpKernel(Operation &op, value_map_t &valueMapping) {
  auto squareOp = cast<tpu::SquareOp>(op);
  assert(squareOp);
  LLVM_DEBUG(llvm::outs() << " Square op: [" << squareOp.name() << "]\n";);

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = squareOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  LLVM_DEBUG(llvm::outs() << "    =>required memory size: [" << size << "]\n";);
  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

  this->name = squareOp.name().str();
  this->op_type = op.getName().getStringRef().str();
  set_datatype(getOpQuant(&op).str());

  // get tensors
  input_data = opTensors[0];
  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}
void SquareOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " Square op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> SquareOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void SquareOpKernel::invoke() {
  size_t output_size = output_data->size();
#pragma omp parallel for schedule(static, omp_schedule(output_size))
  for (size_t i = 0; i < output_size; ++i) {
    output_data->at(i) = input_data->at(i) * input_data->at(i);
  }
  if (datatype == DataType::INT8) {
    llvm_unreachable("No int8 sqaure");
  } else if (datatype == DataType::BF16) {
    clean16bitmantissa(output_data->data(), output_data->data(),
                       output_data->size());
  }
}

void SquareOpKernel::dump() { OpKernel::dump(); }

TanHOpKernel::TanHOpKernel(Operation &op, value_map_t &valueMapping) {
  auto tanhOp = cast<tpu::TanHOp>(op);
  assert(tanhOp);
  LLVM_DEBUG(llvm::outs() << " TanH op: [" << tanhOp.name() << "]\n";);

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = tanhOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  LLVM_DEBUG(llvm::outs() << "    =>required memory size: [" << size << "]\n";);
  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

  this->name = tanhOp.name().str();
  this->op_type = op.getName().getStringRef().str();
  set_datatype(getOpQuant(&op).str());
  if (datatype == DataType::INT8) {
    y0_table_op = opTensors[1];
    slope_table = opTensors[2];
  } else if (datatype == DataType::BF16) {
    y0_bf16_table_op = opTensors[1];
    y0_bf16_slope_table = opTensors[2];
    bf16_min_range = tanhOp.min_range().convertToFloat();
    bf16_max_range = tanhOp.max_range().convertToFloat();
  }

  // get tensors
  input_data = opTensors[0];
  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}
void TanHOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " TanH op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> TanHOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void TanHOpKernel::invoke() {
  size_t output_size = output_data->size();
  if (datatype == DataType::INT8) {
#pragma omp parallel for schedule(static, omp_schedule(output_size))
    for (size_t i = 0; i < output_size; ++i) {
      output_data->at(i) = y0_table_op->at((unsigned char)input_data->at(i));
    }
  } else if (datatype == DataType::BF16) {
    bf16_lut_slope(input_data->data(), output_data->data(), output_data->size(),
                   *y0_bf16_table_op, *y0_bf16_slope_table, bf16_min_range,
                   bf16_max_range);
  } else {

#pragma omp parallel for schedule(static, omp_schedule(output_size))
    for (size_t i = 0; i < output_size; ++i) {
      output_data->at(i) = std::tanh(input_data->at(i));
    }
  }
}

void TanHOpKernel::dump() {
  OpKernel::dump();

  if (this->datatype == DataType::BF16) {
    llvm::outs() << "\tBf16 Range: " << bf16_min_range << " ~ "
                 << bf16_max_range << "\n";
  }
}
} // namespace mlir