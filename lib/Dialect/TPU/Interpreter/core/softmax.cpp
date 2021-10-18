#include "tpuc/Interpreter/cpu/softmax.hpp"
#include "internal.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Interpreter/cpu/lut_func.hpp"
#include "tpuc/MlirModuleInterpreter.h"
#include "tpuc/NativeCpuImplementation.h"

namespace mlir {

SoftmaxOpKernel::SoftmaxOpKernel(Operation &op, value_map_t &valueMapping,
                                 weight_map_t &weightMapping, bool cpu)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  if (cpu) {
    auto castOp = cast<tpu::SoftmaxCpuOp>(op);
    this->axis = castOp.axis();
    this->do_log = false;
  } else {
    auto castOp = cast<tpu::SoftmaxOp>(op);
    this->axis = castOp.axis();
    this->do_log = castOp.do_log();
  }
  assert(this->axis >= 0);
  // get tensors
  input_data = this->opdTensors[0];
  if (this->opdTensors.size() == 5) {
    exp_table = this->opdTensors[1];
    exp_slope_table = this->opdTensors[2];
    reciprocal_table = this->opdTensors[3];
    reciprocal_mantissa_table = this->opdTensors[4];
  }
  output_data = this->resTensor;

  outer_dim = 1;
  for (int i = 0; i < axis; ++i) {
    outer_dim *= shape[i];
  }

  inner_dim = 1;
  for (int i = axis + 1; i < (int)shape.size(); ++i) {
    inner_dim *= shape[i];
  }
  channel = shape[axis];
  max_arr = new float[inner_dim];
  sum_arr = new float[inner_dim];
  ex_arr = new float[channel * inner_dim];
  sub_arr = new float[channel * inner_dim];
}

void SoftmaxOpKernel::invoke_fp32() {
  auto bottom_data = input_data->data();
  auto top_data = output_data->data();

  for (int i = 0; i < outer_dim; ++i) {
    // find max value accross channel
    int c_offset = i * channel * inner_dim;
    memcpy(max_arr, bottom_data + c_offset, inner_dim * sizeof(float));
    for (int j = 0; j < channel; ++j, c_offset += inner_dim) {
      for (int k = 0; k < inner_dim; k++) {
        if (max_arr[k] < bottom_data[c_offset + k])
          max_arr[k] = bottom_data[c_offset + k];
      }
    }

    // calculate exp(x)
    c_offset = i * channel * inner_dim;
    memset(sum_arr, 0, inner_dim * sizeof(float));
    for (int j = 0; j < channel; ++j, c_offset += inner_dim) {
      for (int k = 0; k < inner_dim; k++) {
        sub_arr[j * inner_dim + k] = bottom_data[c_offset + k] - max_arr[k];
        top_data[c_offset + k] = std::exp(sub_arr[j * inner_dim + k]);
        sum_arr[k] += top_data[c_offset + k];
      }
    }

    c_offset = i * channel * inner_dim;
    for (int j = 0; j < channel; ++j, c_offset += inner_dim) {
      for (int k = 0; k < inner_dim; k++) {
        if (do_log == false) {
          top_data[c_offset + k] /= sum_arr[k];
        } else {
          top_data[c_offset + k] = sub_arr[j * inner_dim + k] - std::log(sum_arr[k]);
        }
      }
    }
  }
}

void SoftmaxOpKernel::invoke_bf16() {
  auto bottom_data = input_data->data();
  auto top_data = output_data->data();

  for (int i = 0; i < outer_dim; ++i) {
    // find max value accross channel
    int c_offset = i * channel * inner_dim;
    memcpy(max_arr, bottom_data + c_offset, inner_dim * sizeof(float));
    for (int j = 0; j < channel; ++j, c_offset += inner_dim) {
      for (int k = 0; k < inner_dim; k++) {
        if (max_arr[k] < bottom_data[c_offset + k])
          max_arr[k] = bottom_data[c_offset + k];
      }
    }

    // calculate x - max
    c_offset = i * channel * inner_dim;
    for (int j = 0; j < channel; ++j, c_offset += inner_dim) {
      for (int k = 0; k < inner_dim; k++) {
        auto idx = j * inner_dim + k;
        sub_arr[idx] = BF16(bottom_data[c_offset + k] - max_arr[k]);
      }
    }

    // e^x
    bf16_lut_slope("exp", sub_arr, ex_arr, channel * inner_dim,
                   exp_table->data(), exp_slope_table->data());

    // sum of (e^x)
    float const_val = BF16(BF16(1.0 * channel) / channel);
    memset(sum_arr, 0, inner_dim * sizeof(float));
    for (int j = 0; j < channel; ++j) {
      for (int k = 0; k < inner_dim; k++) {
        auto idx = j * inner_dim + k;
        sum_arr[k] += ex_arr[idx] * const_val;
      }
    }

    // convert to bf16
    for (int k = 0; k < inner_dim; k++) {
      sum_arr[k] = BF16(sum_arr[k]);
    }
    if (do_log == false) {
      // 1 / (e ^x)
      bf16_lut_mantissa(sum_arr, sum_arr, inner_dim, reciprocal_table->data(),
                        reciprocal_mantissa_table->data());
    } else {
      bf16_lut_slope("log", sum_arr, sum_arr, inner_dim, reciprocal_table->data(),
                        reciprocal_mantissa_table->data());
    }

    c_offset = i * channel * inner_dim;
    for (int j = 0; j < channel; ++j, c_offset += inner_dim) {
      for (int k = 0; k < inner_dim; k++) {
        auto idx = j * inner_dim + k;
        if (do_log == false) {
          top_data[c_offset + k] = BF16(ex_arr[idx] * sum_arr[k]);
        } else {
          top_data[c_offset + k] = BF16(sub_arr[idx] - sum_arr[k]);
        }
      }
    }
  }
}

void SoftmaxOpKernel::invoke() {
  if (datatype == DataType::FP32) {
    return invoke_fp32();
  } else if (datatype == DataType::BF16) {
    return invoke_bf16();
  } else {
    llvm_unreachable("TODO");
  }
}

} // namespace mlir
