#include "tpuc/Interpreter/cpu/activation.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Interpreter/cpu/lut_func.hpp"
#include "tpuc/MlirModuleInterpreter.h"
#include "tpuc/NativeCpuImplementation.h"

#include "internal.hpp"

#include <cmath>

namespace mlir {

AbsOpKernel::AbsOpKernel(Operation &op, value_map_t &valueMapping,
                         weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void AbsOpKernel::invoke() {
  size_t output_size = output_data->size();
#pragma omp parallel for schedule(static, omp_schedule(output_size))
  for (size_t i = 0; i < output_size; ++i) {
    output_data->at(i) = std::fabs(input_data->at(i));
  }
}

ExpOpKernel::ExpOpKernel(Operation &op, value_map_t &valueMapping,
                         weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  auto expOp = cast<tpu::ExpOp>(op);
  if (datatype == DataType::INT8) {
    y0_table_op = this->opdTensors[1];
    slope_table = this->opdTensors[2];
  } else if (datatype == DataType::BF16) {
    y0_bf16_table_op = this->opdTensors[1];
    y0_bf16_slope_table = this->opdTensors[2];
  }
  scale = expOp.scale().convertToFloat();
  bias = expOp.bias().convertToFloat();
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void ExpOpKernel::invoke() {
  size_t output_size = output_data->size();
  if (datatype == DataType::INT8) {
#pragma omp parallel for schedule(static, omp_schedule(output_size))
    for (size_t i = 0; i < output_size; ++i) {
      output_data->at(i) = y0_table_op->at((unsigned char)input_data->at(i));
    }
  } else if (datatype == DataType::BF16) {
    bf16_lut_slope("exp", input_data->data(), output_data->data(),
                   output_data->size(), y0_bf16_table_op->data(),
                   y0_bf16_slope_table->data());
  } else {
#pragma omp parallel for schedule(static, omp_schedule(output_size))
    for (size_t i = 0; i < output_size; ++i) {
      output_data->at(i) = scale * std::exp(input_data->at(i)) + bias;
    }
  }
}

MishOpKernel::MishOpKernel(Operation &op, value_map_t &valueMapping,
                           weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  auto mishOp = cast<tpu::MishOp>(op);
  if (datatype == DataType::INT8) {
    y0_table_op = this->opdTensors[1];
    slope_table = this->opdTensors[2];
  } else if (datatype == DataType::BF16) {
    y0_bf16_table_op = this->opdTensors[1];
    y0_bf16_slope_table = this->opdTensors[2];
    bf16_min_range = mishOp.min_range().convertToFloat();
    bf16_max_range = mishOp.max_range().convertToFloat();
  }
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
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
    bf16_lut_slope("mish", input_data->data(), output_data->data(), size,
                   y0_bf16_table_op->data(), y0_bf16_slope_table->data());
  } else {
#pragma omp parallel for schedule(static, omp_schedule(size))
    for (size_t i = 0; i < output_data->size(); ++i) {
      output_data->at(i) = my_mish_activate(input_data->at(i));
    }
  }
}

LeakyReluOpKernel::LeakyReluOpKernel(Operation &op, value_map_t &valueMapping,
                                     weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  auto leaky_reluOp = cast<tpu::LeakyReluOp>(op);

  this->negative_slope = leaky_reluOp.negative_slope().convertToFloat();
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;

  if (datatype == DataType::INT8) {
    assert(this->opdTensors[5]);
    assert(this->opdTensors[6]);
    assert(this->opdTensors[7]);
    assert(this->opdTensors[8]);
    rshift_postive = this->opdTensors[5];
    multiplier_postive = this->opdTensors[6];
    rshift_negative = this->opdTensors[7];
    multiplier_negative = this->opdTensors[8];
  }
}

void LeakyReluOpKernel::invoke() {
  int size = std::accumulate(std::begin(shape), std::end(shape), 1,
                         std::multiplies<>());
  if (datatype == DataType::FP32) {
    leaky_relu(input_data->data(), output_data->data(), size, negative_slope);
  } else if (datatype == DataType::INT8) {
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
  } else if (datatype == DataType::BF16) {
    leaky_relu(input_data->data(), output_data->data(), size, negative_slope);
    BF16(output_data->data(), output_data->data(), output_data->size());
  }
}

ReluOpKernel::ReluOpKernel(Operation &op, value_map_t &valueMapping,
                           weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void ReluOpKernel::invoke() {
  relu(input_data->data(), output_data->data(), output_data->size());
}

PReluOpKernel::PReluOpKernel(Operation &op, value_map_t &valueMapping,
                             weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;

  this->slope_data = this->opdTensors[1];
  if (datatype == DataType::INT8) {
    assert(this->opdTensors[6]);
    assert(this->opdTensors[7]);
    assert(this->opdTensors[8]);
    this->rshift_postive = this->opdTensors[6];
    this->multiplier_postive = this->opdTensors[7];
    this->rshift_negative = this->opdTensors[8];
  }
}

void PReluOpKernel::invoke() {
  int64_t n,c,h,w;
  getNCHW(shape, n, c, h, w);
  int size = n * c * h * w;
  int planner = w * h;
  for (int batch = 0; batch < n; ++batch) {
    for (int channel = 0; channel < c; ++channel) {
      int index = batch * c * w * h + channel * w * h;
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
    BF16(output_data->data(), output_data->data(), output_data->size());
  }
}

ReshapeOpKernel::ReshapeOpKernel(Operation &op, value_map_t &valueMapping,
                                 weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  // get tensors
  input_data = this->opdTensors[0];
  this->resTensor.reset();
  this->resTensor = this->opdTensors[0];
  output_data = this->resTensor;
  valueMapping[op.getResult(0)] = this->resTensor;
}

void ReshapeOpKernel::invoke() {}

SigmoidOpKernel::SigmoidOpKernel(Operation &op, value_map_t &valueMapping,
                                 weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  auto sigmoidOp = cast<tpu::SigmoidOp>(op);
  if (datatype == DataType::INT8) {
    y0_table_op = this->opdTensors[1];
    slope_table = this->opdTensors[2];
  } else if (datatype == DataType::BF16) {
    y0_bf16_table_op = this->opdTensors[1];
    y0_bf16_slope_table = this->opdTensors[2];
  }
  scale = sigmoidOp.scale().convertToFloat();
  bias = sigmoidOp.bias().convertToFloat();
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void SigmoidOpKernel::invoke() {
  size_t output_size = output_data->size();
  if (datatype == DataType::INT8) {
#pragma omp parallel for schedule(static, omp_schedule(output_size))
    for (size_t i = 0; i < output_size; ++i) {
      output_data->at(i) = y0_table_op->at((unsigned char)input_data->at(i));
    }
  } else if (datatype == DataType::BF16) {
    bf16_lut_slope("sigmoid", input_data->data(), output_data->data(),
                   output_size, y0_bf16_table_op->data(),
                   y0_bf16_slope_table->data());
  } else {
#pragma omp parallel for schedule(static, omp_schedule(output_size))
    for (size_t i = 0; i < output_size; ++i) {
      output_data->at(i) = scale / (1 + expf(-1 * input_data->at(i))) + bias;
    }
  }
}

SwishOpKernel::SwishOpKernel(Operation &op, value_map_t &valueMapping,
                             weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  auto swishOp = cast<tpu::SwishOp>(op);
  if (datatype == DataType::INT8) {
    y0_table_op = this->opdTensors[1];
    slope_table = this->opdTensors[2];
  } else if (datatype == DataType::BF16) {
    y0_bf16_table_op = this->opdTensors[1];
    y0_bf16_slope_table = this->opdTensors[2];
    bf16_min_range = swishOp.min_range().convertToFloat();
    bf16_max_range = swishOp.max_range().convertToFloat();
  }
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void SwishOpKernel::invoke() {
  size_t output_size = output_data->size();
  if (datatype == DataType::INT8) {
#pragma omp parallel for schedule(static, omp_schedule(output_size))
    for (size_t i = 0; i < output_size; ++i) {
      output_data->at(i) = y0_table_op->at((unsigned char)input_data->at(i));
    }
  } else if (datatype == DataType::BF16) {
    bf16_lut_slope("swish", input_data->data(), output_data->data(),
                   output_size, y0_bf16_table_op->data(),
                   y0_bf16_slope_table->data());
  } else {
#pragma omp parallel for schedule(static, omp_schedule(output_size))
    for (size_t i = 0; i < output_size; ++i) {
      auto val = input_data->at(i);
      output_data->at(i) = val / (1 + std::exp(-val));
    }
  }
}

EluOpKernel::EluOpKernel(Operation &op, value_map_t &valueMapping,
                         weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  auto eluOp = cast<tpu::EluOp>(op);
  if (datatype == DataType::INT8) {
    y0_table_op = this->opdTensors[1];
    slope_table = this->opdTensors[2];
  } else if (datatype == DataType::BF16) {
    y0_bf16_table_op = this->opdTensors[1];
    y0_bf16_slope_table = this->opdTensors[2];
    bf16_min_range = eluOp.min_range().convertToFloat();
    bf16_max_range = eluOp.max_range().convertToFloat();
  }
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void EluOpKernel::invoke() {
  size_t output_size = output_data->size();
  if (datatype == DataType::INT8) {
#pragma omp parallel for schedule(static, omp_schedule(output_size))
    for (size_t i = 0; i < output_size; ++i) {
      output_data->at(i) = y0_table_op->at((unsigned char)input_data->at(i));
    }
  } else if (datatype == DataType::BF16) {
    bf16_lut_slope("elu", input_data->data(), output_data->data(), output_size,
                   y0_bf16_table_op->data(), y0_bf16_slope_table->data());
  } else {
#pragma omp parallel for schedule(static, omp_schedule(output_size))
    for (size_t i = 0; i < output_size; ++i) {
      auto val = input_data->at(i);
      output_data->at(i) = (val >= 0) ? val : (std::exp(val) - 1);
    }
  }
}

SoftPlusOpKernel::SoftPlusOpKernel(Operation &op, value_map_t &valueMapping,
                                   weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  auto spOp = cast<tpu::SoftPlusOp>(op);
  if (datatype == DataType::INT8) {
    y0_table_op = this->opdTensors[1];
    slope_table = this->opdTensors[2];
  } else if (datatype == DataType::BF16) {
    y0_bf16_table_op = this->opdTensors[1];
    y0_bf16_slope_table = this->opdTensors[2];
  }
  scale = spOp.scale().convertToFloat();
  bias = spOp.bias().convertToFloat();
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void SoftPlusOpKernel::invoke() {
  size_t output_size = output_data->size();
  if (datatype == DataType::INT8) {
#pragma omp parallel for schedule(static, omp_schedule(output_size))
    for (size_t i = 0; i < output_size; ++i) {
      output_data->at(i) = y0_table_op->at((unsigned char)input_data->at(i));
    }
  } else if (datatype == DataType::BF16) {
    bf16_lut_slope("softplus", input_data->data(), output_data->data(),
                   output_data->size(), y0_bf16_table_op->data(),
                   y0_bf16_slope_table->data());
  } else {
#pragma omp parallel for schedule(static, omp_schedule(output_size))
    for (size_t i = 0; i < output_data->size(); ++i) {
      output_data->at(i) = scale * logf(expf(input_data->at(i)) + 1) + bias;
    }
  }
}

PowOpKernel::PowOpKernel(Operation &op, value_map_t &valueMapping,
                           weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  auto powOp = cast<tpu::PowOp>(op);
  if (datatype == DataType::INT8) {
    y0_table_op = this->opdTensors[1];
    slope_table = this->opdTensors[2];
  } else if (datatype == DataType::BF16) {
    y0_bf16_table_op = this->opdTensors[1];
    y0_bf16_slope_table = this->opdTensors[2];
    bf16_min_range = powOp.min_range().convertToFloat();
    bf16_max_range = powOp.max_range().convertToFloat();
  }
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
  coeff = powOp.coeff().convertToFloat();
}

void PowOpKernel::invoke() {
  size_t output_size = output_data->size();
  if (datatype == DataType::INT8) {
#pragma omp parallel for schedule(static, omp_schedule(output_size))
    for (size_t i = 0; i < output_size; ++i) {
      output_data->at(i) = y0_table_op->at((unsigned char)input_data->at(i));
    }
  } else if (datatype == DataType::BF16) {
    bf16_lut_mantissa(input_data->data(), output_data->data(),
                      input_data->size(), y0_bf16_table_op->data(),
                      y0_bf16_slope_table->data());
  } else {
#pragma omp parallel for schedule(static, omp_schedule(output_size))
    for (size_t i = 0; i < output_size; ++i) {
      if (coeff != (int)(coeff) && input_data->at(i) < 0) {
        output_data->at(i) = 0.0f;
      } else {
        output_data->at(i) = std::pow(input_data->at(i), coeff);
      }
    }
  }
}

TanHOpKernel::TanHOpKernel(Operation &op, value_map_t &valueMapping,
                           weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  auto tanhOp = cast<tpu::TanHOp>(op);
  if (datatype == DataType::INT8) {
    y0_table_op = this->opdTensors[1];
    slope_table = this->opdTensors[2];
  } else if (datatype == DataType::BF16) {
    y0_bf16_table_op = this->opdTensors[1];
    y0_bf16_slope_table = this->opdTensors[2];
    bf16_min_range = tanhOp.min_range().convertToFloat();
    bf16_max_range = tanhOp.max_range().convertToFloat();
  }
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void TanHOpKernel::invoke() {
  size_t output_size = output_data->size();
  if (datatype == DataType::INT8) {
#pragma omp parallel for schedule(static, omp_schedule(output_size))
    for (size_t i = 0; i < output_size; ++i) {
      output_data->at(i) = y0_table_op->at((unsigned char)input_data->at(i));
    }
  } else if (datatype == DataType::BF16) {
    bf16_lut_slope("tanh", input_data->data(), output_data->data(),
                   output_data->size(), y0_bf16_table_op->data(),
                   y0_bf16_slope_table->data());
  } else {

#pragma omp parallel for schedule(static, omp_schedule(output_size))
    for (size_t i = 0; i < output_size; ++i) {
      output_data->at(i) = std::tanh(input_data->at(i));
    }
  }
}

LogOpKernel::LogOpKernel(Operation &op, value_map_t &valueMapping,
                         weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  if (datatype == DataType::INT8) {
    y0_table_op = this->opdTensors[1];
    slope_table = this->opdTensors[2];
  } else if (datatype == DataType::BF16) {
    y0_bf16_table_op = this->opdTensors[1];
    y0_bf16_mantissa_table = this->opdTensors[2];
  }
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void LogOpKernel::invoke() {
  size_t output_size = output_data->size();
  if (datatype == DataType::INT8) {
#pragma omp parallel for schedule(static, omp_schedule(output_size))
    for (size_t i = 0; i < output_size; ++i) {
      output_data->at(i) = y0_table_op->at((unsigned char)input_data->at(i));
    }
  } else if (datatype == DataType::BF16) {
    bf16_lut_mantissa(input_data->data(), output_data->data(),
                      output_data->size(), y0_bf16_table_op->data(),
                      y0_bf16_mantissa_table->data(), "log");
  } else {

#pragma omp parallel for schedule(static, omp_schedule(output_size))
    for (size_t i = 0; i < output_size; ++i) {
      output_data->at(i) = std::log(input_data->at(i));
    }
  }
}

} // namespace mlir
