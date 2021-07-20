#include "tpuc/Interpreter/cpu/activation.hpp"
#include "tpuc/Interpreter/cpu/lut_func.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"
#include "tpuc/NativeCpuImplementation.h"

// FIXME: this head should not be here,
//  using interpreter own float convert is better
#include "bmkernel/bm1880v2/1880v2_fp_convert.h"
#include "internal.hpp"

#include <cmath>

namespace mlir {

AbsOpKernel::AbsOpKernel(Operation &op, value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
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

ExpOpKernel::ExpOpKernel(Operation &op, value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  auto expOp = cast<tpu::ExpOp>(op);
  if (datatype == DataType::INT8) {
    y0_table_op = this->opdTensors[1];
    slope_table = this->opdTensors[2];
  } else if (datatype == DataType::BF16) {
    y0_bf16_table_op = this->opdTensors[1];
    y0_bf16_slope_table = this->opdTensors[2];
    bf16_min_range = expOp.min_range().convertToFloat();
    bf16_max_range = expOp.max_range().convertToFloat();
  }
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
    bf16_lut_slope("exp", input_data->data(), output_data->data(), output_data->size(),
                   *y0_bf16_table_op, *y0_bf16_slope_table);
  } else {
#pragma omp parallel for schedule(static, omp_schedule(output_size))
    for (size_t i = 0; i < output_size; ++i) {
      output_data->at(i) = std::exp(input_data->at(i));
    }
  }
}

MishOpKernel::MishOpKernel(Operation &op, value_map_t &valueMapping)
  : CPUOpKernel(op, valueMapping) {
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
                   *y0_bf16_table_op, *y0_bf16_slope_table);
  } else {
#pragma omp parallel for schedule(static, omp_schedule(size))
    for (size_t i = 0; i < output_data->size(); ++i) {
      output_data->at(i) = my_mish_activate(input_data->at(i));
    }
  }
}

LeakyReluOpKernel::LeakyReluOpKernel(Operation &op, value_map_t &valueMapping)
  : CPUOpKernel(op, valueMapping) {
  auto leaky_reluOp = cast<tpu::LeakyReluOp>(op);

  this->negative_slope = leaky_reluOp.negative_slope().convertToFloat();
  this->is_asymmetric = isOpQuantAsymmetric(&op);
  if (is_asymmetric) {
    this->input_offset = -getPreviousOpZeroPoint(&op);
    this->output_offset = getOpZeroPoint(&op);
  }
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

ReluOpKernel::ReluOpKernel(Operation &op, value_map_t &valueMapping)
  : CPUOpKernel(op, valueMapping) {
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void ReluOpKernel::invoke() {
  relu(input_data->data(), output_data->data(), output_data->size());
}

PReluOpKernel::PReluOpKernel(Operation &op, value_map_t &valueMapping)
  : CPUOpKernel(op, valueMapping) {
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

ReciprocalOpKernel::ReciprocalOpKernel(Operation &op,
                                       value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {

  auto rOp = cast<tpu::ReciprocalOp>(op);
  if (datatype == DataType::INT8) {
    y0_table_op = this->opdTensors[1];
    slope_table = this->opdTensors[2];
  } else if (datatype == DataType::BF16) {
    y0_bf16_table_op = this->opdTensors[1];
    y0_bf16_slope_table = this->opdTensors[2];
    bf16_min_range = rOp.min_range().convertToFloat();
    bf16_max_range = rOp.max_range().convertToFloat();
  }

  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
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

ReshapeOpKernel::ReshapeOpKernel(Operation &op, value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  // get tensors
  input_data = this->opdTensors[0];
  this->resTensor.reset();
  this->resTensor = this->opdTensors[0];
  output_data = this->resTensor;
  valueMapping[op.getResult(0)] = this->resTensor;
}

void ReshapeOpKernel::invoke() {
}

SigmoidOpKernel::SigmoidOpKernel(Operation &op, value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  auto sigmoidOp = cast<tpu::SigmoidOp>(op);
  if (datatype == DataType::INT8) {
    y0_table_op = this->opdTensors[1];
    slope_table = this->opdTensors[2];
  } else if (datatype == DataType::BF16) {
    y0_bf16_table_op = this->opdTensors[1];
    y0_bf16_slope_table = this->opdTensors[2];
    bf16_min_range = sigmoidOp.min_range().convertToFloat();
    bf16_max_range = sigmoidOp.max_range().convertToFloat();
  }
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
    bf16_lut_slope("sigmoid", input_data->data(), output_data->data(), output_size,
                   *y0_bf16_table_op, *y0_bf16_slope_table);
  } else {
#pragma omp parallel for schedule(static, omp_schedule(output_size))
    for (size_t i = 0; i < output_size; ++i) {
      output_data->at(i) = 0.5 * tanh(0.5 * input_data->at(i)) + 0.5;
    }
  }
}

SwishOpKernel::SwishOpKernel(Operation &op, value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  auto sigmoidOp = cast<tpu::SwishOp>(op);
  if (datatype == DataType::INT8) {
    y0_table_op = this->opdTensors[1];
    slope_table = this->opdTensors[2];
  } else if (datatype == DataType::BF16) {
    y0_bf16_table_op = this->opdTensors[1];
    y0_bf16_slope_table = this->opdTensors[2];
    bf16_min_range = sigmoidOp.min_range().convertToFloat();
    bf16_max_range = sigmoidOp.max_range().convertToFloat();
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
    bf16_lut_slope("swish", input_data->data(), output_data->data(), output_size,
                   *y0_bf16_table_op, *y0_bf16_slope_table);
  } else {
#pragma omp parallel for schedule(static, omp_schedule(output_size))
    for (size_t i = 0; i < output_size; ++i) {
      auto val = input_data->at(i);
      output_data->at(i) = val / (1 + std::exp(-val));
    }
  }
}

SoftPlusOpKernel::SoftPlusOpKernel(Operation &op, value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  auto spOp = cast<tpu::SoftPlusOp>(op);
  if (datatype == DataType::INT8) {
    y0_table_op = this->opdTensors[1];
    slope_table = this->opdTensors[2];
  } else if (datatype == DataType::BF16) {
    y0_bf16_table_op = this->opdTensors[1];
    y0_bf16_slope_table = this->opdTensors[2];
    bf16_min_range = spOp.min_range().convertToFloat();
    bf16_max_range = spOp.max_range().convertToFloat();
  }
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
    bf16_lut_slope("softplus", input_data->data(), output_data->data(), output_data->size(),
                   *y0_bf16_table_op, *y0_bf16_slope_table);
  } else {
#pragma omp parallel for schedule(static, omp_schedule(output_size))
    for (size_t i = 0; i < output_data->size(); ++i) {
      output_data->at(i) = logf(expf(input_data->at(i)) + 1);
    }
  }
}

SqrtOpKernel::SqrtOpKernel(Operation &op, value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  auto sqrtOp = cast<tpu::SqrtOp>(op);
  if (datatype == DataType::INT8) {
    y0_table_op = this->opdTensors[1];
    slope_table = this->opdTensors[2];
  } else if (datatype == DataType::BF16) {
    y0_bf16_table_op = this->opdTensors[1];
    y0_bf16_slope_table = this->opdTensors[2];
    bf16_min_range = sqrtOp.min_range().convertToFloat();
    bf16_max_range = sqrtOp.max_range().convertToFloat();
  }
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
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

SquareOpKernel::SquareOpKernel(Operation &op, value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
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


TanHOpKernel::TanHOpKernel(Operation &op, value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
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
    bf16_lut_slope("tanh", input_data->data(), output_data->data(), output_data->size(),
                   *y0_bf16_table_op, *y0_bf16_slope_table);
  } else {

#pragma omp parallel for schedule(static, omp_schedule(output_size))
    for (size_t i = 0; i < output_size; ++i) {
      output_data->at(i) = std::tanh(input_data->at(i));
    }
  }
}

} // namespace mlir