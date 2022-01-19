#include "tpuc/Interpreter/cpu/reduce.hpp"
#include "internal.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Interpreter/cpu/lut_func.hpp"
#include "tpuc/MlirModuleInterpreter.h"
#include <algorithm>
namespace mlir {

ReduceOpKernel::ReduceOpKernel(Operation &op, value_map_t &valueMapping,
                               weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  auto input_shape = getTensorShape(op.getOperand(0));
  auto axes_v = op.getAttr("axes").cast<ArrayAttr>();
  arrayAttrToVector(axes_v, this->axes);
  if (datatype == DataType::INT8) {
    auto quant_rshift = this->opdTensors[3];
    auto quant_multiplier = this->opdTensors[4];
    this->rshift = quant_rshift != nullptr ? quant_rshift->at(0) : 0;

    this->multiplier =
        quant_multiplier != nullptr ? quant_multiplier->at(0) : 1;
  }
  if (auto castOp = llvm::dyn_cast_or_null<tpu::ReduceL2Op>(&op)) {
    coeff = castOp.coeff().convertToFloat();
  }
  lut = this->opdTensors[1];
  mantissa_lut = this->opdTensors[2];
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
  // calc dims
  int num_dims = input_shape.size();
  int num_axes = axes.size();

  for (int i = 1; i < num_axes; i++) {
    assert(axes[i] == axes[i - 1] + 1);
    assert(axes[i] < num_dims);
  }
  int start_axis = axes[0];
  int end_axis = axes[num_axes - 1] + 1;
  outer_dims =
      std::accumulate(input_shape.begin(), input_shape.begin() + start_axis, 1,
                      std::multiplies<int64_t>());
  axis_dims = std::accumulate(input_shape.begin() + start_axis,
                              input_shape.begin() + end_axis, 1,
                              std::multiplies<int64_t>());
  inner_dims =
      std::accumulate(input_shape.begin() + end_axis, input_shape.end(), 1,
                      std::multiplies<int64_t>());
}

void ReduceOpKernel::invoke() {
  auto input = input_data->data();
  auto output = output_data->data();
  for (int o = 0; o < outer_dims; o++) {
    for (int i = 0; i < inner_dims; i++) {
      switch (type) {
      case REDUCE_MEAN:
      case REDUCE_SUM: {
        float sum = 0.0f;
        if (inner_dims == 1) {
          sum = std::accumulate(input + o * axis_dims,
                                input + (o + 1) * axis_dims, 0.0f);
        } else {
          for (int a = 0; a < axis_dims; a++) {
            sum += input[o * axis_dims * inner_dims + a * inner_dims + i];
          }
        }
        if (type == REDUCE_SUM) {
          output[o * inner_dims + i] = sum;
        } else {
          if (datatype != DataType::INT8) {
            sum = sum / axis_dims;
          }
          output[o * inner_dims + i] = sum;
        }
      } break;
      case REDUCE_MAX:
      case REDUCE_MIN: {
        float target = input[o * axis_dims * inner_dims + i];
        for (int a = 1; a < axis_dims; a++) {
          auto v = input[o * axis_dims * inner_dims + a * inner_dims + i];
          if (type == REDUCE_MAX && v > target) {
            target = v;
          } else if (type == REDUCE_MIN && v < target) {
            target = v;
          }
        }
        output[o * inner_dims + i] = target;
      } break;
      case REDUCE_L2: {
        float sum = 0.0f;
        for (int a = 0; a < axis_dims; a++) {
          sum += std::pow(
              input[o * axis_dims * inner_dims + a * inner_dims + i], 2);
        }
        if (datatype == DataType::BF16) {
          bf16_lut_mantissa(&sum, &sum, 1, lut->data(), mantissa_lut->data());
          output[o * inner_dims + i] = sum;
        } else {
          output[o * inner_dims + i] = std::pow(sum, coeff);
        }
      } break;
      }
    }
  }
  if (datatype == DataType::INT8 && (rshift != 0 || multiplier != 1)) {
    for (size_t i = 0; i < output_data->size(); ++i) {
      output_data->at(i) = (float)applyMultiplierAndRShiftAndSaturateInt8(
          output_data->at(i), (uint32_t)rshift, (uint32_t)multiplier, false);
    }
  } else if (datatype == DataType::BF16) {
    BF16(output_data->data(), output_data->data(), output_data->size());
  }
}

} // namespace mlir
