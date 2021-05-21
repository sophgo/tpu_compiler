#include "tpuc/Interpreter/cpu/reduce.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {
inline int count(std::vector<int64_t> &shape, int start_axis, int end_axis) {
  int64_t count = 1;
  for (int i = start_axis; i < end_axis; ++i) {
    count *= shape[i];
  }
  return count;
}

void reduce_mean(float *input, float *output,
                 std::vector<int64_t> &org_input_shape,
                 std::vector<int> &axes) {
  assert(axes.size() > 0);
  auto input_shape = org_input_shape;
  int size = count(input_shape, 0, input_shape.size());
  std::vector<float> tmp(size, 0);
  float *_output = tmp.data();

  for (int i = 0; i < (int)axes.size(); i++) {
    int dim = input_shape.size();
    int axis = axes[i];
    assert(dim > axis);

    int inner = count(input_shape, axis + 1, input_shape.size());
    int next_inner = inner * input_shape[axis];
    int outer = count(input_shape, 0, axis);

    for (int i = 0; i < outer; i++) {
      std::vector<float> inner_sum(inner, 0);
      for (int s = 0; s < input_shape[axis]; s++) {
        for (int j = 0; j < inner; j++) {
          inner_sum[j] += input[i * next_inner + s * inner + j];
        }
      }

      // mean
      for (int j = 0; j < inner; j++) {
        _output[i * inner + j] = inner_sum[j] / input_shape[axis];
      }
    }

    input_shape[axis] = 1;
    input = _output;
  }

  // export
  size = count(input_shape, 0, input_shape.size());
  std::copy(_output, _output + size, output);
}

void reduce_mean_int8(float *input, float *output,
                      std::vector<int64_t> &org_input_shape,
                      std::vector<int> &axes, int avg_const, int rshift) {
  assert(axes.size() > 0);
  auto input_shape = org_input_shape;
  int size = count(input_shape, 0, input_shape.size());
  std::vector<int> tmp (size, 0);
  int* _output = tmp.data();
  std::vector<int> tmp2 (size, 0);
  int* _input = tmp2.data();

  // Convert integer format
  for (int i = 0; i < size; i++)
    _input[i] = (int)(input[i] * avg_const);

  for (int i = 0; i < (int)axes.size(); i++) {
    int dim = input_shape.size();
    int axis = axes[i];
    assert(dim > axis);

    int inner = count(input_shape, axis + 1, input_shape.size());
    int next_inner = inner * input_shape[axis];
    int outer = count(input_shape, 0, axis);

    llvm::errs() << "  [" << i << "] inner " << inner
                 << ", outer " << outer
                 << ", axis shape:" << input_shape[axis]
                 << ", axis " << axis << "\n";

    for (int i = 0; i < outer; i++) {
      std::vector<int> inner_sum (inner, 0);
      for (int s = 0; s < input_shape[axis]; s++) {
        for (int j = 0; j < inner; j++) {
          inner_sum[j] += _input[i * next_inner + s * inner + j];
        }
      }

      for (int j = 0; j < inner; j++) {
        _output[i * inner + j] = inner_sum[j];
      }
    }

    input_shape[axis] = 1;
    _input = _output;
  }

  // quant down and
  // Store float format
  size = count(input_shape, 0, input_shape.size());
  for (int i = 0; i < size; i++) {
    int val = _output[i];
    val >>= rshift - 1;
    val += 1; // round up
    val >>= 1;
    val = std::max(val, -128);
    val = std::min(val, 127);
    output[i] = (float)val;
  }

}

void reduce_max(float *input, float *output, std::vector<int64_t> &input_shape,
                std::vector<int> &axes) {
  assert(axes.size() > 0);
  int axis = axes[0];
  // only support one axis, if has two axis, should be continous
  int total = count(input_shape, 0, input_shape.size());
  int n = count(input_shape, 0, axis);
  int c = input_shape[axis];
  int hw = total / (n * c);

  for (int nidx = 0; nidx < n; nidx++) {
    for (int inner_idx = 0; inner_idx < hw; inner_idx++) {
      for (int cidx = 0; cidx < c; cidx++) {
        float tmp = input[nidx * c * hw + cidx * hw + inner_idx];
        if (cidx == 0)
          output[nidx * hw + inner_idx] = tmp;
        output[nidx * hw + inner_idx] =
            std::max(tmp, output[nidx * hw + inner_idx]);
      }
    }
  }
}

//   reduced = np.sqrt(np.sum(
//               a=np.square(data), axis=axes, keepdims=keepdims == 1))
void reduce_l2(float *input, float *output,
                 std::vector<int64_t> &org_input_shape,
                 std::vector<int> &axes) {
  assert(axes.size() > 0);
  auto input_shape = org_input_shape;
  int size = count(input_shape, 0, input_shape.size());
  std::vector<float> tmp(size, 0);
  float *_output = tmp.data();

  for (int i = 0; i < (int)axes.size(); i++) {
    int dim = input_shape.size();
    int axis = axes[i];
    assert(dim > axis);

    // NOTICE: only verify dim [1, xx]
    int inner = count(input_shape, axis + 1, input_shape.size());
    int next_inner = inner * input_shape[axis];
    int outer = count(input_shape, 0, axis);

    for (int i = 0; i < outer; i++) {
      std::vector<float> inner_sum(inner, 0);
      for (int s = 0; s < input_shape[axis]; s++) {
        for (int j = 0; j < inner; j++) {
          inner_sum[j] += std::pow(input[i * next_inner + s * inner + j], 2);
        }
      }

      // l2
      for (int j = 0; j < inner; j++) {
        _output[i * inner + j] = std::sqrt(inner_sum[j]);
      }
    }

    input_shape[axis] = 1;
    input = _output;
  }

  // export
  size = count(input_shape, 0, input_shape.size());
  std::copy(_output, _output + size, output);
}

ReduceL2OpKernel::ReduceL2OpKernel(Operation &op, value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  auto reducemaxOp = cast<tpu::ReduceL2Op>(op);
  auto input_type = reducemaxOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();
  arrayAttrToVector(reducemaxOp.axes().getValue(), this->axes);
  if (datatype == DataType::INT8) {
    auto quant_rshift = this->opdTensors[3];
    auto quant_multiplier = this->opdTensors[4];
    assert(quant_rshift);
    assert(quant_multiplier);
    this->rshift = quant_rshift->at(0);
    this->multiplier = quant_multiplier->at(0);
  }
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void ReduceL2OpKernel::invoke() {
  reduce_l2(input_data->data(), output_data->data(), input_shape, axes);
  if (datatype == DataType::INT8) {
    for (size_t i = 0; i < output_data->size(); ++i) {
      output_data->at(i) = (float)applyMultiplierAndRShiftAndSaturateInt8(
          output_data->at(i), (uint32_t)rshift, (uint32_t)multiplier, false);
    }
  } else if (datatype == DataType::BF16) {
    clean16bitmantissa(output_data->data(), output_data->data(),
                       output_data->size());
  }
}

ReduceMaxOpKernel::ReduceMaxOpKernel(Operation &op, value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  auto reducemaxOp = cast<tpu::ReduceMaxOp>(op);
  auto input_type = reducemaxOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();
  arrayAttrToVector(reducemaxOp.axes().getValue(), this->axes);

  if (datatype == DataType::INT8) {
    auto quant_rshift = this->opdTensors[3];
    auto quant_multiplier = this->opdTensors[4];
    assert(quant_rshift);
    assert(quant_multiplier);
    this->rshift = quant_rshift->at(0);
    this->multiplier = quant_multiplier->at(0);
  }
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void ReduceMaxOpKernel::invoke() {
  reduce_max(input_data->data(), output_data->data(), input_shape, axes);
  if (datatype == DataType::INT8) {
    for (size_t i = 0; i < output_data->size(); ++i) {
      output_data->at(i) = (float)applyMultiplierAndRShiftAndSaturateInt8(
          output_data->at(i), (uint32_t)rshift, (uint32_t)multiplier, false);
    }
  } else if (datatype == DataType::BF16) {
    clean16bitmantissa(output_data->data(), output_data->data(),
                       output_data->size());
  }
}

ReduceMeanOpKernel::ReduceMeanOpKernel(Operation &op,
                                       value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  auto reducemeanOp = cast<tpu::ReduceMeanOp>(op);
  auto input_type = reducemeanOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();
  arrayAttrToVector(reducemeanOp.axes().getValue(), this->axes);

  if (datatype == DataType::INT8) {
    auto quant_rshift = this->opdTensors[3];
    auto quant_multiplier = this->opdTensors[4];
    assert(quant_rshift);
    assert(quant_multiplier);
    this->rshift = quant_rshift->at(0);
    this->multiplier = quant_multiplier->at(0);
  }
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void ReduceMeanOpKernel::invoke() {

  if (datatype == DataType::FP32) {
    reduce_mean(input_data->data(), output_data->data(), input_shape, axes);
  } else if (datatype == DataType::INT8) {
    reduce_mean_int8(input_data->data(), output_data->data(), input_shape, axes,
                     (int)multiplier, (int)rshift);
  } else if (datatype == DataType::BF16) {
    reduce_mean(input_data->data(), output_data->data(), input_shape, axes);
    clean16bitmantissa(output_data->data(), output_data->data(),
                       output_data->size());
  }
}

} // namespace mlir
