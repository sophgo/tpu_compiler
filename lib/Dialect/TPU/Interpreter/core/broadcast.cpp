#include "tpuc/Interpreter/cpu/broadcast.hpp"
#include "internal.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/MlirModuleInterpreter.h"

namespace mlir {

BroadcastAddOpKernel::BroadcastAddOpKernel(Operation &op,
                                           value_map_t &valueMapping,
                                           weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  auto broadcastaddOp = cast<tpu::BroadcastAddOp>(op);

  auto shape1 = getTensorShape(op.getOperand(0));
  auto shape2 = getTensorShape(op.getOperand(1));
  if (shape2.size() == 4 && shape2[1] == 1 && shape2[2] == 1) {
    shape1[1] *= shape1[2];
    shape1[2] = 1;
  }
  this->inputs_shape.push_back(shape1);
  this->inputs_shape.push_back(shape2);

  do_relu = broadcastaddOp.do_relu();
  int axis = broadcastaddOp.axis();
  if (axis != 1) {
    llvm_unreachable("only support axis 1 broadcast mul");
  }
  if (datatype == DataType::INT8) {
    auto quant_rshift = this->opdTensors[4];
    auto quant_multiplier = this->opdTensors[5];
    assert(quant_rshift && quant_multiplier);
    rshift = quant_rshift->at(0);
    multiplier.assign(quant_multiplier->begin(), quant_multiplier->end());
  }
  // get tensors
  inputs_data = this->opdTensors;
  output_data = this->resTensor;
}

void BroadcastAddOpKernel::i8_invoke() {
  int n0 = inputs_shape[0][0];
  int c0 = inputs_shape[0][1];
  int h0 = inputs_shape[0].size() > 2 ? inputs_shape[0][2] : 1;
  int w0 = inputs_shape[0].size() > 3 ? inputs_shape[0][3] : 1;
  int n1 = inputs_shape[1][0];
  int c1 = inputs_shape[1][1];
  int h1 = inputs_shape[1].size() > 2 ? inputs_shape[0][2] : 1;
  int w1 = inputs_shape[1].size() > 3 ? inputs_shape[0][3] : 1;
  size_t input_number = 2;
  std::vector<std::vector<float>> i8_inputs(input_number);
  for (size_t i = 0; i < input_number; ++i) {
    i8_inputs[i].assign(inputs_data[i]->begin(), inputs_data[i]->end());
  }

  for (size_t i = 0; i < input_number; ++i) {
    for (size_t j = 0; j < i8_inputs[i].size(); ++j) {
      i8_inputs[i][j] *= (int8_t)multiplier.at(i);
    }
  }

  size_t idx = 0;
  for (int n = 0; n < (int)n0; ++n) {
    for (int c = 0; c < (int)c0; ++c) {
      for (int h = 0; h < (int)h0; ++h) {
        for (int w = 0; w < (int)w0; ++w) {
          float a = i8_inputs[0][w + h * w0 + c * h0 * w0 + n * c0 * h0 * w0];
          float b = i8_inputs[1][(w1 == 1 ? 0 : w) + (h1 == 1 ? 0 : h) * w1 +
                                 (c1 == 1 ? 0 : c) * h1 * w1 +
                                 (n1 == 1 ? 0 : n) * c1 * h1 * w1];
          output_data->at(idx++) = a + b;
        }
      }
    }
  }
  if (do_relu) {
    relu(output_data->data(), output_data->data(), output_data->size());
  }

  for (size_t i = 0; i < output_data->size(); ++i) {
    output_data->at(i) =
        (float)applyRShiftAndSaturateInt8(output_data->at(i), (uint32_t)rshift);
  }
}

void BroadcastAddOpKernel::fp32_invoke() {
  int n0 = inputs_shape[0][0];
  int c0 = inputs_shape[0][1];
  int h0 = inputs_shape[0].size() > 2 ? inputs_shape[0][2] : 1;
  int w0 = inputs_shape[0].size() > 3 ? inputs_shape[0][3] : 1;
  int n1 = inputs_shape[1][0];
  int c1 = inputs_shape[1][1];
  int h1 = inputs_shape[1].size() > 2 ? inputs_shape[1][2] : 1;
  int w1 = inputs_shape[1].size() > 3 ? inputs_shape[1][3] : 1;

  size_t idx = 0;
  for (int n = 0; n < (int)n0; ++n) {
    for (int c = 0; c < (int)c0; ++c) {
      for (int h = 0; h < (int)h0; ++h) {
        for (int w = 0; w < (int)w0; ++w) {
          float a =
              inputs_data[0]->at(w + h * w0 + c * h0 * w0 + n * c0 * h0 * w0);
          int index = (w1 == 1 ? 0 : w) + (h1 == 1 ? 0 : h) * w1 +
                      (c1 == 1 ? 0 : c) * h1 * w1 +
                      (n1 == 1 ? 0 : n) * c1 * h1 * w1;
          float b = inputs_data[1]->at(index);
          output_data->at(idx++) = a + b;
        }
      }
    }
  }
}

void BroadcastAddOpKernel::invoke() {
  if (datatype == DataType::FP32) {
    fp32_invoke();
  } else if (datatype == DataType::INT8) {
    i8_invoke();
  } else {
    fp32_invoke();
    BF16(output_data->data(), output_data->data(), output_data->size());
  }
}

BroadcastMulOpKernel::BroadcastMulOpKernel(Operation &op,
                                           value_map_t &valueMapping,
                                           weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  auto broadcastmulOp = cast<tpu::BroadcastMulOp>(op);
  do_relu = broadcastmulOp.do_relu();
  int axis = broadcastmulOp.axis();
  if (axis != 1) {
    llvm_unreachable("only support axis 1 broadcast mul");
  }

  this->scale_shape = getTensorShape(op.getOperand(1));
  if (datatype == DataType::INT8) {
    auto quant_rshift = this->opdTensors[4];
    auto quant_multiplier = this->opdTensors[5];
    assert(quant_rshift && quant_multiplier);
    rshift = quant_rshift->at(0);
    multiplier = quant_multiplier->at(0);
  }
  // get tensors
  input_data = this->opdTensors[0];
  scale = this->opdTensors[1];
  output_data = this->resTensor;
}

void BroadcastMulOpKernel::invoke() {
  int64_t n, c, h, w, bn, bc, bh, bw;
  getNCHW(shape, n, c, h, w);
  getNCHW(scale_shape, bn, bc, bh, bw);
  assert(bn == n || bn == 1);
  if (bh == 1 && bw == 1) {
    if (bn == n) {
      c = n * c;
      n = 1;
    }
    for (int ni = 0; ni < n; ++ni) {
      for (int ci = 0; ci < c; ++ci) {
        for (int i = 0; i < h * w; ++i) {
          auto x = input_data->at(ni * c * h * w + ci * h * w + i);
          auto y = x * scale->at(ci);
          output_data->at(ni * c * h * w + ci * h * w + i) = y;
        }
      }
    }
  } else if (bh == h && bw == w && bc == 1) {
    // bcast mul
    if (bn == 1) {
      c = n * c;
      n = 1;
    }
    int hw = h * w;
    for (int k = 0; k < n; k++) {
      for (int j = 0; j < c; j++) {
        for (int i = 0; i < hw; i++) {
          int offset = (k * c + j) * hw + i;
          output_data->at(offset) =
              input_data->at(offset) * scale->at(k * hw + i);
        }
      }
    }
  } else {
    llvm_unreachable("not support now");
  }
  if (do_relu) {
    relu(output_data->data(), output_data->data(), output_data->size());
  }
  if (datatype == DataType::INT8) {
    for (size_t i = 0; i < output_data->size(); ++i) {
      output_data->at(i) = (float)applyMultiplierAndRShiftAndSaturateInt8(
          output_data->at(i), (uint32_t)rshift, (uint32_t)multiplier, true);
    }
  } else if (datatype == DataType::BF16) {
    BF16(output_data->data(), output_data->data(), output_data->size());
  }
}

BroadcastSubOpKernel::BroadcastSubOpKernel(Operation &op,
                                           value_map_t &valueMapping,
                                           weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  auto broadcastsubOp = cast<tpu::BroadcastSubOp>(op);
  do_relu = broadcastsubOp.do_relu();
  int axis = broadcastsubOp.axis();
  if (axis != 1) {
    llvm_unreachable("only support axis 1 broadcast mul");
  }

  if (datatype == DataType::INT8) {
    auto quant_rshift = this->opdTensors[4];
    auto quant_multiplier = this->opdTensors[5];
    assert(quant_rshift && quant_multiplier);
    rshift = quant_rshift->at(0);
    multiplier.assign(quant_multiplier->begin(), quant_multiplier->end());
  }

  this->inputs_shape.push_back(getTensorShape(op.getOperand(0)));
  this->inputs_shape.push_back(getTensorShape(op.getOperand(1)));
  // get tensors
  inputs_data = this->opdTensors;
  output_data = this->resTensor;
}

void BroadcastSubOpKernel::i8_invoke() {
  int n0 = inputs_shape[0][0];
  int c0 = inputs_shape[0][1];
  int h0 = inputs_shape[0].size() > 2 ? inputs_shape[0][2] : 1;
  int w0 = inputs_shape[0].size() > 3 ? inputs_shape[0][3] : 1;
  int n1 = inputs_shape[1][0];
  int c1 = inputs_shape[1][1];
  int h1 = inputs_shape[1].size() > 2 ? inputs_shape[0][2] : 1;
  int w1 = inputs_shape[1].size() > 3 ? inputs_shape[0][3] : 1;
  size_t input_number = 2;
  std::vector<std::vector<float>> i8_inputs(input_number);
  for (size_t i = 0; i < input_number; ++i) {
    i8_inputs[i].assign(inputs_data[i]->begin(), inputs_data[i]->end());
  }

  for (size_t i = 0; i < input_number; ++i) {
    for (size_t j = 0; j < i8_inputs[i].size(); ++j) {
      i8_inputs[i][j] *= (int8_t)multiplier.at(i);
    }
  }

  size_t idx = 0;
  for (int n = 0; n < (int)n0; ++n) {
    for (int c = 0; c < (int)c0; ++c) {
      for (int h = 0; h < (int)h0; ++h) {
        for (int w = 0; w < (int)w0; ++w) {
          float a = i8_inputs[0][w + h * w0 + c * h0 * w0 + n * c0 * h0 * w0];
          float b = i8_inputs[1][(w1 == 1 ? 0 : w) + (h1 == 1 ? 0 : h) * w1 +
                                 (c1 == 1 ? 0 : c) * h1 * w1 +
                                 (n1 == 1 ? 0 : n) * c1 * h1 * w1];
          output_data->at(idx++) = a - b;
        }
      }
    }
  }
  if (do_relu) {
    relu(output_data->data(), output_data->data(), output_data->size());
  }

  for (size_t i = 0; i < output_data->size(); ++i) {
    output_data->at(i) =
        (float)applyRShiftAndSaturateInt8(output_data->at(i), (uint32_t)rshift);
  }
}

void BroadcastSubOpKernel::fp32_invoke() {
  int n0 = inputs_shape[0][0];
  int c0 = inputs_shape[0][1];
  int h0 = inputs_shape[0].size() > 2 ? inputs_shape[0][2] : 1;
  int w0 = inputs_shape[0].size() > 3 ? inputs_shape[0][3] : 1;
  int n1 = inputs_shape[1][0];
  int c1 = inputs_shape[1][1];
  int h1 = inputs_shape[1].size() > 2 ? inputs_shape[0][2] : 1;
  int w1 = inputs_shape[1].size() > 3 ? inputs_shape[0][3] : 1;

  size_t idx = 0;
  for (int n = 0; n < (int)n0; ++n) {
    for (int c = 0; c < (int)c0; ++c) {
      for (int h = 0; h < (int)h0; ++h) {
        for (int w = 0; w < (int)w0; ++w) {
          float a =
              inputs_data[0]->at(w + h * w0 + c * h0 * w0 + n * c0 * h0 * w0);
          float b = inputs_data[1]->at(
              (w1 == 1 ? 0 : w) + (h1 == 1 ? 0 : h) * w1 +
              (c1 == 1 ? 0 : c) * h1 * w1 + (n1 == 1 ? 0 : n) * c1 * h1 * w1);
          output_data->at(idx++) = a - b;
        }
      }
    }
  }
}

void BroadcastSubOpKernel::invoke() {
  if (datatype == DataType::FP32) {
    fp32_invoke();
  } else if (datatype == DataType::INT8) {
    i8_invoke();
  } else {
    fp32_invoke();
    BF16(output_data->data(), output_data->data(), output_data->size());
  }
}

} // namespace mlir