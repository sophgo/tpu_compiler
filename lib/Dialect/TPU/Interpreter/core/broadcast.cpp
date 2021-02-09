#include "tpuc/Interpreter/cpu/broadcast.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Interpreter/cpu/activation.hpp"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {

BroadcastAddOpKernel::BroadcastAddOpKernel(Operation &op,
                                           value_map_t &valueMapping) {
  auto broadcastaddOp = cast<tpu::BroadcastAddOp>(op);
  assert(broadcastaddOp);
  LLVM_DEBUG(llvm::outs() << " BroadcastAdd op: [" << broadcastaddOp.name()
                          << "]\n";);

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = broadcastaddOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  LLVM_DEBUG(llvm::outs() << "    =>required memory size: [" << size << "]\n";);

  this->shape = getTensorShape(result);

  this->inputs_shape.push_back(getTensorShape(op.getOperand(0)));
  this->inputs_shape.push_back(getTensorShape(op.getOperand(1)));

  do_relu = broadcastaddOp.do_relu();
  int axis = broadcastaddOp.axis();
  if (axis != 1) {
    llvm_unreachable("only support axis 1 broadcast mul");
  }

  this->name = broadcastaddOp.name().str();
  this->op_type = op.getName().getStringRef().str();

  set_datatype(getOpQuant(&op).str());
  if (datatype == DataType::INT8) {
    auto quant_rshift = opTensors[4];
    auto quant_multiplier = opTensors[5];
    assert(quant_rshift && quant_multiplier);
    rshift = quant_rshift->at(0);
    multiplier.assign(quant_multiplier->begin(), quant_multiplier->end());
  }

  // get tensors
  inputs_data = opTensors;
  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
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
    clean16bitmantissa(output_data->data(), output_data->data(),
                       output_data->size());
  }
}
void BroadcastAddOpKernel::set_tensor(const std::vector<float> &data) {
  llvm_unreachable("TODO!");
};

std::vector<float> BroadcastAddOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void BroadcastAddOpKernel::dump() { OpKernel::dump(); }

BroadcastMulOpKernel::BroadcastMulOpKernel(Operation &op,
                                           value_map_t &valueMapping) {
  auto broadcastmulOp = cast<tpu::BroadcastMulOp>(op);
  assert(broadcastmulOp);
  LLVM_DEBUG(llvm::outs() << " BroadcastMul op: [" << broadcastmulOp.name()
                          << "]\n";);

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = broadcastmulOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  LLVM_DEBUG(llvm::outs() << "    =>required memory size: [" << size << "]\n";);
  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();
  this->scale_shape = getTensorShape(op.getOperand(1));

  do_relu = broadcastmulOp.do_relu();
  int axis = broadcastmulOp.axis();
  if (axis != 1) {
    llvm_unreachable("only support axis 1 broadcast mul");
  }

  this->name = broadcastmulOp.name().str();
  this->op_type = op.getName().getStringRef().str();

  set_datatype(getOpQuant(&op).str());
  if (datatype == DataType::INT8) {
    auto quant_rshift = opTensors[4];
    auto quant_multiplier = opTensors[5];
    assert(quant_rshift && quant_multiplier);
    rshift = quant_rshift->at(0);
    mutliplier = quant_multiplier->at(0);
  }
  // get tensors
  input_data = opTensors[0];
  scale = opTensors[1];
  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
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
          output_data->at(i), (uint32_t)rshift, (uint32_t)mutliplier, true);
    }
  } else if (datatype == DataType::BF16) {
    clean16bitmantissa(output_data->data(), output_data->data(),
                       output_data->size());
  }
}
void BroadcastMulOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " BroadcastMul op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};
std::vector<float> BroadcastMulOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}
void BroadcastMulOpKernel::dump() { OpKernel::dump(); }

BroadcastSubOpKernel::BroadcastSubOpKernel(Operation &op,
                                           value_map_t &valueMapping) {
  auto broadcastsubOp = cast<tpu::BroadcastSubOp>(op);
  assert(broadcastsubOp);
  LLVM_DEBUG(llvm::outs() << " BroadcastSub op: [" << broadcastsubOp.name()
                          << "]\n";);

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = broadcastsubOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  LLVM_DEBUG(llvm::outs() << "    =>required memory size: [" << size << "]\n";);

  this->shape = getTensorShape(result);

  this->inputs_shape.push_back(getTensorShape(op.getOperand(0)));
  this->inputs_shape.push_back(getTensorShape(op.getOperand(1)));

  do_relu = broadcastsubOp.do_relu();
  int axis = broadcastsubOp.axis();
  if (axis != 1) {
    llvm_unreachable("only support axis 1 broadcast mul");
  }

  this->name = broadcastsubOp.name().str();
  this->op_type = op.getName().getStringRef().str();

  set_datatype(getOpQuant(&op).str());
  if (datatype == DataType::INT8) {
    auto quant_rshift = opTensors[4];
    auto quant_multiplier = opTensors[5];
    assert(quant_rshift && quant_multiplier);
    rshift = quant_rshift->at(0);
    multiplier.assign(quant_multiplier->begin(), quant_multiplier->end());
  }

  // get tensors
  inputs_data = opTensors;
  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
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
    clean16bitmantissa(output_data->data(), output_data->data(),
                       output_data->size());
  }
}
void BroadcastSubOpKernel::set_tensor(const std::vector<float> &data) {
  llvm_unreachable("TODO!");
};

std::vector<float> BroadcastSubOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void BroadcastSubOpKernel::dump() { OpKernel::dump(); }

} // namespace mlir