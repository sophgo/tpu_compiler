#include "tpuc/Interpreter/cpu/gru.hpp"
#include "bmkernel/bm1880v2/1880v2_fp_convert.h"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"
#include "tpuc/NativeCpuImplementation.h"
#include "tpuc/Interpreter/cpu/activation.hpp"

namespace mlir {
double GruOpKernel::sigmoid_(double data) {
  if (datatype == DataType::BF16) {
    float var = data;
    bf16_lut_slope(&var, &var, 1, sigmoid_lut, sigmoid_slope_lut, -8, 8);
    return var;
  } else {
    return 0.5 * tanh(0.5 * data) + 0.5;
  }
}
double GruOpKernel::tanh_(double data) {
  if (datatype == DataType::BF16) {
    float var = data;
    bf16_lut_slope(&var, &var, 1, tanh_lut, tanh_slope_lut, -8, 8);
    return var;
  } else {
    return tanh(data);
  }
}

GruOpKernel::GruOpKernel(Operation &op, value_map_t &valueMapping) {
  auto gruOp = cast<tpu::GruOp>(op);
  assert(gruOp);
  LLVM_DEBUG(llvm::outs() << " GruOp op: [" << gruOp.name() << "]\n";);

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = gruOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  LLVM_DEBUG(llvm::outs() << "    =>required memory size: [" << size << "]\n";);
  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

  auto input_type = gruOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();
  this->name = gruOp.name().str();
  this->op_type = op.getName().getStringRef().str();
  set_datatype(getOpQuant(&op).str());
  if (shape.size() == 4) {
    seq_length = shape[0];
    num_dir = shape[1];
    batch_size = shape[2];
    hidden_size = shape[3];
    assert(input_shape[0] == seq_length);
    only_last = false;
  } else {
    seq_length = input_shape[0];
    num_dir = shape[0];
    batch_size = shape[1];
    hidden_size = shape[2];
    only_last = true;
  }
  assert(input_shape.size() == 3);
  assert(input_shape[1] == batch_size);
  input_size = input_shape[2];
  linear_before_reset = gruOp.linear_before_reset();
  assert(linear_before_reset == true);
  bidirectional = gruOp.bidirectional();
  // get tensors
  input_data = opTensors[0];
  recurrence = opTensors[1];
  bias = opTensors[2];
  initial_h = opTensors[3];
  if (initial_h == nullptr) {
    initial_h = std::make_shared<std::vector<float>>(
        num_dir * batch_size * hidden_size, 0.0f);
  }
  if (datatype == DataType::BF16) {
    sigmoid_lut.assign(opTensors[4]->begin(), opTensors[4]->end());
    sigmoid_slope_lut.assign(opTensors[5]->begin(), opTensors[5]->end());
    tanh_lut.assign(opTensors[6]->begin(), opTensors[6]->end());
    tanh_slope_lut.assign(opTensors[7]->begin(), opTensors[7]->end());
  }
  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}

void GruOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " GruOp op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> GruOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void GruOpKernel::update_addr(bool forward) {
  if (forward) {
    r_z = recurrence->data();
    r_bz = bias->data();
    output = output_data->data();
    prev_hidden_state = initial_h->data();
    input = input_data->data();
  } else {
    r_z = recurrence->data() + 3 * hidden_size * hidden_size;
    r_bz = bias->data() + 3 * hidden_size;
    output = output_data->data() + batch_size * hidden_size;
    prev_hidden_state = initial_h->data() + batch_size * hidden_size;
    input = input_data->data() + 3 * hidden_size;
  }
  r_r = r_z + hidden_size * hidden_size;
  r_h = r_r + hidden_size * hidden_size;
  r_br = r_bz + hidden_size;
  r_bh = r_br + hidden_size;
}

void GruOpKernel::compute(bool forward) {
  update_addr(forward);
  std::vector<float> update_gate(batch_size * hidden_size); // zt
  std::vector<float> reset_gate(batch_size * hidden_size);  // rt
  std::vector<float> hidden_gate(batch_size * hidden_size); // ht

  for (int t = 0; t < seq_length; ++t) {
    int seq_idx = forward ? t : (seq_length - t - 1);
    // zt = sigmoid(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
    // rt = sigmoid(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
    // ht = tanh(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh)
    // H = (1-zt) * ht + zt * Ht
    float *xt = input + seq_idx * batch_size * input_size;
    mkldnn_ip(prev_hidden_state, r_z, r_bz, update_gate.data(), batch_size,
              hidden_size, hidden_size, false);
    mkldnn_ip(prev_hidden_state, r_r, r_br, reset_gate.data(), batch_size,
              hidden_size, hidden_size, false);
    mkldnn_ip(prev_hidden_state, r_h, r_bh, hidden_gate.data(), batch_size,
              hidden_size, hidden_size, false);
    for (int batch = 0; batch < batch_size; batch++) {
      float *xz = xt + batch * input_size;
      float *xr = xz + hidden_size;
      float *xh = xr + hidden_size;
      float *ug = update_gate.data() + batch * hidden_size;
      float *rg = reset_gate.data() + batch * hidden_size;
      float *hg = hidden_gate.data() + batch * hidden_size;
      float *hidden_state = nullptr;
      float *pre_state = prev_hidden_state + batch * hidden_size;
      if (only_last) {
        hidden_state = output + batch * hidden_size;
      } else {
        hidden_state = output + (seq_idx * num_dir * batch_size + batch) * hidden_size;
      }
#pragma omp parallel for schedule(static, omp_schedule(hidden_size))
      for (int i = 0; i < hidden_size; ++i) {
        ug[i] = sigmoid_(ug[i] + xz[i]);
        rg[i] = sigmoid_(rg[i] + xr[i]);
        hg[i] = tanh_(rg[i] * hg[i] + xh[i]);
        hidden_state[i] = (1 - ug[i]) * hg[i] + ug[i] * pre_state[i];
      }
    }
    if (only_last) {
      prev_hidden_state = output;
    } else {
      prev_hidden_state = output + seq_idx * num_dir * batch_size * hidden_size;
    }
  }
}

void GruOpKernel::invoke() {
  compute();
  if (bidirectional) {
    compute(false);
  }
  if (datatype == DataType::BF16) {
    clean16bitmantissa(output_data->data(), output_data->data(),
                       output_data->size());
  }
}

void GruOpKernel::dump() { OpKernel::dump(); }
} // namespace mlir