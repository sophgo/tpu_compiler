#include "tpuc/Interpreter/cpu/lstm.hpp"
#include "bmkernel/bm1880v2/1880v2_fp_convert.h"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Interpreter/cpu/activation.hpp"
#include "tpuc/ModuleInterpreter.h"
#include "tpuc/NativeCpuImplementation.h"

namespace mlir {
double LstmOpKernel::sigmoid_(double data) {
  if (datatype == DataType::BF16) {
    float var = data;
    bf16_lut_slope(&var, &var, 1, *sigmoid_lut, *sigmoid_slope_lut, -8, 8);
    return var;
  } else {
    return 0.5 * tanh(0.5 * data) + 0.5;
  }
}
double LstmOpKernel::tanh_(double data) {
  if (datatype == DataType::BF16) {
    float var = data;
    bf16_lut_slope(&var, &var, 1, *tanh_lut, *tanh_slope_lut, -8, 8);
    return var;
  } else {
    return tanh(data);
  }
}

LstmOpKernel::LstmOpKernel(Operation &op, value_map_t &valueMapping) {
  auto lstmOp = cast<tpu::LstmOp>(op);
  assert(lstmOp);
  LLVM_DEBUG(llvm::outs() << " LstmOp op: [" << lstmOp.name() << "]\n";);

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = lstmOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  LLVM_DEBUG(llvm::outs() << "    =>required memory size: [" << size << "]\n";);
  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

  auto input_type = lstmOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();
  this->name = lstmOp.name().str();
  this->op_type = op.getName().getStringRef().str();
  set_datatype(getOpQuant(&op).str());
  assert(shape.size() == 4);
  seq_length = shape[0];
  num_dir = shape[1];
  batch_size = shape[2];
  hidden_size = shape[3];
  assert(input_shape[0] == seq_length);

  assert(input_shape.size() == 3);
  assert(input_shape[1] == batch_size);
  input_size = input_shape[2];
  bidirectional = lstmOp.bidirectional();
  // get tensors
  input_data = opTensors[0];
  recurrence = opTensors[1];
  bias = opTensors[2];
  if (bias == nullptr) {
    bias =
        std::make_shared<std::vector<float>>(num_dir * 4 * hidden_size, 0.0f);
  }
  initial_h = opTensors[3];
  if (initial_h == nullptr) {
    initial_h = std::make_shared<std::vector<float>>(
        num_dir * batch_size * hidden_size, 0.0f);
  }
  initial_c = opTensors[4];
  if (initial_c == nullptr) {
    initial_c = std::make_shared<std::vector<float>>(
        num_dir * batch_size * hidden_size, 0.0f);
  }
  sigmoid_lut = opTensors[5];
  sigmoid_slope_lut = opTensors[6];
  tanh_lut = opTensors[7];
  tanh_slope_lut = opTensors[8];
  output_data = resultTensor;
  // record mapping table for nexi op connecting
  valueMapping[result] = std::move(resultTensor);
}

void LstmOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " LstmOp op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> LstmOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void LstmOpKernel::update_addr(bool forward) {
  if (forward) {
    r_i = recurrence->data();
    r_bi = bias->data();
    output = output_data->data();
    pre_state_h = initial_h->data();
    pre_state_c = initial_c->data();
    input = input_data->data();
  } else {
    r_i = recurrence->data() + 4 * hidden_size * hidden_size;
    r_bi = bias->data() + 4 * hidden_size;
    output = output_data->data() + batch_size * hidden_size;
    pre_state_h = initial_h->data() + batch_size * hidden_size;
    pre_state_c = initial_c->data() + batch_size * hidden_size;
    input = input_data->data() + 4 * hidden_size;
  }
  r_o = r_i + hidden_size * hidden_size;
  r_f = r_o + hidden_size * hidden_size;
  r_c = r_f + hidden_size * hidden_size;
  r_bo = r_bi + hidden_size;
  r_bf = r_bo + hidden_size;
  r_bc = r_bf + hidden_size;
}

void LstmOpKernel::compute(bool forward) {
  update_addr(forward);
  std::vector<float> gate_i(batch_size * hidden_size);
  std::vector<float> gate_o(batch_size * hidden_size);
  std::vector<float> gate_f(batch_size * hidden_size);
  std::vector<float> gate_c(batch_size * hidden_size);

  for (int t = 0; t < seq_length; ++t) {
    int seq_idx = forward ? t : (seq_length - t - 1);
    float *x = input + seq_idx * batch_size * input_size;
    mkldnn_ip(pre_state_h, r_i, r_bi, gate_i.data(), batch_size, hidden_size,
              hidden_size, false);
    mkldnn_ip(pre_state_h, r_o, r_bo, gate_o.data(), batch_size, hidden_size,
              hidden_size, false);
    mkldnn_ip(pre_state_h, r_f, r_bf, gate_f.data(), batch_size, hidden_size,
              hidden_size, false);
    mkldnn_ip(pre_state_h, r_c, r_bc, gate_c.data(), batch_size, hidden_size,
              hidden_size, false);
    if (datatype == DataType::BF16) {
      clean16bitmantissa(gate_i.data(), gate_i.data(), gate_i.size());
      clean16bitmantissa(gate_o.data(), gate_o.data(), gate_o.size());
      clean16bitmantissa(gate_f.data(), gate_f.data(), gate_f.size());
      clean16bitmantissa(gate_c.data(), gate_c.data(), gate_c.size());
    }
    for (int batch = 0; batch < batch_size; batch++) {
      float *xi = x + batch * input_size;
      float *xo = xi + hidden_size;
      float *xf = xo + hidden_size;
      float *xc = xf + hidden_size;
      float *gi = gate_i.data() + batch * hidden_size;
      float *go = gate_o.data() + batch * hidden_size;
      float *gf = gate_f.data() + batch * hidden_size;
      float *gc = gate_c.data() + batch * hidden_size;
      float *cell_state = pre_state_c + batch * hidden_size;
      float *hidden_state =
          output + (seq_idx * num_dir * batch_size + batch) * hidden_size;
#pragma omp parallel for schedule(static, omp_schedule(hidden_size))
      for (int i = 0; i < hidden_size; ++i) {
        gi[i] = sigmoid_(gi[i] + xi[i]);
        gf[i] = sigmoid_(gf[i] + xf[i]);
        gc[i] = tanh_(gc[i] + xc[i]);
        go[i] = sigmoid_(go[i] + xo[i]);
        if (datatype != DataType::BF16) {
          cell_state[i] = gf[i] * cell_state[i] + gi[i] * gc[i];
          hidden_state[i] = go[i] * tanh_(cell_state[i]);
        } else {
          cell_state[i] =
              BF16(BF16(gf[i] * cell_state[i]) + BF16(gi[i] * gc[i]));
          hidden_state[i] = BF16(go[i] * tanh_(cell_state[i]));
        }
      }
    }
    pre_state_h = output + seq_idx * num_dir * batch_size * hidden_size;
  }
}

void LstmOpKernel::invoke() {
  compute();
  if (bidirectional) {
    compute(false);
  }
  if (datatype == DataType::BF16) {
    clean16bitmantissa(output_data->data(), output_data->data(),
                       output_data->size());
  }
}

void LstmOpKernel::dump() { OpKernel::dump(); }
} // namespace mlir