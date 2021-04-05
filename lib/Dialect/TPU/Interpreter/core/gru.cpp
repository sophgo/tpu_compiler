#include "tpuc/Interpreter/cpu/gru.hpp"
#include "bmkernel/bm1880v2/1880v2_fp_convert.h"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"
#include "tpuc/NativeCpuImplementation.h"

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
  auto weight_type = gruOp.weight().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();
  this->weight_shape = weight_type.getShape();
  this->name = gruOp.name().str();
  this->op_type = op.getName().getStringRef().str();
  set_datatype(getOpQuant(&op).str());
  seq_length = input_shape[0];
  batch_size = input_shape[1];
  input_size = input_shape[2];
  num_dir = weight_shape[0];
  hidden_size = weight_shape[1] / 3; // 3 gates
  linear_before_reset = gruOp.linear_before_reset();
  assert(linear_before_reset == true);
  bidirectional = gruOp.bidirectional();
  // get tensors
  input_data = opTensors[0];
  weight = opTensors[1];
  recurrence = opTensors[2];
  bias = opTensors[3];
  initial_h = opTensors[4];
  if (initial_h == nullptr) {
    initial_h = std::make_shared<std::vector<float>>(
        num_dir * batch_size * hidden_size, 0.0f);
  }
  if (datatype == DataType::BF16) {
    sigmoid_lut.assign(opTensors[5]->begin(), opTensors[5]->end());
    sigmoid_slope_lut.assign(opTensors[6]->begin(), opTensors[6]->end());
    tanh_lut.assign(opTensors[7]->begin(), opTensors[7]->end());
    tanh_slope_lut.assign(opTensors[8]->begin(), opTensors[8]->end());
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
    w_z = weight->data();
    r_z = recurrence->data();
    w_bz = bias->data();
    output = output_data->data();
    prev_hidden_state = initial_h->data();
  } else {
    w_z = weight->data() + 3 * hidden_size * input_size;
    r_z = recurrence->data() + 3 * hidden_size * hidden_size;
    w_bz = bias->data() + 6 * hidden_size;
    output = output_data->data() + batch_size * hidden_size;
    prev_hidden_state = initial_h->data() + batch_size * hidden_size;
  }
  w_r = w_z + hidden_size * input_size;
  w_h = w_r + hidden_size * input_size;
  r_r = r_z + hidden_size * hidden_size;
  r_h = r_r + hidden_size * hidden_size;
  w_br = w_bz + hidden_size;
  w_bh = w_br + hidden_size;
  r_bz = w_bh + hidden_size;
  r_br = r_bz + hidden_size;
  r_bh = r_br + hidden_size;
}

void GruOpKernel::compute(bool forward) {
  update_addr(forward);
  std::vector<double> update_gate(hidden_size, 0.0); // zt
  std::vector<double> reset_gate(hidden_size, 0.0);  // rt
  std::vector<double> hidden_gate(hidden_size, 0.0); // ht

  for (int t = 0; t < seq_length; ++t) {
    int seq_idx = t;
    if (forward == false) {
      seq_idx = seq_length - t - 1;
    }
    // zt = sigmoid(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
    // rt = sigmoid(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
    // ht = tanh(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when
    // linear_before_reset != 0 Wzrh: hidden_size * input_size Rzrh: hidden_size
    // * hidden_size Xt: seq_len * batch_size * input_size
    for (int batch = 0; batch < batch_size; batch++) {
      float *xt =
          input_data->data() + (seq_idx * batch_size + batch) * input_size;
      update_gate.assign(hidden_size, 0.0);
      reset_gate.assign(hidden_size, 0.0);
      hidden_gate.assign(hidden_size, 0.0);
      for (int i = 0; i < hidden_size; ++i) {
        float *wz = w_z + i * input_size;
        float *wr = w_r + i * input_size;
        float *wh = w_h + i * input_size;
        float *rz = r_z + i * hidden_size;
        float *rr = r_r + i * hidden_size;

        for (int j = 0; j < input_size; ++j) {
          update_gate[i] += wz[j] * xt[j];
          reset_gate[i] += wr[j] * xt[j];
          hidden_gate[i] += wh[j] * xt[j];
        }

        for (int j = 0; j < hidden_size; ++j) {
          update_gate[i] += rz[j] * prev_hidden_state[batch * hidden_size + j];
          reset_gate[i] += rr[j] * prev_hidden_state[batch * hidden_size + j];
        }
        update_gate[i] = sigmoid_(update_gate[i] + w_bz[i] + r_bz[i]);
        reset_gate[i] = sigmoid_(reset_gate[i] + w_br[i] + r_br[i]);
        hidden_gate[i] += w_bh[i];
      }

      // second part of hidden gate
      // (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset != 0
      for (int i = 0; i < hidden_size; ++i) {
        float *rh = r_h + i * hidden_size;
        double hidden_gate_acc = r_bh[i];
        for (int j = 0; j < hidden_size; ++j) {
          hidden_gate_acc += rh[j] * prev_hidden_state[batch * hidden_size + j];
        }
        hidden_gate[i] =
            tanh_(hidden_gate[i] + reset_gate[i] * hidden_gate_acc);
      }

      // Ht = (1 - zt) (.) ht + zt (.) Ht-1
      float *hidden_state =
          output + (seq_idx * num_dir * batch_size + batch) * hidden_size;
      for (int i = 0; i < hidden_size; ++i) {
        hidden_state[i] =
            (1 - update_gate[i]) * hidden_gate[i] +
            update_gate[i] * prev_hidden_state[batch * hidden_size + i];
      }
    }
    prev_hidden_state = output + seq_idx * num_dir * batch_size * hidden_size;
  }
}

void GruOpKernel::invoke() {
  compute();
  if (bidirectional) {
    compute(false);
  }
}

void GruOpKernel::dump() { OpKernel::dump(); }
} // namespace mlir