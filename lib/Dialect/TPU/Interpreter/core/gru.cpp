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
  assert(bidirectional == false);
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

void GruOpKernel::invoke() {
  // TODO: optimize gru implementation, ex: use mkldnn
  float *prev_hidden_state = initial_h->data();      // ht
  std::vector<double> update_gate(hidden_size, 0.0); // zt
  std::vector<double> reset_gate(hidden_size, 0.0);  // rt
  std::vector<double> hidden_gate(hidden_size, 0.0); // ht

  float *w_z = weight->data();
  float *w_r = w_z + hidden_size * input_size;
  float *w_h = w_r + hidden_size * input_size;

  float *r_z = recurrence->data();
  float *r_r = r_z + hidden_size * hidden_size;
  float *r_h = r_r + hidden_size * hidden_size;

  float *w_bz = bias->data();
  float *w_br = w_bz + hidden_size;
  float *w_bh = w_br + hidden_size;
  float *r_bz = w_bh + hidden_size;
  float *r_br = r_bz + hidden_size;
  float *r_bh = r_br + hidden_size;

  for (int t = 0; t < seq_length; ++t) {
    // zt = sigmoid(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
    // rt = sigmoid(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
    // ht = tanh(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when
    // linear_before_reset != 0 Wzrh: hidden_size * input_size Rzrh: hidden_size
    // * hidden_size Xt: seq_len * batch_size * input_size
    float *xt = input_data->data() + (t * input_size);
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
        update_gate[i] += rz[j] * prev_hidden_state[j];
        reset_gate[i] += rr[j] * prev_hidden_state[j];
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
        hidden_gate_acc += rh[j] * prev_hidden_state[j];
      }
      hidden_gate[i] = tanh_(hidden_gate[i] + reset_gate[i] * hidden_gate_acc);
    }

    // Ht = (1 - zt) (.) ht + zt (.) Ht-1
    float *hidden_state = output_data->data() + t * hidden_size;
    for (int i = 0; i < hidden_size; ++i) {
      hidden_state[i] = (1 - update_gate[i]) * hidden_gate[i] +
                        update_gate[i] * prev_hidden_state[i];
    }
    prev_hidden_state = hidden_state;
  }
}

void GruOpKernel::dump() { OpKernel::dump(); }
} // namespace mlir