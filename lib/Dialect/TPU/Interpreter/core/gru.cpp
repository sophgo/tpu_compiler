#include "tpuc/Interpreter/cpu/gru.hpp"
#include "bmkernel/bm1880v2/1880v2_fp_convert.h"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"
#include "tpuc/NativeCpuImplementation.h"

namespace mlir {
template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 0.5 * tanh(0.5 * x) + 0.5;
}

static void my_gru(float *input, float *output, float *weight,
                   float *recurrence, float *bias, float *initial_h,
                   int seq_len, int batch_size, int input_size, int hidden_size,
                   bool b_bidirectional, bool b_linear_before_reset) {
  assert(b_bidirectional == false);
  assert(b_linear_before_reset == true);
  assert(batch_size == 1);
  // TODO: optimize gru implementation, ex: use mkldnn
  // weight: Concatenation of weight matrix for update, reset, and hidden gates.
  // shape = [num_directions, 3*hidden_size, input_size] recurrence:
  // Concatenation of recurrence weight matrix for update, reset, and hidden
  // gates bias: Concatenation of Wb[update, reset, hidden gates] and Rb[update,
  // reset, hidden gates], shape = [num_directions, 6*hidden_size] initial_h:
  // [num_directions, batch_size, hidden_size]

  // int num_directions = b_bidirectional ? 2 : 1;
  // int gate_weight_size = hidden_size * input_size;
  float *prev_hidden_state = initial_h;              // ht
  std::vector<double> update_gate(hidden_size, 0.0); // zt
  std::vector<double> reset_gate(hidden_size, 0.0);  // rt
  std::vector<double> hidden_gate(hidden_size, 0.0); // ht

  float *w_z = weight;
  float *w_r = w_z + hidden_size * input_size;
  float *w_h = w_r + hidden_size * input_size;

  float *r_z = recurrence;
  float *r_r = r_z + hidden_size * hidden_size;
  float *r_h = r_r + hidden_size * hidden_size;

  float *w_bz = bias;
  float *w_br = w_bz + hidden_size;
  float *w_bh = w_br + hidden_size;
  float *r_bz = w_bh + hidden_size;
  float *r_br = r_bz + hidden_size;
  float *r_bh = r_br + hidden_size;

  for (int t = 0; t < seq_len; ++t) {
    // zt = sigmoid(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
    // rt = sigmoid(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
    // ht = tanh(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when
    // linear_before_reset != 0 Wzrh: hidden_size * input_size Rzrh: hidden_size
    // * hidden_size Xt: seq_len * batch_size * input_size
    float *xt = input + (t * input_size);
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

      update_gate[i] = sigmoid(update_gate[i] + w_bz[i] + r_bz[i]);
      reset_gate[i] = sigmoid(reset_gate[i] + w_br[i] + r_br[i]);
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
      hidden_gate[i] = tanh(hidden_gate[i] + reset_gate[i] * hidden_gate_acc);
    }

    // Ht = (1 - zt) (.) ht + zt (.) Ht-1
    float *hidden_state = output + t * hidden_size;
    for (int i = 0; i < hidden_size; ++i) {
      hidden_state[i] = (1 - update_gate[i]) * hidden_gate[i] +
                        update_gate[i] * prev_hidden_state[i];
    }
    prev_hidden_state = hidden_state;
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
  my_gru(input_data->data(), output_data->data(), weight->data(),
         recurrence->data(), bias->data(), initial_h->data(), seq_length,
         batch_size, input_size, hidden_size, bidirectional,
         linear_before_reset);
}

void GruOpKernel::dump() { OpKernel::dump(); }
} // namespace mlir