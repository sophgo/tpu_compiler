#ifndef INTERPRETER_CPU_LSTM_H
#define INTERPRETER_CPU_LSTM_H

#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {

class LstmOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPULstmOp";

  LstmOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  double sigmoid_(double data);
  double tanh_(double data);
  void update_addr(bool forward = true);
  void compute(bool forward = true);

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedData recurrence;
  SyncedData bias;
  SyncedData initial_h;
  SyncedData initial_c;
  SyncedData sigmoid_lut;
  SyncedData sigmoid_slope_lut;
  SyncedData tanh_lut;
  SyncedData tanh_slope_lut;
  SyncedDataShape input_shape;

  // param
  int seq_length;
  int hidden_size;
  int batch_size;
  int input_size;
  int num_dir;
  bool bidirectional;

  // addr
  float *r_i;
  float *r_f;
  float *r_c;
  float *r_o;
  float *r_bi;
  float *r_bf;
  float *r_bc;
  float *r_bo;
  float *input;
  float *output;
  float *pre_state_h;
  float *pre_state_c;
};
} // namespace mlir
#endif