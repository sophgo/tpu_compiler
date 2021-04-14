#ifndef INTERPRETER_CPU_GRU_H
#define INTERPRETER_CPU_GRU_H

#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {

class GruOpKernel : public CPUOpKernel<GruOpKernel> {
public:
  static constexpr const char *OpName = "CPUGruOp";

  GruOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
  void set_tensor(const std::vector<float> &data) override;
  std::vector<float> get_tensor() override;
  void dump() override;

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
  SyncedDataShape input_shape;

  // param
  int seq_length;
  int hidden_size;
  int batch_size;
  int input_size;
  int num_dir;
  bool bidirectional;
  bool linear_before_reset;
  bool only_last;
  // addr
  float *r_z;
  float *r_r;
  float *r_h;
  float *r_bz;
  float *r_br;
  float *r_bh;
  float *input;
  float *output;
  float *prev_hidden_state;

  // bf16 only
  std::vector<uint16_t> sigmoid_lut;
  std::vector<uint16_t> sigmoid_slope_lut;
  std::vector<uint16_t> tanh_lut;
  std::vector<uint16_t> tanh_slope_lut;
};
} // namespace mlir
#endif