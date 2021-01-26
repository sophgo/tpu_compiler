#ifndef INTERPRETER_CPU_DIALATE_H
#define INTERPRETER_CPU_DIALATE_H

#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {

void dilateActivation(float *input, float *output, int pad_h_t, int pad_h_b,
                      int ins_h, int ins_h_l, int pad_w_l, int pad_w_r,
                      int ins_w, int ins_w_l, int n, int c, int h, int w,
                      int fill_constant);
class DilateOpKernel : public CPUOpKernel<DilateOpKernel> {
public:
  static constexpr const char *OpName = "CPUDilateOp";

  DilateOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
  void set_tensor(const std::vector<float> &data) override;
  std::vector<float> get_tensor() override;
  void dump() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;

  // param
  int ins_h = 0;
  int ins_w = 0;
  int fill_constant;
};
} // namespace mlir
#endif