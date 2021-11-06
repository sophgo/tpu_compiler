#ifndef INTERPRETER_CPU_CONV_H
#define INTERPRETER_CPU_CONV_H

#include "mkldnn.hpp"
#include "tpuc/Interpreter/cpukernel.h"
#include "tpuc/NativeCpuImplementation.h"
#include <memory>
namespace mlir {

class Conv2DOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUConv2DOp";

  Conv2DOpKernel(Operation &op, value_map_t &valueMapping,
                 weight_map_t &weightMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedData filter_data;
  SyncedData bias_data;
  SyncedDataShape input_shape;
  SyncedDataShape filter_shape;
  SyncedDataShape bias_shape;

  SyncedData zero_bias;
  std::vector<float> tmp_input_data;
  // int8
  SyncedData quant_rshift;
  SyncedData quant_multiplier;
  // bf16
  SyncedData quant_scale;
  SyncedData quant_zeropoint;

  // param
  bool is_dw;
  bool with_bias;
  bool do_relu;
  int n;
  int ic;
  int ih;
  int iw;
  int oc;
  int oh;
  int ow;
  int g;
  int kh;
  int kw;
  int ins_h;
  int ins_w;
  int sh;
  int sw;
  int pt;
  int pb;
  int pl;
  int pr;
  int dh;
  int dw;
  int pad_value;
  bool is_deconv = false;
  bool do_bias_later = false;
  bool do_relu_later = false;
  bool activation_bf16 = false;

  // mkldnn setting
  MKLConv conv;

};
} // namespace mlir

#endif
