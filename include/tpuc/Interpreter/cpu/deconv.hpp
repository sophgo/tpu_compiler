#ifndef INTERPRETER_CPU_DECONV_H
#define INTERPRETER_CPU_DECONV_H

#include "mkldnn.hpp"
#include "tpuc/Interpreter/cpukernel.h"
#include "tpuc/NativeCpuImplementation.h"
#include <memory>
namespace mlir {

class DeConv2DOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUDeConv2DOp";

  DeConv2DOpKernel(Operation &op, value_map_t &valueMapping,
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
  // int8
  std::vector<float> rshift;
  std::vector<float> multiplier;

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

  // mkldnn setting
  MKLDeconv deconv;

};
} // namespace mlir

#endif