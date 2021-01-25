#ifndef INTERPRETER_CPU_DECONV_H
#define INTERPRETER_CPU_DECONV_H

#include "mkldnn.hpp"
#include "tpuc/Interpreter/cpukernel.h"
#include <memory>
namespace mlir {

class DeConv2DOpKernel : public CPUOpKernel<DeConv2DOpKernel> {
public:
  static constexpr const char *OpName = "CPUDeConv2DOp";

  DeConv2DOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
  void set_tensor(const std::vector<float> &data) override;
  std::vector<float> get_tensor() override;
  void dump();

private:
  void fp32_invoke();
  void i8_invoke();

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
  bool is_asymmetric;

  // mkldnn setting
  mkldnn::engine mkl_eng;
  mkldnn::stream mkl_stream;

  std::vector<mkldnn::primitive> mkl_net;
  std::vector<std::unordered_map<int, mkldnn::memory>> mkl_net_args;

  // int8 param
  bool is_perchannel = false;
  bool use_multiplier = false;
};
} // namespace mlir

#endif