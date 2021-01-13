#ifndef INTERPRETER_CPU_CONV_H
#define INTERPRETER_CPU_CONV_H

#include "mkldnn.hpp"
#include "tpuc/Interpreter/cpukernel.h"
#include <memory>
namespace mlir {

class Conv2DOpKernel : public CPUOpKernel<Conv2DOpKernel> {
public:
  static constexpr const char *OpName = "CPUConv2DOp";

  Conv2DOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
  void set_tensor(const std::vector<float> &data) override;
  std::vector<float> get_tensor() override;
  void dump();

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedData filter_data;
  SyncedData bias_data;
  SyncedDataShape input_shape;
  SyncedDataShape filter_shape;
  SyncedDataShape bias_shape;
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
};
} // namespace mlir

#endif