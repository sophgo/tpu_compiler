#ifndef INTERPRETER_CPU_POOLING_H
#define INTERPRETER_CPU_POOLING_H

#include "mkldnn.hpp"
#include "tpuc/Interpreter/cpukernel.h"

#include <memory>
namespace mlir {

class PoolingOpKernel : public CPUOpKernel<PoolingOpKernel> {

  enum class POOL_METHOD { MAX, AVG };

public:
  static constexpr const char *OpName = "CPUPoolingOp";

  PoolingOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
  void set_tensor(const std::vector<float> &data) override;
  std::vector<float> get_tensor() override;
  void dump() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;
  SyncedDataShape scale_shape;
  // param
  POOL_METHOD pool_method;
  bool is_global;
  bool do_relu;
  bool count_include_pad;
  int n;
  int c;
  int ih;
  int iw;
  int oh;
  int ow;
  int kh;
  int kw;
  int sh;
  int sw;
  int pt;
  int pb;
  int pl;
  int pr;
  int pad_value = 0;
  bool is_asymmetric = false;

  // mkldnn setting
  mkldnn::engine mkl_eng;
  mkldnn::stream mkl_stream;

  std::vector<mkldnn::primitive> mkl_net;
  std::vector<std::unordered_map<int, mkldnn::memory>> mkl_net_args;
};
} // namespace mlir

#endif