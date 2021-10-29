#ifndef INTERPRETER_CPU_POOLING_H
#define INTERPRETER_CPU_POOLING_H

#include "mkldnn.hpp"
#include "tpuc/Interpreter/cpukernel.h"
#include "tpuc/NativeCpuImplementation.h"

#include <memory>
namespace mlir {

class PoolingOpKernel : public CPUOpKernel {

  enum class POOL_METHOD { MAX, AVG };

public:
  static constexpr const char *OpName = "CPUPoolingOp";

  PoolingOpKernel(Operation &op, value_map_t &valueMapping,
                  weight_map_t &weightMapping);

  void invoke() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;
  SyncedDataShape scale_shape;
  MKLPooling pool;
  MKLInt8AvgPooling int8_avg_pool;
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

  // int8 param
  float rshift;
  float multiplier;
  SyncedData filter_data;

  // mkldnn setting
  mkldnn::engine mkl_eng;
  mkldnn::stream mkl_stream;

  std::vector<mkldnn::primitive> mkl_net;
  std::vector<std::unordered_map<int, mkldnn::memory>> mkl_net_args;
};
} // namespace mlir

#endif