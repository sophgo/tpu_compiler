#ifndef INTERPRETER_CPU_FULLYCONNECTED_H
#define INTERPRETER_CPU_FULLYCONNECTED_H

#include "mkldnn.hpp"
#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {

class FullyConnectedOpKernel : public CPUOpKernel<FullyConnectedOpKernel> {
public:
  static constexpr const char *OpName = "CPUFullyConnectedOp";

  FullyConnectedOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
  void set_tensor(const std::vector<float> &data) override;
  std::vector<float> get_tensor() override;
  void dump() override{};

private:
  SyncedData input_data;
  SyncedData filter_data;
  SyncedData bias_data;
  SyncedData output_data;
  SyncedDataShape input_shape;
  SyncedDataShape output_shape;
  SyncedDataShape filter_shape;
  SyncedDataShape bias_shape;

  // param
  int m;
  int k;
  int n;
  // mkldnn setting
  mkldnn::engine mkl_eng;
  mkldnn::stream mkl_stream;

  std::vector<mkldnn::primitive> mkl_net;
  std::vector<std::unordered_map<int, mkldnn::memory>> mkl_net_args;
};
} // namespace mlir

#endif