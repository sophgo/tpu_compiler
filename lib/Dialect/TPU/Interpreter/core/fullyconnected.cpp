#include "tpuc/Interpreter/cpu/fullyconnected.hpp"
#include "internal.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/MlirModuleInterpreter.h"
#include "tpuc/NativeCpuImplementation.h"
namespace mlir {
FullyConnectedOpKernel::FullyConnectedOpKernel(Operation &op,
                                               value_map_t &valueMapping,
                                               weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  auto fcOp = cast<tpu::FullyConnectedOp>(op);
  ltrans = fcOp.input_transpose();
  otrans = fcOp.output_transpose();
  parseFullyConnectedParam<tpu::FullyConnectedOp>(&op, batch_high, batch_low, m,
                                                  k, n);
  batch = batch_low * batch_high;
  // get tensors
  input_data = this->opdTensors[0];
  filter_data = this->opdTensors[1];
  bias_data = this->opdTensors[2];
  output_data = this->resTensor;

  this->do_relu = fcOp.do_relu();

  if (datatype == DataType::INT8) {
    rshift_data = this->opdTensors[5];
    multiplier_data = this->opdTensors[6];
  }
} // namespace mlir

void FullyConnectedOpKernel::invoke() {
  float *p_bias = nullptr;
  if (bias_data) {
    p_bias = bias_data->data();
  }
  float *l_data = input_data->data();
  float *o_data = output_data->data();
  std::vector<float> l_buff;
  std::vector<float> o_buff;
  if (ltrans) {
    l_buff.resize(input_data->size());
    my_permute(l_data, l_buff.data(), batch_high, m, batch_low, k, 0, 2, 1, 3);
    l_data = l_buff.data();
  }
  if (otrans) {
    o_buff.resize(output_data->size());
    o_data = o_buff.data();
  }
#pragma omp parallel for schedule(static, omp_schedule(batch))
  for (int i = 0; i < batch; i++) {
    int ret = mkldnn_ip(l_data + m * k * i, filter_data->data() + k * n * i,
                        p_bias == nullptr ? p_bias : (p_bias + i * n),
                        o_data + m * n * i, m, k, n, false);
    assert(ret == 0);
  }
  if (otrans) {
    my_permute(o_data, output_data->data(), batch_high, batch_low, m, n, 0, 2,
               1, 3);
  }
  if (do_relu) {
    relu(output_data->data(), output_data->data(), output_data->size());
  }

  int isz = m * n;
  if (datatype == DataType::INT8) {
    for (int i = 0; i < batch; i++) {
#pragma omp parallel for schedule(static, omp_schedule(isz))
      for (int j = 0; j < isz; ++j) {
        output_data->at(i * isz + j) =
            (float)applyMultiplierAndRShiftAndSaturateInt8(
                output_data->at(i * isz + j), (uint32_t)rshift_data->at(i),
                (uint32_t)multiplier_data->at(i), true);
      }
    }
  } else if (datatype == DataType::BF16) {
    BF16(output_data->data(), output_data->data(), output_data->size(), true);
  }
}

} // namespace mlir