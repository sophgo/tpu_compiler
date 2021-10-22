#include "tpuc/Interpreter/cpu/convfc.hpp"
#include "internal.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/MlirModuleInterpreter.h"
#include "tpuc/NativeCpuImplementation.h"

namespace mlir {

ConvFcOpKernel::ConvFcOpKernel(Operation &op, value_map_t &valueMapping,
                               weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  auto castOp = cast<tpu::ConvFcOp>(op);
  input_data = this->opdTensors[0];
  filter_data = this->opdTensors[1];
  output_data = this->resTensor;
  qscale = castOp.qscale().convertToFloat();
  M = shape[0];
  N = shape[1];
  auto input_shape = getTensorShape(castOp.input());
  K = input_shape[1];
}

void ConvFcOpKernel::invoke() {
  switch (datatype) {
  case DataType::FP32: {
    int ret = mkldnn_ip(input_data->data(), filter_data->data(), nullptr,
                        output_data->data(), M, K, N, false);
    assert(ret == 0);
  } break;
  case DataType::BF16: {
    int filter_size = filter_data->size();
    std::vector<float> new_filter(filter_size);
    for (int i = 0; i < filter_size; i++) {
      new_filter[i] = BF16(filter_data->at(i) * qscale);
    }
    int ret = mkldnn_ip(input_data->data(), new_filter.data(), nullptr,
                        output_data->data(), M, K, N, false);
    assert(ret == 0);
    BF16(output_data->data(), output_data->data(), output_data->size(), true);
  } break;
  default:
    break;
  }
}

} // namespace mlir