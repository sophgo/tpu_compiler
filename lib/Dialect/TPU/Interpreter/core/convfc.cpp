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
  quant_scale = this->opdTensors[2];
  quant_zeropoint = this->opdTensors[3];
  output_data = this->resTensor;
  M = shape[0];
  N = shape[1];
  auto input_shape = getTensorShape(castOp.input());
  K = input_shape[1];
  activation_bf16 = false;
  if (datatype == DataType::BF16) {
    if (getOpQuantParamType(&op) == "ACTIVATION_BF16") {
      activation_bf16 = true;
      int filter_size = filter_data->size();
      for (int i = 0; i < filter_size; i++) {
        filter_data->at(i) =
            BF16(BF16(filter_data->at(i) * quant_scale->at(i % K) +
                      quant_zeropoint->at(i % K)));
      }
    }
  }
}

void ConvFcOpKernel::invoke() {
  int ret = mkldnn_ip(input_data->data(), filter_data->data(), nullptr,
                      output_data->data(), M, K, N, false);
  assert(ret == 0);
  if (DataType::BF16 == datatype) {
    BF16(output_data->data(), output_data->data(), output_data->size(), true);
  }
}

} // namespace mlir