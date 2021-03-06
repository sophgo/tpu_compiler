#include "tpuc/Interpreter/cpu/embedding.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/MlirModuleInterpreter.h"

namespace mlir {
EmbeddingOpKernel::EmbeddingOpKernel(Operation &op, value_map_t &valueMapping,
                                     weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  auto embeddingOp = cast<tpu::EmbeddingOp>(op);
  auto input_type = embeddingOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();

  auto table_type = embeddingOp.table().getType().template cast<TensorType>();
  this->table_shape = table_type.getShape();

  auto output_type = embeddingOp.output().getType().template cast<TensorType>();
  this->output_shape = output_type.getShape();
  // get tensors
  input_data = this->opdTensors[0];
  table_data = this->opdTensors[1];
  scale_data = this->opdTensors[2];
  zeropoint_data = this->opdTensors[3];
  output_data = this->resTensor;
  mix_bf16 = false;
  if (datatype == DataType::BF16) {
    if (getOpQuantParamType(&op) == "MIX_BF16") {
      mix_bf16 = true;
    }
  }
}

void EmbeddingOpKernel::invoke() {
  auto input = input_data->data();
  auto table = table_data->data();
  auto output = output_data->data();

  auto feature_dim = table_shape.back();
  assert(output_shape.back() == feature_dim && "must be the same feature dim");
  int64_t count = std::accumulate(input_shape.begin(), input_shape.end(), 1,
                                  std::multiplies<int64_t>());
  for (int64_t i = 0; i < count; i++) {
    auto index = (size_t)input[i];
    size_t table_offset = (size_t)index * feature_dim;
    auto out_offset = i * feature_dim;
    if (mix_bf16 == false) {
      memcpy(output + out_offset, table + table_offset,
             feature_dim * sizeof(float));
    } else {
      for (int64_t j = 0; j < feature_dim; j++) {
        output[out_offset + j] =
            BF16(BF16(table[table_offset + j] * scale_data->at(j)) + zeropoint_data->at(j));
      }
    }
  }
}

} // namespace mlir
