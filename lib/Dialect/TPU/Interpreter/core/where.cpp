#include "tpuc/Interpreter/cpu/where.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {
WhereOpKernel::WhereOpKernel(Operation &op, value_map_t &valueMapping,
                             weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  auto whereOp = cast<tpu::WhereOp>(op);
  auto input_type = whereOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();

  auto condition_type = whereOp.condition().getType().template cast<TensorType>();
  this->condition_shape = condition_type.getShape();

  auto output_type = whereOp.output().getType().template cast<TensorType>();
  this->output_shape = output_type.getShape();

  this->fill_constant = whereOp.fill_constant().convertToFloat();

  if (datatype == DataType::INT8) {
    auto quant_rshift = this->opdTensors[5];
    auto quant_multiplier = this->opdTensors[6];
    if (quant_rshift != nullptr && quant_multiplier != nullptr) {
      need_quant = true;
      rshift.assign(quant_rshift->begin(), quant_rshift->end());
      multiplier.assign(quant_multiplier->begin(), quant_multiplier->end());
    }
  }

  // TODO: check mask and input can broadcastable

  // get tensors
  input_data = this->opdTensors[0];
  condition_data = this->opdTensors[1];

  if (!isTensorNone(whereOp.x())) {
      auto x_type = whereOp.x().getType().template cast<TensorType>();
      this->x_shape = x_type.getShape();
      x_data = this->opdTensors[2];
      llvm_unreachable("not support x as tensor case");
  }

  output_data = this->resTensor;
}

void WhereOpKernel::invoke() {
  auto input = input_data->data();
  auto output = output_data->data();
  auto condition = condition_data->data();
#define RANKS (4)
  std::vector<int> nchw(RANKS, 1); 
  std::vector<int> condition_nchw(RANKS, 1); 
  std::vector<int> condition_stride(RANKS, 0); 
  // extend to 4 dims
  int shift_dim = RANKS - input_shape.size();
  for (int i = 0; i < (int)input_shape.size(); i++) {
      nchw[i + shift_dim] = input_shape[i];
      condition_nchw[i + shift_dim] = condition_shape[i];
  }
  condition_stride[3] = condition_nchw[3] == 1 ? 0 : 1;
  condition_stride[2] = condition_nchw[2] * condition_nchw[3] == 1 ? 0 : condition_nchw[3];
  condition_stride[1] = condition_nchw[1] * condition_nchw[2] * condition_nchw[3] == 1 ? 0 : (condition_nchw[3] * condition_nchw[2]);
  condition_stride[0] = condition_nchw[0] * condition_nchw[1] * condition_nchw[2] * condition_nchw[3] == 1 ? 0 : (condition_nchw[3] * condition_nchw[2] * condition_nchw[1]);

  for (int n = 0; n < nchw[0]; n++) {
      for (int c = 0; c < nchw[1]; c++) {
          for (int h = 0; h < nchw[2]; h++) {
              for (int w = 0; w < nchw[3]; w++) {
                  int idx = n * (nchw[1] * nchw[2] * nchw[3])
                      + c * (nchw[2] * nchw[3])
                      + h * nchw[3]
                      + w;
                  int mask_idx = n * condition_stride[0] +
                      c * condition_stride[1] +
                      h * condition_stride[2] +
                      w * condition_stride[3];
                  output[idx] = condition[mask_idx]
                      ? fill_constant : input[idx];
                  if (need_quant) {
                      output[idx] = (float)applyMultiplierAndRShiftAndSaturateInt8(
                                  output[idx],
                                  (uint32_t)rshift.at(0),
                                  (uint32_t)multiplier.at(0));
                  }
              }
          }
      }
  }
}

} // namespace mlir
