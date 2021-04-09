#include "tpuc/Interpreter/cpu/matmul.hpp"
#include "bmkernel/bm1880v2/1880v2_fp_convert.h"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Interpreter/cpu/activation.hpp"
#include "tpuc/ModuleInterpreter.h"
#include "tpuc/NativeCpuImplementation.h"

namespace mlir {

MatMulOpKernel::MatMulOpKernel(Operation &op, value_map_t &valueMapping) {
  auto castOp = cast<tpu::MatMulOp>(op);
  assert(castOp);
  LLVM_DEBUG(llvm::outs() << " MatMulOp op: [" << castOp.name() << "]\n";);

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = castOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  LLVM_DEBUG(llvm::outs() << "    =>required memory size: [" << size << "]\n";);
  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();
  this->name = castOp.name().str();
  this->op_type = op.getName().getStringRef().str();
  this->do_relu = castOp.do_relu();
  set_datatype(getOpQuant(&op).str());
  parseMatMulParam(op.getOperand(0), op.getOperand(1), batch, M, K, N);

  if (datatype == DataType::INT8) {
    auto quant_rshift = opTensors[4];
    auto quant_multiplier = opTensors[5];
    assert(quant_rshift);
    assert(quant_multiplier);
    rshift = quant_rshift->at(0);
    multiplier = quant_multiplier->at(0);
  }
  left_data = opTensors[0];
  right_data = opTensors[1];
  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}
void MatMulOpKernel::set_tensor(const std::vector<float> &data) {
  llvm_unreachable("not support now!!");
};

std::vector<float> MatMulOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void MatMulOpKernel::invoke() {
  std::vector<float> transposed_right(right_data->size());
  float *r_data = right_data->data();
  float *l_data = left_data->data();
  float *o_data = output_data->data();
  float *t_data = transposed_right.data();
  for (int b = 0; b < batch; b++) {
    int offset = b * K * N;
    for (int i = 0; i < K; i++) {
      for (int j = 0; j < N; j++) {
        t_data[offset + j * K + i] = r_data[offset + i * N + j];
      }
    }
  }
  for (int i = 0; i < batch; i++) {
    int ret = mkldnn_ip(l_data + M * K * i, t_data + K * N * i, nullptr,
                        o_data + M * N * i, M, K, N, false);
    assert(ret == 0);
  }
  if (do_relu) {
    relu(output_data->data(), output_data->data(), output_data->size());
  }
  switch (datatype) {
  case DataType::INT8:
    if (rshift != 0 || multiplier != 0) {
      for (size_t i = 0; i < output_data->size(); ++i) {
        output_data->at(i) = (float)applyMultiplierAndRShiftAndSaturateInt8(
            output_data->at(i), (uint32_t)rshift, (uint32_t)multiplier, true);
      }
    }
    break;
  case DataType::BF16:
    clean16bitmantissa(output_data->data(), output_data->data(),
                       output_data->size());
    break;
  default:
    break;
  }
}

void MatMulOpKernel::dump() { OpKernel::dump(); }
} // namespace mlir