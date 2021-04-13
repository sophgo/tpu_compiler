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
  l_trans = castOp.left_transpose();
  r_trans = castOp.right_transpose();
  o_trans = castOp.output_transpose();
  parseMatMulParam(op.getOperand(0), op.getOperand(1), op.getResult(0), M, K, N,
                   batch_high, batch_low, l_trans, r_trans, o_trans);
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
  float *r_data = right_data->data();
  float *l_data = left_data->data();
  float *o_data = output_data->data();
  std::vector<float> l_buff;
  std::vector<float> r_buff;
  std::vector<float> o_buff;
  if (l_trans) {
    l_buff.resize(left_data->size());
    my_permute(l_data, l_buff.data(), batch_high, M, batch_low, K, 0, 2, 1, 3);
    l_data = l_buff.data();
  }
  r_buff.resize(right_data->size());
  if (r_trans) {
    my_permute(r_data, r_buff.data(), batch_high, K, batch_low, N, 0, 2, 3, 1);
    r_data = r_buff.data();
  } else {
    my_permute(r_data, r_buff.data(), batch_high, batch_low, K, N, 0, 1, 3, 2);
    r_data = r_buff.data();
  }
  if (o_trans) {
    o_buff.resize(output_data->size());
    o_data = o_buff.data();
  }
  int batch = batch_low * batch_high;
#pragma omp parallel for schedule(static, batch / omp_get_num_threads())
  for (int i = 0; i < batch; i++) {
    int ret = mkldnn_ip(l_data + M * K * i, r_data + K * N * i, nullptr,
                        o_data + M * N * i, M, K, N, false);
    assert(ret == 0);
  }
  if (o_trans) {
    my_permute(o_data, output_data->data(), batch_high, batch_low, M, N, 0, 2,
               1, 3);
  }
  if (do_relu) {
    relu(output_data->data(), output_data->data(), output_data->size());
  }
  switch (datatype) {
  case DataType::INT8:
    if (rshift != 0 || multiplier != 0) {
      int out_size = output_data->size();
#pragma omp parallel for schedule(static, out_size / omp_get_num_threads())
      for (int i = 0; i < out_size; ++i) {
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