//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file implements a translation from MLIR SPIR-V ModuleOp to SPIR-V
// binary module.
//
//===----------------------------------------------------------------------===//

#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/QuantizationArithmetic.h"
#include "tpuc/CustomOpPlugin.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Support/FileUtilities.h"
#include "tpuc/Support/TensorFile.h"
#include "cvikernel/cvikernel.h"
#include <fstream>

#define DEBUG_TYPE "tg_codegen"

using namespace mlir;


#include "backend/backend_tg_api.h"
#include "backend/backend_tl_api.h"

namespace mlir {

LogicalResult tpu::TG_INT8_AbsOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  int32_t input_number = op->getNumOperands();
  auto ga_inputs = new gaddr_t[input_number];
  for(int32_t i = 0; i < input_number; i++){
    ga_inputs[i] = getPreviousOpAddress(op, i);
  }
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_eltwise_abs_kernel(
        *backend_ctx,
        layer_id,     // layer_id
        ga_inputs,    // gaddr_t ga_input[]
        ga_output,    // gaddr_t ga_output
        1,            // int input_size
        n, c, h, w,
        0,      // bool do_relu
        false, 0, 0,
        0, 0, NULL, // rshift, *multipliers, coeffs
        CVK_FMT_I8);

  delete[] ga_inputs;

  return success();
}


LogicalResult tpu::TG_BF16_AbsOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  int32_t input_number = op->getNumOperands();
  auto ga_inputs = new gaddr_t[input_number];
  for(int32_t i = 0; i < input_number; i++){
    ga_inputs[i] = getPreviousOpAddress(op, i);
  }

  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_eltwise_abs_kernel(
        *backend_ctx,
        layer_id,     // layer_id
        ga_inputs,    // gaddr_t ga_input[]
        ga_output,    // gaddr_t ga_output
        1,            // int input_size
        n, c, h, w,
        0,      // bool do_relu
        false, 0, 0,
        0, 0, NULL, // rshift, *multipliers, coeffs
        CVK_FMT_BF16);

  delete[] ga_inputs;

  return success();
}

LogicalResult tpu::TG_INT8_ArgMaxOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(input(), shape, input_size);
  if (shape.size() == 2) {
    n = shape[0];
    w = shape[1];
    h = 1;
    c = 1;
  } else {
    getNCHW(shape, n, c, h, w);
  }
  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  llvm::errs() << "argmax:" << n << "," << c << ","<< h << "," << w << "\n";

  cvi_backend_tg_argmax_kernel(
      *backend_ctx,
      layer_id,     // layer_id
      ga_input,    // gaddr_t ga_input[]
      ga_output,    // gaddr_t ga_output
      n, c, h, w,
      256, CVK_FMT_I8);
  return success();
}

LogicalResult tpu::TG_BF16_ArgMaxOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(input(), shape, input_size);
  if (shape.size() == 2) {
    n = shape[0];
    w = shape[1];
    h = 1;
    c = 1;
  } else {
    getNCHW(shape, n, c, h, w);
  }
  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  llvm::errs() << "argmax:" << n << "," << c << ","<< h << "," << w << "\n";

  cvi_backend_tg_argmax_kernel(
      *backend_ctx,
      layer_id,     // layer_id
      ga_input,    // gaddr_t ga_input[]
      ga_output,    // gaddr_t ga_output
      n, c, h, w,
      256, CVK_FMT_BF16);
  return success();
}

LogicalResult tpu::TG_INT8_ScaleOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  int64_t bn, bc, bh, bw;
  std::vector<int64_t> bshape = getTensorShape(op->getOperand(1));
  getNCHW(bshape, bn, bc, bh, bw);
  assert(bn == 1 || bn == n);
  if (bn == n && ((n * c) <= (4095 - 32))) {
    // [4,3,28,28] x [4,3,1,1] => [1,12,28,28] x [1,12,1,1]
    c = n * c;
    n = 1;
  }

  bool do_relu = this->param().do_relu().getValue();

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  gaddr_t ga_scale = getOpAddress(filter().getDefiningOp());
  gaddr_t ga_pc_info = getWeightOpAddress(pc_info().getDefiningOp());
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_scale_kernel(*backend_ctx, // ctx
                              layer_id,     // layer_id
                              ga_input,     // input_addr
                              ga_scale,     // scale_addr
                              ga_pc_info,   // bias_addr
                              ga_output,    // output_addr
                              n, c, h, w,
                              n * c,   // scale_dim (axis = 1  =>  n * c)
                              h * w,   // inner_dim (axis = 1  =>  h * w)
                              false,   // is_scale_const
                              0,       // const_scale
                              do_relu, // do_activation,
                              false,   // with_bias
                              CVK_FMT_I8);

  return success();
}

LogicalResult tpu::TG_BF16_ScaleOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  int64_t bn, bc, bh, bw;
  std::vector<int64_t> bshape = getTensorShape(op->getOperand(1));
  getNCHW(bshape, bn, bc, bh, bw);
  assert(bn == 1 || bn == n);
  if (bn == n && ((n * c) <= (4095 - 32))) {
    // [4,3,28,28] x [4,3,1,1] => [1,12,28,28] x [1,12,1,1]
    c = n * c;
    n = 1;
  }
  bool do_relu = this->param().do_relu().getValue();

  int64_t input_size_1;
  std::vector<int64_t> shape_1;
  getTensorShapeAndSize(op->getOperand(1), shape_1, input_size_1);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  gaddr_t ga_scale = getOpAddress(filter().getDefiningOp());
  // FIXME: support bias
  // gaddr_t ga_pc_info = getWeightOpAddress(pc_info().getDefiningOp());
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_scale_kernel(*backend_ctx, // ctx
                              layer_id,     // layer_id
                              ga_input,     // input_addr
                              ga_scale,     // scale_addr
                              GA_INVALID,   // bias_addr
                              ga_output,    // output_addr
                              n, c, h, w,
                              n * c,   // scale_dim (axis = 1  =>  n * c)
                              h * w,   // inner_dim (axis = 1  =>  h * w)
                              false,   // is_scale_const
                              0,       // const_scale
                              do_relu, // do_activation
                              false, CVK_FMT_BF16);

  return success();
}

LogicalResult tpu::TG_INT8_BroadcastMulOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n");

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  int64_t n, c, h, w;
  std::vector<int64_t> shape = getTensorShape(op->getOperand(0));
  getNCHW(shape, n, c, h, w);

  int64_t bn, bc, bh, bw;
  std::vector<int64_t> bshape = getTensorShape(op->getOperand(1));
  getNCHW(bshape, bn, bc, bh, bw);

  bool do_relu = this->do_relu();

  int32_t rshift = 0;
  int32_t multiplier[2] = {1, 1}; // only one multiplier

  if (this->rshift().hasValue() && this->m_i8_inputs().hasValue()) {
    auto rshift_int8 = this->rshift().getValue();
    rshift = static_cast<int32_t>(rshift_int8);

    llvm::errs() << "broadcast mul rshift: " << rshift;

    std::vector<int32_t> multiplier_int32(2);
    arrayAttrToVector(this->m_i8_inputs().getValue(), multiplier_int32);

    for (int32_t i = 0; i < 2; i++ ){
      multiplier[i] = static_cast<int32_t>(multiplier_int32[i]);
      llvm::errs() << "broadcast mul multiplier: " << multiplier[i];
    }
  }

  gaddr_t ga_inputs[2];
  ga_inputs[0] = getPreviousOpAddress(op, 0);
  ga_inputs[1] = getPreviousOpAddress(op, 1);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);
  assert(bc == 1);

  cvi_backend_tg_int8_broadcast_mul_kernel(
    *backend_ctx, layer_id,
    ga_inputs, ga_output, n, c,
    h, w, bn, bc, bh, bw, do_relu, rshift, multiplier);

  return success();
}

LogicalResult tpu::TG_BF16_BroadcastMulOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n");

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  int64_t n, c, h, w;
  std::vector<int64_t> shape = getTensorShape(op->getOperand(0));
  getNCHW(shape, n, c, h, w);

  int64_t bn, bc, bh, bw;
  std::vector<int64_t> bshape = getTensorShape(op->getOperand(1));
  getNCHW(bshape, bn, bc, bh, bw);

  bool do_relu = this->do_relu();

  gaddr_t ga_inputs[2];
  ga_inputs[0] = getPreviousOpAddress(op, 0);
  ga_inputs[1] = getPreviousOpAddress(op, 1);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);
  assert(bc == 1);

  cvi_backend_tg_bf16_broadcast_mul_kernel(*backend_ctx, layer_id, ga_inputs,
                                           ga_output, n, c, h, w, bn, bc, bh,
                                           bw, do_relu);

  return success();
}

LogicalResult tpu::TG_INT8_BroadcastAddOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n");

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  int64_t input_size, n, c, h, w;
  std::vector<int64_t> shape;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  int64_t bn, bc, bh, bw;
  getTensorShapeAndSize(op->getOperand(1), shape, input_size);
  getNCHW(shape, bn, bc, bh, bw);

  bool do_relu = this->do_relu();

  int32_t rshift;
  int32_t multiplier[2];

  if (this->rshift().hasValue() && this->m_i8_inputs().hasValue()) {
    auto rshift_int8 = this->rshift().getValue();
    rshift = static_cast<int32_t>(rshift_int8);

    llvm::errs() << "broadcast add rshift: " << rshift;

    std::vector<int32_t> multiplier_int32(2);
    arrayAttrToVector(this->m_i8_inputs().getValue(), multiplier_int32);

    for (int32_t i = 0; i < 2; i++ ){
      multiplier[i] = static_cast<int32_t>(multiplier_int32[i]);
      llvm::errs() << "broadcast add multiplier: " << multiplier[i];
    }
  }

  gaddr_t ga_inputs[2];
  ga_inputs[0] = getPreviousOpAddress(op, 0);
  ga_inputs[1] = getPreviousOpAddress(op, 1);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_int8_broadcast_add_kernel(
    *backend_ctx, layer_id,
    ga_inputs, ga_output, n, c,
    h, w, bn, bc, bh, bw, do_relu, rshift, multiplier);
  return success();
}

LogicalResult tpu::TG_BF16_BroadcastAddOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  int64_t input_size, n, c, h, w;
  std::vector<int64_t> shape;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  int64_t bn, bc, bh, bw;
  getTensorShapeAndSize(op->getOperand(1), shape, input_size);
  getNCHW(shape, bn, bc, bh, bw);

  bool do_relu = this->do_relu();

  gaddr_t ga_inputs[2];
  ga_inputs[0] = getPreviousOpAddress(op, 0);
  ga_inputs[1] = getPreviousOpAddress(op, 1);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_bf16_broadcast_add_kernel(
    *backend_ctx, layer_id,
    ga_inputs, ga_output, n, c,
    h, w, bn, bc, bh, bw, do_relu);

  return success();
}

LogicalResult tpu::TG_INT8_BroadcastSubOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n");

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  int64_t input_size, n, c, h, w;
  std::vector<int64_t> shape;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  int64_t bn, bc, bh, bw;
  getTensorShapeAndSize(op->getOperand(1), shape, input_size);
  getNCHW(shape, bn, bc, bh, bw);

  bool do_relu = this->do_relu();

  int32_t rshift;
  int32_t multiplier[2];

  if (this->rshift().hasValue() && this->m_i8_inputs().hasValue()) {
    auto rshift_int8 = this->rshift().getValue();
    rshift = static_cast<int32_t>(rshift_int8);

    llvm::errs() << "broadcast sub rshift: " << rshift;

    std::vector<int32_t> multiplier_int32(2);
    arrayAttrToVector(this->m_i8_inputs().getValue(), multiplier_int32);

    for (int32_t i = 0; i < 2; i++ ){
      multiplier[i] = static_cast<int32_t>(multiplier_int32[i]);
      llvm::errs() << "broadcast sub multiplier: " << multiplier[i];
    }
  }

  gaddr_t ga_inputs[2];
  ga_inputs[0] = getPreviousOpAddress(op, 0);
  ga_inputs[1] = getPreviousOpAddress(op, 1);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_int8_broadcast_sub_kernel(
    *backend_ctx, layer_id,
    ga_inputs, ga_output, n, c,
    h, w, bn, bc, bh, bw, do_relu, rshift, multiplier);
  return success();
}

LogicalResult tpu::TG_BF16_BroadcastSubOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n");

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  int64_t input_size, n, c, h, w;
  std::vector<int64_t> shape;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  int64_t bn, bc, bh, bw;
  getTensorShapeAndSize(op->getOperand(1), shape, input_size);
  getNCHW(shape, bn, bc, bh, bw);

  bool do_relu = this->do_relu();

  gaddr_t ga_inputs[2];
  ga_inputs[0] = getPreviousOpAddress(op, 0);
  ga_inputs[1] = getPreviousOpAddress(op, 1);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_bf16_broadcast_sub_kernel(
    *backend_ctx, layer_id,
    ga_inputs, ga_output, n, c,
    h, w, bn, bc, bh, bw, do_relu);

  return success();
}

LogicalResult tpu::TG_INT8_ConcatOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  int axis = this->axis();
  int layer_id = getOpLayerId(op);
  unsigned nInputs = op->getNumOperands();
  std::vector<gaddr_t> ga_inputs(nInputs);
  for (unsigned i = 0; i < nInputs; i++) {
    ga_inputs[i] = getPreviousOpAddress(op, i);
  }
  gaddr_t ga_output = getOpAddress(op);

  // prepare shape info
  std::vector<int32_t> axis_dims;
  for (unsigned i = 0; i < nInputs; i++) {
    std::vector<int64_t> shape = getTensorShape(op->getOperand(i));
    axis_dims.push_back(shape[axis]);
  }
  std::vector<int> output_dim;
  std::vector<int64_t> shape = getTensorShape(this->getResult());
  int output_dim_size = shape.size();
  assert(output_dim_size <= 4);
  for (int i = 0; i < output_dim_size; i++) {
    output_dim.push_back(shape[i]);
  }

  // prepare quant info
  std::vector<int32_t> rshift;
  std::vector<int32_t> m_i8_input;
  const int32_t *p_rshift = nullptr;
  const int32_t *p_m_i8 = nullptr;
  if (this->rshift().hasValue() && this->m_i8_inputs().hasValue()) {
    arrayAttrToVector(this->m_i8_inputs().getValue(), m_i8_input);
    assert(m_i8_input.size() == nInputs);
    arrayAttrToVector(this->rshift().getValue(), rshift);
    assert(rshift.size() == nInputs);
    p_rshift = rshift.data();
    p_m_i8 = m_i8_input.data();
  }

  cvi_backend_tg_concat_kernel(*backend_ctx, layer_id, nInputs,
                               ga_inputs.data(), ga_output, axis_dims.data(),
                               axis, output_dim_size, output_dim.data(),
                               do_relu(), p_rshift, p_m_i8, CVK_FMT_I8);

  return success();
}

LogicalResult tpu::TG_BF16_ConcatOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  int nInputs = op->getNumOperands();
  std::vector<gaddr_t> ga_inputs(nInputs);
  for (int i = 0; i < nInputs; i++) {
    ga_inputs[i] = getPreviousOpAddress(op, i);
  }
  gaddr_t ga_output = getOpAddress(op);
  int axis = this->axis();
  int layer_id = getOpLayerId(op);

  std::vector<int32_t> axis_dims;
  for (int i = 0; i < nInputs; i++) {
    std::vector<int64_t> shape = getTensorShape(op->getOperand(i));
    axis_dims.push_back(shape[axis]);
  }

  std::vector<int> output_dim;
  std::vector<int64_t> shape = getTensorShape(this->getResult());
  int output_dim_size = shape.size();
  assert(output_dim_size <= 4);
  for (int i = 0; i < output_dim_size; i++) {
    output_dim.push_back(shape[i]);
  }

  cvi_backend_tg_concat_kernel(
      *backend_ctx, layer_id, nInputs, ga_inputs.data(),
      ga_output, axis_dims.data(), axis, output_dim_size, output_dim.data(),
      do_relu(), nullptr, nullptr, CVK_FMT_BF16);

  return success();
}

LogicalResult tpu::TG_INT8_CropOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = getOpLayerId(op);
  gaddr_t input_gaddr = getPreviousOpAddress(op);

  gaddr_t output_gaddr = getOpAddress(op);
  std::vector<int64_t> input_shape = getTensorShape(op->getOperand(0));
  std::vector<int64_t> output_shape = getTensorShape(this->getResult());

  // prepare data
  std::vector<int> i_s;
  std::vector<int> o_s;
  std::vector<int> offsets;

  i_s.assign(input_shape.begin(), input_shape.end());
  for (uint32_t i = i_s.size(); i < 4; i++) {
    i_s.push_back(1);
  }
  o_s.assign(output_shape.begin(), output_shape.end());
  for (uint32_t i = o_s.size(); i < 4; i++) {
    o_s.push_back(1);
  }
  arrayAttrToVector(this->crop_offset().getValue(), offsets);
  for (uint32_t i = offsets.size(); i < 4; i++) {
    offsets.push_back(0);
  }

  cvi_backend_tg_crop_kernel(*backend_ctx, layer_id, input_gaddr, output_gaddr,
                             i_s.data(), o_s.data(), offsets.data(),
                             CVK_FMT_I8);

  return success();
}

LogicalResult tpu::TG_BF16_CropOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = mlir::getOpLayerId(op);
  gaddr_t input_gaddr = getPreviousOpAddress(op);

  gaddr_t output_gaddr = getOpAddress(op);
  std::vector<int64_t> input_shape = getTensorShape(op->getOperand(0));
  std::vector<int64_t> output_shape = getTensorShape(this->getResult());

  // prepare data
  std::vector<int> i_s;
  std::vector<int> o_s;
  std::vector<int> offsets;

  i_s.assign(input_shape.begin(), input_shape.end());
  for (uint32_t i = i_s.size(); i < 4; i++) {
    i_s.push_back(1);
  }
  o_s.assign(output_shape.begin(), output_shape.end());
  for (uint32_t i = o_s.size(); i < 4; i++) {
    o_s.push_back(1);
  }
  arrayAttrToVector(this->crop_offset().getValue(), offsets);
  for (uint32_t i = offsets.size(); i < 4; i++) {
    offsets.push_back(0);
  }

  cvi_backend_tg_crop_kernel(*backend_ctx, layer_id, input_gaddr, output_gaddr,
                             i_s.data(), o_s.data(), offsets.data(),
                             CVK_FMT_BF16);

  return success();
}

LogicalResult tpu::TG_INT8_PT_Conv2DOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb, pl, pr, dh, dw, pad_value;
  parseConvParam(param(), false, input(), output(), filter(), n, ic, ih, iw, oc,
                 oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb, pl, pr, dh, dw, is_dw,
                 with_bias, do_relu, pad_value);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  gaddr_t ga_filter = getWeightOpAddress(filter().getDefiningOp());
  gaddr_t ga_bias = GA_INVALID;
  if ( with_bias ) {
    assert(!isTensorNone(pc_info()));
    ga_bias =  getWeightOpAddress(pc_info().getDefiningOp());
  }
  assert(pt_rshift().hasValue());
  int8_t rshift = pt_rshift().getValue();
  int layer_id = getOpLayerId(op);
  bool do_ic_alignment = this->do_ic_alignment().hasValue()
                         ? this->do_ic_alignment().getValue() : false;

  // check if fused with a leakyrelu
  int fused_leakyrelu_pos_rshift = 0;
  int fused_leakyrelu_pos_m_i8 = 0;
  int fused_leakyrelu_neg_rshift = 0;
  int fused_leakyrelu_neg_m_i8 = 0;
  float fused_negative_slope = 0.0f;
  if (this->do_leaky_relu()) {
    int8_t pos_rshift, pos_m_i8, neg_rshift, neg_m_i8;
    float negativeSlope;
    parseLeakyReluParam<tpu::TG_INT8_PT_Conv2DOp>(op, pos_rshift, pos_m_i8,
                          neg_rshift, neg_m_i8, negativeSlope);
    assert(neg_m_i8);

    // TODO: fix the type in backend API
    fused_leakyrelu_pos_rshift = static_cast<int>(pos_rshift);
    fused_leakyrelu_pos_m_i8   = static_cast<int>(pos_m_i8);
    fused_leakyrelu_neg_rshift = static_cast<int>(neg_rshift);
    fused_leakyrelu_neg_m_i8   = static_cast<int>(neg_m_i8);
    fused_negative_slope       = negativeSlope;
    do_relu = true;

    LLVM_DEBUG(llvm::errs() << "  fused leaky relu, pos ("
        << fused_leakyrelu_pos_m_i8 << ", " << fused_leakyrelu_pos_rshift
        << "), neg ("
        << fused_leakyrelu_neg_m_i8 << ", " << fused_leakyrelu_neg_rshift
        << ")\n";);
  }

  cvi_backend_tg_fixed_conv_kernel(
      *backend_ctx,
      layer_id, // layer_id,
      ga_input,  // input_data_gaddr,
      ga_output, // output_data_gaddr,
      ga_filter, // weight_data_gaddr,
      ga_bias,   // bias_data_gaddr,
      n, ic, ih, iw,
      g, // group,
      oc,
      kh, kw,
      dh, dw,
      pt, pb, pl, pr, // pad (t, b, l, r)
      ins_h, ins_w, //ins_h, ins_w
      sh, sw,
      with_bias, // bias_term,
      do_relu ? 1 : 0, // do_activation,
      do_relu ? & fused_negative_slope : nullptr,   // activation_arg,
      fused_leakyrelu_pos_m_i8,           // activation_gt_scale,
      fused_leakyrelu_pos_rshift,         // activation_gt_rshift,
      fused_leakyrelu_neg_m_i8,           // activation_le_scale,
      fused_leakyrelu_neg_rshift,         // activation_le_rshift,
      (int)rshift, // right_shift_width,
      false,     // do_chl_quan
      do_ic_alignment,
      0,         // store_cmpr_act
      0,         // load_cmpr_act
      false,     // do_cmpr_wgt
      0,         // store_cmpr_act_c_step
      0,         // load_cmpr_act_c_step
      0,         // store_cmpr_act_h_step
      0,         // load_cmpr_act_h_step
      pad_value  // pad_value
      );

  return success();
}

LogicalResult tpu::TG_INT8_PC_Conv2DOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb, pl, pr, dh, dw, pad_value;
  parseConvParam(param(), false, input(), output(), filter(), n, ic, ih, iw, oc,
                 oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb, pl, pr, dh, dw, is_dw,
                 with_bias, do_relu, pad_value);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  gaddr_t ga_filter = getWeightOpAddress(filter().getDefiningOp());
  gaddr_t ga_pc_info = getWeightOpAddress(pc_info().getDefiningOp());
  int layer_id = getOpLayerId(op);
  bool do_ic_alignment = this->do_ic_alignment().hasValue()
                            ? this->do_ic_alignment().getValue() : false;

  // check if fused with a leakyrelu
  int fused_leakyrelu_pos_rshift = 0;
  int fused_leakyrelu_pos_m_i8 = 0;
  int fused_leakyrelu_neg_rshift = 0;
  int fused_leakyrelu_neg_m_i8 = 0;
  float fused_negative_slope = 0.0f;

  // in layer group, conv + leaky will be one op
  if (this->do_leaky_relu()) {
      int8_t pos_rshift, pos_m_i8, neg_rshift, neg_m_i8;
      float negativeSlope;
      parseLeakyReluParam<tpu::TG_INT8_PC_Conv2DOp>(
          op, pos_rshift, pos_m_i8, neg_rshift, neg_m_i8, negativeSlope);
      assert(neg_m_i8);

      fused_leakyrelu_pos_rshift = static_cast<int>(pos_rshift);
      fused_leakyrelu_pos_m_i8   = static_cast<int>(pos_m_i8);
      fused_leakyrelu_neg_rshift = static_cast<int>(neg_rshift);
      fused_leakyrelu_neg_m_i8   = static_cast<int>(neg_m_i8);
      fused_negative_slope       = negativeSlope;
      do_relu = true;
  }

  int store_cmpr_act = this->store_compr_act().hasValue() ?
                       this->store_compr_act().getValue() : 0;
  int store_cmpr_act_c_step = 0, store_cmpr_act_h_step = 0;
  if (store_cmpr_act) {
    store_cmpr_act =
        this->store_compr_act_param().getValue().step_size().getInt();
    store_cmpr_act_c_step =
        this->store_compr_act_param().getValue().c_step().getInt();
    store_cmpr_act_h_step =
        this->store_compr_act_param().getValue().h_step().getInt();
  }
  int load_cmpr_act = this->load_compr_act().hasValue() ?
                      this->load_compr_act().getValue() : 0;
  int load_cmpr_act_c_step = 0, load_cmpr_act_h_step = 0;
  if (load_cmpr_act) {
    load_cmpr_act =
        this->load_compr_act_param().getValue().step_size().getInt();
    load_cmpr_act_c_step =
        this->load_compr_act_param().getValue().c_step().getInt();
    load_cmpr_act_h_step =
        this->load_compr_act_param().getValue().h_step().getInt();
  }

  bool do_cmpr_wgt = this->compressed_weight().hasValue() ?
                     this->compressed_weight().getValue() : false;

  // Backend fuse previous scale lut op
  gaddr_t ga_scale_lut = GA_INVALID;
  // auto prevOp = op->getOperand(0).getDefiningOp();
  // if (auto prevTpuOp = llvm::dyn_cast<tpu::TG_INT8_ScaleLutOp>(prevOp)) {
  //   ga_scale_lut = getWeightOpAddress(prevTpuOp.table().getDefiningOp());
  //   ga_input = getPreviousOpAddress(prevOp);
  // }

  cvi_backend_tg_fixed_conv_kernel(
      *backend_ctx,
      layer_id,   // layer_id,
      ga_input,   // input_data_gaddr,
      ga_output,  // output_data_gaddr,
      ga_filter,  // weight_data_gaddr,
      ga_pc_info, // bias_data_gaddr,
      n, ic, ih, iw,
      g,                                  // group,
      oc,
      kh, kw,
      dh, dw,
      pt, pb, pl, pr, // pad (t, b, l, r)
      ins_h, ins_w,                               // ins_h, ins_w
      sh, sw,
      with_bias,                                 // bias_term,
      do_relu ? 1 : 0,                           // do_activation,
      do_relu ? &fused_negative_slope : nullptr, // activation_arg,
      fused_leakyrelu_pos_m_i8,                  // activation_gt_scale,
      fused_leakyrelu_pos_rshift,                // activation_gt_rshift,
      fused_leakyrelu_neg_m_i8,                  // activation_le_scale,
      fused_leakyrelu_neg_rshift,                // activation_le_rshift,
      0,    // (int)rshift[0], //right_shift_width,
      true, // do_chl_quan
      do_ic_alignment,
      store_cmpr_act,
      load_cmpr_act,
      do_cmpr_wgt,
      store_cmpr_act_c_step,
      load_cmpr_act_c_step,
      store_cmpr_act_h_step,
      load_cmpr_act_h_step,
      pad_value, // pad_value
      ga_scale_lut);

  return success();
}

LogicalResult tpu::TG_BF16_Conv2DOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb, pl, pr, dh, dw, pad_value;
  parseConvParam(param(), false, input(), output(), filter(), n, ic, ih, iw, oc,
                 oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb, pl, pr, dh,
                 dw, is_dw, with_bias, do_relu, pad_value);

  std::vector<int32_t> ins;
  arrayAttrToVector(param().ins(), ins);
  ins.resize(2, 0);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  gaddr_t ga_filter = getWeightOpAddress(filter().getDefiningOp());
  gaddr_t ga_bias = GA_INVALID;
  if ( with_bias ) {
    assert(!isTensorNone(pc_info()));
    ga_bias =  getWeightOpAddress(pc_info().getDefiningOp());
  }
  int layer_id = getOpLayerId(op);

  int store_cmpr_act = this->store_compr_act().hasValue() ?
                       this->store_compr_act().getValue() : 0;
  int store_cmpr_act_c_step = 0, store_cmpr_act_h_step;
  if (store_cmpr_act) {
    store_cmpr_act =
        this->store_compr_act_param().getValue().step_size().getInt();
    store_cmpr_act_c_step =
        this->store_compr_act_param().getValue().c_step().getInt();
    store_cmpr_act_h_step =
        this->store_compr_act_param().getValue().h_step().getInt();
  }

  int load_cmpr_act = this->load_compr_act().hasValue() ?
                      this->load_compr_act().getValue() : 0;
  int load_cmpr_act_c_step = 0, load_cmpr_act_h_step;
  if (load_cmpr_act) {
    load_cmpr_act =
        this->load_compr_act_param().getValue().step_size().getInt();
    load_cmpr_act_c_step =
        this->load_compr_act_param().getValue().c_step().getInt();
    load_cmpr_act_h_step =
        this->load_compr_act_param().getValue().h_step().getInt();
  }
  bool do_cmpr_wgt = this->compressed_weight().hasValue() ?
                     this->compressed_weight().getValue() : false;

  cvi_backend_tg_bf16_conv_kernel(
      *backend_ctx,
      layer_id,  // layer_id
      ga_input,
      ga_output,
      ga_filter,
      ga_bias,
      n, ic, ih, iw,
      g, // group
      oc,
      kh, kw,
      dh, dw,
      pt, pb, pl, pr, // pad (t, b, l, r)
      ins_h, ins_w,                               // ins_h, ins_w
      sh, sw,
      with_bias,
      do_relu ? 1 : 0,
      false, // fp32_output
      store_cmpr_act,
      load_cmpr_act,
      do_cmpr_wgt,
      store_cmpr_act_c_step,
      load_cmpr_act_c_step,
      store_cmpr_act_h_step,
      load_cmpr_act_h_step
      );

  return success();
}

LogicalResult tpu::TG_INT8_PT_DeConv2DOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  //CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  //Operation *op = this->getOperation();

  std::string errorMsg = "unsupported tg op " + getOpName().str();
  llvm_unreachable(errorMsg.c_str());
}

LogicalResult tpu::TG_INT8_PC_DeConv2DOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, pt, pb, pl, pr, dh, dw,
      pad_value;
  int no_use0, no_use1;
  parseConvParam(param(), false, input(), output(), filter(), n, ic, ih, iw, oc,
                 oh, ow, g, kh, kw, no_use0, no_use1, sh, sw, pt, pb, pl, pr,
                 dh, dw, is_dw, with_bias, do_relu, pad_value);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  gaddr_t ga_filter = getWeightOpAddress(filter().getDefiningOp());
  gaddr_t ga_pc_info = getWeightOpAddress(pc_info().getDefiningOp());
  int layer_id = getOpLayerId(op);

  if (this->do_leaky_relu()) {
    assert(0);
  }

  int kh_ext = (kh - 1) * dh + 1;
  int kw_ext = (kw - 1) * dw + 1;
  int ins_h = sh - 1;
  int ins_w = sw - 1;
  int pad_t = kh_ext - pt - 1;
  int pad_l = kw_ext - pl - 1;
  int pad_b = oh + pt - (ih - 1) * sh - 1;
  int pad_r = ow + pr - (iw - 1) * sw - 1;
  int stride_h = 1;
  int stride_w = 1;
  bool do_chl_quan = true;

  cvi_backend_tg_fixed_conv_kernel(
      *backend_ctx,
      layer_id,   // layer_id,
      ga_input,   // input_data_gaddr,
      ga_output,  // output_data_gaddr,
      ga_filter,  // weight_data_gaddr,
      ga_pc_info, // bias_data_gaddr,
      n, ic, ih, iw,
      g, // group,
      oc,
      kh, kw,
      dh, dw,
      pad_t, pad_b, pad_l, pad_r,
      ins_h, ins_w,
      stride_h, stride_w,
      with_bias, // bias_term,
      do_relu ? 1 : 0, // do_activation,
      nullptr,   // activation_arg,
      0,         // activation_gt_scale,
      0,         // activation_gt_rshift,
      0,         // activation_le_scale,
      0,         // activation_le_rshift,
      0,         // (int)rshift[0], //right_shift_width,
      do_chl_quan,      // do_chl_quan
      false,
      0,         // store_compr_act
      0,         // load_compr_act
      false,     // compressed_weight
      0,         // store_cmpr_act_c_step
      0,         // load_cmpr_act_c_step
      0,         // store_cmpr_act_h_step
      0          // load_cmpr_act_h_step
      );

  return success();
}

LogicalResult tpu::TG_BF16_DeConv2DOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, pt, pb, pl, pr, dh, dw,
      pad_value;
  int no_use0, no_use1;
  parseConvParam(param(), false, input(), output(), filter(), n, ic, ih, iw, oc,
                 oh, ow, g, kh, kw, no_use0, no_use1, sh, sw, pt, pb, pl, pr,
                 dh, dw, is_dw, with_bias, do_relu, pad_value);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  gaddr_t ga_filter = getWeightOpAddress(filter().getDefiningOp());
  gaddr_t ga_bias = GA_INVALID;
  if ( with_bias ) {
    assert(!isTensorNone(pc_info()));
    ga_bias =  getWeightOpAddress(pc_info().getDefiningOp());
  }
  int layer_id = getOpLayerId(op);

  if (this->do_leaky_relu()) {
    assert(0);
  }

  int kh_ext = (kh - 1) * dh + 1;
  int kw_ext = (kw - 1) * dw + 1;
  int ins_h = sh - 1;
  int ins_w = sw - 1;
  int pad_t = kh_ext - pt - 1;
  int pad_l = kw_ext - pl - 1;
  int pad_b = oh + pt - (ih - 1) * sh - 1;
  int pad_r = ow + pr - (iw - 1) * sw - 1;
  sh = 1;
  sw = 1;

  cvi_backend_tg_bf16_conv_kernel(
      *backend_ctx,
      layer_id,   // layer_id,
      ga_input,   // input_data_gaddr,
      ga_output,  // output_data_gaddr,
      ga_filter,  // weight_data_gaddr,
      ga_bias, // bias_data_gaddr,
      n, ic, ih, iw,
      g, // group,
      oc,
      kh, kw,
      dh, dw,
      pad_t, pad_b, pad_l, pad_r,
      ins_h, ins_w,
      sh, sw,
      with_bias, // bias_term,
      do_relu ? 1 : 0, // do_activation,
      false, // fp32_output
      0,     // store_compr_act
      0,     // load_compr_act
      false, // compr_wgt
      0,     // store_cmpr_act_c_step
      0,     // load_cmpr_act_c_step
      0,     // store_cmpr_act_h_step
      0      // load_cmpr_act_h_step
      );

  return success();
}

LogicalResult tpu::TG_BF16_Conv3DOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  bool is_dw, with_bias, do_relu;
  int n, ic, id, ih, iw, oc, od, oh, ow, g, kd, kh, kw, sd, sh, sw;
  int pd0, pd1, pt, pb, pl, pr, dd, dh, dw;
  parseConv3dParam(param(), false, input(), output(), filter(),
                   n, ic, id, ih, iw,
                   oc, od, oh, ow, g,
                   kd, kh, kw,
                   sd, sh, sw,
                   pd0, pd1, pt, pb, pl, pr,
                   dd, dh, dw,
                   is_dw, with_bias, do_relu);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  gaddr_t ga_filter = getWeightOpAddress(filter().getDefiningOp());
  gaddr_t ga_bias = GA_INVALID;
  if ( with_bias ) {
    assert(!isTensorNone(pc_info()));
    ga_bias =  getWeightOpAddress(pc_info().getDefiningOp());
  }
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_bf16_conv3d_kernel(
      *backend_ctx,
      layer_id,  // layer_id
      ga_input,
      ga_output,
      ga_filter,
      ga_bias,
      n, ic, id, ih, iw,
      oc, od, oh, ow,
      kd, kh, kw,
      dd, dh, dw,
      pd0, pd1, pt, pb, pl, pr,
      sd, sh, sw,
      with_bias,
      do_relu ? 1 : 0
      );

  return success();
}

LogicalResult tpu::TG_INT8_DilateOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = getOpLayerId(op);
  gaddr_t input_gaddr = getPreviousOpAddress(op);

  auto fill_constant = this->fill_constant();
  gaddr_t output_gaddr = getOpAddress(op);

  std::vector<int64_t> input_shape;
  int64_t input_size, in, ic, ih, iw;
  getTensorShapeAndSize(op->getOperand(0), input_shape, input_size);
  getNCHW(input_shape, in, ic, ih, iw);

  std::vector<int64_t> output_shape;
  int64_t output_size, on, oc, oh, ow;
  getTensorShapeAndSize(this->getResult(), output_shape, output_size);
  getNCHW(output_shape, on, oc, oh, ow);

  assert(in == on && "only support dilate h/w");
  assert(ic == oc && "only support dilate h/w");

  // get is dilate activation
  std::vector<int32_t> ins;
  arrayAttrToVector(this->ins().getValue(), ins);

  int ins_w = 0;
  int ins_h = 0;
  if (ins.size()) {
    ins_w = ins[0];
    ins_h = 0;
    if (ins.size() > 1) {
      ins_h = ins[1];
    }

    cvi_backend_tg_fixed_dilate_kernel(*backend_ctx, // ctx,
        layer_id,
        input_gaddr,  // bottom_gaddr,
        output_gaddr, // top_gaddr
        in, ic, ih, iw,
        oh, ow,
        fill_constant,
        ins_h, ins_w,
        CVK_FMT_I8);
  }

  return success();
}

LogicalResult tpu::TG_BF16_DilateOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = getOpLayerId(op);
  gaddr_t input_gaddr = getPreviousOpAddress(op);

  auto fill_constant = this->fill_constant();
  gaddr_t output_gaddr = getOpAddress(op);

  std::vector<int64_t> input_shape;
  int64_t input_size, in, ic, ih, iw;
  getTensorShapeAndSize(op->getOperand(0), input_shape, input_size);
  getNCHW(input_shape, in, ic, ih, iw);

  std::vector<int64_t> output_shape;
  int64_t output_size, on, oc, oh, ow;
  getTensorShapeAndSize(this->getResult(), output_shape, output_size);
  getNCHW(output_shape, on, oc, oh, ow);

  assert(in == on && "only support dilate h/w");
  assert(ic == oc && "only support dilate h/w");

  // get is dilate activation
  std::vector<int32_t> ins;
  arrayAttrToVector(this->ins().getValue(), ins);

  int ins_w = 0;
  int ins_h = 0;
  if (ins.size()) {
    ins_w = ins[0];
    ins_h = 0;
    if (ins.size() > 1) {
      ins_h = ins[1];
    }

    cvi_backend_tg_fixed_dilate_kernel(*backend_ctx, // ctx,
        layer_id,
        input_gaddr,  // bottom_gaddr,
        output_gaddr, // top_gaddr
        in, ic, ih, iw,
        oh, ow,
        fill_constant,
        ins_h, ins_w,
        CVK_FMT_BF16);
  }

  return success();
}

LogicalResult tpu::TG_INT8_EmbeddingOp::codegen(void *ctx) {
  std::string errorMsg = "unsupported tg op " + getOpName().str() + "\n";
  llvm_unreachable(errorMsg.c_str());
  return success();
}

LogicalResult tpu::TG_BF16_EmbeddingOp::codegen(void *ctx) {
  std::string errorMsg = "unsupported tg op " + getOpName().str() + "\n";
  llvm_unreachable(errorMsg.c_str());
  return success();
}

LogicalResult tpu::TG_INT8_EltwiseAddOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  std::vector<int64_t> output_shape;
  int64_t output_size, oh, ow;
  getTensorShapeAndSize(op->getResult(0), output_shape, output_size);
  oh = output_shape[2];
  ow = output_shape[3];
  bool do_relu = this->do_relu();
  bool do_early_stride = this->do_early_stride();
  int32_t early_stride_h = this->early_stride_h();
  int32_t early_stride_w = this->early_stride_w();
  if (do_early_stride) {
    assert(oh == h / early_stride_h);
    assert(ow == w / early_stride_w);
  }

  int32_t input_number = op->getNumOperands();
  auto ga_inputs = new gaddr_t[input_number];
  for(int32_t i = 0; i < input_number; i++){
    ga_inputs[i] = getPreviousOpAddress(op, i);
  }
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  bool do_quant_rescale = false;
  int8_t rshift;
  auto m_i8_input = new int8_t[input_number];
  if (this->rshift().hasValue() && this->m_i8_inputs().hasValue()) {
    do_quant_rescale = true;
    rshift = this->rshift().getValue();

    std::vector<int32_t> m_i8_inputs_array;
    arrayAttrToVector(this->m_i8_inputs().getValue(), m_i8_inputs_array);
    assert(m_i8_inputs_array.size() == op->getNumOperands());
    for (int32_t i = 0; i < input_number; i++ ){
      m_i8_input[i] = static_cast<int8_t>(m_i8_inputs_array[i]);
    }
  }

  // TODO: should change on backend API, rather than doing cast
  int rshift_int;
  auto m_int = new int32_t[input_number];
  if (do_quant_rescale) {
    rshift_int = static_cast<int>(rshift);
    for (int i = 0; i < input_number; i++ ){
      m_int[i] = static_cast<int>(m_i8_input[i]);
    }
  }

  //Asymmetric
  int output_offset = 0;
  std::vector<int32_t> inputs_offset;

  if (this->output_offset().hasValue() && this->input_offset().hasValue()) {
    output_offset = this->output_offset().getValue();
    arrayAttrToVector(this->input_offset().getValue(), inputs_offset);
    assert(inputs_offset.size() == op->getNumOperands());
  }

  std::vector<int>coeffs(input_number, 1);

  int store_cmpr_act = this->store_compr_act().hasValue() ?
                       this->store_compr_act().getValue() : 0;
  int store_cmpr_act_c_step = 0;
  if (store_cmpr_act) {
    assert(this->store_compr_act_param().hasValue());
    store_cmpr_act =
        this->store_compr_act_param().getValue().step_size().getInt();
    store_cmpr_act_c_step =
        this->store_compr_act_param().getValue().c_step().getInt();
  }

  int load_cmpr_act = this->load_compr_act().hasValue() ?
                      this->load_compr_act().getValue() : 0;
  int load_cmpr_act_c_step = 0;
  if (load_cmpr_act) {
    assert(this->load_compr_act_param().hasValue());
    load_cmpr_act =
        this->load_compr_act_param().getValue().step_size().getInt();
    load_cmpr_act_c_step =
        this->load_compr_act_param().getValue().c_step().getInt();
  }

  cvi_backend_tg_fixed_eltwise_add_kernel(
      *backend_ctx, layer_id,
      ga_inputs, ga_output,
      input_number, n, c, h, w,
      do_relu, do_early_stride,
      early_stride_h, early_stride_w,
      do_quant_rescale ? rshift_int : 0,
      do_quant_rescale ? m_int : nullptr,
      coeffs.data(),
      inputs_offset.data(), output_offset,
      store_cmpr_act, load_cmpr_act,
      store_cmpr_act_c_step, load_cmpr_act_c_step);

  delete[] m_i8_input;
  delete[] m_int;
  delete[] ga_inputs;
  return success();
}

LogicalResult tpu::TG_INT8_EltwiseMaxOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  std::vector<int64_t> output_shape;
  int64_t output_size, oh, ow;
  getTensorShapeAndSize(op->getResult(0), output_shape, output_size);
  oh = output_shape[2];
  ow = output_shape[3];
  bool do_relu = this->do_relu();
  bool do_early_stride = this->do_early_stride();
  int32_t early_stride_h = this->early_stride_h();
  int32_t early_stride_w = this->early_stride_w();
  if (do_early_stride) {
    assert(oh == h / early_stride_h);
    assert(ow == w / early_stride_w);
  }

  int32_t input_number = op->getNumOperands();
  auto ga_inputs = new gaddr_t[input_number];
  for(int32_t i = 0; i < input_number; i++){
    ga_inputs[i] = getPreviousOpAddress(op, i);
  }
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  bool do_quant_rescale = false;
  int8_t rshift;
  auto m_i8_input = new int8_t [input_number];
  if (this->rshift().hasValue() && this->m_i8_inputs().hasValue()) {
    do_quant_rescale = true;
    rshift = this->rshift().getValue();

    std::vector<int32_t> m_i8_inputs_array;
    arrayAttrToVector(this->m_i8_inputs().getValue(), m_i8_inputs_array);
    assert(m_i8_inputs_array.size() == op->getNumOperands());
    for (int i = 0; i < input_number; i++ ){
      m_i8_input[i] = static_cast<int8_t>(m_i8_inputs_array[i]);
    }
  }

  // TODO: should change on backend API, rather than doing cast
  int rshift_int;
  auto m_int = new int[input_number];
  if (do_quant_rescale) {
    rshift_int = static_cast<int>(rshift);
    for (int i = 0; i < input_number; i++ ){
    m_int[i] = static_cast<int>(m_i8_input[i]);
    }
  }
  const int coeffs[2] = {1, 1};
  cvi_backend_tg_fixed_eltwise_max_kernel(
      *backend_ctx, layer_id,
      ga_inputs, ga_output,
      2, n, c, h, w,
      do_relu, do_early_stride,
      early_stride_h, early_stride_w,
      do_quant_rescale ? rshift_int : 0,
      do_quant_rescale ? m_int : nullptr,
      coeffs);

  delete[] ga_inputs;
  delete[] m_i8_input;
  delete[] m_int;
  return success();
}

LogicalResult tpu::TG_INT8_EltwiseMinOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  std::vector<int64_t> output_shape;
  int64_t output_size, oh, ow;
  getTensorShapeAndSize(op->getResult(0), output_shape, output_size);
  oh = output_shape[2];
  ow = output_shape[3];
  bool do_relu = this->do_relu();
  bool do_early_stride = this->do_early_stride();
  int32_t early_stride_h = this->early_stride_h();
  int32_t early_stride_w = this->early_stride_w();
  if (do_early_stride) {
    assert(oh == h / early_stride_h);
    assert(ow == w / early_stride_w);
  }

  int32_t input_number = op->getNumOperands();
  auto ga_inputs = new gaddr_t[input_number];
  for(int32_t i = 0; i < input_number; i++){
    ga_inputs[i] = getPreviousOpAddress(op, i);
  }
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  bool do_quant_rescale = false;
  int8_t rshift;
  auto m_i8_input = new int8_t[input_number];
  if (this->rshift().hasValue() && this->m_i8_inputs().hasValue()) {
    do_quant_rescale = true;
    rshift = this->rshift().getValue();

    std::vector<int32_t> m_i8_inputs_array;
    arrayAttrToVector(this->m_i8_inputs().getValue(), m_i8_inputs_array);
    assert(m_i8_inputs_array.size() == op->getNumOperands());
    for (int32_t i = 0; i < input_number; i++ ){
      m_i8_input[i] = static_cast<int8_t>(m_i8_inputs_array[i]);
    }
  }

  // TODO: should change on backend API, rather than doing cast
  int rshift_int;
  auto m_int = new int32_t[input_number];
  if (do_quant_rescale) {
    rshift_int = static_cast<int>(rshift);
    for (int32_t i = 0; i < input_number; i++ ){
      m_int[i] = static_cast<int>(m_i8_input[i]);
    }
  }
  const int coeffs[2] = {1, 1};
  cvi_backend_tg_fixed_eltwise_min_kernel(
      *backend_ctx, layer_id,
      ga_inputs, ga_output,
      2, n, c, h, w,
      do_relu, do_early_stride,
      early_stride_h, early_stride_w,
      do_quant_rescale ? rshift_int : 0,
      do_quant_rescale ? m_int : nullptr,
      coeffs);

  delete[] ga_inputs;
  delete[] m_i8_input;
  delete[] m_int;
  return success();
}

LogicalResult tpu::TG_INT8_EltwiseMulOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  std::vector<int64_t> output_shape;
  int64_t output_size;
  getTensorShapeAndSize(op->getResult(0), output_shape, output_size);
  bool do_relu = this->do_relu();
  assert(!do_early_stride());

  gaddr_t ga_inputs[2];

  for (int i = 0; i < 2; i++) {
    auto defOp = op->getOperand(i).getDefiningOp();
    if (isa<tpu::LoadWeightOp>(defOp)) {
      ga_inputs[i] = getWeightOpAddress(defOp);
    } else {
      ga_inputs[i] = getOpAddress(defOp);
    }
  }

  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  assert(this->rshift().hasValue());
  int8_t rshift = this->rshift().getValue();
  assert(this->m_i32_output().hasValue());
  int32_t m_i32_output = this->m_i32_output().getValue();

  // TODO: should change on backend API, rather than doing cast
  int rshift_int = static_cast<int>(rshift);
  int32_t m_int = static_cast<int32_t>(m_i32_output);
  const int coeffs[2] = {1, 1};
  cvi_backend_tg_fixed_eltwise_mul_kernel(
      *backend_ctx, layer_id,
      ga_inputs, ga_output,
      2, n, c, h, w,
      do_relu, false,
      1, 1,
      rshift_int,
      &m_int,
      coeffs);

  return success();
}

LogicalResult tpu::TG_BF16_EltwiseAddOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  std::vector<int64_t> output_shape;
  int64_t output_size, oh, ow;
  getTensorShapeAndSize(op->getResult(0), output_shape, output_size);
  oh = output_shape[2];
  ow = output_shape[3];
  bool do_relu = this->do_relu();
  bool do_early_stride = this->do_early_stride();
  int32_t early_stride_h = this->early_stride_h();
  int32_t early_stride_w = this->early_stride_w();
  if (do_early_stride) {
    assert(oh == h / early_stride_h);
    assert(ow == w / early_stride_w);
  }
  std::vector<float> coeffs;
  arrayAttrToVector(this->coeff().getValue(), coeffs);

  int32_t input_number = op->getNumOperands();
  auto ga_inputs = new gaddr_t[input_number];
  for (int i = 0; i < input_number; i++) {
    ga_inputs[i] = getPreviousOpAddress(op, i);
  }
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  int store_cmpr_act = this->store_compr_act().hasValue() ?
                       this->store_compr_act().getValue() : 0;
  int store_cmpr_act_c_step = 0;
  if (store_cmpr_act) {
    assert(this->store_compr_act_param().hasValue());
    store_cmpr_act =
        this->store_compr_act_param().getValue().step_size().getInt();
    store_cmpr_act_c_step =
        this->store_compr_act_param().getValue().c_step().getInt();
  }

  int load_cmpr_act = this->load_compr_act().hasValue() ?
                      this->load_compr_act().getValue() : 0;
  int load_cmpr_act_c_step = 0;
  if (load_cmpr_act) {
    assert(this->load_compr_act_param().hasValue());
    load_cmpr_act =
        this->load_compr_act_param().getValue().step_size().getInt();
    load_cmpr_act_c_step =
        this->load_compr_act_param().getValue().c_step().getInt();
  }

  cvi_backend_tg_bf16_eltwise_add_kernel(
      *backend_ctx,
      layer_id,     // layer_id
      ga_inputs,    // gaddr_t ga_input[]
      ga_output,    // gaddr_t ga_output
      input_number,            // int input_size
      n, c, h, w,
      do_relu,      // bool do_relu
      do_early_stride, early_stride_h, early_stride_w,
      coeffs.data(),
      store_cmpr_act, load_cmpr_act,
      store_cmpr_act_c_step, load_cmpr_act_c_step);

  delete[] ga_inputs;
  return success();
}

LogicalResult tpu::TG_BF16_EltwiseMaxOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  std::vector<int64_t> output_shape;
  int64_t output_size, oh, ow;
  getTensorShapeAndSize(op->getResult(0), output_shape, output_size);
  oh = output_shape[2];
  ow = output_shape[3];
  bool do_relu = this->do_relu();
  bool do_early_stride = this->do_early_stride();
  int32_t early_stride_h = this->early_stride_h();
  int32_t early_stride_w = this->early_stride_w();
  if (do_early_stride) {
    assert(oh == h / early_stride_h);
    assert(ow == w / early_stride_w);
  }

  int32_t input_number = op->getNumOperands();
  auto ga_inputs = new gaddr_t[input_number];
  for(int32_t i = 0; i < input_number; i++){
    ga_inputs[i] = getPreviousOpAddress(op, i);
  }
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  const float coeffs[2] = {1, 1};
  cvi_backend_tg_bf16_eltwise_max_kernel(
      *backend_ctx, layer_id,
      ga_inputs, ga_output,
      2, n, c, h, w,
      do_relu, do_early_stride,
      early_stride_h, early_stride_w,
      coeffs);

  delete[] ga_inputs;
  return success();
}

LogicalResult tpu::TG_BF16_EltwiseMinOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  std::vector<int64_t> output_shape;
  int64_t output_size, oh, ow;
  getTensorShapeAndSize(op->getResult(0), output_shape, output_size);
  oh = output_shape[2];
  ow = output_shape[3];
  bool do_relu = this->do_relu();
  bool do_early_stride = this->do_early_stride();
  int32_t early_stride_h = this->early_stride_h();
  int32_t early_stride_w = this->early_stride_w();
  if (do_early_stride) {
    assert(oh == h / early_stride_h);
    assert(ow == w / early_stride_w);
  }

  int32_t input_number = op->getNumOperands();
  auto ga_inputs = new gaddr_t[input_number];
  for(int32_t i = 0; i < input_number; i++){
    ga_inputs[i] = getPreviousOpAddress(op, i);
  }
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  const float coeffs[2] = {1, 1};
  cvi_backend_tg_bf16_eltwise_min_kernel(
      *backend_ctx, layer_id,
      ga_inputs, ga_output,
      2, n, c, h, w,
      do_relu, do_early_stride,
      early_stride_h, early_stride_w,
      coeffs);

  delete[] ga_inputs;
  return success();
}

LogicalResult tpu::TG_BF16_EltwiseMulOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  std::vector<int64_t> output_shape;
  int64_t output_size;
  getTensorShapeAndSize(op->getResult(0), output_shape, output_size);
  bool do_relu = this->do_relu();
  assert(!do_early_stride());

  gaddr_t ga_inputs[2];
  ga_inputs[0] = getPreviousOpAddress(op, 0);
  ga_inputs[1] = getPreviousOpAddress(op, 1);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  const float coeffs[2] = {1.0, 1.0};

  cvi_backend_tg_bf16_eltwise_mul_kernel(
      *backend_ctx,
      layer_id,     // layer_id
      ga_inputs,    // gaddr_t ga_input[]
      ga_output,    // gaddr_t ga_output
      2,            // int input_size
      n, c, h, w,
      do_relu,      // bool do_relu
      false, 1, 1,
      coeffs);

  return success();
}

LogicalResult tpu::TG_INT8_FullyConnectedOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  int m, k, n;
  parseFullyConnectedParam(input(), output(), filter(), m, k, n);
  bool do_relu = this->do_relu();
  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  auto rhs = filter().getDefiningOp();
  gaddr_t ga_filter;
  if (isa<tpu::LoadWeightOp>(rhs)) {
    ga_filter = getWeightOpAddress(rhs);
  }
  else {
    ga_filter = getOpAddress(rhs);
  }

  gaddr_t ga_bias = GA_INVALID;
  bool with_bias = false;
  if ( !isTensorNone(bias()) ) {
    ga_bias = getWeightOpAddress(bias().getDefiningOp());
    with_bias = true;
  }
  int layer_id = getOpLayerId(op);

  int8_t rshift_int8 = rshift().getValue();
  int32_t multiplier = this->mutliplier().getValue();
  int rshift = static_cast<int>(rshift_int8);

  auto fcOp = dyn_cast<tpu::TG_INT8_FullyConnectedOp>(op);
  std::vector<int> compr_weight_poss;
  if (fcOp.compr_weight_poss().hasValue())
    arrayAttrToVector(fcOp.compr_weight_poss().getValue(), compr_weight_poss);
  else {
    if (fcOp.compressed_weight().hasValue() &&
        fcOp.compressed_weight().getValue()) {
      llvm::errs() << "  compressed weight enabled, but no poss\n";
      return failure();
    }
  }

  cvi_backend_tg_fixed_fc_kernel(
      *backend_ctx, layer_id, ga_input, ga_filter, ga_bias, ga_output, m, k, n,
      with_bias, do_relu, rshift, multiplier, compr_weight_poss);

  return success();
}

LogicalResult tpu::TG_BF16_FullyConnectedOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  int m, k, n;
  parseFullyConnectedParam(input(), output(), filter(), m, k, n);
  bool do_relu = this->do_relu();
  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  gaddr_t ga_filter;
  auto rhs = filter().getDefiningOp();
  if (isa<tpu::LoadWeightOp>(rhs)) {
    ga_filter = getWeightOpAddress(rhs);
  }
  else {
    ga_filter = getOpAddress(rhs);
  }
  gaddr_t ga_bias = GA_INVALID;
  bool with_bias = false;
  if ( !isTensorNone(bias()) ) {
    ga_bias = getWeightOpAddress(bias().getDefiningOp());
    with_bias = true;
  }
  int layer_id = getOpLayerId(op);

  auto fcOp = dyn_cast<tpu::TG_BF16_FullyConnectedOp>(op);
  std::vector<int> compr_weight_poss;
  if (fcOp.compr_weight_poss().hasValue())
    arrayAttrToVector(fcOp.compr_weight_poss().getValue(), compr_weight_poss);
  else {
    if (fcOp.compressed_weight().hasValue() &&
        fcOp.compressed_weight().getValue()) {
      llvm::errs() << "  compressed weight enabled, but no poss\n";
      return failure();
    }
  }

  cvi_backend_tg_bf16_fc_kernel(*backend_ctx, layer_id, ga_input, ga_filter,
                                ga_bias, ga_output, m, k, n, with_bias, do_relu,
                                compr_weight_poss);

  return success();
}

LogicalResult tpu::TG_INT8_GenericTpuOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = getOpLayerId(op);
  auto op_name = operation_name().str();
  auto operandShapes = getOperandShapes(op);
  auto resultShape = getTensorShape(getResult());

  std::vector<uint64_t> operandGaddrs;
  for (auto operand : op->getOperands()) {
    auto addr = getOpAddress(operand.getDefiningOp());
    operandGaddrs.push_back(addr);
  }
  cvi::OpParam param;
  convertAttributesToOpParam(this->param(), param);

  cvi::CustomOpPlugin *plugin = cvi::CustomOpPlugin::load();
  assert(plugin);
  plugin->int8CodeGen(op_name.c_str(), param, cvi_backend_get_cvk_ctx(*backend_ctx),
                      operandShapes, operandGaddrs, resultShape,
                      getOpAddress(op), layer_id);
  return success();
}

LogicalResult tpu::TG_BF16_GenericTpuOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = getOpLayerId(op);
  auto op_name = operation_name().str();
  auto operandShapes = getOperandShapes(op);
  auto resultShape = getTensorShape(getResult());

  std::vector<uint64_t> operandGaddrs;
  for (auto operand : op->getOperands()) {
    auto addr = getOpAddress(operand.getDefiningOp());
    operandGaddrs.push_back(addr);
  }
  cvi::OpParam param;
  convertAttributesToOpParam(this->param(), param);

  cvi::CustomOpPlugin *plugin = cvi::CustomOpPlugin::load();
  assert(plugin);
  plugin->bf16CodeGen(op_name.c_str(), param, cvi_backend_get_cvk_ctx(*backend_ctx),
                      operandShapes, operandGaddrs, resultShape,
                      getOpAddress(op), layer_id);
  return success();
}

LogicalResult tpu::TG_INT8_InterpOp::codegen(void *ctx) {
  /*
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = getOpLayerId(op);
  */
  std::string errorMsg = "unsupported tg op " + getOpName().str() + "\n";
  llvm_unreachable(errorMsg.c_str());
  return success();
}

LogicalResult tpu::TG_BF16_InterpOp::codegen(void *ctx) {
  /*
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = getOpLayerId(op);
  */
  std::string errorMsg = "unsupported tg op " + getOpName().str() + "\n";
  llvm_unreachable(errorMsg.c_str());
  return success();
}

LogicalResult tpu::TG_INT8_LeakyReluOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  int8_t pos_rshift, pos_m_i8, neg_rshift, neg_m_i8;
  float negativeSlope;
  parseLeakyReluParam<tpu::TG_INT8_LeakyReluOp>(
      op, pos_rshift, pos_m_i8, neg_rshift, neg_m_i8, negativeSlope);
  assert(neg_m_i8);

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);
  int output_offset = 0;
  int input_offset = 0;
  if (this->output_offset().hasValue() && this->input_offset().hasValue()) {
    output_offset = this->output_offset().getValue();
    input_offset = this->input_offset().getValue();
  }

  cvi_backend_tg_fixed_leakyrelu_kernel(
    *backend_ctx,         // ctx
    layer_id,             // layer_id
    ga_input,             // input_gaddr
    ga_output,            // output_gaddr
    n,                    // input_n
    c,                    // input_c
    h,                    // input_h
    w,                    // input_w
    pos_rshift,           // GT_right_shift_width
    neg_rshift,           // LE_right_shift_width
    pos_m_i8,             // GT_scale
    neg_m_i8,              // LE_scale
    input_offset,                    // input_offset
    output_offset                    // output_offset
  );

  return success();
}

LogicalResult tpu::TG_BF16_LeakyReluOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);
  float ga_negative_slope = this->negative_slope().getValue().convertToFloat();

  cvi_backend_tg_bf16_leakyrelu_kernel(
    *backend_ctx,        // ctx
    layer_id,            // layer_id,
    ga_input,            // input_gaddr
    ga_output,           // output_gaddr
    ga_negative_slope,   // ga_negative_slope
    n,                   // input_n
    c,                   // input_c
    h,                   // input_h
    w                   // input_w
  );

  return success();
}

LogicalResult tpu::TG_INT8_LrnOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  gaddr_t ga_power_lut = getWeightOpAddress(power_lut().getDefiningOp());
  gaddr_t ga_sqr_lut = getWeightOpAddress(sqr_lut().getDefiningOp());
  int layer_id = getOpLayerId(op);
  cvi_backend_tg_fixed_lrn_kernel(
      *backend_ctx, layer_id, input_gaddr, output_gaddr,
      ga_sqr_lut, ga_power_lut, n, c, h, w,
      local_size(), sum_rshift(),
      lrn_rshift(), quant_data0(),
      quant_data1());
  return success();
}

LogicalResult tpu::TG_BF16_LrnOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  gaddr_t exp_gaddr = getWeightOpAddress(sqr_lut().getDefiningOp());
  gaddr_t matissa_gaddr = getWeightOpAddress(power_lut().getDefiningOp());
  int local_size = this->local_size();
  float alpha = this->alpha().convertToFloat();
  float k = this->k().convertToFloat();
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_bf16_lrn_kernel(
      *backend_ctx, layer_id, input_gaddr, output_gaddr,
      exp_gaddr, matissa_gaddr, n, c, h, w,
      local_size, alpha, k);

  return success();
}

LogicalResult tpu::TG_INT8_LutOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  gaddr_t y0_table_gaddr = getWeightOpAddress(table().getDefiningOp());
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_lut_kernel(*backend_ctx,
                             layer_id, // layer_id,
                             input_gaddr, output_gaddr, y0_table_gaddr, n, c, h,
                             w, CVK_FMT_I8);

  return success();
}

LogicalResult tpu::TG_INT8_PoolMaskOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_pool_mask_kernel(*backend_ctx,
                                  layer_id, // layer_id,
                                  input_gaddr, output_gaddr, n, c, h, w,
                                  scale(), CVK_FMT_I8);

  return success();
}

LogicalResult tpu::TG_BF16_PoolMaskOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_pool_mask_kernel(*backend_ctx,
                                  layer_id, // layer_id,
                                  input_gaddr, output_gaddr, n, c, h, w,
                                  scale(), CVK_FMT_BF16);

  return success();
}

LogicalResult tpu::TG_INT8_ScaleLutOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  gaddr_t table_gaddr = getWeightOpAddress(table().getDefiningOp());
  int layer_id = getOpLayerId(op);

  // backend tg conv fusion only
  // if (op->getResult(0).hasOneUse()) {
  //   for (auto &use : op->getResult(0).getUses()) {
  //     auto useOp = use.getOwner();
  //     if (llvm::dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(useOp)) {
  //       LLVM_DEBUG(llvm::dbgs() << "  fused to conv\n";);
  //       return success();
  //     }
  //   }
  // }

  cvi_backend_tg_scale_lut_kernel(*backend_ctx,
                                  layer_id, // layer_id,
                                  input_gaddr, output_gaddr, table_gaddr, n, c,
                                  h, w, CVK_FMT_I8);

  return success();
}

LogicalResult tpu::TG_BF16_GruOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t size, seq_len, batch_size, input_size, garbage;
  getTensorShapeAndSize(op->getOperand(0), shape, size);
  getNCHW(shape, seq_len, batch_size, input_size, garbage);
  bool only_last = false;
  int64_t seq_len2, num_dir, batch_size2, hidden_size;
  getTensorShapeAndSize(this->getResult(), shape, size);
  if (shape.size() == 4) {
    getNCHW(shape, seq_len2, num_dir, batch_size2, hidden_size);
    assert(seq_len == seq_len2);
  } else {
    getNCHW(shape, num_dir, batch_size2, hidden_size, garbage);
    only_last = true;
  }
  assert(batch_size == batch_size2);
  assert(input_size == num_dir * 3 * hidden_size);

  bool with_bias = (!isTensorNone(bias()));
  gaddr_t ga_bias = GA_INVALID;
  if ( with_bias ) {
    ga_bias =  getWeightOpAddress(bias().getDefiningOp());
  }
  bool with_h0 = (!isTensorNone(initial_h()));
  gaddr_t initial_h_gaddr = GA_INVALID;
  if (with_h0) {
    auto h_op = initial_h().getDefiningOp();
    auto cast_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(h_op);
    if (cast_op) {
      initial_h_gaddr = getWeightOpAddress(h_op);
    } else {
      initial_h_gaddr = getOpAddress(h_op);
    }
  }

  bool is_linear_before_reset = this->linear_before_reset();
  bool is_bidirectional = this->bidirectional();

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  gaddr_t recurrence_gaddr = getWeightOpAddress(recurrence().getDefiningOp());
  gaddr_t sigmoid_table_data_lut_gaddr = getWeightOpAddress(sigmoid_table().getDefiningOp());
  gaddr_t sigmoid_slope_table_data_lut_gaddr = getWeightOpAddress(sigmoid_slope_table().getDefiningOp());
  gaddr_t tanh_table_data_lut_gaddr = getWeightOpAddress(tanh_table().getDefiningOp());
  gaddr_t tanh_slope_table_data_lut_gaddr = getWeightOpAddress(tanh_slope_table().getDefiningOp());
  int layer_id = getOpLayerId(op);

  LLVM_DEBUG(llvm::errs() << "input_gaddr: " << input_gaddr << "\n"
                          << "recurrence_gaddr: " << recurrence_gaddr << "\n"
                          << "ga_bias: " << ga_bias << "\n"
                          << "initial_h_gaddr: " << initial_h_gaddr << "\n"
                          << "sigmoid_table_data_lut_gaddr: " << sigmoid_table_data_lut_gaddr << "\n"
                          << "sigmoid_slope_table_data_lut_gaddr: " << sigmoid_slope_table_data_lut_gaddr << "\n"
                          << "tanh_table_data_lut_gaddr: " << tanh_table_data_lut_gaddr << "\n"
                          << "tanh_slope_table_data_lut_gaddr: " << tanh_slope_table_data_lut_gaddr << "\n"
                          << "output_gaddr: " << output_gaddr << "\n"
                          << "seq_len: " << seq_len << "\n"
                          << "with_bias: " << with_bias << "\n"
                          << "is_linear_before_reset: " << is_linear_before_reset << "\n"
                          << "is_bidirectional: " << is_bidirectional << "\n"
                          << "\n";);

  cvi_backend_tg_bf16_gru_kernel(*backend_ctx, layer_id,
                  input_gaddr, recurrence_gaddr,
                  ga_bias, initial_h_gaddr,
                  sigmoid_table_data_lut_gaddr, sigmoid_slope_table_data_lut_gaddr,
                  tanh_table_data_lut_gaddr, tanh_slope_table_data_lut_gaddr,
                  output_gaddr,
                  seq_len, num_dir, batch_size, hidden_size,
                  with_bias, with_h0, is_linear_before_reset, is_bidirectional, only_last);
  return success();
}

LogicalResult tpu::TG_INT8_GruOp::codegen(void *ctx) {
  assert(0);
  return success();
}

LogicalResult tpu::TG_BF16_LayerNormOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  float eps = this->eps().convertToFloat();

  std::vector<int32_t> ln_shape;
  int64_t tensorSize = getTensorSize(op->getOperand(0));
  arrayAttrToVector(normalized_shape(), ln_shape);
  int normalized_size =
      std::accumulate(ln_shape.begin(), ln_shape.end(), 1, std::multiplies<>());
  assert(tensorSize % normalized_size == 0);
  int batch_size = tensorSize / normalized_size;

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  gaddr_t ga_table = getWeightOpAddress(table().getDefiningOp());
  gaddr_t ga_mantissa_table =
      getWeightOpAddress(mantissa_table().getDefiningOp());
  int layer_id = getOpLayerId(op);
  gaddr_t ga_scale = GA_INVALID, ga_bias = GA_INVALID;
  bool affine = false;
  if (false == isTensorNone(scale()) && false == isTensorNone(bias())) {
    ga_scale = getWeightOpAddress(scale().getDefiningOp());
    ga_bias = getWeightOpAddress(bias().getDefiningOp());
    affine = true;
  }

  cvi_backend_tg_bf16_layernorm_kernel(
      *backend_ctx, layer_id, input_gaddr, ga_table, ga_mantissa_table,
      ga_scale, ga_bias, output_gaddr, batch_size, normalized_size, eps, affine);
  return success();
}

LogicalResult tpu::TG_BF16_LstmOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t size, seq_len, batch_size, input_size, garbage;
  getTensorShapeAndSize(op->getOperand(0), shape, size);
  getNCHW(shape, seq_len, batch_size, input_size, garbage);
  int64_t seq_len2, num_dir, batch_size2, hidden_size;
  getTensorShapeAndSize(this->getResult(), shape, size);
  assert(shape.size() == 4);
  getNCHW(shape, seq_len2, num_dir, batch_size2, hidden_size);
  assert(seq_len == seq_len2);
  assert(batch_size == batch_size2);
  assert(input_size == num_dir * 4 * hidden_size);

  bool with_bias = (!isTensorNone(bias()));
  gaddr_t ga_bias = GA_INVALID;
  if (with_bias) {
    ga_bias = getWeightOpAddress(bias().getDefiningOp());
  }
  bool with_h0 = (!isTensorNone(initial_h()));
  gaddr_t initial_h_gaddr = GA_INVALID;
  if (with_h0) {
    auto h_op = initial_h().getDefiningOp();
    auto cast_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(h_op);
    if (cast_op) {
      initial_h_gaddr = getWeightOpAddress(h_op);
    } else {
      initial_h_gaddr = getOpAddress(h_op);
    }
  }
  bool with_c0 = (!isTensorNone(initial_c()));
  gaddr_t initial_c_gaddr = GA_INVALID;
  if (with_c0) {
    auto c_op = initial_c().getDefiningOp();
    auto cast_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(c_op);
    if (cast_op) {
      initial_c_gaddr = getWeightOpAddress(c_op);
    } else {
      initial_c_gaddr = getOpAddress(c_op);
    }
  }

  bool is_bidirectional = this->bidirectional();
  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  gaddr_t recurrence_gaddr = getWeightOpAddress(recurrence().getDefiningOp());
  gaddr_t sigmoid_table_data_lut_gaddr =
      getWeightOpAddress(sigmoid_table().getDefiningOp());
  gaddr_t sigmoid_slope_table_data_lut_gaddr =
      getWeightOpAddress(sigmoid_slope_table().getDefiningOp());
  gaddr_t tanh_table_data_lut_gaddr =
      getWeightOpAddress(tanh_table().getDefiningOp());
  gaddr_t tanh_slope_table_data_lut_gaddr =
      getWeightOpAddress(tanh_slope_table().getDefiningOp());
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_bf16_lstm_kernel(
      *backend_ctx, layer_id, input_gaddr, recurrence_gaddr, ga_bias,
      initial_h_gaddr, initial_c_gaddr, sigmoid_table_data_lut_gaddr,
      sigmoid_slope_table_data_lut_gaddr, tanh_table_data_lut_gaddr,
      tanh_slope_table_data_lut_gaddr, output_gaddr, seq_len, num_dir,
      batch_size, hidden_size, with_bias, with_h0, with_c0, is_bidirectional);
  return success();
}

LogicalResult tpu::TG_INT8_LstmOp::codegen(void *ctx) {
  assert(0);
  return success();
}

LogicalResult tpu::TG_INT8_SoftmaxOp::codegen(void *ctx) {
  assert(0);
  return success();
}

LogicalResult tpu::TG_BF16_SoftmaxOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  int axis = this->axis();

  std::vector<int64_t> shape = getTensorShape(op->getOperand(0));
  int dimension = shape.size();

  int layer_id = getOpLayerId(op);
  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  gaddr_t exponential_table_data_lut_gaddr =
      getWeightOpAddress(exponential_table().getDefiningOp());
  gaddr_t exponential_slope_table_data_lut_gaddr =
      getWeightOpAddress(exponential_slope_table().getDefiningOp());
  gaddr_t reciprocal_table_data_lut_gaddr =
      getWeightOpAddress(reciprocal_table().getDefiningOp());
  gaddr_t reciprocal_mantissa_table_data_lut_gaddr =
      getWeightOpAddress(reciprocal_mantissa_table().getDefiningOp());

  LLVM_DEBUG(llvm::errs() << "input_gaddr: " << input_gaddr << "\n"
                          << "exponential_table_data_lut_gaddr: "
                          << exponential_table_data_lut_gaddr << "\n"
                          << "exponential_slope_table_data_lut_gaddr: "
                          << exponential_slope_table_data_lut_gaddr << "\n"
                          << "reciprocal_table_data_lut_gaddr: "
                          << reciprocal_table_data_lut_gaddr << "\n"
                          << "reciprocal_mantissa_table_data_lut_gaddr: "
                          << reciprocal_mantissa_table_data_lut_gaddr << "\n"
                          << "output_gaddr: " << output_gaddr << "\n"
                          << "\n";);

  cvi_backend_tg_bf16_softmax_kernel(
      *backend_ctx, layer_id, input_gaddr, exponential_table_data_lut_gaddr,
      exponential_slope_table_data_lut_gaddr, reciprocal_table_data_lut_gaddr,
      reciprocal_mantissa_table_data_lut_gaddr, output_gaddr, shape.data(),
      axis, dimension);
  return success();
}

LogicalResult tpu::TG_BF16_SquareOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_bf16_square_kernel(
      *backend_ctx,
      layer_id,     // layer_id
      ga_input,    // gaddr_t ga_input[]
      ga_output,    // gaddr_t ga_output
      n, c, h, w,
      false);
  return success();
}

LogicalResult tpu::TG_BF16_QuadraticSumOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  auto shape = getTensorShape(op->getOperand(0));
  int n = shape[0];
  int c = shape[1];
  int h = shape[2];
  int w = shape[3];

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_bf16_conv_kernel(
      *backend_ctx,
      layer_id,  // layer_id
      ga_input,
      ga_output,
      ga_input,
      GA_INVALID,
      n, c, h, w,
      c, // group
      c,
      h, w,
      1, 1,
      0, 0, 0, 0,
      0, 0,
      h, w,
      false,
      false,
      this->high_precision().getValue(),
      0,     // store_compr_act
      0,     // load_compr_act
      false, // compr_wgt
      0,     // store_cmpr_act_c_step
      0,     // load_cmpr_act_c_step
      0,     // store_cmpr_act_h_step
      0      // load_cmpr_act_h_step
      );

  return success();
}

LogicalResult tpu::TG_INT8_MatMulOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  int batch_high = 0, batch_low = 0, M, K, N;
  bool lt = left_transpose();
  bool rt = right_transpose();
  bool ot = output_transpose();
  parseMatMulParam(op->getOperand(0), op->getOperand(1), op->getResult(0),
                   M, K, N, batch_high, batch_low, lt, rt, ot);
  bool do_relu = this->do_relu();
  gaddr_t ga_left = getOpAddress(op->getOperand(0).getDefiningOp());
  gaddr_t ga_right = getOpAddress(op->getOperand(1).getDefiningOp());
  gaddr_t ga_output = getOpAddress(op);

  int layer_id = getOpLayerId(op);

  int8_t rshift_int8 = rshift().getValue();
  int32_t multiplier = this->mutliplier().getValue();
  int rshift = static_cast<int>(rshift_int8);
  std::vector<int> compr_weight_poss;

  cvi_backend_tg_fixed_fc_kernel(*backend_ctx, layer_id, ga_left, ga_right,
                                 GA_INVALID, ga_output, M, K, N, false, do_relu,
                                 rshift, multiplier, compr_weight_poss,
                                 batch_high, batch_low, lt, rt, ot);

  return success();
}

LogicalResult tpu::TG_BF16_MatMulOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  int batch_high = 0, batch_low = 0, M, K, N;
  bool lt = left_transpose();
  bool rt = right_transpose();
  bool ot = output_transpose();
  parseMatMulParam(op->getOperand(0), op->getOperand(1), op->getResult(0),
                   M, K, N, batch_high, batch_low, lt, rt, ot);
  bool do_relu = this->do_relu();
  gaddr_t ga_left = getOpAddress(op->getOperand(0).getDefiningOp());
  gaddr_t ga_right = getOpAddress(op->getOperand(1).getDefiningOp());
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);
  std::vector<int> compr_weight_poss;

  cvi_backend_tg_bf16_fc_kernel(
      *backend_ctx, layer_id, ga_left, ga_right, GA_INVALID, ga_output, M, K, N,
      false, do_relu, compr_weight_poss, batch_high, batch_low, lt, rt, ot);

  return success();
}

LogicalResult tpu::TG_BF16_LutOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  gaddr_t table_data_lut = getWeightOpAddress(table().getDefiningOp());
  gaddr_t table_data_mantissa_lut = getWeightOpAddress(table_mantissa().getDefiningOp());


  int layer_id = getOpLayerId(op);
  auto lut_method = method().getValue().str();
  LLVM_DEBUG(llvm::errs() << "lut method:" << lut_method << " [" << getOpName()
                          << "]\n";);
  if(lut_method == "mantissa") {
    cvi_backend_tg_bf16_lut_scientific_kernel(
        *backend_ctx, layer_id, input_gaddr, output_gaddr, table_data_lut,
        table_data_mantissa_lut, n, c, h, w, CVK_FMT_BF16);
  } else if (lut_method == "slope") {
    // 256 stand for lookup table index
    // we re-range [min, max] to table index
    float scale = 256.0 / (this->max_range().convertToFloat() - this->min_range().convertToFloat());
    cvi_backend_tg_bf16_lut_interpolation_kernel(
        *backend_ctx, layer_id, input_gaddr, output_gaddr, table_data_lut,
        table_data_mantissa_lut, n, c, h, w,
        this->min_range().convertToFloat(),
        this->max_range().convertToFloat(),
        scale);
  } else {
    std::string errorMsg = "unsupported lut method op: (manntissa or slope)" + lut_method + "\n";
    llvm_unreachable(errorMsg.c_str());
  }
  return success();
}

LogicalResult tpu::TG_INT8_PermuteOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  auto input_type = input().getType().template cast<TensorType>();
  std::vector<int64_t> i_s(input_type.getShape());

  std::vector<int> orders;
  orders.push_back(this->order0());
  orders.push_back(this->order1());
  orders.push_back(this->order2());
  orders.push_back(this->order3());

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_permute_kernel(
      *backend_ctx, layer_id, input_gaddr, output_gaddr, i_s[0], i_s[1], i_s[2],
      i_s[3], orders[0], orders[1], orders[2], orders[3], CVK_FMT_I8);
  return success();
}

LogicalResult tpu::TG_BF16_PermuteOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  auto input_type = input().getType().template cast<TensorType>();
  std::vector<int64_t> i_s(input_type.getShape());

  std::vector<int64_t> i_nchw(4,1);
  for (uint64_t i = 0; i < i_s.size(); i++) {
    i_nchw[i] = i_s[i];
  }

  std::vector<int> orders;
  orders.push_back(this->order0());
  orders.push_back(this->order1());
  orders.push_back(this->order2());
  orders.push_back(this->order3());

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_permute_kernel(*backend_ctx, layer_id, input_gaddr,
                                output_gaddr, i_nchw[0], i_nchw[1], i_nchw[2],
                                i_nchw[3], orders[0], orders[1], orders[2],
                                orders[3], CVK_FMT_BF16);
  return success();
}

LogicalResult tpu::TG_INT8_PoolAvg2DOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  // parse param
  bool is_global, do_relu, count_include_pad;
  int n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  parsePoolParam(param(), input(), output(),
                 n, c, ih, iw, oh, ow,
                 kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
                 is_global, do_relu, count_include_pad);
  assert(!do_relu);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  assert(this->rshift().hasValue());
  int8_t rshift = this->rshift().getValue();
  assert(this->m_i8().hasValue());
  int8_t m_i8 = this->m_i8().getValue();

  // TODO: should change on backend API, rather than doing cast
  int rshift_int = static_cast<int>(rshift);
  int m_int = static_cast<int>(m_i8);

  cvi_backend_tg_fixed_avg_pooling_kernel(
      *backend_ctx,
      layer_id, // layer_id,
      ga_input,            // input_data_gaddr,
      ga_output,           // output_data_gaddr,
      n, c, ih, iw,
      kh, kw,
      pt, pb, pl, pr, // pad (t, b, l, r)
      sh, sw,
      do_relu,        // int do_relu,
      rshift_int,     // int right_shift_width,
      m_int,         // &threshold_x_quantized,
      true);

  return success();
}

LogicalResult tpu::TG_INT8_PoolMax2DOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  // parse param
  bool is_global, do_relu, count_include_pad;
  int n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  parsePoolParam(param(), input(), output(),
                 n, c, ih, iw, oh, ow,
                 kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
                 is_global, do_relu, count_include_pad);
  assert(!do_relu);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  assert(!this->rshift().hasValue());
  assert(!this->m_i8().hasValue());

  int store_cmpr_act = this->store_compr_act().hasValue() ?
                       this->store_compr_act().getValue() : 0;
  int store_cmpr_act_c_step = 0;
  if (store_cmpr_act) {
    store_cmpr_act =
        this->store_compr_act_param().getValue().step_size().getInt();
    store_cmpr_act_c_step =
        this->store_compr_act_param().getValue().c_step().getInt();
  }
  int load_cmpr_act = this->load_compr_act().hasValue() ?
                      this->load_compr_act().getValue() : 0;
  int load_cmpr_act_c_step = 0;
  if (load_cmpr_act) {
    load_cmpr_act =
        this->load_compr_act_param().getValue().step_size().getInt();
    load_cmpr_act_c_step =
        this->load_compr_act_param().getValue().c_step().getInt();
  }

  cvi_backend_tg_fixed_max_pooling_kernel(
      *backend_ctx,
      layer_id, // layer_id,
      ga_input,            // input_data_gaddr,
      ga_output,           // output_data_gaddr,
      n, c, ih, iw,
      kh, kw,
      pt, pb, pl, pr, // pad (t, b, l, r)
      sh, sw,
      do_relu,        // int do_relu,
      true,
      store_cmpr_act,
      load_cmpr_act,
      store_cmpr_act_c_step,
      load_cmpr_act_c_step);

  return success();
}

LogicalResult tpu::TG_BF16_PoolAvg2DOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  // parse param
  bool is_global, do_relu, count_include_pad;
  int n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  parsePoolParam(param(), input(), output(),
                 n, c, ih, iw, oh, ow,
                 kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
                 is_global, do_relu, count_include_pad);
  assert(!do_relu);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_bf16_pooling_kernel(
      *backend_ctx,
      layer_id, // layer_id,
      ga_input,            // input_data_gaddr,
      ga_output,           // output_data_gaddr,
      GA_INVALID, // index_data_gaddr,
      GA_INVALID, // o_findex_data_gaddr,
      n, c, ih, iw,
      kh, kw,
      pt, pb, pl, pr, // pad (t, b, l, r)
      sh, sw,
      1,              // is_avg_pooling,
      0.0f,           // float avg_const,  // default(passing 0.0f) is 1/kh*kw
      do_relu,        // int do_relu,
      true);

  return success();
}

LogicalResult tpu::TG_BF16_PoolMax2DOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  // parse param
  bool is_global, do_relu, count_include_pad;
  int n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  parsePoolParam(param(), input(), output(),
                 n, c, ih, iw, oh, ow,
                 kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
                 is_global, do_relu, count_include_pad);
  assert(!do_relu);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_bf16_pooling_kernel(
      *backend_ctx,
      layer_id, // layer_id,
      ga_input,            // input_data_gaddr,
      ga_output,           // output_data_gaddr,
      GA_INVALID, // index_data_gaddr,
      GA_INVALID, // o_findex_data_gaddr,
      n, c, ih, iw,
      kh, kw,
      pt, pb, pl, pr, // pad (t, b, l, r)
      sh, sw,
      0,              // is_avg_pooling,
      0.0f,           // float avg_const,  // default(passing 0.0f) is 1/kh*kw
      do_relu,        // int do_relu,
      true);

  return success();
}

LogicalResult tpu::TG_INT8_PoolMax3DOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  return failure();
}

LogicalResult tpu::TG_BF16_PoolMax3DOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  // parse param
  bool is_global, do_relu, count_include_pad;
  int n, c, id, ih, iw, od, oh, ow, kd, kh, kw, sd, sh, sw;
  int pd0, pd1, pt, pb, pl, pr;
  parsePool3dParam(param(), input(), output(),
                   n, c, id, ih, iw, od, oh, ow,
                   kd, kh, kw, sd, sh, sw,
                   pd0, pd1, pt, pb, pl, pr,
                   is_global, do_relu, count_include_pad);
  assert(!do_relu);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_bf16_max_pooling3d_kernel(
      *backend_ctx,
      layer_id,
      ga_input,
      ga_output,
      n, c, id, ih, iw,
      od, oh, ow,
      kd, kh, kw,
      pd0, pd1, pt, pb, pl, pr, // pad (d0, d1, t, b, l, r)
      sd, sh, sw,
      do_relu,        // int do_relu,
      true);

  return success();
}

LogicalResult tpu::TG_INT8_PReluOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  gaddr_t negative_scope_gaddr =
      getWeightOpAddress(negative_slope().getDefiningOp());
  int layer_id = getOpLayerId(op);

  assert(this->rshift_pos().hasValue());
  int8_t rshift_pos = this->rshift_pos().getValue();
  assert(this->m_i8_pos().hasValue());
  int8_t m_i8_pos = this->m_i8_pos().getValue();
  assert(this->rshift_neg().hasValue());
  int8_t rshift_neg = this->rshift_neg().getValue();
  cvi_backend_tg_fixed_prelu_kernel(*backend_ctx, layer_id, ga_input, ga_output,
                                    negative_scope_gaddr, n, c, h, w,
                                    rshift_pos, m_i8_pos, rshift_neg);

  return success();
}

LogicalResult tpu::TG_BF16_PReluOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_neg_slope = getWeightOpAddress(op->getOperand(1).getDefiningOp());
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = mlir::getOpLayerId(op);

  cvi_backend_tg_bf16_prelu_kernel(
    *backend_ctx,        // ctx
    layer_id,            // layer_id,
    ga_input,            // input_gaddr
    ga_output,           // output_gaddr
    ga_neg_slope,        // ga_negative_slope
    n,                   // input_n
    c,                   // input_c
    h,                   // input_h
    w                   // input_w
  );
  return success();
}

static cvk_fmt_t get_fmt(std::string fmt_str){
  if (fmt_str == "FP32" || fmt_str == "NONE") {
    return CVK_FMT_F32;
  }
  if (fmt_str == "BF16") {
    return CVK_FMT_BF16;
  }
  if (fmt_str == "INT8") {
    return CVK_FMT_I8;
  }
  if (fmt_str == "UINT8") {
    return CVK_FMT_U8;
  }
  llvm_unreachable("unsupport other type cast");
}

LogicalResult tpu::TG_QuantOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName() << " [" << getOpName()
               << "]\n";);

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  int layer_id = getOpLayerId(op);
  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);

  std::vector<int64_t> shape;
  int64_t input_size, n, c, d, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  if (shape.size() == 5) {
    getNCDHW(shape, n, c, d, h, w);
    w = h * w;
    h = d;
  } else {
    getNCHW(shape, n, c, h, w);
  }
  cvk_fmt_t from = get_fmt(this->from().str());
  cvk_fmt_t to = get_fmt(this->to().str());
  float scale = this->scale().convertToFloat();
  int offset = 0;
  if(this->zero_point().hasValue()){
    offset = this->zero_point().getValue();
  }

  int load_cmpr_act = this->load_compr_act().hasValue() ?
                      this->load_compr_act().getValue() : 0;
  int load_cmpr_act_c_step = 0;
  if (load_cmpr_act) {
    load_cmpr_act =
        this->load_compr_act_param().getValue().step_size().getInt();
    load_cmpr_act_c_step =
        this->load_compr_act_param().getValue().c_step().getInt();
  }

  //  quant to int8
  cvi_backend_tg_quant_kernel(*backend_ctx, layer_id, from, to, ga_input,
                              ga_output, n, c, h, w, scale, offset,
                              load_cmpr_act, load_cmpr_act_c_step);

  return success();
}

LogicalResult tpu::TG_ReQuantOp::codegen(void *ctx) {
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  int layer_id = getOpLayerId(op);
  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  float scale = this->qscale().getValue().convertToFloat();
  int input_offset = this->input_offset().getValue();
  int output_offset = this->output_offset().getValue();


  //  quant to int8
  cvi_backend_tg_requant_kernel(*backend_ctx, layer_id, ga_input,
                              ga_output, n, c, h, w, input_offset,
                              output_offset, scale);

  return success();
}

LogicalResult tpu::TG_INT8_ReluOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_relu_kernel(*backend_ctx, layer_id, ga_input, ga_output, n, c,
                             h, w, CVK_FMT_I8);

  return success();
}


LogicalResult tpu::TG_BF16_ReluOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = mlir::getOpLayerId(op);

  cvi_backend_tg_relu_kernel(*backend_ctx, layer_id, ga_input, ga_output, n, c,
                             h, w, CVK_FMT_BF16);

  return success();
}

LogicalResult tpu::TG_INT8_ReorgOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  uint32_t stride = this->stride();

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_reorg_kernel(*backend_ctx, layer_id, input_gaddr, output_gaddr,
                              n, c, h, w, stride, CVK_FMT_I8);

  return success();
}

LogicalResult tpu::TG_BF16_ReorgOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  uint32_t stride = this->stride();

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_reorg_kernel(*backend_ctx, layer_id, input_gaddr, output_gaddr,
                              n, c, h, w, stride, CVK_FMT_BF16);

  return success();
}

LogicalResult tpu::TG_INT8_ReverseOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape = getTensorShape(op->getOperand(0));
  int64_t n, c, h, w;
  getNCHW(shape, n, c, h, w);
  int32_t axis = this->axis();

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  int layer_id = getOpLayerId(op);
  cvi_backend_tg_reverse_kernel(*backend_ctx, layer_id, input_gaddr,
                                        output_gaddr, n, c, h, w, axis,
                                        CVK_FMT_I8);
  return success();
}

LogicalResult tpu::TG_BF16_ReverseOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape = getTensorShape(op->getOperand(0));
  int64_t n, c, h, w;
  getNCHW(shape, n, c, h, w);
  int32_t axis = this->axis();

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  int layer_id = getOpLayerId(op);
  cvi_backend_tg_reverse_kernel(*backend_ctx, layer_id, input_gaddr,
                                        output_gaddr, n, c, h, w, axis,
                                        CVK_FMT_BF16);
  return success();
}

LogicalResult tpu::TG_INT8_ShuffleChannelOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape = getTensorShape(op->getOperand(0));
  int64_t n, c, h, w;
  getNCHW(shape, n, c, h, w);
  uint32_t group = this->group();

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_permute_kernel(*backend_ctx, layer_id, input_gaddr,
                                output_gaddr, n, group, c / group, h*w, 0, 2, 1,
                                3, CVK_FMT_I8);
  return success();
}

LogicalResult tpu::TG_BF16_ShuffleChannelOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape = getTensorShape(op->getOperand(0));
  int64_t n, c, h, w;
  getNCHW(shape, n, c, h, w);
  uint32_t group = this->group();

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  int layer_id = getOpLayerId(op);
  cvi_backend_tg_permute_kernel(*backend_ctx, layer_id, input_gaddr,
                                output_gaddr, n, group, c / group, h*w, 0, 2, 1,
                                3, CVK_FMT_BF16);
  return success();
}

LogicalResult tpu::TG_INT8_SwapChannelOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> input_shape = getTensorShape(input());
  std::vector<int> input_shape_fix;
    for (auto &dim : input_shape) {
    input_shape_fix.push_back((int)dim);
  }

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  int layer_id = getOpLayerId(op);
  std::vector<int32_t> order;
  arrayAttrToVector(this->channel_order(), order);
  cvi_backend_tg_swap_channel_kernel(*backend_ctx, layer_id,
                                       input_gaddr, output_gaddr,  (int)input_shape_fix.size(),
                                       input_shape_fix.data(), order.data(), CVK_FMT_I8);
  return success();
}

LogicalResult tpu::TG_BF16_SwapChannelOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> input_shape = getTensorShape(input());
  std::vector<int> input_shape_fix;
    for (auto &dim : input_shape) {
    input_shape_fix.push_back((int)dim);
  }
  std::vector<int32_t> order;
  arrayAttrToVector(this->channel_order(), order);

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  int layer_id = getOpLayerId(op);
  cvi_backend_tg_swap_channel_kernel(*backend_ctx, layer_id,
                                       input_gaddr, output_gaddr,  (int)input_shape_fix.size(),
                                       input_shape_fix.data(), order.data(), CVK_FMT_BF16);
  return success();
}

LogicalResult tpu::TG_INT8_TileOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  // backend not ok now
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> input_shape = getTensorShape(input());
  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  int64_t input_n, input_c, input_h, input_w;
  getNCHW(input_shape, input_n, input_c, input_h, input_w);

  cvi_backend_tg_tile_kernel(*backend_ctx, layer_id, input_gaddr, output_gaddr,
                             input_n, input_c, input_h, input_w, axis(),
                             tiles(), CVK_FMT_I8);

  return success();
}

LogicalResult tpu::TG_BF16_TileOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  // backend not ok now
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> input_shape = getTensorShape(input());
  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  int64_t input_n, input_c, input_h, input_w;
  getNCHW(input_shape, input_n, input_c, input_h, input_w);

  cvi_backend_tg_tile_kernel(*backend_ctx, layer_id, input_gaddr, output_gaddr,
                             input_n, input_c, input_h, input_w, axis(),
                             tiles(), CVK_FMT_BF16);

  return success();
}

LogicalResult tpu::TG_INT8_PixelShuffleOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  // tranform from pytorch define
  n = shape[0];
  c = shape[1];
  h = shape[2];
  w = shape[3];
  uint32_t upscale_factor = this->upscale_factor();

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  int layer_id = getOpLayerId(op);
  bool isDCR = this->mode().str() == "DCR";

  cvi_backend_tg_fixed_pixel_shuffle_kernel(*backend_ctx, layer_id, input_gaddr,
                                            output_gaddr, n, c, h, w,
                                            upscale_factor, isDCR);

  return success();
}

LogicalResult tpu::TG_BF16_PixelShuffleOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  // tranform from pytorch define
  n = shape[0];
  c = shape[1];
  h = shape[2];
  w = shape[3];
  uint32_t upscale_factor = this->upscale_factor();

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  int layer_id = getOpLayerId(op);
  bool isDCR = this->mode().str() == "DCR";

  cvi_backend_tg_bf16_pixel_shuffle_kernel(*backend_ctx, layer_id, input_gaddr,
                                           output_gaddr, n, c, h, w,
                                           upscale_factor, isDCR);

  return success();
}

LogicalResult tpu::TG_INT8_ClipOp::codegen(void *ctx) {
  std::string errorMsg = "unsupported tg op " + getOpName().str() + "\n";
  llvm_unreachable(errorMsg.c_str());
}

LogicalResult tpu::TG_BF16_ClipOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName() << " [" << getOpName()
               << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;

  getTensorShapeAndSize(op->getOperand(0), shape, input_size);

  // tranform from pytorch define
  n = shape[0];
  c = shape[1];
  h = shape[2];
  w = shape[3];

  float min = this->min().convertToFloat();
  float max = this->max().convertToFloat();

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);

  int layer_id = getOpLayerId(op);
  bool do_relu = false;
  float coeffs[2];
  gaddr_t ga_inputs[1];
  ga_inputs[0] = input_gaddr;

  // leverage min/max op rather than clip
  // FIXME: using EltwiseMaxOp/EltwiseMinOp

  // op definition refer to \TgFixedEltwiseKernel.cpp
  if (0) {
    coeffs[0] = {max};
    cvi_backend_tg_bf16_eltwise_max_kernel(
        *backend_ctx,
        layer_id,     // layer_id
        ga_inputs,    // gaddr_t ga_input[]
        output_gaddr,    // gaddr_t ga_output
        1,            // int input_size
        n, c, h, w,
        do_relu,      // bool do_relu
        false, 0, 0,
        coeffs);

    coeffs[0] = {min};
    cvi_backend_tg_bf16_eltwise_min_kernel(
        *backend_ctx,
        layer_id,     // layer_id
        ga_inputs,    // gaddr_t ga_input[]
        output_gaddr,    // gaddr_t ga_output
        1,            // int input_size
        n, c, h, w,
        do_relu,      // bool do_relu
        false, 0, 0,
        coeffs);
  }
  else {
    coeffs[0] = {max};
    coeffs[1] = {min};
    cvi_backend_tg_bf16_eltwise_min_max_kernel(
        *backend_ctx,
        layer_id,     // layer_id
        ga_inputs,    // gaddr_t ga_input[]
        output_gaddr,    // gaddr_t ga_output
        1,            // int input_size
        n, c, h, w,
        do_relu,      // bool do_relu
        false, 0, 0,
        coeffs);
  }

  return success();
}

LogicalResult tpu::ReshapeOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  //CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  //Operation *op = this->getOperation();

  return success();
}

LogicalResult tpu::TG_INT8_SliceOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  int axis = this->axis();
  std::vector<int64_t> input_shape = getTensorShape(input());
  std::vector<int> input_shape_fix;
  for (auto &dim : input_shape) {
    input_shape_fix.push_back((int)dim);
  }
  int index = 0;
  for (;index < axis; index++) {
    if (input_shape[index] != 1) {
      break;
    }
  }

  if (index == axis) {
    LLVM_DEBUG(llvm::errs() << "  no copy\n";);
    return success();
  }
  int offset = this->offset();
  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  int layer_id = getOpLayerId(op);
  std::vector<int64_t> output_shape = getTensorShape(this->getResult());
  cvi_backend_tg_slice_kernel(*backend_ctx, layer_id, input_gaddr,
                       output_gaddr, (int)input_shape.size(),
                       input_shape_fix.data(), axis, offset,
                       (int)output_shape[axis], CVK_FMT_I8);

  return success();
}

LogicalResult tpu::TG_BF16_SliceOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  int axis = this->axis();
  std::vector<int64_t> input_shape = getTensorShape(input());
  std::vector<int> input_shape_fix;
  for (auto &dim : input_shape) {
    input_shape_fix.push_back((int)dim);
  }

  int index = 0;
  for (;index < axis && input_shape[index] == 1; index++);
  if (index == axis) {
    LLVM_DEBUG(llvm::errs() << "  no copy\n";);
    return success();
  }
  int offset = this->offset();
  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  int layer_id = getOpLayerId(op);
  std::vector<int64_t> output_shape = getTensorShape(this->getResult());
  cvi_backend_tg_slice_kernel(*backend_ctx, layer_id, input_gaddr,
                       output_gaddr, (int)input_shape.size(),
                       input_shape_fix.data(), axis, offset,
                       (int)output_shape[axis], CVK_FMT_BF16);

  return success();
}

LogicalResult tpu::TG_INT8_UpsampleOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  int32_t scale_h = this->scale_h();
  int32_t scale_w = this->scale_w();

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_upsample_kernel(*backend_ctx, layer_id, ga_input, ga_output, n,
                                 c, h, w, scale_h, scale_w, CVK_FMT_I8);

  return success();
}

LogicalResult tpu::TG_BF16_UpsampleOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  int32_t scale_h = this->scale_h();
  int32_t scale_w = this->scale_w();

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_upsample_kernel(*backend_ctx, layer_id, ga_input, ga_output, n,
                                 c, h, w, scale_h, scale_w, CVK_FMT_BF16);

  return success();
}

LogicalResult tpu::TG_INT8_PadOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  // parse param
  std::vector<int32_t> pads;
  auto const_val = this->const_val().convertToFloat();
  auto mode = this->mode().str().c_str();
  arrayAttrToVector(this->pads().getValue(), pads);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_pad_kernel(
      *backend_ctx,
      layer_id,
      ga_input,
      ga_output,
      n,
      c,
      h,
      w,
      pads.data(),
      const_val,
      mode,
      CVK_FMT_I8
  );

  return success();
}

LogicalResult tpu::TG_BF16_PadOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  // parse param
  std::vector<int32_t> pads;
  auto const_val = this->const_val().convertToFloat();
  auto mode = this->mode().str().c_str();
  arrayAttrToVector(this->pads().getValue(), pads);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_pad_kernel(
      *backend_ctx,
      layer_id,
      ga_input,
      ga_output,
      n,
      c,
      h,
      w,
      pads.data(),
      const_val,
      mode,
      CVK_FMT_BF16
  );

  return success();
}

LogicalResult tpu::TG_INT8_ReduceMeanOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);
  std::vector<int64_t> input_shape = getTensorShape(input());

  int rshift = 0;
  if (this->rshift().hasValue())
    rshift = this->rshift().getValue();

  int multiplier = 1;
  if (this->m_i8().hasValue())
    multiplier = this->m_i8().getValue();

  int num_axes = 0;
  int *axes = nullptr;
  if (this->axes().hasValue()) {
    std::vector<int32_t> axes_array;
    arrayAttrToVector(this->axes().getValue(), axes_array);
    num_axes = axes_array.size();
    axes = new int[num_axes];
    for (unsigned i = 0; i < axes_array.size(); ++i)
      axes[i] = axes_array[i];
  }

  cvi_backend_tg_fixed_reduce_mean_kernel(*backend_ctx,
                                          layer_id,
                                          ga_input,
                                          ga_output,
                                          (int)input_shape[0],
                                          (int)input_shape[1],
                                          (int)input_shape[2],
                                          (int)input_shape[3],
                                          rshift,
                                          multiplier,
                                          axes,
                                          num_axes
                                          );

  delete[] axes;

  return success();
}

LogicalResult tpu::TG_INT8_ReduceMaxOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);
  std::vector<int64_t> input_shape = getTensorShape(input());

  int num_axes = 0;
  int *axes = nullptr;
  if (this->axes().hasValue()) {
    std::vector<int32_t> axes_array;
    arrayAttrToVector(this->axes().getValue(), axes_array);
    num_axes = axes_array.size();
    axes = new int[num_axes];
    for (unsigned i = 0; i < axes_array.size(); ++i)
      axes[i] = axes_array[i];
  }

  if (num_axes == 1 && input_shape.size() == 5 && input_shape[axes[0]] == 1) {
    // Replace with tdma global memory copy via TG permute
    std::vector<int64_t> output_shape = getTensorShape(output());
    cvi_backend_tg_permute_kernel(*backend_ctx, layer_id, ga_input, ga_output,
                                  (uint32_t)output_shape[0], // input_n
                                  (uint32_t)output_shape[1], // input_c
                                  (uint32_t)output_shape[2], // input_h
                                  (uint32_t)output_shape[3], // input_w
                                  0, 1, 2, 3,                // order
                                  CVK_FMT_I8);
  } else {
    cvi_backend_tg_fixed_reduce_max_kernel(*backend_ctx,
                                          layer_id,
                                          ga_input,
                                          ga_output,
                                          (int)input_shape[0],
                                          (int)input_shape[1],
                                          (int)input_shape[2],
                                          (int)input_shape[3],
                                          axes,
                                          num_axes
                                          );
  }

  delete[] axes;

  return success();
}

LogicalResult tpu::TG_BF16_ReduceMeanOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);
  std::vector<int64_t> input_shape = getTensorShape(input());

  int num_axes = 0;
  int *axes = nullptr;
  if (this->axes().hasValue()) {
    std::vector<int32_t> axes_array;
    arrayAttrToVector(this->axes().getValue(), axes_array);
    num_axes = axes_array.size();
    axes = new int[num_axes];
    for (unsigned i = 0; i < axes_array.size(); ++i)
      axes[i] = axes_array[i];
  }

  cvi_backend_tg_bf16_reduce_mean_kernel(*backend_ctx,
                                         layer_id,
                                         ga_input,
                                         ga_output,
                                         (int)input_shape[0],
                                         (int)input_shape[1],
                                         (int)input_shape[2],
                                         (int)input_shape[3],
                                         axes,
                                         num_axes
                                         );

  delete[] axes;

  return success();
}

LogicalResult tpu::TG_BF16_ReduceMaxOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);
  std::vector<int64_t> input_shape = getTensorShape(input());

  int num_axes = 0;
  int *axes = nullptr;
  if (this->axes().hasValue()) {
    std::vector<int32_t> axes_array;
    arrayAttrToVector(this->axes().getValue(), axes_array);
    num_axes = axes_array.size();
    axes = new int[num_axes];
    for (unsigned i = 0; i < axes_array.size(); ++i)
      axes[i] = axes_array[i];
  }

  if (num_axes == 1 && input_shape.size() == 5 && input_shape[axes[0]] == 1) {
    // Replace with tdma global memory copy via TG permute
    std::vector<int64_t> output_shape = getTensorShape(output());
    cvi_backend_tg_permute_kernel(*backend_ctx, layer_id, ga_input, ga_output,
                                  (uint32_t)output_shape[0], // input_n
                                  (uint32_t)output_shape[1], // input_c
                                  (uint32_t)output_shape[2], // input_h
                                  (uint32_t)output_shape[3], // input_w
                                  0,                         // order_n
                                  1,                         // order_c
                                  2,                         // order_h
                                  3,                         // order_w
                                  CVK_FMT_BF16);
  } else {
    cvi_backend_tg_bf16_reduce_max_kernel(*backend_ctx,
                                          layer_id,
                                          ga_input,
                                          ga_output,
                                          (int)input_shape[0],
                                          (int)input_shape[1],
                                          (int)input_shape[2],
                                          (int)input_shape[3],
                                          axes,
                                          num_axes
                                          );
  }

  delete[] axes;

  return success();
}

LogicalResult tpu::TG_INT8_Yuv420CscOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> output_shape = getTensorShape(this->getResult());
  int64_t n, c, h, w;
  getNCHW(output_shape, n, c, h, w);
  std::vector<int32_t> order;
  if (this->channel_order().hasValue()) {
    arrayAttrToVector(this->channel_order().getValue(), order);
  }

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  int layer_id = getOpLayerId(op);
  cvi_backend_tg_yuv420_csc_kernel(*backend_ctx, layer_id, input_gaddr,
                                   output_gaddr, n, c, h, w, order,
                                   CVK_FMT_U8);
  return success();
}

LogicalResult tpu::TG_CallOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_ConcatNOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_INT8_MergeConvConvPoolOp::codegen(void *ctx) {
  return success();
}

}

