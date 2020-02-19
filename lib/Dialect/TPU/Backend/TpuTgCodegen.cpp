//===- ConvertToBinary.cpp - MLIR SPIR-V module to binary conversion ------===//
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

#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
#include "mlir/Dialect/TPU/QuantizationArithmetic.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
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
#include "mlir/Support/TensorFile.h"

#include <fstream>

#define DEBUG_TYPE "mlir-to-cmdbuf"

using namespace mlir;

extern int BF16_TABLE_START;
extern int BF16_TABLE_END;

#include "backend/backend_tg_api.h"
#include "backend/backend_tl_api.h"

namespace mlir {

LogicalResult tpu::TG_INT8_ConcatOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  //BM1880v2BackendContext *backend_ctx = (BM1880v2BackendContext *)ctx;
  //Operation *op = this->getOperation();

  assert(false);
  return success();
}

LogicalResult tpu::TG_BF16_ConcatOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  //BM1880v2BackendContext *backend_ctx = (BM1880v2BackendContext *)ctx;
  //Operation *op = this->getOperation();

  assert(false);
  return success();
}

LogicalResult tpu::TG_INT8_PT_Conv2DOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  BM1880v2BackendContext *backend_ctx = (BM1880v2BackendContext *)ctx;
  Operation *op = this->getOperation();

  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, ph, pw, dh, dw;
  parseConvParam(param(), input(), output(), filter(),
                 n, ic, ih, iw, oc, oh, ow, g,
                 kh, kw, sh, sw, ph, pw, dh, dw, is_dw, with_bias, do_relu);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  gaddr_t ga_filter = getWeightOpAddress(filter()->getDefiningOp());
  gaddr_t ga_bias = INVALID_GLOBAL_ADDR;
  if ( with_bias ) {
    assert(!isTensorNone(pc_info()));
    ga_bias =  getWeightOpAddress(pc_info()->getDefiningOp());
  }
  assert(pt_rshift().hasValue());
  int8_t rshift = pt_rshift().getValue().getLimitedValue();
  int layer_id = mlir::getOpLayerId(op);

  bmnet_conv_parallel_fixed_forward_bmkernel(
      *backend_ctx,
      0, // stream_id,
      0, // inst_id,
      layer_id, // layer_id,
      nullptr,  // depends
      0, // depends_len
      ga_input,  // input_data_gaddr,
      ga_output, // output_data_gaddr,
      ga_filter, // weight_data_gaddr,
      ga_bias,   // bias_data_gaddr,
      INVALID_GLOBAL_ADDR, // bn_mean_data_gaddr,
      INVALID_GLOBAL_ADDR, // bn_variance_data_gaddr,
      INVALID_GLOBAL_ADDR, // scale_gaddr,
      INVALID_GLOBAL_ADDR, // scale_bias_gaddr,
      n, ic, ih, iw,
      g, // group,
      oc,
      kh, kw,
      dh, dw,
      ph, ph, pw, pw, // pad (t, b, l, r)
      sh, sw,
      0,         // result_add
      with_bias, // bias_term,
      0,         // do_bn,
      0,         // do_scale,
      0,         // do_scale_bias,
      do_relu ? 1 : 0, // do_activation,
      1.0f,      // bn_scale,
      1e-5,      // eps,
      0,         // param.activation(), method, 0 -> RELU, all others are invalide for now
      nullptr,   // activation_arg,
      INVALID_GLOBAL_ADDR, //global_slope_gaddr,
      false,     //channel_shared,
      0,         //activation_gt_scale,
      0,         //activation_gt_rshift,
      0,         //activation_le_scale, // slope, TODO
      0,         //activation_le_rshift,
      (int)rshift, // right_shift_width,
      0,         //bn_right_shift_width,
      0,         //scale_right_shift_width,
      false,     //use_winograd
      0,         // right_shift_array_len
      0          // ga_per_channel
      );

  return success();
}

LogicalResult tpu::TG_INT8_PC_Conv2DOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  BM1880v2BackendContext *backend_ctx = (BM1880v2BackendContext *)ctx;
  Operation *op = this->getOperation();

  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, ph, pw, dh, dw;
  parseConvParam(param(), input(), output(), filter(),
                 n, ic, ih, iw, oc, oh, ow, g,
                 kh, kw, sh, sw, ph, pw, dh, dw, is_dw, with_bias, do_relu);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  gaddr_t ga_filter = getWeightOpAddress(filter()->getDefiningOp());
  gaddr_t ga_pc_info = getWeightOpAddress(pc_info()->getDefiningOp());
  int layer_id = mlir::getOpLayerId(op);

  bmnet_conv_parallel_fixed_forward_bmkernel(
      *backend_ctx,
      0, // stream_id,
      0, // inst_id,
      layer_id, // layer_id,
      nullptr, // depends
      0, // depends_len
      ga_input,   // input_data_gaddr,
      ga_output,  // output_data_gaddr,
      ga_filter,  // weight_data_gaddr,
      ga_pc_info, // bias_data_gaddr,
      INVALID_GLOBAL_ADDR, // bn_mean_data_gaddr,
      INVALID_GLOBAL_ADDR, // bn_variance_data_gaddr,
      INVALID_GLOBAL_ADDR, // scale_gaddr,
      INVALID_GLOBAL_ADDR, // scale_bias_gaddr,
      n, ic, ih, iw,
      g, // group,
      oc,
      kh, kw,
      dh, dw,
      ph, ph, pw, pw, // pad (t, b, l, r)
      sh, sw,
      0,         // result_add
      with_bias, // bias_term,
      0,         // do_bn,
      0,         // do_scale,
      0,         // do_scale_bias,
      do_relu ? 1 : 0, // do_activation,
      1.0f,      // bn_scale,
      1e-5,      // eps,
      0,         // param.activation(), method, 0 -> RELU, all others are invalide for now
      nullptr,   // activation_arg,
      INVALID_GLOBAL_ADDR, //global_slope_gaddr,
      false,     //channel_shared,
      0,         //activation_gt_scale,
      0,         //activation_gt_rshift,
      0,         //activation_le_scale, // slope, TODO
      0,         //activation_le_rshift,
      0,         //(int)rshift[0], //right_shift_width,
      0,         //bn_right_shift_width,
      0,         //scale_right_shift_width,
      false,     //use_winograd
      oc,        // right_shift_array_len
      ga_pc_info // ga_per_channel
      );

  return success();
}

LogicalResult tpu::TG_BF16_Conv2DOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  BM1880v2BackendContext *backend_ctx = (BM1880v2BackendContext *)ctx;
  Operation *op = this->getOperation();

  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, ph, pw, dh, dw;
  parseConvParam(param(), input(), output(), filter(),
                 n, ic, ih, iw, oc, oh, ow, g,
                 kh, kw, sh, sw, ph, pw, dh, dw, is_dw, with_bias, do_relu);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  gaddr_t ga_filter = getWeightOpAddress(filter()->getDefiningOp());
  gaddr_t ga_bias = INVALID_GLOBAL_ADDR;
  if ( with_bias ) {
    assert(!isTensorNone(pc_info()));
    ga_bias =  getWeightOpAddress(pc_info()->getDefiningOp());
  }
  int layer_id = mlir::getOpLayerId(op);

  bmnet_bf16_conv_forward_kernel(
      *backend_ctx,
      layer_id,  // layer_id
      ga_input,
      ga_output,
      ga_filter,
      ga_bias,
      INVALID_GLOBAL_ADDR, // ga_bn_mean
      INVALID_GLOBAL_ADDR, // ga_bn_variance
      INVALID_GLOBAL_ADDR, // ga_scale
      INVALID_GLOBAL_ADDR, // ga_scale_bias
      n, ic, ih, iw,
      g, // group
      oc,
      kh, kw,
      dh, dw,
      ph, ph, pw, pw, // pad (t, b, l, r)
      sh, sw,
      with_bias,
      0,         // do_bn
      0,         // do_scale
      0,         // do_scale_bias
      do_relu ? 1 : 0,
      1.0f,      // bn_scale
      1e-5,      // eps
      0,         // param.activation(), method, 0 -> RELU, all others are invalid for now
      nullptr,   // activation_arg,
      INVALID_GLOBAL_ADDR //global_slope_gaddr
      );

  return success();
}

static void arrayAttrToVector(const ArrayAttr &arrayAttr,
    std::vector<int32_t> &vector) {
  vector.clear();
  for (auto en : llvm::enumerate(arrayAttr)) {
    auto attr = en.value().dyn_cast<IntegerAttr>();
    vector.push_back(attr.getInt());
  }
}

LogicalResult tpu::TG_INT8_EltwiseAddOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  BM1880v2BackendContext *backend_ctx = (BM1880v2BackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  bool do_relu = this->do_relu();

  gaddr_t ga_inputs[2];
  ga_inputs[0] = getPreviousOpAddress(op, 0);
  ga_inputs[1] = getPreviousOpAddress(op, 1);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = mlir::getOpLayerId(op);

  assert(this->rshift().hasValue());
  int8_t rshift = this->rshift().getValue().getLimitedValue();

  int8_t m_i8_input[2];
  assert(this->m_i8_inputs().hasValue());
  std::vector<int32_t> m_i8_inputs_array;
  arrayAttrToVector(this->m_i8_inputs().getValue(), m_i8_inputs_array);
  assert(m_i8_inputs_array.size() == op->getNumOperands());
  assert(m_i8_inputs_array.size() >= 2);
  m_i8_input[0] = static_cast<int8_t>(m_i8_inputs_array[0]);
  m_i8_input[1] = static_cast<int8_t>(m_i8_inputs_array[1]);

  // TODO: should change on backend API, rather than doing cast
  int rshift_int = static_cast<int>(rshift);
  int m_int[2];
  m_int[0] = static_cast<int>(m_i8_input[0]);
  m_int[1] = static_cast<int>(m_i8_input[1]);
  const int coeffs[2] = {1, 1};

  bmnet_eltwise_fixed_forward_bmkernel(
      *backend_ctx,
      0,            // stream_id,
      0,            // inst_id,
      layer_id,     // layer_id,
      nullptr,      // depends
      0,            // depends_len
      ga_inputs,    // gaddr_t ga_input[],
      ga_output,    // gaddr_t ga_output,
      2,            // int input_size,
      1,            // int op,  0, prod, 1, sum, 2, max
      n, c, h, w,
      do_relu,      // bool do_relu,
      0.0f,         // float relu_slope,
      rshift_int,   // int right_shift_width,
      m_int,
      coeffs);

  return success();
}

LogicalResult tpu::TG_INT8_EltwiseMaxOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  //BM1880v2BackendContext *backend_ctx = (BM1880v2BackendContext *)ctx;
  //Operation *op = this->getOperation();

  assert(false);
  return success();
}

LogicalResult tpu::TG_INT8_EltwiseMulOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  BM1880v2BackendContext *backend_ctx = (BM1880v2BackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  bool do_relu = this->do_relu();

  gaddr_t ga_inputs[2];
  ga_inputs[0] = getPreviousOpAddress(op, 0);
  ga_inputs[1] = getPreviousOpAddress(op, 1);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = mlir::getOpLayerId(op);

  assert(this->rshift().hasValue());
  int8_t rshift = this->rshift().getValue().getLimitedValue();
  assert(this->m_i8_output().hasValue());
  int8_t m_i8_output = this->m_i8_output().getValue().getLimitedValue();

  // TODO: should change on backend API, rather than doing cast
  int rshift_int = static_cast<int>(rshift);
  int m_int = static_cast<int>(m_i8_output);
  const int coeffs[2] = {1, 1};

  bmnet_eltwise_fixed_forward_bmkernel(
      *backend_ctx,
      0,            // stream_id,
      0,            // inst_id,
      layer_id,     // layer_id,
      nullptr,      // depends
      0,            // depends_len
      ga_inputs,    // gaddr_t ga_input[],
      ga_output,    // gaddr_t ga_output,
      2,            // int input_size,
      0,            // int op,  0, prod, 1, sum, 2, max
      n, c, h, w,
      do_relu,      // bool do_relu,
      0.0f,         // float relu_slope,
      rshift_int,   // int right_shift_width,
      &m_int,
      coeffs
      );

  return success();
}

LogicalResult tpu::TG_BF16_EltwiseAddOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  BM1880v2BackendContext *backend_ctx = (BM1880v2BackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  bool do_relu = this->do_relu();

  gaddr_t ga_inputs[2];
  ga_inputs[0] = getPreviousOpAddress(op, 0);
  ga_inputs[1] = getPreviousOpAddress(op, 1);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = mlir::getOpLayerId(op);

  const float coeffs[2] = {1.0, 1.0};

  bf16_eltwise_forward_kernel(
      *backend_ctx,
      layer_id,     // layer_id
      ga_inputs,    // gaddr_t ga_input[]
      ga_output,    // gaddr_t ga_output
      2,            // int input_size
      1,            // int op, 0: prod, 1: sum, 2: max
      n, c, h, w,
      do_relu,      // bool do_relu
      0.0f,         // float relu_slope
      coeffs);

  return success();
}

LogicalResult tpu::TG_BF16_EltwiseMaxOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  //BM1880v2BackendContext *backend_ctx = (BM1880v2BackendContext *)ctx;
  //Operation *op = this->getOperation();

  assert(false);
  return success();
}

LogicalResult tpu::TG_BF16_EltwiseMulOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  BM1880v2BackendContext *backend_ctx = (BM1880v2BackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  bool do_relu = this->do_relu();

  gaddr_t ga_inputs[2];
  ga_inputs[0] = getPreviousOpAddress(op, 0);
  ga_inputs[1] = getPreviousOpAddress(op, 1);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = mlir::getOpLayerId(op);

  const float coeffs[2] = {1.0, 1.0};

  bf16_eltwise_forward_kernel(
      *backend_ctx,
      layer_id,     // layer_id
      ga_inputs,    // gaddr_t ga_input[]
      ga_output,    // gaddr_t ga_output
      2,            // int input_size
      0,            // int op, 0: prod, 1: sum, 2: max
      n, c, h, w,
      do_relu,      // bool do_relu
      0.0f,         // float relu_slope
      coeffs);

  return success();
}

LogicalResult tpu::TG_INT8_LeakyReluOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  //BM1880v2BackendContext *backend_ctx = (BM1880v2BackendContext *)ctx;
  //Operation *op = this->getOperation();

  assert(false);
  return success();
}

LogicalResult tpu::TG_BF16_LeakyReluOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  //BM1880v2BackendContext *backend_ctx = (BM1880v2BackendContext *)ctx;
  //Operation *op = this->getOperation();

  assert(false);
  return success();
}

LogicalResult tpu::TG_INT8_PoolAvg2DOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  BM1880v2BackendContext *backend_ctx = (BM1880v2BackendContext *)ctx;
  Operation *op = this->getOperation();

  // parse param
  bool is_global, do_relu;
  int n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr;
  parsePoolParam(param(), input(), output(),
                 n, c, ih, iw, oh, ow,
                 kh, kw, sh, sw, pt, pb, pl, pr,
                 is_global, do_relu);
  assert(!do_relu);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = mlir::getOpLayerId(op);

  assert(this->rshift().hasValue());
  int8_t rshift = this->rshift().getValue().getLimitedValue();
  assert(this->m_i8().hasValue());
  int8_t m_i8 = this->m_i8().getValue().getLimitedValue();

  // TODO: should change on backend API, rather than doing cast
  int rshift_int = static_cast<int>(rshift);
  int m_int = static_cast<int>(m_i8);

  bmnet_pooling_fixed_forward_bmkernel(
      *backend_ctx,
      0, // stream_id,
      0, // inst_id,
      layer_id, // layer_id,
      nullptr, // depends
      0, // depends_len
      ga_input,            // input_data_gaddr,
      ga_output,           // output_data_gaddr,
      INVALID_GLOBAL_ADDR, // index_data_gaddr,
      INVALID_GLOBAL_ADDR, // o_findex_data_gaddr,
      n, c, ih, iw,
      kh, kw,
      pt, pb, pl, pr, // pad (t, b, l, r)
      sh, sw,
      1,              // is_avg_pooling,
      0.0f,           // float avg_const,  // default(passing 0.0f) is 1/kh*kw
      do_relu,        // int do_relu,
      rshift_int,     // int right_shift_width,
      &m_int,         // &threshold_x_quantized,
      true);

  return success();
}

LogicalResult tpu::TG_INT8_PoolMax2DOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  BM1880v2BackendContext *backend_ctx = (BM1880v2BackendContext *)ctx;
  Operation *op = this->getOperation();

  // parse param
  bool is_global, do_relu;
  int n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr;
  parsePoolParam(param(), input(), output(),
                 n, c, ih, iw, oh, ow,
                 kh, kw, sh, sw, pt, pb, pl, pr,
                 is_global, do_relu);
  assert(!do_relu);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = mlir::getOpLayerId(op);

  assert(!this->rshift().hasValue());
  assert(!this->m_i8().hasValue());

  bmnet_pooling_fixed_forward_bmkernel(
      *backend_ctx,
      0, // stream_id,
      0, // inst_id,
      layer_id, // layer_id,
      nullptr, // depends
      0, // depends_len
      ga_input,            // input_data_gaddr,
      ga_output,           // output_data_gaddr,
      INVALID_GLOBAL_ADDR, // index_data_gaddr,
      INVALID_GLOBAL_ADDR, // o_findex_data_gaddr,
      n, c, ih, iw,
      kh, kw,
      pt, pb, pl, pr, // pad (t, b, l, r)
      sh, sw,
      0,              // is_avg_pooling,
      0.0f,           // float avg_const,  // default(passing 0.0f) is 1/kh*kw
      do_relu,        // int do_relu,
      0,              // int right_shift_width,
      nullptr,        // &threshold_x_quantized,
      true);

  return success();
}

LogicalResult tpu::TG_BF16_PoolAvg2DOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  BM1880v2BackendContext *backend_ctx = (BM1880v2BackendContext *)ctx;
  Operation *op = this->getOperation();

  // parse param
  bool is_global, do_relu;
  int n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr;
  parsePoolParam(param(), input(), output(),
                 n, c, ih, iw, oh, ow,
                 kh, kw, sh, sw, pt, pb, pl, pr,
                 is_global, do_relu);
  assert(!do_relu);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = mlir::getOpLayerId(op);

  bf16_pooling_forward_kernel(
      *backend_ctx,
      layer_id, // layer_id,
      ga_input,            // input_data_gaddr,
      ga_output,           // output_data_gaddr,
      INVALID_GLOBAL_ADDR, // index_data_gaddr,
      INVALID_GLOBAL_ADDR, // o_findex_data_gaddr,
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
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  BM1880v2BackendContext *backend_ctx = (BM1880v2BackendContext *)ctx;
  Operation *op = this->getOperation();

  // parse param
  bool is_global, do_relu;
  int n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr;
  parsePoolParam(param(), input(), output(),
                 n, c, ih, iw, oh, ow,
                 kh, kw, sh, sw, pt, pb, pl, pr,
                 is_global, do_relu);
  assert(!do_relu);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = mlir::getOpLayerId(op);

  bf16_pooling_forward_kernel(
      *backend_ctx,
      layer_id, // layer_id,
      ga_input,            // input_data_gaddr,
      ga_output,           // output_data_gaddr,
      INVALID_GLOBAL_ADDR, // index_data_gaddr,
      INVALID_GLOBAL_ADDR, // o_findex_data_gaddr,
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

LogicalResult tpu::TG_INT8_UpsampleOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  BM1880v2BackendContext *backend_ctx = (BM1880v2BackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  int32_t scale = this->scale().getLimitedValue();

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = mlir::getOpLayerId(op);

  upsample_fixed_bmkernel(
      *backend_ctx,
      0, //stream_id,
      0, //inst_id,
      layer_id, //layer_id,
      nullptr, //const u32 *depends,
      0, //depends_len,
      ga_input,
      ga_output,
      n,
      c,
      h,
      w,
      scale,
      scale);

  return success();
}

LogicalResult tpu::TG_BF16_UpsampleOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  //BM1880v2BackendContext *backend_ctx = (BM1880v2BackendContext *)ctx;
  //Operation *op = this->getOperation();

  assert(false);
  return success();
}

}
