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

#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/QuantizationArithmetic.h"
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

#include "backend/backend_common.h" // cvk_fmt_t

#define DEBUG_TYPE "mlir-to-cmdbuf"

using namespace mlir;


#include "backend/backend_tg_api.h"
#include "backend/backend_tl_api.h"

namespace mlir {

static void parseTLLeakyReluParam(Operation *op,
    int8_t &pos_rshift, int8_t &pos_m_i8,
    int8_t &neg_rshift, int8_t &neg_m_i8,
    float & negative_slope) {
  auto lreluOp = llvm::dyn_cast<tpu::TL_LG_INT8_LeakyReluOp>(op);
  assert(lreluOp);

  if (lreluOp.m_i8_pos().hasValue()) {
    pos_m_i8 = lreluOp.m_i8_pos().getValue();
    pos_rshift = lreluOp.rshift_pos().getValue();
  } else {
    pos_m_i8 = 0;
    pos_rshift = 0;
  }

  if (lreluOp.m_i8_neg().hasValue()) {
    neg_m_i8 = lreluOp.m_i8_neg().getValue();
    neg_rshift = lreluOp.rshift_neg().getValue();
  } else {
    neg_m_i8 = 0;
    neg_rshift = 0;
  }
  negative_slope = lreluOp.negative_slope().convertToFloat();
}

static void parseTLConvLeakyParam(Operation *op,
    int8_t &pos_rshift, int8_t &pos_m_i8,
    int8_t &neg_rshift, int8_t &neg_m_i8,
    float & negative_slope) {
  auto lreluOp = llvm::dyn_cast<tpu::TL_LG_INT8_Conv2DOp>(op);
  assert(lreluOp);

  if (lreluOp.m_i8_pos().hasValue()) {
    pos_m_i8 = lreluOp.m_i8_pos().getValue();
    pos_rshift = lreluOp.rshift_pos().getValue();
  } else {
    pos_m_i8 = 0;
    pos_rshift = 0;
  }

  if (lreluOp.m_i8_neg().hasValue()) {
    neg_m_i8 = lreluOp.m_i8_neg().getValue();
    neg_rshift = lreluOp.rshift_neg().getValue();
  } else {
    neg_m_i8 = 0;
    neg_rshift = 0;
  }
  negative_slope = lreluOp.negative_slope().getValue().convertToFloat();
}

LogicalResult tpu::TL_LG_INT8_AbsOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  auto input_shape = getTensorShape(op->getOperand(0));
  int64_t n, c, h, w;
  getNCHW(input_shape, n, c, h, w);

  int nInputs = op->getNumOperands();
  std::vector<int32_t> la_input_array;
  auto la_input = new laddr_t[nInputs];
  la_input[0] = this->la_input();

  laddr_t la_output = this->la_output();

  // parse param
  int layer_id = getOpLayerId(op);

  int op_code = 3; // abs
  cvi_backend_tl_eltwise( *backend_ctx,
      layer_id, /*u32 layer_id,*/
      la_input,
      la_output,
      -1 /*la_working*/,
      n, c, h, w, nInputs,
      op_code,
      0 /*rshift*/,
      0 /*m_i8_input*/,
      0 /*use_default_coeff*/,
      0 /*do_relu*/,
      0 /*relu_slope*/,
      NULL /*coeffs*/,
      0,
      0, 0, 0 /*do_early_stride, early_stride_h, early_stride_w*/
      );

  delete[] la_input;

  return success();
}

LogicalResult tpu::TL_LG_BF16_AbsOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
    CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  auto input_shape = getTensorShape(op->getOperand(0));
  int64_t n, c, h, w;
  getNCHW(input_shape, n, c, h, w);

  int nInputs = op->getNumOperands();
  std::vector<int32_t> la_input_array;
  auto la_input = new laddr_t[nInputs];
  la_input[0] = this->la_input();

  laddr_t la_output = this->la_output();

  // parse param
  int layer_id = getOpLayerId(op);

  int op_code = 3; // abs

  cvi_backend_bf16_tl_eltwise(*backend_ctx,
      layer_id, /*u32 layer_id,*/
      la_input,
      la_output,
      n, c, h, w, nInputs,
      op_code,
      0 /*use_default_coeff*/,
      0 /*do_relu*/,
      0 /*relu_slope*/,
      NULL /*coeffs*/,
      0 /*do_early_stride*/,
      0, 0 /*early_stride_h, early_stride_w*/
      );

  delete[] la_input;
  return success();
}

LogicalResult tpu::TL_LG_INT8_Conv2DOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb, pl,
      pr, dh, dw, pad_value;
  parseConvParam(param(), false, input(), output(), n, ic, ih, iw, oc,
                 oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb, pl, pr, dh,
                 dw, is_dw, with_bias, do_relu, pad_value);

  laddr_t la_input = this->la_input();
  laddr_t la_output = this->la_output();
  laddr_t la_weight = this->la_filter();
  laddr_t la_perchanel = this->la_bias();
  laddr_t la_working = this->la_working();
  bool do_ic_alignment = this->do_ic_alignment().hasValue()
                            ? this->do_ic_alignment().getValue() : false;

  // pad is not "SAME", can not get from conv param
  int ph_t = this->pad_top_h();
  int ph_b = this->pad_bottom_h();
  int pw_l = this->pad_left_w();
  int pw_r = this->pad_right_w();
  int layer_id = getOpLayerId(op);

  int8_t pos_rshift = 0, pos_m_i8 = 0;
  int8_t neg_rshift = 0, neg_m_i8 = 0;
  float neg_slope = 0.0;

  if (this->do_leaky_relu())
    parseTLConvLeakyParam(op, pos_rshift, pos_m_i8,
                          neg_rshift, neg_m_i8, neg_slope);

  cvi_backend_tl_conv(
    *backend_ctx,
    layer_id,
    la_input, la_output, la_weight, la_working, la_perchanel,
    n, ic, ih, iw,
    g, oc, oh, ow, kh, kw, dh,
    dw, ph_t, ph_b, pw_l, pw_r, sh, sw, ins_h, ins_w,
    0,/*result_add*/
    0, /*ctrl*/
    with_bias,
    do_relu,
    neg_slope,
    0,/*rshift,*/
    oc, /*right_shift_len,*/
    pos_rshift, /*rshift_pos*/
    neg_rshift, /*rshift8_neg*/
    pos_m_i8, /*m_i8_pos*/
    neg_m_i8, /*m_i8_neg*/
    do_ic_alignment
    );

  return success();
}

LogicalResult tpu::TL_LG_BF16_Conv2DOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w;
  int sh, sw, pt, pb, pl, pr, dh, dw, pad_value;
  parseConvParam(param(), false, input(), output(),
                 n, ic, ih, iw, oc, oh, ow, g,
                 kh, kw, ins_h, ins_w, sh, sw, pt, pb, pl, pr,
                 dh, dw, is_dw, with_bias, do_relu, pad_value);

  laddr_t la_input = this->la_input();
  laddr_t la_output = this->la_output();
  laddr_t la_weight = this->la_filter();
  laddr_t la_working = this->la_working();

  laddr_t la_bias = 0;
  if (with_bias)
    la_bias = this->la_bias();

  // pad is not "SAME", can not get from conv param
  int ph_t = this->pad_top_h();
  int ph_b = this->pad_bottom_h();
  int pw_l = this->pad_left_w();
  int pw_r = this->pad_right_w();
  int layer_id = getOpLayerId(op);

  cvi_backend_bf16_tl_conv(
    *backend_ctx,
    layer_id,
    la_input, la_output, la_weight, la_working, la_bias,
    n, ic, ih, iw,
    g, oc, oh, ow, kh, kw, dh,
    dw, ph_t, ph_b, pw_l, pw_r, sh, sw, ins_h, ins_w,
    with_bias,
    do_relu
    );

  return success();
}


LogicalResult tpu::TL_LG_INT8_DeConv2DOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, pt, pb, pl, pr, dh, dw, pad_value;
  int no_use0, no_use1;
  parseConvParam(param(), false, input(), output(), n, ic, ih, iw, oc,
                 oh, ow, g, kh, kw, no_use0, no_use1, sh, sw, pt, pb, pl, pr, dh, dw, is_dw,
                 with_bias, do_relu, pad_value);

  laddr_t la_input = this->la_input();
  laddr_t la_output = this->la_output();
  laddr_t la_weight = this->la_filter();
  laddr_t la_perchannel = this->la_bias();
  bool do_ic_alignment = this->do_ic_alignment().hasValue()
                            ? this->do_ic_alignment().getValue() : false;

  // pad is not "SAME", can not get from conv param
  int ph_t = this->pad_top_h();
  int ph_b = this->pad_bottom_h();
  int pw_l = this->pad_left_w();
  int pw_r = this->pad_right_w();
  int ins_h = this->ins_h();
  int ins_last_h = this->ins_last_h();
  int ins_w = this->ins_w();
  int ins_last_w = this->ins_last_w();
  int layer_id = getOpLayerId(op);

  cvi_backend_tl_deconv(
    *backend_ctx,
    layer_id,
    la_input, la_output, la_weight, la_perchannel,
    n, ic, ih, iw,
    g, oc, oh, ow, kh, kw, dh, dw,
    ins_h, ins_last_h, ins_w, ins_last_w,
    ph_t, ph_b, pw_l, pw_r, sh, sw,
    with_bias,
    do_relu,   // do_activation,
    0,         //right_shift_width,
    oc,       // right_shift_array_len
    do_ic_alignment
    );

  return success();
}

LogicalResult tpu::TL_LG_BF16_DeConv2DOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g;
  int kh, kw, sh, sw, pt, pb, pl, pr, dh, dw, pad_value;
  int no_use0, no_use1;
  parseConvParam(param(), false, input(), output(), n, ic, ih, iw, oc,
                 oh, ow, g, kh, kw, no_use0, no_use1, sh, sw, pt, pb, pl, pr, dh, dw, is_dw,
                 with_bias, do_relu, pad_value);

  laddr_t la_input = this->la_input();
  laddr_t la_output = this->la_output();
  laddr_t la_weight = this->la_filter();
  laddr_t la_bias = 0;
  if (with_bias)
    la_bias = this->la_bias();

  // pad is not "SAME", can not get from conv param
  int ph_t = this->pad_top_h();
  int ph_b = this->pad_bottom_h();
  int pw_l = this->pad_left_w();
  int pw_r = this->pad_right_w();
  int ins_h = this->ins_h();
  int ins_last_h = this->ins_last_h();
  int ins_w = this->ins_w();
  int ins_last_w = this->ins_last_w();
  int layer_id = getOpLayerId(op);

  cvi_backend_tl_bf16_deconv(
    *backend_ctx,
    layer_id,
    la_input, la_output, la_weight, la_bias,
    n, ic, ih, iw,
    g, oc, oh, ow, kh, kw, dh, dw,
    ins_h, ins_last_h, ins_w, ins_last_w,
    ph_t, ph_b, pw_l, pw_r, sh, sw,
    with_bias,
    do_relu);

  return success();
}

LogicalResult tpu::TL_LG_INT8_EltwiseAddOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = getOpLayerId(op);

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  bool do_relu = this->do_relu();
  int nInputs = op->getNumOperands();
  assert(op->getNumOperands() == 2 && "support 2 inputs only");

  int64_t output_size, oh, ow;
  getTensorShapeAndSize(op->getResult(0), shape, output_size);
  oh = shape[2];
  ow = shape[3];

  std::vector<int32_t> la_input_array;
  auto la_input = new laddr_t[nInputs];
  arrayAttrToVector(this->la_input().getValue(), la_input_array);
  for (int i = 0; i < nInputs; ++i) {
      la_input[i] = la_input_array[i];
  }

  laddr_t la_output = this->la_output();
  laddr_t la_working = this->la_working();

  int8_t rshift;
  int8_t m_i8_input[2];
  if (this->rshift().hasValue() && this->m_i8().hasValue()) {
    rshift = this->rshift().getValue();

    std::vector<int32_t> m_i8_inputs_array;
    arrayAttrToVector(this->m_i8().getValue(), m_i8_inputs_array);
    assert(m_i8_inputs_array.size() == op->getNumOperands());
    assert(m_i8_inputs_array.size() >= 2);
    m_i8_input[0] = static_cast<int8_t>(m_i8_inputs_array[0]);
    m_i8_input[1] = static_cast<int8_t>(m_i8_inputs_array[1]);
  }

  bool do_early_stride = this->do_early_stride();
  int32_t early_stride_h = this->early_stride_h();
  int32_t early_stride_w = this->early_stride_w();
  if (do_early_stride) {
    assert(oh == h / early_stride_h);
    assert(ow == w / early_stride_w);
  }

  // op code PROD = 0; SUM = 1; MAX = 2;
  int op_code = 1;
  const int coeffs[2] = {1, 1};

  cvi_backend_tl_eltwise( *backend_ctx,
                          layer_id, /*u32 layer_id,*/
                          la_input,
                          la_output,
                          la_working,
                          n, c, h, w, nInputs,
                          op_code,
                          rshift,
                          m_i8_input,
                          true, /*use_default_coeff,*/
                          do_relu,
                          0, /*relu_slope,*/
                          coeffs,
                          0,
                          do_early_stride, early_stride_h, early_stride_w);
  delete[] la_input;
  return success();
}

LogicalResult tpu::TL_LG_BF16_EltwiseAddOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = mlir::getOpLayerId(op);

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  bool do_relu = this->do_relu();
  int nInputs = op->getNumOperands();
  assert(op->getNumOperands() == 2 && "support 2 inputs only");

  int64_t output_size, oh, ow;
  getTensorShapeAndSize(op->getResult(0), shape, output_size);
  oh = shape[2];
  ow = shape[3];

  std::vector<int32_t> la_input_array;
  laddr_t la_input[nInputs];
  arrayAttrToVector(this->la_input().getValue(), la_input_array);
  for (int i = 0; i < nInputs; ++i) {
      la_input[i] = la_input_array[i];
  }
  std::vector<float> coeffs;
  arrayAttrToVector(this->coeff().getValue(), coeffs);

  laddr_t la_output = this->la_output();

  bool do_early_stride = this->do_early_stride();
  int32_t early_stride_h = this->early_stride_h();
  int32_t early_stride_w = this->early_stride_w();
  if (do_early_stride) {
    assert(oh == h / early_stride_h);
    assert(ow == w / early_stride_w);
  }

  // op code PROD = 0; SUM = 1; MAX = 2;
  int op_code = 1;
  cvi_backend_bf16_tl_eltwise(*backend_ctx,
                              layer_id, /*u32 layer_id,*/
                              la_input,
                              la_output,
                              n, c, h, w, nInputs,
                              op_code,
                              true, /*use_default_coeff,*/
                              do_relu,
                              0, /*relu_slope,*/
                              coeffs.data(),
                              do_early_stride,
                              early_stride_h, early_stride_w);

  return success();
}

LogicalResult tpu::TL_LG_INT8_EltwiseMulOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = getOpLayerId(op);

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  bool do_relu = this->do_relu();
  int nInputs = op->getNumOperands();
  assert(op->getNumOperands() == 2 && "support 2 inputs only");

  std::vector<int32_t> la_input_array;
  auto la_input = new laddr_t[nInputs];
  arrayAttrToVector(this->la_input().getValue(), la_input_array);
  for (int i = 0; i < nInputs; ++i) {
    la_input[i] = la_input_array[i];
  }

  laddr_t la_output = this->la_output();
  laddr_t la_working = this->la_working();

  int8_t rshift;
  int32_t m_i32;
  if (this->rshift().hasValue() && this->m_i32_output().hasValue()) {
    rshift = this->rshift().getValue();
    m_i32 = this->m_i32_output().getValue();
  }

  // op code PROD = 0; SUM = 1; MAX = 2;
  int op_code = 0;
  const int coeffs[2] = {1, 1};

  cvi_backend_tl_eltwise( *backend_ctx,
                          layer_id, /*u32 layer_id,*/
                          la_input,
                          la_output,
                          la_working,
                          n, c, h, w, nInputs,
                          op_code,
                          rshift,
                          0, /*m_i8*/
                          true, /*use_default_coeff,*/
                          do_relu,
                          0, /*relu_slope,*/
                          coeffs,
                          m_i32,
                          0, 0, 0);
  delete[] la_input;
  return success();
}

LogicalResult tpu::TL_LG_BF16_EltwiseMulOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = mlir::getOpLayerId(op);

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  bool do_relu = this->do_relu();
  int nInputs = op->getNumOperands();
  assert(op->getNumOperands() == 2 && "support 2 inputs only");

  std::vector<int32_t> la_input_array;
  laddr_t la_input[nInputs];
  arrayAttrToVector(this->la_input().getValue(), la_input_array);
  for (int i = 0; i < nInputs; ++i) {
      la_input[i] = la_input_array[i];
  }

  laddr_t la_output = this->la_output();

  // op code PROD = 0; SUM = 1; MAX = 2;
  int op_code = 0;
  const float coeffs[2] = {1.0, 1.0};

  cvi_backend_bf16_tl_eltwise(*backend_ctx,
                              layer_id, /*u32 layer_id,*/
                              la_input,
                              la_output,
                              n, c, h, w, nInputs,
                              op_code,
                              true, /*use_default_coeff,*/
                              do_relu,
                              0, /*relu_slope,*/
                              coeffs,
                              0,
                              0, 0);

  return success();
}

LogicalResult tpu::TL_LG_INT8_LrnOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = getOpLayerId(op);

  laddr_t la_input = this->la_input();
  laddr_t la_output = this->la_output();
  laddr_t la_working = this->la_working();
  laddr_t la_sqrt = this->la_sqrt();
  laddr_t la_power = this->la_power();
  int local_size = this->local_size();
  int8_t sum_rshift_i8 = this->sum_rshift();
  int8_t lrn_rshift_i8 = this->lrn_rshift();
  int8_t m_i8[2];
  m_i8[0] = this->quant_data0();
  m_i8[1] = this->quant_data1();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  cvi_backend_tl_lrn( *backend_ctx,
                      layer_id,
                      la_input,
                      la_output,
                      la_sqrt,
                      la_power,
                      la_working,
                      n, c, h, w, local_size,
                      sum_rshift_i8,
                      lrn_rshift_i8,
                      m_i8);
  return success();
}

LogicalResult tpu::TL_LG_BF16_LrnOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = mlir::getOpLayerId(op);

  laddr_t la_input = this->la_input();
  laddr_t la_output = this->la_output();
  laddr_t la_working = this->la_working();
  laddr_t la_power_exp_table = this->la_sqrt();
  laddr_t la_power_mantissa_table = this->la_power();
  int local_size = this->local_size();
  float alpha = this->alpha().convertToFloat();
  float k = this->k().convertToFloat();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  cvi_backend_bf16_tl_lrn( *backend_ctx,
                          layer_id,
                          la_input,
                          la_output,
                          la_power_exp_table,
                          la_power_mantissa_table,
                          la_working,
                          n, c, h, w, local_size,
                          alpha, k);

  return success();
}

LogicalResult tpu::TL_LG_INT8_MulConstOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = getOpLayerId(op);

  laddr_t la_input = this->la_input();
  laddr_t la_output = this->la_output();
  bool do_relu = this->do_relu();
  float const_val = this->const_val().convertToFloat();

  auto shape = getTensorShape(input());
  int64_t n, c, h, w;
  getNCHW(shape, n, c, h, w);

  cvi_backend_tl_mul_const(*backend_ctx, layer_id, la_input, la_output,
                           const_val, do_relu, n, c, h, w, CVK_FMT_I8);
  return success();
}

LogicalResult tpu::TL_LG_BF16_MulConstOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = getOpLayerId(op);

  laddr_t la_input = this->la_input();
  laddr_t la_output = this->la_output();
  bool do_relu = this->do_relu();
  float const_val = this->const_val().convertToFloat();

  auto shape = getTensorShape(input());
  int64_t n, c, h, w;
  getNCHW(shape, n, c, h, w);

  cvi_backend_tl_mul_const(*backend_ctx, layer_id, la_input, la_output,
                           const_val, do_relu, n, c, h, w, CVK_FMT_BF16);
  return success();
}

LogicalResult tpu::TL_LG_INT8_LutOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = getOpLayerId(op);

  laddr_t la_input = this->la_input();
  laddr_t la_output = this->la_output();
  laddr_t la_y_table = this->la_y_table();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  cvi_backend_int8_tl_lut( *backend_ctx,
                      layer_id,
                      la_input,
                      la_output,
                      la_y_table,
                      n, c, h, w);
  return success();

}


// two kinds of bf16 lookup table usage:
// 1. For tanh/sigmoid, calculate y0 + slope at [-8, 8] to close result
// 2. For reciprocal/sqrt/power, calcuate mantissa and exp, and mul them
LogicalResult tpu::TL_LG_BF16_LutOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = getOpLayerId(op);

  laddr_t la_input = this->la_input();
  laddr_t la_output = this->la_output();
  laddr_t la_working = this->la_working();
  laddr_t la_slope_lut = this->la_slope_lut();
  laddr_t la_y_table = this->la_y_table();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  float table_thresh_min = this->min_range().convertToFloat();
  float table_thresh_max = this->max_range().convertToFloat();
  auto lut_method = method().getValue().str();
  // method 0: mantissa, 1: slope
  int method = 0;
  if(lut_method == "mantissa")
    method = 0;
  else if (lut_method == "slope")
    method = 1;

  cvi_backend_bf16_tl_lut( *backend_ctx,
                          layer_id,
                          la_input,
                          la_output,
                          la_working,
                          la_y_table,
                          la_slope_lut,
                          table_thresh_min,
                          table_thresh_max,
                          n, c, h, w, method);
  return success();
}

LogicalResult tpu::TL_LG_INT8_ScaleLutOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = getOpLayerId(op);

  laddr_t la_input = this->la_input();
  laddr_t la_output = this->la_output();
  laddr_t la_table = this->la_table();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  cvi_backend_tl_scale_lut(*backend_ctx, layer_id, la_input, la_output,
                           la_table, n, c, h, w);
  return success();
}

LogicalResult tpu::TL_LG_QuantOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = getOpLayerId(op);

  laddr_t la_input = this->la_input();
  laddr_t la_output = this->la_output();
  laddr_t la_working = this->la_working();
  bool bExtraInput = this->bExtraInput();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  cvk_fmt_t from, to;
  if (this->from() == "BF16") {
    from = CVK_FMT_BF16;
  } else if (this->from() == "INT8") {
    from = CVK_FMT_I8;
  } else if (this->from() == "UINT8") {
    from = CVK_FMT_U8;
  } else {
    llvm_unreachable("current `from` only support int8/bf16");
  }

  if (this->to() == "BF16") {
    to = CVK_FMT_BF16;
  } else if (this->to() == "INT8") {
    to = CVK_FMT_I8;
  } else {
    llvm_unreachable("current `to` only support int8/bf16");
  }

  // FIXME: support U8 type
  cvi_backend_tl_quant(*backend_ctx,
                      layer_id,
                      la_input,
                      la_output,
                      la_working,
                      from, to,
                      this->const_scale().convertToFloat(),
                      n, c, h, w,
                      bExtraInput);
  return success();
}

LogicalResult tpu::TL_LG_INT8_ConcatOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  unsigned nInputs = op->getNumOperands();
  int layer_id = getOpLayerId(op);
  int axis = this->axis();

  std::vector<int32_t> la_input_array;
  std::vector<laddr_t> la_input(nInputs);
  arrayAttrToVector(this->la_input().getValue(), la_input_array);
  for (unsigned i = 0; i < nInputs; ++i) {
    la_input[i] = static_cast<laddr_t>(la_input_array[i]);
  }

  laddr_t la_output = this->la_output();

  std::vector<int32_t> input_dims(nInputs);
  for (unsigned i = 0; i < nInputs; i++) {
    std::vector<int64_t> shape = getTensorShape(op->getOperand(i));
    input_dims[i] = shape[axis];
  }
  std::vector<int> output_dim;
  std::vector<int64_t> output_shape = getTensorShape(getResult());
  for (auto &dim : output_shape) {
    output_dim.push_back(dim);
  }
  for (uint32_t i = output_dim.size(); i < 4; i++) {
    output_dim.push_back(1); // fill to 4 dim
  }

  // prepare quant info
  int32_t *r_i8 = nullptr;
  int32_t *m_i8 = nullptr;
  std::vector<int32_t> m_i8_array;
  std::vector<int32_t> r_i8_array;
  if (this->r_i8().hasValue() && this->m_i8().hasValue()) {
    arrayAttrToVector(this->m_i8().getValue(), m_i8_array);
    m_i8 = m_i8_array.data();
    arrayAttrToVector(this->r_i8().getValue(), r_i8_array);
    r_i8 = r_i8_array.data();
  }

  cvi_backend_tl_concat(*backend_ctx, layer_id, input_dims.data(), nInputs,
                        output_dim.data(), la_input.data(), la_output,
                        do_relu(), r_i8, m_i8);
  return success();
}

LogicalResult tpu::TL_LG_BF16_ConcatOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  unsigned nInputs = op->getNumOperands();
  int layer_id = getOpLayerId(op);
  int axis = this->axis();

  std::vector<int32_t> la_input_array;
  std::vector<laddr_t> la_input(nInputs);
  arrayAttrToVector(this->la_input().getValue(), la_input_array);
  for (unsigned i = 0; i < nInputs; ++i) {
    la_input[i] = static_cast<laddr_t>(la_input_array[i]);
  }

  laddr_t la_output = this->la_output();

  std::vector<int32_t> input_dims(nInputs);
  for (unsigned i = 0; i < nInputs; i++) {
    std::vector<int64_t> shape = getTensorShape(op->getOperand(i));
    input_dims[i] = shape[axis];
  }
  std::vector<int> output_dim;
  std::vector<int64_t> shape = getTensorShape(this->getResult());
  for (auto &dim : shape) {
    output_dim.push_back(dim);
  }
  for (uint32_t i = output_dim.size(); i < 4; i++) {
    output_dim.push_back(1); // fill to 4 dim
  }

  cvi_backend_tl_bf16_concat(*backend_ctx, layer_id, input_dims.data(), nInputs,
                             output_dim.data(), la_input.data(), la_output,
                             do_relu());
  return success();
}

LogicalResult tpu::TL_LG_LoadNeuronOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL Load Neuron codegen.\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = getOpLayerId(op);

  std::vector<int64_t> shape;
  int64_t n, c, h, w;
  shape = getTensorShape(op->getOperand(0));
  getNCHW(shape, n, c, h, w);

  int global_c = c;
  int global_h = h;
  int global_w = w;

  shape = getTensorShape(op->getResult(0));
  getNCHW(shape, n, c, h, w);

  int local_n = n;
  int local_c = c;
  int local_h = h;
  int local_w = w;


  gaddr_t src_gaddr = this->gaddr().getValue();
  laddr_t dst_laddr = this->laddr().getValue();
  bool transpose = this->transpose();
  bool aligned = this->align();
  bool isNeuron = true;

  cvk_fmt_t from, to;
  RankedTensorType in_type = this->getOperand().getType().cast<RankedTensorType>();
  RankedTensorType out_type = this->getResult().getType().cast<RankedTensorType>();

  // convert type to `cvi_backend_fmt`
  if (in_type.getElementType().isBF16()) {
    from = CVK_FMT_BF16;
  }
  else if (in_type.getElementType().isInteger(8)) {
    // int8
    from = CVK_FMT_I8;
  }
  else {
    llvm_unreachable("current `from` only support int8/bf16");
  }

  // convert type to `cvi_backend_fmt`
  if (out_type.getElementType().isBF16()) {
    to = CVK_FMT_BF16;
  }
  else if (out_type.getElementType().isInteger(8)) {
    // int8
    to = CVK_FMT_I8;
  }
  else {
    llvm_unreachable("current `to` only support int8/bf16");
  }

  bool do_decompress = this->load_compr_act().hasValue() ?
                       this->load_compr_act().getValue() : false;

  if (!do_decompress) {
    cvi_backend_tl_load_stride( *backend_ctx,
                                layer_id,
                                src_gaddr,
                                dst_laddr,
                                local_n, local_c, local_h, local_w,
                                global_c, global_h, global_w,
                                transpose, aligned, isNeuron,
                                from, to
                                );
  } else {
    int step_size = 0;
    int c_step = local_c;
    int h_step = local_h;
    if (this->compr_act_param().hasValue()) {
      c_step = this->compr_act_param().getValue().c_step().getInt();
      h_step = this->compr_act_param().getValue().h_step().getInt();
      step_size = this->compr_act_param().getValue().step_size().getInt();
    }

    cvi_backend_tl_load_compressed( *backend_ctx,
                                    layer_id,
                                    src_gaddr,
                                    dst_laddr,
                                    local_n, local_c, local_h, local_w,
                                    global_c, global_h, global_w,
                                    transpose, aligned, isNeuron,
                                    from, to,
                                    h_step, step_size, c_step
                                    );
  }
  return success();
}

LogicalResult tpu::TL_LG_LoadCoeffOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL Load Coeff codegen.\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = 0;

  std::vector<int64_t> shape;
  int64_t n, c, h, w;
  shape = getTensorShape(op->getResult(0));
  getNCHW(shape, n, c, h, w);


  int local_n = n;
  int local_c = c;
  int local_h = h;
  int local_w = w;

  int global_c = c;
  int global_h = h;
  int global_w = w;

  gaddr_t src_gaddr = this->offset().getValue();
  laddr_t dst_laddr = this->laddr().getValue();
  bool transpose = this->transpose();
  bool aligned = this->align();
  bool isNeuron = false;
  bool bcompressed = false;
  if (this->compressed_weight().hasValue())
    bcompressed = this->compressed_weight().getValue();

  cvk_fmt_t from, to;
  // Coeff dont need to quant, just check result type
  RankedTensorType out_type = this->getResult().getType().cast<RankedTensorType>();

  // convert type to `cvi_backend_fmt`
  if (out_type.getElementType().isBF16() ||
      out_type.getElementType().isInteger(16)) {
    from = CVK_FMT_BF16;
    to = CVK_FMT_BF16;
  } else if (out_type.getElementType().isInteger(8)) {
    // int8
    from = CVK_FMT_I8;
    to = CVK_FMT_I8;
  } else if (out_type.getElementType().isInteger(16)) {
    // int16
    from = CVK_FMT_I8;
    to = CVK_FMT_I8;
    local_w *= 2;
    global_w *= 2;
  } else {
    llvm_unreachable("current `from/to` only support int8/bf16");
  }

  cvi_backend_tl_load_stride( *backend_ctx,
                              layer_id,
                              src_gaddr,
                              dst_laddr,
                              local_n, local_c, local_h, local_w,
                              global_c, global_h, global_w,
                              transpose, aligned, isNeuron,
                              from, to,
                              bcompressed);
  return success();
}

LogicalResult tpu::TL_LG_StoreOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL Store codegen.\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = getOpLayerId(op);

  std::vector<int64_t> shape;
  int64_t n, c, h, w;
  shape = getTensorShape(op->getOperand(0));
  getNCHW(shape, n, c, h, w);

  int local_n = n;
  int local_c = c;
  int local_h = h;
  int local_w = w;

  shape = getTensorShape(op->getResult(0));
  getNCHW(shape, n, c, h, w);

  int global_c = c;
  int global_h = h;
  int global_w = w;

  gaddr_t src_gaddr = this->gaddr().getValue();
  laddr_t dst_laddr = this->laddr().getValue();
  bool transpose = this->transpose();
  bool aligned = this->align();
  bool isNeuron = true;

  cvk_fmt_t from, to;
  RankedTensorType in_type = this->getOperand().getType().cast<RankedTensorType>();
  RankedTensorType out_type = this->getResult().getType().cast<RankedTensorType>();

  // convert type to `cvi_backend_fmt`
  if (in_type.getElementType().isBF16()) {
    from = CVK_FMT_BF16;
  }
  else if (in_type.getElementType().isInteger(8)) {
    // int8
    from = CVK_FMT_I8;
  }
  else {
    llvm_unreachable("current `from` only support int8/bf16");
  }

  // convert type to `cvi_backend_fmt`
  if (out_type.getElementType().isBF16()) {
    to = CVK_FMT_BF16;
  }
  else if (out_type.getElementType().isInteger(8)) {
    // int8
    to = CVK_FMT_I8;
  }
  else {
    llvm_unreachable("current `to` only support int8/bf16");
  }

  bool do_compress = this->store_compr_act().hasValue() ?
                     this->store_compr_act().getValue() : false;
  int step_size = 0;
  int c_step = local_c;
  int h_step = local_h;
  if (this->compr_act_param().hasValue()) {
    h_step = this->compr_act_param().getValue().h_step().getInt();
    c_step = this->compr_act_param().getValue().c_step().getInt();
    step_size = this->compr_act_param().getValue().step_size().getInt();
  }

  if (!do_compress) {
    cvi_backend_tl_store_stride( *backend_ctx,
                                  layer_id,
                                  src_gaddr,
                                  dst_laddr,
                                  local_n, local_c, local_h, local_w,
                                  global_c, global_h, global_w,
                                  transpose, aligned, isNeuron,
                                  from, to
                                  );
  } else {
    cvi_backend_tl_store_compressed( *backend_ctx,
                                     layer_id,
                                     src_gaddr,
                                     dst_laddr,
                                     local_n, local_c, local_h, local_w,
                                     global_c, global_h, global_w,
                                     transpose, aligned, isNeuron,
                                     from, to,
                                     h_step, step_size, c_step
                                     );
  }

  return success();
}

LogicalResult tpu::TL_LG_JoinOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL join codegen.\n";);
  return success();
}

LogicalResult tpu::TL_LG_CopyOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL Copy codegen.\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = 0;

  std::vector<int64_t> shape;
  int64_t n, c, h, w;
  shape = getTensorShape(op->getOperand(0));
  getNCHW(shape, n, c, h, w);

  laddr_t la_src = this->la_src().getValue();
  laddr_t la_dst = this->la_dst().getValue();
  bool align = this->align();
  cvk_fmt_t fmt = CVK_FMT_I8;
  RankedTensorType out_type = this->getResult().getType().cast<RankedTensorType>();

  // convert type to `cvi_backend_fmt`
  if (out_type.getElementType().isBF16() ||
      out_type.getElementType().isInteger(16))
      fmt = CVK_FMT_BF16;

  cvi_backend_tl_copy(*backend_ctx,
                      layer_id,
                      la_src,
                      la_dst,
                      n, c, h, w,
                      align,
                      fmt);

  return success();
}

LogicalResult tpu::TL_LG_INT8_PoolAvg2DOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = getOpLayerId(op);

  bool is_global, do_relu, count_include_pad;
  int n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  parsePoolParam(param(), input(), output(),
                 n, c, ih, iw, oh, ow,
                 kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
                 is_global, do_relu, count_include_pad);

  laddr_t la_input = this->la_input();
  laddr_t la_output = this->la_output();

  assert(this->rshift().hasValue());
  int8_t rshift_i8 = this->rshift().getValue();
  assert(this->m_i8().hasValue());
  int8_t m_i8 = this->m_i8().getValue();

  cvi_backend_tl_pooling( *backend_ctx,
                          layer_id,
                          la_input, la_output,
                          n, c, ih, iw,
                          n, c, oh, ow,
                          kh, kw, sh, sw,
                          pt, pb, pl, pr,
                          true, /*is_avg_pooling,*/
                          rshift_i8,
                          m_i8);

  return success();
}

LogicalResult tpu::TL_LG_BF16_PoolAvg2DOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = getOpLayerId(op);

  bool is_global, do_relu, count_include_pad;
  int n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  parsePoolParam(param(), input(), output(),
                 n, c, ih, iw, oh, ow,
                 kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
                 is_global, do_relu, count_include_pad);

  laddr_t la_input = this->la_input();
  laddr_t la_output = this->la_output();

  cvi_backend_tl_bf16_pooling( *backend_ctx,
                                layer_id,
                                la_input, la_output,
                                n, c, ih, iw,
                                n, c, oh, ow,
                                kh, kw, sh, sw,
                                pt, pb, pl, pr,
                                true/*is_avg_pooling,*/);
  return success();
}

LogicalResult tpu::TL_LG_INT8_PoolMax2DOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = getOpLayerId(op);

  bool is_global, do_relu, count_include_pad;
  int n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  parsePoolParam(param(), input(), output(),
                 n, c, ih, iw, oh, ow,
                 kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
                 is_global, do_relu, count_include_pad);

  laddr_t la_input = this->la_input();
  laddr_t la_output = this->la_output();
  int8_t rshift_i8 = 0, multiplier_i8 = 1;

  cvi_backend_tl_pooling( *backend_ctx,
                          layer_id,
                          la_input, la_output,
                          n, c, ih, iw,
                          n, c, oh, ow,
                          kh, kw, sh, sw,
                          pt, pb, pl, pr,
                          false, /*is_avg_pooling,*/
                          rshift_i8,
                          multiplier_i8);

  return success();
}

LogicalResult tpu::TL_LG_BF16_PoolMax2DOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
              << " [" << getOpName() << "]\n";);

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = getOpLayerId(op);

  bool is_global, do_relu, count_include_pad;
  int n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  parsePoolParam(param(), input(), output(),
                 n, c, ih, iw, oh, ow,
                 kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
                 is_global, do_relu, count_include_pad);

  laddr_t la_input = this->la_input();
  laddr_t la_output = this->la_output();

  cvi_backend_tl_bf16_pooling( *backend_ctx,
                                layer_id,
                                la_input, la_output,
                                n, c, ih, iw,
                                n, c, oh, ow,
                                kh, kw, sh, sw,
                                pt, pb, pl, pr,
                                false/*is_avg_pooling,*/);

  return success();
}

LogicalResult tpu::TL_LG_INT8_ScaleOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  bool do_relu = this->do_relu();

  laddr_t la_input = this->la_input();
  laddr_t la_output = this->la_output();
  laddr_t la_scale = this->la_scale();
  laddr_t la_bias = this->la_bias();
  int layer_id = getOpLayerId(op);

  cvi_backend_tl_scale(
      *backend_ctx, // ctx
      layer_id,     // layer_id
      la_input,     // input_addr
      la_scale,    // scale_addr
      la_bias,      // pack_addr
      la_output,    // output_addr
      n, c, h, w,
      n * c,        // scale_dim (axis = 1  =>  n * c)
      h * w,        // inner_dim (axis = 1  =>  h * w)
      false,        // is_scale_const
      0,            // const_scale
      0,
      do_relu,      // do_activation,
      0,            // activation_method
      nullptr,      // activation_arg
      nullptr,      // multiplier
      false);        // with_bias

  return success();
}

LogicalResult tpu::TL_LG_BF16_ScaleOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);

 CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  bool do_relu = this->do_relu();

  laddr_t la_input = this->la_input();
  laddr_t la_output = this->la_output();
  laddr_t la_scale = this->la_scale();
  laddr_t la_bias = this->la_bias();
  int layer_id = mlir::getOpLayerId(op);

  cvi_backend_bf16_tl_scale(
      *backend_ctx, // ctx
      layer_id,     // layer_id
      la_input,     // input_addr
      la_scale,    // scale_addr
      la_bias,      // pack_addr
      la_output,    // output_addr
      n, c, h, w,
      do_relu,      // do_activation,
      0,            // activation_method
      nullptr,      // activation_arg
      false);        // with_bias

  return success();
}

LogicalResult tpu::TL_LG_INT8_UpsampleOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  laddr_t la_input = this->la_input();
  laddr_t la_output = this->la_output();
  auto scale_h = this->scale_h();
  auto scale_w = this->scale_w();
  int layer_id = getOpLayerId(op);

  cvi_backend_tl_upsample(
      *backend_ctx,
      layer_id, //layer_id,
      la_input,
      la_output,
      n,
      c,
      h,
      w,
      scale_h,
      scale_w
  );
  return success();
}

LogicalResult tpu::TL_LG_BF16_UpsampleOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  assert(0);
  return success();
}

LogicalResult tpu::TL_LG_INT8_LeakyReluOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  int8_t pos_rshift, pos_m_i8, neg_rshift, neg_m_i8;
  float neg_slope;
  parseTLLeakyReluParam(
    op, pos_rshift, pos_m_i8, neg_rshift, neg_m_i8, neg_slope);

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  laddr_t la_input = this->la_input();
  laddr_t la_output = this->la_output();
  int layer_id = getOpLayerId(op);

  cvi_backend_tl_leaky_relu(
      *backend_ctx,
      layer_id, //layer_id,
      la_input,
      la_output,
      n,
      c,
      h,
      w,
      pos_rshift, neg_rshift, pos_m_i8, neg_m_i8
  );
  return success();
}

LogicalResult tpu::TL_LG_BF16_LeakyReluOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  auto lreluOp = llvm::dyn_cast<tpu::TL_LG_BF16_LeakyReluOp>(op);
  float negative_slope = lreluOp.negative_slope().convertToFloat();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  laddr_t la_input = this->la_input();
  laddr_t la_output = this->la_output();
  laddr_t la_working = this->la_working();
  int layer_id = mlir::getOpLayerId(op);

  cvi_backend_bf16_tl_leaky_relu(
      *backend_ctx,
      layer_id, //layer_id,
      la_input,
      la_output,
      la_working,
      n,
      c,
      h,
      w,
      negative_slope);

  return success();
}

LogicalResult tpu::TL_LG_INT8_PReluOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  int8_t r_i8_pos, m_i8_pos, r_i8_neg;
  auto prelu_op = llvm::dyn_cast<tpu::TL_LG_INT8_PReluOp>(op);
  assert(prelu_op);

  if (prelu_op.m_i8_pos().hasValue()) {
    m_i8_pos = prelu_op.m_i8_pos().getValue();
    r_i8_pos = prelu_op.r_i8_pos().getValue();
  } else {
    m_i8_pos = 0;
    r_i8_pos = 0;
  }

  if (prelu_op.r_i8_neg().hasValue()) {
    r_i8_neg = prelu_op.r_i8_neg().getValue();
  } else {
    r_i8_neg = 0;
  }

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  laddr_t la_input = this->la_input();
  laddr_t la_output = this->la_output();
  laddr_t la_slope = this->la_slope();
  int layer_id = getOpLayerId(op);

  cvi_backend_tl_prelu(
      *backend_ctx,
      layer_id, //layer_id,
      la_input,
      la_output,
      la_slope,
      n,
      c,
      h,
      w,
      r_i8_pos, m_i8_pos, r_i8_neg);

  return success();
}

LogicalResult tpu::TL_LG_BF16_PReluOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  auto input_shape = getTensorShape(op->getOperand(0));
  int64_t n, c, h, w;
  getNCHW(input_shape, n, c, h, w);

  laddr_t la_input = this->la_input();
  laddr_t la_output = this->la_output();
  laddr_t la_slope = this->la_slope();
  int layer_id = mlir::getOpLayerId(op);

  cvi_backend_tl_bf16_prelu(
      *backend_ctx,
      layer_id, //layer_id,
      la_input,
      la_output,
      la_slope,
      n,
      c,
      h,
      w);

  return success();
}

LogicalResult tpu::TL_LG_INT8_PadOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  auto input_shape = getTensorShape(op->getOperand(0));
  auto output_shape = getTensorShape(op->getResult(0));

  laddr_t la_input = this->la_input();
  laddr_t la_output = this->la_output();

  // parse param
  int layer_id = getOpLayerId(op);
  std::vector<int32_t> pads;
  auto const_val = this->const_val().convertToFloat();
  arrayAttrToVector(this->pads(), pads);

  cvi_backend_tl_pad(
      *backend_ctx,
      layer_id, //layer_id,
      input_shape.data(),
      output_shape.data(),
      la_input,
      la_output,
      const_val,
      pads.data()
  );
  return success();
}

LogicalResult tpu::TL_LG_BF16_PadOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  auto input_shape = getTensorShape(op->getOperand(0));
  auto output_shape = getTensorShape(op->getResult(0));

  laddr_t la_input = this->la_input();
  laddr_t la_output = this->la_output();

  // parse param
  int layer_id = getOpLayerId(op);
  std::vector<int32_t> pads;
  auto const_val = this->const_val().convertToFloat();
  arrayAttrToVector(this->pads(), pads);

  cvi_backend_tl_bf16_pad(
      *backend_ctx,
      layer_id, //layer_id,
      input_shape.data(),
      output_shape.data(),
      la_input,
      la_output,
      const_val,
      pads.data()
  );
  return success();
}

LogicalResult tpu::TL_LG_INT8_CropOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  auto input_shape = getTensorShape(op->getOperand(0));
  auto output_shape = getTensorShape(op->getResult(0));

  for (uint32_t i = input_shape.size(); i < 4; i++) {
    input_shape.push_back(1);
  }
  for (uint32_t i = output_shape.size(); i < 4; i++) {
    output_shape.push_back(1);
  }

  laddr_t la_input = this->la_input();
  laddr_t la_output = this->la_output();

  // parse param
  int layer_id = getOpLayerId(op);
  std::vector<int32_t> crop_offsets;
  arrayAttrToVector(this->crop_offsets(), crop_offsets);

  cvi_backend_tl_crop(*backend_ctx, layer_id, input_shape.data(),
                      output_shape.data(), la_input, la_output,
                      crop_offsets.data());
  return success();
}

LogicalResult tpu::TL_LG_BF16_CropOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  auto input_shape = getTensorShape(op->getOperand(0));
  auto output_shape = getTensorShape(op->getResult(0));
  for (uint32_t i = input_shape.size(); i < 4; i++) {
    input_shape.push_back(1);
  }
  for (uint32_t i = output_shape.size(); i < 4; i++) {
    output_shape.push_back(1);
  }
  laddr_t la_input = this->la_input();
  laddr_t la_output = this->la_output();

  // parse param
  int layer_id = mlir::getOpLayerId(op);
  std::vector<int32_t> crop_offsets;
  arrayAttrToVector(this->crop_offsets(), crop_offsets);

  cvi_backend_tl_bf16_crop(*backend_ctx, layer_id, input_shape.data(),
                           output_shape.data(), la_input, la_output,
                           crop_offsets.data());
  return success();
}

LogicalResult tpu::TL_LG_INT8_ReluOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  auto input_shape = getTensorShape(op->getOperand(0));
  int64_t n, c, h, w;
  getNCHW(input_shape, n, c, h, w);

  laddr_t la_input = this->la_input();
  laddr_t la_output = this->la_output();

  // parse param
  int layer_id = getOpLayerId(op);

  cvi_backend_tl_relu(
      *backend_ctx,
      layer_id, //layer_id,
      n, c, h, w,
      la_input,
      la_output
  );
  return success();
}

LogicalResult tpu::TL_LG_BF16_ReluOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
    CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  auto input_shape = getTensorShape(op->getOperand(0));
  int64_t n, c, h, w;
  getNCHW(input_shape, n, c, h, w);

  laddr_t la_input = this->la_input();
  laddr_t la_output = this->la_output();

  // parse param
  int layer_id = getOpLayerId(op);

  cvi_backend_tl_bf16_relu(
      *backend_ctx,
      layer_id, //layer_id,
      n, c, h, w,
      la_input,
      la_output
  );
  return success();
}

LogicalResult tpu::TL_LG_INT8_SliceOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  auto input_shape = getTensorShape(op->getOperand(0));
  auto output_shape = getTensorShape(op->getResult(0));

  laddr_t la_input = this->la_input();
  laddr_t la_output = this->la_output();
  int offset = this->offset();
  int axis = this->axis();
  int layer_id = getOpLayerId(op);

  cvi_backend_tl_slice(
      *backend_ctx,
      layer_id, //layer_id,
      input_shape.data(),
      output_shape.data(),
      la_input,
      la_output,
      axis,
      offset
  );
  return success();
}

LogicalResult tpu::TL_LG_BF16_SliceOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  auto input_shape = getTensorShape(op->getOperand(0));
  auto output_shape = getTensorShape(op->getResult(0));

  laddr_t la_input = this->la_input();
  laddr_t la_output = this->la_output();
  int layer_id = getOpLayerId(op);
  auto offset = this->offset();
  auto axis = this->axis();

  cvi_backend_tl_bf16_slice(
      *backend_ctx,
      layer_id, //layer_id,
      input_shape.data(),
      output_shape.data(),
      la_input,
      la_output,
      axis,
      offset
  );
  return success();
}

LogicalResult tpu::TL_LG_INT8_SwapChannelOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  auto input_shape = getTensorShape(op->getOperand(0));
  int64_t n, c, h, w;
  getNCHW(input_shape, n, c, h, w);
  laddr_t la_input = this->la_input();
  laddr_t la_output = this->la_output();
  int layer_id = getOpLayerId(op);
  std::vector<int32_t> order;
  arrayAttrToVector(this->channel_order(), order);
  cvi_backend_tl_swap_channel(*backend_ctx, layer_id, la_input, la_output, n, c,
                              h, w, order.data(), CVK_FMT_I8);
  return success();
}

LogicalResult tpu::TL_LG_BF16_SwapChannelOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  auto input_shape = getTensorShape(op->getOperand(0));
  int64_t n, c, h, w;
  getNCHW(input_shape, n, c, h, w);
  laddr_t la_input = this->la_input();
  laddr_t la_output = this->la_output();
  int layer_id = getOpLayerId(op);
  std::vector<int32_t> order;
  arrayAttrToVector(this->channel_order(), order);
  cvi_backend_tl_swap_channel(*backend_ctx, layer_id, la_input, la_output, n, c,
                              h, w, order.data(), CVK_FMT_BF16);
  return success();
}

LogicalResult tpu::TL_LG_BF16_LayerNormOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  auto input_shape = getTensorShape(op->getOperand(0));
  int64_t n, c, h, w;
  getNCHW(input_shape, n, c, h, w);
  laddr_t la_input = this->la_input();
  laddr_t la_output = this->la_output();
  laddr_t la_table = this->la_table();
  laddr_t la_mantissa_table = this->la_mantissa_table();
  laddr_t la_scale = this->la_scale();
  laddr_t la_bias = this->la_bias();
  laddr_t la_working = this->la_working();
  bool affine = this->affine();
  float eps = this->eps().convertToFloat();
  int layer_id = getOpLayerId(op);
  cvi_backend_tl_bf16_layernorm(*backend_ctx, layer_id, la_input, la_output,
                                la_table, la_mantissa_table, la_scale, la_bias,
                                la_working, affine, eps, n, c, h, w);
  return success();
}
}
