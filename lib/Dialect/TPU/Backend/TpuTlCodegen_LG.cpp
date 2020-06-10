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

static void parseTLLeakyReluParam(Operation *op,
    int8_t &pos_rshift, int8_t &pos_m_i8,
    int8_t &neg_rshift, int8_t &neg_m_i8,
    float & negative_slope) {
  auto lreluOp = llvm::dyn_cast<tpu::TL_LG_LeakyReluOp>(op);
  assert(lreluOp);

  if (lreluOp.m_i8_pos().hasValue()) {
    pos_m_i8 = lreluOp.m_i8_pos().getValue().getLimitedValue();
    pos_rshift = lreluOp.rshift_pos().getValue().getLimitedValue();
    assert(pos_m_i8);
  } else {
    pos_m_i8 = 0;
    pos_rshift = 0;
  }

  if (lreluOp.m_i8_neg().hasValue()) {
    neg_m_i8 = lreluOp.m_i8_neg().getValue().getLimitedValue();
    neg_rshift = lreluOp.rshift_neg().getValue().getLimitedValue();
    assert(neg_m_i8);
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
  auto lreluOp = llvm::dyn_cast<tpu::TL_LG_Conv2DOp>(op);
  assert(lreluOp);

  if (lreluOp.m_i8_pos().hasValue()) {
    pos_m_i8 = lreluOp.m_i8_pos().getValue().getLimitedValue();
    pos_rshift = lreluOp.rshift_pos().getValue().getLimitedValue();
    assert(pos_m_i8);
  } else {
    pos_m_i8 = 0;
    pos_rshift = 0;
  }

  if (lreluOp.m_i8_neg().hasValue()) {
    neg_m_i8 = lreluOp.m_i8_neg().getValue().getLimitedValue();
    neg_rshift = lreluOp.rshift_neg().getValue().getLimitedValue();
    assert(neg_m_i8);
  } else {
    neg_m_i8 = 0;
    neg_rshift = 0;
  }
  negative_slope = lreluOp.negative_slope().getValue().convertToFloat();
}

LogicalResult tpu::TL_LG_Conv2DOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, pt, pb, pl, pr, dh, dw;
  parseConvParam(param(), false, input(), output(), filter(),
                 n, ic, ih, iw, oc, oh, ow, g,
                 kh, kw, sh, sw, pt, pb, pl, pr, dh, dw, is_dw, with_bias, do_relu);

  laddr_t la_input = this->la_input().getLimitedValue();
  laddr_t la_output = this->la_output().getLimitedValue();
  laddr_t la_weight = this->la_filter().getLimitedValue();
  laddr_t la_perchanel = this->la_bias().getLimitedValue();
  laddr_t la_working = this->la_working().getLimitedValue();
  bool do_ic_alignment = this->do_ic_alignment().hasValue()
                            ? this->do_ic_alignment().getValue() : false;

  // pad is not "SAME", can not get from conv param
  int ph_t = this->pad_top_h().getLimitedValue();
  int ph_b = this->pad_bottom_h().getLimitedValue();
  int pw_l = this->pad_left_w().getLimitedValue();
  int pw_r = this->pad_right_w().getLimitedValue();
  int layer_id = mlir::getOpLayerId(op);

  int8_t pos_rshift = 0, pos_m_i8 = 0;
  int8_t neg_rshift = 0, neg_m_i8 = 0;
  float neg_slope = 0.0;

  if (this->fused_leaky())
    parseTLConvLeakyParam(op, pos_rshift, pos_m_i8,
                          neg_rshift, neg_m_i8, neg_slope);

  cvi_backend_tl_conv(
    *backend_ctx,
    layer_id,
    la_input, la_output, la_weight, la_working, la_perchanel,
    n, ic, ih, iw,
    g, oc, oh, ow, kh, kw, dh,
    dw, ph_t, ph_b, pw_l, pw_r, sh, sw,
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

LogicalResult tpu::TL_LG_DeConv2DOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, pt, pb, pl, pr, dh, dw;
  parseConvParam(param(), false, input(), output(), filter(),
                 n, ic, ih, iw, oc, oh, ow, g,
                 kh, kw, sh, sw, pt, pb, pl, pr, dh, dw, is_dw, with_bias, do_relu);

  laddr_t la_input = this->la_input().getLimitedValue();
  laddr_t la_output = this->la_output().getLimitedValue();
  laddr_t la_weight = this->la_filter().getLimitedValue();
  laddr_t la_perchannel = this->la_bias().getLimitedValue();
  bool do_ic_alignment = this->do_ic_alignment().hasValue()
                            ? this->do_ic_alignment().getValue() : false;

  // pad is not "SAME", can not get from conv param
  int ph_t = this->pad_top_h().getLimitedValue();
  int ph_b = this->pad_bottom_h().getLimitedValue();
  int pw_l = this->pad_left_w().getLimitedValue();
  int pw_r = this->pad_right_w().getLimitedValue();
  int ins_h = this->ins_h().getLimitedValue();
  int ins_last_h = this->ins_last_h().getLimitedValue();
  int ins_w = this->ins_w().getLimitedValue();
  int ins_last_w = this->ins_last_w().getLimitedValue();
  int layer_id = mlir::getOpLayerId(op);

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


LogicalResult tpu::TL_LG_EltwiseAddOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = mlir::getOpLayerId(op);
  TensorFile *wTF = getWeightTensorFile(op);

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
  for (unsigned i = 0; i < nInputs; ++i) {
      la_input[i] = la_input_array[i];
  }

  laddr_t la_output = this->la_output().getLimitedValue();
  laddr_t la_working = this->la_working().getLimitedValue();

  bool do_quant_rescale = false;
  int8_t rshift;
  int8_t m_i8_input[2];
  if (this->rshift().hasValue() && this->m_i8_inputs().hasValue()) {
    do_quant_rescale = true;
    rshift = this->rshift().getValue().getLimitedValue();

    std::vector<int32_t> m_i8_inputs_array;
    arrayAttrToVector(this->m_i8_inputs().getValue(), m_i8_inputs_array);
    assert(m_i8_inputs_array.size() == op->getNumOperands());
    assert(m_i8_inputs_array.size() >= 2);
    m_i8_input[0] = static_cast<int8_t>(m_i8_inputs_array[0]);
    m_i8_input[1] = static_cast<int8_t>(m_i8_inputs_array[1]);
  }

  bool do_early_stride = this->do_early_stride();
  int32_t early_stride_h = this->early_stride_h().getLimitedValue();
  int32_t early_stride_w = this->early_stride_w().getLimitedValue();
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

  return success();
}

LogicalResult tpu::TL_LG_EltwiseMulOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = mlir::getOpLayerId(op);
  TensorFile *wTF = getWeightTensorFile(op);

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
  for (unsigned i = 0; i < nInputs; ++i) {
      la_input[i] = la_input_array[i];
  }

  laddr_t la_output = this->la_output().getLimitedValue();
  laddr_t la_working = this->la_working().getLimitedValue();

  bool do_quant_rescale = false;
  int8_t rshift;
  int32_t m_i32;
  if (this->rshift().hasValue() && this->m_i32().hasValue()) {
    do_quant_rescale = true;
    rshift = this->rshift().getValue().getLimitedValue();
    m_i32 = this->m_i32().getValue().getLimitedValue();
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

  return success();
}

LogicalResult tpu::TL_LG_LrnOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = mlir::getOpLayerId(op);

  laddr_t la_input = this->la_input().getLimitedValue();
  laddr_t la_output = this->la_output().getLimitedValue();
  laddr_t la_working = this->la_working().getLimitedValue();
  laddr_t la_sqrt = this->la_sqrt().getLimitedValue();
  laddr_t la_power = this->la_power().getLimitedValue();
  int local_size = this->local_size().getLimitedValue();
  int8_t sum_rshift_i8 = this->sum_rshift().getLimitedValue();
  int8_t lrn_rshift_i8 = this->lrn_rshift().getLimitedValue();
  int8_t m_i8[2];
  m_i8[0] = this->quant_data0().getLimitedValue();
  m_i8[1] = this->quant_data1().getLimitedValue();

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
}

LogicalResult tpu::TL_LG_LutOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = mlir::getOpLayerId(op);

  laddr_t la_input = this->la_input().getLimitedValue();
  laddr_t la_output = this->la_output().getLimitedValue();
  laddr_t la_working = this->la_working().getLimitedValue();
  laddr_t la_y_table = this->la_y_table().getLimitedValue();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  cvi_backend_tl_lut( *backend_ctx,
                      layer_id,
                      la_input,
                      la_output,
                      la_working,
                      la_y_table,
                      n, c, h, w);
  return success();

}

LogicalResult tpu::TL_LG_ConcatOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  unsigned nInputs = op->getNumOperands();
  int layer_id = mlir::getOpLayerId(op);
  int axis = this->axis().getLimitedValue();

  std::vector<int32_t> la_input_array;
  laddr_t la_input[nInputs];
  arrayAttrToVector(this->la_input().getValue(), la_input_array);
  for (unsigned i = 0; i < nInputs; ++i) {
      la_input[i] = static_cast<laddr_t>(la_input_array[i]);
    }

  laddr_t la_output = this->la_output().getLimitedValue();
  laddr_t la_working = this->la_working().getLimitedValue();


  #define SHAPE_DIM 4
  int32_t input_dims[nInputs * SHAPE_DIM];
  for ( unsigned i = 0; i < nInputs; i++) {
    std::vector<int64_t> shape;
    int64_t size;
    getTensorShapeAndSize(op->getOperand(i), shape, size);
    // TODO: this looks very strange. 4 allocated for each input
    // TODO: but only 1 is set for each input
    input_dims[i] = shape[axis];
  }
  int output_dim[SHAPE_DIM];
  int output_dim_size;
  {
    std::vector<int64_t> shape;
    int64_t size;
    getTensorShapeAndSize(this->getResult(), shape, size);
    output_dim[0] = shape[0];
    output_dim[1] = shape[1];
    output_dim[2] = shape[2];
    output_dim[3] = shape[3];
    output_dim_size = shape.size();
  }

  // prepare quant info
  bool do_quant_rescale = false;
  int8_t r_i8;
  int8_t m_i8[nInputs];
  if (this->r_i8().hasValue() && this->m_i8().hasValue()) {
    do_quant_rescale = true;
    r_i8 = this->r_i8().getValue().getLimitedValue();

    std::vector<int32_t> m_i8_array;
    arrayAttrToVector(this->m_i8().getValue(), m_i8_array);
    assert(m_i8_array.size() == nInputs);
    for (unsigned i = 0; i < nInputs; ++i) {
      m_i8[i] = static_cast<int8_t>(m_i8_array[i]);
    }
  }

  cvi_backend_tl_concat( *backend_ctx,
                      layer_id,
                      input_dims,
                      nInputs,
                      output_dim,
                      la_input,
                      la_output,
                      la_working,
                      r_i8,
                      m_i8);
  return success();

}


LogicalResult tpu::TL_LG_LoadNeuronOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL Load Neuron codegen.\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = 0;

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
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


  gaddr_t src_gaddr = this->gaddr()->getLimitedValue();
  laddr_t dst_laddr = this->laddr()->getLimitedValue();
  bool transpose = this->transpose();
  bool aligned = this->align();
  bool isNeuron = true;

  cvi_backend_tl_load_stride( *backend_ctx,
                              layer_id,
                              src_gaddr,
                              dst_laddr,
                              local_n, local_c, local_h, local_w,
                              global_c, global_h, global_w,
                              transpose, aligned, isNeuron);
  return success();
}

LogicalResult tpu::TL_LG_LoadCoeffOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL Load Coeff codegen.\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = 0;

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  int64_t r_n, r_c, r_h, r_w;
  shape = getTensorShape(op->getResult(0));
  getNCHW(shape, r_n, r_c, r_h, r_w);
  getNCHW(shape, n, c, h, w);

  // load coeff as shape (1, oc, kh * kw, ic/g)
  if (this->tensor_type() == "CONV_COEFF") {
    n = 1;
    c = r_n;
    h = r_h * r_w;
    w = r_c;
    LLVM_DEBUG(llvm::errs()
      << "conv coeff load shape(nchw): ( " << n << " ,"
      << c << " ," << h << ", " << w << ")\n";);
  }

  int local_n = n;
  int local_c = c;
  int local_h = h;
  int local_w = w;

  int global_c = c;
  int global_h = h;
  int global_w = w;

  gaddr_t src_gaddr = this->gaddr()->getLimitedValue();
  laddr_t dst_laddr = this->laddr()->getLimitedValue();
  bool transpose = this->transpose();
  bool aligned = this->align();
  bool isNeuron = false;

  cvi_backend_tl_load_stride( *backend_ctx,
                              layer_id,
                              src_gaddr,
                              dst_laddr,
                              local_n, local_c, local_h, local_w,
                              global_c, global_h, global_w,
                              transpose, aligned, isNeuron);
  return success();
}

LogicalResult tpu::TL_LG_StoreOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL Store codegen.\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = 0;

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
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

  gaddr_t src_gaddr = this->gaddr()->getLimitedValue();
  laddr_t dst_laddr = this->laddr()->getLimitedValue();
  bool transpose = this->transpose();
  bool aligned = this->align();
  bool isNeuron = true;

  cvi_backend_tl_store_stride( *backend_ctx,
                                layer_id,
                                src_gaddr,
                                dst_laddr,
                                local_n, local_c, local_h, local_w,
                                global_c, global_h, global_w,
                                transpose, aligned, isNeuron);
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
  int64_t input_size, n, c, h, w;
  shape = getTensorShape(op->getOperand(0));
  getNCHW(shape, n, c, h, w);

  laddr_t la_src = this->la_src()->getLimitedValue();
  laddr_t la_dst = this->la_dst()->getLimitedValue();
  bool align = this->align();

  cvi_backend_tl_copy(*backend_ctx,
                      layer_id,
                      la_src,
                      la_dst,
                      n, c, h, w,
                      align
                      );

  return success();
}

LogicalResult tpu::TL_LG_INT8_PoolAvg2DOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL int8 pool avg codegen.\n";);

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = mlir::getOpLayerId(op);
  TensorFile *wTF = getWeightTensorFile(op);

  bool is_global, do_relu;
  int n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr;
  parsePoolParam(param(), input(), output(),
                 n, c, ih, iw, oh, ow,
                 kh, kw, sh, sw, pt, pb, pl, pr,
                 is_global, do_relu);

  laddr_t la_input = this->la_input().getLimitedValue();
  laddr_t la_output = this->la_output().getLimitedValue();

  assert(this->rshift().hasValue());
  int8_t rshift_i8 = this->rshift().getValue().getLimitedValue();
  assert(this->m_i8().hasValue());
  int8_t m_i8 = this->m_i8().getValue().getLimitedValue();

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

LogicalResult tpu::TL_LG_INT8_PoolMax2DOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL int8 pool max codegen.\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = mlir::getOpLayerId(op);
  TensorFile *wTF = getWeightTensorFile(op);

  bool is_global, do_relu;
  int n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr;
  parsePoolParam(param(), input(), output(),
                 n, c, ih, iw, oh, ow,
                 kh, kw, sh, sw, pt, pb, pl, pr,
                 is_global, do_relu);

  laddr_t la_input = this->la_input().getLimitedValue();
  laddr_t la_output = this->la_output().getLimitedValue();
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

LogicalResult tpu::TL_LG_BroadcastMulOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  bool do_relu = this->do_relu();

  laddr_t la_input = this->la_input().getLimitedValue();
  laddr_t la_output = this->la_output().getLimitedValue();
  laddr_t la_scale = this->la_scale().getLimitedValue();
  laddr_t la_bias = this->la_bias().getLimitedValue();
  int layer_id = mlir::getOpLayerId(op);

  cvi_backend_tl_broadcast_mul(
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

LogicalResult tpu::TL_LG_UpsampleOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  laddr_t la_input = this->la_input().getLimitedValue();
  laddr_t la_output = this->la_output().getLimitedValue();
  auto scale = this->scale().getLimitedValue();
  int layer_id = mlir::getOpLayerId(op);

  cvi_backend_tl_upsample(
      *backend_ctx,
      layer_id, //layer_id,
      la_input,
      la_output,
      n,
      c,
      h,
      w,
      scale,
      scale
  );
  return success();
}

LogicalResult tpu::TL_LG_LeakyReluOp::codegen(void *ctx) {
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

  laddr_t la_input = this->la_input().getLimitedValue();
  laddr_t la_output = this->la_output().getLimitedValue();
  int layer_id = mlir::getOpLayerId(op);

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

LogicalResult tpu::TL_LG_PReluOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  int8_t r_i8_pos, m_i8_pos, r_i8_neg;
  auto prelu_op = llvm::dyn_cast<tpu::TL_LG_PReluOp>(op);
  assert(prelu_op);

  if (prelu_op.m_i8_pos().hasValue()) {
    m_i8_pos = prelu_op.m_i8_pos().getValue().getLimitedValue();
    r_i8_pos = prelu_op.r_i8_pos().getValue().getLimitedValue();
    assert(m_i8_pos);
  } else {
    m_i8_pos = 0;
    r_i8_pos = 0;
  }

  if (prelu_op.r_i8_neg().hasValue()) {
    r_i8_neg = prelu_op.r_i8_neg().getValue().getLimitedValue();
    assert(r_i8_neg);
  } else {
    r_i8_neg = 0;
  }

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  laddr_t la_input = this->la_input().getLimitedValue();
  laddr_t la_output = this->la_output().getLimitedValue();
  laddr_t la_slope = this->la_slope().getLimitedValue();
  int layer_id = mlir::getOpLayerId(op);

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

}
