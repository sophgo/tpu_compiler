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


LogicalResult tpu::TL_LA_Conv2DOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, ph, pw, dh, dw;
  parseConvParam(param(), false, input(), output(), filter(),
                 n, ic, ih, iw, oc, oh, ow, g,
                 kh, kw, sh, sw, ph, pw, dh, dw, is_dw, with_bias, do_relu);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  gaddr_t ga_filter = getWeightOpAddress(filter()->getDefiningOp());
  gaddr_t ga_pc_info = getWeightOpAddress(pc_info()->getDefiningOp());
  int layer_id = mlir::getOpLayerId(op);
  bool do_ic_alignment = (this->do_ic_alignment().hasValue()) ? this->do_ic_alignment().getValue() : false;

  LLVM_DEBUG(llvm::errs() << "    TL_LA_Conv2DOp, layer_id = " << layer_id << "\n";);
  cvi_backend_tl_conv_LA(*backend_ctx, layer_id,
      ga_input, ga_output, ga_filter, ga_pc_info,
      n, ic, ih, iw, g, oc, oh, ow, kh, kw,
      dh, dw, ph, ph, pw, pw, sh, sw,
      false, with_bias, do_relu, do_ic_alignment);
  return success();
}

LogicalResult tpu::TL_LW_Conv2DOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, ph, pw, dh, dw;
  parseConvParam(param(), false, input(), output(), filter(),
                 n, ic, ih, iw, oc, oh, ow, g,
                 kh, kw, sh, sw, ph, pw, dh, dw, is_dw, with_bias, do_relu);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  gaddr_t ga_filter = getWeightOpAddress(filter()->getDefiningOp());
  gaddr_t ga_pc_info = getWeightOpAddress(pc_info()->getDefiningOp());
  laddr_t la_input = this->la_input().getLimitedValue();
  laddr_t la_output = this->la_output().getLimitedValue();
  laddr_t la_working = this->la_working().getLimitedValue();

  // leaky_relu
  bool do_leaky_relu = this->do_leaky_relu();
  bool do_ic_alignment = (this->do_ic_alignment().hasValue()) ? this->do_ic_alignment().getValue() : false;
  int8_t pos_rshift = 0, pos_m_i8 = 0, neg_rshift = 0, neg_m_i8 = 0;
  if (do_leaky_relu) {
    if (this->m_i8_pos().hasValue()) {
      pos_m_i8 = this->m_i8_pos().getValue().getLimitedValue();
      pos_rshift = this->rshift_pos().getValue().getLimitedValue();
    } else {
      pos_m_i8 = 0;
      pos_rshift = 0;
    }
    if (this->m_i8_neg().hasValue()) {
      neg_m_i8 = this->m_i8_neg().getValue().getLimitedValue();
      neg_rshift = this->rshift_neg().getValue().getLimitedValue();
    } else {
      neg_m_i8 = 0;
      neg_rshift = 0;
    }
  }

  int layer_id = mlir::getOpLayerId(op);

  LLVM_DEBUG(
    llvm::errs() << "    TL_LW_Conv2DOp,  layer_id = " << layer_id;
    llvm::errs() << ", " << this->lm_layout();
    if (tl_load_flag())
      llvm::errs() << ", LD";
    if (tl_store_flag())
      llvm::errs() << ", ST";
    if (!tl_load_flag() && !tl_store_flag())
      llvm::errs() << ", FUSED";
    llvm::errs() << "\n";
  );

  if (tl_load_flag()) {
    cvi_backend_tl_load(*backend_ctx, layer_id,
        la_input, ga_input, n, ic, ih, iw);
  }

  bool compressed_weight = false;
  auto convOp = dyn_cast<tpu::TL_LW_Conv2DOp>(op);
  if (convOp.compressed_weight().hasValue())
    compressed_weight = convOp.compressed_weight().getValue();

  #if 0
  //
  // V0: Weight Only version, with no parallel for load/store activations
  // (only consider load weight parallel)
  //
  cvi_backend_tl_conv_LW(*backend_ctx, layer_id,
      la_input, la_output, la_working,
      ga_filter, ga_pc_info,
      n, ic, ih, iw, g, oc, oh, ow, kh, kw,
      dh, dw, ph, ph, pw, pw, sh, sw,
      false, with_bias, do_relu);
  if (op.tl_store_flag()) {
    cvi_backend_tl_store(*backend_ctx, layer_id,
        la_output, ga_output, n, oc, oh, ow);
  }
  #endif
  #if 1
  //
  // V1: Weight and Store version
  //   this only do parallel on both Weight load and Activation Store
  //   but load activation is not handled in parallel
  //
  if (tl_store_flag()) {
    cvi_backend_tl_conv_LW(*backend_ctx, layer_id,
        la_input, la_output, la_working,
        ga_filter, ga_pc_info,
        n, ic, ih, iw, g, oc, oh, ow, kh, kw,
        dh, dw, ph, ph, pw, pw, sh, sw,
        false, with_bias, do_relu,
        true, ga_output,
        do_leaky_relu, pos_rshift, pos_m_i8, neg_rshift, neg_m_i8,
        do_ic_alignment, compressed_weight);
  } else {
    cvi_backend_tl_conv_LW(*backend_ctx, layer_id,
        la_input, la_output, la_working,
        ga_filter, ga_pc_info,
        n, ic, ih, iw, g, oc, oh, ow, kh, kw,
        dh, dw, ph, ph, pw, pw, sh, sw,
        false, with_bias, do_relu,
        false, GA_INVALID,
        do_leaky_relu, pos_rshift, pos_m_i8, neg_rshift, neg_m_i8,
        do_ic_alignment, compressed_weight);
  }
  #endif
  return success();
}

LogicalResult tpu::TL_EltwiseAddOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  std::vector<int64_t> output_shape;
  int64_t output_size, on, oc, oh, ow;
  getTensorShapeAndSize(op->getResult(0), output_shape, output_size);
  getNCHW(output_shape, on, oc, oh, ow);
  bool do_relu = this->do_relu();
  bool do_early_stride = this->do_early_stride();
  int32_t early_stride_h = this->early_stride_h().getLimitedValue();
  int32_t early_stride_w = this->early_stride_w().getLimitedValue();
  if (do_early_stride) {
    assert(oh == h / early_stride_h);
    assert(ow == w / early_stride_w);
  }

  assert(op->getNumOperands() == 2 && "support 2 inputs only");

  // Always make sure the long path to be input, the short path as augend
  // TODO: do real search
  // workaround as followings
  // 1. for resnet50
  //    - if opd0 is not Conv, it MUST be short path
  //    - if opd0 is conv, check in_short_path() flag
  // 2. for mobilenet_v2 (always both conv, and the is_short_path() is not valid)
  //    - if opd0 has more than one use, it is the short path
  int augend_idx = 0;
  auto prev_op = op->getOperand(0)->getDefiningOp();
  auto prev_conv_op = llvm::dyn_cast<tpu::TL_LW_Conv2DOp>(prev_op);
  if (!prev_conv_op) {
    augend_idx = 1;
  } else {
    if (!op->getOperand(0)->hasOneUse()) {
      augend_idx = 1;
    } else if (prev_conv_op.in_short_path().getValue()) {
      augend_idx = 1;
    }
  }

  gaddr_t ga_input = getPreviousOpAddress(op, augend_idx);
  gaddr_t ga_addend = getPreviousOpAddress(op, 1 - augend_idx);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = mlir::getOpLayerId(op);

  laddr_t la_input = LA_INVALID;
  laddr_t la_output = LA_INVALID;
  laddr_t la_working = LA_INVALID;
  if (this->lm_layout() != "NONE") {
    la_input = this->la_input().getLimitedValue();
    la_output = this->la_output().getLimitedValue();
    la_working = this->la_working().getLimitedValue();
  }

  int8_t rshift = this->rshift().getLimitedValue();
  int8_t m_i8_input[2];
  if(this->m_i8_inputs().hasValue()){
  std::vector<int32_t> m_i8_inputs_array;
    arrayAttrToVector(this->m_i8_inputs().getValue(), m_i8_inputs_array);
    assert(m_i8_inputs_array.size() == 2);
    m_i8_input[0] = static_cast<int8_t>(m_i8_inputs_array[0]);
    m_i8_input[1] = static_cast<int8_t>(m_i8_inputs_array[1]);
  }

  LLVM_DEBUG(
    llvm::errs() << "    TL_EltwiseAddOp, layer_id = " << layer_id;
    llvm::errs() << ", " << this->lm_layout();
    if (tl_load_flag())
      llvm::errs() << ", LD";
    if (tl_store_flag())
      llvm::errs() << ", ST";
    if (!tl_load_flag() && !tl_store_flag())
      llvm::errs() << ", FUSED";
    llvm::errs() << "\n";
  );

  // op code PROD = 0; SUM = 1; MAX = 2;
  int op_code = 1;
  cvi_backend_tl_eltwise_op(
    *backend_ctx, layer_id,
    la_input, la_output, la_working,
    ga_input, ga_output, ga_addend,
    op_code, n, c, h, w, do_relu,
    do_early_stride, early_stride_h, early_stride_w,
    rshift, m_i8_input[augend_idx], m_i8_input[1-augend_idx], 0,
    tl_load_flag(), tl_store_flag());

  return success();
}

LogicalResult tpu::TL_EltwiseMulOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = mlir::getOpLayerId(op);

  int augend_idx = 0;
  auto prev_op = op->getOperand(0)->getDefiningOp();
  auto prev_conv_op = llvm::dyn_cast<tpu::TL_LW_Conv2DOp>(prev_op);
  if (!prev_conv_op) {
    augend_idx = 1;
  } else {
    if (!op->getOperand(0)->hasOneUse()) {
      augend_idx = 1;
    } else if (prev_conv_op.in_short_path().getValue()) {
      augend_idx = 1;
    }
  }

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  bool do_relu = this->do_relu();
  int nInputs = op->getNumOperands();
  assert(op->getNumOperands() == 2 && "support 2 inputs only");

  gaddr_t ga_input = getPreviousOpAddress(op, augend_idx); //Closest op
  gaddr_t ga_input2 = getPreviousOpAddress(op, 1 - augend_idx);
  gaddr_t ga_input2_pre_input =  getPreviousOpAddress(op->getOperand(augend_idx)->getDefiningOp());
  bool isAllInLocalMem = (ga_input2_pre_input == ga_input2) && (tl_load_flag() == false) && (tl_store_flag() == false);
  //Fix me: now use global address to present it's unique ID. 
  gaddr_t ga_output = getOpAddress(op);

  laddr_t la_input = this->la_input().getLimitedValue();
  laddr_t la_output = this->la_output().getLimitedValue();
  laddr_t la_working = this->la_working().getLimitedValue();

  bool do_quant_rescale = false;
  int8_t rshift = this->rshift().getLimitedValue();

  // op code PROD = 0; SUM = 1; MAX = 2;
  int op_code = 0;
  const int coeffs[2] = {1, 1};
  const int i32Multiplier = (this->m_i32_output().hasValue()) ? this->m_i32_output().getValue().getLimitedValue() : 0;

  if(!isAllInLocalMem) {
    cvi_backend_tl_eltwise_op(
      *backend_ctx, layer_id,
      la_input, la_output, la_working,
      ga_input, ga_output, ga_input2,
      op_code, n, c, h, w, do_relu,
      false, 1, 1,
      rshift, 0, 0, i32Multiplier,
      tl_load_flag(), tl_store_flag());
  } else {
    LLVM_DEBUG(llvm::errs() << "TL_codegen: " << "cvi_backend_tl_eltwise"
               << " [" << getOpName() << "]\n";);
    laddr_t la_input_tmp[2];
    la_input_tmp[0] = la_input;
    la_input_tmp[1] = la_output;
    cvi_backend_tl_eltwise(
        *backend_ctx, layer_id,
        la_input_tmp, la_output, la_working,
        n, c, h, w, 2,
        op_code,
        rshift, nullptr,
        true,
        do_relu, 0, coeffs, i32Multiplier);
  }
  return success();
}

LogicalResult tpu::TL_LutOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
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

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  gaddr_t y0_table_gaddr = getWeightOpAddress(table()->getDefiningOp());
  int layer_id = mlir::getOpLayerId(op);

  laddr_t la_input = LA_INVALID;
  laddr_t la_output = LA_INVALID;
  laddr_t la_working = LA_INVALID;
  if (this->lm_layout() != "NONE") {
    la_input = this->la_input().getLimitedValue();
    la_output = this->la_output().getLimitedValue();
    la_working = this->la_working().getLimitedValue();
  }

  LLVM_DEBUG(
    llvm::errs() << "    TL_LutOp, layer_id = " << layer_id;
    llvm::errs() << ", " << this->lm_layout();
    if (tl_load_flag())
      llvm::errs() << ", LD";
    if (tl_store_flag())
      llvm::errs() << ", ST";
    if (!tl_load_flag() && !tl_store_flag())
      llvm::errs() << ", FUSED";
    llvm::errs() << "\n";
  );

  cvi_backend_tl_lut(
    *backend_ctx, layer_id,
    la_input, la_output, la_working,
    ga_input, ga_output, y0_table_gaddr,
    n, c, h, w,
    tl_load_flag(), tl_store_flag());
  return success();
}

// MemRefType dummy
LogicalResult tpu::TL_MemRef_LutOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TL_MemRef_EltwiseAddOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TL_MemRef_EltwiseMulOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TL_MemRef_LA_Conv2DOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TL_MemRef_LW_Conv2DOp::codegen(void *ctx) {
  return success();
}

}
