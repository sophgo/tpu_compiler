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

#define DEBUG_TYPE "mlir-to-cmdbuf"

using namespace mlir;

#include "backend/backend_tg_api.h"
#include "backend/backend_tl_api.h"

namespace mlir {


LogicalResult tpu::TL_LA_Conv2DOp::codegen(void *ctx) {
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

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  gaddr_t ga_filter = getWeightOpAddress(filter().getDefiningOp());
  gaddr_t ga_pc_info = getWeightOpAddress(pc_info().getDefiningOp());
  int layer_id = getOpLayerId(op);
  bool do_ic_alignment = (this->do_ic_alignment().hasValue()) ? this->do_ic_alignment().getValue() : false;

  LLVM_DEBUG(llvm::errs() << "    TL_LA_Conv2DOp, layer_id = " << layer_id << "\n";);
  cvi_backend_tl_conv_LA(*backend_ctx, layer_id,
      ga_input, ga_output, ga_filter, ga_pc_info,
      n, ic, ih, iw, g, oc, oh, ow, kh, kw,
      dh, dw, pt, pb, pl, pr, sh, sw, ins_h, ins_w,
      false, with_bias, do_relu, do_ic_alignment);
  return success();
}

LogicalResult tpu::TL_LW_Conv2DOp::codegen(void *ctx) {
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

  gaddr_t ga_input = tl_load_flag() ? getPreviousOpAddress(op) : GA_INVALID;
  gaddr_t ga_output = tl_store_flag() ? getOpAddress(op) : GA_INVALID;
  gaddr_t ga_filter = getWeightOpAddress(filter().getDefiningOp());
  gaddr_t ga_pc_info = getWeightOpAddress(pc_info().getDefiningOp());
  laddr_t la_input = this->la_input();
  laddr_t la_output = this->la_output();
  laddr_t la_working = this->la_working();

  // leaky_relu
  bool do_leaky_relu = this->do_leaky_relu();
  bool do_ic_alignment = (this->do_ic_alignment().hasValue()) ? this->do_ic_alignment().getValue() : false;
  int8_t pos_rshift = 0, pos_m_i8 = 0, neg_rshift = 0, neg_m_i8 = 0;
  if (do_leaky_relu) {
    if (this->m_i8_pos().hasValue()) {
      pos_m_i8 = this->m_i8_pos().getValue();
      pos_rshift = this->rshift_pos().getValue();
    } else {
      pos_m_i8 = 0;
      pos_rshift = 0;
    }
    if (this->m_i8_neg().hasValue()) {
      neg_m_i8 = this->m_i8_neg().getValue();
      neg_rshift = this->rshift_neg().getValue();
    } else {
      neg_m_i8 = 0;
      neg_rshift = 0;
    }
  }

  int layer_id = getOpLayerId(op);

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
    bool do_decompress = load_compr_act().hasValue() ?
                         load_compr_act().getValue() : false;

    if (!do_decompress) {
      cvi_backend_tl_load(*backend_ctx,
                           layer_id,
                           la_input,
                           ga_input,
                           CVK_FMT_I8,
                           n, ic, ih, iw);
    } else {
      int step_size = 0;
      int c_step = ic;
      int h_step = ih;
      if (this->load_compr_act_param().hasValue()) {
        c_step = this->load_compr_act_param().getValue().c_step().getInt();
        h_step = this->load_compr_act_param().getValue().h_step().getInt();
        step_size = this->load_compr_act_param().getValue().step_size().getInt();
      }

      cvi_backend_tl_load_compressed(*backend_ctx,
                                     layer_id,
                                     ga_input,
                                     la_input,
                                     n, ic, ih, iw,
                                     ic, ih, iw,
                                     /*transpose=*/false,
                                     /*aligned=*/true,
                                     /*isNeuron=*/true,
                                     /*from=*/CVK_FMT_I8,
                                     /*to=*/CVK_FMT_I8,
                                     h_step, step_size, c_step
                                     );

    }
  }

  bool compressed_weight = false;
  auto convOp = dyn_cast<tpu::TL_LW_Conv2DOp>(op);
  if (convOp.compressed_weight().hasValue())
    compressed_weight = convOp.compressed_weight().getValue();

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
        dh, dw, pt, pb, pl, pr, sh, sw, ins_h, ins_w,
        false, with_bias, do_relu,
        true, ga_output,
        do_leaky_relu, pos_rshift, pos_m_i8, neg_rshift, neg_m_i8,
        do_ic_alignment, compressed_weight);
  } else {
    cvi_backend_tl_conv_LW(*backend_ctx, layer_id,
        la_input, la_output, la_working,
        ga_filter, ga_pc_info,
        n, ic, ih, iw, g, oc, oh, ow, kh, kw,
        dh, dw, pt, pb, pl, pr, sh, sw, ins_h, ins_w,
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
  int32_t early_stride_h = this->early_stride_h();
  int32_t early_stride_w = this->early_stride_w();
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
  gaddr_t ga_input = GA_INVALID;
  if (tl_load_flag()) {
    auto weightOp = op->getOperand(0).getDefiningOp();
    if (isa<tpu::LoadWeightOp>(weightOp)) {
      // load from weight
      ga_input = getWeightOpAddress(weightOp);
    }
    else {
      ga_input = getPreviousOpAddress(op, 0);
    }
  }
  gaddr_t ga_addend = getPreviousOpAddress(op, 1);
  gaddr_t ga_output = tl_store_flag() ? getOpAddress(op) : GA_INVALID;
  int layer_id = getOpLayerId(op);

  laddr_t la_input = LA_INVALID;
  laddr_t la_output = LA_INVALID;
  laddr_t la_working = LA_INVALID;
  if (this->lm_layout() != "NONE") {
    la_input = this->la_input();
    la_output = this->la_output();
    la_working = this->la_working();
  }

  int8_t rshift = this->rshift();
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
    rshift, m_i8_input[0], m_i8_input[1], 0,
    tl_load_flag(), tl_store_flag());

  return success();
}

LogicalResult tpu::TL_EltwiseMulOp::codegen(void *ctx) {
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
  assert(op->getNumOperands() == 2 && "support 2 inputs only");

  gaddr_t ga_input = tl_load_flag() ? getPreviousOpAddress(op, 0) : GA_INVALID; //Closest op
  auto opd2 = op->getOperand(1).getDefiningOp();
  gaddr_t ga_input2 = opd2->getAttr("gaddr") ?
                      opd2->getAttr("gaddr").cast<IntegerAttr>().getInt() :
                      (opd2->getAttr("offset") ? opd2->getAttr("offset").cast<IntegerAttr>().getInt() :
                                                  GA_INVALID);
  bool isAllInLocalMem = (ga_input2 == GA_INVALID) && (tl_load_flag() == false);
  //Fix me: now use global address to present it's unique ID.
  gaddr_t ga_output = tl_store_flag() ? getOpAddress(op) : GA_INVALID;

  laddr_t la_input = this->la_input();
  laddr_t la_output = this->la_output();
  laddr_t la_working = this->la_working();

  int8_t rshift = this->rshift();

  // op code PROD = 0; SUM = 1; MAX = 2;
  int op_code = 0;
  const int coeffs[2] = {1, 1};
  const int i32Multiplier = (this->m_i32_output().hasValue()) ? this->m_i32_output().getValue() : 0;

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
        do_relu, 0, coeffs, i32Multiplier,
        0, 1, 1);
    if(tl_store_flag()) {
      cvi_backend_tl_store(*backend_ctx, layer_id, la_output, ga_output,
                           CVK_FMT_I8, n, c, h, w);
    }
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
  int64_t output_size, on, oc, oh, ow;
  getTensorShapeAndSize(op->getResult(0), output_shape, output_size);
  getNCHW(output_shape, on, oc, oh, ow);

  gaddr_t ga_input = tl_load_flag() ? getPreviousOpAddress(op) : GA_INVALID;
  gaddr_t ga_output = tl_store_flag() ? getOpAddress(op) : GA_INVALID;
  gaddr_t y0_table_gaddr = getWeightOpAddress(table().getDefiningOp());
  int layer_id = getOpLayerId(op);

  laddr_t la_input = LA_INVALID;
  laddr_t la_output = LA_INVALID;
  laddr_t la_working = LA_INVALID;
  if (this->lm_layout() != "NONE") {
    la_input = this->la_input();
    la_output = this->la_output();
    la_working = this->la_working();
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

  cvi_backend_tl_lut_LA(
    *backend_ctx, layer_id,
    la_input, la_output, la_working,
    ga_input, ga_output, y0_table_gaddr,
    n, c, h, w,
    tl_load_flag(), tl_store_flag());
  return success();
}

LogicalResult tpu::TL_PoolAvg2DOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  bool is_global, do_relu, count_include_pad;
  int n, c, ih, iw, oh, ow, kw, sw, pb, pl, pr, pad_value;
  int kh, sh, ph;
  parsePoolParam(param(), input(), output(),
                n, c, ih, iw, oh, ow,
                kh, kw, sh, sw, ph, pb, pl, pr, pad_value,
                is_global, do_relu, count_include_pad);
  int8_t rshift = (this->rshift().hasValue()) ? this->rshift().getValue() : 0;
  int8_t m_i8 = (this->m_i8().hasValue()) ? this->m_i8().getValue() : 0;

  gaddr_t ga_input = tl_load_flag() ? getPreviousOpAddress(op) : GA_INVALID;
  gaddr_t ga_output = tl_store_flag() ? getOpAddress(op) : GA_INVALID;
  int layer_id = getOpLayerId(op);

  laddr_t la_input = LA_INVALID;
  laddr_t la_output = LA_INVALID;
  if (this->lm_layout() != "NONE") {
    la_input = this->la_input();
    la_output = this->la_output();
  }

  LLVM_DEBUG(
    llvm::errs() << "    TL_PoolAvg2DOp, layer_id = " << layer_id;
    llvm::errs() << ", " << this->lm_layout();
    if (tl_load_flag())
      llvm::errs() << ", LD";
    if (tl_store_flag())
      llvm::errs() << ", ST";
    if (!tl_load_flag() && !tl_store_flag())
      llvm::errs() << ", FUSED";
    llvm::errs() << "\n";
  );
  if(tl_load_flag()) {
    cvi_backend_tl_load(*backend_ctx, layer_id, la_input, ga_input, CVK_FMT_I8,
                        n, c, ih, iw);
  }
  cvi_backend_tl_pooling(
    *backend_ctx, layer_id,
    la_input, la_output,
    n, c, ih, iw,
    n, c, oh, ow,
    kh, kw, sh, sw,
    ph, pb, pl, pr,
    true, //avg_pooling
    rshift, m_i8);
  if(tl_store_flag()) {
    cvi_backend_tl_store(*backend_ctx, layer_id, la_output, ga_output,
                         CVK_FMT_I8, n, c, oh, ow);
  }
  return success();
}

LogicalResult tpu::TL_ScaleOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  bool do_relu = this->param().do_relu().getValue();

  gaddr_t ga_input = tl_load_flag() ? getPreviousOpAddress(op) : GA_INVALID;
  gaddr_t ga_output = tl_store_flag() ? getOpAddress(op) : GA_INVALID;
  gaddr_t ga_scale = getOpAddress(filter().getDefiningOp());
  gaddr_t ga_pc_info = getWeightOpAddress(pc_info().getDefiningOp());
  int layer_id = getOpLayerId(op);

  laddr_t la_input = LA_INVALID;
  laddr_t la_output = LA_INVALID;
  laddr_t la_working = LA_INVALID;
  if (this->lm_layout() != "NONE") {
    la_input = this->la_input();
    la_output = this->la_output();
    la_working = this->la_working();
  }

  LLVM_DEBUG(
    llvm::errs() << "    TL_ScaleOp, layer_id = " << layer_id;
    llvm::errs() << ", " << this->lm_layout();
    if (tl_load_flag())
      llvm::errs() << ", LD";
    if (tl_store_flag())
      llvm::errs() << ", ST";
    if (!tl_load_flag() && !tl_store_flag())
      llvm::errs() << ", FUSED";
    llvm::errs() << "\n";
  );

  if(tl_load_flag()) {
    cvi_backend_tl_load(*backend_ctx, layer_id, la_input, ga_input, CVK_FMT_I8,
                        n, c, h, w);
  }
  cvi_backend_tl_scale_qi32(
      *backend_ctx, // ctx
      layer_id,     // layer_id
      la_input, la_output, la_working,
      ga_scale, // scale_addr
      ga_pc_info,   // pack_addr
      n, c, h, w,
      n * c,        // scale_dim (axis = 1  =>  n * c)
      h * w,        // inner_dim (axis = 1  =>  h * w)
      false,        // is_scale_const
      0,            // const_scale
      do_relu,      // do_activation,
      0,            // activation_method
      nullptr,      // activation_arg
      false,        // with_bias
      false         // second_is_load_weight
      );
  if(tl_store_flag()) {
    cvi_backend_tl_store(*backend_ctx, layer_id, la_output, ga_output,
                         CVK_FMT_I8, n, c, h, w);
  }
  return success();
}

LogicalResult tpu::TL_PixelShuffleOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, in, ic, ih, iw;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, in, ic, ih, iw);
  uint32_t factor = this->factor();
  uint32_t oc = ic / (factor * factor);
  uint32_t oh = ih * factor;
  uint32_t ow = iw * factor;

  gaddr_t ga_input = tl_load_flag() ? getPreviousOpAddress(op) : GA_INVALID;
  gaddr_t ga_output = tl_store_flag() ? getOpAddress(op) : GA_INVALID;
  uint32_t layer_id = getOpLayerId(op);

  laddr_t la_input = LA_INVALID;
  laddr_t la_output = LA_INVALID;
  if (this->lm_layout() != "NONE") {
    la_input = this->la_input();
    la_output = this->la_output();
  }
  LLVM_DEBUG(
    llvm::errs() << "    TL_PixelShuffleOp, layer_id = " << layer_id;
    llvm::errs() << ", " << this->lm_layout();
    if (tl_load_flag())
      llvm::errs() << ", LD";
    if (tl_store_flag())
      llvm::errs() << ", ST";
    if (!tl_load_flag() && !tl_store_flag())
      llvm::errs() << ", FUSED";
    llvm::errs() << "\n";
  );

  cvi_backend_tl_pixel_shuffle_LA(*backend_ctx, layer_id, la_input, la_output,
                                  ga_input, (uint32_t)in, (uint32_t)ic,
                                  (uint32_t)ih, (uint32_t)iw, factor);

  if(tl_store_flag()) {
    cvi_backend_tl_store(*backend_ctx, layer_id, la_output, ga_output,
                         CVK_FMT_I8, in, oc, oh, ow);
  }
  return success();
}

LogicalResult tpu::TL_PReluOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TL_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  laddr_t la_input = LA_INVALID;
  laddr_t la_output = LA_INVALID;
  laddr_t la_working = LA_INVALID;
  gaddr_t ga_filter = getWeightOpAddress(filter().getDefiningOp());
  gaddr_t ga_input = tl_load_flag() ? getPreviousOpAddress(op) : GA_INVALID;
  gaddr_t ga_output = tl_store_flag() ? getOpAddress(op) : GA_INVALID;

  if (this->lm_layout() != "NONE") {
    la_input = this->la_input();
    la_output = this->la_output();
    la_working = this->la_working();
  }

  int layer_id = getOpLayerId(op);
  assert(this->rshift_pos().hasValue());
  int8_t rshift_pos = this->rshift_pos().getValue();
  assert(this->m_i8_pos().hasValue());
  int8_t m_i8_pos = this->m_i8_pos().getValue();
  assert(this->rshift_neg().hasValue());
  int8_t rshift_neg = this->rshift_neg().getValue();

  cvi_backend_tl_load(*backend_ctx, layer_id, la_working, ga_filter, CVK_FMT_I8,
                      1, c, 1, 1);
  if(tl_load_flag()) {
    cvi_backend_tl_load(*backend_ctx, layer_id, la_input, ga_input, CVK_FMT_I8,
                        n, c, h, w);
  }

  cvi_backend_tl_prelu(
      *backend_ctx,
      layer_id, //layer_id,
      la_input,
      la_output,
      la_working, // filter
      n,
      c,
      h,
      w,
      rshift_pos, m_i8_pos, rshift_neg);

  if(tl_store_flag()) {
    cvi_backend_tl_store(*backend_ctx, layer_id, la_output, ga_output,
                         CVK_FMT_I8, n, c, h, w);
  }

  return success();
}

}
