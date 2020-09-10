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
#include "mlir/Dialect/TPU/CustomOpPlugin.h"
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
#include "cvikernel/cvikernel.h"
#include <fstream>

#define DEBUG_TYPE "tg_codegen"

using namespace mlir;

extern int BF16_TABLE_START;
extern int BF16_TABLE_END;

#include "backend/backend_tg_api.h"
#include "backend/backend_tl_api.h"

namespace mlir {

LogicalResult tpu::TG_INT8_BroadcastMulOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  bool do_relu = this->param().do_relu().getValue();;

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  gaddr_t ga_scale = getOpAddress(filter()->getDefiningOp());
  gaddr_t ga_pc_info = getWeightOpAddress(pc_info()->getDefiningOp());
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_fixed_scale_qi32_kernel(
      *backend_ctx, // ctx
      0,            // stream_id
      0,            // inst_id
      layer_id,     // layer_id
      nullptr,      // depends
      0,            // depends_len
      ga_input,     // input_addr
      ga_scale, // scale_addr
      ga_pc_info,   // pack_addr
      ga_output,    // output_addr
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

  return success();
}

LogicalResult tpu::TG_BF16_BroadcastMulOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  bool do_relu = this->param().do_relu().getValue();;

  int64_t input_size_1;
  std::vector<int64_t> shape_1;
  getTensorShapeAndSize(op->getOperand(1), shape_1, input_size_1);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  gaddr_t ga_scale = getOpAddress(filter()->getDefiningOp());
  // FIXME: support bias
  //gaddr_t ga_pc_info = getWeightOpAddress(pc_info()->getDefiningOp());
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_bf16_scale_kernel(
      *backend_ctx, // ctx
      0,            // stream_id
      0,            // inst_id
      layer_id,     // layer_id
      nullptr,      // depends
      0,            // depends_len
      ga_input,     // input_addr
      ga_scale, // scale_addr
      GA_INVALID,   // pack_addr
      ga_output,    // output_addr
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

  return success();
}

LogicalResult tpu::TG_CastOp::codegen(void *ctx) {
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
  int layer_id = getOpLayerId(op);

  if (from() == "FP32" && to() == "BF16") {
    convert_fp32_bf16_kernel(*backend_ctx, 0, 0, layer_id, nullptr, 0,
                                       input_gaddr, output_gaddr, n, c,
                                       h, w);
  } else if (from() == "BF16" && to() == "FP32") {
    convert_bf16_fp32_kernel(*backend_ctx, 0, 0, layer_id, nullptr, 0,
                                       input_gaddr, output_gaddr, n, c,
                                       h, w);
  } else {
    llvm_unreachable("unsupport other type cast");
  }

  return success();
}

LogicalResult tpu::TG_INT8_ConcatOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  unsigned nInputs = op->getNumOperands();
  auto ga_inputs = new gaddr_t[nInputs];
  for ( unsigned i = 0; i < nInputs; i++) {
    ga_inputs[i] = getPreviousOpAddress(op, i);
  }
  gaddr_t ga_output = getOpAddress(op);
  int axis = this->axis().getLimitedValue();
  int layer_id = getOpLayerId(op);

  // prepare shape info
  #define SHAPE_DIM 4
  auto input_dims = new int32_t[nInputs * SHAPE_DIM];
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
  std::vector<int64_t> shape;
  int64_t size;
  getTensorShapeAndSize(this->getResult(), shape, size);
  output_dim[0] = shape[0];
  output_dim[1] = shape[1];
  output_dim[2] = shape[2];
  output_dim[3] = shape[3];
  output_dim_size = shape.size();

  // prepare quant info
  bool do_quant_rescale = false;
  int8_t rshift;
  auto m_i8_input = new int32_t[nInputs];
  if (this->rshift().hasValue() && this->m_i8_inputs().hasValue()) {
    do_quant_rescale = true;
    rshift = this->rshift().getValue().getLimitedValue();

    std::vector<int32_t> m_i8_inputs_array;
    arrayAttrToVector(this->m_i8_inputs().getValue(), m_i8_inputs_array);
    assert(m_i8_inputs_array.size() == nInputs);
    for (unsigned i = 0; i < nInputs; ++i) {
      m_i8_input[i] = static_cast<int8_t>(m_i8_inputs_array[i]);
    }
  }

  // TODO: should change on backend API, rather than doing cast
  auto rshift_int = new int32_t[nInputs];
  auto m_int = new int32_t[nInputs];
  if (do_quant_rescale) {
    for (unsigned i = 0; i < nInputs; ++i) {
      rshift_int[i] = static_cast<int>(rshift);
      m_int[i] = static_cast<int>(m_i8_input[i]);
    }
  }

  cvi_backend_tg_fixed_concat_kernel(
      *backend_ctx,
      0, // u32 stream_id,
      0, //u32 inst_id,
      layer_id, // u32 layer_id,
      nullptr, // const u32 *depends,
      0,// u32 depends_len,
      ga_inputs,       // gaddr_t input_gaddrs[],
      ga_output,       // gaddr_t output_gaddr,
      input_dims,      // int input_dims[],
      nInputs,         //int input_num,
      axis,            // int concat_axis,
      output_dim_size, // int output_dim_size,
      output_dim,      // int *output_dim,
      do_relu(),
      do_quant_rescale ? nInputs : 0,     // const int need_quantize_num,
      do_quant_rescale ? rshift_int : 0,  // const int *right_shift_width,
      do_quant_rescale ? m_int : nullptr  // const int *threshold_x_quantized
      );

  delete[] ga_inputs;
  delete[] m_i8_input;
  delete[] rshift_int;
  delete[] m_int;
  delete[] input_dims;
  return success();
}

LogicalResult tpu::TG_BF16_ConcatOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  int nInputs = op->getNumOperands();
  auto ga_inputs = new gaddr_t[nInputs];
  for ( int i = 0; i < nInputs; i++) {
    ga_inputs[i] = getPreviousOpAddress(op, i);
  }
  gaddr_t ga_output = getOpAddress(op);
  int axis = this->axis().getLimitedValue();
  int layer_id = getOpLayerId(op);

  // prepare shape info
  #define SHAPE_DIM 4
  auto input_dims = new int32_t[nInputs * SHAPE_DIM];
  for ( int i = 0; i < nInputs; i++) {
    std::vector<int64_t> shape;
    int64_t size;
    getTensorShapeAndSize(op->getOperand(i), shape, size);
    // TODO: this looks very strange. 4 allocated for each input
    // TODO: but only 1 is set for each input
    input_dims[i] = shape[axis];
  }

  int output_dim[SHAPE_DIM];
  int output_dim_size;
  std::vector<int64_t> shape;
  int64_t size;
  getTensorShapeAndSize(this->getResult(), shape, size);
  output_dim[0] = shape[0];
  output_dim[1] = shape[1];
  output_dim[2] = shape[2];
  output_dim[3] = shape[3];
  output_dim_size = shape.size();

  cvi_backend_tg_bf16_concat_kernel(
      *backend_ctx,
      0, // stream_id,
      0, // inst_id,
      layer_id, // layer_id,
      nullptr, // depends
      0, // depends_len
      ga_inputs,       // gaddr_t input_gaddrs[],
      ga_output,       // gaddr_t output_gaddr,
      input_dims,      // int input_dims[],
      nInputs,         // int input_num,
      axis,            // int concat_axis,
      output_dim_size, // int output_dim_size,
      output_dim,      // int *output_dim,
      do_relu(),
      0,               // int need_quantize_num
      nullptr          // threshold_x_quantized,
      );

  delete[] ga_inputs;
  delete[] input_dims;
  return success();
}

LogicalResult tpu::TG_INT8_CropOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = getOpLayerId(op);
  gaddr_t input_gaddr = getPreviousOpAddress(op);

  gaddr_t output_gaddr = getOpAddress(op);
  std::vector<int64_t> input_shape1 = getTensorShape(op->getOperand(0));
  std::vector<int64_t> output_shape = getTensorShape(this->getResult());

  // prepare data
  std::vector<int> i1_s;
  std::vector<int> i2_s;
  std::vector<int> o_s;
  std::vector<int> offsets;

  i1_s.assign(input_shape1.begin(), input_shape1.end());
  arrayAttrToVector(this->crop_shape().getValue(), i2_s);
  o_s.assign(output_shape.begin(), output_shape.end());
  arrayAttrToVector(this->crop_offset().getValue(), offsets);

  cvi_backend_tg_fixed_crop_kernel(*backend_ctx, // ctx,
                              0,            // stream_id
                              0,            // inst_id
                              layer_id,
                              nullptr,      // depends
                              0,            // depends_len
                              input_gaddr,  // bottom_gaddr,
                              output_gaddr, // top_gaddr
                              i1_s.data(), i2_s.data(), o_s.data(),
                              offsets.data(), CVK_FMT_I8);

  return success();
}

LogicalResult tpu::TG_BF16_CropOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = mlir::getOpLayerId(op);
  gaddr_t input_gaddr = getPreviousOpAddress(op);

  gaddr_t output_gaddr = getOpAddress(op);
  std::vector<int64_t> input_shape1 = getTensorShape(op->getOperand(0));
  std::vector<int64_t> output_shape = getTensorShape(this->getResult());

  // prepare data
  std::vector<int> i1_s;
  std::vector<int> i2_s;
  std::vector<int> o_s;
  std::vector<int> offsets;

  i1_s.assign(input_shape1.begin(), input_shape1.end());
  arrayAttrToVector(this->crop_shape().getValue(), i2_s);
  o_s.assign(output_shape.begin(), output_shape.end());
  arrayAttrToVector(this->crop_offset().getValue(), offsets);

  cvi_backend_tg_fixed_crop_kernel(*backend_ctx, // ctx,
                              0,            // stream_id
                              0,            // inst_id
                              layer_id,
                              nullptr,      // depends
                              0,            // depends_len
                              input_gaddr,  // bottom_gaddr,
                              output_gaddr, // top_gaddr
                              i1_s.data(), i2_s.data(), o_s.data(),
                              offsets.data(), CVK_FMT_BF16);

  return success();
}

LogicalResult tpu::TG_INT8_PT_Conv2DOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, pt, pb, pl, pr, dh, dw;
  parseConvParam(param(), false, input(), output(), filter(), n, ic, ih, iw, oc,
                 oh, ow, g, kh, kw, sh, sw, pt, pb, pl, pr, dh, dw, is_dw,
                 with_bias, do_relu);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  gaddr_t ga_filter = getWeightOpAddress(filter()->getDefiningOp());
  gaddr_t ga_bias = GA_INVALID;
  if ( with_bias ) {
    assert(!isTensorNone(pc_info()));
    ga_bias =  getWeightOpAddress(pc_info()->getDefiningOp());
  }
  assert(pt_rshift().hasValue());
  int8_t rshift = pt_rshift().getValue().getLimitedValue();
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
      GA_INVALID, // bn_mean_data_gaddr,
      GA_INVALID, // bn_variance_data_gaddr,
      GA_INVALID, // scale_gaddr,
      GA_INVALID, // scale_bias_gaddr,
      n, ic, ih, iw,
      g, // group,
      oc,
      kh, kw,
      dh, dw,
      pt, pb, pl, pr, // pad (t, b, l, r)
      0, 0, //ins_h, ins_w
      sh, sw,
      with_bias, // bias_term,
      0,         // do_bn,
      0,         // do_scale,
      0,         // do_scale_bias,
      do_relu ? 1 : 0, // do_activation,
      1.0f,      // bn_scale,
      1e-5,      // eps,
      0,         // param.activation(), method, 0 -> RELU, all others are invalide for now
      do_relu ? & fused_negative_slope : nullptr,   // activation_arg,
      GA_INVALID, //global_slope_gaddr,
      false,     //channel_shared,
      fused_leakyrelu_pos_m_i8,           // activation_gt_scale,
      fused_leakyrelu_pos_rshift,         // activation_gt_rshift,
      fused_leakyrelu_neg_m_i8,           // activation_le_scale,
      fused_leakyrelu_neg_rshift,         // activation_le_rshift,
      (int)rshift, // right_shift_width,
      0,         //bn_right_shift_width,
      0,         //scale_right_shift_width,
      false,     // do_chl_quan
      do_ic_alignment,
      false,     // store_compr_act
      false      // load_compr_act
      );

  return success();
}

LogicalResult tpu::TG_INT8_PC_Conv2DOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, pt, pb, pl, pr, dh, dw;
  parseConvParam(param(), false, input(), output(), filter(), n, ic, ih, iw, oc,
                 oh, ow, g, kh, kw, sh, sw, pt, pb, pl, pr, dh, dw, is_dw,
                 with_bias, do_relu);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  gaddr_t ga_filter = getWeightOpAddress(filter()->getDefiningOp());
  gaddr_t ga_pc_info = getWeightOpAddress(pc_info()->getDefiningOp());
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

  bool storeComprAct = this->store_compr_act().hasValue() ?
                       this->store_compr_act().getValue() : false;
  bool loadComprAct = this->load_compr_act().hasValue() ?
                      this->load_compr_act().getValue() : false;

  cvi_backend_tg_fixed_conv_kernel(
      *backend_ctx,
      layer_id, // layer_id,
      ga_input,   // input_data_gaddr,
      ga_output,  // output_data_gaddr,
      ga_filter,  // weight_data_gaddr,
      ga_pc_info, // bias_data_gaddr,
      GA_INVALID, // bn_mean_data_gaddr,
      GA_INVALID, // bn_variance_data_gaddr,
      GA_INVALID, // scale_gaddr,
      GA_INVALID, // scale_bias_gaddr,
      n, ic, ih, iw,
      g, // group,
      oc,
      kh, kw,
      dh, dw,
      pt, pb, pl, pr, // pad (t, b, l, r)
      0, 0, //ins_h, ins_w
      sh, sw,
      with_bias, // bias_term,
      0,         // do_bn,
      0,         // do_scale,
      0,         // do_scale_bias,
      do_relu ? 1 : 0, // do_activation,
      1.0f,      // bn_scale,
      1e-5,      // eps,
      0,         // param.activation(), method, 0 -> RELU, all others are invalide for now
      do_relu ? & fused_negative_slope : nullptr,   // activation_arg,
      GA_INVALID, // global_slope_gaddr,
      false,     // channel_shared,
      fused_leakyrelu_pos_m_i8,           // activation_gt_scale,
      fused_leakyrelu_pos_rshift,         // activation_gt_rshift,
      fused_leakyrelu_neg_m_i8,           // activation_le_scale,
      fused_leakyrelu_neg_rshift,         // activation_le_rshift,
      0,         // (int)rshift[0], //right_shift_width,
      0,         // bn_right_shift_width,
      0,         // scale_right_shift_width,
      true,      // do_chl_quan
      do_ic_alignment,
      storeComprAct,
      loadComprAct
      );

  return success();
}

LogicalResult tpu::TG_BF16_Conv2DOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, pt, pb, pl, pr, dh, dw;
  parseConvParam(param(), false, input(), output(), filter(),
                 n, ic, ih, iw, oc, oh, ow, g,
                 kh, kw, sh, sw, pt, pb, pl, pr, dh, dw, is_dw, with_bias, do_relu);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  gaddr_t ga_filter = getWeightOpAddress(filter()->getDefiningOp());
  gaddr_t ga_bias = GA_INVALID;
  if ( with_bias ) {
    assert(!isTensorNone(pc_info()));
    ga_bias =  getWeightOpAddress(pc_info()->getDefiningOp());
  }
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_bf16_conv_kernel(
      *backend_ctx,
      layer_id,  // layer_id
      ga_input,
      ga_output,
      ga_filter,
      ga_bias,
      GA_INVALID, // ga_bn_mean
      GA_INVALID, // ga_bn_variance
      GA_INVALID, // ga_scale
      GA_INVALID, // ga_scale_bias
      n, ic, ih, iw,
      g, // group
      oc,
      kh, kw,
      dh, dw,
      pt, pb, pl, pr, // pad (t, b, l, r)
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
      GA_INVALID //global_slope_gaddr
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
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, pt, pb, pl, pr, dh, dw;
  parseConvParam(param(), false, input(), output(), filter(),
                 n, ic, ih, iw, oc, oh, ow, g,
                 kh, kw, sh, sw, pt, pb, pl, pr, dh, dw, is_dw, with_bias, do_relu);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  gaddr_t ga_filter = getWeightOpAddress(filter()->getDefiningOp());
  gaddr_t ga_pc_info = getWeightOpAddress(pc_info()->getDefiningOp());
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
      GA_INVALID, // bn_mean_data_gaddr,
      GA_INVALID, // bn_variance_data_gaddr,
      GA_INVALID, // scale_gaddr,
      GA_INVALID, // scale_bias_gaddr,
      n, ic, ih, iw,
      g, // group,
      oc,
      kh, kw,
      dh, dw,
      pad_t, pad_b, pad_l, pad_r,
      ins_h, ins_w,
      stride_h, stride_w,
      with_bias, // bias_term,
      0,         // do_bn,
      0,         // do_scale,
      0,         // do_scale_bias,
      do_relu ? 1 : 0, // do_activation,
      0,         // bn_scale,
      0,         // eps,
      0,         // param.activation(), method, 0 -> RELU, all others are invalide for now
      nullptr,   // activation_arg,
      GA_INVALID, // global_slope_gaddr,
      false,     // channel_shared,
      0,         // activation_gt_scale,
      0,         // activation_gt_rshift,
      0,         // activation_le_scale,
      0,         // activation_le_rshift,
      0,         // (int)rshift[0], //right_shift_width,
      0,         // bn_right_shift_width,
      0,         // scale_right_shift_width,
      do_chl_quan,      // do_chl_quan
      false,
      false,
      false
      );

  return success();
}

LogicalResult tpu::TG_BF16_DeConv2DOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  //CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  //Operation *op = this->getOperation();

  std::string errorMsg = "unsupported tg op " + getOpName().str();
  llvm_unreachable(errorMsg.c_str());
}

LogicalResult tpu::TG_INT8_DilateOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = getOpLayerId(op);
  gaddr_t input_gaddr = getPreviousOpAddress(op);

  auto fill_constant = this->fill_constant().getLimitedValue();
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
        0,            // stream_id
        0,            // inst_id
        layer_id,
        nullptr,      // depends
        0,            // depends_len
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
  //CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  //Operation *op = this->getOperation();
  std::string errorMsg = "unsupported tg op " + getOpName().str();
  llvm_unreachable(errorMsg.c_str());
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
  int32_t early_stride_h = this->early_stride_h().getLimitedValue();
  int32_t early_stride_w = this->early_stride_w().getLimitedValue();
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
    rshift = this->rshift().getValue().getLimitedValue();

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
  std::vector<int>coeffs(input_number, 1);

   cvi_backend_tg_fixed_eltwise_add_kernel(
      *backend_ctx, layer_id,
      ga_inputs, ga_output,
      input_number, n, c, h, w,
      do_relu, do_early_stride,
      early_stride_h, early_stride_w,
      do_quant_rescale ? rshift_int : 0,
      do_quant_rescale ? m_int : nullptr,
      coeffs.data());

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
  int32_t early_stride_h = this->early_stride_h().getLimitedValue();
  int32_t early_stride_w = this->early_stride_w().getLimitedValue();
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
    rshift = this->rshift().getValue().getLimitedValue();

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
  int32_t early_stride_h = this->early_stride_h().getLimitedValue();
  int32_t early_stride_w = this->early_stride_w().getLimitedValue();
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
    rshift = this->rshift().getValue().getLimitedValue();

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
    auto defOp = op->getOperand(i)->getDefiningOp();
    if (isa<tpu::LoadWeightOp>(defOp)) {
      ga_inputs[i] = getWeightOpAddress(defOp);
    } else {
      ga_inputs[i] = getOpAddress(defOp);
    }
  }

  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  assert(this->rshift().hasValue());
  int8_t rshift = this->rshift().getValue().getLimitedValue();
  assert(this->m_i32_output().hasValue());
  int32_t m_i32_output = this->m_i32_output().getValue().getLimitedValue();

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
  int32_t early_stride_h = this->early_stride_h().getLimitedValue();
  int32_t early_stride_w = this->early_stride_w().getLimitedValue();
  if (do_early_stride) {
    assert(oh == h / early_stride_h);
    assert(ow == w / early_stride_w);
  }

  int32_t input_number = op->getNumOperands();
  auto ga_inputs = new gaddr_t[input_number];
  for (int i = 0; i < input_number; i++) {
    ga_inputs[i] = getPreviousOpAddress(op, i);
  }
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  // only need two coeff now, here just for safety coding
  std::vector<float> coeffs(input_number, 1.0f);

  cvi_backend_tg_bf16_eltwise_kernel(
      *backend_ctx,
      layer_id,     // layer_id
      ga_inputs,    // gaddr_t ga_input[]
      ga_output,    // gaddr_t ga_output
      input_number,            // int input_size
      1,            // int op, 0: prod, 1: sum, 2: max
      n, c, h, w,
      do_relu,      // bool do_relu
      0.0f,         // float relu_slope
      do_early_stride, early_stride_h, early_stride_w,
      coeffs.data());

  delete[] ga_inputs;
  return success();
}

LogicalResult tpu::TG_BF16_EltwiseMaxOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  //CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  //Operation *op = this->getOperation();

  std::string errorMsg = "unsupported tg op " + getOpName().str() + "\n";
  llvm_unreachable(errorMsg.c_str());
}

LogicalResult tpu::TG_BF16_EltwiseMinOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  //CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  //Operation *op = this->getOperation();

  std::string errorMsg = "unsupported tg op " + getOpName().str() + "\n";
  llvm_unreachable(errorMsg.c_str());
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

  cvi_backend_tg_bf16_eltwise_kernel(
      *backend_ctx,
      layer_id,     // layer_id
      ga_inputs,    // gaddr_t ga_input[]
      ga_output,    // gaddr_t ga_output
      2,            // int input_size
      0,            // int op, 0: prod, 1: sum, 2: max
      n, c, h, w,
      do_relu,      // bool do_relu
      0.0f,         // float relu_slope
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
  gaddr_t ga_filter = getWeightOpAddress(filter()->getDefiningOp());
  gaddr_t ga_bias = GA_INVALID;
  bool with_bias = false;
  if ( !isTensorNone(bias()) ) {
    ga_bias = getWeightOpAddress(bias()->getDefiningOp());
    with_bias = true;
  }
  int layer_id = getOpLayerId(op);

  int8_t rshift_int8 = rshift().getValue().getLimitedValue();
  int rshift = static_cast<int>(rshift_int8);

  cvi_backend_tg_fixed_fc_kernel(
      *backend_ctx,
      0, // stream_id,
      0, // inst_id,
      layer_id, // layer_id,
      nullptr, // depends
      0, // depends_len
      ga_input, // input_data_gaddr,
      ga_filter, // weight_data_gaddr,
      ga_bias, // bias_data_gaddr,
      ga_output, // output_data_gaddr,
      m, // int in_row,
      k, // int in_col,
      n, // int out_col,
      with_bias, // int have_bias,
      do_relu ? 1 : 0, // do_activation,
      0, // activation_method,
      GA_INVALID, // activation_ga_slope,
      0, // int activation_channel_shared,
      0, // int activation_gt_scale,
      0, // int activation_gt_rshift,
      0, // int activation_le_scale,
      0, // int activation_le_rshift,
      false, // weight_tp,
      3,     // int left_shift_width, // #define DEFAULT_FC_LEFT_SHIFT 3
      (int)rshift, // rshift
      0,     //int threshold_x_quantized_len,
      nullptr, //const int *threshold_x_quantized,
      nullptr  //const int *right_shift_array
      );

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
  gaddr_t ga_filter = getWeightOpAddress(filter()->getDefiningOp());
  gaddr_t ga_bias = GA_INVALID;
  bool with_bias = false;
  if ( !isTensorNone(bias()) ) {
    ga_bias = getWeightOpAddress(bias()->getDefiningOp());
    with_bias = true;
  }
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_bf16_fc_kernel(
      *backend_ctx,
      layer_id, // layer_id
      ga_input, // input_data_gaddr
      ga_filter, // weight_data_gaddr
      ga_bias, // bias_data_gaddr
      ga_output, // output_data_gaddr
      m, // int in_row
      k, // int in_col
      n, // in out_col,
      with_bias, // has_bias
      do_relu ? 1 : 0, // do_activation
      0  // activation_method
      );

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
    auto addr = getOpAddress(operand->getDefiningOp());
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
    auto addr = getOpAddress(operand->getDefiningOp());
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

  cvi_backend_tg_fixed_leakyrelu_kernel(
    *backend_ctx,         // ctx
    0,                    // stream_id
    0,                    // inst_id
    layer_id,             // layer_id
    nullptr,              // depends
    0,                    // depends_len
    ga_input,             // input_gaddr
    ga_output,            // output_gaddr
    n,                    // input_n
    c,                    // input_c
    h,                    // input_h
    w,                    // input_w
    pos_rshift,           // GT_right_shift_width
    neg_rshift,           // LE_right_shift_width
    pos_m_i8,             // GT_scale
    neg_m_i8,             // LE_scale
    0,                    // threshold_x_quantized_len
    nullptr,              // threshold_x_quantized
    nullptr               // right_shift_array
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
  gaddr_t power_lut_gaddr = getWeightOpAddress(power_lut()->getDefiningOp());
  gaddr_t sqr_lut_gaddr = getWeightOpAddress(sqr_lut()->getDefiningOp());
  int layer_id = getOpLayerId(op);
  cvi_backend_tg_fixed_lrn_kernel(
      *backend_ctx, 0, 0, layer_id, nullptr, 0, input_gaddr, output_gaddr,
      sqr_lut_gaddr, power_lut_gaddr, n, c, h, w,
      local_size().getLimitedValue(), sum_rshift().getLimitedValue(),
      lrn_rshift().getLimitedValue(), quant_data0().getLimitedValue(),
      quant_data1().getLimitedValue());
  return success();
}

LogicalResult tpu::TG_BF16_LrnOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  // TODO:
  // CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  // Operation *op = this->getOperation();
  std::string errorMsg = "unsupported tg op " + getOpName().str() + "\n";
  llvm_unreachable(errorMsg.c_str());
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
  gaddr_t y0_table_gaddr = getWeightOpAddress(table()->getDefiningOp());
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_fixed_lut_kernel(*backend_ctx,
                             0,        // stream_id,
                             0,        // inst_id,
                             layer_id, // layer_id,
                             nullptr,  // const u32 *depends,
                             0,        // depends_len,
                             input_gaddr, output_gaddr, y0_table_gaddr, n, c, h,
                             w, CVK_FMT_I8);

  return success();
}

LogicalResult tpu::TG_BF16_GruOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t tensorSize, seq_len, batchSize, inputSize, garbage;
  getTensorShapeAndSize(op->getOperand(0), shape, tensorSize);
  getNCHW(shape, seq_len, batchSize, inputSize, garbage);

  int64_t seq_len2, outputC, outputH, hiddenSize;
  getTensorShapeAndSize(this->getResult(), shape, tensorSize);
  getNCHW(shape, seq_len2, outputC, outputH, hiddenSize);
  assert(seq_len == seq_len2);

  bool with_bias = (!isTensorNone(bias()));
  gaddr_t ga_bias = GA_INVALID;
  if ( with_bias ) {
    ga_bias =  getWeightOpAddress(bias()->getDefiningOp());
  }

  bool is_linear_before_reset = this->linear_before_reset();
  bool is_bidirectional = this->bidirectional();

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  gaddr_t weight_gaddr = getWeightOpAddress(weight()->getDefiningOp());
  gaddr_t recurrence_gaddr = getWeightOpAddress(recurrence()->getDefiningOp());
  gaddr_t initial_h_gaddr = getWeightOpAddress(initial_h()->getDefiningOp());
  gaddr_t sigmoid_table_data_lut_gaddr = getWeightOpAddress(sigmoid_table()->getDefiningOp());
  gaddr_t sigmoid_slope_table_data_lut_gaddr = getWeightOpAddress(sigmoid_slope_table()->getDefiningOp());
  gaddr_t tanh_table_data_lut_gaddr = getWeightOpAddress(tanh_table()->getDefiningOp());
  gaddr_t tanh_slope_table_data_lut_gaddr = getWeightOpAddress(tanh_slope_table()->getDefiningOp());
  int layer_id = getOpLayerId(op);

  LLVM_DEBUG(llvm::errs() << "input_gaddr: " << input_gaddr << "\n"
                          << "weight_gaddr: " << weight_gaddr << "\n"
                          << "recurrence_gaddr: " << recurrence_gaddr << "\n"
                          << "ga_bias: " << ga_bias << "\n"
                          << "initial_h_gaddr: " << initial_h_gaddr << "\n"
                          << "sigmoid_table_data_lut_gaddr: " << sigmoid_table_data_lut_gaddr << "\n"
                          << "sigmoid_slope_table_data_lut_gaddr: " << sigmoid_slope_table_data_lut_gaddr << "\n"
                          << "tanh_table_data_lut_gaddr: " << tanh_table_data_lut_gaddr << "\n"
                          << "tanh_slope_table_data_lut_gaddr: " << tanh_slope_table_data_lut_gaddr << "\n"
                          << "output_gaddr: " << output_gaddr << "\n"
                          << "seq_len: " << seq_len << "\n"
                          << "batchSize: " << batchSize << "\n"
                          << "inputSize: " << inputSize << "\n"
                          << "hiddenSize: " << hiddenSize << "\n"
                          << "with_bias: " << with_bias << "\n"
                          << "is_linear_before_reset: " << is_linear_before_reset << "\n"
                          << "is_bidirectional: " << is_bidirectional << "\n"
                          << "\n";);

  cvi_backend_tg_bf16_gru_kernel(*backend_ctx, layer_id,
                  input_gaddr, weight_gaddr, recurrence_gaddr,
                  ga_bias, initial_h_gaddr,
                  sigmoid_table_data_lut_gaddr, sigmoid_slope_table_data_lut_gaddr,
                  tanh_table_data_lut_gaddr, tanh_slope_table_data_lut_gaddr,
                  output_gaddr,
                  seq_len, batchSize, inputSize, hiddenSize,
                  with_bias, is_linear_before_reset, is_bidirectional);
  return success();
}

LogicalResult tpu::TG_INT8_GruOp::codegen(void *ctx) {
  assert(0);
  return success();
}

LogicalResult tpu::TG_BF16_LstmOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t tensorSize, seq_len, batchSize, inputSize, garbage;
  getTensorShapeAndSize(op->getOperand(0), shape, tensorSize);
  getNCHW(shape, seq_len, batchSize, inputSize, garbage);

  int64_t seq_len2, outputC, outputH, hiddenSize;
  getTensorShapeAndSize(this->getResult(), shape, tensorSize);
  getNCHW(shape, seq_len2, outputC, outputH, hiddenSize);

  bool with_bias = (!isTensorNone(bias()));
  gaddr_t ga_bias = GA_INVALID;
  if ( with_bias ) {
    ga_bias =  getWeightOpAddress(bias()->getDefiningOp());
  }

  bool is_bidirectional = this->bidirectional();

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  gaddr_t weight_gaddr = getWeightOpAddress(weight()->getDefiningOp());
  gaddr_t recurrence_gaddr = getWeightOpAddress(recurrence()->getDefiningOp());
  gaddr_t initial_h_gaddr = getWeightOpAddress(initial_h()->getDefiningOp());
  gaddr_t initial_c_gaddr = getWeightOpAddress(initial_c()->getDefiningOp());
  gaddr_t sigmoid_table_data_lut_gaddr = getWeightOpAddress(sigmoid_table()->getDefiningOp());
  gaddr_t sigmoid_slope_table_data_lut_gaddr = getWeightOpAddress(sigmoid_slope_table()->getDefiningOp());
  gaddr_t tanh_table_data_lut_gaddr = getWeightOpAddress(tanh_table()->getDefiningOp());
  gaddr_t tanh_slope_table_data_lut_gaddr = getWeightOpAddress(tanh_slope_table()->getDefiningOp());
  int layer_id = getOpLayerId(op);

  LLVM_DEBUG(llvm::errs() << "input_gaddr: " << input_gaddr << "\n"
                                                       << "weight_gaddr: " << weight_gaddr << "\n"
                                                       << "recurrence_gaddr: " << recurrence_gaddr << "\n"
                                                       << "ga_bias: " << ga_bias << "\n"
                                                       << "initial_h_gaddr: " << initial_h_gaddr << "\n"
                                                       << "initial_c_gaddr: " << initial_c_gaddr << "\n"
                                                       << "sigmoid_table_data_lut_gaddr: " << sigmoid_table_data_lut_gaddr << "\n"
                                                       << "sigmoid_slope_table_data_lut_gaddr: " << sigmoid_slope_table_data_lut_gaddr << "\n"
                                                       << "tanh_table_data_lut_gaddr: " << tanh_table_data_lut_gaddr << "\n"
                                                       << "tanh_slope_table_data_lut_gaddr: " << tanh_slope_table_data_lut_gaddr << "\n"
                                                       << "output_gaddr: " << output_gaddr << "\n"
                                                       << "seq_len: " << seq_len << "\n"
                                                       << "batchSize: " << batchSize << "\n"
                                                       << "inputSize: " << inputSize << "\n"
                                                       << "hiddenSize: " << hiddenSize << "\n"
                                                       << "with_bias: " << with_bias << "\n"
                                                       << "is_bidirectional: " << is_bidirectional << "\n"
                                                       << "\n";);

  cvi_backend_tg_bf16_lstm_kernel(*backend_ctx, layer_id,
                                     input_gaddr, weight_gaddr, recurrence_gaddr,
                                     ga_bias, initial_h_gaddr, initial_c_gaddr,
                                     sigmoid_table_data_lut_gaddr, sigmoid_slope_table_data_lut_gaddr,
                                     tanh_table_data_lut_gaddr, tanh_slope_table_data_lut_gaddr,
                                     output_gaddr,
                                     seq_len, batchSize, inputSize, hiddenSize,
                                     with_bias, is_bidirectional);
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
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  int axis = this->axis().getLimitedValue();

  std::vector<int64_t> shape;
  int64_t tensorSize, outer_size, inner_size;
  getTensorShapeAndSize(op->getOperand(0), shape, tensorSize);
  int dimension = shape.size();
  if (shape.size() == 2) {
    outer_size = shape[0];
    inner_size = shape[1];
  } else if (shape.size() == 4) {
    assert(axis == 1 && "Support only axis = 1 (Align c)");
    if(shape[2] * shape[3] == 1) {
      outer_size = shape[0]; //n
      inner_size = shape[1]; //c
    } else {
      // assert(0);
    }
  } else if (shape.size() == 3) {
    assert(axis == 2 && "Support only axis = 2");
    outer_size = shape[0] * shape[1]; //c * h
    inner_size = shape[2]; //w
  }


  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  gaddr_t exponential_table_data_lut_gaddr = getWeightOpAddress(exponential_table()->getDefiningOp());
  gaddr_t exponential_slope_table_data_lut_gaddr = getWeightOpAddress(exponential_slope_table()->getDefiningOp());
  gaddr_t reciprocal_table_data_lut_gaddr = getWeightOpAddress(reciprocal_table()->getDefiningOp());
  gaddr_t reciprocal_mantissa_table_data_lut_gaddr = getWeightOpAddress(reciprocal_mantissa_table()->getDefiningOp());
  int layer_id = getOpLayerId(op);

  LLVM_DEBUG(llvm::errs() << "input_gaddr: " << input_gaddr << "\n"
                                                       << "exponential_table_data_lut_gaddr: " << exponential_table_data_lut_gaddr << "\n"
                                                       << "exponential_slope_table_data_lut_gaddr: " << exponential_slope_table_data_lut_gaddr << "\n"
                                                       << "reciprocal_table_data_lut_gaddr: " << reciprocal_table_data_lut_gaddr << "\n"
                                                       << "reciprocal_mantissa_table_data_lut_gaddr: " << reciprocal_mantissa_table_data_lut_gaddr << "\n"
                                                       << "output_gaddr: " << output_gaddr << "\n"
                                                       << "outer_size: " << outer_size << "\n"
                                                       << "inner_size: " << inner_size << "\n"
                                                       << "\n";);

  cvi_backend_tg_bf16_softmax_kernel(*backend_ctx, layer_id,
                                            input_gaddr,
                                            exponential_table_data_lut_gaddr, exponential_slope_table_data_lut_gaddr,
                                            reciprocal_table_data_lut_gaddr, reciprocal_mantissa_table_data_lut_gaddr,
                                            output_gaddr,
                                            shape.data(), axis, dimension);
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
  gaddr_t table_data_lut = getWeightOpAddress(table()->getDefiningOp());
  gaddr_t table_data_mantissa_lut = getWeightOpAddress(table_mantissa()->getDefiningOp());


  int layer_id = getOpLayerId(op);
  auto lut_method = method().getValue().str();
  LLVM_DEBUG(llvm::errs() << "lut method:" << lut_method << " [" << getOpName()
                          << "]\n";);
  if(lut_method == "mantissa") {
    cvi_backend_tg_bf16_lut_scientific_kernel(*backend_ctx,
                                  0,        // stream_id,
                                  0,        // inst_id,
                                  layer_id, // layer_id,
                                  nullptr,  // const u32 *depends,
                                  0,        // depends_len,
                                  input_gaddr, output_gaddr, table_data_lut, table_data_mantissa_lut,
                                  n, c, h, w, CVK_FMT_BF16);
  } else if (lut_method == "slope") {
    cvi_backend_tg_bf16_lut_interpolation_kernel(
        *backend_ctx,
        0, // strean_id
        0, // inst_id,
        layer_id, nullptr, 0, input_gaddr, output_gaddr, table_data_lut,
        table_data_mantissa_lut, n, c, h, w, BF16_TABLE_START, BF16_TABLE_END,
        16 // scale
    );
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

  auto input_type = input()->getType().template cast<TensorType>();
  std::vector<int64_t> i_s(input_type.getShape());
  auto output_type = output()->getType().template cast<TensorType>();
  std::vector<int64_t> o_s(output_type.getShape());

  std::vector<int> orders;
  orders.push_back(this->order0().getLimitedValue());
  orders.push_back(this->order1().getLimitedValue());
  orders.push_back(this->order2().getLimitedValue());
  orders.push_back(this->order3().getLimitedValue());

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  int layer_id = getOpLayerId(op);
  // Check if we need to reorder the data or keep it.
  bool need_permute_ = false;
  int num_axes_ = i_s.size();

  for (int i = 0; i < num_axes_; ++i) {
    if (orders[i] != i) {
      // As long as there is one order which is different from the natural order
      // of the data, we need to permute. Otherwise, we share the data and diff.
      need_permute_ = true;
      break;
    }
  }
  cvi_backend_tg_fixed_premute_kernel(
      *backend_ctx,
      0, //stream_id,
      0, //inst_id,
      layer_id, //layer_id,
      nullptr, //const u32 *depends,
      0, //depends_len,
      input_gaddr,
      output_gaddr,
      i_s[0], i_s[1], i_s[2], i_s[3],
      o_s[0], o_s[1], o_s[2], o_s[3],
      orders[0], orders[1], orders[2], orders[3],
      need_permute_);
  return success();
}

LogicalResult tpu::TG_BF16_PermuteOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  auto input_type = input()->getType().template cast<TensorType>();
  std::vector<int64_t> i_s(input_type.getShape());
  auto output_type = output()->getType().template cast<TensorType>();
  std::vector<int64_t> o_s(output_type.getShape());

  std::vector<int64_t> i_nchw(4,1);
  std::vector<int64_t> o_nchw(4,1);

  for (uint64_t i = 0; i < i_s.size(); i++) {
    i_nchw[i] = i_s[i];
  }

  for (uint64_t i = 0; i < o_s.size(); i++) {
    o_nchw[i] = o_s[i];
  }

  std::vector<int> orders;
  orders.push_back(this->order0().getLimitedValue());
  orders.push_back(this->order1().getLimitedValue());
  orders.push_back(this->order2().getLimitedValue());
  orders.push_back(this->order3().getLimitedValue());

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  int layer_id = getOpLayerId(op);
  // Check if we need to reorder the data or keep it.
  bool need_permute_ = false;
  int num_axes_ = i_s.size();
  for (int i = 0; i < num_axes_; ++i) {
    if (orders[i] != i) {
      // As long as there is one order which is different from the natural order
      // of the data, we need to permute. Otherwise, we share the data and diff.
      need_permute_ = true;
      break;
    }
  }

  cvi_backend_tg_bf16_premute_kernel(
      *backend_ctx,
      0, //stream_id,
      0, //inst_id,
      layer_id, //layer_id,
      nullptr, //const u32 *depends,
      0, //depends_len,
      input_gaddr,
      output_gaddr,
      i_nchw[0], i_nchw[1], i_nchw[2], i_nchw[3],
      o_nchw[0], o_nchw[1], o_nchw[2], o_nchw[3],
      orders[0], orders[1], orders[2], orders[3],
      need_permute_);
  return success();
}

LogicalResult tpu::TG_INT8_PoolAvg2DOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  // parse param
  bool is_global, do_relu, count_include_pad;
  int n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr;
  parsePoolParam(param(), input(), output(),
                 n, c, ih, iw, oh, ow,
                 kh, kw, sh, sw, pt, pb, pl, pr,
                 is_global, do_relu, count_include_pad);
  assert(!do_relu);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  assert(this->rshift().hasValue());
  int8_t rshift = this->rshift().getValue().getLimitedValue();
  assert(this->m_i8().hasValue());
  int8_t m_i8 = this->m_i8().getValue().getLimitedValue();

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
  int n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr;
  parsePoolParam(param(), input(), output(),
                 n, c, ih, iw, oh, ow,
                 kh, kw, sh, sw, pt, pb, pl, pr,
                 is_global, do_relu, count_include_pad);
  assert(!do_relu);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  assert(!this->rshift().hasValue());
  assert(!this->m_i8().hasValue());


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
      true);

  return success();
}

LogicalResult tpu::TG_BF16_PoolAvg2DOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  // parse param
  bool is_global, do_relu, count_include_pad;
  int n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr;
  parsePoolParam(param(), input(), output(),
                 n, c, ih, iw, oh, ow,
                 kh, kw, sh, sw, pt, pb, pl, pr,
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
  int n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr;
  parsePoolParam(param(), input(), output(),
                 n, c, ih, iw, oh, ow,
                 kh, kw, sh, sw, pt, pb, pl, pr,
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
      getWeightOpAddress(negative_slope()->getDefiningOp());
  int layer_id = getOpLayerId(op);

  assert(this->rshift_pos().hasValue());
  int8_t rshift_pos = this->rshift_pos().getValue().getLimitedValue();
  assert(this->m_i8_pos().hasValue());
  int8_t m_i8_pos = this->m_i8_pos().getValue().getLimitedValue();
  assert(this->rshift_neg().hasValue());
  int8_t rshift_neg = this->rshift_neg().getValue().getLimitedValue();
  cvi_backend_tg_fixed_prelu_kernel(
      *backend_ctx,
      layer_id,             // layer_id,
      ga_input,             // input_data_gaddr,
      ga_output,            // output_data_gaddr,
      negative_scope_gaddr, // float negative_slope,
      n, c, h, w, rshift_pos, m_i8_pos, rshift_neg, CVK_FMT_I8);

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
  gaddr_t ga_neg_slope = getWeightOpAddress(op->getOperand(1)->getDefiningOp());
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

LogicalResult tpu::TG_INT8_QuantOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  // FIXME: rename to dequant, from low accuricy to higher
  // plz refre LowerToTG.cpp:867 for more details
  cvk_fmt_t from, to;
  if (this->from() == "INT8") {
    from = CVK_FMT_I8;
  } else if (this->from() == "UINT8") {
    from = CVK_FMT_U8;
  } else {
    std::stringstream err_msg;
    err_msg << " not support " << this->from().str() << "type\n";
    throw std::runtime_error(err_msg.str());
  }

  if (this->to() == "NONE") {
    to = CVK_FMT_F32;
  } else {
    to = CVK_FMT_BF16;
    assert(this->to() == "BF16");
  }

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  int layer_id = getOpLayerId(op);
  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  // dequant:
  // output[i] = input[i] * threshold / 128.0;
  float threshold = this->threshold().getValue().convertToFloat();
  float dequant = threshold / 128.0;

  // dequant to bf16/fp32
  mixed_precision_dequant(
          *backend_ctx,//CviBackendContext &ctx,
          layer_id,//u32 layer_id,
          from, to,
          ga_input,//gaddr_t bottom_gaddr,
          ga_output,//gaddr_t top_gaddr,
          n,//int input_n,
          c,//int input_c,
          h,//int input_h,
          w,//int input_w,
          dequant//float const_scale
          );

  return success();
}

LogicalResult tpu::TG_BF16_QuantOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";

  cvk_fmt_t from, to;
  if (this->from() == "NONE") {
    from = CVK_FMT_F32;
  }
  else {
    from = CVK_FMT_BF16;
    assert(this->from() == "BF16");
  }
  assert(this->to() == "INT8");
  to = CVK_FMT_I8;

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  int layer_id = getOpLayerId(op);
  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  // quant:
  // output[i] = (float)saturateInt8(input[i] * 128.0 / threshold);
  float threshold = this->threshold().getValue().convertToFloat();
  float quant = 128.0 / threshold;

  //  quant to int8
  mixed_precision_quant(
          *backend_ctx,//CviBackendContext &ctx,
          layer_id,//u32 layer_id,
          from, to,
          ga_input,//gaddr_t bottom_gaddr,
          ga_output,//gaddr_t top_gaddr,
          n,//int input_n,
          c,//int input_c,
          h,//int input_h,
          w,//int input_w,
          quant//float const_scale
          );
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

  cvi_backend_tg_fixed_relu_kernel(
      *backend_ctx,
      0, //u32 stream_id,
      0, //u32 inst_id
      layer_id,
      NULL,
      0, //
      ga_input,             // input_data_gaddr,
      ga_output,            // output_data_gaddr,
      -1, // float negative_slope,
      n, c, h, w,
      0,
      NULL, // *threshold_x_quantized,
      NULL, // *right_shift_array,
      CVK_FMT_I8);

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

  cvi_backend_tg_fixed_relu_kernel(
      *backend_ctx,
      0, //u32 stream_id,
      0, //u32 inst_id
      layer_id,
      NULL,
      0, //
      ga_input,             // input_data_gaddr,
      ga_output,            // output_data_gaddr,
      -1, // float negative_slope,
      n, c, h, w,
      0,
      NULL, // *threshold_x_quantized,
      NULL, // *right_shift_array,
      CVK_FMT_BF16);

  return success();
}

LogicalResult tpu::TG_INT8_ReorgOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  uint32_t stride = this->stride().getLimitedValue();

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_fixed_reorg_kernel(*backend_ctx, 0, 0, layer_id, nullptr, 0,
                                       input_gaddr, output_gaddr, n, c,
                                       h, w, stride);

  return success();
}

LogicalResult tpu::TG_BF16_ReorgOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  uint32_t stride = this->stride().getLimitedValue();

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_bf16_reorg_kernel(*backend_ctx, layer_id,
                            input_gaddr, output_gaddr,
                            n, c, h, w, stride);

  return success();
}

LogicalResult tpu::TG_INT8_ShuffleChannelOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  int frame_size = h * w;
  uint32_t group = this->group().getLimitedValue();

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  int layer_id = getOpLayerId(op);
  cvi_backend_tg_shuffle_channel_kernel(*backend_ctx, 0, 0, layer_id, nullptr, 0,
                                       input_gaddr, output_gaddr, n, c,
                                       frame_size, group, CVK_FMT_I8);
  return success();
}

LogicalResult tpu::TG_BF16_ShuffleChannelOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  int frame_size = h * w;
  uint32_t group = this->group().getLimitedValue();

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  int layer_id = getOpLayerId(op);
  cvi_backend_tg_shuffle_channel_kernel(*backend_ctx, 0, 0, layer_id, nullptr, 0,
                                       input_gaddr, output_gaddr, n, c,
                                       frame_size, group, CVK_FMT_BF16);
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
  arrayAttrToVector(this->channel_order().getValue(), order);
  cvi_backend_tg_swap_channel_kernel(*backend_ctx, 0, 0, layer_id, nullptr, 0,
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
  arrayAttrToVector(this->channel_order().getValue(), order);

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  int layer_id = getOpLayerId(op);
  cvi_backend_tg_swap_channel_kernel(*backend_ctx, 0, 0, layer_id, nullptr, 0,
                                       input_gaddr, output_gaddr,  (int)input_shape_fix.size(),
                                       input_shape_fix.data(), order.data(), CVK_FMT_BF16);
  return success();
}

LogicalResult tpu::TG_INT8_TileOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  // backend not ok now
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> input_shape = getTensorShape(input());
  std::vector<int64_t> output_shape = getTensorShape(this->getResult());

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  int layer_id = getOpLayerId(op);
  std::vector<int32_t> resp;
  arrayAttrToVector(this->resp().getValue(), resp);

  assert(resp[0] == 1 && resp[1] == 1 && "only support hw tile");
  int tile_h = resp[2];
  int tile_w = resp[3];
  resp.clear();
  resp = {tile_h, tile_w};


  int64_t input_n, input_c, input_h, input_w;
  int64_t output_n, output_c, output_h, output_w;
  getNCHW(input_shape, input_n, input_c, input_h, input_w);
  getNCHW(output_shape, output_n, output_c, output_h, output_w);

  // axis order is nchw
  // only support resp.size() == 2, hw tile
  // resp[0] is h tile, resp[1] is w tile
  cvi_backend_tg_tile_kernel(*backend_ctx,
      input_gaddr, input_n, input_c, input_h, input_w, CVK_FMT_I8,
      output_gaddr, output_n, output_c, output_h, output_w, CVK_FMT_I8,
      resp.data(), resp.size(), layer_id);

  return success();
}

LogicalResult tpu::TG_BF16_TileOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  // backend not ok now
#if 0
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> input_shape = getTensorShape(input());
  std::vector<int64_t> output_shape = getTensorShape(this->getResult());

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  int layer_id = getOpLayerId(op);
  std::vector<int32_t> resp;
  arrayAttrToVector(this->resp().getValue(), resp);
  tile_forward_kernel(*backend_ctx, 0, 0, layer_id, nullptr, 0,
                      input_gaddr, output_gaddr,
                      input_shape.size(), input_shape.data(),
                      output_shape.size(), output_shape.data(),
                      resp.size(), resp.data(), CVK_FMT_BF16);
#endif
  std::string errorMsg = "unsupported tg op " + getOpName().str() + "\n";
  llvm_unreachable(errorMsg.c_str());
  return success();
}

LogicalResult tpu::TG_INT8_TileInterpOp::codegen(void *ctx) {
  /*
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> input_shape = getTensorShape(input());

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  int layer_id = getOpLayerId(op);
  std::vector<int32_t> resp;
  arrayAttrToVector(this->resp().getValue(), resp);
  */
  std::string errorMsg = "unsupported tg op " + getOpName().str() + "\n";
  llvm_unreachable(errorMsg.c_str());
  return success();
}

LogicalResult tpu::TG_BF16_TileInterpOp::codegen(void *ctx) {
  /*
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  */
  std::string errorMsg = "unsupported tg op " + getOpName().str() + "\n";
  llvm_unreachable(errorMsg.c_str());

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
  uint32_t upscale_factor = this->upscale_factor().getLimitedValue();

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  int layer_id = getOpLayerId(op);
  cvi_backend_tg_fixed_pixel_shuffle_kernel(
      *backend_ctx, //const CviBackendContext &ctx,
      0, //u32 stream_id,
      0, //u32 inst_id,
      layer_id,
      NULL, //const u32 *depends,
      0, //u32 depends_len,
      input_gaddr,
      output_gaddr,
      n,
      c,
      h,
      w,
      upscale_factor);

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
  c = shape[1] * shape[2] * shape[3];
  h = shape[4];
  w = shape[5];
  uint32_t upscale_factor = this->upscale_factor().getLimitedValue();

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_bf16_pixel_shuffle_kernel(
      *backend_ctx, //const CviBackendContext &ctx,
      0, //u32 stream_id,
      0, //u32 inst_id,
      layer_id,
      NULL, //const u32 *depends,
      0, //u32 depends_len,
      input_gaddr,
      output_gaddr,
      n,
      c,
      h,
      w,
      upscale_factor);

  return success();
}

LogicalResult tpu::TG_INT8_ClipOp::codegen(void *ctx) {
  std::string errorMsg = "unsupported tg op " + getOpName().str() + "\n";
  llvm_unreachable(errorMsg.c_str());
}

LogicalResult tpu::TG_BF16_ClipOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName() << " [" << getOpName()
               << "]\n";
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

  // op definition refer to \bf16_eltwise_kernel.cpp
  if (0) {
    coeffs[0] = {max};
    cvi_backend_tg_bf16_eltwise_kernel(
        *backend_ctx,
        layer_id,     // layer_id
        ga_inputs,    // gaddr_t ga_input[]
        output_gaddr,    // gaddr_t ga_output
        1,            // int input_size
        2,            // int op, 0: prod, 1: sum, 2: max
        n, c, h, w,
        do_relu,      // bool do_relu
        0.0f,         // float relu_slope
        false, 0, 0,
        coeffs);

    coeffs[0] = {min};
    cvi_backend_tg_bf16_eltwise_kernel(
        *backend_ctx,
        layer_id,     // layer_id
        ga_inputs,    // gaddr_t ga_input[]
        output_gaddr,    // gaddr_t ga_output
        1,            // int input_size
        3,            // int op, 0: prod, 1: sum, 2: max, 3: min
        n, c, h, w,
        do_relu,      // bool do_relu
        0.0f,         // float relu_slope
        false, 0, 0,
        coeffs);
  }
  else {
    coeffs[0] = {max};
    coeffs[1] = {min};
    cvi_backend_tg_bf16_eltwise_kernel(
        *backend_ctx,
        layer_id,     // layer_id
        ga_inputs,    // gaddr_t ga_input[]
        output_gaddr,    // gaddr_t ga_output
        1,            // int input_size
        4,            // int op, 0: prod, 1: sum, 2: max, 3: min, 4:max_min
        n, c, h, w,
        do_relu,      // bool do_relu
        0.0f,         // float relu_slope
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

  int axis = this->axis().getLimitedValue();
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
  int offset = this->offset().getLimitedValue();
  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  int layer_id = getOpLayerId(op);
  std::vector<int64_t> output_shape = getTensorShape(this->getResult());
  cvi_backend_tg_slice_kernel(*backend_ctx, 0, 0, layer_id, nullptr, 0, input_gaddr,
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

  int axis = this->axis().getLimitedValue();
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
  int offset = this->offset().getLimitedValue();
  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  int layer_id = getOpLayerId(op);
  std::vector<int64_t> output_shape = getTensorShape(this->getResult());
  cvi_backend_tg_slice_kernel(*backend_ctx, 0, 0, layer_id, nullptr, 0, input_gaddr,
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
  int32_t scale_h = this->scale_h().getLimitedValue();
  int32_t scale_w = this->scale_w().getLimitedValue();

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_fixed_upsample_kernel(
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
      scale_h,
      scale_w
  );

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
  int32_t scale_h = this->scale_h().getLimitedValue();
  int32_t scale_w = this->scale_w().getLimitedValue();

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = getOpLayerId(op);

  cvi_backend_tg_bf16_upsample_kernel(
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
      scale_h,
      scale_w
  );

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
      CVK_FMT_BF16
  );

  return success();
}

LogicalResult tpu::TG_INT8_ReduceMeanOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  assert(0);

  return success();
}

LogicalResult tpu::TG_INT8_ReduceMaxOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  assert(0);

  return success();
}

LogicalResult tpu::TG_BF16_ReduceMeanOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  assert(0);

  return success();
}

LogicalResult tpu::TG_BF16_ReduceMaxOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";);
  assert(0);

  return success();
}

LogicalResult tpu::TG_INT8_ZeroMaskOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> input_shape = getTensorShape(input());

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  int layer_id = getOpLayerId(op);
  // y = relu(x * 1 + 1)
  cvi_backend_tg_fixed_mac_const_kernel(
      *backend_ctx, 0, 0, layer_id, nullptr, 0, input_gaddr, output_gaddr,
      (int)input_shape[0], (int)input_shape[1], (int)input_shape[2],
      (int)input_shape[3], 1, 1, true);
  return success();
}

LogicalResult tpu::TG_BF16_ZeroMaskOp::codegen(void *ctx) {
  LLVM_DEBUG(llvm::errs() << "TG_codegen: " << getOperationName() << " ["
                          << getOpName() << "]\n";);
  // backend not ok now
  std::string errorMsg = "unsupported tg op " + getOpName().str() + "\n";
  llvm_unreachable(errorMsg.c_str());
  return success();
}

// MemRefType dummy
LogicalResult tpu::TG_MemRef_INT8_BroadcastMulOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_BF16_BroadcastMulOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_ConcatOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_BF16_ConcatOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_CropOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_BF16_CropOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_ClipOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_BF16_ClipOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_PT_Conv2DOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_PC_Conv2DOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_BF16_Conv2DOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_PT_DeConv2DOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_PC_DeConv2DOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_BF16_DeConv2DOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_EltwiseAddOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_EltwiseMaxOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_EltwiseMinOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_EltwiseMulOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_BF16_EltwiseAddOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_BF16_EltwiseMaxOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_BF16_EltwiseMinOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_BF16_EltwiseMulOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_BF16_FullyConnectedOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_FullyConnectedOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_BF16_GenericTpuOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_GenericTpuOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_BF16_LeakyReluOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_LeakyReluOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_LutOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_BF16_LutOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_GruOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_BF16_GruOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_LstmOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_BF16_LstmOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_SoftmaxOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_BF16_SoftmaxOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_LrnOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_BF16_LrnOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_BF16_PermuteOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_PermuteOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_PoolAvg2DOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_BF16_PoolAvg2DOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_PoolMax2DOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_BF16_PoolMax2DOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_PReluOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_BF16_PReluOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_QuantOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_BF16_QuantOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_ReluOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_BF16_ReluOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_ReorgOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_BF16_ReorgOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_ReduceMeanOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_ReduceMaxOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_BF16_ReduceMeanOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_BF16_ReduceMaxOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_ShuffleChannelOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_BF16_ShuffleChannelOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_SwapChannelOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_BF16_SwapChannelOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_PixelShuffleOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_BF16_PixelShuffleOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_CastOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_ReshapeOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_SliceOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_BF16_SliceOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_UpsampleOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_BF16_UpsampleOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_PadOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_BF16_PadOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_ZeroMaskOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_BF16_ZeroMaskOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_CallOp::codegen(void *ctx) {
  return success();
}
}

