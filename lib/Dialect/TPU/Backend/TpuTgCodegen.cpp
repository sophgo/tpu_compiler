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


template <typename OpTy>
static Operation* getNextOp(const OpTy &op) {
  assert(op->getResult()->hasOneUse());
  Operation *nextOp = nullptr;
  for (auto &use : op->getResult()->getUses()) {
    nextOp = use.getOwner();
    break;
  }
  assert(nextOp);
  return nextOp;
}

static void parseTgLeakyReluParam(Operation *op,
    int8_t &pos_rshift, int8_t &pos_m_i8,
    int8_t &neg_rshift, int8_t &neg_m_i8,
    float &negative_slope) {
  auto lreluOp = llvm::dyn_cast<tpu::TG_INT8_LeakyReluOp>(op);
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

LogicalResult tpu::TG_INT8_BroadcastMulOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
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
  int layer_id = mlir::getOpLayerId(op);

  scale_fixed_forward_qi32(
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
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  //CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  //Operation *op = this->getOperation();

  assert(false);
  return success();
}

LogicalResult tpu::TG_INT8_ConcatOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  unsigned nInputs = op->getNumOperands();
  gaddr_t ga_inputs[nInputs];
  for ( unsigned i = 0; i < nInputs; i++) {
    ga_inputs[i] = getPreviousOpAddress(op, i);
  }
  gaddr_t ga_output = getOpAddress(op);
  int axis = this->axis().getLimitedValue();
  int layer_id = mlir::getOpLayerId(op);

  // prepare shape info
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
  int8_t rshift;
  int8_t m_i8_input[nInputs];
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
  int rshift_int[nInputs];
  int m_int[nInputs];
  if (do_quant_rescale) {
    for (unsigned i = 0; i < nInputs; ++i) {
      rshift_int[i] = static_cast<int>(rshift);
      m_int[i] = static_cast<int>(m_i8_input[i]);
    }
  }

  bmnet_concat_fixed_forward_bmkernel(
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
      do_quant_rescale ? nInputs : 0,     // const int need_quantize_num,
      do_quant_rescale ? rshift_int : 0,  // const int *right_shift_width,
      do_quant_rescale ? m_int : nullptr  // const int *threshold_x_quantized
      );

  return success();
}

LogicalResult tpu::TG_BF16_ConcatOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  int nInputs = op->getNumOperands();
  gaddr_t ga_inputs[nInputs];
  for ( int i = 0; i < nInputs; i++) {
    ga_inputs[i] = getPreviousOpAddress(op, i);
  }
  gaddr_t ga_output = getOpAddress(op);
  int axis = this->axis().getLimitedValue();
  int layer_id = mlir::getOpLayerId(op);

  // prepare shape info
  #define SHAPE_DIM 4
  int32_t input_dims[nInputs * SHAPE_DIM];
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

  bf16_concat_fixed_forward_bmkernel(
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
      0,               // int need_quantize_num
      nullptr          // threshold_x_quantized,
      );

  return success();
}

LogicalResult tpu::TG_INT8_CropOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName() << " [" << getOpName()
               << "]\n";
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

  crop_fixed_forward_bmkernel(*backend_ctx, // ctx,
                              0,            // stream_id
                              0,            // inst_id
                              layer_id,
                              nullptr,      // depends
                              0,            // depends_len
                              input_gaddr,  // bottom_gaddr,
                              output_gaddr, // top_gaddr
                              i1_s.data(), i2_s.data(), o_s.data(),
                              offsets.data(), FMT_I8);

  return success();
}

LogicalResult tpu::TG_BF16_CropOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  //CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  //Operation *op = this->getOperation();

  assert(false);
  return success();
}

LogicalResult tpu::TG_INT8_PT_Conv2DOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
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
  gaddr_t ga_bias = GA_INVALID;
  if ( with_bias ) {
    assert(!isTensorNone(pc_info()));
    ga_bias =  getWeightOpAddress(pc_info()->getDefiningOp());
  }
  assert(pt_rshift().hasValue());
  int8_t rshift = pt_rshift().getValue().getLimitedValue();
  int layer_id = mlir::getOpLayerId(op);

  // check if fused with a leakyrelu
  int fused_leakyrelu_pos_rshift = 0;
  int fused_leakyrelu_pos_m_i8 = 0;
  int fused_leakyrelu_neg_rshift = 0;
  int fused_leakyrelu_neg_m_i8 = 0;
  float fused_negative_slope = 0.0f;
  if (this->fuse_next()) {
    Operation *nextOp = getNextOp(this);
    int8_t pos_rshift, pos_m_i8, neg_rshift, neg_m_i8;
    float negativeSlope;
    parseTgLeakyReluParam(nextOp,
        pos_rshift, pos_m_i8, neg_rshift, neg_m_i8, negativeSlope);

    assert(neg_m_i8);
    // TODO: fix the type in backend API
    fused_leakyrelu_pos_rshift = static_cast<int>(pos_rshift);
    fused_leakyrelu_pos_m_i8   = static_cast<int>(pos_m_i8);
    fused_leakyrelu_neg_rshift = static_cast<int>(neg_rshift);
    fused_leakyrelu_neg_m_i8   = static_cast<int>(neg_m_i8);
    fused_negative_slope       = negativeSlope;
    do_relu = true;

    llvm::errs() << "  fused leaky relu, pos ("
        << fused_leakyrelu_pos_m_i8 << ", " << fused_leakyrelu_pos_rshift
        << "), neg ("
        << fused_leakyrelu_neg_m_i8 << ", " << fused_leakyrelu_neg_rshift
        << ")\n";

    // finally, change gaddr to the nextOp's
    ga_output = getOpAddress(nextOp);
  }

  cvi_backend_tg_int8_conv(
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
      GA_INVALID, // bn_mean_data_gaddr,
      GA_INVALID, // bn_variance_data_gaddr,
      GA_INVALID, // scale_gaddr,
      GA_INVALID, // scale_bias_gaddr,
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
      false,     //use_winograd
      0,         // right_shift_array_len
      0          // ga_per_channel
      );

  return success();
}

LogicalResult tpu::TG_INT8_PC_Conv2DOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
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

  // check if fused with a leakyrelu
  int fused_leakyrelu_pos_rshift = 0;
  int fused_leakyrelu_pos_m_i8 = 0;
  int fused_leakyrelu_neg_rshift = 0;
  int fused_leakyrelu_neg_m_i8 = 0;
  float fused_negative_slope = 0.0f;
  if (this->fuse_next()) {
    Operation *nextOp = getNextOp(this);
    int8_t pos_rshift, pos_m_i8, neg_rshift, neg_m_i8;
    float negativeSlope;
    parseTgLeakyReluParam(nextOp,
        pos_rshift, pos_m_i8, neg_rshift, neg_m_i8, negativeSlope);
    assert(neg_m_i8);

    // TODO: fix the type in backend API
    fused_leakyrelu_pos_rshift = static_cast<int>(pos_rshift);
    fused_leakyrelu_pos_m_i8   = static_cast<int>(pos_m_i8);
    fused_leakyrelu_neg_rshift = static_cast<int>(neg_rshift);
    fused_leakyrelu_neg_m_i8   = static_cast<int>(neg_m_i8);
    fused_negative_slope       = negativeSlope;
    do_relu = true;

    llvm::errs() << "  fused leaky relu, pos ("
        << fused_leakyrelu_pos_m_i8 << ", " << fused_leakyrelu_pos_rshift
        << "), neg ("
        << fused_leakyrelu_neg_m_i8 << ", " << fused_leakyrelu_neg_rshift
        << ")\n";

    // finally, change gaddr to the nextOp's
    ga_output = getOpAddress(nextOp);
  }

  cvi_backend_tg_int8_conv(
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
      GA_INVALID, // bn_mean_data_gaddr,
      GA_INVALID, // bn_variance_data_gaddr,
      GA_INVALID, // scale_gaddr,
      GA_INVALID, // scale_bias_gaddr,
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
      false,     // use_winograd
      oc,        // right_shift_array_len
      ga_pc_info // ga_per_channel
      );

  return success();
}

LogicalResult tpu::TG_BF16_Conv2DOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
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
  gaddr_t ga_bias = GA_INVALID;
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
      GA_INVALID, // ga_bn_mean
      GA_INVALID, // ga_bn_variance
      GA_INVALID, // ga_scale
      GA_INVALID, // ga_scale_bias
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
      GA_INVALID //global_slope_gaddr
      );

  return success();
}

LogicalResult tpu::TG_INT8_PT_DeConv2DOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  //CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  //Operation *op = this->getOperation();

  assert(false);
  return success();
}

LogicalResult tpu::TG_INT8_PC_DeConv2DOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
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

  // check if fused with a leakyrelu
  int fused_leakyrelu_pos_rshift = 0;
  int fused_leakyrelu_pos_m_i8 = 0;
  int fused_leakyrelu_neg_rshift = 0;
  int fused_leakyrelu_neg_m_i8 = 0;
  float negativeSlope;
  if (this->fuse_next()) {
    Operation *nextOp = getNextOp(this);
    int8_t pos_rshift, pos_m_i8, neg_rshift, neg_m_i8;
    parseTgLeakyReluParam(nextOp,
        pos_rshift, pos_m_i8, neg_rshift, neg_m_i8, negativeSlope);
    assert(neg_m_i8);

    // TODO: fix the type in backend API
    fused_leakyrelu_pos_rshift = static_cast<int>(pos_rshift);
    fused_leakyrelu_pos_m_i8   = static_cast<int>(pos_m_i8);
    fused_leakyrelu_neg_rshift = static_cast<int>(neg_rshift);
    fused_leakyrelu_neg_m_i8   = static_cast<int>(neg_m_i8);

    llvm::errs() << "  fused leaky relu, pos ("
        << fused_leakyrelu_pos_m_i8 << ", " << fused_leakyrelu_pos_rshift
        << "), neg ("
        << fused_leakyrelu_neg_m_i8 << ", " << fused_leakyrelu_neg_rshift
        << ")\n";

    // finally, change gaddr to the nextOp's
    ga_output = getOpAddress(nextOp);
  }

  deconv_fixed_forward_bmkernel(
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
      n,
      ic,
      ih,
      iw,
      g, // group,
      oc,
      oh,
      ow,
      kh,
      kw,
      dh,
      dw,
      ph, // pad_h_top,
      ph, // pad_h_bottom,
      pw, // pad_w_left,
      pw, // pad_w_right,
      sh,
      sw,
      with_bias, // do_bias
      false,     // result_add
      do_relu,   // do_activation,
      0,         //right_shift_width,
      false,     //use_winograd,
      oc,        // right_shift_array_len
      ga_pc_info // ga_per_channel
      );

  return success();
}

LogicalResult tpu::TG_BF16_DeConv2DOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  //CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  //Operation *op = this->getOperation();

  assert(false);
  return success();
}

LogicalResult tpu::TG_INT8_EltwiseAddOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  bool do_relu = this->do_relu();

  assert(op->getNumOperands() == 2 && "support 2 inputs only");

  gaddr_t ga_inputs[2];
  ga_inputs[0] = getPreviousOpAddress(op, 0);
  ga_inputs[1] = getPreviousOpAddress(op, 1);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = mlir::getOpLayerId(op);

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

  // TODO: should change on backend API, rather than doing cast
  int rshift_int;
  int m_int[2];
  if (do_quant_rescale) {
    rshift_int = static_cast<int>(rshift);
    m_int[0] = static_cast<int>(m_i8_input[0]);
    m_int[1] = static_cast<int>(m_i8_input[1]);
  }
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
      do_quant_rescale ? rshift_int : 0,   // int right_shift_width,
      do_quant_rescale ? m_int : nullptr,
      coeffs);

  return success();
}

LogicalResult tpu::TG_INT8_EltwiseMaxOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  //CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  //Operation *op = this->getOperation();

  assert(false);
  return success();
}

LogicalResult tpu::TG_INT8_EltwiseMulOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
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
  assert(this->m_i32_output().hasValue());
  int32_t m_i32_output = this->m_i32_output().getValue().getLimitedValue();

  // TODO: should change on backend API, rather than doing cast
  int rshift_int = static_cast<int>(rshift);
  int32_t m_int = static_cast<int32_t>(m_i32_output);
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
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
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
  //CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  //Operation *op = this->getOperation();

  assert(false);
  return success();
}

LogicalResult tpu::TG_BF16_EltwiseMulOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
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

LogicalResult tpu::TG_INT8_FullyConnectedOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
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
  int layer_id = mlir::getOpLayerId(op);

  int8_t rshift_int8 = rshift().getValue().getLimitedValue();
  int rshift = static_cast<int>(rshift_int8);

  bmnet_fc_fixed_forward_bmkernel(
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
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
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
  int layer_id = mlir::getOpLayerId(op);

  bf16_fc_forward_kernel(
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

LogicalResult tpu::TG_INT8_InputOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  //CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  //Operation *op = this->getOperation();

  return success();
}

LogicalResult tpu::TG_BF16_InputOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  //CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  //Operation *op = this->getOperation();

  return success();
}

LogicalResult tpu::TG_INT8_LeakyReluOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  if (this->fuse_prev()) {
    // fused out, do nothing
    return success();
  }
  int8_t pos_rshift, pos_m_i8, neg_rshift, neg_m_i8;
  float negativeSlope;
  parseTgLeakyReluParam(op,
      pos_rshift, pos_m_i8, neg_rshift, neg_m_i8, negativeSlope);
  assert(neg_m_i8);

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = mlir::getOpLayerId(op);

  bmnet_leakyrelu_fixed_forward_bmkernel(
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
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";

  // if (this->fuse_prev()) {
  //   // fused out, do nothing
  //   return success();
  // }

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = mlir::getOpLayerId(op);
  float ga_negative_slope = this->negative_slope().convertToFloat();

  bf16_leakyrelu_forward_kernel(
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

LogicalResult tpu::TG_INT8_LutOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName() << " [" << getOpName()
               << "]\n";
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  gaddr_t y0_table_gaddr = getWeightOpAddress(table()->getDefiningOp());
  int layer_id = mlir::getOpLayerId(op);

  lut_fixed_forward_bmkernel(*backend_ctx,
                             0,        // stream_id,
                             0,        // inst_id,
                             layer_id, // layer_id,
                             nullptr,  // const u32 *depends,
                             0,        // depends_len,
                             input_gaddr, output_gaddr, y0_table_gaddr, n, c, h,
                             w, FMT_I8);

  return success();
}

LogicalResult tpu::TG_BF16_LutOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName() << " [" << getOpName()
               << "]\n";
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

  int layer_id = mlir::getOpLayerId(op);

  bf16_lut_scientific_forward_kernel(*backend_ctx,
                                  0,        // stream_id,
                                  0,        // inst_id,
                                  layer_id, // layer_id,
                                  nullptr,  // const u32 *depends,
                                  0,        // depends_len,
                                  input_gaddr, output_gaddr, table_data_lut,table_data_mantissa_lut,
                                  n, c, h, w, FMT_BF16);

  return success();
}

LogicalResult tpu::TG_INT8_PermuteOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName() << " [" << getOpName()
               << "]\n";
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
  int layer_id = mlir::getOpLayerId(op);
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
  permute_fixed_forward_kernel(
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
  llvm::errs() << "TG_codegen: " << getOperationName() << " [" << getOpName()
               << "]\n";
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
  int layer_id = mlir::getOpLayerId(op);
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

  bf16_permute_fixed_forward_kernel(
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
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
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
      GA_INVALID, // index_data_gaddr,
      GA_INVALID, // o_findex_data_gaddr,
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
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
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
      GA_INVALID, // index_data_gaddr,
      GA_INVALID, // o_findex_data_gaddr,
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
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
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
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
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
  llvm::errs() << "TG_codegen: " << getOperationName() << " [" << getOpName()
               << "]\n";
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
  int layer_id = mlir::getOpLayerId(op);

  assert(this->rshift_pos().hasValue());
  int8_t rshift_pos = this->rshift_pos().getValue().getLimitedValue();
  assert(this->m_i8_pos().hasValue());
  int8_t m_i8_pos = this->m_i8_pos().getValue().getLimitedValue();
  assert(this->rshift_neg().hasValue());
  int8_t rshift_neg = this->rshift_neg().getValue().getLimitedValue();
  bmnet_prelu_fixed_forward_bmkernel(
      *backend_ctx,
      layer_id,             // layer_id,
      ga_input,             // input_data_gaddr,
      ga_output,            // output_data_gaddr,
      negative_scope_gaddr, // float negative_slope,
      n, c, h, w, rshift_pos, m_i8_pos, rshift_neg, FMT_I8);

  return success();
}

LogicalResult tpu::TG_BF16_PReluOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName() << " [" << getOpName()
               << "]\n";
  // CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  // Operation *op = this->getOperation();
  assert(false);
  return success();
}

LogicalResult tpu::TG_INT8_ReluOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName() << " [" << getOpName()
               << "]\n";
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = mlir::getOpLayerId(op);

  bmnet_relu_fixed_forward_bmkernel(
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
      FMT_I8);

  return success();
}

LogicalResult tpu::TG_BF16_ReluOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName() << " [" << getOpName()
               << "]\n";
  // CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  // Operation *op = this->getOperation();
  assert(false);
  return success();
}

LogicalResult tpu::TG_INT8_ShuffleChannelOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName() << " [" << getOpName()
               << "]\n";
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
  int layer_id = mlir::getOpLayerId(op);
  shuffle_channel_forward_kernel(*backend_ctx, 0, 0, layer_id, nullptr, 0,
                                       input_gaddr, output_gaddr, n, c,
                                       frame_size, group, FMT_I8);
  return success();
}

LogicalResult tpu::TG_BF16_ShuffleChannelOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName() << " [" << getOpName()
               << "]\n";
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
  int layer_id = mlir::getOpLayerId(op);
  shuffle_channel_forward_kernel(*backend_ctx, 0, 0, layer_id, nullptr, 0,
                                       input_gaddr, output_gaddr, n, c,
                                       frame_size, group, FMT_BF16);
  return success();
}

LogicalResult tpu::ReshapeOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  //CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  //Operation *op = this->getOperation();

  return success();
}

LogicalResult tpu::TG_INT8_SliceOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  int axis = this->axis().getLimitedValue();
  std::vector<int64_t> input_shape = getTensorShape(input());
  std::vector<int> input_shape_fix;
  for (auto &dim : input_shape) {
    input_shape_fix.push_back((int)dim);
  }

  if (axis == 1 && input_shape[0] == 1) {
    llvm::errs() << "  no copy\n";
    return success();
  }
  int offset = this->offset().getLimitedValue();
  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  int layer_id = mlir::getOpLayerId(op);
  std::vector<int64_t> output_shape = getTensorShape(this->getResult());
  slice_forward_kernel(*backend_ctx, 0, 0, layer_id, nullptr, 0, input_gaddr,
                       output_gaddr, (int)input_shape.size(),
                       input_shape_fix.data(), axis, offset,
                       (int)output_shape[axis], FMT_I8);

  return success();
}

LogicalResult tpu::TG_BF16_SliceOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  int axis = this->axis().getLimitedValue();
  std::vector<int64_t> input_shape = getTensorShape(input());
  std::vector<int> input_shape_fix;
  for (auto &dim : input_shape) {
    input_shape_fix.push_back((int)dim);
  }

  if (axis == 1 && input_shape[0] == 1) {
    llvm::errs() << "  no copy\n";
    return success();
  }
  int offset = this->offset().getLimitedValue();
  gaddr_t input_gaddr = getPreviousOpAddress(op);
  gaddr_t output_gaddr = getOpAddress(op);
  int layer_id = mlir::getOpLayerId(op);
  std::vector<int64_t> output_shape = getTensorShape(this->getResult());
  slice_forward_kernel(*backend_ctx, 0, 0, layer_id, nullptr, 0, input_gaddr,
                       output_gaddr, (int)input_shape.size(),
                       input_shape_fix.data(), axis, offset,
                       (int)output_shape[axis], FMT_BF16);

  return success();
}

LogicalResult tpu::TG_INT8_UpsampleOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
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
      scale
  );

  return success();
}

LogicalResult tpu::TG_BF16_UpsampleOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  int32_t scale = this->scale().getLimitedValue();

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = mlir::getOpLayerId(op);

  bf16_upsample_fixed_bmkernel(
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
      scale
  );

  return success();
}

// MemRefType dummy
LogicalResult tpu::TG_MemRef_INT8_PC_Conv2DOp::codegen(void *ctx) {
    return success();
}

LogicalResult tpu::TG_MemRef_INT8_EltwiseAddOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_FullyConnectedOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_InputOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_PoolAvg2DOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_INT8_PoolMax2DOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_QuantOp::codegen(void *ctx) {
  return success();
}

LogicalResult tpu::TG_MemRef_ReshapeOp::codegen(void *ctx) {
  return success();
}

}
