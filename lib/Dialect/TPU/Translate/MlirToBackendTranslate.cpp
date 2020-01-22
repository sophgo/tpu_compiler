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
#include "mlir/Dialect/TPU/QuantizationArithmetic.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
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

#include "backend/backend_tg_api.h"
#include "backend/backend_tl_api.h"

static BM1880v2BackendContext *backend_ctx = nullptr;

template <typename T>
static std::unique_ptr<std::vector<T>>
getWeightFromOperandTensor(Operation &op, int opdIndex) {
  auto weightOp = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
      op.getOperand(opdIndex)->getDefiningOp());
  assert(weightOp);

  auto loadFileOp = llvm::dyn_cast_or_null<tpu::LoadFileOp>(
      weightOp.getOperand()->getDefiningOp());
  assert(loadFileOp);
  auto weightTensorFile = openInputTensorFile(loadFileOp.filename());

  assert(weightOp.name().hasValue());
  auto tensor_name = weightOp.name().getValue();
  auto type = weightOp.getResult()->getType().cast<TensorType>();
  auto weight = weightTensorFile->readTensor<T>(tensor_name, type);

  return weight;
}

static int8_t getRshiftFromOperandTensor(Operation &op, int opdIndex) {
  auto weight = getWeightFromOperandTensor<float>(op, opdIndex);
  return (int8_t)weight->at(0);
}

static LogicalResult runOperation(Operation &opInst) {
  LLVM_DEBUG(llvm::errs() << "  op " << opInst.getName() << "\n";);

  if (auto op = dyn_cast<tpu::TL_LA_Conv2DOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "TL_LA_Conv2DOp" << "\n";);

    bool with_bias, do_relu;
    int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, ph, pw, dh, dw;
    getConv2DOpParam(op, n, ic, ih, iw, oc, oh, ow, g,
        kh, kw, sh, sw, ph, pw, dh, dw, with_bias, do_relu);

    gaddr_t ga_input = getPreviousOpAddress(op);
    gaddr_t ga_output = op.offset().getValue().getLimitedValue();
    gaddr_t ga_filter = getWeightOpAddress(op.getOperand(1)->getDefiningOp());
    gaddr_t ga_perchannel = getWeightOpAddress(op.getOperand(2)->getDefiningOp());
    int layer_id = op.layer_id().getValue().getLimitedValue();

    llvm::errs() << "TL_LA_Conv2DOp, layer_id = " << layer_id << "\n";
    cvi_backend_tl_conv_LA(*backend_ctx, layer_id,
        ga_input, ga_output, ga_filter, ga_perchannel,
        n, ic, ih, iw, g, oc, oh, ow, kh, kw,
        dh, dw, ph, ph, pw, pw, sh, sw,
        false, with_bias, do_relu);

    return success();
  }
  if (auto op = dyn_cast<tpu::TL_LW_Conv2DOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "TL_LW_Conv2DOp" << "\n";);

    bool with_bias, do_relu;
    int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, ph, pw, dh, dw;
    getConv2DOpParam(op, n, ic, ih, iw, oc, oh, ow, g,
        kh, kw, sh, sw, ph, pw, dh, dw, with_bias, do_relu);

    gaddr_t ga_input = getPreviousOpAddress(op);
    gaddr_t ga_output = op.offset().getValue().getLimitedValue();
    gaddr_t ga_filter = getWeightOpAddress(op.getOperand(1)->getDefiningOp());
    gaddr_t ga_perchannel = getWeightOpAddress(op.getOperand(2)->getDefiningOp());
    laddr_t la_input = op.la_input().getLimitedValue();
    laddr_t la_output = op.la_output().getLimitedValue();
    laddr_t la_working = op.la_working().getLimitedValue();
    int layer_id = op.layer_id().getValue().getLimitedValue();

    llvm::errs() << "TL_LW_Conv2DOp, layer_id = " << layer_id << "\n";
    if (op.tl_load_flag()) {
      cvi_backend_tl_load(*backend_ctx, layer_id,
          la_input, ga_input, n, ic, ih, iw);
    }
    cvi_backend_tl_conv_LW(*backend_ctx, layer_id,
        la_input, la_output, la_working,
        ga_filter, ga_perchannel,
        n, ic, ih, iw, g, oc, oh, ow, kh, kw,
        dh, dw, ph, ph, pw, pw, sh, sw,
        false, with_bias, do_relu);
    if (op.tl_store_flag()) {
      cvi_backend_tl_store(*backend_ctx, layer_id,
          la_output, ga_output, n, oc, oh, ow);
    }
    return success();
  }

  if (auto op = dyn_cast<tpu::ConcatOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "concat ConcatOp" << "\n";);
    auto num = op.getOperation()->getNumOperands();
    gaddr_t input_gaddrs[num];
    auto axis = op.dimension();
    int output_dim[4];
    LLVM_DEBUG(llvm::errs() << "concat num :" << num << "\n";);
    LLVM_DEBUG(llvm::errs() << "concat axis :" << axis << "\n";);
    int32_t input_dims[num * 4];
    int output_dim_size;
    std::vector<int64_t> shape = op.res()->getType().cast<TensorType>().getShape();
    output_dim[0] = shape[0];
    output_dim[1] = shape[1];
    output_dim[2] = shape[2];
    output_dim[3] = shape[3];
    output_dim_size = shape.size();

    for ( int i = 0; i < num; i++) {
      int32_t n, c, h, w;
      input_gaddrs[i] = getPreviousOpAddress(op, i);
      std::vector<int64_t> shape =  op.getOperand(i)->getType().cast<TensorType>().getShape();
      n = shape[0];
      c = shape[1];
      h = shape[2];
      w = shape[3];
      input_dims[i] = shape[1];//shape[axis];
      LLVM_DEBUG(llvm::errs() << "shape n:" << n << " c:" << c << " h:"<< h << " w:"<< w <<"\n";);
    }
    gaddr_t output_gaddr = op.offset().getValue().getLimitedValue();

    int layer_id = op.layer_id().getValue().getLimitedValue();
    LLVM_DEBUG(llvm::errs() << "Concat id=" << layer_id << "\n";);
    LLVM_DEBUG(llvm::errs() << "Concat quant=" << op.quant() << "\n";);

    if (op.quant() == "INT8") {

      std::vector<float> threshold_x(num);
      float threshold_y;
      // determine multiplier and rshift according each threshold_x
      // scale[i] = threshold_x[i] / threshold_y
      // each scale will be implemented by hardware as
      // scale[i] = multiplier / (1 << rshift)
      // find a rshift, that put max(multiplier) into range (64, 127)
      int rshift;
      int8_t multiplier[num];

      for (int index = 0; index < num; ++index) {
        // get threshold_x
        threshold_x[index] = getPreviousOpThreshold(op, index);
      }
      // get threshold_y
      threshold_y = op.threshold_y().getValue().convertToFloat();

      // determine rshift for all inputs, and multiplier for each input
      // use max threshold_x to find rshift first
      float max_threshold_x = *std::max_element(
          std::begin(threshold_x), std::end(threshold_x));
      rshift = findRShiftAndMultiplierFromQScale(max_threshold_x / threshold_y);
      for (int index = 0; index < num; ++index) {
        float qscale = threshold_x[index] / threshold_y;
        multiplier[index] = (int8_t)findMultiplierFromQScaleAndRShift(qscale, rshift);
      }

      int threshold_x_quantized[num];
      int rshift_in[num];
      for (int i = 0; i < num; ++i) {
        threshold_x_quantized[i] = (int)multiplier[i];
        rshift_in[i] = rshift;
      }

      bmnet_concat_fixed_forward_bmkernel(
           *backend_ctx,
           0, // u32 stream_id,
           0, //u32 inst_id,
           layer_id, // u32 layer_id,
           nullptr, // const u32 *depends,
           0,// u32 depends_len,
           input_gaddrs, // gaddr_t input_gaddrs[],
           output_gaddr, // gaddr_t output_gaddr,
           input_dims, // int input_dims[],
           num, //int input_num,
           1, // int concat_axis,
           output_dim_size, // int output_dim_size,
           output_dim, // int *output_dim,
           num, // const int need_quantize_num,
           rshift_in, // const int *right_shift_width,
           threshold_x_quantized // const int *threshold_x_quantized
      );

    } else if (op.quant() == "BF16") {

      bf16_concat_fixed_forward_bmkernel(
          *backend_ctx,
          0, // stream_id,
          0, // inst_id,
          layer_id, // layer_id,
          nullptr, // depends
          0, // depends_len
          input_gaddrs, // gaddr_t ga_input[],
          output_gaddr, // gaddr_t ga_output,
          input_dims, // int input_dims[],
          num, // int input_num
          1, // concat_axis
          output_dim_size, //int output_dim_size
          output_dim, //int *output_dim
          0, //int need_quantize_num
          0  // threshold_x_quantized,
      );
    } else {
      llvm::errs() << "not support yet \n";
      assert(0);
    }
    return success();
  }

  if (auto op = dyn_cast<tpu::Conv2DOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "Conv2DOp" << "\n";);

    bool with_bias, do_relu;
    int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, ph, pw, dh, dw;
    getConv2DOpParam<tpu::Conv2DOp>(op, n, ic, ih, iw, oc, oh, ow, g,
                     kh, kw, sh, sw, ph, pw, dh, dw, with_bias, do_relu);

    gaddr_t input_gaddr = getPreviousOpAddress(op);
    gaddr_t output_gaddr = op.offset().getValue().getLimitedValue();
    gaddr_t filter_gaddr = getWeightOpAddress(op.getOperand(1)->getDefiningOp());

    int layer_id = op.layer_id().getValue().getLimitedValue();

    if (op.quant() == "INT8") {

    gaddr_t bias_gaddr = INVALID_GLOBAL_ADDR;
    //int with_bias = 0;
    if (opInst.getNumOperands() > 3) {
      with_bias = 1;
    }
    int rshift_opd_index = 2;
    if (with_bias) {
      bias_gaddr = getWeightOpAddress(op.getOperand(2)->getDefiningOp());
      rshift_opd_index = 3;
    }
    int8_t rshift = getRshiftFromOperandTensor(opInst, rshift_opd_index);

    bmnet_conv_parallel_fixed_forward_bmkernel(
        *backend_ctx,
        0, // stream_id,
        0, // inst_id,
        layer_id, // layer_id,
        nullptr, // depends
        0, // depends_len
        input_gaddr, // input_data_gaddr,
        output_gaddr, // output_data_gaddr,
        filter_gaddr, // weight_data_gaddr,
        bias_gaddr, // bias_data_gaddr,
        INVALID_GLOBAL_ADDR, // bn_mean_data_gaddr,
        INVALID_GLOBAL_ADDR, // bn_variance_data_gaddr,
        INVALID_GLOBAL_ADDR,
        INVALID_GLOBAL_ADDR,
        n,
        ic,
        ih,
        iw,
        g, // group,
        oc,
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
        0, // result_add
        with_bias, // bias_term,
        0, // do_bn,
        0, // do_scale,
        0, // do_scale_bias,
        do_relu ? 1 : 0, // do_activation,
        1.0f, // bn_scale,
        1e-5, // eps,
        0, // param.activation(), method, 0 -> RELU, all others are invalide for now
        nullptr, // activation_arg,
        INVALID_GLOBAL_ADDR, //global_slope_gaddr,
        false, //channel_shared,
        0, //activation_gt_scale,
        0, //activation_gt_rshift,
        0, //activation_le_scale, // slope, TODO
        0, //activation_le_rshift,
        (int)rshift, // right_shift_width,
        0, //bn_right_shift_width,
        0, //scale_right_shift_width,
        false, //use_winograd
        0, // right_shift_array_len
        0 // ga_per_channel
        );

    } else if (op.quant() == "INT8_MULTIPLIER") {

    gaddr_t bias_gaddr = getWeightOpAddress(op.getOperand(2)->getDefiningOp());
    // TODO: assuming always with_bias
    int with_bias = 1;
    bmnet_conv_parallel_fixed_forward_bmkernel(
        *backend_ctx,
        0, // stream_id,
        0, // inst_id,
        layer_id, // layer_id,
        nullptr, // depends
        0, // depends_len
        input_gaddr, // input_data_gaddr,
        output_gaddr, // output_data_gaddr,
        filter_gaddr, // weight_data_gaddr,
        bias_gaddr, // bias_data_gaddr,
        INVALID_GLOBAL_ADDR, // bn_mean_data_gaddr,
        INVALID_GLOBAL_ADDR, // bn_variance_data_gaddr,
        INVALID_GLOBAL_ADDR,
        INVALID_GLOBAL_ADDR,
        n,
        ic,
        ih,
        iw,
        g, // group,
        oc,
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
        0, // result_add
        with_bias, // bias_term,
        0, // do_bn,
        0, // do_scale,
        0, // do_scale_bias,
        do_relu ? 1 : 0, // do_activation,
        1.0f, // bn_scale,
        1e-5, // eps,
        0, // param.activation(), method, 0 -> RELU, all others are invalide for now
        nullptr, // activation_arg,
        INVALID_GLOBAL_ADDR, //global_slope_gaddr,
        false, //channel_shared,
        0, //activation_gt_scale,
        0, //activation_gt_rshift,
        0, //activation_le_scale, // slope, TODO
        0, //activation_le_rshift,
        0, //(int)rshift[0], //right_shift_width,
        0, //bn_right_shift_width,
        0, //scale_right_shift_width,
        false, //use_winograd
        oc, // right_shift_array_len
        bias_gaddr // ga_per_channel
        );
    } else if (op.quant() == "BF16") {

      gaddr_t bias_gaddr = getWeightOpAddress(op.getOperand(2)->getDefiningOp());
      // TODO: assuming always with_bias
      int with_bias = 1;

      bmnet_bf16_conv_forward_kernel(
        *backend_ctx,
        layer_id,  // layer_id
        input_gaddr,
        output_gaddr,
        filter_gaddr,
        bias_gaddr,
        INVALID_GLOBAL_ADDR,  // ga_bn_mean
        INVALID_GLOBAL_ADDR, // ga_bn_variance
        INVALID_GLOBAL_ADDR, // ga_scale
        INVALID_GLOBAL_ADDR, // ga_scale_bias
        n,
        ic,
        ih,
        iw,
        g, // group
        oc,
        kh,
        kw,
        dh,
        dw,
        ph, // pad_h_top
        ph, // pd_h_bottom
        pw, // pad_w_left
        pw, // pad_w_right
        sh,
        sw,
        with_bias,
        0, // do_bn
        0, // do_scale
        0, // do_scale_bias
        do_relu ? 1 : 0,
        1.0f, // bn_scale
        1e-5, // eps
        0, // param.activation(), method, 0 -> RELU, all others are invalid for now
        nullptr, // activation_arg,
        INVALID_GLOBAL_ADDR //global_slope_gaddr
      );

    } else {
      assert(false);
    }

    return success();
  }
  if (auto op = dyn_cast<tpu::CropOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "Cropop" << op.name() 
                            << "\n";);

    int layer_id = op.layer_id().getValue().getLimitedValue();
    // gen cmdbuf
    gaddr_t input_gaddr = getPreviousOpAddress(op, 0);
    gaddr_t output_gaddr = op.offset().getValue().getLimitedValue();

    auto input_1_type = op.input1()->getType().cast<TensorType>();
    std::vector<int> i1_s;
    i1_s.assign(input_1_type.getShape().begin(), input_1_type.getShape().end());
    auto input_2_type = op.input2()->getType().cast<TensorType>();
    std::vector<int> i2_s;
    i2_s.assign(input_2_type.getShape().begin(), input_2_type.getShape().end());

    auto output_type = op.output()->getType().cast<TensorType>();
    std::vector<int> o_s;
    o_s.assign(output_type.getShape().begin(), output_type.getShape().end());

    auto output_size =
        std::accumulate(std::begin(o_s), std::end(o_s), 1, std::multiplies<>());

    std::vector<int> offsets;
    offsets.assign({op.crop_offset_n().getValue().getLimitedValue(),
                    op.crop_offset_c().getValue().getLimitedValue(),
                    op.crop_offset_h().getValue().getLimitedValue(),
                    op.crop_offset_w().getValue().getLimitedValue()});

    if (op.quant() == "INT8") {
      // TODO: wait for backend
      crop_fixed_forward_bmkernel(
          *backend_ctx, // ctx, 
          0, //stream_id
          0, // inst_id
          layer_id,
          nullptr, //depends
          0, //depends_len
          input_gaddr, // bottom_gaddr,
          output_gaddr, //top_gaddr 
          i1_s.data(), 
          i2_s.data(), 
          offsets.data(),
          o_s.size());

    } else if (op.quant() == "BF16") {
      assert(0 && "not support now");
    }
    // gen cmdbuf end

    return success();
  }
  if (auto op = dyn_cast<tpu::Pool2DOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "Pool2DOp" << "\n";);

    bool is_average_pool, do_relu;
    int n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr;
    getPool2DOpParam(op, is_average_pool, n, c, ih, iw, oh, ow,
                     kh, kw, sh, sw, pt, pb, pl, pr, do_relu);

    float threshold_x;
    float threshold_y;
    uint32_t rshift = 0;
    // multiplier is taking avg_const into account
    uint32_t multiplier = 0;
    if (op.quant() == "INT8" && is_average_pool) {
      threshold_x = getPreviousOpThreshold(op);
      threshold_y = op.threshold_y().getValue().convertToFloat();
      // determine multiplier and rshift according to threshold_x
      // scale = threshold_x / threshold_y
      // scale will be implemented by hardware as
      // scale = multiplier / (1 << rshift)
      // find a rshift, that put max(multiplier) into range (64, 127)
      //uint32_t rshift;
      //int8_t multiplier;
      float scale = threshold_x / threshold_y;
      float scale_and_avg_const = scale / (kh * kw);
      //rshift = findRShiftAndMultiplierFromQScale(scale_and_avg_const, &multiplier, false, 127);
      rshift = findRShiftAndMultiplierFromQScale(scale_and_avg_const, &multiplier, false, 255);
    }

    // gen cmdbuf
    gaddr_t input_gaddr = getPreviousOpAddress(op);
    gaddr_t output_gaddr = op.offset().getValue().getLimitedValue();
    int layer_id = op.layer_id().getValue().getLimitedValue();

    if (op.quant() == "INT8") {
      int threshold_x_quantized = multiplier;
      bmnet_pooling_fixed_forward_bmkernel(
          *backend_ctx,
          0, // stream_id,
          0, // inst_id,
          layer_id, // layer_id,
          nullptr, // depends
          0, // depends_len
          input_gaddr, // input_data_gaddr,
          output_gaddr, // output_data_gaddr,
          INVALID_GLOBAL_ADDR, // index_data_gaddr,
          INVALID_GLOBAL_ADDR, // o_findex_data_gaddr,
          n,
          c,
          ih,
          iw,
          kh,
          kw,
          pt, // int pad_top,
          pb, // int pad_bot,
          pl, // int pad_left,
          pr, // int pad_right,
          sh, // int stride_h,
          sw, // int stride_w,
          is_average_pool, //is_avg_pooling,
          0.0f, // float avg_const,  // default(passing 0.0f) is 1/kh*kw
          0, // int do_relu,
          is_average_pool ? rshift : 0, //int right_shift_width,
          is_average_pool ? &threshold_x_quantized : nullptr, // &threshold_x_quantized,
          true);
    }
    else if (op.quant() == "BF16") {
      bf16_pooling_forward_kernel(
          *backend_ctx,
          layer_id, // layer_id,
          input_gaddr, // input_data_gaddr,
          output_gaddr, // output_data_gaddr,
          INVALID_GLOBAL_ADDR, // index_data_gaddr,
          INVALID_GLOBAL_ADDR, // o_findex_data_gaddr,
          n,
          c,
          ih,
          iw,
          kh,
          kw,
          pt, // int pad_top,
          pb, // int pad_bot,
          pl, // int pad_left,
          pr, // int pad_right,
          sh, // int stride_h,
          sw, // int stride_w,
          is_average_pool, //is_avg_pooling,
          0.0f, // float avg_const,  // default(passing 0.0f) is 1/kh*kw
          0, // int do_relu,
          true);
    } else {
      llvm::errs() << "op.quant = " << op.quant();
      assert(0);
    }
    // gen cmdbuf end

    return success();
  }

  if (auto op = dyn_cast<tpu::FullyConnectedOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "FullyConnectedOp" << "\n";);

    bool with_transpose, with_bias, do_relu;
    int m, k, n;
    getFullyConnectedOpParam(op, with_transpose, m, k, n, with_bias, do_relu);
    assert(with_transpose == false);

    gaddr_t input_gaddr = getPreviousOpAddress(op);
    gaddr_t output_gaddr = op.offset().getValue().getLimitedValue();
    gaddr_t filter_gaddr = getWeightOpAddress(op.getOperand(1)->getDefiningOp());
    gaddr_t bias_gaddr = INVALID_GLOBAL_ADDR;
    //int with_bias = 0;
    if (opInst.getNumOperands() > 2) {
      with_bias = 1;
      bias_gaddr = getWeightOpAddress(op.getOperand(2)->getDefiningOp());
    }


    int layer_id = op.layer_id().getValue().getLimitedValue();

    if (op.quant() == "INT8") {

      int rshift_opd_index = 2;
      if (with_bias) {
        rshift_opd_index = 3;
      }
      int8_t rshift = getRshiftFromOperandTensor(opInst, rshift_opd_index);

      bmnet_fc_fixed_forward_bmkernel(
          *backend_ctx,
          0, // stream_id,
          0, // inst_id,
          layer_id, // layer_id,
          nullptr, // depends
          0, // depends_len
          input_gaddr, // input_data_gaddr,
          filter_gaddr, // weight_data_gaddr,
          bias_gaddr, // bias_data_gaddr,
          output_gaddr, // output_data_gaddr,
          m, // int in_row,
          k, // int in_col,
          n, // int out_col,
          with_bias, // int have_bias,
          do_relu ? 1 : 0, // do_activation,
          0, // activation_method,
          INVALID_GLOBAL_ADDR, // activation_ga_slope,
          0, // int activation_channel_shared,
          0, // int activation_gt_scale,
          0, // int activation_gt_rshift,
          0, // int activation_le_scale,
          0, // int activation_le_rshift,
          false, // weight_tp,
          3, // int left_shift_width, // #define DEFAULT_FC_LEFT_SHIFT 3
          (int)rshift, // rshift
          0, //int threshold_x_quantized_len,
          nullptr, //const int *threshold_x_quantized,
          nullptr //const int *right_shift_array
          );

    } else if (op.quant() == "BF16") {
      // Note:
      //  1880v2 tdma does not support transposed matrix load
      //  Weight tranpose must be handled before backend

      bf16_fc_forward_kernel(
        *backend_ctx,
        layer_id, // layer_id
        input_gaddr, // input_data_gaddr
        filter_gaddr, // weight_data_gaddr
        bias_gaddr, // bias_data_gaddr
        output_gaddr, // output_data_gaddr
        m, // int in_row
        k, // int in_col
        n, // in out_col,
        with_bias, // has_bias
        do_relu ? 1 : 0, // do_activation
        0  // activation_method
      );
    } else {
      assert(0);
    }

    return success();
  }
  if (auto op = dyn_cast<tpu::ReluOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "ReluOp" << "\n";);
    assert(0 && "consider fuse relu first");

    int n, c, h, w;
    float negative_slope = op.negative_slope().convertToFloat();
    LLVM_DEBUG(llvm::errs() << "  negative_slope " << negative_slope << "\n";);
    auto input_type = op.x()->getType().cast<TensorType>();
    std::vector<int64_t> i_s(input_type.getShape());
    auto output_type = op.y()->getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());
    assert((i_s == o_s) && "input shape not equal to output shape");
    n = i_s[0];
    c = i_s[1];
    h = i_s[2];
    w = i_s[3];

    gaddr_t input_gaddr = getPreviousOpAddress(op);
    gaddr_t output_gaddr = op.offset().getValue().getLimitedValue();

    int layer_id = op.layer_id().getValue().getLimitedValue();

    bmnet_relu_fixed_forward_bmkernel(
        *backend_ctx,
        0, // stream_id,
        0, // inst_id,
        layer_id, // layer_id,
        nullptr, // depends
        0, // depends_len
        input_gaddr, // input_data_gaddr,
        output_gaddr, // output_data_gaddr,
        0.0f, // float negative_slope,
        n,
        c,
        h,
        w,
        0, // int threshold_x_quantized_len,
        nullptr, // const int *threshold_x_quantized,
        nullptr, //const int *right_shift_array
        FMT_I8
        );

    return success();
  }
  if (auto op = dyn_cast<tpu::PReluOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "PReluOp"
                            << "\n";);

    int n, c, h, w;
    auto input_type = op.x()->getType().cast<TensorType>();
    std::vector<int64_t> i_s(input_type.getShape());
    auto output_type = op.y()->getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());
    assert((i_s == o_s) && "input shape not equal to output shape");
    n = i_s[0];
    c = i_s[1];
    h = i_s[2];
    w = i_s[3];
    gaddr_t negative_scope_gaddr = getWeightOpAddress(op.getOperand(1)->getDefiningOp());
    gaddr_t input_gaddr = getPreviousOpAddress(op);
    gaddr_t output_gaddr = op.offset().getValue().getLimitedValue();

    int layer_id = op.layer_id().getValue().getLimitedValue();

    bmnet_prelu_fixed_forward_bmkernel(
        *backend_ctx,
        layer_id,             // layer_id,
        input_gaddr,          // input_data_gaddr,
        output_gaddr,         // output_data_gaddr,
        negative_scope_gaddr, // float negative_slope,
        n, c, h, w,
        0,       // int threshold_x_quantized_len,
        nullptr, // const int *threshold_x_quantized,
        nullptr  // const int *right_shift_array
    );

    return success();
  }
  if (auto op = dyn_cast<tpu::EltwiseOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "EltwiseOp" << "\n";);

#define MAX_ELTWISE_INPUT (2)
    int n, c, h, w;
    auto input_1_type = op.x1()->getType().cast<TensorType>();
    std::vector<int64_t> i1_s(input_1_type.getShape());
    auto input_2_type = op.x2()->getType().cast<TensorType>();
    std::vector<int64_t> i2_s(input_2_type.getShape());
    auto output_type = op.y()->getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());
    assert((i1_s == i2_s) && "two input shapes not equal");
    assert((i1_s == o_s) && "input shape not equal to output shape");
    n = o_s[0];
    c = o_s[1];
    h = o_s[2];
    w = o_s[3];

    bool do_relu = false;
    if (op.fused_activation_function() == "NONE") {
    } else if (op.fused_activation_function() == "RELU") {
      do_relu = true;
    } else {
      assert(0);
    }


    gaddr_t ga_inputs[2];
    ga_inputs[0] = getPreviousOpAddress(op, 0);
    ga_inputs[1] = getPreviousOpAddress(op, 1);
    gaddr_t output_gaddr = op.offset().getValue().getLimitedValue();

    int layer_id = op.layer_id().getValue().getLimitedValue();

    if (op.quant() == "INT8") {
      std::vector<float> threshold_x(MAX_ELTWISE_INPUT);
      float threshold_y;
      // determine multiplier and rshift according each threshold_x
      // scale[i] = threshold_x[i] / threshold_y
      // each scale will be implemented by hardware as
      // scale[i] = multiplier / (1 << rshift)
      // find a rshift, that put max(multiplier) into range (64, 127)
      uint32_t rshift;
      uint32_t multiplier_prod;
      for (int index = 0; index < MAX_ELTWISE_INPUT; ++index) {
        // get threshold_x
        threshold_x[index] = getPreviousOpThreshold(op, index);
      }
      // get threshold_y
      threshold_y = op.threshold_y().getValue().convertToFloat();
      if (op.method() == "SUM") {

        const int nInputs = opInst.getNumOperands();
        assert((nInputs - 2) == MAX_ELTWISE_INPUT &&
               "Elt sum only support two inputs now");

        int8_t rshift_opd_index = MAX_ELTWISE_INPUT;
        int8_t rshift = getRshiftFromOperandTensor(opInst, rshift_opd_index);

        int8_t multiplier_opd_index = MAX_ELTWISE_INPUT + 1;
        auto fmultiplier =
            getWeightFromOperandTensor<float>(opInst, multiplier_opd_index);

        int multiplier_int8[MAX_ELTWISE_INPUT];
        for (size_t i = 0; i < fmultiplier.get()->size(); ++i) {
          multiplier_int8[i] = static_cast<int8_t>(fmultiplier->at(i));
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
            output_gaddr, // gaddr_t ga_output,
            2,            // int input_size,
            1,            // int op,  0, prod, 1, sum, 2, max
            n, c, h, w,
            do_relu, // bool do_relu,
            0.0f,    // float relu_slope,
            rshift,  // int right_shift_width,
            multiplier_int8, coeffs);
      } else if (op.method() == "PROD") {
        float threshold_prod = std::accumulate(
            threshold_x.begin(), threshold_x.end(), 1.0, std::multiplies<>());
        float qscale = threshold_prod / threshold_y / 127.0;
        rshift = findRShiftAndMultiplierFromQScale(qscale, &multiplier_prod,
                                                   true, 255);
        int threshold_x_quantized[MAX_ELTWISE_INPUT];

        for (int i = 0; i < MAX_ELTWISE_INPUT; ++i) {
          threshold_x_quantized[i] = multiplier_prod;
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
            output_gaddr, // gaddr_t ga_output,
            2,            // int input_size,
            0,            // int op,  0, prod, 1, sum, 2, max
            n, c, h, w,
            do_relu, // bool do_relu,
            0.0f,    // float relu_slope,
            rshift,  // int right_shift_width,
            threshold_x_quantized, coeffs);
      } else {
        assert("not support");
      }
    } else if (op.quant() == "BF16") {
      if (op.method() == "SUM") {
        const float coeffs[2] = {1.0, 1.0};

        bf16_eltwise_forward_kernel(*backend_ctx,
                                    layer_id,     // layer_id
                                    ga_inputs,    // gaddr_t ga_input[]
                                    output_gaddr, // gaddr_t ga_output
                                    2,            // int input_size
                                    1, // int op, 0: prod, 1: sum, 2: max
                                    n, c, h, w,
                                    do_relu, // bool do_relu
                                    0.0f,    // float relu_slope
                                    coeffs);
      }
    }

    // gen cmd end

    return success();
  }
/*  if (auto op = dyn_cast<tpu::SqrtOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "SqrtOp(" << op.name() << ")\n";);

    int n, c, h, w;
    auto input_type = op.input()->getType().cast<TensorType>();
    std::vector<int64_t> i_s(input_type.getShape());
    auto output_type = op.output()->getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());
    assert((i_s == o_s) && "input shape not equal to output shape");
    n = i_s[0];
    c = i_s[1];
    h = i_s[2];
    w = i_s[3];

    gaddr_t input_gaddr = getPreviousOpAddress(op);
    gaddr_t output_gaddr = op.offset().getValue().getLimitedValue();
    gaddr_t y0_table_gaddr = getWeightOpAddress(op.getOperand(1)->getDefiningOp());

    int layer_id = op.layer_id().getValue().getLimitedValue();

    if (op.quant() == "INT8") {
      sqrt_fixed_forward_bmkernel(
          *backend_ctx,
          0, //stream_id,
          0, //inst_id,
          layer_id, //layer_id,
          nullptr, //const u32 *depends,
          0, //depends_len,
          input_gaddr,
          output_gaddr,
          y0_table_gaddr,
          n,
          c,
          h,
          w);
    }
    else {
      llvm::errs() << "not support yet \n";
      assert(0);
    }

    return success();
  }  */

  if (auto op = dyn_cast<tpu::TanHOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "TanHOp" << "\n";);

    int n, c, h, w;
    float scale = op.scale().convertToFloat();
    LLVM_DEBUG(llvm::errs() << "  its scale " << scale << "\n";);
    auto input_type = op.x()->getType().cast<TensorType>();
    std::vector<int64_t> i_s(input_type.getShape());
    auto output_type = op.y()->getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());
    assert((i_s == o_s) && "input shape not equal to output shape");
    n = i_s[0];
    c = i_s[1];
    h = i_s[2];
    w = i_s[3];

    gaddr_t input_gaddr = getPreviousOpAddress(op);
    gaddr_t output_gaddr = op.offset().getValue().getLimitedValue();
    gaddr_t y0_table_gaddr = getWeightOpAddress(op.getOperand(1)->getDefiningOp());
    gaddr_t slope_gaddr = getWeightOpAddress(op.getOperand(2)->getDefiningOp());

    int layer_id = op.layer_id().getValue().getLimitedValue();

    bf16_tanh_forward_kernel(
        *backend_ctx,
        0, // stream_id,
        0, // inst_id,
        layer_id, // layer_id,
        nullptr, // depends
        0, // depends_len
        input_gaddr, // input_data_gaddr,
        output_gaddr, // output_data_gaddr,
        y0_table_gaddr,
        slope_gaddr,
        n,
        c,
        h,
        w,
        scale
        );

    return success();
  }
  if (auto op = dyn_cast<tpu::SigmoidOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "SigmoidOp"
                            << "\n";);
    int n, c, h, w;
    auto input_type = op.input()->getType().cast<TensorType>();
    std::vector<int64_t> i_s(input_type.getShape());
    auto output_type = op.output()->getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());
    assert((i_s == o_s) && "input shape not equal to output shape");
    n = i_s[0];
    c = i_s[1];
    h = i_s[2];
    w = i_s[3];
    gaddr_t input_gaddr = getPreviousOpAddress(op);
    gaddr_t output_gaddr = op.offset().getValue().getLimitedValue();
    gaddr_t y0_table_gaddr =
        getWeightOpAddress(op.getOperand(1)->getDefiningOp());
    int layer_id = op.layer_id().getValue().getLimitedValue();
    if (op.quant() == "INT8") {
      sigmoid_fixed_forward_bmkernel(*backend_ctx,
                                     0,        // stream_id,
                                     0,        // inst_id,
                                     layer_id, // layer_id,
                                     nullptr,  // const u32 *depends,
                                     0,        // depends_len,
                                     input_gaddr, output_gaddr, y0_table_gaddr,
                                     n, c, h, w);

    } else {
      llvm::errs() << "not support yet \n";
      assert(0);
    }
    return success();
  }
  return success();
}

static LogicalResult runBlock(Block &bb) {
  // Traverse operations.
  for (auto &op : bb) {
    if (failed(runOperation(op)))
      return failure();
  }

  return success();
}

static LogicalResult runOneFunction(FuncOp func) {
  LLVM_DEBUG(llvm::errs() << "func " << func.getName() << "\n";);

  // Then, run blocks one by one.
  for (Block &bb : func.getBlocks()) {
    if (failed(runBlock(bb)))
      return failure();
  }

  return success();
}

LogicalResult translateModule(ModuleOp module, llvm::raw_ostream &output) {
  if (!module)
    return failure();

  std::vector<int8_t> weight_data;
  backend_ctx = bmnet_create_backend_context(weight_data);

  for (FuncOp function : module.getOps<FuncOp>()) {
    LLVM_DEBUG(llvm::errs() << "run " << function.getName() << "\n";);

    if (!function.getName().equals("tpu_func")) {
      //continue;
      assert(0);
    }
    if (failed(runOneFunction(function)))
      return failure();
  }

  bmnet_submit(backend_ctx);
  std::vector<uint8_t> cmdbuf;
  bmnet_read_cmdbuf(backend_ctx, cmdbuf);

  output.write(reinterpret_cast<char *>(cmdbuf.data()), cmdbuf.size());

  return success();
}

static TranslateFromMLIRRegistration
    registration("mlir-to-cmdbuf",
                 [](ModuleOp module, llvm::raw_ostream &output) {
                   return translateModule(module, output);
                 });
