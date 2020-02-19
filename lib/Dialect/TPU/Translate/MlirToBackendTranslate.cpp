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

extern int BF16_TABLE_START;
extern int BF16_TABLE_END;

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

// \threshold_x_quantized number should eq \input_nr
static void getI8Multiplier(Operation* opInst,
    float threshold_y,
    int input_nr,
    int* threshold_x_quantized) {

  std::vector<float> threshold_x;
  // determine multiplier and rshift according each threshold_x
  // scale[i] = threshold_x[i] / threshold_y
  // each scale will be implemented by hardware as
  // scale[i] = multiplier / (1 << rshift)
  // find a rshift, that put max(multiplier) into range (64, 127)

  for (int index = 0; index < input_nr; ++index) {
    // get threshold_x
    threshold_x[index] = getPreviousOpThreshold(opInst, index);
  }


  // determine rshift for all inputs, and multiplier for each input
  // use max threshold_x to find rshift first
  float max_threshold_x = *std::max_element(
      std::begin(threshold_x), std::end(threshold_x));
  int8_t rshift = findRShiftAndMultiplierFromQScale(max_threshold_x / threshold_y);
  for (int index = 0; index < input_nr; ++index) {
    float qscale = threshold_x[index] / threshold_y;
    threshold_x_quantized[index] = findMultiplierI8FromQScaleAndRShift(qscale, rshift);
  }
}

static LogicalResult runOperation(Operation &opInst) {
  LLVM_DEBUG(llvm::errs() << "  op " << opInst.getName() << "\n";);

  if (auto tpuTGOp = llvm::dyn_cast<tpu::TpuTGOpCodegenInterface>(opInst)) {
    return tpuTGOp.codegen((void *)backend_ctx);
  }

  if (auto op = dyn_cast<tpu::TL_LA_Conv2DOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "TL_LA_Conv2DOp" << "\n";);

    bool is_dw, with_bias, do_relu;
    int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, ph, pw, dh, dw;
    parseConvParam(op.param(), op.input(), op.output(), op.filter(),
                   n, ic, ih, iw, oc, oh, ow, g,
                   kh, kw, sh, sw, ph, pw, dh, dw, is_dw, with_bias, do_relu);

    gaddr_t ga_input = getPreviousOpAddress(op);
    gaddr_t ga_output = op.offset().getValue().getLimitedValue();
    gaddr_t ga_filter = getWeightOpAddress(op.getOperand(1)->getDefiningOp());
    gaddr_t ga_perchannel = getWeightOpAddress(op.getOperand(2)->getDefiningOp());
    int layer_id = op.layer_id().getValue().getLimitedValue();

    LLVM_DEBUG(llvm::errs() << "TL_LA_Conv2DOp, layer_id = " << layer_id << "\n";);
    cvi_backend_tl_conv_LA(*backend_ctx, layer_id,
        ga_input, ga_output, ga_filter, ga_perchannel,
        n, ic, ih, iw, g, oc, oh, ow, kh, kw,
        dh, dw, ph, ph, pw, pw, sh, sw,
        false, with_bias, do_relu);

    return success();
  }

  if (auto op = dyn_cast<tpu::TL_LW_Conv2DOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "TL_LW_Conv2DOp" << "\n";);

    bool is_dw, with_bias, do_relu;
    int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, ph, pw, dh, dw;
    parseConvParam(op.param(), op.input(), op.output(), op.filter(),
                   n, ic, ih, iw, oc, oh, ow, g,
                   kh, kw, sh, sw, ph, pw, dh, dw, is_dw, with_bias, do_relu);

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
    #if 0
    //
    // V0: Weight Only version, with no parallel for load/store activations
    // (only consider load weight parallel)
    //
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
    #endif
    #if 1
    //
    // V1: Weight and Store version
    //   this only do parallel on both Weight load and Activation Store
    //   but load activation is not handled in parallel
    //
    if (op.tl_store_flag()) {
      cvi_backend_tl_conv_LW(*backend_ctx, layer_id,
          la_input, la_output, la_working,
          ga_filter, ga_perchannel,
          n, ic, ih, iw, g, oc, oh, ow, kh, kw,
          dh, dw, ph, ph, pw, pw, sh, sw,
          false, with_bias, do_relu,
          true, ga_output);
    } else {
      cvi_backend_tl_conv_LW(*backend_ctx, layer_id,
          la_input, la_output, la_working,
          ga_filter, ga_perchannel,
          n, ic, ih, iw, g, oc, oh, ow, kh, kw,
          dh, dw, ph, ph, pw, pw, sh, sw,
          false, with_bias, do_relu);
    }
    #endif
    #if 0
    //
    // V2: Tiling version
    //    make for loops outside of the backend api, handle tiling outside
    //
    // TODO:
    #endif
    return success();
  }
#if 0
  if (auto op = dyn_cast<tpu::ConcatOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "concat ConcatOp" << "\n";);
    int num = op.getOperation()->getNumOperands();
    gaddr_t input_gaddrs[num];

    auto axis = op.dimension().getLimitedValue();
    #define SHAPE_DIM 4
    int output_dim[SHAPE_DIM];
    LLVM_DEBUG(llvm::errs() << "concat num :" << num << "\n";);
    LLVM_DEBUG(llvm::errs() << "concat axis :" << axis << "\n";);
    int32_t input_dims[num * SHAPE_DIM];
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
      input_dims[i] = shape[axis];
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
      int8_t rshift;
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
        multiplier[index] = findMultiplierI8FromQScaleAndRShift(qscale, rshift);
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
           axis, // int concat_axis,
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
          axis, // concat_axis
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
#endif

  if (auto op = dyn_cast<tpu::DeConv2DOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "DeConv2DOp" << "\n";);

    bool with_bias = false, do_relu = false;
    int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, ph, pw, dh, dw;
    getDeConv2DOpParam(op, n, ic, ih, iw, oc, oh, ow, g,
                     kh, kw, sh, sw, ph, pw, dh, dw, with_bias);

    gaddr_t input_gaddr = getPreviousOpAddress(op);
    gaddr_t output_gaddr = op.offset().getValue().getLimitedValue();
    gaddr_t filter_gaddr = getWeightOpAddress(op.getOperand(1)->getDefiningOp());
    gaddr_t bias_gaddr = INVALID_GLOBAL_ADDR;
    if (with_bias) {
      bias_gaddr = getWeightOpAddress(op.getOperand(2)->getDefiningOp());
    }

    int layer_id = op.layer_id().getValue().getLimitedValue();

    if (op.quant() == "INT8") {
      assert(false);
    } else if (op.quant() == "INT8_MULTIPLIER") {
      // assuming padding == 0
      assert(ph == 0);
      assert(pw == 0);
      deconv_fixed_forward_bmkernel(
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
          false, // result_add
          do_relu, // do_activation,
          0, //right_shift_width,
          false, //use_winograd,
          oc, // right_shift_array_len
          bias_gaddr // ga_per_channel
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

    //auto output_size =
    //    std::accumulate(std::begin(o_s), std::end(o_s), 1, std::multiplies<>());

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
          o_s.data(),
          offsets.data(),
          FMT_I8
          );

    } else if (op.quant() == "BF16") {
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
          o_s.data(),
          offsets.data(),
          FMT_BF16
          );
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

  if (auto op = dyn_cast<tpu::PermuteOp>(opInst)) {
    LLVM_DEBUG(LLVM_DEBUG(llvm::errs() << "PermuteOp" << "\n";););

    int i_nchw[] = {1, 1, 1, 1};
    int o_nchw[] = {1, 1, 1, 1};

    auto input_type = op.input()->getType().cast<TensorType>();
    std::vector<int64_t> i_s(input_type.getShape());
    auto output_type = op.output()->getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());

    for (uint64_t i = 0; i < i_s.size(); i++) {
      i_nchw[i] = i_s[i];
    }

    for (uint64_t i = 0; i < o_s.size(); i++) {
      o_nchw[i] = o_s[i];
    }

    // FIXME: check orders.size() != 4
    std::vector<int> orders;

    orders.push_back(op.order0().getLimitedValue());

    orders.push_back(op.order1().getLimitedValue());

    orders.push_back(op.order2().getLimitedValue());

    orders.push_back(op.order3().getLimitedValue());

    gaddr_t input_gaddr = getPreviousOpAddress(op);
    gaddr_t output_gaddr = op.offset().getValue().getLimitedValue();

    int layer_id = op.layer_id().getValue().getLimitedValue();

    int num_axes_ = i_s.size();

    // Check if we need to reorder the data or keep it.
    bool need_permute_ = false;
    for (int i = 0; i < num_axes_; ++i) {
      if (orders[i] != i) {
        // As long as there is one order which is different from the natural order
        // of the data, we need to permute. Otherwise, we share the data and diff.
        need_permute_ = true;
        break;
      }
    }

    if (op.quant() == "INT8") {
      permute_fixed_forward_kernel(
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
    }
    else {
      // if (op.quant() == "BF16") {
      assert(0 && "plz implement it");
    }

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

    if (op.quant() == "INT8" || op.quant() == "INT8_MULTIPLIER") {
      int GT_right_shift_width = static_cast<int>(getWeightFromOperandTensor<float>(opInst, 2)->at(0));
      int GT_scale = static_cast<int>(getWeightFromOperandTensor<float>(opInst, 3)->at(0));
      int LE_right_shift_width = static_cast<int>(getWeightFromOperandTensor<float>(opInst, 4)->at(0));

      LLVM_DEBUG(llvm::errs() <<
          "GT_right_shift_width = " << GT_right_shift_width << "\n"
          "LE_right_shift_width = " << LE_right_shift_width << "\n"
          "GT_scale = " << GT_scale << "\n";);

      bmnet_prelu_fixed_forward_bmkernel(
          *backend_ctx,
          layer_id,             // layer_id,
          input_gaddr,          // input_data_gaddr,
          output_gaddr,         // output_data_gaddr,
          negative_scope_gaddr, // float negative_slope,
          n, c, h, w,
          GT_right_shift_width,
          GT_scale,
          LE_right_shift_width,
          FMT_I8
      );
    } else if (op.quant() == "BF16"){
      LLVM_DEBUG(llvm::errs() <<
          "run PRelu Backend BF16\n";);
      bf16_prelu_forward_kernel(
          *backend_ctx,
          layer_id,
          input_gaddr,
          output_gaddr,
          negative_scope_gaddr,
          n, c, h, w
      );
    } else {
      assert(0 && "UNKNOW Quant Type.");
    }
    return success();
  }

  if (auto op = dyn_cast<tpu::SqrtOp>(opInst)) {
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
    if (op.quant() == "INT8"|| op.quant() == "INT8_PER_CHANNEL"||op.quant() == "INT8_MULTIPLIER"){
      sqrt_fixed_forward_bmkernel(*backend_ctx,
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

  if (auto op = dyn_cast<tpu::DivOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "DivOp(" << op.name() << ")\n";);

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


    if (op.quant() == "INT8"|| op.quant() == "INT8_PER_CHANNEL"||op.quant() == "INT8_MULTIPLIER"){
      reciprocal_fixed_forward_bmkernel(
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
      LLVM_DEBUG(llvm::errs() << "not support yet \n";);
      assert(0);
    }

    return success();
  }

  if (auto op = dyn_cast<tpu::PowerOp>(opInst)) {
    // TODO: fuse relu, power implement by depthwise, it could be fused
    LLVM_DEBUG(llvm::errs() << "PowerOp(" << op.name() << ")\n";);

    float power = op.power().convertToFloat();
    auto input_type = op.x()->getType().cast<TensorType>();
    std::vector<int64_t> i_s(input_type.getShape());
    auto output_type = op.y()->getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());
    assert((i_s == o_s) && "input shape not equal to output shape");
    int nchw[4] = {1, 1, 1, 1};
    for (uint64_t i = 0; i < i_s.size(); i++) {
      nchw[i] = i_s[i];
    }

    gaddr_t input_gaddr = getPreviousOpAddress(op);
    gaddr_t output_gaddr = op.offset().getValue().getLimitedValue();
    gaddr_t scale_offset = getWeightOpAddress(op.getOperand(1)->getDefiningOp());
    gaddr_t shift_offset = getWeightOpAddress(op.getOperand(2)->getDefiningOp());
    int layer_id = op.layer_id().getValue().getLimitedValue();

    float threshold_y,threshold_x,qscale;
    int8_t rshift;
    uint32_t multiplier;
    if (op.quant() != "NONE"){

      threshold_y = op.threshold_y().getValue().convertToFloat();
      threshold_x = getPreviousOpThreshold(op);

      qscale = (threshold_x*threshold_x) /(127.0*threshold_y);
    }

    if (op.quant() == "INT8"|| op.quant() == "INT8_PER_CHANNEL") {
      rshift = findRShiftAndMultiplierFromQScale(qscale);
      multiplier = findMultiplierI8FromQScaleAndRShift(qscale, rshift);
    }else if(op.quant() == "INT8_MULTIPLIER"){
      rshift = (float)findRShiftAndMultiplierFromQScale(qscale, &multiplier, true,255);
    }
    if (op.quant() == "INT8"|| op.quant() == "INT8_PER_CHANNEL") {

      int right_shift_width = (int)rshift;
      int threshold_x_quantized = (int)multiplier;
      LLVM_DEBUG(llvm::errs() << "powerop rshift (" << op.name() << ") is "<< rshift << "\n";);

      llvm::errs() << llvm::format("input_gaddr 0x%lx,output_gaddr 0x%lx, scale_offset 0x%lx, shift_offset 0x%lx\n",input_gaddr, output_gaddr, scale_offset,shift_offset);
      bmnet_power_fixed_forward_bmkernel(
          *backend_ctx, 0, 0, layer_id,
          nullptr, 0,
          input_gaddr, output_gaddr, nchw[0], nchw[1], nchw[2], nchw[3],
          power, scale_offset, shift_offset, right_shift_width, threshold_x_quantized, FMT_I8);
    } else if(op.quant() == "INT8_MULTIPLIER"){
      assert(0 && "not support per channel multiplier power backend api now");
    } else if (op.quant() == "BF16") {
      assert(0 && "not support now");
    }
    return success();
  }


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
    gaddr_t y0_table_gaddr = getWeightOpAddress(op.getOperand(1)->getDefiningOp());
    gaddr_t slope_gaddr = INVALID_GLOBAL_ADDR;

    int layer_id = op.layer_id().getValue().getLimitedValue();
    if (op.quant() == "INT8") {
      sigmoid_fixed_forward_bmkernel(*backend_ctx,
                                     0,        // stream_id,
                                     0,        // inst_id,
                                     layer_id, // layer_id,
                                     nullptr,  // const u32 *depends,
                                     0,        // depends_len,
                                     input_gaddr, output_gaddr, y0_table_gaddr,
                                     slope_gaddr, n, c, h, w, 0, 0, FMT_I8);

    } else if (op.quant() == "BF16"){
      llvm::errs() << BF16_TABLE_START << ",  " << BF16_TABLE_END
                   << "\n";
      slope_gaddr = getWeightOpAddress(op.getOperand(2)->getDefiningOp());
      sigmoid_fixed_forward_bmkernel(*backend_ctx,
                                     0,        // stream_id,
                                     0,        // inst_id,
                                     layer_id, // layer_id,
                                     nullptr,  // const u32 *depends,
                                     0,        // depends_len,
                                     input_gaddr, output_gaddr, y0_table_gaddr,
                                     slope_gaddr, n, c, h, w, BF16_TABLE_START, BF16_TABLE_END, FMT_BF16);
    } else {
      llvm::errs() << op.quant() << "not support yet \n";
      assert(0);
    }
    return success();
  }
  if (auto op = dyn_cast<tpu::ScaleOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "ScaleOp(" << op.name() << ")\n";);

#define SCALE_INPUT_NUM (2)
    int n, c, h, w;
    auto input_1_type = op.x()->getType().cast<TensorType>();
    std::vector<int64_t> i1_s(input_1_type.getShape());
    auto input_2_type = op.scale()->getType().cast<TensorType>();
    std::vector<int64_t> i2_s(input_2_type.getShape());
    auto output_type = op.y()->getType().cast<TensorType>();
    auto second_is_load_weight = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
        op.getOperand(1)->getDefiningOp());
    std::vector<int64_t> o_s(output_type.getShape());
    LLVM_DEBUG(llvm::errs() << "input[1] shape_size is " << i2_s.size() << " "
                            << std::to_string(second_is_load_weight)<< "\n";);
    n = o_s[0];
    c = o_s[1];
    h = (o_s.size() >=4)? o_s[2] : 1;
    w = (o_s.size() >=4)? o_s[3] : 1;

    LLVM_DEBUG(llvm::errs() << "{n, c, h, w} = { "
                            << n << ", " << c << ", " << h << ", " << w << " }\n";);

    gaddr_t ga_inputs = getPreviousOpAddress(op, 0);
    gaddr_t scale_gaddr;
    gaddr_t bias_gaddr = INVALID_GLOBAL_ADDR;
    gaddr_t pack_gaddr = INVALID_GLOBAL_ADDR;
    gaddr_t output_gaddr = op.offset().getValue().getLimitedValue();
    bool do_bias = op.with_bias();
    int layer_id = op.layer_id().getValue().getLimitedValue();
    int scale_dim;
    // TODO: support axis > 0, now
    int inner_dim = h * w;
    // TODO: support variable input[1] shape, currently ONLY verify <n,c,h,w> X <n,c>
    if (second_is_load_weight && op.quant() == "INT8_PER_CHANNEL") {
      // scale from weight
      scale_gaddr = getWeightOpAddress(op.getOperand(1)->getDefiningOp());
      scale_dim = n * c;
      // int8 will pack pack rshift and multipiler no matter if has bias
      bias_gaddr = getWeightOpAddress(op.getOperand(2)->getDefiningOp());
    } else if (!second_is_load_weight && op.quant() == "INT8") {
      // scale from input
      scale_gaddr = getPreviousOpAddress(op, 1);
      scale_dim = n * c;
      // int8 pack rshift and multipiler
      pack_gaddr = getWeightOpAddress(op.getOperand(2)->getDefiningOp());
    } else if (op.quant() == "BF16") {
      if(second_is_load_weight){
        scale_gaddr = getWeightOpAddress(op.getOperand(1)->getDefiningOp());
        if (do_bias) {
          bias_gaddr = getWeightOpAddress(op.getOperand(2)->getDefiningOp());
        }
      }else{
        scale_gaddr = getPreviousOpAddress(op, 1);
      }
      scale_dim = n * c;

    } else{
      assert(0 && "not supprt this condiction");
    }

#define RELU (0)
    bool do_relu = false;
    int activation = RELU;
    float activation_arg[1] = {0.0f};
    if (op.fused_activation_function() == "NONE") {
    } else if (op.fused_activation_function() == "RELU") {
      do_relu = true;
    } else {
      assert(0 && "fused activation mode not support");
    }

    //uint32_t rshift = 0;
    // Per layer
    if (op.quant() == "INT8") {
      if (second_is_load_weight){
        assert(0 && "TODO: perlayer, or you can use perchannel, more better");
      } else {
        scale_fixed_forward_qi32(*backend_ctx, // ctx
                                 0,            // stream_id
                                 0,            // inst_id
                                 layer_id,     // layer_id
                                 nullptr,      // depends
                                 0,            // depends_len
                                 ga_inputs,    // input_addr
                                 scale_gaddr,  // scale_addr
                                 pack_gaddr,   // pack_addr
                                 output_gaddr, // output_addr
                                 n, c, h, w,
                                 scale_dim,      // scale_dim
                                 inner_dim,      // inner_dim
                                 false,          // is_scale_const
                                 0,              // const_scale
                                 do_relu,        // do_activation,
                                 activation,     // activation_method
                                 activation_arg, // activation_arg
                                 do_bias, second_is_load_weight);
      }

    } else if (op.quant() == "INT8_PER_CHANNEL"){
      // Per Channel only when the second input is from weight
      assert(second_is_load_weight &&
             "Per Channel only when the second input is from weight");

      scale_fixed_forward_qi32(*backend_ctx, // ctx
                                0,            // stream_id
                                0,            // inst_id
                                layer_id,     // layer_id
                                nullptr,      // depends
                                0,            // depends_len
                                ga_inputs,    // input_addr
                                scale_gaddr,  // scale_addr
                                bias_gaddr,   // bias_addr
                                output_gaddr, // output_addr
                                n, c, h, w,
                                scale_dim,      // scale_dim
                                inner_dim,      // inner_dim
                                false,          // is_scale_const
                                0,              // const_scale
                                do_relu,        // do_activation,
                                activation,     // activation_method
                                activation_arg, // activation_arg
                                do_bias, second_is_load_weight);

    } else if (op.quant() == "BF16") {
      bf16_scale_forward_kernel(
          *backend_ctx, // ctx
          0,            // stream_id
          0,            // inst_id
          layer_id,     // layer_id
          nullptr,      // depends
          0,            // depends_len
          ga_inputs,    // input_addr
          scale_gaddr,  // scale_addr
          bias_gaddr,   // bias_addr
          output_gaddr, // output_addr
          n, c, h, w,
          scale_dim,      // scale_dim
          inner_dim,      // inner_dim
          false,          // is_scale_const
          0,              // const_scale
          do_relu,        // do_activation,
          activation,     // activation_method
          activation_arg, // activation_arg
          do_bias, second_is_load_weight);

    } else {
      assert(0 && "op quant type not support");
    }
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
