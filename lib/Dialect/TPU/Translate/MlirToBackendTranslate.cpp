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
#include "llvm/Support/Debug.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/TensorFile.h"

#include <fstream>

#define DEBUG_TYPE "mlir-to-cmdbuf"

using namespace mlir;

#include "backend/backend_tg_api.h"
static BM1880v2BackendContext *backend_ctx = nullptr;

#define calcConv2DSpatialOutput(_i_, _k_, _s_, _p_, _d_) \
    (((_i_) + 2 * (_p_) - (_d_) * ((_k_) - 1) - 1) / (_s_) + 1)

static int64_t findPadForSamePadding(int64_t i, int64_t o, int64_t k, int64_t s, int64_t d) {
  //llvm::errs() << "i: " << i << ", o: " << o << ", k: " << k << ", s: " << s << ", d: " << d << "\n";
  if (k == 1) {
    return 0;
  }
  for (int64_t p = 1; p <= k - 1; ++p) {
    if (calcConv2DSpatialOutput(i, k, s, p, d) == o) {
      return p;
    }
  }
  assert(false);
  return 0;
}

static int8_t getRshiftFromOperandTensor(Operation &op, int opdIndex) {
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
  auto weight = weightTensorFile->readTensor<float>(tensor_name, type);

  return (int8_t)weight->at(0);
}

static LogicalResult runOperation(Operation &opInst) {
  LLVM_DEBUG(llvm::errs() << "  op " << opInst.getName() << "\n";);

  if (auto op = dyn_cast<tpu::Conv2DOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "Conv2DOp" << "\n";);

    int n, ic, ih, iw, oc, oh, ow, kh, kw, sh, sw, ph, pw, dh, dw;
    dh = op.dilation_h_factor().getLimitedValue();  // APInt, use .getLimitedValue(); to get uint65_t
    dw = op.dilation_w_factor().getLimitedValue();
    sh = op.stride_h().getLimitedValue();
    sw = op.stride_w().getLimitedValue();
    auto input_type = op.input()->getType().cast<TensorType>();
    std::vector<int64_t> i_s(input_type.getShape());
    auto output_type = op.output()->getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());
    auto filter_type = op.filter()->getType().cast<TensorType>();
    std::vector<int64_t> f_s(filter_type.getShape());
    assert((i_s[0] == o_s[0]) && "input N not equal to output N");
    n = i_s[0];
    ih = i_s[2];
    iw = i_s[3];
    oc = f_s[0];
    ic = f_s[1];
    kh = f_s[2];
    kw = f_s[3];
    oh = o_s[2];
    ow = o_s[3];
    if (op.padding() == "SAME") {
      ph = findPadForSamePadding(ih, oh, kh, sh, dh);
      pw = findPadForSamePadding(iw, ow, kw, sw, dw);
    } else if (op.padding() == "VALID") {
      ph = 0;
      pw = 0;
    } else {
      assert(false);
    }
    bool do_relu = false;
    if (op.fused_activation_function() == "NONE") {
    } else if (op.fused_activation_function() == "RELU") {
      do_relu = true;
    } else {
      assert(0);
    }

    gaddr_t input_gaddr = getPreviousOpAddress(op);
    gaddr_t output_gaddr = op.offset().getValue().getLimitedValue();
    gaddr_t filter_gaddr = getWeightOpAddress(op.getOperand(1)->getDefiningOp());

    if (op.quant() == "INT8") {

    gaddr_t bias_gaddr = INVALID_GLOBAL_ADDR;
    int with_bias = 0;
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
        0, // layer_id,
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
        1, // group,
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
        0, //int threshold_x_quantized_len,
        nullptr, //const int *threshold_x_quantized,
        nullptr //const int *right_shift_array
        );

    } else if (op.quant() == "INT8_MULTIPLIER") {

    gaddr_t bias_gaddr = getWeightOpAddress(op.getOperand(2)->getDefiningOp());
    // TODO: assuming always with_bias
    int with_bias = 1;

    bmnet_conv_parallel_fixed_forward_bmkernel_qdm(
        *backend_ctx,
        0, // stream_id,
        0, // inst_id,
        0, // layer_id,
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
        1, // group,
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
        0, //int threshold_x_quantized_len,
        nullptr, //const int *threshold_x_quantized,
        nullptr //const int *right_shift_array
        );

    } else {
      assert(false);
    }

    return success();
  }
  if (auto op = dyn_cast<tpu::Pool2DOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "Pool2DOp" << "\n";);
    auto pool_method = op.getAttrOfType<StringAttr>("pool");

    bool is_average_pool;
    if (pool_method.getValue() == "AVE") {
      is_average_pool = true;
    } else if (pool_method.getValue() == "MAX") {
      is_average_pool = false;
    } else {
      assert(false);
    }
    int n, c, ih, iw, oh, ow, kh, kw, sh, sw, ph, pw;
    kh = op.filter_height().getLimitedValue();
    kw = op.filter_width().getLimitedValue();
    sh = op.stride_h().getLimitedValue();
    sw = op.stride_w().getLimitedValue();
    auto input_type = op.input()->getType().cast<TensorType>();
    std::vector<int64_t> i_s(input_type.getShape());
    auto output_type = op.output()->getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());
    assert((i_s[0] == o_s[0]) && "input N not equal to output N");
    assert((i_s[1] == o_s[1]) && "input C not equal to output C");
    n = i_s[0];
    c = i_s[1];
    ih = i_s[2];
    iw = i_s[3];
    oh = o_s[2];
    ow = o_s[3];
    auto padding_attr = op.getAttrOfType<StringAttr>("padding");
    if (padding_attr.getValue() == "SAME") {
      ph = findPadForSamePadding(ih, oh, kh, sh, 1);
      pw = findPadForSamePadding(iw, ow, kw, sw, 1);
    } else if (padding_attr.getValue() == "VALID") {
      ph = 0;
      pw = 0;
    } else {
      assert(false);
    }

    // for INT8, get threshold_x and make copy of input first
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

    int threshold_x_quantized = multiplier;
    bmnet_pooling_fixed_forward_bmkernel(
        *backend_ctx,
        0, // stream_id,
        0, // inst_id,
        0, // layer_id,
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
        ph, // int pad_top,
        ph, // int pad_bot,
        pw, // int pad_left,
        pw, // int pad_right,
        sh, // int stride_h,
        sw, // int stride_w,
        is_average_pool, //is_avg_pooling,
        0.0f, // float avg_const,  // default(passing 0.0f) is 1/kh*kw
        0, // int do_relu,
        is_average_pool ? rshift : 0, //int right_shift_width,
        is_average_pool ? &threshold_x_quantized : nullptr, // &threshold_x_quantized,
        true);
    // gen cmdbuf end

    return success();
  }
  if (auto op = dyn_cast<tpu::FullyConnectedOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "FullyConnectedOp" << "\n";);

    int m, k, n;
    bool transpose = false;
    auto input_type = op.input()->getType().cast<TensorType>();
    std::vector<int64_t> i_s(input_type.getShape());
    auto output_type = op.output()->getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());
    auto filter_type = op.filter()->getType().cast<TensorType>();
    std::vector<int64_t> f_s(filter_type.getShape());
    assert((i_s[0] == o_s[0]) && "input M not equal to output M");
    m = i_s[0];
    // assuming transpose is false
    assert(transpose == false);
    assert((i_s[1] == f_s[1]) && "input K not equal to filter K");
    k = i_s[1];
    assert((f_s[0] == o_s[1]) && "filter N not equal to output N");
    n = o_s[1];

    gaddr_t input_gaddr = getPreviousOpAddress(op);
    gaddr_t output_gaddr = op.offset().getValue().getLimitedValue();
    gaddr_t filter_gaddr = getWeightOpAddress(op.getOperand(1)->getDefiningOp());
    gaddr_t bias_gaddr = INVALID_GLOBAL_ADDR;
    int with_bias = 0;
    if (op.quant() == "INT8") {
      if (opInst.getNumOperands() > 3) {
        with_bias = 1;
      }
    } else {
      assert(0);
    }
    int rshift_opd_index = 2;
    if (with_bias) {
      bias_gaddr = getWeightOpAddress(op.getOperand(2)->getDefiningOp());
      rshift_opd_index = 3;
    }
    int8_t rshift = getRshiftFromOperandTensor(opInst, rshift_opd_index);

    bmnet_fc_fixed_forward_bmkernel(
        *backend_ctx,
        0, // stream_id,
        0, // inst_id,
        0, // layer_id,
        nullptr, // depends
        0, // depends_len
        input_gaddr, // input_data_gaddr,
        filter_gaddr, // weight_data_gaddr,
        bias_gaddr, // bias_data_gaddr,
        output_gaddr, // output_data_gaddr,
        m, // int in_row,
        k, // int in_col,
        n, // int out_col,
        1, // int have_bias,
        0, // do_activation,
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

    bmnet_relu_fixed_forward_bmkernel(
        *backend_ctx,
        0, // stream_id,
        0, // inst_id,
        0, // layer_id,
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
        nullptr //const int *right_shift_array
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

    std::vector<float> threshold_x(MAX_ELTWISE_INPUT);
    float threshold_y;
    // determine multiplier and rshift according each threshold_x
    // scale[i] = threshold_x[i] / threshold_y
    // each scale will be implemented by hardware as
    // scale[i] = multiplier / (1 << rshift)
    // find a rshift, that put max(multiplier) into range (64, 127)
    uint32_t rshift;
    int8_t multiplier[MAX_ELTWISE_INPUT];
    if (op.quant() == "INT8") {
      for (int index = 0; index < MAX_ELTWISE_INPUT; ++index) {
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
      for (int index = 0; index < 2; ++index) {
        float qscale = threshold_x[index] / threshold_y;
        multiplier[index] = (int8_t)findMultiplierFromQScaleAndRShift(qscale, rshift);
      }
    }
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

    int threshold_x_quantized[MAX_ELTWISE_INPUT];
    for (int i = 0; i < MAX_ELTWISE_INPUT; ++i) {
      threshold_x_quantized[i] = (int)multiplier[i];
    }
    const int coeffs[2] = {1, 1};
    bmnet_eltwise_fixed_forward_bmkernel(
        *backend_ctx,
        0, // stream_id,
        0, // inst_id,
        0, // layer_id,
        nullptr, // depends
        0, // depends_len
        ga_inputs, // gaddr_t ga_input[],
        output_gaddr, // gaddr_t ga_output,
        2, // int input_size,
        1, // int op,  0, prod, 1, sum, 2, max
        n,
        c,
        h,
        w,
        do_relu, // bool do_relu,
        0.0f, // float relu_slope,
        rshift, //int right_shift_width,
        threshold_x_quantized,
        coeffs);
    // gen cmd end

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

LogicalResult translateModule(ModuleOp module, StringRef outputFilename) {
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

  auto file = openOutputFile(outputFilename);
  if (!file)
    return failure();

  file->os().write(reinterpret_cast<char *>(cmdbuf.data()),
                   cmdbuf.size());
  file->keep();

  return success();
}

static TranslateFromMLIRRegistration
    registration("mlir-to-cmdbuf",
                 [](ModuleOp module, StringRef outputFilename) {
                   return translateModule(module, outputFilename);
                 });
