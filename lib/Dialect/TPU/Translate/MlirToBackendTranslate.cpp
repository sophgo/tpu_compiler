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

static LogicalResult runOperation(Operation &opInst) {
  LLVM_DEBUG(llvm::errs() << "  op " << opInst.getName() << "\n";);

  if (auto tpuTGOp = llvm::dyn_cast<tpu::TpuTGOpCodegenInterface>(opInst)) {
    return tpuTGOp.codegen((void *)backend_ctx);
  }

  if (auto op = dyn_cast<tpu::TL_LA_Conv2DOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "TL_LA_Conv2DOp" << "\n";);

    bool is_dw, with_bias, do_relu;
    int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, ph, pw, dh, dw;
    parseConvParam(op.param(), false, op.input(), op.output(), op.filter(),
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
    parseConvParam(op.param(), false, op.input(), op.output(), op.filter(),
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
  if (auto op = dyn_cast<tpu::FullyConnectedOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "FullyConnectedOp" << "\n";);

    bool with_transpose, with_bias, do_relu;
    int m, k, n;
    getFullyConnectedOpParam(op, with_transpose, m, k, n, with_bias, do_relu);
    parseFullyConnectedParam(op.input(), op.output(), op.filter());

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
#endif

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
    if (op.quant() == "INT8"|| op.quant() == "INT8_PER_CHANNEL"||op.quant() == "INT8_MULTIPLIER"){

    gaddr_t input_gaddr = getPreviousOpAddress(op);
    gaddr_t output_gaddr = op.offset().getValue().getLimitedValue();
    gaddr_t y0_table_gaddr = getWeightOpAddress(op.getOperand(1)->getDefiningOp());

    int layer_id = op.layer_id().getValue().getLimitedValue();
    lut_fixed_forward_bmkernel(*backend_ctx,
                               0,        // stream_id,
                               0,        // inst_id,
                               layer_id, // layer_id,
                               nullptr,  // const u32 *depends,
                               0,        // depends_len,
                               input_gaddr, output_gaddr, y0_table_gaddr, 
                               n, c, h, w, FMT_I8);

    } else if(op.quant() == "BF16") {

      gaddr_t input_gaddr = getPreviousOpAddress(op);
      gaddr_t output_gaddr = op.offset().getValue().getLimitedValue();
      gaddr_t table_data_lut = getWeightOpAddress(op.getOperand(1)->getDefiningOp());
      gaddr_t table_data_mantissa_lut = getWeightOpAddress(op.getOperand(2)->getDefiningOp());


      int layer_id = op.layer_id().getValue().getLimitedValue();
      // TODO: set lut backend
      assert(false);
    }else {
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
    if (op.quant() == "INT8"|| op.quant() == "INT8_PER_CHANNEL"||op.quant() == "INT8_MULTIPLIER"){

    gaddr_t input_gaddr = getPreviousOpAddress(op);
    gaddr_t output_gaddr = op.offset().getValue().getLimitedValue();
    gaddr_t y0_table_gaddr = getWeightOpAddress(op.getOperand(1)->getDefiningOp());

    int layer_id = op.layer_id().getValue().getLimitedValue();
    lut_fixed_forward_bmkernel(*backend_ctx,
                               0,        // stream_id,
                               0,        // inst_id,
                               layer_id, // layer_id,
                               nullptr,  // const u32 *depends,
                               0,        // depends_len,
                               input_gaddr, output_gaddr, y0_table_gaddr, n, c,
                               h, w, FMT_I8);

    }else if(op.quant() == "BF16") {

      gaddr_t input_gaddr = getPreviousOpAddress(op);
      gaddr_t output_gaddr = op.offset().getValue().getLimitedValue();
      gaddr_t table_data_lut = getWeightOpAddress(op.getOperand(1)->getDefiningOp());
      gaddr_t table_data_mantissa_lut = getWeightOpAddress(op.getOperand(2)->getDefiningOp());


      int layer_id = op.layer_id().getValue().getLimitedValue();

      // TODO: set lut backend
      assert(false);
    }else {
      llvm::errs() << "not support yet \n";
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

    lut_interpolation_forward_kernel(*backend_ctx,
                                     0,            // stream_id,
                                     0,            // inst_id,
                                     layer_id,     // layer_id,
                                     nullptr,      // depends
                                     0,            // depends_len
                                     input_gaddr,  // input_data_gaddr,
                                     output_gaddr, // output_data_gaddr,
                                     y0_table_gaddr, slope_gaddr, n, c, h, w,
                                     -8, 8, scale);

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
