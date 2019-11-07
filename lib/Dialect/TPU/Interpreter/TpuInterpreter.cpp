//===- TpuInterpreter.cpp - Implementation of TPU Op Interpreter ---------===//
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
// This file implements the TPU dialect Interpreter.
//
//===----------------------------------------------------------------------===//


#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/Interpreter.h"
#include "mlir/Dialect/TPU/QuantizationArithmetic.h"
#include "mlir/Dialect/TPU/NativeCpuImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/TensorFile.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/CommandLine.h"

#include <numeric>
#include <functional>

#define DEBUG_TYPE "interpreter"

//#define QUANT_DEQUANT_EVERY_LAYER
#define ENABLE_GEN_CMDBUF

using namespace std;

static llvm::cl::OptionCategory clOptionsCategory("interpreter options");

static llvm::cl::opt<std::string> clAllTensorFilename(
    "dump-all-tensor",
    llvm::cl::desc("dump all tensor into a npz file"),
    llvm::cl::init("-"),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<std::string> clCmdBufFilename(
    "generate-cmdbuf",
    llvm::cl::desc("generate cmdbuf and save into a bin file"),
    llvm::cl::init("-"),
    llvm::cl::cat(clOptionsCategory));

#ifdef ENABLE_GEN_CMDBUF
#include "backend/backend_tg_api.h"
static BM1880v2BackendContext *backend_ctx = nullptr;
#endif

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

namespace mlir {

std::vector<std::shared_ptr<std::vector<float> > >
    ModuleInterpreter::getOperandTensors(Operation &opInst,
    value_map_t &valueMapping) {
  std::vector<std::shared_ptr<std::vector<float> > > opdT;
  for (auto operand : opInst.getOperands()) {
    auto it = valueMapping.find(operand);
    assert(it != valueMapping.end());
    opdT.push_back(it->second);
  }
  return opdT;
}

LogicalResult ModuleInterpreter::runOperation(Operation &opInst) {
  // #include "mlir/Dialect/LLVMIR/LLVMConversions.inc"
  if (auto loadFileOp = dyn_cast<tpu::LoadFileOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "LoadFileOp" << "\n";);
    auto filename = loadFileOp.getAttrOfType<StringAttr>("filename").getValue();
    LLVM_DEBUG(llvm::errs() << "  filename " << filename << "\n";);
    weight_is = std::make_unique<std::ifstream>(filename.str(),
        std::ios::in | std::ios::binary);
    auto filename_tensorfile = llvm::sys::path::stem(filename).str() + ".npz";
    weight_file = openInputTensorFile(filename_tensorfile);

    return success();
  }
  if (auto loadWeightOp = dyn_cast<tpu::LoadWeightOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "LoadWeightOp" << "\n";);

    auto result = loadWeightOp.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    if (loadWeightOp.name().hasValue()) {
      auto tensor_name = loadWeightOp.name().getValue();
      LLVM_DEBUG(llvm::errs() << "  tensor_name " << tensor_name << "\n";);
      auto type = result->getType().cast<TensorType>();
      auto tensor = weight_file->readTensor<float>(tensor_name, type);

      valueMapping[result] = std::move(tensor);
    } else {
      assert(loadWeightOp.offset().hasValue());
      auto offset = loadWeightOp.offset().getValue().getLimitedValue();
      LLVM_DEBUG(llvm::errs() << "  offset " << offset << "\n";);
      std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
      assert(shape.size() <= 4);
      auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
      auto weight_data = std::make_unique<std::vector<float> >(size);

      weight_is.get()->seekg(offset, std::ios::beg);
      weight_is.get()->read((char*)weight_data.get()->data(), size * sizeof(float));

      valueMapping[result] = std::move(weight_data);
    }
    return success();
  }
  if (auto op = dyn_cast<tpu::InputOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "InputOp" << "\n";);
    auto opdT = getOperandTensors(opInst, valueMapping);
    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() == 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto resultT = std::make_shared<std::vector<float> >(size);

    // use copy for now
    resultT->assign(opdT[0]->begin(), opdT[0]->end());

    valueMapping[result] = std::move(resultT);

    return success();
  }
  if (auto op = dyn_cast<tpu::Conv2DOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "Conv2DOp" << "\n";);
    auto opdT = getOperandTensors(opInst, valueMapping);
    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() == 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto resultT = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here
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
    float *mkldnn_input = (float *)opdT[0]->data();
    float *mkldnn_weight = (float *)opdT[1]->data();
    float *mkldnn_bias = nullptr;
    float *rshift = nullptr;
    float *multiplier = nullptr;
    if (op.quant() == "NONE") {
      if (opdT.size() > 2) {
        assert(opdT.size() == 3);
        mkldnn_bias = (float *)opdT[2]->data();
      }
    } else if (op.quant() == "INT8" || op.quant() == "INT8_PER_CHANNEL") {
      if (opdT.size() > 3) {
        assert(opdT.size() == 4);
        mkldnn_bias = (float *)opdT[2]->data();
        rshift = (float *)opdT[3]->data();
      } else {
        assert(opdT.size() == 3);
        rshift = (float *)opdT[2]->data();
      }
    } else if (op.quant() == "INT8_MULTIPLIER") {
      if (opdT.size() > 4) {
        assert(opdT.size() == 5);
        mkldnn_bias = (float *)opdT[2]->data();
        rshift = (float *)opdT[3]->data();
        multiplier = (float *)opdT[4]->data();
      } else if (opdT.size() == 4) {
        rshift = (float *)opdT[2]->data();
        multiplier = (float *)opdT[3]->data();
      } else {
        assert(opdT.size() == 3);
        // fake some data for now, we only needs the cmdbuf for now
        mkldnn_bias = (float *)opdT[1]->data();
        rshift = (float *)opdT[1]->data();
        multiplier = (float *)opdT[1]->data();
      }
    } else {
      assert(false);
    }

#ifdef QUANT_DEQUANT_EVERY_LAYER
    // do quantize on input
    // remove this when the network is full int8, and passed legalization
    // copy the input first
    std::vector<float> input_copy(*opdT[0]);
    mkldnn_input = input_copy.data();
    if (op.quant() == "INT8" || op.quant() == "INT8_PER_CHANNEL"
        || op.quant() == "INT8_MULTIPLIER") {
      float threshold_x = getPreviousOpThreshold(op);
      LLVM_DEBUG(llvm::errs() << "  conv input quantize, threshold_x = "
                              << std::to_string(threshold_x) << "\n";);
      for (size_t i = 0; i < opdT[0]->size(); ++i) {
        mkldnn_input[i] = (float)saturateInt8(mkldnn_input[i] * 128.0 / threshold_x);
      }
    }
#endif

    float *mkldnn_output = (float *)resultT.get()->data();
    int mkldnn_ret = mkldnn_conv(mkldnn_input, mkldnn_weight, mkldnn_bias, mkldnn_output,
        n, ic, ih, iw, oc, oh, ow, kh, kw, sh, sw, ph, pw);
    assert(mkldnn_ret == 0);
    //dump_data_float_abs("mkldnn_output", mkldnn_output, n, oc, oh, ow);

    if (op.fused_activation_function() == "NONE") {
    } else if (op.fused_activation_function() == "RELU") {
      my_relu(mkldnn_output, mkldnn_output, n, oc, oh, ow, 0.0f);
    } else {
      assert(0);
    }

    // rshift and saturate on output
    if (op.quant() == "INT8") {
      assert(rshift);
      for (int i = 0; i < size; ++i) {
        mkldnn_output[i] = (float)applyRShiftAndSaturateInt8(mkldnn_output[i], (uint32_t)rshift[0]);
      }
    } else if (op.quant() == "INT8_PER_CHANNEL") {
      assert(rshift);
      int inner_size = size / oc;
      for (int i = 0; i < oc; ++i) {
        for (int j = 0; j < inner_size; ++j) {
          mkldnn_output[i * inner_size + j] =
              (float)applyRShiftAndSaturateInt8(mkldnn_output[i * inner_size + j],
                                                (uint32_t)rshift[i]);
        }
      }
    } else if (op.quant() == "INT8_MULTIPLIER") {
      assert(multiplier);
      int inner_size = size / oc;
      for (int i = 0; i < oc; ++i) {
        for (int j = 0; j < inner_size; ++j) {
          mkldnn_output[i * inner_size + j] =
              (float)applyMultiplierAndRShiftAndSaturateInt8(mkldnn_output[i * inner_size + j],
                                                             rshift[i], multiplier[i], true);
        }
      }
    }

#ifdef QUANT_DEQUANT_EVERY_LAYER
    // do dequantize on output
    // remove this when the network is full int8, and passed legalization
    if (op.quant() == "INT8" || op.quant() == "INT8_PER_CHANNEL"
        || op.quant() == "INT8_MULTIPLIER") {
      float threshold_y = op.threshold_y().getValue().convertToFloat();
      LLVM_DEBUG(llvm::errs() << "  conv output dequantize, threshold_y = "
                   << std::to_string(threshold_y) << "\n";);
      for (int i = 0; i < size; ++i) {
        mkldnn_output[i] = mkldnn_output[i] * threshold_y / 128.0;
      }
    }
#endif
    // TODO: End of compute, need refactor

    valueMapping[result] = std::move(resultT);

#ifdef ENABLE_GEN_CMDBUF
    if (clCmdBufFilename != "-") {

    gaddr_t input_gaddr = getPreviousOpAddress(op);
    gaddr_t output_gaddr = op.offset().getValue().getLimitedValue();
    gaddr_t filter_gaddr = getWeightOpAddress(op.getOperand(1)->getDefiningOp());

    if (op.quant() == "INT8") {

    gaddr_t bias_gaddr = INVALID_GLOBAL_ADDR;
    int with_bias = 0;
    if (opdT.size() > 3) {
      with_bias = 1;
    }
    if (with_bias) {
      bias_gaddr = getWeightOpAddress(op.getOperand(2)->getDefiningOp());
    }

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
        0, // do_activation,
        1.0f, // bn_scale,
        1e-5, // eps,
        0, // param.activation(), method
        nullptr, // activation_arg,
        INVALID_GLOBAL_ADDR, //global_slope_gaddr,
        false, //channel_shared,
        0, //activation_gt_scale,
        0, //activation_gt_rshift,
        0, //activation_le_scale, // slope, TODO
        0, //activation_le_rshift,
        (int)rshift[0], //right_shift_width,
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
        0, // do_activation,
        1.0f, // bn_scale,
        1e-5, // eps,
        0, // param.activation(), method
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

    } // clCmdBufFilename
#endif

    return success();
  }
  if (auto op = dyn_cast<tpu::Pool2DOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "Pool2DOp" << "\n";);
    auto opdT = getOperandTensors(opInst, valueMapping);
    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto resultT = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here
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
    float *mkldnn_input = (float *)opdT[0]->data();

    // for INT8, get threshold_x and make copy of input first
    std::vector<float> input_copy;
    float threshold_x;
    float threshold_y;
    if (op.quant() == "INT8" && is_average_pool) {
      // make copy
      std::vector<float> &src_vec = *opdT[0];
      std::copy(src_vec.begin(), src_vec.end(), back_inserter(input_copy));
      mkldnn_input = input_copy.data();

      threshold_x = getPreviousOpThreshold(op);
      threshold_y = op.threshold_y().getValue().convertToFloat();
    }

#ifdef QUANT_DEQUANT_EVERY_LAYER
    // do quantize on input
    // remove this when the network is full int8, and passed legalization
    if (op.quant() == "INT8" && is_average_pool) {
      for (size_t i = 0; i < opdT[0]->size(); ++i) {
        mkldnn_input[i] = (float)saturateInt8(mkldnn_input[i] * 128.0 / threshold_x);
      }
    }
#endif

    float *mkldnn_output = (float *)resultT.get()->data();
    int mkldnn_ret = mkldnn_pool(mkldnn_input, mkldnn_output,
        n, c, ih, iw, oh, ow, kh, kw, sh, sw, ph, pw, is_average_pool);
    assert(mkldnn_ret == 0);
    //dump_data_float_abs("mkldnn_output", mkldnn_output, n, c, oh, ow);

    uint32_t rshift = 0;
    // multiplier is taking avg_const into account
    uint32_t multiplier = 0;
    // do quantize for average pooling, max poolings are bypassed
    if (op.quant() == "INT8" && is_average_pool) {
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

      // apply multiplier, rshift and saturate
      for (int i = 0; i < size; ++i) {
        // restore sum value first
        int sum = (int)(mkldnn_output[i] * kh * kw + 0.5);
        mkldnn_output[i] = (float)applyMultiplierAndRShiftAndSaturateInt8(
                                      sum, rshift, multiplier);
      }
    }

#ifdef QUANT_DEQUANT_EVERY_LAYER
    // do dequantize on output
    // remove this when the network is full int8, and passed legalization
    if (op.quant() == "INT8" && is_average_pool) {
      LLVM_DEBUG(llvm::errs() << "  avg pool output dequantize, threshold_y = "
                 << std::to_string(threshold_y) << "\n";);
      for (int i = 0; i < size; ++i) {
        mkldnn_output[i] = mkldnn_output[i] * threshold_y / 128.0;
      }
    }
#endif
    // TODO: End of compute, need refactor

    valueMapping[result] = std::move(resultT);

#ifdef ENABLE_GEN_CMDBUF
    if (clCmdBufFilename != "-") {

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

    } // clCmdBufFilename
#endif

    return success();
  }
  if (auto op = dyn_cast<tpu::FullyConnectedOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "FullyConnectedOp" << "\n";);
    auto opdT = getOperandTensors(opInst, valueMapping);
    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() == 2);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto resultT = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here
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
    assert((i_s[1] == f_s[1]) && "input K not equal to filter K");
    k = i_s[1];
    assert((f_s[0] == o_s[1]) && "filter N not equal to output N");
    n = o_s[1];

    float *mkldnn_input = (float *)opdT[0]->data();
    float *mkldnn_weight = (float *)opdT[1]->data();
    float *mkldnn_bias = nullptr;
    float *rshift = nullptr;
    if (op.quant() == "NONE") {
      if (opdT.size() > 2) {
        assert(opdT.size() == 3);
        mkldnn_bias = (float *)opdT[2]->data();
      }
    } else if (op.quant() == "INT8") {
      if (opdT.size() > 3) {
        assert(opdT.size() == 4);
        mkldnn_bias = (float *)opdT[2]->data();
        rshift = (float *)opdT[3]->data();
      } else {
        assert(opdT.size() == 3);
        rshift = (float *)opdT[2]->data();
      }
    } else {
      assert(false);
    }

#ifdef QUANT_DEQUANT_EVERY_LAYER
    // do quantize on input
    // remove this when the network is full int8, and passed legalization
    // copy the input first
    std::vector<float> input_copy(*opdT[0]);
    mkldnn_input = input_copy.data();
    if (op.quant() == "INT8") {
      float threshold_x = getPreviousOpThreshold(op;
      LLVM_DEBUG(llvm::errs() << "  fc input quantize, threshold_x = "
                              << std::to_string(threshold_x) << "\n";);
      for (size_t i = 0; i < opdT[0]->size(); ++i) {
        mkldnn_input[i] = (float)saturateInt8(mkldnn_input[i] * 128.0 / threshold_x);
      }
    }
#endif

    float *mkldnn_output = (float *)resultT.get()->data();
    int mkldnn_ret = mkldnn_ip(mkldnn_input, mkldnn_weight, mkldnn_bias,
        mkldnn_output, m, k, n, transpose);
    assert(mkldnn_ret == 0);
    //dump_data_float_abs("mkldnn_output", mkldnn_output, 1, 1, m, n);

    // rshift and saturate on output
    if (op.quant() == "INT8") {
      assert(rshift);
      for (int i = 0; i < size; ++i) {
        mkldnn_output[i] = (float)applyRShiftAndSaturateInt8(mkldnn_output[i],
                                                             (uint32_t)rshift[0]);
      }
    }

#ifdef QUANT_DEQUANT_EVERY_LAYER
    // do dequantize on output
    // remove this when the network is full int8, and passed legalization
    if (op.quant() == "INT8") {
      float threshold_y = op.threshold_y().getValue().convertToFloat();
      LLVM_DEBUG(llvm::errs() << "  fc output dequantize, threshold_y = "
                   << std::to_string(threshold_y) << "\n";);
      for (int i = 0; i < size; ++i) {
        mkldnn_output[i] = mkldnn_output[i] * threshold_y / 128.0;
      }
    }
#endif

    // TODO: End of compute, need refactor

    valueMapping[result] = std::move(resultT);

#ifdef ENABLE_GEN_CMDBUF
    if (clCmdBufFilename != "-") {

    gaddr_t input_gaddr = getPreviousOpAddress(op);
    gaddr_t output_gaddr = op.offset().getValue().getLimitedValue();
    gaddr_t filter_gaddr = getWeightOpAddress(op.getOperand(1)->getDefiningOp());
    int with_bias = 0;
    gaddr_t bias_gaddr = INVALID_GLOBAL_ADDR;
    if (op.quant() == "NONE") {
      if (opdT.size() > 2) {
        with_bias = 1;
      }
    } else if (op.quant() == "INT8" || op.quant() == "INT8_PER_CHANNEL"
               || op.quant() == "INT8_MULTIPLIER") {
      if (opdT.size() > 3) {
        with_bias = 1;
      }
    }
    if (with_bias) {
      bias_gaddr = getWeightOpAddress(op.getOperand(2)->getDefiningOp());
    }

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
        rshift[0],
        0, //int threshold_x_quantized_len,
        nullptr, //const int *threshold_x_quantized,
        nullptr //const int *right_shift_array
        );

    } // clCmdBufFilename
#endif

    return success();
  }
  if (auto op = dyn_cast<tpu::ReluOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "ReluOp" << "\n";);
    auto opdT = getOperandTensors(opInst, valueMapping);
    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto resultT = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here
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
    float *input = (float *)opdT[0]->data();
    float *output = (float *)resultT.get()->data();
    int ret = my_relu(input, output, n, c, h, w, negative_slope);
    assert(ret == 0);
    //dump_data_float_abs("mkldnn_output", mkldnn_output, n, c, oh, ow);
    // TODO: End of compute, need refactor

    valueMapping[result] = std::move(resultT);

#ifdef ENABLE_GEN_CMDBUF
    if (clCmdBufFilename != "-") {

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

    } // clCmdBufFilename
#endif

    return success();
  }
  if (auto op = dyn_cast<tpu::SoftmaxOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "SoftmaxOp" << "\n";);
    auto opdT = getOperandTensors(opInst, valueMapping);
    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() == 2);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto resultT = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here
    int n, c;
    auto input_type = op.x()->getType().cast<TensorType>();
    std::vector<int64_t> i_s(input_type.getShape());
    auto output_type = op.y()->getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());
    assert((i_s == o_s) && "input shape not equal to output shape");
    n = i_s[0];
    c = i_s[1];
    float *input = (float *)opdT[0]->data();

    // do dequantization
    if (0) {
      float threshold_x = getPreviousOpThreshold(op);
      LLVM_DEBUG(llvm::errs() << "  softmax dequantize, threshold_x = "
                              << std::to_string(threshold_x) << "\n";);
      for (size_t i = 0; i < opdT[0]->size(); ++i) {
        input[i] = input[i] * threshold_x / 128.0;
      }
    }

    float *output = (float *)resultT.get()->data();
    int ret = my_softmax(input, output, n, c);
    assert(ret == 0);
    //dump_data_float_abs("mkldnn_output", mkldnn_output, n, c, oh, ow);
    // TODO: End of compute, need refactor

    valueMapping[result] = std::move(resultT);
    return success();
  }
  if (auto op = dyn_cast<tpu::BatchNormOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "BatchNormOp" << "\n";);
    auto opdT = getOperandTensors(opInst, valueMapping);
    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto resultT = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here
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
    float *input = (float *)opdT[0]->data();
    float *mean = (float *)opdT[1]->data();
    float *variance = (float *)opdT[2]->data();
    float *scale = (float *)opdT[3]->data();
    float *output = (float *)resultT.get()->data();
    int ret = my_bn(input, mean, variance, scale, output, n, c, h, w);
    assert(ret == 0);
    //dump_data_float_abs("mkldnn_output", mkldnn_output, n, c, oh, ow);
    // TODO: End of compute, need refactor

    valueMapping[result] = std::move(resultT);

#ifdef ENABLE_GEN_CMDBUF
    if (clCmdBufFilename != "-") {

    assert(false && "GEN_CMDBUF does not support bn, bn should change to scale");

    }
#endif

    return success();
  }
  if (auto op = dyn_cast<tpu::ScaleOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "ScaleOp" << "\n";);
    auto opdT = getOperandTensors(opInst, valueMapping);
    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto resultT = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here
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
    float *input = (float *)opdT[0]->data();
    float *scale = (float *)opdT[1]->data();
    float *bias = nullptr;
    if (opdT.size() > 2) {
      assert(opdT.size() == 3);
      bias = (float *)opdT[2]->data();
    }
    float *output = (float *)resultT.get()->data();
    int ret = my_scale(input, scale, bias, output, n, c, h, w);
    assert(ret == 0);
    //dump_data_float_abs("mkldnn_output", mkldnn_output, n, c, oh, ow);
    // TODO: End of compute, need refactor

    valueMapping[result] = std::move(resultT);

#ifdef ENABLE_GEN_CMDBUF
    if (clCmdBufFilename != "-") {

    assert(false && "GEN_CMDBUF does not support scale, scale should merge into conv");

    }
#endif

    return success();
  }
  if (auto op = dyn_cast<tpu::EltwiseOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "EltwiseOp" << "\n";);
    auto opdT = getOperandTensors(opInst, valueMapping);
    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto resultT = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here
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
    float *input[MAX_ELTWISE_INPUT];
    for (int index = 0; index < MAX_ELTWISE_INPUT; ++index) {
      input[index] = (float *)opdT[index]->data();
    }
    float *output = (float *)resultT.get()->data();

    // for INT8, get threshold_x and make copy of input first
    std::vector<float> input_copy[MAX_ELTWISE_INPUT];
    std::vector<float> threshold_x(MAX_ELTWISE_INPUT);
    float threshold_y;
    if (op.quant() == "INT8") {
      for (int index = 0; index < MAX_ELTWISE_INPUT; ++index) {
        // make copy
        std::vector<float> &src_vec = *opdT[index];
        std::copy(src_vec.begin(), src_vec.end(), back_inserter(input_copy[index]));
        input[index] = input_copy[index].data();

        // get threshold_x
        threshold_x[index] = getPreviousOpThreshold(op, index);
      }
      // get threshold_y
      threshold_y = op.threshold_y().getValue().convertToFloat();
    }

#ifdef QUANT_DEQUANT_EVERY_LAYER
    // do quantize on input
    // remove this when the network is full int8, and passed legalization
    if (op.quant() == "INT8") {
      for (int index = 0; index < MAX_ELTWISE_INPUT; ++index) {
        for (size_t i = 0; i < opdT[index]->size(); ++i) {
          input[index][i] = (float)saturateInt8(input[index][i] * 128.0 / threshold_x[index]);
        }
      }
    }
#endif

    // determine multiplier and rshift according each threshold_x
    // scale[i] = threshold_x[i] / threshold_y
    // each scale will be implemented by hardware as
    // scale[i] = multiplier / (1 << rshift)
    // find a rshift, that put max(multiplier) into range (64, 127)
    uint32_t rshift;
    int8_t multiplier[MAX_ELTWISE_INPUT];
    if (op.quant() == "INT8") {
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

    // apply multiplier
    if (op.quant() == "INT8") {
      for (int index = 0; index < MAX_ELTWISE_INPUT; ++index) {
        for (size_t i = 0; i < opdT[index]->size(); ++i) {
          input[index][i] = input[index][i] * multiplier[index];
        }
      }
    }

    int ret = my_eltwise(input[0], input[1], output, n, c, h, w, 1);
    assert(ret == 0);
    //dump_data_float_abs("output", mkldnn_output, n, c, oh, ow);

    if (op.fused_activation_function() == "NONE") {
    } else if (op.fused_activation_function() == "RELU") {
      my_relu(output, output, n, c, h, w, 0.0f);
    } else {
      assert(0);
    }

    // rshift and saturate on output
    if (op.quant() == "INT8") {
      //assert(rshift);
      for (int i = 0; i < size; ++i) {
        output[i] = (float)applyRShiftAndSaturateInt8(output[i], (uint32_t)rshift);
      }
    }

#ifdef QUANT_DEQUANT_EVERY_LAYER
    // do dequantize on output
    // remove this when the network is full int8, and passed legalization
    if (op.quant() == "INT8") {
      LLVM_DEBUG(llvm::errs() << "  fc output dequantize, threshold_y = "
                   << std::to_string(threshold_y) << "\n";);
      for (int i = 0; i < size; ++i) {
        output[i] = output[i] * threshold_y / 128.0;
      }
    }
#endif
    // TODO: End of compute, need refactor

    valueMapping[result] = std::move(resultT);

#ifdef ENABLE_GEN_CMDBUF
    if (clCmdBufFilename != "-") {

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
        false, // bool do_relu,
        0.0f, // float relu_slope,
        rshift, //int right_shift_width,
        threshold_x_quantized,
        coeffs);
    // gen cmd end

    } // clCmdBufFilename
#endif

    return success();
  }
  if (auto op = dyn_cast<tpu::ReshapeOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "ReshapeOp" << "\n";);
    auto opdT = getOperandTensors(opInst, valueMapping);
    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto resultT = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here
    auto input_type = op.input()->getType().cast<TensorType>();
    std::vector<int64_t> i_s(input_type.getShape());
    auto output_type = op.output()->getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());
    auto i_size = std::accumulate(std::begin(i_s), std::end(i_s), 1, std::multiplies<>());
    auto o_size = std::accumulate(std::begin(o_s), std::end(o_s), 1, std::multiplies<>());
    assert((i_size == o_size) && "input size not equal to output size");

    // use copy for now
    resultT.get()->assign(opdT[0]->begin(), opdT[0]->end());
    // TODO: End of compute, need refactor

    valueMapping[result] = std::move(resultT);

    return success();
  }
  if (auto op = dyn_cast<tpu::QuantizationOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "QuantizationOp" << "\n";);
    auto opdT = getOperandTensors(opInst, valueMapping);
    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto resultT = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here
    if (op.quant() == "INT8") {
      float *input = (float *)opdT[0]->data();
      float *output = (float *)resultT.get()->data();
      float threshold = op.threshold().getValue().convertToFloat();
      LLVM_DEBUG(llvm::errs() << "  quantization, threshold = "
                   << std::to_string(threshold) << "\n";);
      for (int i = 0; i < size; ++i) {
        output[i] = (float)quantizeNeuron(input[i], threshold);
      }
    }
    // TODO: End of compute, need refactor

    valueMapping[result] = std::move(resultT);

    return success();
  }
  if (auto op = dyn_cast<tpu::DequantizationOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "DequantizationOp" << "\n";);
    auto opdT = getOperandTensors(opInst, valueMapping);
    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto resultT = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here
    if (op.quant() == "INT8") {
      float *input = (float *)opdT[0]->data();
      float *output = (float *)resultT.get()->data();
      float threshold = op.threshold().getValue().convertToFloat();
      LLVM_DEBUG(llvm::errs() << "  quantization, threshold = "
                   << std::to_string(threshold) << "\n";);
      for (int i = 0; i < size; ++i) {
        output[i] = dequantizeNeuron((int8_t)input[i], threshold);
      }
    }
    // TODO: End of compute, need refactor

    valueMapping[result] = std::move(resultT);

    return success();
  }

  if (auto op = dyn_cast<ConstantOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "ConstantOp" << "\n";);
    //op.dump();
    // TODO: use specific Op for null operand
    // only support zero constant for now
    // TODO: check isZero

    // it it safe to ignore, put null pointer to the valueMapping
    auto result = op.getResult();
    valueMapping[result] = std::move(nullptr);

    return success();
  }

  if (auto op = dyn_cast<ReturnOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "ReturnOp" << "\n";);
    auto opdT = getOperandTensors(opInst, valueMapping);
    //copy the value into outputs_
    assert(outputs_.size() == 1);
    outputs_[0]->assign(opdT[0]->begin(), opdT[0]->end());

    return success();
  }

  return opInst.emitError("unsupported operation: ")
         << opInst.getName();
}

LogicalResult ModuleInterpreter::runBlock(Block &bb) {
  // Traverse operations.
  for (auto &op : bb) {
    if (failed(runOperation(op)))
      return failure();
  }

  return success();
}

LogicalResult ModuleInterpreter::runOneFunction(FuncOp func) {
  LLVM_DEBUG(llvm::errs() << "func " << func.getName() << "\n";);
  // Clear the value mappings, it is only relevant within one function.
  valueMapping.clear();

  // Add function arguments to the value remapping table.
  unsigned int argIdx = 0;
  assert(inputs_.size() == 1);
  for (auto arg : func.getArguments()) {
    LLVM_DEBUG(
      llvm::errs() << "arg " << argIdx << ": ";
      arg->getType().dump();
      llvm::errs() << "\n";
    );

    // copy the inputs_[0] into a unique_ptr pointed vector
    // TODO: pass input as unique_ptr directly
    auto input = std::make_unique<std::vector<float> >();
    input->swap(*inputs_[0]);
    valueMapping[arg] = std::move(input);
    argIdx++;
  }
  assert(argIdx == 1);

#ifdef ENABLE_GEN_CMDBUF
  if (clCmdBufFilename != "-") {
    std::vector<int8_t> weight_data;
    backend_ctx = bmnet_create_backend_context(weight_data);
  }
#endif

  // Then, run blocks one by one.
  for (Block &bb : func.getBlocks()) {
    if (failed(runBlock(bb)))
      return failure();
  }

#ifdef ENABLE_GEN_CMDBUF
  if (clCmdBufFilename != "-") {
    bmnet_submit(backend_ctx);
    std::vector<uint8_t> cmdbuf;
    bmnet_read_cmdbuf(backend_ctx, cmdbuf);
    std::fstream output(clCmdBufFilename, std::ios::out | std::ios::trunc | std::ios::binary);
    output.write((char *)cmdbuf.data(), cmdbuf.size());
  }
#endif

  if (clAllTensorFilename != "-") {
    // dump all values
    LLVM_DEBUG(llvm::errs() << "valueMapping size " << valueMapping.size() << "\n";);
    auto TensorOut = openOutputTensorFile(clAllTensorFilename);
    for (auto it = valueMapping.begin(); it != valueMapping.end(); it++ ) {
      auto op = it->first->getDefiningOp();
      if (!op) {
        //it->first->dump();
        continue;
      }
      LLVM_DEBUG(llvm::errs() << op->getName() << " : " << getOpName(op) << "\n";);
      auto vec = it->second.get();
      assert(vec);
      auto type = it->first->getType().dyn_cast<mlir::TensorType>();
      LLVM_DEBUG(llvm::errs() << "  vec size = " << vec->size() << "\n";);
      TensorOut->addTensor(getOpName(op), vec, type);
    }
    TensorOut->keep();
  }

  return success();
}

LogicalResult ModuleInterpreter::runFunctions() {
  for (FuncOp function : mlirModule.getOps<FuncOp>()) {
    LLVM_DEBUG(llvm::errs() << "run " << function.getName() << "\n";);

    if (!function.getName().equals("tpu_func")) {
      //continue;
      assert(0);
    }
    if (failed(runOneFunction(function)))
      return failure();
  }

  return success();
}

LogicalResult runTpuModule(ModuleOp m,
    std::vector<std::vector<float> *> &inputs,
    std::vector<std::vector<float> *> &outputs) {
  return ModuleInterpreter::runModule<>(m, inputs, outputs);
}

} // namespace mlir
