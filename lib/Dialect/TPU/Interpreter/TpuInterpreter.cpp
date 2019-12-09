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

using namespace std;

static llvm::cl::OptionCategory clOptionsCategory("interpreter options");

static llvm::cl::opt<std::string> clAllTensorFilename(
    "dump-all-tensor",
    llvm::cl::desc("dump all tensor into a npz file"),
    llvm::cl::init("-"),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<float> clInputScale(
    "input-scale",
    llvm::cl::desc("input scale to apply on the input values"),
    llvm::cl::cat(clOptionsCategory));

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
    assert(loadWeightOp.name().hasValue());
    auto tensor_name = loadWeightOp.name().getValue();
    LLVM_DEBUG(llvm::errs() << "  tensor_name " << tensor_name << "\n";);

    auto type = result->getType().cast<TensorType>();
    std::unique_ptr<std::vector<float> > tensor= nullptr;
    if (type.getElementType().isF32()) {
      tensor = std::move(weight_file->readTensor<float>(tensor_name, type));
    } else if (type.getElementType().isInteger(8)) {
      // TODO: we still save int8 weight as fp32 for now
      assert(0);
    } else if (type.getElementType().isBF16()) {
      auto tensor_bf16 = weight_file->readTensor<bfloat16>(tensor_name, type);

      // TODO: convert bf16 to fp32 here for now
      // as valueMapping is hardcoded as std::vector<float>
      // TODO: more generic valueMapping
      tensor = std::move(std::make_unique<std::vector<float> >(tensor_bf16->size()));
      BFloat16ToFloat(tensor_bf16->data(), tensor->data(), tensor_bf16->size());
    } else {
      assert(0);
    }

    valueMapping[result] = std::move(tensor);
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

    float inputScale =
        (clInputScale.getNumOccurrences() > 0) ? clInputScale : 1.0f;

    if (inputScale != 1.0f) {
      llvm::errs() << "Apply input_scale = " << clInputScale << "\n";
      for(auto it = resultT->begin(); it != resultT->end(); it++ ) {
        *it *= inputScale;
      }
    }

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

    bool with_bias, do_relu;
    int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, ph, pw, dh, dw;
    getConv2DOpParam(op, n, ic, ih, iw, oc, oh, ow, g,
                     kh, kw, sh, sw, ph, pw, dh, dw, with_bias, do_relu);

    std::shared_ptr<std::vector<float> > input = opdT[0];
    std::shared_ptr<std::vector<float> > filter = opdT[1];
    std::shared_ptr<std::vector<float> > bias = nullptr;
    std::shared_ptr<std::vector<float> > rshift = nullptr;
    std::shared_ptr<std::vector<float> > multiplier = nullptr;
    std::shared_ptr<std::vector<float> > per_channel_info = nullptr;
    std::shared_ptr<std::vector<float> > eltwise_input = nullptr;
    if (op.per_channel_info_is_aggregated()) {
      llvm::errs() << "Not support interpret with per_channel_info aggreated\n";
      assert(0);
    }
    getConv2DOpVariadicTensors(op, opdT, bias, rshift, multiplier,
        per_channel_info, eltwise_input);

    float *output_data = resultT->data();
    int mkldnn_ret = mkldnn_conv(input->data(), filter->data(),
        bias?bias->data():nullptr, output_data,
        n, ic, ih, iw, oc, oh, ow, kh, kw, sh, sw, ph, pw, g);
    assert(mkldnn_ret == 0);
    //dump_data_float_abs("mkldnn_output", mkldnn_output, n, oc, oh, ow);

    if (do_relu) {
      my_relu(output_data, output_data, n, oc, oh, ow, 0.0f);
    }

    if (op.fused_eltwise_method() == "SUM") {
      assert(eltwise_input);
      my_eltwise(eltwise_input->data(), output_data, output_data, n, oc, oh, ow, 1);

      if (op.fused_activation_function_after_eltwise() == "RELU") {
        my_relu(output_data, output_data, n, oc, oh, ow, 0.0f);
      } else {
        assert(op.fused_activation_function_after_eltwise() == "NONE");
      }
    } else {
      assert(eltwise_input == nullptr);
      assert(op.fused_eltwise_method() == "NONE");
    }

    // rshift and saturate on output
    if (op.quant() == "INT8") {
      assert(rshift);
      for (int i = 0; i < size; ++i) {
        output_data[i] = (float)applyRShiftAndSaturateInt8(output_data[i],
            (uint32_t)rshift->at(0));
      }
    } else if (op.quant() == "INT8_PER_CHANNEL") {
      assert(rshift);
      int inner_size = size / oc;
      for (int i = 0; i < oc; ++i) {
        for (int j = 0; j < inner_size; ++j) {
          output_data[i * inner_size + j] = (float)applyRShiftAndSaturateInt8(
              output_data[i * inner_size + j], (uint32_t)rshift->at(i));
        }
      }
    } else if (op.quant() == "INT8_MULTIPLIER") {
      assert(multiplier);
      int inner_size = size / oc;
      for (int i = 0; i < oc; ++i) {
        for (int j = 0; j < inner_size; ++j) {
          output_data[i * inner_size + j] =
              (float)applyMultiplierAndRShiftAndSaturateInt8(
                  output_data[i * inner_size + j],
                  rshift->at(i), multiplier->at(i), true);
        }
      }
    } else if (op.quant() == "BF16") {
      auto tensor_bf16 = std::make_unique<std::vector<bfloat16> >(resultT->size());
      FloatToBFloat16(resultT->data(), tensor_bf16->data(), resultT->size()); // with rounding
      BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
    }

    valueMapping[result] = std::move(resultT);

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

    bool is_average_pool, do_relu;
    int n, c, ih, iw, oh, ow, kh, kw, sh, sw, ph, pw;
    getPool2DOpParam(op, is_average_pool, n, c, ih, iw, oh, ow,
                     kh, kw, sh, sw, ph, pw, do_relu);

    std::shared_ptr<std::vector<float> > input = opdT[0];

    // for INT8, get threshold_x and make copy of input first
    std::vector<float> input_copy;
    float threshold_x;
    float threshold_y;
    if (op.quant() == "INT8" && is_average_pool) {
      // make copy
      auto input_copy = make_shared<std::vector<float> >();
      input_copy->assign(input->begin(), input->end());
      input = input_copy;
      // get threshold
      threshold_x = getPreviousOpThreshold(op);
      threshold_y = op.threshold_y().getValue().convertToFloat();
    }

    float *output_data = resultT->data();
    int mkldnn_ret = mkldnn_pool(input->data(), output_data,
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
      //rshift = findRShiftAndMultiplierFromQScale(scale_and_avg_const,
      //                                           &multiplier, false, 127);
      rshift = findRShiftAndMultiplierFromQScale(scale_and_avg_const,
                                                 &multiplier, false, 255);

      // apply multiplier, rshift and saturate
      for (int i = 0; i < size; ++i) {
        // restore sum value first
        int sum = (int)(output_data[i] * kh * kw + 0.5);
        output_data[i] = (float)applyMultiplierAndRShiftAndSaturateInt8(
                                      sum, rshift, multiplier);
      }
    }

    if (op.quant() == "BF16" && is_average_pool) {
      auto tensor_bf16 = std::make_unique<std::vector<bfloat16> >(resultT->size());
      FloatToBFloat16(resultT->data(), tensor_bf16->data(), resultT->size()); // with rounding
      BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
    }

    valueMapping[result] = std::move(resultT);

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

    bool with_transpose, with_bias, do_relu;
    int m, k, n;
    getFullyConnectedOpParam(op, with_transpose, m, k, n, with_bias, do_relu);
    assert(with_transpose == false);

    std::shared_ptr<std::vector<float> > input = opdT[0];
    std::shared_ptr<std::vector<float> > filter = opdT[1];
    std::shared_ptr<std::vector<float> > bias = nullptr;
    std::shared_ptr<std::vector<float> > rshift = nullptr;
    getFullyConnectedOpVariadicTensors(op, opdT, bias, rshift);

    float *output_data = (float *)resultT->data();
    int mkldnn_ret = mkldnn_ip(input->data(), filter->data(),
        bias?bias->data():nullptr, output_data, m, k, n, with_transpose);
    assert(mkldnn_ret == 0);
    //dump_data_float_abs("output_data", output_data, 1, 1, m, n);

    // rshift and saturate on output
    if (op.quant() == "INT8") {
      assert(rshift);
      for (int i = 0; i < size; ++i) {
        output_data[i] = (float)applyRShiftAndSaturateInt8(output_data[i],
                                                           (uint32_t)rshift->at(0));
      }
    } else if (op.quant() == "BF16") {
      auto tensor_bf16 = std::make_unique<std::vector<bfloat16> >(resultT->size());
      FloatToBFloat16(resultT->data(), tensor_bf16->data(), resultT->size()); // with rounding
      BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
    }

    valueMapping[result] = std::move(resultT);

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

    valueMapping[result] = std::move(resultT);

    return success();
  }
  if (auto op = dyn_cast<tpu::SoftmaxOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "SoftmaxOp" << "\n";);
    auto opdT = getOperandTensors(opInst, valueMapping);
    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() == 2 || shape.size() == 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto resultT = std::make_unique<std::vector<float> >(size);

    int n, c;
    auto input_type = op.x()->getType().cast<TensorType>();
    std::vector<int64_t> i_s(input_type.getShape());
    auto output_type = op.y()->getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());
    assert((i_s == o_s) && "input shape not equal to output shape");
    n = i_s[0];
    c = i_s[1];
    if (i_s.size() == 4) {
      assert(i_s[2] == 1 && i_s[3] == 1);
    }
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

    valueMapping[result] = std::move(resultT);

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

    valueMapping[result] = std::move(resultT);

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

    assert(op.method() == "SUM");

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
    float *output = (float *)resultT->data();

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
    } else if (op.quant() == "BF16") {
      auto tensor_bf16 = std::make_unique<std::vector<bfloat16> >(resultT->size());
      FloatToBFloat16(resultT->data(), tensor_bf16->data(), resultT->size()); // with rounding
      BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
    }

    valueMapping[result] = std::move(resultT);

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

    auto input_type = op.input()->getType().cast<TensorType>();
    std::vector<int64_t> i_s(input_type.getShape());
    auto output_type = op.output()->getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());
    auto i_size = std::accumulate(std::begin(i_s), std::end(i_s), 1, std::multiplies<>());
    auto o_size = std::accumulate(std::begin(o_s), std::end(o_s), 1, std::multiplies<>());
    assert((i_size == o_size) && "input size not equal to output size");

    // use copy for now
    resultT.get()->assign(opdT[0]->begin(), opdT[0]->end());

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

    if (op.quant() == "INT8") {
      float *input = (float *)opdT[0]->data();
      float *output = (float *)resultT->data();
      float threshold = op.threshold().getValue().convertToFloat();
      LLVM_DEBUG(llvm::errs() << "  quantization, threshold = "
                   << std::to_string(threshold) << "\n";);
      for (int i = 0; i < size; ++i) {
        output[i] = (float)quantizeNeuron(input[i], threshold);
      }
    } else if (op.quant() == "BF16") {
      resultT->assign(opdT[0]->begin(), opdT[0]->end());
    } else {
      assert(0);
    }

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

    if (op.quant() == "INT8") {
      float *input = (float *)opdT[0]->data();
      float *output = (float *)resultT->data();
      float threshold = op.threshold().getValue().convertToFloat();
      LLVM_DEBUG(llvm::errs() << "  quantization, threshold = "
                   << std::to_string(threshold) << "\n";);
      for (int i = 0; i < size; ++i) {
        output[i] = dequantizeNeuron((int8_t)input[i], threshold);
      }
    } else if (op.quant() == "BF16") {
      resultT->assign(opdT[0]->begin(), opdT[0]->end());
    } else {
      assert(0);
    }

    valueMapping[result] = std::move(resultT);

    return success();
  }

  if (auto op = dyn_cast<ConstantOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "ConstantOp" << "\n";);
    //op.dump();
    // we don't use this Op anymore
    assert(0);

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
      if (auto loadWeightOp = dyn_cast<tpu::LoadWeightOp>(op)) {
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
