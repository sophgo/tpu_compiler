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
#include <algorithm>

#define DEBUG_TYPE "interpreter"

using namespace std;

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
    auto filename_tensorfile = llvm::sys::path::stem(filename).str() + ".npz";
    weightFile_ = openInputTensorFile(filename_tensorfile);

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
      tensor = std::move(weightFile_->readTensor<float>(tensor_name, type));
    } else if (type.getElementType().isInteger(8)) {
      // TODO: we still save int8 weight as fp32 for now
      assert(0);
    } else if (type.getElementType().isBF16()) {
      auto tensor_bf16 = weightFile_->readTensor<bfloat16>(tensor_name, type);

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

    valueMapping[result] = std::move(resultT);

    return success();
  }
  if (auto op = dyn_cast<tpu::Conv2DOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "Conv2DOp" << "\n";);
    auto opdT = getOperandTensors(opInst, valueMapping);
    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  name " << op.name() << "\n";);
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() == 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto resultT = std::make_unique<std::vector<float> >(size);

    bool with_bias, do_relu;
    int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, ph, pw, dh, dw;
    getConv2DOpParam<tpu::Conv2DOp>(op, n, ic, ih, iw, oc, oh, ow, g,
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

    int mkldnn_ret = mkldnn_conv(input->data(), filter->data(),
        bias?bias->data():nullptr, resultT->data(),
        n, ic, ih, iw, oc, oh, ow, kh, kw, sh, sw, ph, pw, g);
    assert(mkldnn_ret == 0);
    //dump_data_float_abs("mkldnn_output", mkldnn_output, n, oc, oh, ow);

    if (do_relu) {
      my_relu(resultT->data(), resultT->data(), n, oc, oh, ow, 0.0f);
    }

    // rshift and saturate on output
    if (op.quant() == "INT8") {
      assert(rshift);
      for (int i = 0; i < size; ++i) {
        resultT->at(i) = (float)applyRShiftAndSaturateInt8(resultT->at(i),
            (uint32_t)rshift->at(0));
      }
    } else if (op.quant() == "INT8_PER_CHANNEL") {
      assert(rshift);
      int isz = size / oc;
      for (int i = 0; i < oc; ++i) {
        for (int j = 0; j < isz; ++j) {
          resultT->at(i * isz + j) = (float)applyRShiftAndSaturateInt8(
              resultT->at(i * isz + j), (uint32_t)rshift->at(i));
        }
      }
    } else if (op.quant() == "INT8_MULTIPLIER") {
      assert(multiplier);
      int isz = size / oc;
      for (int i = 0; i < oc; ++i) {
        for (int j = 0; j < isz; ++j) {
          resultT->at(i * isz + j) =
              (float)applyMultiplierAndRShiftAndSaturateInt8(
                  resultT->at(i * isz + j),
                  rshift->at(i), multiplier->at(i), true);
        }
      }
    } else if (op.quant() == "BF16") {
      auto tensor_bf16 = std::make_unique<std::vector<bfloat16> >(resultT->size());
      FloatToBFloat16(resultT->data(), tensor_bf16->data(), resultT->size()); // with rounding
      BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
    } else if (op.quant() == "NONE") {
    } else {
      assert(0);
    }

    // apply eltwise if needed
    if (op.fused_eltwise_method() == "SUM") {
      assert(eltwise_input);

      if (op.quant() == "INT8" || op.quant() == "INT8_PER_CHANNEL"
          ||op.quant() == "INT8_MULTIPLIER") {
        // fused eltwise support 2 inputs only
        std::vector<float> eltwise_threshold_x(2);
        eltwise_threshold_x[0] = getPreviousOpThreshold(op, op.getNumOperands() - 1);
        eltwise_threshold_x[1] = op.threshold_y_before_eltwise().getValue().convertToFloat();
        float eltwise_threshold_y = op.threshold_y().getValue().convertToFloat();

        // determine rshift for all inputs, and multiplier for each input
        // use max threshold_x to find rshift first
        uint32_t eltwise_rshift;
        std::vector<float> eltwise_multiplier(2);
        float max_threshold_x = *std::max_element(
            std::begin(eltwise_threshold_x), std::end(eltwise_threshold_x));
        eltwise_rshift = findRShiftAndMultiplierFromQScale(max_threshold_x / eltwise_threshold_y);
        LLVM_DEBUG(llvm::errs() << "  threshold_y = " << std::to_string(eltwise_threshold_y)
                                << ", rshift = " << std::to_string(eltwise_rshift) << "\n");
        for (int index = 0; index < 2; ++index) {
          float qscale = eltwise_threshold_x[index] / eltwise_threshold_y;
          eltwise_multiplier[index] = (int8_t)findMultiplierFromQScaleAndRShift(qscale, eltwise_rshift);
          LLVM_DEBUG(llvm::errs()
              << "  threshold_x[" << index << "] = " << std::to_string(eltwise_threshold_x[index])
              << ", multiplier["  << index << "] = " << std::to_string(eltwise_multiplier[index])
              << "\n");
        }

        // make copy of inputs
        std::vector<std::shared_ptr<std::vector<float> > > input_copy(2);
        for (int index = 0; index < 2; ++index) {
          input_copy[index] = make_shared<std::vector<float> >();
        }
        input_copy[0]->assign(eltwise_input->begin(), eltwise_input->end());
        input_copy[1]->assign(resultT->begin(), resultT->end());

        // apply multiplier
        for (int index = 0; index < 2; ++index) {
          for (size_t i = 0; i < input_copy[index]->size(); ++i) {
            (*input_copy[index])[i] = (*input_copy[index])[i] *eltwise_multiplier[index];
          }
        }

        my_eltwise(input_copy[0]->data(), input_copy[1]->data(),
                   resultT->data(), n, oc, oh, ow, 1);

        for (int i = 0; i < size; ++i) {
          resultT->at(i) = (float)applyRShiftAndSaturateInt8(resultT->at(i),
              (uint32_t)eltwise_rshift);
        }

      } else if (op.quant() == "BF16") {
        my_eltwise(eltwise_input->data(), resultT->data(), resultT->data(),
            n, oc, oh, ow, 1);
        auto tensor_bf16 = std::make_unique<std::vector<bfloat16> >(resultT->size());
        // with rounding
        FloatToBFloat16(resultT->data(), tensor_bf16->data(), resultT->size());
        BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());

      } else if (op.quant() == "NONE") {
        my_eltwise(eltwise_input->data(), resultT->data(), resultT->data(),
            n, oc, oh, ow, 1);
      }

      if (op.fused_activation_function_after_eltwise() == "RELU") {
        my_relu(resultT->data(), resultT->data(), n, oc, oh, ow, 0.0f);
      } else {
        assert(op.fused_activation_function_after_eltwise() == "NONE");
      }

    } else {
      assert(eltwise_input == nullptr);
      assert(op.fused_eltwise_method() == "NONE");
    }

    valueMapping[result] = std::move(resultT);

    return success();
  }
    if (auto op = dyn_cast<tpu::DeConv2DOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "DeConv2DOp" << "\n";);
    auto opdT = getOperandTensors(opInst, valueMapping);
    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  name " << op.name() << "\n";);
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() == 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto resultT = std::make_unique<std::vector<float> >(size);

    bool with_bias;
    int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, ph, pw, dh, dw;
    getDeConv2DOpParam(op, n, ic, ih, iw, oc, oh, ow, g,
                     kh, kw, sh, sw, ph, pw, dh, dw, with_bias);

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
    getDeConv2DOpVariadicTensors(op, opdT, bias, rshift, multiplier,
        per_channel_info, eltwise_input);

    int mkldnn_ret = mkldnn_deconv(input->data(), filter->data(),
        bias?bias->data():nullptr, resultT->data(),
        n, ic, ih, iw, oc, oh, ow, kh, kw, sh, sw, ph, pw, g);
    assert(mkldnn_ret == 0);
    //dump_data_float_abs("mkldnn_output", mkldnn_output, n, oc, oh, ow);


    // rshift and saturate on output
    if (op.quant() == "INT8") {
      assert(rshift);
      for (int i = 0; i < size; ++i) {
        resultT->at(i) = (float)applyRShiftAndSaturateInt8(resultT->at(i),
            (uint32_t)rshift->at(0));
      }
    } else if (op.quant() == "INT8_PER_CHANNEL") {
      assert(rshift);
      int isz = size / oc;
      for (int i = 0; i < oc; ++i) {
        for (int j = 0; j < isz; ++j) {
          resultT->at(i * isz + j) = (float)applyRShiftAndSaturateInt8(
              resultT->at(i * isz + j), (uint32_t)rshift->at(i));
        }
      }
    } else if (op.quant() == "INT8_MULTIPLIER") {
      assert(multiplier);
      int isz = size / oc;
      for (int i = 0; i < oc; ++i) {
        for (int j = 0; j < isz; ++j) {
          resultT->at(i * isz + j) =
              (float)applyMultiplierAndRShiftAndSaturateInt8(
                  resultT->at(i * isz + j),
                  rshift->at(i), multiplier->at(i), true);
        }
      }
    } else if (op.quant() == "BF16") {
      auto tensor_bf16 = std::make_unique<std::vector<bfloat16> >(resultT->size());
      FloatToBFloat16(resultT->data(), tensor_bf16->data(), resultT->size()); // with rounding
      BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
    } else if (op.quant() == "NONE") {
    } else {
      assert(0);
    }
    assert(eltwise_input == nullptr);
    assert(op.fused_eltwise_method() == "NONE");

    valueMapping[result] = std::move(resultT);

    return success();
  }
  if (auto op = dyn_cast<tpu::Pool2DOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "Pool2DOp" << "\n";);
    auto opdT = getOperandTensors(opInst, valueMapping);
    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump();
               llvm::errs() << "\n";);
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape),
                                1, std::multiplies<>());
    auto resultT = std::make_unique<std::vector<float> >(size);

    bool is_average_pool, do_relu;
    int n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr;
    getPool2DOpParam(op, is_average_pool, n, c, ih, iw, oh, ow,
                     kh, kw, sh, sw, pt, pb, pl, pr, do_relu);

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
        n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, is_average_pool);
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


    if (do_relu) {
      my_relu(resultT->data(), resultT->data(), 1, 1, 1, n, 0.0f);
    }

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

    int ret = 0 ;
    if(i_s.size() == 4){

    n = i_s[0];
    c = i_s[1];
    h = i_s[2];
    w = i_s[3];

    float *input = (float *)opdT[0]->data();
    float *output = (float *)resultT.get()->data();
    ret = my_relu(input, output, n, c, h, w, negative_slope);

    }else if(i_s.size() == 2){ //for (h w) shape relu

      n = 1;
      c = 1;
      h = i_s[0];
      w = i_s[1];
      float *input = (float *)opdT[0]->data();
      float *output = (float *)resultT.get()->data();
      ret = my_relu(input, output, n, c, h, w, negative_slope);

    }
    assert(ret == 0);

    valueMapping[result] = std::move(resultT);

    return success();
  }
  if (auto op = dyn_cast<tpu::PReluOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "PReluOp" << "\n";);
    auto opdT = getOperandTensors(opInst, valueMapping);
    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  name " << op.name() << "\n"
                            << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    std::vector<int64_t> shape =
        result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1,
                                std::multiplies<>());
    auto resultT = std::make_unique<std::vector<float>>(size);

    // ---- checked ----
    int n, c, h, w;
    float *negative_slope = opdT[1]->data();
    
    auto input_type = op.x()->getType().cast<TensorType>();
    std::vector<int64_t> i_s(input_type.getShape());
    auto output_type = op.y()->getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());
    assert((i_s == o_s) && "input shape not equal to output shape");
    assert((i_s.size() == 4) && "PRelu support shape size of 4 now.");
    
    n = i_s[0];
    c = i_s[1];
    h = i_s[2];
    w = i_s[3];
    float *input = (float *)opdT[0]->data();
    float *output = (float *)resultT.get()->data();
    int ret = my_prelu(input, output, n, c, h, w, negative_slope);
    assert(ret == 0);

    std::shared_ptr<std::vector<float> > rshift = nullptr;
    std::shared_ptr<std::vector<float> > multiplier = nullptr;

    getPReluOpVariadicTensors(op, opdT, rshift, multiplier);

    // rshift and saturate on output
    if (op.quant() == "INT8" || op.quant() == "INT8_PER_CHANNEL") {
      assert(rshift);
      for (int i = 0; i < size; ++i) {
        resultT->at(i) = (float)applyRShiftAndSaturateInt8(resultT->at(i),
            (uint32_t)rshift->at(0));
      }
    } else if (op.quant() == "INT8_MULTIPLIER") {
      assert(multiplier);
      for (int i = 0; i < size; ++i) {
        resultT->at(i) = (float)applyMultiplierAndRShiftAndSaturateInt8(
            resultT->at(i), (uint32_t)rshift->at(0), multiplier->at(0), true);
      }
    } else if (op.quant() == "BF16") {
      // auto tensor_bf16 = std::make_unique<std::vector<bfloat16> >(resultT->size());
      // FloatToBFloat16(resultT->data(), tensor_bf16->data(), resultT->size()); // with rounding
      // BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
    } else if (op.quant() == "NONE") {
    } else {
      assert(0);
    }

    valueMapping[result] = std::move(resultT);

    return success();
  }
  if (auto op = dyn_cast<tpu::SigmoidOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "SigmoidOp"
                            << "\n";);
    auto opdT = getOperandTensors(opInst, valueMapping);
    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump();
               llvm::errs() << "\n";);
    std::vector<int64_t> shape =
        result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1,
                                std::multiplies<>());
    auto resultT = std::make_unique<std::vector<float>>(size);
    auto input_type = op.input()->getType().cast<TensorType>();
    std::vector<int64_t> i_s(input_type.getShape());
    auto output_type = op.output()->getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());
    assert((i_s == o_s) && "input shape not equal to output shape");
    int n, c, h, w;
    n = i_s[0];
    c = i_s[1];
    h = i_s[2];
    w = i_s[3];
    float *input = (float *)opdT[0]->data();
    float *output = (float *)resultT.get()->data();
    int ret;
    if (op.quant() == "INT8"){
      std::vector<int> data(256, 0);
      float threshold_x = getPreviousOpThreshold(op);
      float threshold_y = getOpThreshold(op);

      assert(threshold_x != 0.0);
      for (int idx = 0; idx < 256; ++idx) {
        char lutInput = static_cast<char>(idx);
        float index = -lutInput * threshold_x / 128.0;
        float lutOutput = 1.0 / (1 + std::exp(index)) * 128.0 / threshold_y;
        int lutOutputI32 = std::floor(lutOutput + 0.5);
        lutOutputI32 = (lutOutputI32 > 127)
                           ? 127
                           : (lutOutputI32 < -128) ? -128 : lutOutputI32;
        data[idx] = lutOutputI32;
      }
      for (int i = 0; i < size; ++i) {
        output[i] = data[(unsigned char)input[i]];
      }
      ret = 0;
    } else {
      ret = my_sigmoid(input, output, n, c, h, w);
    }
    assert(ret == 0);
    valueMapping[result] = std::move(resultT);
    return success();
  }
  if (auto op = dyn_cast<tpu::DummyDataOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "DummyDataOp"
                            << "\n";);
    auto opdT = getOperandTensors(opInst, valueMapping);
    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump();
               llvm::errs() << "\n";);
    std::vector<int64_t> shape =
        result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1,
                                std::multiplies<>());
    auto resultT = std::make_unique<std::vector<float>>(size);
    valueMapping[result] = std::move(resultT);

    return success();
  }

  if (auto op = dyn_cast<tpu::TanHOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "TanHOp" << "\n";);
    auto opdT = getOperandTensors(opInst, valueMapping);
    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto resultT = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here
    int n, c, h, w;
    //float negative_slope = op.negative_slope().convertToFloat();
    auto input_type = op.x()->getType().cast<TensorType>();
    std::vector<int64_t> i_s(input_type.getShape());
    auto output_type = op.y()->getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());
    assert((i_s == o_s) && "input shape not equal to output shape");
    n = i_s[0];
    c = i_s[1];
    h = i_s[2];
    w = i_s[3];
    float *_input = (float *)opdT[0]->data();
    float *input;
    float *output = (float *)resultT.get()->data();

    auto type = result->getType().cast<TensorType>();
    input = _input;
    if (type.getElementType().isBF16()) {
      input = output;
      // do dequantization
      float threshold_x = getPreviousOpThreshold(op);
      //float threshold_x = 8.0; //<! FIXME: not harcode here
      LLVM_DEBUG(llvm::errs() << "  tanh dequantize, threshold_x = "
                              << std::to_string(threshold_x) << "\n";);
      // FIXME: find value by calibration
      // dirty output
      for (size_t i = 0; i < opdT[0]->size(); ++i) {
        output[i] = input[i];
        if (output[i] > threshold_x) {
          output[i] = threshold_x;
        }
        else if(output[i] < -1.0 * threshold_x) {
          output[i] = -1.0 * threshold_x;
        }
      }
    }

    int ret = my_tanh(input, output, n, c, h, w);
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

    int n, c, h, w;
    auto input_type = op.x()->getType().cast<TensorType>();
    std::vector<int64_t> i_s(input_type.getShape());
    auto output_type = op.y()->getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());
    assert((i_s == o_s) && "input shape not equal to output shape");

    assert((i_s.size() == 4 || i_s.size() == 2) &&
           "BatchNorm support shape size of 4 or 2 now." );

    n = i_s[0];
    c = i_s[1];
    h = (i_s.size() == 2) ? 1 : i_s[2];
    w = (i_s.size() == 2) ? 1 : i_s[3];

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
    // mode 0: load scale from wegiht
    // mode 1: load scale from input2
    int mode = 1;
    // check if scale second is load weight op
    if(auto weight_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
        op.getOperand(1)->getDefiningOp())){
        mode = 0;
    }
    LLVM_DEBUG(llvm::errs() << "ScaleOp" << "\n";);
    auto opdT = getOperandTensors(opInst, valueMapping);
    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto resultT = std::make_unique<std::vector<float> >(size);
    uint32_t multiplier_prod;
    int n, c, h, w;
    auto input_type = op.x()->getType().cast<TensorType>();
    std::vector<int64_t> i_s(input_type.getShape());
    auto output_type = op.y()->getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());
    assert((i_s == o_s) && "input shape not equal to output shape");

    assert((i_s.size() == 4 || i_s.size() == 2) &&
           "BatchNorm support shape size of 4 or 2 now." );

    n = i_s[0];
    c = i_s[1];
    h = (i_s.size() == 2) ? 1 : i_s[2];
    w = (i_s.size() == 2) ? 1 : i_s[3];

    float *input = (float *)opdT[0]->data();
    float *scale = (float *)opdT[1]->data();
    float *output = (float *)resultT.get()->data();
    float *bias = nullptr;
    uint32_t rshift;
    if (opdT.size() > 2) {
      assert(opdT.size() == 3);
      bias = (float *)opdT[2]->data();
    }

    if (op.quant() == "INT8") {
      if (mode == 1) {
        assert(opdT.size() == 2);
        std::vector<float> threshold_x(2);
        float threshold_y;

        for (int index = 0; index < 2; ++index) {
          // get threshold_x
          threshold_x[index] = getPreviousOpThreshold(op, index);
        }
        // get threshold_y
        threshold_y = op.threshold_y().getValue().convertToFloat();
        // determine rshift for all inputs, and multiplier for each input
        // use max threshold_x to find rshift first
        float threshold_prod = std::accumulate(
            threshold_x.begin(), threshold_x.end(), 1.0, std::multiplies<>());
        float qscale = threshold_prod / threshold_y / 127.0;
        rshift = findRShiftAndMultiplierFromQScale(qscale, &multiplier_prod,
                                                   true, 255);
      } else {
        auto weight_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
            op.getOperand(1)->getDefiningOp());
        float threshold_x = getPreviousOpThreshold(op);
        float threshold_y = op.threshold_y().getValue().convertToFloat();
        rshift = 0;

      }
    }
    int ret = my_scale(input, scale, bias, output, n, c, h, w);
    assert(ret == 0);

    // rshift and saturate on output
    if (op.quant() == "INT8") {
      // assert(rshift);
      if(mode == 0){
        for (int i = 0; i < size; ++i) {
          output[i] =
              (float)applyRShiftAndSaturateInt8(output[i], (uint32_t)rshift);
        }
      } else {
        for (int i = 0; i < size; ++i) {
          output[i] = (float)applyMultiplierAndRShiftAndSaturateInt8(
              output[i], rshift, multiplier_prod, false);
        }
      }
    }
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
        std::copy(src_vec.begin(), src_vec.end(),
                  back_inserter(input_copy[index]));
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
    uint32_t multiplier_prod;
    if (op.quant() == "INT8") {
      if (op.method() == "SUM" || op.method() == "MAX") {
        // determine rshift for all inputs, and multiplier for each input
        // use max threshold_x to find rshift first
        float max_threshold_x =
            *std::max_element(std::begin(threshold_x), std::end(threshold_x));
        rshift =
            findRShiftAndMultiplierFromQScale(max_threshold_x / threshold_y);
        LLVM_DEBUG(llvm::errs()
                   << "  threshold_y = " << std::to_string(threshold_y)
                   << ", rshift = " << std::to_string(rshift) << "\n");
        for (int index = 0; index < MAX_ELTWISE_INPUT; ++index) {
          float qscale = threshold_x[index] / threshold_y;
          multiplier[index] =
              (int8_t)findMultiplierFromQScaleAndRShift(qscale, rshift);
          LLVM_DEBUG(llvm::errs()
                     << "  threshold_x[" << index
                     << "] = " << std::to_string(threshold_x[index])
                     << ", multiplier[" << index
                     << "] = " << std::to_string(multiplier[index]) << "\n");
        }
        // apply multiplier
        for (int index = 0; index < MAX_ELTWISE_INPUT; ++index) {
          for (size_t i = 0; i < opdT[index]->size(); ++i) {
            input[index][i] = input[index][i] * multiplier[index];
          }
        }
      } else if (op.method() == "PROD") {
        float threshold_prod = std::accumulate(
            threshold_x.begin(), threshold_x.end(), 1.0, std::multiplies<>());
        float qscale = threshold_prod / threshold_y / 127.0;
        rshift = findRShiftAndMultiplierFromQScale(qscale, &multiplier_prod, true, 255);
      } else {
        assert(0); // not support
      }
    }

    int ret;
    if (op.method() == "SUM") {
      ret = my_eltwise(input[0], input[1], output, n, c, h, w, 1);
    } else if (op.method() == "PROD") {
      ret = my_eltwise(input[0], input[1], output, n, c, h, w, 0);
    } else if (op.method() == "MAX") {
      ret = my_eltwise(input[0], input[1], output, n, c, h, w, 2);
    }
    assert(ret == 0);
    if (op.fused_activation_function() == "NONE") {
    } else if (op.fused_activation_function() == "RELU") {
      my_relu(output, output, n, c, h, w, 0.0f);
    } else {
      assert(0);
    }

    // rshift and saturate on output
    if (op.quant() == "INT8") {
      // assert(rshift);
      if (op.method() == "PROD") {
        for (int i = 0; i < size; ++i) {
          output[i] = (float)applyMultiplierAndRShiftAndSaturateInt8(
              output[i], rshift, multiplier_prod, true);
        }
      } else {
        for (int i = 0; i < size; ++i) {
          output[i] =
              (float)applyRShiftAndSaturateInt8(output[i], (uint32_t)rshift);
        }
      }
    } else if (op.quant() == "BF16") {
      auto tensor_bf16 =
          std::make_unique<std::vector<bfloat16>>(resultT->size());
      FloatToBFloat16(resultT->data(), tensor_bf16->data(),
                      resultT->size()); // with rounding
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
  if (auto op = dyn_cast<tpu::CropOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "CropOp"
                            << "\n";);

    auto opdT = getOperandTensors(opInst, valueMapping);
    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump();
               llvm::errs() << "\n";);
    std::vector<int64_t> output_shape =
        result->getType().cast<TensorType>().getShape();
    auto size = std::accumulate(std::begin(output_shape),
                                std::end(output_shape), 1, std::multiplies<>());
    auto resultT = std::make_unique<std::vector<float>>(size);
    uint32_t bottom_num = opdT.size();
    assert(bottom_num >= 2 && "bottom num is 0 or 1");

    auto crop_start_axis = op.axis();
    int crop_offset_n = op.crop_offset_n().getValue().getLimitedValue();
    int crop_offset_c = op.crop_offset_c().getValue().getLimitedValue();
    int crop_offset_h = op.crop_offset_h().getValue().getLimitedValue();
    int crop_offset_w = op.crop_offset_w().getValue().getLimitedValue();
    vector<int> crop_offset = {crop_offset_n, crop_offset_c, crop_offset_h, crop_offset_w};
    LLVM_DEBUG (llvm::errs() << crop_offset_n << ", " << crop_offset_c << ", "
               << crop_offset_h << "," << crop_offset_w;);

    auto input1 = op.input1()->getType().cast<TensorType>();
    std::vector<int64_t> input_shape1(input1.getShape());
    auto input2 = op.input2()->getType().cast<TensorType>();
    std::vector<int64_t> input_shape2(input2.getShape());

    float *input = (float *)opdT[0]->data();
    float *output = (float *)resultT.get()->data();
    vector<int >indices(size, 0);
    my_crop(input, output, input_shape1.data(), input_shape2.data(),
            output_shape.data(), 0, crop_offset.data(), indices.data());
    valueMapping[result] = std::move(resultT);
    return success();
  }
  if (auto op = dyn_cast<tpu::ConcatOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "ConcatOp" << "\n";);
    auto opdT = getOperandTensors(opInst, valueMapping);
    auto result = op.res();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto resultT = std::make_unique<std::vector<float> >(size);
    size_t bottom_num = opdT.size();
    uint32_t n, c, h, w;
    auto concat_axis = op.dimension();
    auto tmp_resultT = std::make_unique<std::vector<float> >(0);
    int shift_idx_c=0;
    int shift_idx_h=0;

    assert(bottom_num >= 2 && "bottom num is 0 or 1");
    assert(shape.size() <= 4);
    LLVM_DEBUG(llvm::errs() << "concat_axis =" << concat_axis << "\n";);

    std::vector<float *> input(bottom_num);
    for (size_t index = 0; index < bottom_num; ++index) {
      input[index] = (float *)opdT[index]->data();
    }
    //float *output = (float *)resultT->data();

    // for INT8, get threshold_x and make copy of input first
    std::vector<std::vector<float> >input_copy(bottom_num);
    std::vector<float> threshold_x(bottom_num);
    float threshold_y;
    if (op.quant() == "INT8") {
      for (size_t index = 0; index < bottom_num; ++index) {
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
    std::vector<int8_t> multiplier(bottom_num);
    if (op.quant() == "INT8") {
      // determine rshift for all inputs, and multiplier for each input
      // use max threshold_x to find rshift first
      float max_threshold_x = *std::max_element(
          std::begin(threshold_x), std::end(threshold_x));
      rshift = findRShiftAndMultiplierFromQScale(max_threshold_x / threshold_y);
      LLVM_DEBUG(llvm::errs() << "  threshold_y = " << std::to_string(threshold_y)
                              << ", rshift = " << std::to_string(rshift) << "\n");
      for (size_t index = 0; index < bottom_num; ++index) {
        float qscale = threshold_x[index] / threshold_y;
        multiplier[index] = (int8_t)findMultiplierFromQScaleAndRShift(qscale, rshift);
        LLVM_DEBUG(llvm::errs()
            << "  threshold_x[" << index << "] = " << std::to_string(threshold_x[index])
            << ", multiplier["  << index << "] = " << std::to_string(multiplier[index])
            << "\n");
      }
    }

    // apply multiplier & saturate
    if (op.quant() == "INT8") {
      for (size_t index = 0; index < bottom_num; ++index) {
        for (size_t i = 0; i < opdT[index]->size(); ++i) {
          input[index][i] = input[index][i] * multiplier[index];
          input[index][i] = (float)applyRShiftAndSaturateInt8(input[index][i], (uint32_t)rshift);
        }
      }
    }
    for (uint32_t i = 0; i < bottom_num; i++) {
      std::vector<int64_t> shape =  op.getOperand(i)->getType().cast<TensorType>().getShape();
      n = shape[0];
      c = shape[1];
      h = shape[2];
      w = shape[3];

      LLVM_DEBUG(llvm::errs() << "shape n:" << n << " c:" << c << " h:"<< h << " w:"<< w <<"\n";);
      LLVM_DEBUG(llvm::errs() << "bottom num:" << opdT.size() << "\n";);
      LLVM_DEBUG(llvm::errs() << "data size:" << opdT[i]->size() << "\n";);

      auto *input_data = input[i];

      if (concat_axis == 0) {
        tmp_resultT.get()->insert(tmp_resultT.get()->end(), opdT[i]->begin(), opdT[i]->end());
      }
      else if (concat_axis == 1) {
        for (uint32_t idx_n = 0; idx_n < n; idx_n++) {
          auto shapeT = std::make_unique<std::vector<float> >(c * h * w);
          int insert_offset = ((idx_n + 1) * shift_idx_c  + idx_n * c) * h * w;
          shapeT.get()->assign(&input_data[idx_n * c * h * w], &input_data[(idx_n + 1) * c * h * w]);
          tmp_resultT.get()->insert(tmp_resultT.get()->begin() + insert_offset, shapeT->begin(), shapeT->end());
        }
        shift_idx_c += c;
      } else if (concat_axis == 2) {
        for (uint32_t idx_n = 0; idx_n < n; idx_n++) {
          for (uint32_t idx_c = 0; idx_c < c ;idx_c++) {
            auto shapeT = std::make_unique<std::vector<float> >(h * w);
            int insert_offset = (idx_n * c * h + (idx_c + 1) * shift_idx_h + idx_c * h) * w;
            shapeT.get()->assign(&input_data[(idx_n * c + idx_c) * h * w], &input_data[(idx_n * c + (idx_c + 1)) * h * w]);
            tmp_resultT.get()->insert(tmp_resultT.get()->begin() + insert_offset, shapeT->begin(), shapeT->end());
          }
        }
        shift_idx_h += h;
      } else
        assert(0 && "not support concat_axis >=3 now\n");
    }

    resultT.get()->assign(tmp_resultT.get()->begin(), tmp_resultT.get()->end());
    valueMapping[result] = std::move(resultT);

    return success();
  }
  if (auto op = dyn_cast<tpu::UpsampleOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "UpsampleOp" << "\n";);
    auto opdT = getOperandTensors(opInst, valueMapping);
    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto resultT = std::make_unique<std::vector<float> >(size);

    int n, c, ih, iw, oh, ow, scale;
    auto input_type = op.x()->getType().cast<TensorType>();
    std::vector<int64_t> i_s(input_type.getShape());
    auto output_type = op.y()->getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());
    n = i_s[0];
    c = i_s[1];
    ih = i_s[2];
    iw = i_s[3];
    scale = op.scale().getLimitedValue();
    assert(o_s[0] == n);
    assert(o_s[1] == c);
    oh = o_s[2];
    ow = o_s[3];
    assert(oh ==  ih * scale);
    assert(ow ==  iw * scale);
    float *input = (float *)opdT[0]->data();
    float *output = (float *)resultT.get()->data();
    int ret = my_upsample(input, output, n, c, ih, iw, scale);
    assert(ret == 0);

    valueMapping[result] = std::move(resultT);

    return success();
  }

  if (auto op = dyn_cast<tpu::SliceOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "SliceOp" << "\n";);
    auto opdT = getOperandTensors(opInst, valueMapping);
    auto results = op.getResults();
    int axis = op.axis().getValue().getLimitedValue();
    std::vector<int64_t> i_s = op.getOperand()->getType().cast<TensorType>().getShape();

    float *input = (float *)opdT[0]->data();
    for (uint32_t i = 0; i < results.size(); i++) {
      auto result = results[i];
      LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
      std::vector<int64_t> o_s = result->getType().cast<TensorType>().getShape();
      assert(o_s.size() <= 4);
      auto size = std::accumulate(std::begin(o_s), std::end(o_s), 1, std::multiplies<>());
      auto resultT = std::make_unique<std::vector<float> >(size);

      float *output = (float *)resultT.get()->data();
      int ret = my_slice(input, output, axis, i_s, o_s);
      assert(ret == 0);

      valueMapping[result] = std::move(resultT);
      input += size;
    }

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
      llvm::errs() << i_s[0] << ", " << i_s[1] << ", " << i_s[2] << ", " << i_s[3] << "\n";
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
    // do nothing
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
  // run blocks one by one.
  for (Block &bb : func.getBlocks()) {
    if (failed(runBlock(bb)))
      return failure();
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

static bool isValidTpuOp(Operation &op)
{
  return (!isa<tpu::LoadWeightOp>(op) && !isa<tpu::LoadFileOp>(op) &&
          op.getName().getDialect().str() == "tpu");
}

LogicalResult runTpuModule(ModuleOp m,
    std::vector<int64_t> input_shape, std::vector<float> &input_vec,
    std::map<std::string, std::vector<float> > *results,
    std::map<std::string, std::vector<int64_t> > *shapeMap,
    std::map<std::string, std::vector<float> > *allTensorMap) {
  for (FuncOp function : m.getOps<FuncOp>()) {
    for (Block &bb : function.getBlocks()) {
      for (auto &op : bb) {
        if (!isValidTpuOp(op)) {
          continue;
        }
        // TODO: Only support one output tesor for now.
        auto result = op.getResult(0);
        std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
        (*shapeMap)[getOpName(&op).str()] = shape;
      }
    }
  }

  return ModuleInterpreter::runModule<>(m, input_shape, input_vec,
                                        results, allTensorMap);
}

} // namespace mlir
