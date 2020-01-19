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
#include "mlir/Dialect/TPU/CpuLayer_DetectionOutput.h"
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
//#include <google/protobuf/stubs/common.h>
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
        n, ic, ih, iw, oc, oh, ow, kh, kw, sh, sw, dh,dw, ph, pw, g);
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
  if (auto op = dyn_cast<tpu::PermuteOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "PermuteOp" << "\n";);

    auto opdT = getOperandTensors(opInst, valueMapping);
    auto result = op.getResult();

    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() == 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto resultT = std::make_unique<std::vector<float> >(size);


    int in, ic, ih, iw,on,oc,oh,ow,order0,order1,order2,order3;
    auto input_type = op.input()->getType().cast<TensorType>();
    std::vector<int64_t> i_s(input_type.getShape());
    auto output_type = op.output()->getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());

    //Dirty need to improve!!
    order0 = op.order0().getLimitedValue();
    order1 = op.order1().getLimitedValue();
    order2 = op.order2().getLimitedValue();
    order3 = op.order3().getLimitedValue();

    int ret = 0 ;

    in = i_s[0];
    ic = i_s[1];
    ih = i_s[2];
    iw = i_s[3];

    on = o_s[0];
    oc = o_s[1];
    oh = o_s[2];
    ow = o_s[3];


    //As long as there is one order which is different from the natural order
    // of the data, we need to permute.(from caffe permute layer source code mark)
    if( in==on && ic==oc && ih==oh && iw==ow ){
      valueMapping[result] = std::move(opdT[0]);
    }else{
      float *input = (float *)opdT[0]->data();
      float *output = (float *)resultT.get()->data();
      ret = my_permute(input,output,shape.size(),in,ic,ih,iw,on,oc,oh,ow,order0,order1,order2,order3);
      assert(ret == 0);
      valueMapping[result] = std::move(resultT);
    }
    return success();
  }

  if (auto op = dyn_cast<tpu::NormalizeOp>(opInst)) {
    /*not the same as ssd Normalize op, here only do normalize , reuse "Scale op" for scale operation */
    LLVM_DEBUG(llvm::errs() << "NormalizeOp" << "\n";);
    auto opdT = getOperandTensors(opInst, valueMapping);
    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() == 4);

    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto resultT = std::make_unique<std::vector<float> >(size);

    bool across_spatial = op.across_spatial();
    //bool channel_shared = op.across_spatial();

    //implement for ssd case first
    assert(!across_spatial);

    int n, c, h, w;
    auto input_type = op.input()->getType().cast<TensorType>();
    std::vector<int64_t> i_s(input_type.getShape());

    n = i_s[0];
    c = i_s[1];
    h = i_s[2];
    w = i_s[3];

    float *input = (float *)opdT[0]->data();
    //float *scale = (float *)opdT[1]->data();
    float *output = (float *)resultT.get()->data();

    int ret = 0 ;
    ret = my_normalize(input,output,across_spatial,n,c,h,w);
    assert(ret == 0);
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

    std::shared_ptr<std::vector<float> > rshift_pos = nullptr;
    std::shared_ptr<std::vector<float> > rshift_neg = nullptr;
    std::shared_ptr<std::vector<float> > multiplier_pos = nullptr;
    std::shared_ptr<std::vector<float> > multiplier_neg = nullptr;


    getPReluOpVariadicTensors(op, opdT, rshift_pos, rshift_neg, multiplier_pos, multiplier_neg);

    float threshold_x;
    float threshold_y;
    if (op.quant() != "NONE"){
      threshold_x = getPreviousOpThreshold(op);
      threshold_y = op.threshold_y().getValue().convertToFloat();
    }

    // rshift and saturate on output
    if (op.quant() == "INT8" || op.quant() == "INT8_PER_CHANNEL") {
      assert(rshift_pos);
      assert(rshift_neg);
      for (int i = 0; i < size; ++i) {
        if (input[i] > 0){
          // resultT->at(i) = (threshold_x / threshold_y) * resultT->at(i);
          resultT->at(i) = (float)applyRShiftAndSaturateInt8(resultT->at(i),
              (uint32_t)rshift_pos->at(0));
        } else {
          resultT->at(i) = (float)applyRShiftAndSaturateInt8(resultT->at(i),
              (uint32_t)rshift_neg->at(0));
        }
      }
    } else if (op.quant() == "INT8_MULTIPLIER") {
      assert(multiplier_pos);
      assert(multiplier_neg);
      for (int i = 0; i < size; ++i) {
        if (input[i] > 0){
          // resultT->at(i) = (threshold_x / threshold_y) * resultT->at(i);
          // resultT->at(i) = (float)applyMultiplierAndRShiftAndSaturateInt8(
          //     resultT->at(i), (uint32_t)rshift_pos->at(0), multiplier_pos->at(0), true);
          resultT->at(i) = (float)applyMultiplierAndRShiftAndSaturateInt8(
              resultT->at(i), (uint32_t)rshift_pos->at(0), multiplier_pos->at(0), false);
        } else {
          resultT->at(i) = (float)applyMultiplierAndRShiftAndSaturateInt8(
              resultT->at(i), (uint32_t)rshift_neg->at(0), multiplier_neg->at(0), true);
        }
      }
    } else if (op.quant() == "BF16") {
      assert(0 && "Not support BF16 now.");
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
    // check if scale second is load weight op
    auto sec_blob_weight_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
        op.getOperand(1)->getDefiningOp());

    LLVM_DEBUG(llvm::errs() << "ScaleOp" << "\n";);
    auto opdT = getOperandTensors(opInst, valueMapping);
    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto resultT = std::make_unique<std::vector<float> >(size);

    auto input_type = op.x()->getType().cast<TensorType>();
    std::vector<int64_t> i_s(input_type.getShape());
    auto output_type = op.y()->getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());
    assert((i_s == o_s) && "input shape not equal to output shape");\
    int n, c, h, w;
    n = i_s[0];
    c = i_s[1];
    h = (i_s.size() == 2) ? 1 : i_s[2];
    w = (i_s.size() == 2) ? 1 : i_s[3];
    int oc = o_s[1];

    llvm::errs() << "input shape size : " << "\n";
    llvm::errs() << i_s[0]<<","<<i_s[1]<<","<<i_s[2]<<","<<i_s[3] << "\n";

    uint32_t multiplier_prod;
    float *input = (float*)opdT[0]->data();
    float *scale = (float*)opdT[1]->data();
    std::shared_ptr<std::vector<float>> bias = nullptr;
    if (op.with_bias()) {
      bias = opdT[2];
    }

    auto rshift = std::make_shared<std::vector<float>>(1);
    std::shared_ptr<std::vector<float>> multiplier = nullptr;

    if (op.quant() == "INT8" || op.quant() == "INT8_PER_CHANNEL" ||
        op.quant() == "INT8_MULTIPLIER") {
      // if second input is from blob
      // we need caluate rshift and multiplier,
      // otherwise, read rhisft and multiplier from load weight
      if (sec_blob_weight_op) {
        getScaleOpVariadicTensors(op, opdT, bias, rshift, multiplier);
      }else {
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
        rshift->at(0) = (float)findRShiftAndMultiplierFromQScale(qscale, &multiplier_prod, true,
                                                255);
      }
      }
    int ret;
    if (op.with_bias()) {
      ret =
          my_scale(input, scale, bias->data(), resultT->data(), n, c, h, w);
    }else{
      ret = my_scale(input, scale, nullptr, resultT->data(), n, c, h, w);
    }

    assert(ret == 0);
    // rshift and saturate on output
    if (op.quant() == "INT8") {
      assert(rshift);
        for (int i = 0; i < size; ++i) {
        resultT->at(i) = (float)applyRShiftAndSaturateInt8(
            resultT->at(i), (uint32_t)rshift->at(0));
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
                  resultT->at(i * isz + j), rshift->at(i), multiplier->at(i),
                  true);
        }
      }
    } else if (op.quant() == "BF16") {
      assert("not support now");
    } else if (op.quant() == "NONE") {
    } else {
      assert(0);
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
        rshift = findRShiftAndMultiplierFromQScale(max_threshold_x / threshold_y);
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
    int tmp_w=0;
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

      if(shape.size()==4){


        n = shape[0];
        c = shape[1];
        h = shape[2];
        w = shape[3];


        LLVM_DEBUG(llvm::errs() << "shape n:" << n << " c:" << c << " h:"<< h << " w:"<< w <<"\n";);
        LLVM_DEBUG(llvm::errs() << "bottom num:" << opdT.size() << "\n";);
        LLVM_DEBUG(llvm::errs() << "data size:" << opdT[i]->size() << "\n";);

        auto *input_data = opdT[i]->data();

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
      }else if(shape.size()==2){
          h = shape[0];
          w = shape[1];

          LLVM_DEBUG(llvm::errs() << "shape h:" << h << " w:"<< w <<"\n";);
          LLVM_DEBUG(llvm::errs() << "bottom num:" << opdT.size() << "\n";);
          LLVM_DEBUG(llvm::errs() << "data size:" << opdT[i]->size() << "\n";);
          auto *input_data = opdT[i]->data();
          if (concat_axis == 0) {
            tmp_resultT.get()->insert(tmp_resultT.get()->end(), opdT[i]->begin(), opdT[i]->end());
          }
          else if (concat_axis == 1) {

            for (uint32_t idx_h = 0; idx_h < h; idx_h++) {
              auto shapeT = std::make_unique<std::vector<float> >(w);
              //int insert_offset = ((idx_h + 1) * idx_h) * h * w;
              //int insert_offset = (idx_h  + (idx_h + 1) * shift_idx_h) * (i=0?w:tmp_w);
              int insert_offset = ((idx_h+1)* tmp_w) + idx_h*w;
              shapeT.get()->assign(&input_data[idx_h * w], &input_data[(idx_h + 1) * w]);
              tmp_resultT.get()->insert(tmp_resultT.get()->begin() + insert_offset, shapeT->begin(), shapeT->end());
            }
            tmp_w += w;
          } else
            assert(0 && "not support concat_axis >=2 now\n");
      }else if(shape.size()==3){
        c = shape[0];
        h = shape[1];
        w = shape[2];

        LLVM_DEBUG(llvm::errs() << "shape c:" << c <<"\n";);
        LLVM_DEBUG(llvm::errs() << "shape h:" << h << " w:"<< w <<"\n";);
        LLVM_DEBUG(llvm::errs() << "bottom num:" << opdT.size() << "\n";);
        LLVM_DEBUG(llvm::errs() << "data size:" << opdT[i]->size() << "\n";);

        auto *input_data = input[i];
        if (concat_axis == 0) {
          tmp_resultT.get()->insert(tmp_resultT.get()->end(), opdT[i]->begin(), opdT[i]->end());
        }else if (concat_axis == 2) {

            assert(c==1);
            for (uint32_t idx_h = 0; idx_h < h; idx_h++) {
            auto shapeT = std::make_unique<std::vector<float> >(w);
            int insert_offset = ((idx_h+1)* tmp_w) + idx_h*w;
            shapeT.get()->assign(&input_data[idx_h * w], &input_data[(idx_h + 1) * w]);
            tmp_resultT.get()->insert(tmp_resultT.get()->begin() + insert_offset, shapeT->begin(), shapeT->end());
          }
          tmp_w += w;
        }else {
          assert(0&&"not support shape size =1 and axis = 1 now ");
        }
       }
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
    assert(shape.size() == 2 || shape.size() == 4|| shape.size() == 3);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto resultT = std::make_unique<std::vector<float> >(size);
    float *output = (float *)resultT.get()->data();

    int axis = op.axis().getValue().getLimitedValue();
    int n,c,h,w;
    auto input_type = op.x()->getType().cast<TensorType>();
    std::vector<int64_t> i_s(input_type.getShape());
    auto output_type = op.y()->getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());
    assert((i_s == o_s) && "input shape not equal to output shape");
    float *input = (float *)opdT[0]->data();

    if (shape.size() == 2) {
      n = i_s[0];
      c = i_s[1];

      // do dequantization
      if (0) {
        float threshold_x = getPreviousOpThreshold(op);
        LLVM_DEBUG(llvm::errs() << "  softmax dequantize, threshold_x = "
                                << std::to_string(threshold_x) << "\n";);
        for (size_t i = 0; i < opdT[0]->size(); ++i) {
          input[i] = input[i] * threshold_x / 128.0;
        }
      }

      int ret = my_softmax2D(input, output, n, c);
      assert(ret == 0);
    } else if (shape.size() == 4) {
      int ret = my_softmax4D(input, output, axis, shape);
      assert(ret == 0);
    } else if (shape.size() == 3) {
      c = i_s[0];
      h = i_s[1];
      w = i_s[2];
      //just for axis = 2 now
      assert(axis == 2);
      auto tmp_resultT = std::make_unique<std::vector<float> >(w);

      float *tmp = (float *)tmp_resultT.get()->data();

      for(int ci = 0; ci < c; ci++) {
        for(int hi = 0; hi < h; hi++) {
          for(int wi = 0; wi < w; wi++) {
            tmp[wi] = input[ci * w * h + hi * w + wi];
          }

          int ret = my_softmax2D(tmp, tmp, 1, w);
          assert(ret == 0);
          for(int wi = 0; wi < w; wi++) {
            output[ci * w * h + hi * w + wi] = tmp[wi];
          }
        }  //end for hi
      } //end for ci
    }

    valueMapping[result] = std::move(resultT);
    return success();
  }
  if (auto op = dyn_cast<tpu::DivOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "DivOp" << "\n";);
    auto opdT = getOperandTensors(opInst, valueMapping);
    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() == 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto resultT = std::make_unique<std::vector<float> >(size);
    float eps = 1.0e-5;


    float threshold_y,threshold_x;
    uint32_t multiplier;
    if (op.quant() != "NONE"){

      threshold_y = op.threshold_y().getValue().convertToFloat();
      threshold_x = getPreviousOpThreshold(op);
    }

    float *input = (float *)opdT[0]->data();

    float *output = (float *)resultT->data();    

    if (op.quant() == "INT8"|| op.quant() == "INT8_PER_CHANNEL"||op.quant() == "INT8_MULTIPLIER") {
        assert(threshold_x != 0.0);
        std::vector<int> data(256, 0);

        for (int idx = 0; idx < 256; ++idx) {
          char lutInput = static_cast<char>(idx);
          float index = lutInput * threshold_x / 127.0;
          float lutOutput = 1.0 /(index) * 127.0 / threshold_y;
          int lutOutputI32 = std::floor(lutOutput + 0.5);
          lutOutputI32 = (lutOutputI32 > 127)
                             ? 127
                             : (lutOutputI32 < -127) ? -127 : lutOutputI32;
          data[idx] = lutOutputI32;
        }
        for (int i = 0; i < size; ++i) {
          output[i] = data[(unsigned char)input[i]];
        }
    }else if(op.quant() == "NONE"){
     
      float numerator = op.numerator().convertToFloat();
      auto input_type = op.input()->getType().cast<TensorType>();
      std::vector<int64_t> i_s(input_type.getShape());

      int n,c,h,w;
      if(i_s.size()==4){
        n = i_s[0];
        c = i_s[1];
        h = i_s[2];
        w = i_s[3];
      }else if(i_s.size()==3){
        n = 1;
        c = i_s[0];
        h = i_s[1];
        w = i_s[2];
      }else{
        assert(0&&"only support shape size 4 or 3");
      }

      for (int i = 0; i < n * c * h * w; ++i) {
        output[i] = numerator/(input[i] + eps);
      }
    }else{
      assert(0&&"not support method");
    }

    valueMapping[result] = std::move(resultT);
    return success();
  }
  if (auto op = dyn_cast<tpu::SqrtOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "SqrtOp" << "\n";);
    auto opdT = getOperandTensors(opInst, valueMapping);
    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() == 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto resultT = std::make_unique<std::vector<float> >(size);


    float threshold_y,threshold_x;
    
    if (op.quant() != "NONE"){
      threshold_y = op.threshold_y().getValue().convertToFloat();
      threshold_x = getPreviousOpThreshold(op);
    }

    float *input = (float *)opdT[0]->data();

    float *output = (float *)resultT->data();

    if (op.quant() == "INT8"|| op.quant() == "INT8_PER_CHANNEL"||op.quant() == "INT8_MULTIPLIER") {
      assert(threshold_x != 0.0);
      std::vector<int> data(256, 0);

      for (int idx = 0; idx < 256; ++idx) {
        char lutInput = static_cast<char>(idx);
        float index = lutInput * threshold_x / 128.0;
        float lutOutput = pow(index,0.5) * 128.0 / threshold_y;
        int lutOutputI32 = std::floor(lutOutput + 0.5);
        lutOutputI32 = (lutOutputI32 > 127)
                           ? 127
                           : (lutOutputI32 < -128) ? -128 : lutOutputI32;
        data[idx] = lutOutputI32;
      }
      for (int i = 0; i < size; ++i) {
        output[i] = data[(unsigned char)input[i]];
      }
    }else if(op.quant() == "NONE"){
      auto input_type = op.input()->getType().cast<TensorType>();
      std::vector<int64_t> i_s(input_type.getShape());

      int n,c,h,w;
      if(i_s.size()==4){
        n = i_s[0];
        c = i_s[1];
        h = i_s[2];
        w = i_s[3];
      }else if(i_s.size()==3){
        n = 1;
        c = i_s[0];
        h = i_s[1];
        w = i_s[2];
      }else{
        assert(0&&"only support shape size 4 or 3");
      }

      for (int i = 0; i < n * c * h * w; ++i) {
        output[i] = pow(input[i],0.5);
      }
    }else{
      assert(0&&"no other quant method is support");
    }

    valueMapping[result] = std::move(resultT);
    return success();
  }

  if (auto op = dyn_cast<tpu::PriorBoxOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "PriorBoxOp" << "\n";);
    auto opdT = getOperandTensors(opInst, valueMapping);
    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto resultT = std::make_unique<std::vector<float> >(size);

    float min_size = op.min_size().convertToFloat();
    float max_size = op.max_size().convertToFloat();
    //float aspect_ratio = op.aspect_ratio0().convertToFloat();
    int aspect_ratios_size = op.aspect_ratios_size().getLimitedValue();
    bool flip = op.flip();
    bool clip = op.clip();
    float variance0 = op.variance0().convertToFloat();
    float variance1 = op.variance1().convertToFloat();
    float variance2 = op.variance2().convertToFloat();
    float variance3 = op.variance3().convertToFloat();
    float offset = op.offset().convertToFloat();
    float step = op.step().convertToFloat();
    vector<float> min_sizes_;
    vector<float> max_sizes_;
    vector<float> aspect_ratios;
    vector<float> aspect_ratios_;
    bool flip_;
    int num_priors_;
    bool clip_;
    vector<float> variance_;
    int img_w_;
    int img_h_;
    float step_w_;
    float step_h_;

    float offset_;

    aspect_ratios.push_back(op.aspect_ratio0().convertToFloat()) ;
    if(aspect_ratios_size==2)
      aspect_ratios.push_back(op.aspect_ratio1().getValue().convertToFloat()) ;

    int max_size_size=op.max_size_size().getLimitedValue();
    int min_size_size=op.min_size_size().getLimitedValue();


  for (int i = 0; i < min_size_size; ++i) {
    min_sizes_.push_back(min_size);
    assert(min_sizes_.back()> 0 && "min_size must be positive.");
    assert(i==0); //more than one min size is not support.
  }

    aspect_ratios_.clear();
    aspect_ratios_.push_back(1.);
    flip_ = flip;
    for (int i = 0; i < aspect_ratios_size; ++i) {
          float ar = aspect_ratios[i];
          bool already_exist = false;
          for (size_t j = 0; j < aspect_ratios_.size(); ++j) {
            if (fabs(ar - aspect_ratios_[j]) < 1e-6) {
              already_exist = true;
              break;
            }
          }
          if (!already_exist) {
            aspect_ratios_.push_back(ar);
            if (flip_) {
              aspect_ratios_.push_back(1./ar);
            }
          }
      }

    num_priors_ = aspect_ratios_.size() * min_sizes_.size();


    max_sizes_.push_back(max_size);
    assert(max_sizes_[0]> min_sizes_[0] && "max_size must be greater than min_size.");
    num_priors_ += 1;

    clip_ = clip;

    // Must and only provide 4 variance.
    assert(variance0> 0);
    variance_.push_back(variance0);
    assert(variance1> 0);
    variance_.push_back(variance1);
    assert(variance2> 0);
    variance_.push_back(variance2);
    assert(variance3> 0);
    variance_.push_back(variance3);

    img_h_ = 0;
    img_w_ = 0;

    assert(step>0&&( "step should be larger than 0."));
    step_h_ = step;
    step_w_ = step;

    offset_ = offset;

  std::vector<int64_t> shape1 = op.getOperand(1)->getType().cast<TensorType>().getShape();
  std::vector<int64_t> shape0 = op.getOperand(0)->getType().cast<TensorType>().getShape();
  assert(shape1.size()==4&&shape0.size()==4);
  const int layer_width = shape0[3];
  const int layer_height = shape0[2];

  int img_width, img_height;
  if (img_h_ == 0 || img_w_ == 0) {
    img_width = shape1[3];
    img_height = shape1[2];
  } else {
    img_width = img_w_;
    img_height = img_h_;
  }
  float step_w, step_h;
  if (step_w_ == 0 || step_h_ == 0) {
    step_w = static_cast<float>(img_width) / layer_width;
    step_h = static_cast<float>(img_height) / layer_height;
  } else {
    step_w = step_w_;
    step_h = step_h_;
  }


  float *top_data = (float *)resultT.get()->data();

  int dim = layer_height * layer_width * num_priors_ * 4;
  int idx = 0;
  for (int h = 0; h < layer_height; ++h) {
    for (int w = 0; w < layer_width; ++w) {
      float center_x = (w + offset_) * step_w;
      float center_y = (h + offset_) * step_h;
      float box_width, box_height;
      for (int s = 0; s < min_size_size; ++s) {
        int min_size_ = min_sizes_[s];
        // first prior: aspect_ratio = 1, size = min_size
        box_width = box_height = min_size_;
        // xmin
        top_data[idx++] = (center_x - box_width / 2.) / img_width;
        // ymin
        top_data[idx++] = (center_y - box_height / 2.) / img_height;
        // xmax
        top_data[idx++] = (center_x + box_width / 2.) / img_width;
        // ymax
        top_data[idx++] = (center_y + box_height / 2.) / img_height;

        if (max_size_size>0) {
          int max_size_ = max_sizes_[s];
          // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
          box_width = box_height = sqrt(min_size_ * max_size_);
          // xmin
          top_data[idx++] = (center_x - box_width / 2.) / img_width;
          // ymin
          top_data[idx++] = (center_y - box_height / 2.) / img_height;
          // xmax
          top_data[idx++] = (center_x + box_width / 2.) / img_width;
          // ymax
          top_data[idx++] = (center_y + box_height / 2.) / img_height;
        }

        // rest of priors
        for (size_t r = 0; r < aspect_ratios_.size(); ++r) {
          float ar = aspect_ratios_[r];
          if (fabs(ar - 1.) < 1e-6) {
            continue;
          }
          box_width = min_size_ * sqrt(ar);
          box_height = min_size_ / sqrt(ar);
          // xmin
          top_data[idx++] = (center_x - box_width / 2.) / img_width;
          // ymin
          top_data[idx++] = (center_y - box_height / 2.) / img_height;
          // xmax
          top_data[idx++] = (center_x + box_width / 2.) / img_width;
          // ymax
          top_data[idx++] = (center_y + box_height / 2.) / img_height;
        }
      }
    }
  }
  // clip the prior's coordidate such that it is within [0, 1]
  if (clip_) {
    for (int d = 0; d < dim; ++d) {
      top_data[d] = std::min<float>(std::max<float>(top_data[d], 0.), 1.);
    }
  }

  auto output_type = op.output()->getType().cast<TensorType>();
  std::vector<int64_t> o_s(output_type.getShape());

  // set the variance.
  top_data += (o_s[2]);

  int count = 0;
  for (int h = 0; h < layer_height; ++h) {
    for (int w = 0; w < layer_width; ++w) {
      for (int i = 0; i < num_priors_; ++i) {
        for (int j = 0; j < 4; ++j) {
          top_data[count] = variance_[j];
          ++count;
        }
      }
    }
  }

  valueMapping[result] = std::move(resultT);
  return success();
 }

 if (auto op = dyn_cast<tpu::DetectionOutputOp>(opInst)) {
     LLVM_DEBUG(llvm::errs() << "DetectionOutputOp" << "\n";);

    auto opdT = getOperandTensors(opInst, valueMapping);
    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto resultT = std::make_unique<std::vector<float> >(size);

    int num_classes_ = op.num_classes().getLimitedValue();
    bool share_location_ = op.share_location();
    int num_loc_classes_ = share_location_ ? 1 : num_classes_;
    int background_label_id_ = op.background_label_id().getValue().getLimitedValue();
    Decode_CodeType code_type_;
    if(op.code_type() == "CORNER"){
      code_type_ = PriorBoxParameter_CodeType_CORNER;
    }else if(op.code_type() == "CENTER_SIZE"){
      code_type_ = PriorBoxParameter_CodeType_CENTER_SIZE;
    }else if(op.code_type() == "CORNER_SIZE"){
      code_type_ = PriorBoxParameter_CodeType_CORNER_SIZE;
    }else{
      assert(0);
    }
    bool variance_encoded_in_target_ =  false;

    int keep_top_k_ = op.keep_top_k().getValue().getLimitedValue();
    float confidence_threshold_ = op.confidence_threshold().getValue().convertToFloat();

    // Parameters used in nms.
    float nms_threshold_ = op.nms_threshold().getValue().convertToFloat();
    float eta_ = 1.0;
    int top_k_ = op.top_k().getValue().getLimitedValue();

    auto input_type0 = op.input()[0]->getType().cast<TensorType>();
    std::vector<int64_t> i_s0(input_type0.getShape());
    auto input_type1 = op.input()[1]->getType().cast<TensorType>();
    std::vector<int64_t> i_s1(input_type1.getShape());
    auto input_type2 = op.input()[2]->getType().cast<TensorType>();
    std::vector<int64_t> i_s2(input_type2.getShape());
    int num = i_s0[0];
    int num_priors_ = i_s2[2]/ 4;

    float* loc_data= (float *)opdT[0]->data();
    float* conf_data = (float *)opdT[1]->data();
    float* prior_data= (float *)opdT[2]->data();


    //calc && sort
    vector<map<int, vector<pair<float ,int>> > > all_conf_scores;
    GetConfidenceScores_opt(conf_data, num, num_priors_, num_classes_, confidence_threshold_, &all_conf_scores);
    for (int i = 0; i < num; ++i) {
      for (int c = 0; c < num_classes_; ++c) {
        if (all_conf_scores[i].find(c) == all_conf_scores[i].end()){
          LLVM_DEBUG(std::cout<<"class with no score idx = %d,"<<c<<"\n";);
          continue;
        }
        vector<pair<float,int> >& scores = all_conf_scores[i].find(c)->second;

        if (top_k_ < (int)scores.size()) {
          std::partial_sort (scores.begin(), scores.begin()+top_k_ ,scores.end(), SortScoreCmp0);
        } else {
          std::sort (scores.begin() , scores.end(), SortScoreCmp0);
        }
      }
    }

    //build keep for decode ,recode vilad index
    float *decode_keep_index;
    int buf_length = 0;
    if (share_location_) {
      buf_length = num * num_priors_;
    } else {
      buf_length = num * num_priors_ * num_classes_;
    }
    decode_keep_index = new float[buf_length];
    memset (decode_keep_index , 0 , buf_length*4);
    float *p = decode_keep_index;
    for (int i = 0; i < num; ++i) {
      if (share_location_) {
        p = decode_keep_index + num_priors_*i;
      }
      for (int c = 0; c < num_classes_; ++c) {
        if (!share_location_) {
          p = decode_keep_index + num_priors_*num_classes_*i + num_priors_*c;
        }
        if (c == background_label_id_) {
          // Ignore background class.
          continue;
        }

        if (all_conf_scores[i].find(c) == all_conf_scores[i].end())
          continue;
        vector<pair<float,int> >& scores = all_conf_scores[i].find(c)->second;
        int length = top_k_ < (int)scores.size() ? top_k_ : scores.size();
        for (int k = 0; k < length; ++k) {
          p[scores[k].second] = 1;
        }
      }
    }

    // Retrieve all location predictions.
    vector<LabelBBox_l> all_loc_preds;
    GetLocPredictions_opt(loc_data, num, num_priors_, num_loc_classes_,
                         share_location_, decode_keep_index, &all_loc_preds);

    // Decode all loc predictions to bboxes.
    vector<LabelBBox_l> all_decode_bboxes;
    const bool clip_bbox = false;
    DecodeBBoxesAll_opt(all_loc_preds, num_priors_ ,prior_data , num,
                       share_location_, num_loc_classes_, background_label_id_,
                       code_type_, variance_encoded_in_target_, clip_bbox,decode_keep_index,
                       &all_decode_bboxes);
    delete [] decode_keep_index;

    int num_kept = 0;
    vector<map<int, vector<pair<float,int>>> > all_indices;
    for (int i = 0; i < num; ++i) {
      const LabelBBox_l& decode_bboxes = all_decode_bboxes[i];
      const map<int, vector<pair<float ,int>> >& conf_scores = all_conf_scores[i];
      map<int, vector<pair<float,int>> > indices;
      int num_det = 0;
      for (int c = 0; c < num_classes_; ++c) {
        if (c == background_label_id_) {
          // Ignore background class.
          continue;
        }
        if (conf_scores.find(c) == conf_scores.end())
          continue;
        int label = share_location_ ? -1 : c;
        if (decode_bboxes.find(label) == decode_bboxes.end()) {
          // Something bad happened if there are no predictions for current label.
          llvm::errs() << "Could not find location predictions for label " << label;
          continue;
        }
        const vector<BBox_l>& bboxes = decode_bboxes.find(label)->second;
        const vector<pair<float ,int>>& aa = conf_scores.find(c)->second;
        ApplyNMSFast_opt(bboxes, aa, confidence_threshold_, nms_threshold_, eta_, top_k_, &(indices[c]));

        num_det += indices[c].size();
      }

      if (keep_top_k_ > -1 && num_det > keep_top_k_) {
        vector<pair<float, pair<int, int> > > score_index_pairs;
        for (auto it = indices.begin();
             it != indices.end(); ++it) {
          int label = it->first;

          const vector<pair<float,int>>& label_indices = it->second;
          for (int j = 0; j < (int)label_indices.size(); ++j) {
            score_index_pairs.push_back(std::make_pair(
            label_indices[j].first, std::make_pair(label, label_indices[j].second)));
          }
        }
        // Keep top k results per image.
        std::sort (score_index_pairs.begin(), score_index_pairs.end(),SortScoreCmp1);
        score_index_pairs.resize(keep_top_k_);
        // Store the new indices.
        map<int, vector<pair<float,int>> > new_indices;
        for (int j = 0; j < (int)score_index_pairs.size(); ++j) {

          int label = score_index_pairs[j].second.first;
          int idx = score_index_pairs[j].second.second;
          float s = score_index_pairs[j].first;


          new_indices[label].push_back(make_pair(s , idx));
        }
        all_indices.push_back(new_indices);
        num_kept += keep_top_k_;
      } else {
        all_indices.push_back(indices);
        num_kept += num_det;
      }
    }
    //float *top_data = (float *)opdT[0]->data();

    float *top_data = (float *)resultT.get()->data();

    int output_size = num*keep_top_k_*1*1*7;
    //init output buf
    for (int i = 0; i < output_size; ++i) {
      top_data[i] = -1;
    }

    if (num_kept == 0) {
      LLVM_DEBUG(llvm::errs() << "Couldn't find any detections";);
      // Generate fake results per image.
      for (int i = 0; i < num; ++i) {
        top_data[0] = i;
        top_data += 7;
      }
    } else {
      int count = 0;
      for (int i = 0; i < num; ++i) {
        const LabelBBox_l& decode_bboxes = all_decode_bboxes[i];
        for (auto it = all_indices[i].begin();
            it != all_indices[i].end(); ++it) {
          int label = it->first;
          int loc_label = share_location_ ? -1 : label;
          if (decode_bboxes.find(loc_label) == decode_bboxes.end()) {
            // Something bad happened if there are no predictions for current label.
            llvm::errs() << "Could not find location predictions for " << loc_label;
            continue;
          }
          const vector<BBox_l>& bboxes = decode_bboxes.find(loc_label)->second;
          vector<pair<float,int>>& indices = it->second;
          for (int j = 0; j < (int)indices.size(); ++j) {

            int idx = indices[j].second;
            top_data[count * 7] = i;
            top_data[count * 7 + 1] = label;
            top_data[count * 7 + 2] = indices[j].first;
            const BBox_l& bbox = bboxes[idx];
            top_data[count * 7 + 3] = bbox.xmin;
            top_data[count * 7 + 4] = bbox.ymin;
            top_data[count * 7 + 5] = bbox.xmax;
            top_data[count * 7 + 6] = bbox.ymax;
            ++count;
          }
        }
      }
    }
scal    valueMapping[result] = std::move(resultT);
    return success();
  }
  if (auto op = dyn_cast<tpu::PowerOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "PowerOp" << "\n";);
    auto opdT = getOperandTensors(opInst, valueMapping);
    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto resultT = std::make_unique<std::vector<float> >(size);

    int nchw[4] = {1, 1, 1, 1};
    float power = op.power().convertToFloat();
    float scale = op.scale().convertToFloat();
    float shift = op.shift().convertToFloat();
    //float rshift = op.rshift().getValue().convertToFloat();
    LLVM_DEBUG(llvm::errs() << "  power" << power << ", scale " << scale << ", shift " << shift << "\n";);
    auto input_type = op.x()->getType().cast<TensorType>();
    std::vector<int64_t> i_s(input_type.getShape());
    auto output_type = op.y()->getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());
    assert((i_s == o_s) && "input shape not equal to output shape");
    for (uint64_t i = 0; i < i_s.size(); i++) {
      nchw[i] = i_s[i];
    }

/*
    How to get Qscale value: 

    X = Sx*Qy
    Y = Sy*Qy
    Sx = thrx /127
    Sy = thry /127

    Y=X*X 
    ==> Sy*Qy=Sx*Qx*Sx*Qx
    ==> Qy = ((thrx*thrx/(128))*(Qx*Qx))/thry
    ==> Qscale = (thrx*thrx/128)/thry

*/
    float threshold_y,threshold_x,qscale,rshift;
    uint32_t multiplier;
    if (op.quant() != "NONE"){

      threshold_y = op.threshold_y().getValue().convertToFloat();
      threshold_x = getPreviousOpThreshold(op);

      qscale = (threshold_x*threshold_x) /(127*threshold_y);  
    }
    
    if (op.quant() == "INT8"|| op.quant() == "INT8_PER_CHANNEL") {
      rshift = findRShiftAndMultiplierFromQScale(qscale);
      multiplier = findMultiplierFromQScaleAndRShift(qscale, rshift);
    }else if(op.quant() == "INT8_MULTIPLIER"){
      rshift = (float)findRShiftAndMultiplierFromQScale(qscale, &multiplier, true,255);                                      
    }else if(op.quant() == "NONE"){

    }else{
      assert(0&&"no other quant method is support");
    }

    #define POWER_INPUT_NR (1)
    float *input[POWER_INPUT_NR]; 
    for (int index = 0; index < POWER_INPUT_NR; ++index) {
      input[index] = (float *)opdT[index]->data();
    }

    float *output = (float *)resultT.get()->data();

/*      if (op.quant() == "INT8"|| op.quant() == "INT8_PER_CHANNEL"||op.quant() == "INT8_MULTIPLIER") {
    assert(threshold_x != 0.0);
    std::vector<int> data(256, 0);

    for (int idx = 0; idx < 256; ++idx) {
      char lutInput = static_cast<char>(idx);
      float index = lutInput * threshold_x / 127.0;
      float lutOutput = pow(index,2) * 127.0 / threshold_y;
      int lutOutputI32 = std::floor(lutOutput + 0.5);
      lutOutputI32 = (lutOutputI32 > 127)
                         ? 127
                         : (lutOutputI32 < -128) ? -128 : lutOutputI32;
      data[idx] = lutOutputI32;
    }
    for (int i = 0; i < size; ++i) {
      output[i] = data[(unsigned char)input[0][i]];
    }
  }else */{

    int ret = my_power(input[0], output, nchw[0], nchw[1], nchw[2], nchw[3], scale, shift, power);
    assert(ret == 0);

    // rshift and saturate on output
    
      //assert(rshift);
      for (int i = 0; i < size; ++i) {
        if (op.quant() == "INT8"|| op.quant() == "INT8_PER_CHANNEL") {
          output[i] = output[i]*multiplier;
        output[i] = (float)applyRShiftAndSaturateInt8(output[i], (uint32_t)rshift);
        }else if(op.quant() == "INT8_MULTIPLIER"){
          output[i] = (float)applyMultiplierAndRShiftAndSaturateInt8(output[i],rshift,  multiplier);        
        }
        /*    else if (op.quant() == "BF16"){
              auto tensor_bf16 = std::make_unique<std::vector<bfloat16> >(resultT->size());
              FloatToBFloat16(resultT->data(), tensor_bf16->data(), resultT->size()); // with rounding
              BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
        }*/
      }
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
