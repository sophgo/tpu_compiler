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


#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/ModuleInterpreter.h"
#include "tpuc/NativeCpuImplementation.h"
#include "tpuc/CustomOpParam.h"
#include "tpuc/GPUInplementation.h"
#include "tpuc/CpuLayer_DetectionOutput.h"
#include "tpuc/CpuLayer_FasterRCNN.h"
#include "tpuc/CpuLayer_RetinaFaceDetection.h"
#include "tpuc/CpuLayer_YoloDetection.h"
#include "tpuc/CustomOpPlugin.h"
#include "tpuc/MachineInfo.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "tpuc/Support/TensorFile.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DynamicLibrary.h"

#include <numeric>
#include <functional>
#include <algorithm>
#include <unordered_map>

extern llvm::cl::opt<bool> clUseTPUQuantOp;
extern float BF16_TABLE_START;
extern float BF16_TABLE_END;
namespace mlir {

std::vector<std::shared_ptr<std::vector<float>>>
getOperandTensors(Operation *op, const value_map_t &valueMapping) {
  std::vector<std::shared_ptr<std::vector<float>>> opdT;
  for (auto operand : op->getOperands()) {
    if (isTensorNone(operand)) {
      opdT.push_back(nullptr);
      continue;
    }
    auto it = valueMapping.find(operand);
    if(it == valueMapping.end()){
      llvm::errs() << "not find: " << operand.getDefiningOp()->getName() << "\n";
      llvm_unreachable("value mapping false");
    };
    opdT.push_back(it->second);
  }
  return opdT;
}


LogicalResult tpu::AbsOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);

  // parse param
  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(this->input(), shape, input_size);
  assert(input_size == size);
  getNCHW(shape, n, c, h, w);

  // get tensors
  assert(opdT.size() == 1);
  std::shared_ptr<std::vector<float> > input = opdT[0];
#ifdef USE_GPU
  // compute in fp32
  int ret;
  if (dm == DeviceMode::GPU){
    ret = gpu_abs(input->data(), resultT->data(), n, c, h, w);
  }else{
    ret = my_abs(input->data(), resultT->data(), n, c, h, w);
  }
#else
  int ret = my_abs(input->data(), resultT->data(), n, c, h, w);
#endif
  assert(ret == 0);

  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::ArgMaxOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);
  (void)valueMapping;
  return success();
}

LogicalResult tpu::BatchNormOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(this->input(), shape, input_size);
  assert(input_size == size);
  getNCHW(shape, n, c, h, w);

  float *input = (float *)opdT[0]->data();
  float *mean = (float *)opdT[1]->data();
  float *variance = (float *)opdT[2]->data();
  float *scale = (float *)opdT[3]->data();
  float *output = (float *)resultT.get()->data();
  float variance_epsilon = this->variance_epsilon().convertToFloat();

  int ret = my_bn(input, mean, variance, scale, variance_epsilon, output, n, c, h, w);
  assert(ret == 0);

  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::BroadcastMulOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);

  // parse param
  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w, bn, bc, bh, bw;
  getTensorShapeAndSize(this->input(), shape, input_size);
  assert(input_size == size);
  getNCHW(shape, n, c, h, w);
  auto bshape = getTensorShape(op->getOperand(1));
  getNCHW(bshape, bn, bc, bh, bw);
  bool do_relu = this->do_relu();
  int axis = this->axis();
  assert(axis == 1);

  // get tensors
  assert(opdT.size() == 6);
  std::shared_ptr<std::vector<float> > input = opdT[0];
  std::shared_ptr<std::vector<float> > scale = opdT[1];
  std::shared_ptr<std::vector<float> > quant_rshift = opdT[4];
  std::shared_ptr<std::vector<float> > quant_multiplier = opdT[5];

  // MUL apply qscale on output put, no scaling on input
  assert(bn == n || bn == 1);
  // compute in fp32
  if (bc == c && bh == 1 && bw == 1) {
    // do scale
    if (bn == n) {
      c = n * c;
      n = 1;
    }
    int ret = my_scale(input->data(), scale->data(), nullptr,
                      resultT->data(), n, c, h, w);
    assert(ret == 0);
  } else if (bh == h && bw == w && bc == 1) {
    // bcast mul
    if (bn == 1) {
      c = n * c;
      n = 1;
    }
    int hw = h * w;
    for (int k = 0; k < n; k++){
      for (int j = 0; j < c; j++) {
        for (int i = 0; i < hw; i++) {
          int offset = (k * c + j) * hw + i;
          resultT->at(offset) = input->at(offset) * scale->at(k * hw + i);
        }
      }
    }
  } else {
    assert(0 && "not support now");
  }
  if (do_relu) {
    my_relu(resultT->data(), resultT->data(), n, c, h, w, 0.0f);
  }
  // rshift and saturate on output
  if (mlir::getOpQuant(op) == "NONE") {
    // do nothing
  } else if (mlir::getOpQuant(op) == "INT8") {
    for (int i = 0; i < size; ++i) {
      resultT->at(i) = (float)applyMultiplierAndRShiftAndSaturateInt8(
          resultT->at(i), (uint32_t)quant_rshift->at(0),
          (uint32_t)quant_multiplier->at(0), true);
    }
  } else if (mlir::getOpQuant(op) == "BF16") {
    auto tensor_bf16 = std::make_unique<std::vector<bfloat16> >(resultT->size());
    FloatToBFloat16(resultT->data(), tensor_bf16->data(), resultT->size()); // with rounding
    BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
  } else {
    llvm_unreachable("unsupported type");
  }

  valueMapping[result] = std::move(resultT);
  return success();
}

LogicalResult tpu::ConcatOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float>>> &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name()
                          << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  std::vector<int64_t> output_shape = getTensorShape(result);
  int64_t output_size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float>>(output_size);

  for (uint32_t i = output_shape.size(); i < 4; i++) {
    output_shape.push_back(1); // append to 4 dim
  }

  // parse param
  int concat_axis = this->axis();
  LLVM_DEBUG(llvm::errs() << "concat_axis =" << concat_axis << "\n";);

  size_t nInputs = this->getNumInputs();
  assert(nInputs >= 2 && "bottom num is 0 or 1");

  // quant param
  std::shared_ptr<std::vector<float>> quant_rshift = opdT[nInputs + 2];
  std::shared_ptr<std::vector<float>> quant_multiplier = opdT[nInputs + 3];
  bool need_quant = false;
  float rshift = 0;
  if (mlir::getOpQuant(op) == "INT8" && quant_rshift != nullptr &&
      quant_multiplier != nullptr) {
    need_quant = true;
    rshift = quant_rshift->at(0);
  }

  int global_offset = 0;
  int num_concats = 1;
  std::vector<int64_t> input_shape = getTensorShape(op->getOperand(0));
  for (int idx = 0; idx < concat_axis; idx++) {
    num_concats *= input_shape[idx];
  }

  int concat_output_size = 1;
  for (int idx = concat_axis; idx < 4; idx++) {
    concat_output_size *= output_shape[idx];
  }

  for (uint32_t i = 0; i < nInputs; i++) {
    float *input = (float *)opdT[i]->data();
    std::vector<int64_t> input_shape = getTensorShape(op->getOperand(i));
    for (uint32_t idx = input_shape.size(); idx < 4; idx++) {
      input_shape.push_back(1);
    }

    int concat_input_size = 1;
    for (int idx = concat_axis; idx < 4; idx++) {
      concat_input_size *= input_shape[idx];
    }

    for (int num_idx = 0; num_idx < num_concats; num_idx++) {
      auto inputT = std::make_unique<std::vector<float>>(concat_input_size);
      inputT.get()->assign(&input[num_idx * concat_input_size],
                           &input[(num_idx + 1) * concat_input_size]);

      if (need_quant) {
        for (int idx = 0; idx < concat_input_size; ++idx) {
          inputT->at(idx) = (float)applyMultiplierAndRShiftAndSaturateInt8(
              inputT->at(idx), rshift,
              (uint32_t)quant_multiplier->at(i), false);
        }
      }

      auto offset = global_offset + concat_output_size * num_idx;
      std::memcpy(resultT.get()->data() + offset, inputT.get()->data(),
                  concat_input_size * sizeof(float));
    }
    global_offset += concat_input_size;
  }

  if (do_relu()) {
    my_relu(resultT->data(), resultT->data(), output_shape[0], output_shape[1],
            output_shape[2], output_shape[3], 0.0f);
  }

  valueMapping[result] = std::move(resultT);
  return success();
}

template <typename OpTy>
LogicalResult doConv2DOpInterpret(Operation *op,
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  auto castOp = cast<OpTy>(op);
  assert(castOp);
  bool is_deconv = isa<tpu::DeConv2DOp>(op);
  bool do_bias_later = false;
  bool do_relu_later = false;

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = castOp.getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);

  // parse param
  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, pt, pb, pl, pr, dh, dw, pad_value;
  parseConvParam(castOp.param(), is_deconv, castOp.input(), castOp.output(),
                 castOp.filter(), n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw,
                 pt, pb, pl, pr, dh, dw, is_dw, with_bias, do_relu, pad_value);

  // get tensors
  assert(opdT.size() == 7);
  std::vector<float> input(opdT[0]->begin(), opdT[0]->end());

  std::shared_ptr<std::vector<float> > filter = opdT[1];
  std::shared_ptr<std::vector<float> > bias = opdT[2];
  std::shared_ptr<std::vector<float> > quant_rshift = opdT[5];
  std::shared_ptr<std::vector<float> > quant_multiplier = opdT[6];

  bool is_asymmetric = isOpQuantAsymmetric(op);

  // get is dilate activation
  std::vector<int32_t> ins;
  arrayAttrToVector(castOp.param().ins(), ins);

  if (ins.size()) {
    int ins_w = ins[0];
    int ins_h = 0;
    if (ins.size() > 1) {
      ins_h = ins[1];
    }

    if (ins_w == 0 && ins_h == 0) {
      // no need to dilate
    }
    else {

      int oh = calc_dilute_hw(ih, ins_h, 0, 0, 0);
      int ow = calc_dilute_hw(iw, ins_w, 0, 0, 0);
      int size = n * ic * oh * ow;
      auto dilateActivation = std::vector<float> (size);
      my_dilateActivation (input.data(), dilateActivation.data(), 0, 0, ins_h, 0, 0, 0, ins_w, 0, n, ic, ih, iw);
      // update dilated info
      input = std::vector<float>(dilateActivation);
      ih = oh;
      iw = ow;
    }
  }


  // compute in fp32
  if (!is_deconv) {
#ifdef USE_GPU
    int ret;
    LLVM_DEBUG(llvm::errs() << "  k: (" << kh << "*" << kw << "), "
                 << "s: (" << sh << "*" << sw << "), "
                 << "p: (" << ph << "*" << pw << "), "
                 << "g: " << g << "\n";);
    if (dm == DeviceMode::GPU) {

      ret = gpu_conv(input->data(), filter->data(),
                         bias ? bias->data() : nullptr, resultT->data(), n, ic,
                         ih, iw, oc, oh, ow, kh, kw, sh, sw, dh, dw, ph, pw, g);
    }else{
      ret = mkldnn_conv(input->data(), filter->data(), bias ? bias->data() : nullptr,
                  resultT->data(), n, ic, ih, iw, oc, oh, ow, kh, kw, sh, sw,
                  dh, dw, pt, pb, pl, pr, g, 0);
    }
#else
    float *bias_data = bias ? bias->data() : nullptr;
    if (getOpQuant(op) == "INT8" && isOpQuantPerchannel(op) &&
        getOpQuantParamType(op) == "RSHIFT_AND_M_I32") {
      if (bias_data) {
        assert(with_bias && "with_bias value is false");
        do_bias_later = true;
        bias_data = nullptr;
        if (do_relu) {
          do_relu_later = true;
          do_relu = false;
        }
      }
    }
    int ret = mkldnn_conv(input.data(), filter->data(), bias_data,
                          resultT->data(), n, ic, ih, iw, oc, oh, ow, kh, kw,
                          sh, sw, dh, dw, pt, pb, pl, pr, g, pad_value);
    assert(ret == 0);
#endif
  } else {
    int ret = mkldnn_deconv(input.data(), filter->data(),
        bias ? bias->data() : nullptr, resultT->data(),
        n, ic, ih, iw, oc, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, g);
    assert(ret == 0);
  }

  if (do_relu && !is_asymmetric) {
    int ret = my_relu(resultT->data(), resultT->data(), n, oc, oh, ow, 0.0f);
    assert(ret == 0);
  }
  // rshift and saturate on output
  if (getOpQuant(op) == "NONE") {
    // do nothing
  } else if (getOpQuant(op) == "INT8") {
    if (!isOpQuantPerchannel(op)) {
      assert(getOpQuantParamType(op) == "RSHIFT_ONLY");
      assert(quant_rshift);
      quantizeActivationInt8PerLayerRshift(resultT->data(), resultT->data(),
          size, (uint32_t)quant_rshift->at(0));
    } else if (isOpQuantPerchannel(op)
               && getOpQuantParamType(op) == "RSHIFT_ONLY") {
      assert(quant_rshift);
      quantizeActivationInt8PerChannelRShift(resultT->data(), resultT->data(),
          n, oc, size / oc / n, quant_rshift->data());
    } else if (isOpQuantPerchannel(op)
               && getOpQuantParamType(op) == "RSHIFT_AND_M_I32") {
      assert(quant_rshift);
      assert(quant_multiplier);
      // asymmetric no need to do relu, if value < 0, do offset(-128) will saturate to -128
      quantizeActivationInt8PerChannelMultiplierAndRShift(
          resultT->data(), resultT->data(),
          do_bias_later ? bias->data() : nullptr, do_relu_later && !is_asymmetric, n, oc,
          size / oc / n, quant_rshift->data(), quant_multiplier->data());
    } else {
      assert(false);
    }
  } else if (getOpQuant(op) == "BF16") {
    auto tensor_bf16 = std::make_unique<std::vector<bfloat16> >(resultT->size());
    FloatToBFloat16(resultT->data(), tensor_bf16->data(), resultT->size()); // with rounding
    BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
  } else {
    llvm_unreachable("unsupported type");
  }

  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::Conv2DOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);
  return doConv2DOpInterpret<tpu::Conv2DOp>(op, valueMapping);
}

LogicalResult tpu::DeConv2DOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);
  return doConv2DOpInterpret<tpu::DeConv2DOp>(op, valueMapping);
}

template <typename OpTy>
LogicalResult doConv3DOpInterpret(Operation *op,
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  auto castOp = cast<OpTy>(op);
  assert(castOp);
  bool is_deconv = false; // isa<tpu::DeConv3DOp>(op)
  bool do_bias_later = false;
  bool do_relu_later = false;

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = castOp.getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);

  bool is_dw, with_bias, do_relu;
  int n, ic, id, ih, iw, oc, od, oh, ow, g, kd, kh, kw, sd, sh, sw;
  int pd0, pd1, pt, pb, pl, pr, dd, dh, dw;
  parseConv3dParam(castOp.param(), is_deconv,
                  castOp.input(), castOp.output(), castOp.filter(),
                  n, ic, id, ih, iw,
                  oc, od, oh, ow, g,
                  kd, kh, kw,
                  sd, sh, sw,
                  pd0, pd1, pt, pb, pl, pr,
                  dd, dh, dw,
                  is_dw, with_bias, do_relu);

  // get tensors
  assert(opdT.size() == 7);
  std::shared_ptr<std::vector<float> > input = opdT[0];
  std::shared_ptr<std::vector<float> > filter = opdT[1];
  std::shared_ptr<std::vector<float> > bias = opdT[2];
  //std::shared_ptr<std::vector<float> > quant_scale = opdT[3];
  //std::shared_ptr<std::vector<float> > quant_zeropoint = opdT[4];
  std::shared_ptr<std::vector<float> > quant_rshift = opdT[5];
  std::shared_ptr<std::vector<float> > quant_multiplier = opdT[6];

  // get is dilate activation
  std::vector<int32_t> ins;
  arrayAttrToVector(castOp.param().ins(), ins);
  assert(!ins.size() && "Not support 3d dilated conv");

  // compute in fp32
  if (!is_deconv) {
    float *bias_data = bias ? bias->data() : nullptr;
    if (getOpQuant(op) == "INT8" && isOpQuantPerchannel(op) &&
        getOpQuantParamType(op) == "RSHIFT_AND_M_I32") {
      if (bias_data) {
        assert(with_bias && "with_bias value is false");
        do_bias_later = true;
        bias_data = nullptr;
        if (do_relu) {
          do_relu_later = true;
          do_relu = false;
        }
      }
    }
    /*
    conv3d_float_ref(input->data(), filter->data(), bias_data, resultT->data(),
                     n, ic, id, ih, iw,
                     oc, od, oh, ow,
                     kd, kh, kw,
                     sd, sh, sw,
                     dd, dh, dw,
                     pd0, pd1, pt, pb, pl, pr);
    */
    mkldnn_conv3d(input->data(), filter->data(), bias_data, resultT->data(),
                     n, ic, id, ih, iw,
                     oc, od, oh, ow, g,
                     kd, kh, kw,
                     sd, sh, sw,
                     dd, dh, dw,
                     pd0, pd1, pt, pb, pl, pr);
  } else {
    assert(0 && "Not support 3d deconv");
  }

  if (do_relu) {
    //assert(0 && "Not support 3d conv relu");
    int ret = my_relu(resultT->data(), resultT->data(), n, oc, od*oh, ow, 0.0f);
    assert(ret == 0);
  }

  // rshift and saturate on output
  if (getOpQuant(op) == "NONE") {
    // do nothing
  } else if (getOpQuant(op) == "INT8") {
    if (!isOpQuantPerchannel(op)) {
      assert(getOpQuantParamType(op) == "RSHIFT_ONLY");
      assert(quant_rshift);
      quantizeActivationInt8PerLayerRshift(resultT->data(), resultT->data(),
          size, (uint32_t)quant_rshift->at(0));
    } else if (isOpQuantPerchannel(op)
               && getOpQuantParamType(op) == "RSHIFT_ONLY") {
      assert(quant_rshift);
      quantizeActivationInt8PerChannelRShift(resultT->data(), resultT->data(),
          n, oc, size / oc / n, quant_rshift->data());
    } else if (isOpQuantPerchannel(op)
               && getOpQuantParamType(op) == "RSHIFT_AND_M_I32") {
      assert(quant_rshift);
      assert(quant_multiplier);
      quantizeActivationInt8PerChannelMultiplierAndRShift(
          resultT->data(), resultT->data(),
          do_bias_later ? bias->data() : nullptr, do_relu_later, n, oc,
          size / oc / n, quant_rshift->data(), quant_multiplier->data());
    } else {
      assert(false);
    }
  } else if (getOpQuant(op) == "BF16") {
    auto tensor_bf16 = std::make_unique<std::vector<bfloat16> >(resultT->size());
    FloatToBFloat16(resultT->data(), tensor_bf16->data(), resultT->size()); // with rounding
    BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
  } else {
    llvm_unreachable("unsupported type");
  }

  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::Conv3DOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);
  return doConv3DOpInterpret<tpu::Conv3DOp>(op, valueMapping);
}

LogicalResult tpu::DilateOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float>>(size);

  std::vector<int64_t> shape;
  int64_t input_size, n, ic, ih, iw;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, ic, ih, iw);

  float *input = (float *)opdT[0]->data();
  float *output = (float *)resultT.get()->data();
  auto fill_constant = this->fill_constant();

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
  }

  // check output is valid
  std::vector<int64_t> output_shape;
  int64_t output_size, on, oc, oh, ow;
  getTensorShapeAndSize(result, output_shape, output_size);
  getNCHW(output_shape, on, oc, oh, ow);
  assert(oh == calc_dilute_hw (ih, ins_h, 0, 0, 0) && "mismatch output shape with ins_h");
  assert(ow == calc_dilute_hw (iw, ins_w, 0, 0, 0) && "mismatch output shape with ins_w");


  my_dilateActivation (input, output, 0, 0, ins_h, 0, 0, 0, ins_w, 0, n, ic, ih, iw, fill_constant);

  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::CropOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float>>> &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name()
                          << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float>>(size);

  // parse param

  std::vector<int64_t> input_shape1 = getTensorShape(op->getOperand(0));
  std::vector<int> input_shape2;
  std::vector<int64_t> output_shape = getTensorShape(this->getResult());
  std::vector<int> crop_offset;

  arrayAttrToVector(this->crop_shape().getValue(), input_shape2);
  arrayAttrToVector(this->crop_offset().getValue(), crop_offset);
  assert(output_shape.size() == 4 && " not support dim is not 4\n");
  std::vector<int> indices(input_shape1.size(), 0);
  float *input = (float *)opdT[0]->data();
  float *output = (float *)resultT.get()->data();

  my_crop(input, output, input_shape1.data(), output_shape.data(), 0, crop_offset.data(), indices.data());
  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::DetectionOutputOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);

  std::vector<int64_t> output_shape;
  output_shape = getTensorShape(this->output());

  assert(output_shape.size() <= 4);

    int num_classes_ = this->num_classes();
    bool share_location_ = this->share_location();
    int num_loc_classes_ = share_location_ ? 1 : num_classes_;
    int background_label_id_ = this->background_label_id();
    Decode_CodeType code_type_;
    if(this->code_type() == "CORNER"){
      code_type_ = PriorBoxParameter_CodeType_CORNER;
    }else if(this->code_type() == "CENTER_SIZE"){
      code_type_ = PriorBoxParameter_CodeType_CENTER_SIZE;
    }else if(this->code_type() == "CORNER_SIZE"){
      code_type_ = PriorBoxParameter_CodeType_CORNER_SIZE;
    }else{
      assert(0);
    }
    bool variance_encoded_in_target_ =  false;

    int keep_top_k_ = this->keep_top_k();
    float confidence_threshold_ = this->confidence_threshold().convertToFloat();

    // Parameters used in nms.
    float nms_threshold_ = this->nms_threshold().convertToFloat();
    float eta_ = 1.0;
    int top_k_ = this->top_k();

    auto input_type0 = this->input()[0].getType().cast<TensorType>();
    std::vector<int64_t> i_s0(input_type0.getShape());
    auto input_type1 = this->input()[1].getType().cast<TensorType>();
    std::vector<int64_t> i_s1(input_type1.getShape());
    auto input_type2 = this->input()[2].getType().cast<TensorType>();
    std::vector<int64_t> i_s2(input_type2.getShape());
    int num = i_s0[0];
    int num_priors_ = i_s2[2]/ 4;

    float* loc_data= (float *)opdT[0]->data();
    float* conf_data = (float *)opdT[1]->data();
    float* prior_data= (float *)opdT[2]->data();


    //calc && sort
    std::vector<std::map<int, std::vector<std::pair<float ,int>> > > all_conf_scores;
    GetConfidenceScores_opt(conf_data, num, num_priors_, num_classes_, confidence_threshold_, &all_conf_scores);
    for (int i = 0; i < num; ++i) {
      for (int c = 0; c < num_classes_; ++c) {
        if (all_conf_scores[i].find(c) == all_conf_scores[i].end()){
          LLVM_DEBUG(std::cout<<"class with no score idx = %d,"<<c<<"\n";);
          continue;
        }
        std::vector<std::pair<float,int> >& scores = all_conf_scores[i].find(c)->second;

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
        std::vector<std::pair<float,int> >& scores = all_conf_scores[i].find(c)->second;
        int length = top_k_ < (int)scores.size() ? top_k_ : scores.size();
        for (int k = 0; k < length; ++k) {
          p[scores[k].second] = 1;
        }
      }
    }

    // Retrieve all location predictions.
    std::vector<LabelBBox_l> all_loc_preds;
    GetLocPredictions_opt(loc_data, num, num_priors_, num_loc_classes_,
                         share_location_, decode_keep_index, &all_loc_preds);

    // Decode all loc predictions to bboxes.
    std::vector<LabelBBox_l> all_decode_bboxes;
    const bool clip_bbox = false;
    DecodeBBoxesAll_opt(all_loc_preds, num_priors_ ,prior_data , num,
                       share_location_, num_loc_classes_, background_label_id_,
                       code_type_, variance_encoded_in_target_, clip_bbox,decode_keep_index,
                       &all_decode_bboxes);
    delete [] decode_keep_index;

    int num_kept = 0;
    std::vector<std::map<int, std::vector<std::pair<float,int>>> > all_indices;
    for (int i = 0; i < num; ++i) {
      const LabelBBox_l& decode_bboxes = all_decode_bboxes[i];
      const std::map<int, std::vector<std::pair<float ,int>> >& conf_scores = all_conf_scores[i];
      std::map<int, std::vector<std::pair<float,int>> > indices;
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
        const std::vector<BBox_l>& bboxes = decode_bboxes.find(label)->second;
        const std::vector<std::pair<float ,int>>& aa = conf_scores.find(c)->second;
        ApplyNMSFast_opt(bboxes, aa, confidence_threshold_, nms_threshold_, eta_, top_k_, &(indices[c]));

        num_det += indices[c].size();
      }

      if (keep_top_k_ > -1 && num_det > keep_top_k_) {
        std::vector<std::pair<float, std::pair<int, int> > > score_index_pairs;
        for (auto it = indices.begin();
             it != indices.end(); ++it) {
          int label = it->first;

          const std::vector<std::pair<float,int>>& label_indices = it->second;
          for (int j = 0; j < (int)label_indices.size(); ++j) {
            score_index_pairs.push_back(std::make_pair(
            label_indices[j].first, std::make_pair(label, label_indices[j].second)));
          }
        }
        // Keep top k results per image.
        std::sort (score_index_pairs.begin(), score_index_pairs.end(),SortScoreCmp1);
        score_index_pairs.resize(keep_top_k_);
        // Store the new indices.
        std::map<int, std::vector<std::pair<float,int>> > new_indices;
        for (int j = 0; j < (int)score_index_pairs.size(); ++j) {

          int label = score_index_pairs[j].second.first;
          int idx = score_index_pairs[j].second.second;
          float s = score_index_pairs[j].first;

          new_indices[label].push_back(std::make_pair(s , idx));
        }
        all_indices.push_back(new_indices);
        num_kept += keep_top_k_;
      } else {
        all_indices.push_back(indices);
        num_kept += num_det;
      }
    }
    //float *top_data = (float *)opdT[0]->data();

    float *top_data = (float *)resultT->data();

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
          const std::vector<BBox_l>& bboxes = decode_bboxes.find(loc_label)->second;
          std::vector<std::pair<float,int>>& indices = it->second;
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
    valueMapping[result] = std::move(resultT);
    return success();
}

static LogicalResult doLUTOpInterpret(Operation *op, StringRef &type,
     DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  auto opdT = getOperandTensors(op, valueMapping);
  auto result = op->getResult(0);
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);
  float *input = (float *)opdT[0]->data();
  float *output = (float *)resultT.get()->data();
  std::shared_ptr<std::vector<float> > y0_table_op = opdT[1];
  std::shared_ptr<std::vector<float> > slope_table = opdT[2];
  float* y0_bf16_table;
  float* y0_bf16_slope_table;

  if (getOpQuant(op) == "INT8") {
    for (int i = 0; i < size; ++i) {
      output[i] = y0_table_op->at((unsigned char)input[i]);
    }
  }else if(getOpQuant(op) == "BF16"|| getOpQuant(op) == "NONE"){
    std::vector<int64_t> shape;
    int64_t input_size, n, c, h, w;
    getTensorShapeAndSize(op->getOperand(0), shape, input_size);
    getNCHW(shape, n, c, h, w);
    float _bf16_table_start = BF16_TABLE_START;
    float _bf16_table_end = BF16_TABLE_END;

    if (type == "Reciprocal") {
      for (int i = 0; i < input_size; ++i) {
        output[i] = 1.0 / input[i];
      }
    } else if (type == "Sqrt") {
      for (int i = 0; i < input_size; ++i) {
        output[i] = pow(input[i], 0.5);
      }
    } else if (type == "Sigmoid") {
      auto castOp = dyn_cast<tpu::SigmoidOp>(op);
      BF16_TABLE_START = castOp.min_range().convertToFloat();
      BF16_TABLE_END = castOp.max_range().convertToFloat();

      y0_bf16_table = y0_table_op == nullptr ? nullptr : y0_table_op->data();
      y0_bf16_slope_table = slope_table == nullptr ? nullptr : slope_table->data();
      my_sigmoid(input, output, n, c, h, w,
          y0_bf16_table, y0_bf16_slope_table,
          getOpQuant(op) == "BF16");
    } else if (type == "TanH") {
      auto castOp = dyn_cast<tpu::TanHOp>(op);
      BF16_TABLE_START = castOp.min_range().convertToFloat();
      BF16_TABLE_END = castOp.max_range().convertToFloat();

      y0_bf16_table = y0_table_op == nullptr ? nullptr : y0_table_op->data();
      y0_bf16_slope_table = slope_table == nullptr ? nullptr : slope_table->data();

      my_tanh(input, output, n, c, h, w,
          y0_bf16_table, y0_bf16_slope_table,
          getOpQuant(op) == "BF16");
    } else if (type == "Mish") {
      auto castOp = dyn_cast<tpu::MishOp>(op);
      BF16_TABLE_START = castOp.min_range().convertToFloat();
      BF16_TABLE_END = castOp.max_range().convertToFloat();

      y0_bf16_table = y0_table_op == nullptr ? nullptr : y0_table_op->data();
      y0_bf16_slope_table = slope_table == nullptr ? nullptr : slope_table->data();

      float mish_threshold = castOp.mish_threshold().convertToFloat();
      my_mish(input, output, n, c, h, w, y0_bf16_table, y0_bf16_slope_table,
          getOpQuant(op) == "BF16", mish_threshold);
    } else if (type == "Exp") {
      auto castOp = dyn_cast<tpu::ExpOp>(op);
      BF16_TABLE_START = castOp.min_range().convertToFloat();
      BF16_TABLE_END = castOp.max_range().convertToFloat();

      y0_bf16_table = y0_table_op == nullptr ? nullptr : y0_table_op->data();
      y0_bf16_slope_table = slope_table == nullptr ? nullptr : slope_table->data();

      my_exp(input, output, n, c, h, w,
          y0_bf16_table, y0_bf16_slope_table,
          getOpQuant(op) == "BF16");
    } else if (type == "SoftPlus") {
      auto castOp = dyn_cast<tpu::SoftPlusOp>(op);
      BF16_TABLE_START = castOp.min_range().convertToFloat();
      BF16_TABLE_END = castOp.max_range().convertToFloat();

      y0_bf16_table = y0_table_op == nullptr ? nullptr : y0_table_op->data();
      y0_bf16_slope_table = slope_table == nullptr ? nullptr : slope_table->data();

      float threshold = castOp.threshold().convertToFloat();

      my_softplus(input, output, n, c, h, w,
          y0_bf16_table, y0_bf16_slope_table,
          getOpQuant(op) == "BF16", threshold);
    } else {
      llvm_unreachable("not support LUT op type");
    }

    if (getOpQuant(op) == "BF16"){
      auto tensor_bf16 = std::make_unique<std::vector<bfloat16> >(size);
      // with rounding
      FloatToBFloat16(output, tensor_bf16->data(), size);
      BFloat16ToFloat(tensor_bf16->data(), output, size);
    }

    BF16_TABLE_START = _bf16_table_start;
    BF16_TABLE_END = _bf16_table_end;
  }else{
    llvm_unreachable("not support method");
  }

  valueMapping[result] = std::move(resultT);

  return success();

}

LogicalResult tpu::ReciprocalOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);
  StringRef type = "Reciprocal";
  return doLUTOpInterpret(op,type,valueMapping);
}

LogicalResult tpu::SqrtOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);
  StringRef type = "Sqrt";
  return doLUTOpInterpret(op,type,valueMapping);
}

LogicalResult tpu::SigmoidOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);
  StringRef type = "Sigmoid";
  return doLUTOpInterpret(op,type,valueMapping);
}

LogicalResult tpu::TanHOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float>>> &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name()
                          << "]\n";);
  StringRef type = "TanH";
  return doLUTOpInterpret(op, type, valueMapping);
}

LogicalResult tpu::MishOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float>>> &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name()
                          << "]\n";);
  StringRef type = "Mish";
  return doLUTOpInterpret(op, type, valueMapping);
}

LogicalResult tpu::InstanceNormOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float>>> &valueMapping) {
  llvm_unreachable("InstanceNormOp not support on v1, please use v2\n");
  return failure();
}

LogicalResult tpu::ExpOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float>>> &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name()
                          << "]\n";);
  StringRef type = "Exp";
  return doLUTOpInterpret(op, type, valueMapping);
}

static LogicalResult doBroadcastOpInterpret(Operation *op,
    StringRef &type, bool do_relu,
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  auto opdT = getOperandTensors(op, valueMapping);
  auto result = op->getResult(0);
  auto output_size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(output_size);

  // parse param
  std::vector<int64_t> shape;
  int64_t input0_size, input1_size, n0, c0, h0, w0;
  int64_t n1, c1, h1, w1;
  getTensorShapeAndSize(op->getOperand(0), shape, input0_size);
  assert(input0_size == output_size);
  getNCHW(shape, n0, c0, h0, w0);
  getTensorShapeAndSize(op->getOperand(1), shape, input1_size);
  getNCHW(shape, n1, c1, h1, w1);

  llvm::errs() << llvm::format("interpret, a:%d,%d,%d,%d, b:%d,%d,%d,%d\n",
                  (int)n0, (int)c0, (int)h0, (int)w0, (int)n1, (int)c1, (int)h1, (int)w1);

  // get tensors
  uint32_t nInputs = 2;
  float *input[2];
  input[0] = opdT[0]->data();
  input[1] = opdT[1]->data();

  float *output = resultT->data();
  std::shared_ptr<std::vector<float> > quant_rshift = opdT[nInputs + 2];
  std::shared_ptr<std::vector<float> > quant_multiplier = opdT[nInputs + 3];

  // apply qscale on input tensors before f32 compute
  std::vector<std::vector<float> > input_copy(nInputs);
  if (type == "ADD" || type == "MAX" || type == "MIN" || type == "SUB") {
    if (getOpQuant(op) == "INT8" && getOpQuantParamType(op) != "NONE") {
      for (unsigned i = 0; i < nInputs; ++i) {
        // make copy
        input_copy[i].assign(opdT[i]->begin(),
                             opdT[i]->end());
        input[i] = input_copy[i].data();
      }
      // apply multiplier
      for (unsigned i = 0; i < nInputs; ++i) {
        for (size_t j = 0; j < opdT[i]->size(); ++j) {
          input[i][j] = input[i][j] * (int8_t)quant_multiplier->at(i);
        }
      }
    }
  } else if (type == "MUL") {
    // MUL apply qscale on output put, no scaling on input
  } else {
    llvm_unreachable("unsupported eltwise type");
  }

  // compute in fp32

  size_t idx = 0;
  for (int n = 0; n < (int)n0; ++n) {
    for (int c = 0; c < (int)c0; ++c) {
      for (int h = 0; h < (int)h0; ++h) {
        for (int w = 0; w < (int)w0; ++w) {
          float a = input[0][w + h * w0 + c * h0 * w0 + n * c0 * h0 * w0];
          float b = input[1][(w1 == 1 ? 0 : w) +
                             (h1 == 1 ? 0 : h) * w1 +
                             (c1 == 1 ? 0 : c) * h1 * w1 +
                             (n1 == 1 ? 0 : n) * c1 * h1 * w1];
          if (type == "ADD") {
            output[idx++] = a + b;
          } else if (type == "SUB") {
            output[idx++] = a - b;
          } else if (type == "MAX") {
            output[idx++] = a > b ? a : b;
          } else if (type == "MIN") {
            output[idx++] = a < b ? a : b;
          } else if (type == "MUL") {
            output[idx++] = a * b;
          } else {
            llvm_unreachable("unsupported eltwise type");
          }
        }
      }
    }
  }

  if (do_relu) {
    int ret = my_relu(output, output, n0, c0, h0, w0, 0.0f);
    assert(ret == 0);
  }

  // rshift and saturate on output
  if (getOpQuant(op) == "NONE") {
    // do nothing
  } else if (getOpQuant(op) == "INT8") {
    if (getOpQuantParamType(op) != "NONE") {
      if (type == "ADD" || type == "SUB" || type == "MAX" || type == "MIN") {
        // apply rshift and saturate
        for (int i = 0; i < output_size; ++i) {
          output[i] = (float)applyRShiftAndSaturateInt8(
              output[i], (uint32_t)quant_rshift->at(0));
        }
      } else if (type == "MUL") {
        // apply qscale on output (both rshift and saturate)
        for (int i = 0; i < output_size; ++i) {
          output[i] = (float)applyMultiplierAndRShiftAndSaturateInt8(
              output[i], (uint32_t)quant_rshift->at(0),
              (uint32_t)quant_multiplier->at(0), true);
        }
      }
    }
    for (int i = 0; i < output_size; ++i) {
      if (output[i] > 127.0) {
        output[i] = 127.0;
      } else if (output[i] < -128.0) {
        output[i] = -128.0;
      }
    }
  } else if (getOpQuant(op) == "BF16") {
    auto tensor_bf16 = std::make_unique<std::vector<bfloat16>>(resultT->size());
    FloatToBFloat16(resultT->data(), tensor_bf16->data(),
                    resultT->size()); // with rounding
    BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
  } else {
    assert(false);
  }
  valueMapping[result] = std::move(resultT);
  return success();
}

LogicalResult tpu::BroadcastAddOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);
  StringRef type = "ADD";
  return doBroadcastOpInterpret(op, type, do_relu(), valueMapping);
}

LogicalResult tpu::BroadcastSubOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);
  StringRef type = "SUB";
  return doBroadcastOpInterpret(op, type, do_relu(), valueMapping);
}

static LogicalResult doEltwiseOpInterpret(Operation *op,
    StringRef &type, bool do_relu, std::vector<float> & coeff,
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  auto opdT = getOperandTensors(op, valueMapping);
  auto result = op->getResult(0);
  // auto size = getTensorSize(result);

  // parse param
  std::vector<int64_t> shape;
  int64_t input_size, in, ic, ih, iw;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  auto resultT = std::make_unique<std::vector<float> >(input_size);
  // assert(input_size == size);
  getNCHW(shape, in, ic, ih, iw);
  std::vector<int64_t> output_shape;
  int64_t output_size, on, oc, oh, ow;
  getTensorShapeAndSize(op->getResult(0), output_shape, output_size);
  getNCHW(output_shape, on, oc, oh, ow);
  auto resultReal = std::make_unique<std::vector<float> >(output_size);

  // get tensors
  const unsigned nInputs = op->getNumOperands() - 4;
  std::vector<float *>input(nInputs);
  float *output = resultT->data();

  std::vector<int> input_offset(nInputs, 0);
  int output_offset = 0;
  bool is_asymmetric = isOpQuantAsymmetric(op);

  if (getOpQuant(op) == "INT8" && is_asymmetric){
    for(size_t i = 0; i < nInputs; ++i){
      input_offset.at(i) = -getPreviousOpZeroPoint(op, i);
    }
    output_offset = getOpZeroPoint(op);
  }

  std::shared_ptr<std::vector<float> > quant_rshift = opdT[nInputs + 2];
  std::shared_ptr<std::vector<float> > quant_multiplier = opdT[nInputs + 3];

  // apply qscale on input tensors before f32 compute
  std::vector<std::vector<float> > input_copy(nInputs);
  for (size_t i = 0; i < nInputs; ++i) {
    input[i] = opdT[i]->data();
  }
  if (type == "ADD" || type == "MAX" || type == "MIN") {
    if (getOpQuant(op) == "INT8" && getOpQuantParamType(op) != "NONE") {
      for (unsigned i = 0; i < nInputs; ++i) {
        // make copy
        input_copy[i].assign(opdT[i]->begin(),
                             opdT[i]->end());
        if(is_asymmetric && type != "ADD"){
          llvm_unreachable("TODO: only ADD has asymmetric, other not ready");
        }
        if(input_offset[i] != 0){
          for (auto &data: input_copy[i]) {
            data += input_offset[i];
          }
        }
        input[i] = input_copy[i].data();
      }
      // apply multiplier
      //#pragma omp parallel for schedule(static)
      for (unsigned i = 0; i < nInputs; ++i) {
        for (size_t j = 0; j < opdT[i]->size(); ++j) {
          input[i][j] = input[i][j] * ((int8_t)quant_multiplier->at(i));
        }
      }
    } else if (getOpQuant(op) == "NONE" || getOpQuant(op) == "BF16") {
      for (unsigned i = 0; i < nInputs; ++i) {
        // make copy
        input_copy[i].assign(opdT[i]->begin(),
                             opdT[i]->end());
        for (auto &data: input_copy[i]) {
          data = data * coeff[i];
        }
        input[i] = input_copy[i].data();
      }
    }
  } else if (type == "MUL") {
    // MUL apply qscale on output put, no scaling on input
  } else {
    llvm_unreachable("unsupported eltwise type");
  }
  //#pragma omp parallel for schedule(static)
  for (size_t ni = 0; ni < nInputs; ++ni) {
    for (size_t i = 0; i < (size_t)(in * ic * ih * iw); ++i) {
      if (ni == 0) { // first input
        output[i] = input[ni][i];
      } else {
        if (type == "ADD") {
          output[i] = output[i] + input[ni][i];
        } else if (type == "MAX") {
          output[i] = output[i] > input[ni][i] ? output[i] : input[ni][i];
        } else if (type == "MIN") {
          output[i] = output[i] < input[ni][i] ? output[i] : input[ni][i];
        }  else if (type == "MUL") {
          output[i] = output[i] * input[ni][i];
        } else {
          llvm_unreachable("unsupported eltwise type");
        }
      }
    }
  }

  // compute in fp32
  int ret = 0;
  // asymmetric no need to do relu, if value < 0, do offset(-128) will saturate
  // to -128
  if (do_relu && !is_asymmetric) {
    ret = my_relu(output, output, in, ic, ih, iw, 0.0f);
    assert(ret == 0);
  }

  // rshift and saturate on output
  if (getOpQuant(op) == "NONE") {
    // do nothing
  } else if (getOpQuant(op) == "INT8") {
    if (getOpQuantParamType(op) != "NONE") {
      if (type == "ADD" || type == "MAX" || type == "MIN") {
        // apply rshift and saturate
        for (int i = 0; i < input_size; ++i) {
          output[i] = (float)applyRShiftAndSaturateInt8(
              output[i], (uint32_t)quant_rshift->at(0), output_offset);
        }
      } else if (type == "MUL") {
        // apply qscale on output (both rshift and saturate)
        for (int i = 0; i < input_size; ++i) {
          output[i] = (float)applyMultiplierAndRShiftAndSaturateInt8(
              output[i], (uint32_t)quant_rshift->at(0),
              (uint32_t)quant_multiplier->at(0), true);
        }
      }
    }
    for (int i = 0; i < input_size; ++i) {
      if (output[i] > 127.0) {
        output[i] = 127.0;
      } else if (output[i] < -128.0) {
        output[i] = -128.0;
      }
    }
  } else if (getOpQuant(op) == "BF16") {
    auto tensor_bf16 = std::make_unique<std::vector<bfloat16>>(resultT->size());
    FloatToBFloat16(resultT->data(), tensor_bf16->data(),
                    resultT->size()); // with rounding
    BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
  } else {
    assert(false);
  }

  bool isReshape = !((ih == oh) && (iw == ow));
  if(isReshape){
    float *output_real = resultReal->data();
    int hReshapeScale = ih/oh;
    int wReshapeScale = iw/ow;
    for(int n_counter = 0;  n_counter < in; n_counter++)
      for(int c_counter = 0;  c_counter < ic; c_counter++)
        for(int h_counter = 0;  h_counter < oh; h_counter++)
          for(int w_counter = 0;  w_counter < ow; w_counter++) {
            int index_old = w_counter * wReshapeScale +
                            h_counter * hReshapeScale * iw +
                            c_counter * iw * ih +
                            n_counter * ic * iw * ih;
            int index_new = w_counter +
                            h_counter * ow +
                            c_counter * ow * oh +
                            n_counter * ic * ow * oh;
            output_real[index_new] = output[index_old];
          }
  }
  if(isReshape){
    valueMapping[result] = std::move(resultReal);
  } else {
    valueMapping[result] = std::move(resultT);
  }

  return success();
}

LogicalResult tpu::EltwiseAddOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);
  const unsigned nInputs = op->getNumOperands() - 4;
  std::vector<float> coeff(nInputs);
  if (this->coeff().hasValue()) {
    arrayAttrToVector(this->coeff().getValue(), coeff);
  } else {
    for(unsigned i = 0; i < nInputs;  i++)
      coeff[i] = 1.0;
  }
  StringRef type = "ADD";
  return doEltwiseOpInterpret(op, type, do_relu(), coeff, valueMapping);
}

LogicalResult tpu::EltwiseMaxOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);
  const unsigned nInputs = op->getNumOperands() - 4;
  std::vector<float> coeff(nInputs);
  for(unsigned i = 0; i < nInputs;  i++)
      coeff[i] = 1.0;
  StringRef type = "MAX";
  return doEltwiseOpInterpret(op, type, do_relu(), coeff, valueMapping);
}

LogicalResult tpu::EltwiseMinOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);
  const unsigned nInputs = op->getNumOperands() - 4;
  std::vector<float> coeff(nInputs);
  for(unsigned i = 0; i < nInputs;  i++)
      coeff[i] = 1.0;
  StringRef type = "MIN";
  return doEltwiseOpInterpret(op, type, do_relu(), coeff, valueMapping);
}

LogicalResult tpu::EltwiseMulOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);
  const unsigned nInputs = op->getNumOperands() - 4;
  std::vector<float> coeff(nInputs);
  for(unsigned i = 0; i < nInputs;  i++)
      coeff[i] = 1.0;
  StringRef type = "MUL";
  return doEltwiseOpInterpret(op, type, do_relu(), coeff, valueMapping);
}

LogicalResult tpu::FullyConnectedOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);

  // parse param
  int m, k, n;
  parseFullyConnectedParam(input(), output(), filter(), m, k, n);
  bool do_relu = this->do_relu();


  std::vector<float> input(opdT[0]->begin(), opdT[0]->end());
  std::shared_ptr<std::vector<float> > filter = opdT[1];
  std::shared_ptr<std::vector<float> > bias = opdT[2];
  std::shared_ptr<std::vector<float> > quant_rshift = opdT[5];
  std::shared_ptr<std::vector<float>> quant_multiplier = opdT[6];

  // int input_offset = 0;
  // int output_offset = 0;
  // bool is_asymmetric = isOpQuantAsymmetric();

  int ret = mkldnn_ip(input.data(), filter->data(),
      bias ? bias->data() : nullptr, resultT->data(), m, k, n, false);
  assert(ret == 0);
  if (do_relu) {
    ret = my_relu(resultT->data(), resultT->data(), 1, 1, 1, n * m, 0.0f);
    assert(ret == 0);
  }

  // rshift and saturate on output
  if (getOpQuant() == "NONE") {
    // do nothing
  } else if (getOpQuant() == "INT8") {
    assert(quant_rshift);
    assert(quant_multiplier);
    for (int i = 0; i < size; ++i) {
      resultT->at(i) = (float)applyMultiplierAndRShiftAndSaturateInt8(
          resultT->at(i), (uint32_t)quant_rshift->at(0),
          quant_multiplier->at(0), true);
    }
  } else if (getOpQuant() == "BF16") {
    auto tensor_bf16 = std::make_unique<std::vector<bfloat16> >(resultT->size());
    FloatToBFloat16(resultT->data(), tensor_bf16->data(), resultT->size()); // with rounding
    BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
  } else {
    llvm_unreachable("unsupported type");
  }

  valueMapping[result] = std::move(resultT);

  return success();
}


LogicalResult tpu::GruOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  llvm_unreachable("complete in GruOpKernel");
  return success();
}

LogicalResult tpu::LstmOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);

  auto resultT = std::make_unique<std::vector<float> >(size);

  assert(opdT.size() == 10);
  std::shared_ptr<std::vector<float> > input = opdT[0];
  std::shared_ptr<std::vector<float> > weight = opdT[1];
  std::shared_ptr<std::vector<float> > recurrence = opdT[2];
  std::shared_ptr<std::vector<float> > bias = opdT[3];
  std::shared_ptr<std::vector<float> > initial_h = opdT[4];
  std::shared_ptr<std::vector<float> > initial_c = opdT[5];

  int seq_len = 0;
  int batch_size = 0;
  int input_size = 0;
  int hidden_size = 0;

  parseLstmParam(this->input(), this->weight(), seq_len, batch_size, input_size, hidden_size);
  my_lstm(input->data(), resultT->data(), weight->data(), recurrence->data(), bias->data(), initial_h->data(), initial_c->data(),
              seq_len, batch_size, input_size, hidden_size, this->bidirectional());

  // rshift and saturate on output
  if (getOpQuant() == "NONE") {
    // do nothing
  } else if (getOpQuant() == "INT8") {
    // gru doesn not implement int8 quantization so far
    assert(0);
  } else if (getOpQuant() == "BF16") {
    auto tensor_bf16 = std::make_unique<std::vector<bfloat16> >(resultT->size());
    FloatToBFloat16(resultT->data(), tensor_bf16->data(), resultT->size()); // with rounding
    BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
  } else {
    llvm_unreachable("unsupported type");
  }

  valueMapping[result] = std::move(resultT);
  return success();
}

LogicalResult tpu::FrcnDetectionOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);

  auto resultT = std::make_unique<std::vector<float>>(size, 0);
  std::vector<int64_t> shape = result.getType().cast<TensorType>().getShape();
  assert(shape.size() == 4);

  std::vector<int64_t> rois_shape;
  rois_shape = getTensorShape(op->getOperand(2));

  float *bbox_deltas = (float *)opdT[0]->data();
  float *scores = (float *)opdT[1]->data();
  float *rois = (float *)opdT[2]->data();

  float *output = (float *)resultT.get()->data();
  int32_t class_num = this->class_num();
  int32_t keep_topk = this->keep_topk();
  float nms_threshold = this->nms_threshold().convertToFloat();
  float obj_threshold = this->obj_threshold().convertToFloat();

  int batch = rois_shape[0];
  int num = rois_shape[2];

  for (int b = 0; b < batch; ++b) {
    auto batched_bbox_deltas = bbox_deltas + b * num * class_num * 4;
    auto batched_scores = scores + b * num * class_num;
    auto batched_rois = rois + b * num * 5;

    std::vector<float> boxes(num * 4, 0);
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < 4; ++j) {
        boxes[i*4 + j] = batched_rois[i*5 + j + 1];
      }
    }

    std::vector<float> pred(num * class_num * 4, 0);
    float *pred_data = pred.data();
    std::vector<float> deltas(batched_bbox_deltas, batched_bbox_deltas + num * class_num * 4);
    bbox_transform_inv(boxes.data(), deltas.data(), pred_data, num, class_num);

    int det_num = 0;
    auto dets = new detections[num];

    for (int i = 0; i < num; ++i) {
      for (int j = 1; j < (int)class_num; ++j) {
        if (batched_scores[i*class_num + j] > obj_threshold) {
          dets[det_num].bbox.x1 = pred[i*class_num*4 + j*4 + 0];
          dets[det_num].bbox.y1 = pred[i*class_num*4 + j*4 + 1];
          dets[det_num].bbox.x2 = pred[i*class_num*4 + j*4 + 2];
          dets[det_num].bbox.y2 = pred[i*class_num*4 + j*4 + 3];
          dets[det_num].cls = j;
          dets[det_num].score = batched_scores[i*class_num + j];
          det_num++;
        }
      }
    }

    nms(dets, det_num, nms_threshold);
    auto dets_nms = new detections[det_num];
    int det_idx = 0;
    for (int i = 0; i < det_num; i++) {
      if (dets[i].score > 0) {
        dets_nms[det_idx] = dets[i];
        det_idx ++;
      }
    }

    if (keep_topk > det_idx)
        keep_topk = det_idx;

    long long count = 0;
    auto batched_output = output + b * shape[1] * shape[2] * shape[3];
    for(int i = 0; i < keep_topk; ++i) {
      batched_output[count++] = dets_nms[i].bbox.x1;
      batched_output[count++] = dets_nms[i].bbox.y1;
      batched_output[count++] = dets_nms[i].bbox.x2;
      batched_output[count++] = dets_nms[i].bbox.y2;
      batched_output[count++] = dets_nms[i].cls;
      batched_output[count++] = dets_nms[i].score;
      // printf("x1: %f, y1: %f, x2: %f, y2: %f, cls: %d, score: %f\n",
      //     dets_nms[i].bbox.x1, dets_nms[i].bbox.y1, dets_nms[i].bbox.x2, dets_nms[i].bbox.y2,
      //     dets_nms[i].cls, dets_nms[i].score);
    }
    delete[] dets_nms;
    delete[] dets;
  }

  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::InputOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  // Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  // auto opdT = getOperandTensors(op, valueMapping);
  // auto result = this->getResult();
  // auto resultT = std::make_unique<std::vector<float> >();

  // // use copy for now
  // resultT->assign(opdT[0]->begin(), opdT[0]->end());

  // valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::InterpOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);
  float *input = (float *)opdT[0]->data();
  float *output = (float *)resultT.get()->data();
  auto pad_beg_ = pad_beg();
  auto pad_end_ = pad_end();
  std::string ctd = coordinate_transformation_mode().str();
  std::vector<int64_t> shape;
  int64_t input_size, in, ic, ih, iw;
  int64_t output_size, on, oc, oh, ow;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, in, ic, ih, iw);

  std::vector<int64_t> output_shape;
  getTensorShapeAndSize(result, output_shape, output_size);
  getNCHW(output_shape, on, oc, oh, ow);

  int height_in_ = ih;
  int width_in_ = iw;
  int height_in_eff_ = height_in_ + pad_beg_ + pad_end_;
  int width_in_eff_ = width_in_ + pad_beg_ + pad_end_;
  int height_out_ = -1;
  int width_out_ = -1;
  if (this->shrink_factor() && !this->zoom_factor()) {
    const int shrink_factor = this->shrink_factor();
    assert(shrink_factor >= 1 && "Shrink factor must be positive");
    height_out_ = (height_in_eff_ - 1) / shrink_factor + 1;
    width_out_ = (width_in_eff_ - 1) / shrink_factor + 1;
  } else if (this->zoom_factor() &&
             !this->shrink_factor()) {
    const int zoom_factor = this->zoom_factor();
    assert(zoom_factor >= 1 && "Zoom factor must be positive");
    height_out_ = height_in_eff_ + (height_in_eff_ - 1) * (zoom_factor - 1);
    width_out_ = width_in_eff_ + (width_in_eff_ - 1) * (zoom_factor - 1);
  } else if (this->height() && this->width()) {
    height_out_  = this->height();
    width_out_  = this->width();
  } else if (this->zoom_factor() &&
             this->shrink_factor()) {
    const int shrink_factor = this->shrink_factor();
    const int zoom_factor = this->zoom_factor();
    assert(shrink_factor >= 1 && "Shrink factor must be positive");
    assert(zoom_factor >= 1 && "Zoom factor must be positive");

    height_out_ = (height_in_eff_ - 1) / shrink_factor + 1;
    width_out_ = (width_in_eff_ - 1) / shrink_factor + 1;
    height_out_ = height_out_ + (height_out_ - 1) * (zoom_factor - 1);
    width_out_ = width_out_ + (width_out_ - 1) * (zoom_factor - 1);
  }
  if (ctd == "align_corners") {
    // TODO: verify pad_end_ > 0
    my_interp(in * ic, input, -pad_beg_, -pad_beg_, height_in_eff_,
              width_in_eff_, height_in_, width_in_, output, 0, 0, height_out_,
              width_out_, height_out_, width_out_);
  } else if (ctd == "half_pixel") {
    my_interp_linear(input, output, in, ic, ih, iw, oh, ow);
  }

  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::LeakyReluOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);

  // parse param
  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(this->input(), shape, input_size);
  assert(input_size == size);
  getNCHW(shape, n, c, h, w);
  float negative_slope = this->negative_slope().convertToFloat();

  // get tensors
  assert(opdT.size() == 9);
  std::shared_ptr<std::vector<float> > quant_pos_rshift = opdT[5];
  std::shared_ptr<std::vector<float> > quant_pos_multiplier = opdT[6];
  std::shared_ptr<std::vector<float> > quant_neg_rshift = opdT[7];
  std::shared_ptr<std::vector<float> > quant_neg_multiplier = opdT[8];

  std::vector<float> input(opdT[0]->begin(), opdT[0]->end());
  int input_offset = 0;
  int output_offset = 0;
  bool is_asymmetric = isOpQuantAsymmetric();

  if (getOpQuant() == "INT8" && is_asymmetric) {
    input_offset = -getPreviousOpZeroPoint(op);
    output_offset = getOpZeroPoint(op);
    if (input_offset != 0) {
      for (size_t i = 0; i < input.size(); i++) {
        input.at(i) += input_offset;
      }
    }
  }

  // compute in fp32
  // skipped because if quantization is needed, the negative_slop
  // will be computed by quant rescale

  // rshift and saturate on output
  if (getOpQuant() == "NONE") {
    int ret = my_relu(input.data(), resultT->data(), n, c, h, w, negative_slope);
    assert(ret == 0);
  } else if (getOpQuant() == "INT8") {
    LLVM_DEBUG(llvm::errs() << "    rshift_pos "
               << std::to_string(quant_pos_rshift->at(0)) << "\n";);
    LLVM_DEBUG(llvm::errs() << "    multiplier_pos "
               << std::to_string(quant_pos_multiplier->at(0)) << "\n";);
    LLVM_DEBUG(llvm::errs() << "    rshift_neg "
               << std::to_string(quant_neg_rshift->at(0)) << "\n";);
    LLVM_DEBUG(llvm::errs() << "    multiplier_neg "
               << std::to_string(quant_neg_multiplier->at(0)) << "\n";);

    bool do_pos_scale = (quant_pos_multiplier->at(0) != 0.0) ? true : false;
    LLVM_DEBUG(llvm::errs() << "    do_pos_scale " << std::to_string(do_pos_scale) << "\n";);

    float *data_i = input.data();
    float *data_o = resultT->data();
    for (int i = 0; i < size; ++i) {
      if (data_i[i] > 0){
        if (do_pos_scale) {
          data_o[i] = (float)applyMultiplierAndRShiftAndSaturateInt8(data_i[i],
              (uint32_t)quant_pos_rshift->at(0), quant_pos_multiplier->at(0), false, output_offset);
        } else {
          data_o[i] = data_i[i];
        }
      } else {
        data_o[i] = (float)applyMultiplierAndRShiftAndSaturateInt8(
            data_i[i], (uint32_t)quant_neg_rshift->at(0),
            quant_neg_multiplier->at(0), false, output_offset);
      }
    }
  } else if (getOpQuant() == "BF16") {
    int ret = my_relu(input.data(), resultT->data(), n, c, h, w, negative_slope);
    assert(ret == 0);
    auto tensor_bf16 = std::make_unique<std::vector<bfloat16> >(resultT->size());
    FloatToBFloat16(resultT->data(), tensor_bf16->data(), resultT->size()); // with rounding
    BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
  } else {
    llvm_unreachable("unsupported type");
  }

  valueMapping[result] = std::move(resultT);

  return success();
}

template <typename OpTy>
static LogicalResult doPool2DOpInterpret(Operation *op, bool is_average,
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  auto castOp = dyn_cast<OpTy>(op);
  assert(castOp);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = op->getResult(0);
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);

  // parse param
  bool is_global, do_relu, count_include_pad;
  int n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  parsePoolParam(castOp.param(), castOp.input(), castOp.output(),
                 n, c, ih, iw, oh, ow,
                 kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
                 is_global, do_relu, count_include_pad);
  std::vector<int64_t> input_shape = {n, c, ih, iw};
  // get tensors
  float *output = resultT->data();
  std::vector<float> input(opdT[0]->begin(), opdT[0]->end());
  bool is_asymmetric = isOpQuantAsymmetric(op);
  if (getOpQuant(op) == "INT8" && is_asymmetric && is_average){
    // [Sam]becasue our hardware(cv1835) can't do offset at input and output
    // therefore, we can't implement average use hardware api
    // in average pool case, we can use depthwise conv replace it
    llvm_unreachable("avg pool can't not do asymmetric quantization with our hardware api, please use conv");
  }
  if (!is_asymmetric && pad_value != 0){
    llvm::errs() << "pad value:" << pad_value << "\n";
    llvm_unreachable("symmetric pad is zero");
  }

  std::shared_ptr<std::vector<float> > quant_rshift = nullptr;
  std::shared_ptr<std::vector<float> > quant_multiplier = nullptr;
  if (is_average) {
    assert(opdT.size() == 5);
    quant_rshift = opdT[3];
    quant_multiplier = opdT[4];
  } else {
    assert(opdT.size() == 1);
  }

  // no qscale on input tensors before f32 compute

  // compute in fp32
  int ret;

  ret = mkldnn_pool(input.data(), output, n, c, ih, iw, oh, ow, kh, kw,
                    sh, sw, pt, pb, pl, pr, is_average, count_include_pad, pad_value);

  assert(ret == 0);
  // apply qscale on output for average pooling, max poolings are bypassed
  if (is_average && getOpQuant(op) == "INT8") {
    assert(quant_rshift && quant_multiplier);
    std::vector<float> conv_result(size);
    // In hardware limitation, we can not put avg pool with large kernel
    // if avg pool ih * iw > local memory, in our hardware
    // need to split it then sum
    // TODO:This case mostly happenend in global average,
    /// if avg pool kernel size bigger than local memory size, todo
    int lmem_size = MInfo::lane_num;
    if ((ih * iw) > ((lmem_size - size) / 2) && is_global) {

      std::vector<int> h_slices;
      int h_slice_size = (int)(((32768 - size) / iw) / 2);
      int total_h = ih;
      while (total_h > 0) {
        if (total_h > h_slice_size) {
          total_h -= h_slice_size;
          h_slices.push_back(h_slice_size);
        } else {
          h_slices.push_back(total_h);
          break;
        }
      }
      int offset = 0;
      std::vector<float> output_data(size, 0);
      for(auto &slice: h_slices){
          int filter_shape = c * slice * kw;
          int g = c;
          int oc = c;
          int dh = 1, dw = 1;
          int input_slice_size = n * c * slice * kw;
          std::vector<float> conv_filter(filter_shape, 1);
          std::vector<float> input_slice(input_slice_size);
          std::vector<float> output_tmp_data(size);
          std::vector<int64_t> tmp_shape = {n, c, slice, iw};
          my_slice(input.data(), input_slice.data(), 2, offset, input_shape, tmp_shape);
          int ret =
              mkldnn_conv(input_slice.data(), conv_filter.data(), NULL,
                          output_tmp_data.data(), n, c, slice, iw, oc, 1, 1,
                          slice, kw, sh, sw, dh, dw, pt, pb, pl, pr, g, 0);
          offset += slice;
          assert(ret == 0);
          for (int64_t i = 0; i < size; ++i) {
            float sum = output_tmp_data[i];
            output_tmp_data[i] = (float)applyMultiplierAndRShiftAndSaturateInt8(
                sum, (uint32_t)quant_rshift->at(0),
                (uint32_t)quant_multiplier->at(0), false);
            output_data[i] += output_tmp_data[i];
          }
      }
      for (int64_t i = 0; i < size; ++i) {
        output[i] = output_data[i];
      }
      valueMapping[result] = std::move(resultT);
      return success();
    }


    // sumulate hw that not support Division,
    // we add it in kernel and divide by (rightshift)
    // it should call by "pool sum", we leverage by depthwise conv
    int filter_shape = c * kh * kw;
    int g = c;
    int oc = c;
    int dh = 1, dw = 1;
    std::vector<float> conv_filter(filter_shape, 1);
    int ret = mkldnn_conv(input.data(), conv_filter.data(), NULL,
        conv_result.data(), n, c, ih, iw, oc, oh, ow, kh, kw,
        sh, sw, dh, dw, pt, pb, pl, pr, g, 0);
    assert(ret == 0);

    for (int64_t i = 0; i < size; ++i) {
      // multiplier is taking avg_const into account
      // restore sum value first
      output[i] = (float)applyMultiplierAndRShiftAndSaturateInt8(
          conv_result[i], (uint32_t)quant_rshift->at(0),
          (uint32_t)quant_multiplier->at(0), false);
    }
  } else if (is_average && getOpQuant(op) == "BF16") {
    auto tensor_bf16 = std::make_unique<std::vector<bfloat16> >(resultT->size());
    FloatToBFloat16(resultT->data(), tensor_bf16->data(), resultT->size()); // with rounding
    BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
  }
  valueMapping[result] = std::move(resultT);

  return success();
}

template <typename OpTy>
static LogicalResult doPool3DOpInterpret(Operation *op, bool is_average,
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  auto castOp = dyn_cast<OpTy>(op);
  assert(castOp);
  assert(!is_average && "Only support max pool");

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = op->getResult(0);
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);

  // parse param
  bool is_global, do_relu, count_include_pad;
  int n, c, id, ih, iw, od, oh, ow, kd, kh, kw, sd, sh, sw;
  int pd0, pd1, pt, pb, pl, pr;
  parsePool3dParam(castOp.param(), castOp.input(), castOp.output(),
                   n, c, id, ih, iw,
                   od, oh, ow,
                   kd, kh, kw,
                   sd, sh, sw,
                   pd0, pd1, pt, pb, pl, pr,
                   is_global, do_relu, count_include_pad);

  // get tensors
  float *input = opdT[0]->data();
  float *output = resultT->data();
  std::shared_ptr<std::vector<float> > quant_rshift = nullptr;
  std::shared_ptr<std::vector<float> > quant_multiplier = nullptr;

  // no qscale on input tensors before f32 compute

  // compute in fp32
  pool3d_float_ref(input, output,
                   n, c, id, ih, iw,
                   od, oh, ow,
                   kd, kh, kw,
                   sd, sh, sw,
                   pd0, pd1, pt, pb, pl, pr);

  // apply qscale on output for average pooling, max poolings are bypassed
  if (is_average && getOpQuant(op) == "INT8") {
    assert(quant_rshift && quant_multiplier);
    std::vector<float> conv_result(size);

    {
      // sumulate hw that not support Division,
      // we add it in kernel and divide by (rightshift)
      // it should call by "pool sum", we leverage by depthwise conv
      int filter_shape = c * kh * kw;
      int g = c;
      int oc = c;
      int dh = 1, dw = 1;
      std::vector<float> conv_filter(filter_shape, 1);
      int ret = mkldnn_conv(input, conv_filter.data(), NULL,
          conv_result.data(), n, c, ih, iw, oc, oh, ow, kh, kw,
          sh, sw, dh, dw, pt, pb, pl, pr, g, 0);
      assert(ret == 0);
    }

    for (int64_t i = 0; i < size; ++i) {
      // multiplier is taking avg_const into account
      // restore sum value first
      float sum;
      if (is_global){
        sum = output[i];
      } else {
        sum = conv_result[i];
        //sum = std::round(output[i] * kh * kw);
      }
      output[i] = (float)applyMultiplierAndRShiftAndSaturateInt8(
          sum, (uint32_t)quant_rshift->at(0),
          (uint32_t)quant_multiplier->at(0), false);
    }
  } else if (getOpQuant(op) == "BF16") {
    auto tensor_bf16 = std::make_unique<std::vector<bfloat16> >(resultT->size());
    FloatToBFloat16(resultT->data(), tensor_bf16->data(), resultT->size()); // with rounding
    BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
  }

  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::LayerNormOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float>>> &valueMapping) {
  assert(0);
  return success();
}

LogicalResult tpu::LrnOneOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float>>> &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name()
                          << "]\n";);
  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float>>(size);

  std::vector<int64_t> input_shape;
  int64_t input_size;
  getTensorShapeAndSize(input(), input_shape, input_size);
  assert(input_shape.size() == 4);
  int n, c, h, w;
  n = input_shape[0];
  c = input_shape[1];
  h = input_shape[2];
  w = input_shape[3];
  std::shared_ptr<std::vector<float>> input = opdT[0];
  int ret =
      my_lrn_one(input->data(), resultT->data(), n, c, h, w,
                 local_size(), alpha().convertToFloat());
  assert(ret == 0);

  valueMapping[result] = std::move(resultT);
  return success();
}

LogicalResult tpu::LrnTwoOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float>>> &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name()
                          << "]\n";);
  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float>>(size);

  std::vector<int64_t> input_shape;
  int64_t input_size;
  getTensorShapeAndSize(input(), input_shape, input_size);
  assert(input_shape.size() == 4);
  int n, c, h, w;
  n = input_shape[0];
  c = input_shape[1];
  h = input_shape[2];
  w = input_shape[3];
  std::shared_ptr<std::vector<float>> input = opdT[0];
  int ret = my_lrn_two(input->data(), resultT->data(), n, c, h, w,
                       local_size());
  assert(ret == 0);

  valueMapping[result] = std::move(resultT);
  return success();
}

LogicalResult tpu::LrnThreeOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float>>> &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name()
                          << "]\n";);
  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float>>(size);

  std::vector<int64_t> input_shape;
  int64_t input_size;
  getTensorShapeAndSize(input(), input_shape, input_size);
  assert(input_shape.size() == 4);
  int n, c, h, w;
  n = input_shape[0];
  c = input_shape[1];
  h = input_shape[2];
  w = input_shape[3];
  std::shared_ptr<std::vector<float>> input = opdT[0];
  int ret = my_lrn_three(input->data(), resultT->data(), n, c, h, w,
                         beta().convertToFloat(), k().convertToFloat());
  assert(ret == 0);

  valueMapping[result] = std::move(resultT);
  return success();
}

LogicalResult tpu::LrnOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float>>> &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name()
                          << "]\n";);
  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float>>(size);

  std::vector<int64_t> input_shape;
  int64_t input_size;
  getTensorShapeAndSize(input(), input_shape, input_size);
  assert(input_shape.size() == 4);
  int n, c, h, w;
  n = input_shape[0];
  c = input_shape[1];
  h = input_shape[2];
  w = input_shape[3];
  int ret = 0;
  auto quant = getOpQuant();
  std::shared_ptr<std::vector<float>> input = opdT[0];
  if (quant == "NONE") {
    std::shared_ptr<std::vector<float>> scaleT = opdT[3];
    if (scaleT == NULL) {
      scaleT = std::make_shared<std::vector<float>>(input_size);
      ret = my_lrn_one(input->data(), scaleT->data(), n, c, h, w, local_size(), alpha().convertToFloat());
      assert(ret == 0);
      ret = my_lrn_two(scaleT->data(), resultT->data(), n, c, h, w,local_size());
      assert(ret == 0);
      ret = my_lrn_three(resultT->data(), scaleT->data(), n, c, h, w,
                         beta().convertToFloat(), k().convertToFloat());
      assert(ret == 0);
    }
    ret = my_lrn_main(input->data(), scaleT->data(), resultT->data(), n, c, h, w);
    assert(ret == 0);
  } else if (quant == "INT8") {
    std::shared_ptr<std::vector<float>> sqr_lut = opdT[1];
    std::shared_ptr<std::vector<float>> power_lut = opdT[2];
    ret = my_lrn_int8(
        input->data(), resultT->data(), n, c, h, w,
        local_size(), sqr_lut->data(), power_lut->data(),
        sum_rshift(), lrn_rshift(),
        quant_data0(), quant_data1());
    assert(ret == 0);
  } else if (quant == "BF16") {
    auto scaleT = std::make_unique<std::vector<float>>(size);
    ret = my_lrn_one(input->data(), scaleT->data(), n, c, h, w,
                     local_size(), alpha().convertToFloat());
    assert(ret == 0);
    ret = my_lrn_two(scaleT->data(), resultT->data(), n, c, h, w,
                     local_size());
    assert(ret == 0);
    ret = my_lrn_three(resultT->data(), scaleT->data(), n, c, h, w,
                       beta().convertToFloat(), k().convertToFloat());
    assert(ret == 0);
    ret =
        my_lrn_main(input->data(), scaleT->data(), resultT->data(), n, c, h, w);
    assert(ret == 0);
  } else {
    llvm::errs() << "Quant not supported:" << quant << "\n";
    assert(false);
  }

  valueMapping[result] = std::move(resultT);
  return success();
}

LogicalResult tpu::NormalizeOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  auto opdT = getOperandTensors(op, valueMapping);

  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name()
                          << "]\n";);

  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float>>(size);

  std::vector<int64_t> input_shape;
  std::vector<int64_t> output_shape;
  int64_t input_size, output_size;

  getTensorShapeAndSize(input(), input_shape, input_size);
  getTensorShapeAndSize(output(), output_shape, output_size);

  assert(input_shape.size() == 4);
  assert(output_shape.size() == 4);

  bool across_spatial = this->across_spatial();
  bool channel_shared = this->channel_shared();

  //implement for ssd case first
  assert(!across_spatial);

  int n, c, h, w;

  n = input_shape[0];
  c = input_shape[1];
  h = input_shape[2];
  w = input_shape[3];

  float *scale = (float *)opdT[1]->data();
  std::shared_ptr<std::vector<float>> input = opdT[0];

  int ret = 0 ;
  ret = my_normalize(input->data(),scale,resultT->data(),across_spatial,channel_shared,n,c,h,w);
  assert(ret == 0);
  valueMapping[result] = std::move(resultT);
  return success();
}

LogicalResult tpu::PermuteOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float>>> &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name()
                          << "]\n";);
  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float>>(size);

  std::vector<int64_t> input_shape;
  int64_t input_size;
  getTensorShapeAndSize(input(), input_shape, input_size);

  assert(input_shape.size() == 4);

  int in, ic, ih, iw, order0, order1, order2, order3;

  in = input_shape[0];
  ic = input_shape[1];
  ih = input_shape[2];
  iw = input_shape[3];

  order0 = this->order0();
  order1 = this->order1();
  order2 = this->order2();
  order3 = this->order3();

  std::shared_ptr<std::vector<float>> input = opdT[0];
  int ret = my_permute(input->data(), resultT->data(), in, ic, ih, iw, order0,
                       order1, order2, order3);
  assert(ret == 0);
  valueMapping[result] = std::move(resultT);
  return success();
}

LogicalResult tpu::PixelShuffleOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float>>> &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name()
                          << "]\n";);
  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float>>(size);

  std::vector<int64_t> input_shape;
  std::vector<int64_t> output_shape;

  int64_t input_size, output_size;
  getTensorShapeAndSize(input(), input_shape, input_size);
  getTensorShapeAndSize(output(), output_shape, output_size);


  int in, ic, ih, iw, on, oc, oh, ow;

  int upscale_factor = this->upscale_factor();
  std::string mode = this->mode().str();
  bool dcr_mode = false;

  if(mode == "DCR"){
    dcr_mode = true;
  }
  in = input_shape[0];
  ic = input_shape[1];
  ih = input_shape[2];
  iw = input_shape[3];


  on = output_shape[0];
  oc = output_shape[1];
  oh = output_shape[2];
  ow = output_shape[3];

  assert(ic == oc * upscale_factor * upscale_factor);
  assert(ih * upscale_factor == oh);
  assert(iw * upscale_factor == ow);

  std::shared_ptr<std::vector<float>> input = opdT[0];
  my_pixelshuffle(input->data(), resultT->data(), in, ic, ih, iw, on, oc, oh,
                  ow, upscale_factor, dcr_mode);
  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::ClipOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float>>> &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name()
                          << "]\n";);
  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float>>(size);

  std::vector<int64_t> input_shape;
  std::vector<int64_t> output_shape;

  int64_t input_size, output_size;
  getTensorShapeAndSize(input(), input_shape, input_size);
  getTensorShapeAndSize(output(), output_shape, output_size);


  int in, ic, ih, iw, on, oc, oh, ow;

  float min = this->min().convertToFloat();
  float max = this->max().convertToFloat();

  in = input_shape[0];
  ic = input_shape[1];
  ih = input_shape[2];
  iw = input_shape[3];


  on = output_shape[0];
  oc = output_shape[1];
  oh = output_shape[2];
  ow = output_shape[3];

  std::shared_ptr<std::vector<float>> input = opdT[0];
  // fp32
  my_clip(input->data(), resultT->data(), in, ic, ih, iw, on,
                          oc, oh, ow, min, max);

  // rshift and saturate on output
  if (mlir::getOpQuant(op) == "NONE") {
    // do nothing
  } else if (mlir::getOpQuant(op) == "INT8") {
    // order depends on \TPUOps.td
    std::shared_ptr<std::vector<float> > quant_rshift = opdT[3];
    std::shared_ptr<std::vector<float> > quant_multiplier = opdT[4];

    for (int i = 0; i < size; ++i) {
      resultT->at(i) = (float)applyMultiplierAndRShiftAndSaturateInt8(
          resultT->at(i), (uint32_t)quant_rshift->at(0),
          (uint32_t)quant_multiplier->at(0), false);
    }
  } else if (mlir::getOpQuant(op) == "BF16") {
    auto tensor_bf16 = std::make_unique<std::vector<bfloat16> >(resultT->size());
    FloatToBFloat16(resultT->data(), tensor_bf16->data(), resultT->size()); // with rounding
    BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
  } else {
    llvm_unreachable("unsupported type");
  }

  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::PoolAvg2DOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);
  return doPool2DOpInterpret<tpu::PoolAvg2DOp>(op, true, valueMapping);
}

LogicalResult tpu::PoolMax2DOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);
  return doPool2DOpInterpret<tpu::PoolMax2DOp>(op, false, valueMapping);
}

LogicalResult tpu::PoolMax3DOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);
  return doPool3DOpInterpret<tpu::PoolMax3DOp>(op, false, valueMapping);
  return failure();
}

LogicalResult tpu::PowerOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float>>> &valueMapping) {
  //Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name()
                          << "]\n";);
#if 0
  if (auto op = dyn_cast<tpu::PowerOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "PowerOp" << "\n";);
    assert(false);
    auto opdT = getOperandTensors(opInst, valueMapping);
    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result.getType().dump(); llvm::errs() << "\n";);
    std::vector<int64_t> shape = result.getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto resultT = std::make_unique<std::vector<float> >(size);

    int nchw[4] = {1, 1, 1, 1};
    float power = op.power().convertToFloat();
    float scale = op.scale().convertToFloat();
    float shift = op.shift().convertToFloat();
    //float rshift = op.rshift().getValue().convertToFloat();
    LLVM_DEBUG(llvm::errs() << "  power" << power << ", scale " << scale << ", shift " << shift << "\n";);
    auto input_type = op.x().getType().cast<TensorType>();
    std::vector<int64_t> i_s(input_type.getShape());
    auto output_type = op.y().getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());
    assert((i_s == o_s) && "input shape not equal to output shape");
    for (uint64_t i = 0; i < i_s.size(); i++) {
      nchw[i] = i_s[i];
    }

    ///
    /// How to get Qscale value:
    ///
    /// X = Sx*Qy
    /// Y = Sy*Qy
    /// Sx = thrx /127
    /// Sy = thry /127
    ///
    /// Y=X*X
    /// ==> Sy*Qy=Sx*Qx*Sx*Qx
    /// ==> Qy = ((thrx*thrx/(127))*(Qx*Qx))/thry
    /// ==> Qscale = (thrx*thrx/127)/thry
    ///

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
#if 1
    if (op.quant() == "INT8"|| op.quant() == "INT8_PER_CHANNEL"||op.quant() == "INT8_MULTIPLIER") {
      assert(threshold_x != 0.0);
      std::vector<int> data(256, 0);

      for (int idx = 0; idx < 256; ++idx) {
        char lutInput = static_cast<char>(idx);
        float index = lutInput * threshold_x / 127.0;
        float lutOutput = pow(index,power) * 127.0 / threshold_y;
        int lutOutputI32 = std::floor(lutOutput + 0.5);
        lutOutputI32 = (lutOutputI32 > 127)
                           ? 127
                           : (lutOutputI32 < -128) ? -128 : lutOutputI32;
        data[idx] = lutOutputI32;
      }
      for (int i = 0; i < size; ++i) {
        output[i] = data[(unsigned char)input[0][i]];
      }
    }
    else
#endif
    {

      if (op.quant() == "INT8"|| op.quant() == "INT8_PER_CHANNEL") {
        scale = scale*(threshold_y/threshold_x)*multiplier;
        shift = shift*(threshold_y/127.0)*multiplier;
        scale = (float)applyRShiftAndSaturateInt8(scale, (uint32_t)rshift);
        shift = (float)applyRShiftAndSaturateInt8(shift, (uint32_t)rshift);
      }else if(op.quant() == "INT8_MULTIPLIER"){
        scale = scale*(threshold_y/threshold_x);
        shift = shift*(threshold_y/127.0);
        scale = (float)applyMultiplierAndRShiftAndSaturateInt8(scale,rshift,  multiplier);
        shift = (float)applyMultiplierAndRShiftAndSaturateInt8(shift,rshift,  multiplier);
      }

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
        /* else if (op.quant() == "BF16"){
              auto tensor_bf16 = std::make_unique<std::vector<bfloat16> >(resultT->size());
              FloatToBFloat16(resultT->data(), tensor_bf16->data(), resultT->size()); // with rounding
              BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
        }*/
      }
    }

    valueMapping[result] = std::move(resultT);
    return success();
  }
#endif

  llvm_unreachable("unsupported op");
  return success();
}

LogicalResult tpu::PReluOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float>>> &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name()
                          << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float>>(size);

  // parse param
  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  assert(input_size == size);
  getNCHW(shape, n, c, h, w);

  // get tensors
  std::shared_ptr<std::vector<float>> input = opdT[0];
  std::shared_ptr<std::vector<float>> negative_slope = opdT[1];
  std::shared_ptr<std::vector<float>> rshift_pos = opdT[6];
  std::shared_ptr<std::vector<float>> multiplier_pos = opdT[7];
  std::shared_ptr<std::vector<float>> rshift_neg = opdT[8];

  // compute in fp32
  my_prelu(input->data(), resultT->data(), n, c, h, w, negative_slope->data());

  if (getOpQuant() == "INT8") {

    assert(rshift_pos);
    assert(rshift_neg);
    assert(multiplier_pos);
    for (int i = 0; i < size; ++i) {
      if (input->at(i) > 0) {
        resultT->at(i) = (float)applyMultiplierAndRShiftAndSaturateInt8(
            resultT->at(i), (uint32_t)rshift_pos->at(0), multiplier_pos->at(0),
            false);
      } else {
        resultT->at(i) = (float)applyRShiftAndSaturateInt8(
            resultT->at(i), (uint32_t)rshift_neg->at(0));
      }
    }
  } else if (getOpQuant() == "BF16") {
    auto tensor_bf16 = std::make_unique<std::vector<bfloat16> >(resultT->size());
    FloatToBFloat16(resultT->data(), tensor_bf16->data(), resultT->size()); // with rounding
    BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
  }
  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::CscOp::interpret(
  DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name()
                          << "]\n";);
  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  std::vector<int64_t> input_shape;
  std::vector<int64_t> output_shape;
  int64_t input_size, n, c, h, w;
  int64_t output_size, on, oc, oh, ow;
  getTensorShapeAndSize(op->getOperand(0), input_shape, input_size);
  getTensorShapeAndSize(result, output_shape, output_size);
  getNCHW(input_shape, n, c, h, w);
  getNCHW(output_shape, on, oc, oh, ow);

  auto resultT = std::make_unique<std::vector<float>>(output_size);

  auto pixel_format = this->pixel_format().str();
  auto aligned = this->aligned();

  if (pixel_format == "YUV420_PLANAR") {
    std::vector<int> orders{0, 1, 2};
    my_yuv420_csc(opdT[0]->data(), resultT->data(), on, oc, oh, ow, orders,
                  getOpQuant() == "NONE" ? 0 : 1);
  } else if (aligned) {
    // do crop to make data unaligned
    std::vector<int64_t> crop_shape(input_shape.begin(), input_shape.end());
    crop_shape[3] = (int)(oc * oh * ow / (c * h));
    std::vector<int> crop_offset{0, 0, 0, 0};
    std::vector<int> indices(4, 0);
    llvm::errs() << "crop_shape: <" << crop_shape[0]
                 << "," << crop_shape[1] << ", " << crop_shape[2]
                 << "," << crop_shape[3] << "\n";

    my_crop(opdT[0]->data(), resultT->data(), input_shape.data(),
          crop_shape.data(), 0, crop_offset.data(), indices.data());
  } else {
    resultT->assign(opdT[0]->begin(), opdT[0]->end());
  }
  valueMapping[result] = std::move(resultT);
  return success();
}

LogicalResult tpu::PriorBoxOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float>>> &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name()
                          << "]\n";);
  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float>>(size);

  std::vector<float> min_size;
  arrayAttrToVector(this->min_size(), min_size);
  std::vector<float> max_size;
  arrayAttrToVector(this->max_size(), max_size);
  std::vector<float> aspect_ratios;
  arrayAttrToVector(this->aspect_ratios(), aspect_ratios);
  std::vector<float> variance;
  arrayAttrToVector(this->variance(), variance);

  bool clip = this->clip();
  bool use_default_aspect_ratio = this->use_default_aspect_ratio();
  float offset = this->offset().convertToFloat();
  float step_w = this->step_w().convertToFloat();
  float step_h = this->step_h().convertToFloat();
  int num_priors = this->num_priors();
  int img_width = this->img_w();
  int img_height = this->img_h();

  std::vector<int64_t> shape1 =
      this->getOperand(1).getType().cast<TensorType>().getShape();
  std::vector<int64_t> shape0 =
      this->getOperand(0).getType().cast<TensorType>().getShape();
  assert(shape1.size() == 4 && shape0.size() == 4);
  const int layer_width = shape0[3];
  const int layer_height = shape0[2];

  if (img_width == 0 || img_height == 0) {
    img_width = shape1[3];
    img_height = shape1[2];
  }

  if (step_w == 0 || step_h == 0) {
    step_w = static_cast<float>(img_width) / layer_width;
    step_h = static_cast<float>(img_height) / layer_height;
  }

  float *top_data = (float *)resultT->data();

  int dim = layer_height * layer_width * num_priors * 4;
  int idx = 0;
  for (int h = 0; h < layer_height; ++h) {
    for (int w = 0; w < layer_width; ++w) {
      float center_x = (w + offset) * step_w;
      float center_y = (h + offset) * step_h;
      float box_width, box_height;
      for (size_t s = 0; s < min_size.size(); ++s) {
        int min_size_ = min_size[s];
        if (use_default_aspect_ratio) {
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
        }

        if (max_size.size() > 0) {
          int max_size_ = max_size[s];
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
        for (size_t r = 0; r < aspect_ratios.size(); ++r) {
          float ar = aspect_ratios[r];
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
  if (clip) {
    for (int d = 0; d < dim; ++d) {
      top_data[d] = std::min<float>(std::max<float>(top_data[d], 0.), 1.);
    }
  }

  auto output_type = this->output().getType().cast<TensorType>();
  std::vector<int64_t> o_s(output_type.getShape());

  // set the variance.
  top_data += (o_s[2]);

  int count = 0;
  for (int h = 0; h < layer_height; ++h) {
    for (int w = 0; w < layer_width; ++w) {
      for (int i = 0; i < num_priors; ++i) {
        for (int j = 0; j < 4; ++j) {
          top_data[count] = variance[j];
          ++count;
        }
      }
    }
  }

  valueMapping[result] = std::move(resultT);
  return success();
}

LogicalResult tpu::ProposalOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float>>(size);
  std::vector<int64_t> shape = result.getType().cast<TensorType>().getShape();
  assert(shape.size() == 4);

  std::vector<int64_t> score_shape, bbox_shape;
  int64_t score_size;
  getTensorShapeAndSize(op->getOperand(0), score_shape, score_size);
  bbox_shape = getTensorShape(op->getOperand(1));
  int batch = score_shape[0];
  int channel = score_shape[1];
  int height = score_shape[2];
  int width = score_shape[3];

  float *score = (float *)opdT[0]->data();
  float *bbox_deltas = (float *)opdT[1]->data();
  float *output = (float *)resultT.get()->data();
  auto net_input_h = this->net_input_h();
  auto net_input_w = this->net_input_w();
  auto feat_stride = this->feat_stride();
  auto anchor_base_size = this->anchor_base_size();
  auto rpn_obj_threshold = this->rpn_obj_threshold().convertToFloat();
  auto rpn_nms_threshold = this->rpn_nms_threshold().convertToFloat();
  auto rpn_nms_post_top_n = this->rpn_nms_post_top_n();

  std::vector<float> anchor_scale = {8, 16, 32};
  std::vector<float> anchor_ratio = {0.5, 1, 2};

  std::vector<float> anchor_boxes;
  generate_anchors(anchor_base_size, anchor_scale, anchor_ratio, anchor_boxes);

  float thresh = rpn_obj_threshold;

  for (int b = 0; b < batch; ++b) {
    auto batched_score = score + b * channel * height * width;
    auto batched_bbox_deltas = bbox_deltas + b * bbox_shape[1] * bbox_shape[2] * bbox_shape[3];
    std::vector<std::vector<float>> select_anchor;
    std::vector<float> confidence;
    std::vector<std::vector<float>> bbox;
    int anchor_num = anchor_scale.size() * anchor_ratio.size();

    for (int k = 0; k < anchor_num; k++) {
      float w = anchor_boxes[4 * k + 2] - anchor_boxes[4 * k] + 1;
      float h = anchor_boxes[4 * k + 3] - anchor_boxes[4 * k + 1] + 1;
      float x_ctr = anchor_boxes[4 * k] + 0.5 * (w - 1);
      float y_ctr = anchor_boxes[4 * k + 1] + 0.5 * (h - 1);

      for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
          if (batched_score[anchor_num * height * width + (k * height + i) * width + j] >= thresh) {
            std::vector<float> tmp_anchor;
            std::vector<float> tmp_bbox;

            tmp_anchor.push_back(j * feat_stride+ x_ctr);
            tmp_anchor.push_back(i * feat_stride+ y_ctr);
            tmp_anchor.push_back(w);
            tmp_anchor.push_back(h);
            select_anchor.push_back(tmp_anchor);
            confidence.push_back(batched_score[anchor_num * height * width + (k * height + i) * width + j]);
            tmp_bbox.push_back(batched_bbox_deltas[(4 * k * height + i) * width + j]);
            tmp_bbox.push_back(batched_bbox_deltas[((4 * k +1) * height + i) * width + j]);
            tmp_bbox.push_back(batched_bbox_deltas[((4 * k + 2) * height + i) * width + j]);
            tmp_bbox.push_back(batched_bbox_deltas[((4 * k + 3) * height + i) * width + j]);
            bbox.push_back(tmp_bbox);
          }
        }
      }
    }
    std::vector<std::vector<float>> pred_boxes;
    anchor_box_transform_inv(net_input_w, net_input_h, bbox, select_anchor, pred_boxes);
    anchor_box_nms(pred_boxes, confidence, rpn_nms_threshold);
    int num = pred_boxes.size() > rpn_nms_post_top_n ? rpn_nms_post_top_n : pred_boxes.size();

    auto batched_output = output + b * shape[1] * shape[2] * shape[3];
    for (int i = 0; i < num; i++) {
      batched_output[5 * i] = b;
      batched_output[5 * i + 1] = pred_boxes[i][0];
      batched_output[5 * i + 2] = pred_boxes[i][1];
      batched_output[5 * i + 3] = pred_boxes[i][2];
      batched_output[5 * i + 4] = pred_boxes[i][3];
    }
  }

  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::ReluOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);

  // parse param
  std::vector<int64_t> shape;
  int64_t input_size, n, c, d, h, w;
  getTensorShapeAndSize(this->input(), shape, input_size);
  assert(input_size == size);
  if (shape.size() == 5) {
    getNCDHW(shape, n, c, d, h, w);
    w = h * w;
    h  = d;
  } else {
    getNCHW(shape, n, c, h, w);
  }

  // get tensors
  assert(opdT.size() == 1);
  std::shared_ptr<std::vector<float> > input = opdT[0];
#ifdef USE_GPU
  // compute in fp32
  int ret;
  if (dm == DeviceMode::GPU){
    ret = gpu_relu(input->data(), resultT->data(), n, c, h, w, 0.0f);
  }else{
    ret = my_relu(input->data(), resultT->data(), n, c, h, w, 0.0f);
  }
#else
  int ret = my_relu(input->data(), resultT->data(), n, c, h, w, 0.0f);
#endif
  assert(ret == 0);

  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::ReorgOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float>>(size);
  std::vector<int64_t> shape = result.getType().cast<TensorType>().getShape();
  assert(shape.size() == 4);

  std::vector<int64_t> input_shape, output_shape;
  int64_t input_size, output_size;
  getTensorShapeAndSize(this->input(), input_shape, input_size);
  getTensorShapeAndSize(this->output(), output_shape, output_size);

  int64_t n = input_shape[0];
  int64_t c = input_shape[1];
  int64_t h = input_shape[2];
  int64_t w = input_shape[3];

  float *input = (float *)opdT[0]->data();
  float *output = (float *)resultT.get()->data();
  auto stride = this->stride();
  int ret = my_reorg(input, output, stride, n, c, h, w);

  assert(ret == 0);

  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::ReshapeOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);

  // use copy for now
  resultT.get()->assign(opdT[0]->begin(), opdT[0]->end());

  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::RetinaFaceDetectionOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float>>> &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  std::vector<int64_t> output_shape;
  int64_t output_size;
  getTensorShapeAndSize(result, output_shape, output_size);
  auto resultT = std::make_unique<std::vector<float>>(output_size);
  auto output_data = resultT->data();

  auto confidence_threshold = this->confidence_threshold().convertToFloat();
  auto nms_threshold = this->nms_threshold().convertToFloat();
  auto keep_topk = this->keep_topk();

  std::vector<AnchorCfg> cfg;
  std::unordered_map<std::string, std::vector<AnchorBox>> um_anchors_fpn;
  std::unordered_map<std::string, int> um_num_anchors;
  std::vector<int> feature_stride_fpn{32, 16, 8};

  AnchorCfg cfg1(32, {32, 16}, 16, {1.0}, 9999);
  AnchorCfg cfg2(16, {8, 4}, 16, {1.0}, 9999);
  AnchorCfg cfg3(8, {2, 1}, 16, {1.0}, 9999);
  cfg.push_back(cfg1);
  cfg.push_back(cfg2);
  cfg.push_back(cfg3);

  um_anchors_fpn.clear();
  auto anchors = generate_anchors_fpn(false, cfg);
  for (int i = 0; i < (int)feature_stride_fpn.size(); ++i) {
    std::string key = "stride" + std::to_string(feature_stride_fpn[i]);
    um_anchors_fpn[key] = anchors[i];
    um_num_anchors[key] = anchors[i].size();
  }

  int input_count = opT.size();
  assert(input_count == 9);

  auto batch = output_shape[0];

  for (int b = 0; b < batch; ++b) {
    std::vector<FaceInfo> infos;
    for (size_t i = 0; i < feature_stride_fpn.size(); ++i) {
      int stride = feature_stride_fpn[i];

      size_t landmark_count = opT[input_count-3*i-1]->size() / batch;
      auto landmark_data = opT[input_count-3*i-1]->data() + b * landmark_count;

      size_t bbox_count = opT[input_count-3*i-2]->size() / batch;
      auto bbox_data = opT[input_count-3*i-2]->data() + b * bbox_count;

      size_t score_count = opT[input_count-3*i-3]->size() / batch;
      auto score_data = opT[input_count-3*i-3]->data() + b * score_count;

      auto shape = getTensorShape(op->getOperand(input_count-3*i-1));
      assert(shape.size() == 4);

      size_t height = shape[2];
      size_t width = shape[3];

      std::vector<float> score(score_data + score_count / 2, score_data + score_count);
      std::vector<float> bbox(bbox_data, bbox_data + bbox_count);
      std::vector<float> landmark(landmark_data, landmark_data + landmark_count);

      int count = height * width;
      std::string key = "stride" + std::to_string(stride);
      auto anchors_fpn = um_anchors_fpn[key];
      auto num_anchors = um_num_anchors[key];

      std::vector<AnchorBox> anchors = anchors_plane(height, width, stride, anchors_fpn);

      for (int num = 0; num < num_anchors; ++num) {
        for (int j = 0; j < (int)count; ++j) {
          float confidence = score[j+count*num];
          if (confidence <= confidence_threshold)
            continue;

          float dx = bbox[j+count*(0+num*4)];
          float dy = bbox[j+count*(1+num*4)];
          float dw = bbox[j+count*(2+num*4)];
          float dh = bbox[j+count*(3+num*4)];
          std::vector<float> bbox_deltas{dx,dy,dw,dh};
          auto bbox = bbox_pred(anchors[j+count*num], bbox_deltas);

          std::vector<float> landmark_deltas(10,0);
          for (int k = 0; k < 5; ++k) {
            landmark_deltas[k] = landmark[j+count*(num*10+k*2)];
            landmark_deltas[k+5] = landmark[j+count*(num*10+k*2+1)];
          }

          auto pts = landmark_pred(anchors[j+count*num], landmark_deltas);

          FaceInfo info;
          info.x1 = bbox[0];
          info.y1 = bbox[1];
          info.x2 = bbox[2];
          info.y2 = bbox[3];
          info.score = confidence;
          for(int idx = 0; idx < 5; ++idx) {
              info.x[idx] = pts[idx];
              info.y[idx] = pts[idx+5];
          }

          infos.push_back(info);
        }
      }
    }

    auto preds = nms(infos, nms_threshold);
    if (keep_topk > preds.size())
        keep_topk = preds.size();

    long long count = 0;
    auto batch_output_data = output_data + b * output_size / batch;
    for(int i = 0; i < (int)keep_topk; ++i) {
      batch_output_data[count++] = preds[i].x1;
      batch_output_data[count++] = preds[i].y1;
      batch_output_data[count++] = preds[i].x2;
      batch_output_data[count++] = preds[i].y2;
      batch_output_data[count++] = preds[i].score;
      for(int j = 0; j < 5; ++j) {
        batch_output_data[count++] = preds[i].x[j];
        batch_output_data[count++] = preds[i].y[j];
      }

      LLVM_DEBUG(llvm::errs() << "x1= " << preds[i].x1 << ",y1= " << preds[i].y1
                << ",x2= " << preds[i].x2 << ",y2= " << preds[i].y2
                << ", score= " << preds[i].score
                << ", pts1= " << preds[i].x[0] << ", pts2= " << preds[i].y[0]
                << ", pts3= " << preds[i].x[1] << ", pts4= " << preds[i].y[1]
                << ", pts5= " << preds[i].x[2] << ", pts6= " << preds[i].y[2]
                << ", pts7= " << preds[i].x[3] << ", pts8= " << preds[i].y[3]
                << ", pts9= " << preds[i].x[4] << ", pts10= " << preds[i].y[4] << "\n";);
    }
  }

  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::ROIPoolingOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float>>(size, std::numeric_limits<float>::min());
  std::vector<int64_t> shape = result.getType().cast<TensorType>().getShape();
  assert(shape.size() == 4);

  std::vector<int64_t> rois_shape, data_shape;
  int64_t rois_size, data_size;
  getTensorShapeAndSize(op->getOperand(0), data_shape, data_size);
  getTensorShapeAndSize(op->getOperand(1), rois_shape, rois_size);

  float *data = (float *)opdT[0]->data();
  float *rois = (float *)opdT[1]->data();
  float *output = (float *)resultT.get()->data();
  auto pooled_h = this->pooled_h();
  auto pooled_w = this->pooled_w();
  auto spatial_scale = this->spatial_scale().convertToFloat();

  int ret = my_roipooling(data, rois, output, pooled_h, pooled_w, spatial_scale, rois_shape[0],
          rois_shape[2], data_shape[1], data_shape[2], data_shape[3]);

  assert(ret == 0);

  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::ScaleOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);

  // parse param
  std::vector<int64_t> shape;
  int64_t input_size, n, c, d, h, w;
  getTensorShapeAndSize(this->input(), shape, input_size);
  assert(input_size == size);
  if (shape.size() == 5) {
    getNCDHW(shape, n, c, d, h, w);
    llvm::errs() << "n: " << n << ", c: " << c << ", d: " << d << ", h: " << h << ", w: " << w << "\n";
    w = h * w;
    h = d;
  } else {
    getNCHW(shape, n, c, h, w);
  }
  bool do_relu = this->do_relu();

  std::shared_ptr<std::vector<float> > input = opdT[0];
  std::shared_ptr<std::vector<float> > scale = opdT[1];
  std::shared_ptr<std::vector<float> > bias = opdT[2];

  int ret = my_scale(input->data(), scale->data(), bias?bias->data():nullptr,
                     resultT->data(), n, c, h, w);
  assert(ret == 0);
  if (do_relu) {
    my_relu(resultT->data(), resultT->data(), n, c, h, w, 0.0f);
  }

  valueMapping[result] = std::move(resultT);
  return success();
}

LogicalResult tpu::ScaleLutOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);

  // parse param
  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(this->input(), shape, input_size);
  assert(input_size == size);
  getNCHW(shape, n, c, h, w);

  std::shared_ptr<std::vector<float> > input = opdT[0];
  std::shared_ptr<std::vector<float> > table = opdT[1];
  for (int ni = 0; ni < n; ++ni) {
    for (int ci = 0; ci < c; ++ci) {
      for (int i = 0; i < h * w; ++i) {
        int index = ni * c * h * w + ci * h * w + i;
        auto x = input->at(index);
        auto y = table->at((int)(ci * 256 + x));
        resultT->at(index) = y;
      }
    }
  }

  valueMapping[result] = std::move(resultT);
  return success();
}

LogicalResult tpu::ReverseOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float>>> &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name()
                          << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float>>(size);
  std::vector<int64_t> shape = result.getType().cast<TensorType>().getShape();
  int64_t n, c, h, w;
  getNCHW(shape, n, c, h, w);

  float *input = (float *)opdT[0]->data();
  float *output = (float *)resultT.get()->data();
  int32_t axis = this->axis();
  int ret = my_reverse(input, output, n, c, h, w, axis);

  assert(ret == 0);

  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::ShuffleChannelOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float>>> &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float>>(size);
  std::vector<int64_t> shape = result.getType().cast<TensorType>().getShape();
  assert(shape.size() >= 2);

  std::vector<int64_t> input_shape, output_shape;
  int64_t input_size, output_size;
  getTensorShapeAndSize(this->input(), input_shape, input_size);
  getTensorShapeAndSize(this->output(), output_shape, output_size);

  assert((input_shape == output_shape) && "input shape not equal to output shape");
  assert((input_shape.size() >= 2) && "ShuffleChannel support shape size  must >= 2");

  int64_t n = input_shape[0];
  int64_t c = input_shape[1];
  int64_t frame_size = std::accumulate(
      input_shape.begin() + 2, input_shape.end(), 1, std::multiplies<>());
  float *input = (float *)opdT[0]->data();
  float *output = (float *)resultT.get()->data();
  uint32_t group = this->group();
  int ret = my_shuffle_channel(input, output, group, n, c, frame_size);

  assert(ret == 0);

  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::SliceOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);

  std::vector<int64_t> input_shape;
  int64_t input_size;
  getTensorShapeAndSize(this->input(), input_shape, input_size);
  std::vector<int64_t> output_shape;
  int64_t output_size;
  getTensorShapeAndSize(this->output(), output_shape, output_size);
  int axis = this->axis();
  int offset = this->offset();

  int ret = my_slice(opdT[0]->data(), resultT->data(), axis, offset,
                     input_shape, output_shape);
  assert(ret == 0);
  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::CustomOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float>>> &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);
  auto result = this->getResult();
  auto operandTensors = getOperandTensors(op, valueMapping);
  auto operandShapes = getOperandShapes(op);
  auto resultTensor = std::make_shared<std::vector<float>>(getTensorSize(result));
  auto resultShape = getTensorShape(result);

  cvi::OpParam param;
  convertAttributesToOpParam(this->param(), param);

  auto& pluginFile = ModuleInterpreter::getCustomOpPluginFile();
  cvi::CustomOpPlugin *plugin = cvi::CustomOpPlugin::load(pluginFile);
  assert(plugin);

  if (getOpQuant() == "NONE") {
    plugin->fp32Interpret(
        operation_name().str().c_str(), param, operandTensors,
        operandShapes, resultTensor, resultShape);
  } else if (getOpQuant() == "INT8") {
    plugin->int8Interpret(
        operation_name().str().c_str(), param, operandTensors,
        operandShapes, resultTensor, resultShape);
  } else if (getOpQuant() == "BF16") {
    plugin->bf16Interpret(
        operation_name().str().c_str(), param, operandTensors,
        operandShapes, resultTensor, resultShape);
  } else {
    llvm_unreachable("unsupported type");
  }
  valueMapping[result] = resultTensor;
  return success();
}

LogicalResult tpu::SoftmaxOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  std::vector<int64_t> shape = getTensorShape(result);
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);

  // parse param
  int axis = this->axis();
  bool isBf16 = getOpQuant() == "BF16";

  if (shape.size() == 2) {
    int ret = my_softmax2D(opdT[0]->data(), resultT->data(), shape[0], shape[1], isBf16);
    assert(ret == 0);
  } else if (shape.size() == 4) {
    int ret = my_softmax4D(opdT[0]->data(), resultT->data(), axis, shape, isBf16);
    assert(ret == 0);
  } else if (shape.size() == 3) {
    int ret = my_softmax3D(opdT[0]->data(), resultT->data(), axis, shape, isBf16);
    assert(ret == 0);
  }

  if (getOpQuant() == "NONE") {
    // do nothing
  } else if (getOpQuant() == "INT8") {
    // softmax doesn not implement int8 quantization so far
    assert(0);
  } else if (getOpQuant() == "BF16") {
    auto tensor_bf16 = std::make_unique<std::vector<bfloat16> >(resultT->size());
    FloatToBFloat16(resultT->data(), tensor_bf16->data(), resultT->size()); // with rounding
    BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
  } else {
    llvm_unreachable("unsupported type");
  }

  valueMapping[result] = std::move(resultT);
  return success();
}

LogicalResult tpu::SoftPlusOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float>>> &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name()
                          << "]\n";);
  StringRef type = "SoftPlus";
  return doLUTOpInterpret(op, type, valueMapping);
}

LogicalResult tpu::SoftmaxCpuOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  std::vector<int64_t> shape = getTensorShape(result);
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);

  // parse param
  int axis = this->axis();

  if (shape.size() == 2) {
    int ret = my_softmax2D(opdT[0]->data(), resultT->data(), shape[0], shape[1], false);
    assert(ret == 0);
  } else if (shape.size() == 4) {
    int ret = my_softmax4D(opdT[0]->data(), resultT->data(), axis, shape, false);
    assert(ret == 0);
  } else if (shape.size() == 3) {
    int ret = my_softmax3D(opdT[0]->data(), resultT->data(), axis, shape, false);
    assert(ret == 0);
  }

  if (getOpQuant() == "NONE") {
    // do nothing
  } else if (getOpQuant() == "INT8") {
    // softmax doesn not implement int8 quantization so far
    assert(0);
  } else if (getOpQuant() == "BF16") {
    auto tensor_bf16 = std::make_unique<std::vector<bfloat16> >(resultT->size());
    FloatToBFloat16(resultT->data(), tensor_bf16->data(), resultT->size()); // with rounding
    BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
  } else {
    llvm_unreachable("unsupported type");
  }

  valueMapping[result] = std::move(resultT);
  return success();
}

LogicalResult tpu::SquareOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  std::vector<int64_t> shape = getTensorShape(result);
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);

  size_t total = std::accumulate(shape.begin(), shape.end(), 1,
                                 std::multiplies<int64_t>());
  float *input = opdT[0]->data();
  float *output = resultT->data();
  for (size_t i = 0; i < total; ++i) {
    output[i] = input[i] * input[i];
  }

  if (getOpQuant() == "NONE") {
    // do nothing here
  } else if (getOpQuant() == "BF16") {
    auto tensor_bf16 = std::make_unique<std::vector<bfloat16> >(resultT->size());
    FloatToBFloat16(resultT->data(), tensor_bf16->data(), resultT->size()); // with rounding
    BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
  } else {
    llvm_unreachable("unsupported type");
  }

  valueMapping[result] = std::move(resultT);
  return success();
}

LogicalResult tpu::QuadraticSumOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);
  int axis = this->axis();
  assert(axis == 1);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  std::vector<int64_t> shape = getTensorShape(result);
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);

////
  auto input_shape = getTensorShape(op->getOperand(0));
  assert(input_shape.size() == 4);
  int n = input_shape[0];
  int c = input_shape[1];
  int h = input_shape[2];
  int w = input_shape[3];

  assert(h < 32);
  assert(w < 32);

  // compute in fp32
  mkldnn_conv(opdT[0]->data(), opdT[0]->data(), nullptr,
              resultT->data(), n, c, h, w, c, 1, 1, h, w,
              h, w, 1, 1, 0, 0, 0, 0, c, 0);
  // rshift and saturate on output
  if (getOpQuant() == "NONE" || this->high_precision()) {
    // do nothing
  } else if (getOpQuant() == "BF16") {
    auto tensor_bf16 = std::make_unique<std::vector<bfloat16> >(resultT->size());
    FloatToBFloat16(resultT->data(), tensor_bf16->data(), resultT->size()); // with rounding
    BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
  } else {
    llvm_unreachable("unsupported type");
  }

  valueMapping[result] = std::move(resultT);
  return success();
}

LogicalResult tpu::MatMulOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);

  // parse param
  int m, k, n;
  auto opd_left = op->getOperand(0);
  auto opd_right = op->getOperand(1);

  auto left_type = opd_left.getType().template cast<TensorType>();
  std::vector<int64_t> left_shape(left_type.getShape());
  auto right_type = opd_right.getType().cast<TensorType>();
  std::vector<int64_t> right_shape(right_type.getShape());
  auto output_type = output().getType().template cast<TensorType>();
  std::vector<int64_t> output_shape(output_type.getShape());
  llvm::errs() << "left:" << left_shape[0] << " " << left_shape[1]
               << " right:" << right_shape[0] << " " << right_shape[1]
               << " output:" << output_shape[0] << " " << output_shape[1]
               << "\n";
  assert((left_shape[0] == output_shape[0]) && "input M not equal to output M");
  m = left_shape[0];
  assert((left_shape[1] == right_shape[0]) && "input K not equal to filter K");
  k = left_shape[1];
  assert((right_shape[1] == output_shape[1]) && "filter N not equal to output N");
  n = output_shape[1];
  bool do_relu = this->do_relu();

  std::vector<float> transposed_right(opdT[1]->size());
  float *right_data = opdT[1]->data();
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < n; j++) {
      transposed_right[j * k + i] = right_data[i * n  + j];
    }
  }

  int ret = mkldnn_ip(opdT[0]->data(), transposed_right.data(), nullptr, resultT->data(), m, k, n, false);
  assert(ret == 0);
  if (do_relu) {
    ret = my_relu(resultT->data(), resultT->data(), 1, 1, 1, n * m, 0.0f);
    assert(ret == 0);
  }

  if (getOpQuant() == "NONE") {
    // do nothing here
  } else if (getOpQuant() == "BF16") {
    auto tensor_bf16 = std::make_unique<std::vector<bfloat16> >(resultT->size());
    FloatToBFloat16(resultT->data(), tensor_bf16->data(), resultT->size()); // with rounding
    BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
  } else {
    llvm_unreachable("unsupported type");
  }

  valueMapping[result] = std::move(resultT);
  return success();
}

LogicalResult tpu::SwapChannelOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float>>> &valueMapping) {

  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name()
                          << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float>>(size);
  std::vector<int64_t> shape = result.getType().cast<TensorType>().getShape();
  assert(shape.size() == 4);

  std::vector<int64_t> input_shape, output_shape;
  int64_t input_size, output_size;
  getTensorShapeAndSize(this->input(), input_shape, input_size);
  getTensorShapeAndSize(this->output(), output_shape, output_size);

  assert((input_shape == output_shape) &&
         "input shape not equal to output shape");
  assert((input_shape.size() == 4) &&
         "SwapChannel support shape size  must == 4");
  std::vector<int32_t> order;
  arrayAttrToVector(this->channel_order().getValue(), order);

  float *input = (float *)opdT[0]->data();
  float *output = (float *)resultT.get()->data();
  int ret = my_swap_channel(input, output, input_shape[0], input_shape[1],
                            input_shape[2], input_shape[3], order.data());
  assert(ret == 0);
  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::TileOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float>>> &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float>>(size);
  float *input = (float *)opdT[0]->data();
  float *output = (float *)resultT.get()->data();

  // input
  std::vector<int64_t> input_shape;
  int64_t input_size;
  getTensorShapeAndSize(op->getOperand(0), input_shape, input_size);

  // output
  std::vector<int64_t> output_shape;
  int64_t output_size;
  getTensorShapeAndSize(this->output(), output_shape, output_size);

  // get scale info
  std::vector<int32_t> resp;
  arrayAttrToVector(this->resp().getValue(), resp);

  my_tile(input, output, input_shape, output_shape, resp);

  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::TileInterpOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);
  float *input = (float *)opdT[0]->data();
  float *output = (float *)resultT.get()->data();

  // input
  std::vector<int64_t> shape;
  int64_t input_size, in, ic, ih, iw;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, in, ic, ih, iw);

  // output
  std::vector<int64_t> output_shape;
  int64_t output_size, oh, ow;
  getTensorShapeAndSize(this->output(), output_shape, output_size);
  oh = output_shape[2];
  ow = output_shape[3];

  // get scale info
  std::vector<int32_t> resp;
  arrayAttrToVector(this->resp().getValue(), resp);
  assert(resp.size() == 2 && "oonly support h/w tile");

  // check oh/ow is valid
  int interpOw = ow / resp[0]; // w
  int interpOh = oh / resp[1]; // h
  int interpIw = iw / 2; // 2 for deconv twice in w-axis, plz refer \getTwiceWDeConv
  int interpIh = ih;

  my_interptile(input, output, in, ic, interpOh, interpOw, interpIh, interpIw);

  if (mlir::getOpQuant(op) == "NONE") {
    // do nothing
  } else if (mlir::getOpQuant(op) == "INT8") {
    // order depends on \TPUOps.td
    std::shared_ptr<std::vector<float> > quant_rshift = opdT[3];
    std::shared_ptr<std::vector<float> > quant_multiplier = opdT[4];

    for (int i = 0; i < size; ++i) {
      resultT->at(i) = (float)applyMultiplierAndRShiftAndSaturateInt8(
          resultT->at(i), (uint32_t)quant_rshift->at(0),
          (uint32_t)quant_multiplier->at(0), true);
    }
  } else if (mlir::getOpQuant(op) == "BF16") {
    auto tensor_bf16 = std::make_unique<std::vector<bfloat16> >(resultT->size());
    FloatToBFloat16(resultT->data(), tensor_bf16->data(), resultT->size()); // with rounding
    BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
  } else {
    llvm_unreachable("unsupported type");
  }

  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::UpsampleOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);

  // parse param
  std::vector<int64_t> input_shape;
  int64_t input_size, n, c, ih, iw;
  getTensorShapeAndSize(this->input(), input_shape, input_size);
  getNCHW(input_shape, n, c, ih, iw);
  std::vector<int64_t> output_shape;
  int64_t output_size, oh, ow;
  getTensorShapeAndSize(this->output(), output_shape, output_size);
  oh = output_shape[2];
  ow = output_shape[3];
  int64_t scale_h = this->scale_h();
  int64_t scale_w = this->scale_w();
  assert(oh == ih * scale_h);
  assert(ow == iw * scale_w);

  // get tensors
  // assert(opdT.size() == 1);
  std::shared_ptr<std::vector<float> > input = opdT[0];

  // compute in fp32
  int ret = my_upsample(input->data(), resultT->data(), n, c, ih, iw, scale_h, scale_w);
  assert(ret == 0);

  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::PadOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float>>> &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  std::shared_ptr<std::vector<float> > input = opdT[0];
  auto result = this->getResult();
  std::vector<int64_t> shape = getTensorShape(result);
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);

  // parse param
  std::vector<int32_t> pads;
  auto const_val = this->const_val().convertToFloat();
  arrayAttrToVector(this->pads().getValue(), pads);

  std::vector<int64_t> input_shape = getTensorShape(this->input());

  int on = pads[0] + pads[4] + input_shape[0];
  int oc = pads[1] + pads[5] + input_shape[1];
  int oh = pads[2] + pads[6] + input_shape[2];
  int ow = pads[3] + pads[7] + input_shape[3];

  assert(on == shape[0]);
  assert(oc == shape[1]);
  assert(oh == shape[2]);
  assert(ow == shape[3]);

  int ret = my_pad_constant(input->data(), resultT->data(), input_shape, pads,
                            const_val);
  assert(ret == 0);
  valueMapping[result] = std::move(resultT);
  return success();
}

LogicalResult tpu::ReduceL2Op::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float>>> &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  std::shared_ptr<std::vector<float> > input = opdT[0];
  auto result = this->getResult();
  std::vector<int64_t> shape = getTensorShape(result);
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);

  // parse param
  std::vector<int32_t> axes;
  arrayAttrToVector(this->axes().getValue(), axes);

  std::vector<int64_t> input_shape = getTensorShape(this->input());

  if (getOpQuant() == "INT8") {
    llvm_unreachable("unsupported int8 for reducel2");
  } else {
    // float32, bf16
    int ret = my_reduce_l2(input->data(), resultT->data(), input_shape, axes);
    assert(ret == 0);
  }

  valueMapping[result] = std::move(resultT);
  return success();
}

LogicalResult tpu::ReduceMeanOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float>>> &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  std::shared_ptr<std::vector<float> > input = opdT[0];
  auto result = this->getResult();
  std::vector<int64_t> shape = getTensorShape(result);
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);

  // parse param
  std::vector<int32_t> axes;
  arrayAttrToVector(this->axes().getValue(), axes);

  std::vector<int64_t> input_shape = getTensorShape(this->input());

  if (getOpQuant() == "INT8") {
    // int8
    std::shared_ptr<std::vector<float> > quant_rshift = nullptr;
    std::shared_ptr<std::vector<float> > quant_multiplier = nullptr;
    assert(opdT.size() == 5);
    quant_rshift = opdT[3];
    quant_multiplier = opdT[4];

    int ret = my_reduce_mean_int8(input->data(), resultT->data(), input_shape,
                                  axes, (int)quant_multiplier->at(0),
                                  (int)quant_rshift->at(0));
    assert(ret == 0);
  } else {
    // float32, bf16
    int ret = my_reduce_mean(input->data(), resultT->data(), input_shape, axes);
    assert(ret == 0);
  }

  valueMapping[result] = std::move(resultT);
  return success();
}


LogicalResult tpu::ReduceMaxOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float>>> &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  std::shared_ptr<std::vector<float> > input = opdT[0];
  auto result = this->getResult();
  std::vector<int64_t> shape = getTensorShape(result);
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);

  // parse param
  std::vector<int32_t> axes;
  arrayAttrToVector(this->axes().getValue(), axes);

  std::vector<int64_t> input_shape = getTensorShape(this->input());

  int ret = my_reduce_max(input->data(), resultT->data(), input_shape, axes);
  assert(ret == 0);

  if (mlir::getOpQuant(op) == "NONE") {
    // do nothing
  } else if (getOpQuant() == "INT8") {
    assert(opdT.size() == 5);
    std::shared_ptr<std::vector<float> > quant_rshift = opdT[3];
    std::shared_ptr<std::vector<float> > quant_multiplier = opdT[4];

    for (int64_t i = 0; i < size; ++i) {
      resultT->at(i) = (float)applyMultiplierAndRShiftAndSaturateInt8(
          resultT->at(i), (uint32_t)quant_rshift->at(0),
          (uint32_t)quant_multiplier->at(0), false);
    }
  } else if (mlir::getOpQuant(op) == "BF16") {
    auto tensor_bf16 = std::make_unique<std::vector<bfloat16> >(resultT->size());
    FloatToBFloat16(resultT->data(), tensor_bf16->data(), resultT->size()); // with rounding
    BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
  } else {
    llvm_unreachable("unsupported type");
  }

  valueMapping[result] = std::move(resultT);
  return success();
}

LogicalResult tpu::YoloDetectionOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float>>> &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto output_shape = getTensorShape(result);
  auto resultT = std::make_unique<std::vector<float>>(size);
  auto output_data = resultT->data();

  auto batch = output_shape[0];

  int net_input_h = this->net_input_h();
  int net_input_w = this->net_input_w();
  float obj_threshold = this->obj_threshold().convertToFloat();
  float nms_threshold = this->nms_threshold().convertToFloat();
  int keep_topk = this->keep_topk();
  bool tiny = this->tiny();
  bool yolo_v4 = this->yolo_v4();
  int class_num = this->class_num();
  std::string str_anchors = this->anchors().str();
  std::vector<float> vec_anchors;
  std::stringstream iss(str_anchors);
  std::string s;
  while (std::getline(iss, s, ',')) {
    vec_anchors.push_back(atof(s.c_str()));
  }

  int input_count = opT.size();

  if (tiny) {
    if (vec_anchors.size() == 0) {
      vec_anchors = {
        10,14,  23,27,  37,58,    // layer23-conv (26*26)
        81,82,  135,169,  344,319 // layer16-conv (13*13)
      };
    }
  } else {
    if (vec_anchors.size() == 0) {
      if (yolo_v4) {
        vec_anchors = {
          142, 110, 192, 243, 459, 401, // layer161-conv
          36, 75, 76, 55, 72, 146,// layer150-conv
          12, 16, 19, 36, 40, 28, // layer139-conv
        };
      }
      else {
        vec_anchors = {
          10,13,   16,30,    33,23,      // layer106-conv (52*52)
          30,61,   62,45,    59,119,     // layer94-conv  (26*26)
          116,90,  156,198,  373,326     // layer82-conv  (13*13)
        };
      }
    }
  }

  float anchors[input_count][6];
  assert((int)vec_anchors.size() == input_count * 6);

  for (int i = 0; i < input_count; ++i) {
    for (int j = 0; j < 6; ++j) {
      anchors[i][j] = vec_anchors[i*6+j];
    }
  }

  for (int b = 0; b < batch; ++b) {
    std::vector<std::vector<int>> grid_size;
    std::vector<std::vector<float>> features;

    for (int i = 0; i < input_count; ++i) {
      auto shape = getTensorShape(op->getOperand(i));
      grid_size.push_back(std::vector<int>{(int)shape[2], (int)shape[3]});
      auto data = opT[i]->data() + b * shape[1] * shape[2] * shape[3];
      auto size = opT[i]->size() / batch;
      std::vector<float> bottom_data(data, data + size);
      features.push_back(bottom_data);
    }

    detection det_raw[MAX_DET_RAW];
    detection dets[MAX_DET];
    int det_raw_idx = 0;
    for (size_t i = 0; i < features.size(); i++) {
      process_feature(det_raw, &det_raw_idx, features[i].data(), grid_size[i],
        &anchors[i][0], {net_input_h, net_input_w}, class_num, obj_threshold);
    }
    nms(det_raw, det_raw_idx, nms_threshold);
    int det_idx = 0;
    for (int i = 0; i < det_raw_idx; i++) {
      if (det_raw[i].score > 0) {
        dets[det_idx] = det_raw[i];
        det_idx ++;
      } else {
        //std::cout << "erased: " << det_raw[i].cls << std::endl;
      }
    }

    if (keep_topk > det_idx)
        keep_topk = det_idx;

    long long count = 0;
    auto batched_output_data = output_data + b * output_shape[1] * output_shape[2] * output_shape[3];
    for(int i = 0; i < (int)keep_topk; ++i) {
      batched_output_data[count++] = dets[i].bbox.x;
      batched_output_data[count++] = dets[i].bbox.y;
      batched_output_data[count++] = dets[i].bbox.w;
      batched_output_data[count++] = dets[i].bbox.h;
      batched_output_data[count++] = dets[i].cls;
      batched_output_data[count++] = dets[i].score;

      LLVM_DEBUG(llvm::errs() << "x= " << dets[i].bbox.x << ",y= " << dets[i].bbox.y
                << ",w= " << dets[i].bbox.w << ",h= " << dets[i].bbox.h
                << ", class= " << dets[i].cls
                << ", score= " << dets[i].score<< "\n";);
    }
  }

  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::ZeroMaskOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float>>> &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  std::shared_ptr<std::vector<float> > input = opdT[0];
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);

  for (uint32_t i = 0; i < size; i++) {
    resultT->at(i) = (input->at(i) == 0.0 ? 1.0 : 0.0);
  }

  valueMapping[result] = std::move(resultT);
  return success();
}

LogicalResult tpu::QuantOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);
  int zero_point = this->zero_point();
  float scale = this->scale().convertToFloat();
  LLVM_DEBUG(llvm::errs() << "quantization/dequantization, scale = "
              << std::to_string(scale) << "\n";);

  if (this->from() == "NONE" && this->to() == "INT8") {
    float *input = (float *)opdT[0]->data();
    float *output = (float *)resultT->data();
    auto prevOp = getOperand().getDefiningOp();
    bool useTpuQuantOp = isa<tpu::InputOp>(prevOp) ? false : clUseTPUQuantOp;
    quantizeActivationFromFp32ToInt8(output, input, size, scale,
                                     useTpuQuantOp, zero_point);
  } else if ((this->from() == "INT8" || this->from() == "UINT8") && this->to() == "NONE") {
    float *input = (float *)opdT[0]->data();
    float *output = (float *)resultT->data();
    dequantizeActivationFromInt8ToFp32(output, input, size, scale,
                                       clUseTPUQuantOp, zero_point);
  } else if (this->from() == "NONE" && this->to() == "BF16") {
    auto tensor_bf16 = std::make_unique<std::vector<bfloat16>>(resultT->size());
    FloatToBFloat16(opdT[0]->data(), tensor_bf16->data(),
                    opdT[0]->size()); // with rounding
    BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
  } else if (this->from() == "BF16" && this->to() == "NONE") {
    resultT->assign(opdT[0]->begin(), opdT[0]->end());
  } else if ((this->from() == "INT8" ||
             this->from() == "UINT8") && this->to() == "BF16") {
    float *input = (float *)opdT[0]->data();
    dequantizeActivationFromInt8ToBf16(resultT->data(), input, size, scale);
  } else if (this->from() == "BF16" && this->to() == "INT8") {
    float *input = (float *)opdT[0]->data();
    float *output = (float *)resultT->data();
    quantizeActivationFromBf16ToInt8(output, input, size, scale);
  } else {
    LLVM_DEBUG(llvm::errs() << "  From:  "
                            << this->from().str() << "  To: " << this->to().str() << "\n";);
    llvm_unreachable("unsupported type");
  }

  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::ReQuantOp::interpret(
    DenseMap<Value, std::shared_ptr<std::vector<float>>> &valueMapping) {
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name()
                          << "]\n";);
  Operation *op = this->getOperation();
  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float>>(size);
  float *input = (float *)opdT[0]->data();

  float input_offset = (float)-getPreviousOpZeroPoint(op);
  float output_offset = (float)getOpZeroPoint(op);
  float qscale = this->qscale().convertToFloat();

  auto tensor_bf16 = std::make_unique<std::vector<bfloat16>>(resultT->size());
  bfloat16 input_offset_bf16;
  bfloat16 output_offset_bf16;
  bfloat16 qscale_bf16;
  FloatToBFloat16(&input_offset, &input_offset_bf16, 1);
  BFloat16ToFloat(&input_offset_bf16, &input_offset, 1);
  FloatToBFloat16(&output_offset, &output_offset_bf16, 1);
  BFloat16ToFloat(&output_offset_bf16, &output_offset, 1);
  FloatToBFloat16(&qscale, &qscale_bf16, 1);
  BFloat16ToFloat(&qscale_bf16, &qscale, 1);
  FloatToBFloat16(input, tensor_bf16->data(),
                  resultT->size()); // with rounding
  BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
  for (size_t i = 0; i < tensor_bf16->size(); i++){
    resultT->at(i) += (float)input_offset ;
  }

  FloatToBFloat16(resultT->data(), tensor_bf16->data(),
                  resultT->size()); // with rounding
  BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
  for (size_t i = 0; i < tensor_bf16->size(); i++){
    resultT->at(i) *= qscale ;
  }
  FloatToBFloat16(resultT->data(), tensor_bf16->data(),
                  resultT->size()); // with rounding
  BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
  for (size_t i = 0; i < tensor_bf16->size(); i++){
    resultT->at(i) += (float)output_offset ;
  }
  FloatToBFloat16(resultT->data(), tensor_bf16->data(),
                  resultT->size()); // with rounding
  BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
  // round
  for (size_t i = 0; i < tensor_bf16->size(); i++){
    float sub_part;
    float int_part;
    sub_part = std::modf(resultT->at(i), &int_part);
    // subpart 0.5
    if(std::abs(std::abs(sub_part) - 0.5f) < 0.001){
      resultT->at(i) = std::round(resultT->at(i) / 2) * 2;
    }
    else if(std::abs(sub_part) > 0.0f){
       resultT->at(i) = std::nearbyint(resultT->at(i));
    }
  }


  valueMapping[result] = std::move(resultT);
  return success();
}



LogicalResult ModuleInterpreter::runOperation(Operation &opInst) {
  if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpInterpInterface>(opInst)) {
    return tpuOp.interpret(valueMapping);
  }

  // Bypass load file and weight since is done in constructor
  if (auto weightFileOp = dyn_cast<tpu::WeightFileOp>(opInst)) {
    return success();
  }
  if (auto loadWeightOp = dyn_cast<tpu::LoadWeightOp>(opInst)) {
    return success();
  }
  if (auto noneOp = dyn_cast<tpu::NoneOp>(opInst)) {
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
      llvm_unreachable("only has tpu func");
    }

    if (failed(runOneFunction(function)))
      return failure();
  }

  return success();
}

LogicalResult ModuleInterpreter::doRun(
    std::vector<int64_t> input_shape, std::vector<float> &input_data,
    std::map<std::string, std::vector<float>> *results,
    std::map<std::string, std::vector<float>> *allTensorMap) {
  // set inputs
  auto inputs = getInputsList();
  if (inputs.size() == 1) {
    std::vector<int64_t> shape = inputs[0].getType().template cast<TensorType>().getShape();
    updateValue(inputs[0], input_data);
  } else {
    // parse mlir and get each input, we concatenate all inputs
    // e.g: input is func @tpu_func(%arg0: tensor<1x30720x1xf32>, %arg1: tensor<1x7680x1xf32>, %arg2: tensor<1x1920x1xf32>, %arg3: tensor<1x480x1xf32>, %arg4: tensor<1x120x1xf32>) -> tensor<1x40920x1xf32>
    // 1x30720x1 + 1x7680x1 + 1x1920x1 + 1x480x1 + 1x120x1

    // check the total size is same with input data size
    int64_t input_size = (int64_t)input_data.size();
    int64_t total_input_size = 0;
    for (auto i : inputs) {
      std::vector<int64_t> shape = i.getType().template cast<TensorType>().getShape();
      total_input_size += std::accumulate(shape.begin(), shape.end(), 1,
          std::multiplies<int64_t>());
    }

    // check input data size with total input size
    if (input_size != total_input_size) {
      std::stringstream err_msg;
      err_msg << "input data size(" << input_size
        << ") not the same with total input size required ("<<total_input_size<<")\n";
      llvm_unreachable(err_msg.str().c_str());
    }

    // seperate input data to each  mlir InputOp Value
    total_input_size = 0;
    for (auto i : inputs) {
      std::vector<int64_t> shape = i.getType().template cast<TensorType>().getShape();
      int64_t shape_sz = std::accumulate(shape.begin(), shape.end(), 1,
          std::multiplies<int64_t>());

      // calculate shift
      std::vector<float> input_n(input_data.begin() + total_input_size,
                                 input_data.begin() + total_input_size +
                                     shape_sz);

      updateValue(i, input_n);

      total_input_size += shape_sz;
    }
  }

  // inference
  if (failed(runFunctions()))
    return failure();

  // get results
  assert(results);
  value_map_t resultsMap = getResults();
  for (auto it = resultsMap.begin(); it != resultsMap.end(); it++) {
    auto op = it->first.getDefiningOp();
    assert(op);
    auto vec = it->second.get();
    assert(vec);
    // deep copy
    (*results)[getOpName(op).str()] = *vec;
  }

  // get all tensor data if needed
  if (allTensorMap) {
    value_map_t valueMap = getValueMap();
    for (auto it = valueMap.begin(); it != valueMap.end(); it++) {
      auto op = it->first.getDefiningOp();
      if (!op) {
        //it->first->dump();
        continue;
      }
      if (auto loadWeightOp = dyn_cast<tpu::LoadWeightOp>(op)) {
        continue;
      }
      auto vec = it->second.get();
      assert(vec);
      // deep copy
      (*allTensorMap)[getOpName(op).str()] = *vec;
    }
  }

  return success();
}

static bool isValidTpuOp(Operation &op) {
  return (!isa<tpu::LoadWeightOp>(op) && !isa<tpu::WeightFileOp>(op) &&
          !isa<tpu::NoneOp>(op) &&
          op.getName().getDialect()->getNamespace() == "tpu");
}

std::string ModuleInterpreter::customOpPluginFile_ = "";

void ModuleInterpreter::getShape(
    std::map<std::string, std::vector<int64_t>> *shapeMap) {
  for (FuncOp function : this->mlirModule.getOps<FuncOp>()) {
    for (Block &bb : function.getBlocks()) {
      for (auto &op : bb) {
        if (!isValidTpuOp(op)) {
          continue;
        }
        // TODO: Only support one output tesor for now.
        auto result = op.getResult(0);
        std::vector<int64_t> shape =
            result.getType().cast<TensorType>().getShape();
        (*shapeMap)[getOpName(&op).str()] = shape;
      }
    }
  }
};

LogicalResult
runTpuModule(ModuleOp m, std::string pluginFile,
             std::vector<int64_t> input_shape, std::vector<float> &input_data,
             std::map<std::string, std::vector<float>> *results,
             std::map<std::string, std::vector<int64_t>> *shapeMap,
             std::map<std::string, std::vector<float>> *allTensorMap) {

  ModuleInterpreter::setCustomOpPluginFile(pluginFile);

  LogicalResult ret = failure();
  std::unique_ptr<ModuleInterpreter> interpreter = std::make_unique<ModuleInterpreter>(m);
  ret = interpreter->doRun(input_shape, input_data, results, allTensorMap);
  interpreter->getShape(shapeMap);

  return ret;
}

} // namespace mlir
