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
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
#include "mlir/Dialect/TPU/Interpreter.h"
#include "mlir/Dialect/TPU/NativeCpuImplementation.h"
#include "mlir/Dialect/TPU/CpuOpParam.h"
#include "mlir/Dialect/TPU/CpuLayer_DetectionOutput.h"
#include "mlir/Dialect/TPU/CpuLayer_RetinaFaceDetection.h"
#include "mlir/Dialect/TPU/CpuLayer_YoloDetection.h"
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
#include "llvm/Support/DynamicLibrary.h"

#include <numeric>
#include <functional>
#include <algorithm>
#include <unordered_map>

namespace mlir {

static llvm::cl::opt<std::string>
    clCustomerInterpretPlugin("customer-interpret-plugin",
        llvm::cl::desc("Specify the interpret library to customer op"));

static llvm::sys::DynamicLibrary gCustomerPlugin;

static std::vector<std::shared_ptr<std::vector<float> > >
    getOperandTensors(Operation *op, const value_map_t &valueMapping) {
  std::vector<std::shared_ptr<std::vector<float> > > opdT;
  for (auto operand : op->getOperands()) {
    if ( isTensorNone(operand) ) {
      opdT.push_back(nullptr);
      continue;
    }
    auto it = valueMapping.find(operand);
    assert(it != valueMapping.end());
    opdT.push_back(it->second);
  }
  return opdT;
}

LogicalResult tpu::BatchNormOp::interpret(
    DenseMap<Value *, std::shared_ptr<std::vector<float> > > &valueMapping) {
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
    DenseMap<Value *, std::shared_ptr<std::vector<float> > > &valueMapping) {
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
  bool do_relu = this->do_relu();
  int axis = this->axis().getLimitedValue();
  assert(axis == 1);

  // get tensors
  assert(opdT.size() == 6);
  std::shared_ptr<std::vector<float> > input = opdT[0];
  std::shared_ptr<std::vector<float> > scale = opdT[1];
  assert(scale->size() == (size_t)c * n);
  std::shared_ptr<std::vector<float> > quant_rshift = opdT[4];
  std::shared_ptr<std::vector<float> > quant_multiplier = opdT[5];

  // MUL apply qscale on output put, no scaling on input

  // compute in fp32
  int ret = my_scale(input->data(), scale->data(), nullptr,
                     resultT->data(), n, c, h, w);
  assert(ret == 0);
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
    assert(false);
  }

  valueMapping[result] = std::move(resultT);
  return success();
}

LogicalResult tpu::ConcatOp::interpret(
    DenseMap<Value *, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);

  // parse param
  auto concat_axis = this->axis();
  LLVM_DEBUG(llvm::errs() << "concat_axis =" << concat_axis << "\n";);

  // get tensors
  size_t nInputs = this->getNumInputs();
  assert(nInputs >= 2 && "bottom num is 0 or 1");
  std::vector<float *> input(nInputs);
  for (size_t i = 0; i < nInputs; ++i) {
    input[i] = (float *)opdT[i]->data();
  }
  //float *output = resultT->data();
  std::shared_ptr<std::vector<float> > quant_rshift = opdT[nInputs + 2];
  std::shared_ptr<std::vector<float> > quant_multiplier = opdT[nInputs + 3];

  // apply qscale on input tensors before f32 compute
  std::vector<std::vector<float> > input_copy(nInputs);
  if (mlir::getOpQuant(op) == "INT8") {
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

  // there is no fp32 compute

  // apply rshift and saturate on input directly
  if (mlir::getOpQuant(op) == "INT8") {
    for (unsigned i = 0; i < nInputs; ++i) {
      for (size_t j = 0; j < opdT[i]->size(); ++j) {
        input[i][j] = (float)applyRShiftAndSaturateInt8(input[i][j],
            (uint32_t)quant_rshift->at(0));
      }
    }
  }

  // do concat copy
  auto tmp_resultT = std::make_unique<std::vector<float> >(0);
  int shift_idx_c=0;
  int shift_idx_h=0;
  int tmp_w=0;
  for (uint32_t i = 0; i < nInputs; i++) {
    std::vector<int64_t> shape;
    int64_t input_size, n, c, h, w;
    getTensorShapeAndSize(op->getOperand(i), shape, input_size);

    if (shape.size() == 4) {
      n = shape[0];
      c = shape[1];
      h = shape[2];
      w = shape[3];

      LLVM_DEBUG(llvm::errs() << "  [" << i << "], shape ("
                              << n << ", " << c << ", " << h << ", " << w << ")\n";);
      LLVM_DEBUG(llvm::errs() << "  [" << i << "], size " << input_size << "\n";);

      auto *input_data = input[i];

      if (concat_axis == 0) {
        auto shapeT = std::make_unique<std::vector<float> >(input_size);
        shapeT.get()->assign(&input_data[0], &input_data[input_size]);
        tmp_resultT.get()->insert(tmp_resultT.get()->end(), shapeT->begin(), shapeT->end());
      } else if (concat_axis == 1) {
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
      } else {
        assert(0 && "not support concat_axis >=3 now\n");
      }
    } else if (shape.size() == 2) {
      h = shape[0];
      w = shape[1];

      LLVM_DEBUG(llvm::errs() << "  [" << i << "], shape ("
                              << h << ", " << w << ")\n";);
      LLVM_DEBUG(llvm::errs() << "  [" << i << "], size " << input_size << "\n";);

      auto *input_data = input[i];

      if (concat_axis == 0) {
        auto shapeT = std::make_unique<std::vector<float> >(input_size);
        shapeT.get()->assign(&input_data[0], &input_data[input_size]);
        tmp_resultT.get()->insert(tmp_resultT.get()->end(), shapeT->begin(), shapeT->end());
      } else if (concat_axis == 1) {
        for (uint32_t idx_h = 0; idx_h < h; idx_h++) {
          auto shapeT = std::make_unique<std::vector<float> >(w);
          //int insert_offset = ((idx_h + 1) * idx_h) * h * w;
          //int insert_offset = (idx_h  + (idx_h + 1) * shift_idx_h) * (i=0?w:tmp_w);
          int insert_offset = ((idx_h+1)* tmp_w) + idx_h*w;
          shapeT.get()->assign(&input_data[idx_h * w], &input_data[(idx_h + 1) * w]);
          tmp_resultT.get()->insert(tmp_resultT.get()->begin() + insert_offset, shapeT->begin(), shapeT->end());
        }
        tmp_w += w;
      } else {
        assert(0 && "not support concat_axis >=2 now\n");
      }
    } else if (shape.size() == 3) {
      c = shape[0];
      h = shape[1];
      w = shape[2];

      LLVM_DEBUG(llvm::errs() << "  [" << i << "], shape ("
                              << c << ", " << h << ", " << w << ")\n";);
      LLVM_DEBUG(llvm::errs() << "  [" << i << "], size " << input_size << "\n";);

      auto *input_data = input[i];

      if (concat_axis == 0) {
        auto shapeT = std::make_unique<std::vector<float> >(input_size);
        shapeT.get()->assign(&input_data[0], &input_data[input_size]);
        tmp_resultT.get()->insert(tmp_resultT.get()->end(), shapeT->begin(), shapeT->end());
      } else if (concat_axis == 2) {
        assert(c==1);
        for (uint32_t idx_h = 0; idx_h < h; idx_h++) {
          auto shapeT = std::make_unique<std::vector<float> >(w);
          int insert_offset = ((idx_h+1)* tmp_w) + idx_h*w;
          shapeT.get()->assign(&input_data[idx_h * w], &input_data[(idx_h + 1) * w]);
          tmp_resultT.get()->insert(tmp_resultT.get()->begin() + insert_offset, shapeT->begin(), shapeT->end());
        }
        tmp_w += w;
      } else {
        assert(0&&"not support shape size =1 and axis = 1 now ");
      }
    }
  }
  resultT.get()->assign(tmp_resultT.get()->begin(), tmp_resultT.get()->end());

  valueMapping[result] = std::move(resultT);
  return success();
}

template <typename OpTy>
LogicalResult doConv2DOpInterpret(Operation *op,
    DenseMap<Value *, std::shared_ptr<std::vector<float> > > &valueMapping) {
  auto castOp = cast<OpTy>(op);
  assert(castOp);
  bool is_deconv = isa<tpu::DeConv2DOp>(op);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = castOp.getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);

  // parse param
  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, ph, pw, dh, dw;
  parseConvParam(castOp.param(), is_deconv,
                 castOp.input(), castOp.output(), castOp.filter(),
                 n, ic, ih, iw, oc, oh, ow, g,
                 kh, kw, sh, sw, ph, pw, dh, dw, is_dw, with_bias, do_relu);

  // get tensors
  assert(opdT.size() == 7);
  std::shared_ptr<std::vector<float> > input = opdT[0];
  std::shared_ptr<std::vector<float> > filter = opdT[1];
  std::shared_ptr<std::vector<float> > bias = opdT[2];
  //std::shared_ptr<std::vector<float> > quant_scale = opdT[3];
  //std::shared_ptr<std::vector<float> > quant_zeropoint = opdT[4];
  std::shared_ptr<std::vector<float> > quant_rshift = opdT[5];
  std::shared_ptr<std::vector<float> > quant_multiplier = opdT[6];

  // compute in fp32
  if (!is_deconv) {
    int ret = mkldnn_conv(input->data(), filter->data(),
        bias?bias->data():nullptr, resultT->data(),
        n, ic, ih, iw, oc, oh, ow, kh, kw, sh, sw, dh,dw, ph, pw, g);
    assert(ret == 0);
  } else {
    int ret = mkldnn_deconv(input->data(), filter->data(),
        bias?bias->data():nullptr, resultT->data(),
        n, ic, ih, iw, oc, oh, ow, kh, kw, sh, sw, ph, pw, g);
    assert(ret == 0);
  }
  if (do_relu) {
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
      quantizeActivationInt8PerChannelMultiplierAndRShift(resultT->data(),
          resultT->data(), n, oc, size / oc / n,
          quant_rshift->data(), quant_multiplier->data());
    } else {
      assert(false);
    }
  } else if (getOpQuant(op) == "BF16") {
    auto tensor_bf16 = std::make_unique<std::vector<bfloat16> >(resultT->size());
    FloatToBFloat16(resultT->data(), tensor_bf16->data(), resultT->size()); // with rounding
    BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
  } else {
    assert(false);
  }

  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::Conv2DOp::interpret(
    DenseMap<Value *, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);
  return doConv2DOpInterpret<tpu::Conv2DOp>(op, valueMapping);
}

LogicalResult tpu::DeConv2DOp::interpret(
    DenseMap<Value *, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);
  return doConv2DOpInterpret<tpu::DeConv2DOp>(op, valueMapping);
}

LogicalResult tpu::CropOp::interpret(
    DenseMap<Value *, std::shared_ptr<std::vector<float>>> &valueMapping) {
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

  my_crop(input, output, input_shape1.data(), input_shape2.data(),
          output_shape.data(), 0, crop_offset.data(), indices.data());
  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::DetectionOutputOp::interpret(
    DenseMap<Value *, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);

  std::vector<int64_t> output_shape;
  output_shape = getTensorShape(this->output());

  assert(output_shape.size() <= 4);

    int num_classes_ = this->num_classes().getLimitedValue();
    bool share_location_ = this->share_location();
    int num_loc_classes_ = share_location_ ? 1 : num_classes_;
    int background_label_id_ = this->background_label_id().getLimitedValue();
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

    int keep_top_k_ = this->keep_top_k().getLimitedValue();
    float confidence_threshold_ = this->confidence_threshold().convertToFloat();

    // Parameters used in nms.
    float nms_threshold_ = this->nms_threshold().convertToFloat();
    float eta_ = 1.0;
    int top_k_ = this->top_k().getLimitedValue();

    auto input_type0 = this->input()[0]->getType().cast<TensorType>();
    std::vector<int64_t> i_s0(input_type0.getShape());
    auto input_type1 = this->input()[1]->getType().cast<TensorType>();
    std::vector<int64_t> i_s1(input_type1.getShape());
    auto input_type2 = this->input()[2]->getType().cast<TensorType>();
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
     DenseMap<Value *, std::shared_ptr<std::vector<float> > > &valueMapping) {
  auto opdT = getOperandTensors(op, valueMapping);
  auto result = op->getResult(0);
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);
  float *input = (float *)opdT[0]->data();
  float *output = (float *)resultT.get()->data();
  std::shared_ptr<std::vector<float> > y0_table_op = opdT[1];

  if (getOpQuant(op) == "INT8") {
    for (int i = 0; i < size; ++i) {
      output[i] = y0_table_op->at((unsigned char)input[i]);
    }
  }else if(getOpQuant(op) == "BF16"|| getOpQuant(op) == "NONE"){
    std::vector<int64_t> shape;
    int64_t input_size, n, c, h, w;
    getTensorShapeAndSize(op->getOperand(0), shape, input_size);
    getNCHW(shape, n, c, h, w);

    if (type == "Reciprocal"){
      float eps = 1.0e-5;
      for (int i = 0; i < input_size; ++i) {
        output[i] = 1.0/(input[i] + eps);
      }
    }else if (type == "Sqrt"){
      for (int i = 0; i < input_size; ++i) {
        output[i] = pow(input[i],0.5);
      }
    }else if (type == "Sigmoid"){
      my_sigmoid(input, output, n, c, h, w);
    }else{
      assert(false&&"not support LUT op type");
    }

    if (getOpQuant(op) == "BF16"){
      auto tensor_bf16 = std::make_unique<std::vector<bfloat16> >(size);
      // with rounding
      FloatToBFloat16(output, tensor_bf16->data(), size);
      BFloat16ToFloat(tensor_bf16->data(), output, size);
    }

  }else{
    assert(0&&"not support method");
  }

  valueMapping[result] = std::move(resultT);

  return success();

}

LogicalResult tpu::ReciprocalOp::interpret(
    DenseMap<Value *, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);
  StringRef type = "Reciprocal";
  return doLUTOpInterpret(op,type,valueMapping);
}

LogicalResult tpu::SqrtOp::interpret(
    DenseMap<Value *, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);
  StringRef type = "Sqrt";
  return doLUTOpInterpret(op,type,valueMapping);
}

LogicalResult tpu::SigmoidOp::interpret(
    DenseMap<Value *, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);
  StringRef type = "Sigmoid";
  return doLUTOpInterpret(op,type,valueMapping);
}

static LogicalResult doEltwiseOpInterpret(Operation *op,
    StringRef &type, bool do_relu,
    DenseMap<Value *, std::shared_ptr<std::vector<float> > > &valueMapping) {
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
  int64_t output_size, oh, ow;
  getTensorShapeAndSize(op->getResult(0), output_shape, output_size);
  oh = output_shape[2];
  ow = output_shape[3];
  auto resultReal = std::make_unique<std::vector<float> >(output_size);

  // get tensors
  const unsigned nInputs = op->getNumOperands() - 4;
  std::vector<float *>input(nInputs);
  for (unsigned i = 0; i < nInputs; ++i) {
    input[i] = opdT[i]->data();
  }
  float *output = resultT->data();
  std::shared_ptr<std::vector<float> > quant_rshift = opdT[nInputs + 2];
  std::shared_ptr<std::vector<float> > quant_multiplier = opdT[nInputs + 3];

  // apply qscale on input tensors before f32 compute
  std::vector<std::vector<float> > input_copy(nInputs);
  if (type == "ADD" || type == "MAX") {
    if (getOpQuant(op) == "INT8") {
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
    assert(false);
  }

  // compute in fp32
  assert(nInputs == 2);
  int ret = 0;
  if (type == "ADD") {
    ret = my_eltwise(input[0], input[1], output, in, ic, ih, iw, 1);
  } else if (type == "MAX") {
    ret = my_eltwise(input[0], input[1], output, in, ic, ih, iw, 2);
  } else if (type == "MUL") {
    ret = my_eltwise(input[0], input[1], output, in, ic, ih, iw, 0);
  } else {
    assert(false);
  }
  assert(ret == 0);
  if (do_relu) {
    ret = my_relu(output, output, in, ic, ih, iw, 0.0f);
    assert(ret == 0);
  }

  // rshift and saturate on output
  if (getOpQuant(op) == "NONE") {
    // do nothing
  } else if (getOpQuant(op) == "INT8") {
    if (type == "ADD" || type == "MAX") {
      // apply rshift and saturate
      for (int i = 0; i < input_size; ++i) {
        output[i] =
            (float)applyRShiftAndSaturateInt8(output[i], (uint32_t)quant_rshift->at(0));
      }
    } else if (type == "MUL") {
      // apply qscale on output (both rshift and saturate)
      for (int i = 0; i < input_size; ++i) {
        output[i] = (float)applyMultiplierAndRShiftAndSaturateInt8(
            output[i], (uint32_t)quant_rshift->at(0),
            (uint32_t)quant_multiplier->at(0), true);
      }
    }
  } else if (getOpQuant(op) == "BF16") {
    auto tensor_bf16 = std::make_unique<std::vector<bfloat16> >(resultT->size());
    FloatToBFloat16(resultT->data(), tensor_bf16->data(), resultT->size()); // with rounding
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
    DenseMap<Value *, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);
  StringRef type = "ADD";
  return doEltwiseOpInterpret(op, type, do_relu(), valueMapping);
}

LogicalResult tpu::EltwiseMaxOp::interpret(
    DenseMap<Value *, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);
  StringRef type = "MAX";
  return doEltwiseOpInterpret(op, type, do_relu(), valueMapping);
}

LogicalResult tpu::EltwiseMulOp::interpret(
    DenseMap<Value *, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);
  StringRef type = "MUL";
  return doEltwiseOpInterpret(op, type, do_relu(), valueMapping);
}

LogicalResult tpu::FullyConnectedOp::interpret(
    DenseMap<Value *, std::shared_ptr<std::vector<float> > > &valueMapping) {
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

  std::shared_ptr<std::vector<float> > input = opdT[0];
  std::shared_ptr<std::vector<float> > filter = opdT[1];
  std::shared_ptr<std::vector<float> > bias = opdT[2];
  std::shared_ptr<std::vector<float> > quant_rshift = opdT[5];

  int ret = mkldnn_ip(input->data(), filter->data(),
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
    for (int i = 0; i < size; ++i) {
      resultT->at(i) = (float)applyRShiftAndSaturateInt8(resultT->at(i),
          (uint32_t)quant_rshift->at(0));
    }
  } else if (getOpQuant() == "BF16") {
    auto tensor_bf16 = std::make_unique<std::vector<bfloat16> >(resultT->size());
    FloatToBFloat16(resultT->data(), tensor_bf16->data(), resultT->size()); // with rounding
    BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
  } else {
    assert(false);
  }

  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::InputOp::interpret(
    DenseMap<Value *, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);

  // use copy for now
  resultT->assign(opdT[0]->begin(), opdT[0]->end());

  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::LeakyReluOp::interpret(
    DenseMap<Value *, std::shared_ptr<std::vector<float> > > &valueMapping) {
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
  std::shared_ptr<std::vector<float> > input = opdT[0];
  //std::shared_ptr<std::vector<float> > quant_pos_scale = opdT[1];
  //std::shared_ptr<std::vector<float> > quant_pos_zeropoint = opdT[2];
  //std::shared_ptr<std::vector<float> > quant_neg_scale = opdT[3];
  //std::shared_ptr<std::vector<float> > quant_neg_zeropoint = opdT[4];
  std::shared_ptr<std::vector<float> > quant_pos_rshift = opdT[5];
  std::shared_ptr<std::vector<float> > quant_pos_multiplier = opdT[6];
  std::shared_ptr<std::vector<float> > quant_neg_rshift = opdT[7];
  std::shared_ptr<std::vector<float> > quant_neg_multiplier = opdT[8];

  // compute in fp32
  // skipped because if quantization is needed, the negative_slop
  // will be computed by quant rescale

  // rshift and saturate on output
  if (getOpQuant() == "NONE") {
    int ret = my_relu(input->data(), resultT->data(), n, c, h, w, negative_slope);
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

    float *data_i = input->data();
    float *data_o = resultT->data();
    for (int i = 0; i < size; ++i) {
      if (data_i[i] > 0){
        if (do_pos_scale) {
          data_o[i] = (float)applyMultiplierAndRShiftAndSaturateInt8(data_i[i],
              (uint32_t)quant_pos_rshift->at(0), quant_pos_multiplier->at(0), false);
        } else {
          data_o[i] = data_i[i];
        }
      } else {
        data_o[i] = (float)applyMultiplierAndRShiftAndSaturateInt8(data_i[i],
            (uint32_t)quant_neg_rshift->at(0), quant_neg_multiplier->at(0), false);
      }
    }
  } else if (getOpQuant() == "BF16") {
    int ret = my_relu(input->data(), resultT->data(), n, c, h, w, negative_slope);
    assert(ret == 0);
    auto tensor_bf16 = std::make_unique<std::vector<bfloat16> >(resultT->size());
    FloatToBFloat16(resultT->data(), tensor_bf16->data(), resultT->size()); // with rounding
    BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
  } else {
    assert(false);
  }

  valueMapping[result] = std::move(resultT);

  return success();
}

template <typename OpTy>
static LogicalResult doPool2DOpInterpret(Operation *op, bool is_average,
    DenseMap<Value *, std::shared_ptr<std::vector<float> > > &valueMapping) {
  auto castOp = dyn_cast<OpTy>(op);
  assert(castOp);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = op->getResult(0);
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);

  // parse param
  bool is_global, do_relu;
  int n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr;
  parsePoolParam(castOp.param(), castOp.input(), castOp.output(),
                 n, c, ih, iw, oh, ow,
                 kh, kw, sh, sw, pt, pb, pl, pr,
                 is_global, do_relu);

  // get tensors
  float *input = opdT[0]->data();
  float *output = resultT->data();
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
  if (getOpQuant(op) == "INT8" && is_average && is_global) {
    // Average pool should sum by self, we use conv to help us by filter all 1
    // if use mkldnn, it will dive kh * kw by float,
    // calculate method different from 1880v2
    // 1880v2 will prod (qscale / kh * kw) together

    // Todo: my case only has global average, if your model has other case,
    //       plz add and test
    ret = my_avg_pooling(input, output, n, c, ih, iw, oh,
                         ow, kh, kw, sh, sw, pt, pb, pl, pr);
  } else {
    ret = mkldnn_pool(input, output, n, c, ih, iw, oh, ow, kh, kw,
                      sh, sw, pt, pb, pl, pr, is_average);
  }
  assert(ret == 0);

  // apply qscale on output for average pooling, max poolings are bypassed
  if (is_average && getOpQuant(op) == "INT8") {
    assert(quant_rshift && quant_multiplier);
    for (int64_t i = 0; i < size; ++i) {
      // multiplier is taking avg_const into account
      // restore sum value first
      float sum;
      if (is_global){
        sum = output[i];
      } else {
        sum = std::round(output[i] * kh * kw);
      }
      output[i] = (float)applyMultiplierAndRShiftAndSaturateInt8(
          sum, (uint32_t)quant_rshift->at(0),
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

LogicalResult tpu::NormalizeOp::interpret(
    DenseMap<Value *, std::shared_ptr<std::vector<float> > > &valueMapping) {
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
    DenseMap<Value *, std::shared_ptr<std::vector<float> > > &valueMapping) {
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

  assert(input_shape.size() == 4);

  int in,ic,ih,iw,on,oc,oh,ow,order0,order1,order2,order3;

  in = input_shape[0];
  ic = input_shape[1];
  ih = input_shape[2];
  iw = input_shape[3];

  on = output_shape[0];
  oc = output_shape[1];
  oh = output_shape[2];
  ow = output_shape[3];

  order0 = this->order0().getLimitedValue();
  order1 = this->order1().getLimitedValue();
  order2 = this->order2().getLimitedValue();
  order3 = this->order3().getLimitedValue();

  int ret = 0;
  std::shared_ptr<std::vector<float>> input = opdT[0];
  //As long as there is one order which is different from the natural order
  // of the data, we need to permute.(from caffe permute layer source code mark)
  if( in==on && ic==oc && ih==oh && iw==ow ){
    valueMapping[result] = std::move(opdT[0]);
  } else {
    std::shared_ptr<std::vector<float>> input = opdT[0];
    ret = my_permute(input->data(),resultT->data(),input_shape.size(),in,ic,ih,iw,
              on,oc,oh,ow,
              order0,order1,order2,order3);
    assert(ret == 0);
    valueMapping[result] = std::move(resultT);
  }
  return success();
}

LogicalResult tpu::PixelShuffleOp::interpret(
    DenseMap<Value *, std::shared_ptr<std::vector<float>>> &valueMapping) {
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

  int upscale_factor = this->upscale_factor().getLimitedValue();

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
  my_pixelshuffle(input->data(), resultT->data(), in, ic, ih, iw, on,
                          oc, oh, ow, upscale_factor);
  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::PoolAvg2DOp::interpret(
    DenseMap<Value *, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);
  return doPool2DOpInterpret<tpu::PoolAvg2DOp>(op, true, valueMapping);
}

LogicalResult tpu::PoolMax2DOp::interpret(
    DenseMap<Value *, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);
  return doPool2DOpInterpret<tpu::PoolMax2DOp>(op, false, valueMapping);
}

LogicalResult tpu::PowerOp::interpret(
    DenseMap<Value *, std::shared_ptr<std::vector<float>>> &valueMapping) {
  //Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name()
                          << "]\n";);
#if 0
  if (auto op = dyn_cast<tpu::PowerOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "PowerOp" << "\n";);
    assert(false);
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

  assert(false);
  return success();
}

LogicalResult tpu::PReluOp::interpret(
    DenseMap<Value *, std::shared_ptr<std::vector<float>>> &valueMapping) {
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
  }
  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::PreprocessOp::interpret(
    DenseMap<Value *, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name()
                          << "]\n";);
  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float>>(size);
  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  assert(input_size == size);
  getNCHW(shape, n, c, h, w);
  // use copy for now
  std::vector<int> orders;
  std::vector<float> means;
  if (this->color_order().hasValue()) {
    for (auto o : llvm::enumerate(this->color_order().getValue())) {
      auto attr = o.value().dyn_cast<IntegerAttr>();
      orders.push_back(attr.getInt());
    }
  }
  if (this->mean().hasValue()) {
    for (auto m : llvm::enumerate(this->mean().getValue())) {
      auto attr = m.value().dyn_cast<FloatAttr>();
      means.push_back((float)attr.getValueAsDouble());
    }
  }

  my_preprocess(opdT[0]->data(), resultT->data(), n, c, h, w,
                orders, means, this->raw_scale().convertToFloat(),
                this->scale().convertToFloat());

  valueMapping[result] = std::move(resultT);
  return success();
}

LogicalResult tpu::PriorBoxOp::interpret(
    DenseMap<Value *, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name()
                          << "]\n";);
  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float>>(size);

  float min_size = this->min_size().convertToFloat();
  float max_size = this->max_size().convertToFloat();
  int aspect_ratios_size = this->aspect_ratios_size().getLimitedValue();
  bool flip = this->flip();
  bool clip = this->clip();
  float variance0 = this->variance0().convertToFloat();
  float variance1 = this->variance1().convertToFloat();
  float variance2 = this->variance2().convertToFloat();
  float variance3 = this->variance3().convertToFloat();
  float offset = this->offset().convertToFloat();
  float step = this->step().convertToFloat();
  std::vector<float> min_sizes_;
  std::vector<float> max_sizes_;
  std::vector<float> aspect_ratios;
  std::vector<float> aspect_ratios_;
  bool flip_;
  int num_priors_;
  bool clip_;
  std::vector<float> variance_;
  int img_w_;
  int img_h_;
  float step_w_;
  float step_h_;

  float offset_;

  aspect_ratios.push_back(this->aspect_ratio0().convertToFloat()) ;
  if(aspect_ratios_size==2)
    aspect_ratios.push_back(this->aspect_ratio1().getValue().convertToFloat()) ;

  int max_size_size=this->max_size_size().getLimitedValue();
  int min_size_size=this->min_size_size().getLimitedValue();


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

  std::vector<int64_t> shape1 = this->getOperand(1)->getType().cast<TensorType>().getShape();
  std::vector<int64_t> shape0 = this->getOperand(0)->getType().cast<TensorType>().getShape();
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


  float *top_data = (float *)resultT->data();

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

  auto output_type = this->output()->getType().cast<TensorType>();
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

LogicalResult tpu::ReluOp::interpret(
    DenseMap<Value *, std::shared_ptr<std::vector<float> > > &valueMapping) {
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

  // compute in fp32
  int ret = my_relu(input->data(), resultT->data(), n, c, h, w, 0.0f);
  assert(ret == 0);

  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::ReshapeOp::interpret(
    DenseMap<Value *, std::shared_ptr<std::vector<float> > > &valueMapping) {
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
    DenseMap<Value *, std::shared_ptr<std::vector<float>>> &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float>>(size);
  auto output_data = resultT->data();

  auto confidence_threshold = this->confidence_threshold().convertToFloat();
  auto nms_threshold = this->nms_threshold().convertToFloat();
  auto keep_topk = this->keep_topk().getLimitedValue();

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
  for (int i = 0; i < feature_stride_fpn.size(); ++i) {
    std::string key = "stride" + std::to_string(feature_stride_fpn[i]);
    um_anchors_fpn[key] = anchors[i];
    um_num_anchors[key] = anchors[i].size();
  }

  int input_count = opT.size();
  assert(input_count == 9);

  std::vector<FaceInfo> infos;
  for (size_t i = 0; i < feature_stride_fpn.size(); ++i) {
    int stride = feature_stride_fpn[i];

    auto landmark_data = opT[input_count-3*i-1]->data();
    size_t landmark_count = opT[input_count-3*i-1]->size();

    auto bbox_data = opT[input_count-3*i-2]->data();
    size_t bbox_count = opT[input_count-3*i-2]->size();

    auto score_data = opT[input_count-3*i-3]->data();
    size_t score_count = opT[input_count-3*i-3]->size();

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

    for(size_t num = 0; num < num_anchors; ++num) {
      for(size_t j = 0; j < count; ++j) {
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
        for(size_t k = 0; k < 5; ++k) {
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
  for(int i = 0; i < keep_topk; ++i) {
    output_data[count++] = preds[i].x1;
    output_data[count++] = preds[i].y1;
    output_data[count++] = preds[i].x2;
    output_data[count++] = preds[i].y2;
    output_data[count++] = preds[i].score;
    for(int j = 0; j < 5; ++j) {
      output_data[count++] = preds[i].x[j];
      output_data[count++] = preds[i].y[j];
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

  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::ScaleOp::interpret(
    DenseMap<Value *, std::shared_ptr<std::vector<float> > > &valueMapping) {
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

LogicalResult tpu::ShuffleChannelOp::interpret(
    DenseMap<Value *, std::shared_ptr<std::vector<float>>> &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float>>(size);
  std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
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
  uint32_t group = this->group().getLimitedValue();
  int ret = my_shuffle_channel(input, output, group, n, c, frame_size);

  assert(ret == 0);

  valueMapping[result] = std::move(resultT);

  return success();
}

#if 0
LogicalResult tpu::SigmoidOp::interpret(
    DenseMap<Value *, std::shared_ptr<std::vector<float>>> &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name()
                          << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float>>(size);

  // parse param

  std::vector<int64_t> shape = getTensorShape(op->getOperand(0));
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  assert(input_size == size);
  getNCHW(shape, n, c, h, w);

  std::vector<int> indices(size, 0);
  float *input = (float *)opdT[0]->data();
  float *output = (float *)resultT.get()->data();
  if (getOpQuant() == "INT8") {
    std::vector<int> data(256, 0);
    float threshold_x = getPreviousOpThreshold(op);
    float threshold_y = getOpThreshold(op);

    assert(threshold_x != 0.0);
    for (int idx = 0; idx < 256; ++idx) {
      char lutInput = static_cast<char>(idx);
      float index = -lutInput * threshold_x / 127.0;
      float lutOutput = 1.0 / (1 + std::exp(index)) * 127.0 / threshold_y;
      int lutOutputI32 = std::floor(lutOutput + 0.5);
      lutOutputI32 = (lutOutputI32 > 127)
                         ? 127
                         : (lutOutputI32 < -128) ? -128 : lutOutputI32;
      data[idx] = lutOutputI32;
    }
    for (int i = 0; i < size; ++i) {
      output[i] = data[(unsigned char)input[i]];
    }
  } else {
    my_sigmoid(input, output, n, c, h, w);
  }
  valueMapping[result] = std::move(resultT);

  return success();
}
#endif

LogicalResult tpu::SliceOp::interpret(
    DenseMap<Value *, std::shared_ptr<std::vector<float> > > &valueMapping) {
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
  int axis = this->axis().getLimitedValue();
  int offset = this->offset().getLimitedValue();

  int ret = my_slice(opdT[0]->data(), resultT->data(), axis, offset,
                     input_shape, output_shape);
  assert(ret == 0);
  valueMapping[result] = std::move(resultT);

  return success();
}

static bool getBoolAttributeValue(const Attribute& attr) {
  return (bool)attr.cast<BoolAttr>().getValue();
}

static int32_t getIntegerAttributeValue(const Attribute& attr) {
  return (int32_t)attr.cast<IntegerAttr>().getValue().getSExtValue();
}

static float getFloatAttributeValue(const Attribute& attr) {
  return (float)attr.cast<FloatAttr>().getValue().convertToFloat();
}

static std::string getStringAttributeValue(const Attribute& attr) {
  return attr.cast<StringAttr>().getValue().str();
}

static uint32_t getAttributeArrayKind(const ArrayAttr& array) {
  auto& attr = *(array.begin());
  return attr.getKind();
}

static void convertParamAttributesToOpParam(
    const DictionaryAttr &attrs,
    cvi::OpParam &param) {
  for (auto& a : attrs) {
    auto name = a.first.str();
    auto &attr = a.second;
    switch (attr.getKind()) {
      case StandardAttributes::Bool:
        param.put<bool>(name, getBoolAttributeValue(attr));
        break;
      case StandardAttributes::Integer:
        param.put<int32_t>(name, getIntegerAttributeValue(attr));
        break;
      case StandardAttributes::Float:
        param.put<float>(name, getFloatAttributeValue(attr));
        break;
      case StandardAttributes::String:
        param.put<std::string>(name, getStringAttributeValue(attr));
        break;
      case StandardAttributes::Array: {
        auto array = attr.cast<ArrayAttr>();
        switch (getAttributeArrayKind(array)) {
          case StandardAttributes::Bool: {
            std::vector<bool> vec;
            for (auto& item : array) {
              vec.push_back(getBoolAttributeValue(item));
            }
            param.put<std::vector<bool>>(name, vec);
            break;
          }
          case StandardAttributes::Integer: {
            std::vector<int32_t> vec;
            for (auto& item : array) {
              vec.push_back(getIntegerAttributeValue(item));
            }
            param.put<std::vector<int32_t>>(name, vec);
            break;
          }
          case StandardAttributes::Float: {
            std::vector<float> vec;
            for (auto& item : array) {
              vec.push_back(getFloatAttributeValue(item));
            }
            param.put<std::vector<float>>(name, vec);
            break;
          }
          case StandardAttributes::String: {
            std::vector<std::string> vec;
            for (auto& item : array) {
              vec.push_back(getStringAttributeValue(item));
            }
            param.put<std::vector<std::string>>(name, vec);
            break;
          }
          default:
            assert(false);
        }
        break;
      }
      default:
        assert(false);
    }
  }
}

using CustomerInterpretFn =
    void (*)(std::vector<std::shared_ptr<std::vector<float>>>&,
             std::vector<std::vector<int64_t>>&,
             std::shared_ptr<std::vector<float>>&,
             std::vector<int64_t>&,
             cvi::OpParam&);

LogicalResult tpu::GenericCpuOp::interpret(
    DenseMap<Value*, std::shared_ptr<std::vector<float>>> &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);
  auto result = this->getResult();
  auto operandTensors = getOperandTensors(op, valueMapping);
  std::vector<std::vector<int64_t>> operandShapes;
  for (auto operand : op->getOperands()) {
    operandShapes.push_back(getTensorShape(operand));
  }
  auto resultTensor = std::make_shared<std::vector<float>>(getTensorSize(result));
  auto resultShape = getTensorShape(result);

  assert(gCustomerPlugin.isValid());
  auto fn = this->operation_name().str() + "_interpret";
  std::transform(fn.begin(), fn.end(), fn.begin(), [](char c) {
    return std::tolower(c);
  });

  auto func = reinterpret_cast<CustomerInterpretFn>(
                gCustomerPlugin.getAddressOfSymbol(fn.c_str()));
  llvm::errs() << "[Interpret] to find " << fn << "\n";
  assert(func);
  cvi::OpParam param;
  // parse param
  convertParamAttributesToOpParam(this->param(), param);
  func(operandTensors, operandShapes, resultTensor, resultShape, param);
  valueMapping[result] = resultTensor;
  return success();
}

LogicalResult tpu::SoftmaxOp::interpret(
    DenseMap<Value *, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  std::vector<int64_t> shape = getTensorShape(result);
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);

  // parse param
  int axis = this->axis().getLimitedValue();

  if (shape.size() == 2) {
    int ret = my_softmax2D(opdT[0]->data(), resultT->data(), shape[0], shape[1]);
    assert(ret == 0);
  } else if (shape.size() == 4) {
    int ret = my_softmax4D(opdT[0]->data(), resultT->data(), axis, shape);
    assert(ret == 0);
  } else if (shape.size() == 3) {
    int ret = my_softmax3D(opdT[0]->data(), resultT->data(), axis, shape);
    assert(ret == 0);
  }

  valueMapping[result] = std::move(resultT);
  return success();
}

LogicalResult tpu::SwapChannelOp::interpret(
    DenseMap<Value *, std::shared_ptr<std::vector<float>>> &valueMapping) {

  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name()
                          << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float>>(size);
  std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
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

LogicalResult tpu::TanHOp::interpret(
    DenseMap<Value *, std::shared_ptr<std::vector<float> > > &valueMapping) {
  //Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

#if 0
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
#endif

  assert(false);
  return success();
}

LogicalResult tpu::TransposeOp::interpret(
    DenseMap<Value *, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name()
                          << "]\n";);
  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float>>(size);
  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  assert(input_size == size);
  getNCHW(shape, n, h, w, c);
  my_transpose(opdT[0]->data(), resultT->data(), n, c, h, w);

  valueMapping[result] = std::move(resultT);
  return success();
}

LogicalResult tpu::UpsampleOp::interpret(
    DenseMap<Value *, std::shared_ptr<std::vector<float> > > &valueMapping) {
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
  int64_t scale = this->scale().getLimitedValue();
  assert(oh == ih * scale);
  assert(ow == iw * scale);

  // get tensors
  assert(opdT.size() == 1);
  std::shared_ptr<std::vector<float> > input = opdT[0];

  // compute in fp32
  int ret = my_upsample(input->data(), resultT->data(), n, c, ih, iw, scale);
  assert(ret == 0);

  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::YoloDetectionOp::interpret(
    DenseMap<Value *, std::shared_ptr<std::vector<float>>> &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float>>(size);
  auto output_data = resultT->data();

  auto net_input_h = this->net_input_h().getLimitedValue();
  auto net_input_w = this->net_input_w().getLimitedValue();
  auto obj_threshold = this->obj_threshold().convertToFloat();
  auto nms_threshold = this->nms_threshold().convertToFloat();
  auto keep_topk = this->keep_topk().getLimitedValue();

  int input_count = opT.size();
  assert(input_count == 3);

  const float anchors[3][6] = {
    {10,13,   16,30,    33,23},      // layer106-conv (52*52)
    {30,61,   62,45,    59,119},     // layer94-conv  (26*26)
    {116,90,  156,198,  373,326}     // layer82-conv  (13*13)
  };

  std::vector<std::vector<int>> grid_size(3);
  std::vector<std::vector<float>> features;

  for (int i = 0; i < 3; ++i) {
    auto shape = getTensorShape(op->getOperand(i));
    grid_size[i] = std::vector<int>{shape[2], shape[3]};
    auto data = opT[i]->data();
    auto size = opT[i]->size();
    std::vector<float> bottom_data(data, data + size);
    features.push_back(bottom_data);
  }

  detection det_raw[MAX_DET_RAW];
  detection dets[MAX_DET];
  int det_raw_idx = 0;
  for (int i = 0; i < 3; i++) {
    process_feature(det_raw, &det_raw_idx, features[i].data(), grid_size[i],
      &anchors[i][0], {net_input_h, net_input_w}, 80, obj_threshold);
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
  for(int i = 0; i < keep_topk; ++i) {
    output_data[count++] = dets[i].bbox.x;
    output_data[count++] = dets[i].bbox.y;
    output_data[count++] = dets[i].bbox.w;
    output_data[count++] = dets[i].bbox.h;
    output_data[count++] = dets[i].cls;
    output_data[count++] = dets[i].score;

    LLVM_DEBUG(llvm::errs() << "x= " << dets[i].bbox.x << ",y= " << dets[i].bbox.y
              << ",w= " << dets[i].bbox.w << ",h= " << dets[i].bbox.h
              << ", class= " << dets[i].cls
              << ", score= " << dets[i].score<< "\n";);
  }

  valueMapping[result] = std::move(resultT);

  return success();
}

LogicalResult tpu::QuantOp::interpret(
    DenseMap<Value *, std::shared_ptr<std::vector<float> > > &valueMapping) {
  Operation *op = this->getOperation();
  LLVM_DEBUG(llvm::errs() << getOperationName() << " [" << this->name() << "]\n";);

  auto opdT = getOperandTensors(op, valueMapping);
  auto result = this->getResult();
  auto size = getTensorSize(result);
  auto resultT = std::make_unique<std::vector<float> >(size);

  if (this->from() == "NONE" && this->to() == "INT8") {
    float *input = (float *)opdT[0]->data();
    float *output = (float *)resultT->data();
    float threshold = this->threshold().getValue().convertToFloat();
    LLVM_DEBUG(llvm::errs() << "  quantization, threshold = "
               << std::to_string(threshold) << "\n";);
    quantizeActivationInt8WithThreshold(output, input, size, threshold);
  } else if (this->from() == "INT8" && this->to() == "NONE") {
    float *input = (float *)opdT[0]->data();
    float *output = (float *)resultT->data();
    float threshold = this->threshold().getValue().convertToFloat();
    LLVM_DEBUG(llvm::errs() << "  quantization, threshold = "
               << std::to_string(threshold) << "\n";);
    dequantizeActivationInt8WithThreshold(output, input, size, threshold);
  } else if (this->from() == "NONE" && this->to() == "BF16") {
    auto tensor_bf16 = std::make_unique<std::vector<bfloat16>>(resultT->size());
    FloatToBFloat16(opdT[0]->data(), tensor_bf16->data(),
                    opdT[0]->size()); // with rounding
    BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
  } else if (this->from() == "BF16" && this->to() == "NONE") {
    resultT->assign(opdT[0]->begin(), opdT[0]->end());
  } else if (this->from() == "INT8" && this->to() == "BF16") {
    float *input = (float *)opdT[0]->data();
    float *output = (float *)resultT->data();
    float threshold = this->threshold().getValue().convertToFloat();
    LLVM_DEBUG(llvm::errs() << "  quantization, threshold = "
               << std::to_string(threshold) << "\n";);
    dequantizeActivationInt8WithThreshold(output, input, size, threshold);
    auto tensor_bf16 = std::make_unique<std::vector<bfloat16>>(resultT->size());
    FloatToBFloat16(resultT->data(), tensor_bf16->data(),
                    resultT->size()); // with rounding
    BFloat16ToFloat(tensor_bf16->data(), resultT->data(), resultT->size());
  } else if (this->from() == "BF16" && this->to() == "INT8") {
    float *input = (float *)opdT[0]->data();
    float *output = (float *)resultT->data();
    float threshold = this->threshold().getValue().convertToFloat();
    LLVM_DEBUG(llvm::errs() << "  quantization, threshold = "
               << std::to_string(threshold) << "\n";);
    quantizeActivationInt8WithThreshold(output, input, size, threshold);
  } else {
    assert(0);
  }

  valueMapping[result] = std::move(resultT);

  return success();
}

std::vector<std::shared_ptr<std::vector<float> > >
    ModuleInterpreter::getOperandTensors(Operation &opInst,
    value_map_t &valueMapping) {
  std::vector<std::shared_ptr<std::vector<float> > > opdT;
  for (auto operand : opInst.getOperands()) {
    if ( !operand->getType().dyn_cast_or_null<RankedTensorType>() ) {
      // this is NoneType
      // isa<tpu::NoneOp>(operand->getDefiningOp());
      opdT.push_back(nullptr);
      continue;
    }
    auto it = valueMapping.find(operand);
    assert(it != valueMapping.end());
    opdT.push_back(it->second);
  }
  return opdT;
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
      assert(0);
    }
    if (failed(runOneFunction(function)))
      return failure();
  }

  return success();
}

LogicalResult ModuleInterpreter::doRun(std::vector<int64_t> input_shape, std::vector<float> &input_vec,
                                       std::map<std::string, std::vector<float> > *results,
                                       std::map<std::string, std::vector<float> > *allTensorMap) {
  // load customer interpret plugin firstly if has.
  if (!clCustomerInterpretPlugin.empty() && !gCustomerPlugin.isValid()) {
    std::string ErrorMsg;
    gCustomerPlugin = llvm::sys::DynamicLibrary::getPermanentLibrary(clCustomerInterpretPlugin.c_str(), &ErrorMsg);
    if (!gCustomerPlugin.isValid()) {
      llvm::errs() << "Failed to load customer interpret plugin from path:"
                   << clCustomerInterpretPlugin << ", reason:" << ErrorMsg << "\n";
      assert(0);
    }
  }

  // set inputs
  auto inputs = getInputsList();
  assert(inputs.size() == 1);
  std::vector<int64_t> shape = inputs[0]->getType().template cast<TensorType>().getShape();

  assert(input_shape == shape);
  assert((int64_t)input_vec.size() == std::accumulate(shape.begin(), shape.end(), 1,
                                                        std::multiplies<int64_t>()));
  updateValue(inputs[0], input_vec);

  // inference
  if (failed(runFunctions()))
    return failure();

  // get results
  assert(results);
  value_map_t resultsMap = getResults();
  for (auto it = resultsMap.begin(); it != resultsMap.end(); it++) {
    auto op = it->first->getDefiningOp();
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
      auto op = it->first->getDefiningOp();
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
          op.getName().getDialect().str() == "tpu");
}

LogicalResult runTpuModule(ModuleOp m,
    std::vector<int64_t> input_shape, std::vector<float> &input_vec,
    std::map<std::string, std::vector<float> > *results,
    std::map<std::string, std::vector<int64_t> > *shapeMap,
    std::map<std::string, std::vector<float> > *allTensorMap) {
  return runTpuModule(m, nullptr, input_shape, input_vec, results, shapeMap, allTensorMap);
}

LogicalResult runTpuModule(ModuleOp m, ModuleInterpreter *interpreter,
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

  LogicalResult ret = failure();
  if (interpreter != nullptr) {
    ret = ModuleInterpreter::runModule<>(interpreter, input_shape, input_vec, results, allTensorMap);
  } else {
    ret = ModuleInterpreter::runModule<>(m, input_shape, input_vec, results, allTensorMap);
  }

  return ret;
}

} // namespace mlir
