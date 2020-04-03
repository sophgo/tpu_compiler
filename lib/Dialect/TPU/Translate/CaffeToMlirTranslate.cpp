//===- CaffeToMlirTranslate.cpp - Caffe to MLIR module conversion ----===//
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
// This file implements a translation from caffe module to MLIR TPU ModuleOp.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/TensorFile.h"
#include "mlir/Translation.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"

#include "caffe/caffe.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/util/signal_handler.h"

#include <iostream>
#include <cstring>
#include <numeric>
#include <map>
#include <string>
#include <vector>

#define DEBUG_TYPE "caffe-to-mlir"

using namespace mlir;

static llvm::cl::OptionCategory clOptionsCategory("caffe translate options");

static llvm::cl::opt<std::string>
    clCaffeModelFilename("caffemodel", llvm::cl::desc("Specify the caffemodel filename"),
                         llvm::cl::cat(clOptionsCategory));

/// set static batch size
// TODO: enable by default for now, should set default 0 when shapeinference is
// ready
static llvm::cl::opt<int> clStaticBatchsize(
    "static-batchsize",
    llvm::cl::desc("set static batchsize, dynamic batchsize is used when not set"),
    llvm::cl::cat(clOptionsCategory), llvm::cl::init(1));

//=========================do preprocess=======================================
static llvm::cl::OptionCategory clOptionsPreprocess("input  data preprocess options");

//  preprocess way 1: add preprocess op
static llvm::cl::opt<bool>
    clAddPreprocess("add-preprocess",
                    llvm::cl::desc("add preprocessing to input feature map"),
                    llvm::cl::init(false));

static llvm::cl::opt<bool>
    clPreprocessTranspose("preprocess-transpose",
                          llvm::cl::desc("packed format to planar"),
                          llvm::cl::cat(clOptionsPreprocess), llvm::cl::init(false));

static llvm::cl::opt<std::string>
    clPreprocessMeanFilename("preprocess-mean-file",
                             llvm::cl::desc("Specify the mean filename"),
                             llvm::cl::cat(clOptionsPreprocess));

static llvm::cl::opt<std::string> clPreprocessMean("preprocess-mean",
                                                   llvm::cl::desc("Specify the mean"),
                                                   llvm::cl::cat(clOptionsPreprocess));

static llvm::cl::opt<float> clPreprocessScale("preprocess-scale",
                                              llvm::cl::desc("Specify the scale"),
                                              llvm::cl::cat(clOptionsPreprocess),
                                              llvm::cl::init(1.0f));

static llvm::cl::opt<float> clPreprocessRawScale("preprocess-raw-scale",
                                                 llvm::cl::desc("Specify the raw scale"),
                                                 llvm::cl::cat(clOptionsPreprocess),
                                                 llvm::cl::init(255.0f));

static llvm::cl::opt<std::string>
    clPreprocessColorOrder("preprocess-color-order",
                           llvm::cl::desc("Specify the order of feature map"),
                           llvm::cl::cat(clOptionsPreprocess));

// preprocess way 2: use sub ops
static llvm::cl::opt<bool>
    clResolvePreprocess("resovle-preprocess",
                        llvm::cl::desc("Resovle Preprocess Op into sub Ops"),
                        llvm::cl::cat(clOptionsPreprocess), llvm::cl::init(false));

static llvm::cl::opt<std::string>
    clSwapChannel("swap_channel", llvm::cl::desc("Specify the channel order to swap"),
                  llvm::cl::cat(clOptionsPreprocess));

static llvm::cl::opt<std::string> clMean("mean", llvm::cl::desc("Specify the mean"),
                                         llvm::cl::cat(clOptionsPreprocess));

static llvm::cl::opt<float> clScale("scale", llvm::cl::desc("Specify the scale"),
                                    llvm::cl::cat(clOptionsPreprocess),
                                    llvm::cl::init(1.0f));

static llvm::cl::opt<float> clRawScale("raw_scale",
                                       llvm::cl::desc("Specify the raw scale"),
                                       llvm::cl::cat(clOptionsPreprocess),
                                       llvm::cl::init(1.0f));

// Importer that takes an Caffe model and imports it as an MLIR module in the
// TPU dialect.
class CaffeImporter {
public:
  explicit CaffeImporter(mlir::ModuleOp module)
      : module_(module), builder_(module.getContext()) {}

  // Import the Caffe model file into the MLIR Module.
  LogicalResult Import(const llvm::StringRef inputFilename,
                       llvm::StringRef caffemodelFilename);

private:
  mlir::Type GetTypeFromCaffeShape(const std::vector<int> shape, mlir::Type elementType,
                                   bool transpose = false);

  void ParseNetInputOutput(caffe::Net<float> &net,
                           std::map<std::string, mlir::Type> &inputs,
                           std::map<std::string, mlir::Type> &outputs);

  mlir::Block *CreateOneBlockFunction(std::map<std::string, mlir::Type> &inputs,
                                      std::map<std::string, mlir::Type> &outputs);

  void AddWeightFileOp(mlir::Block *block, const llvm::StringRef &weightFilename);
  TensorFile *GetWeightFile();
  mlir::Value *AddLoadWeightOp(mlir::Block *block, std::string name, TensorType &type);

  void AddReturnOp(mlir::Block *block, std::map<std::string, mlir::Type> &outputs);

  mlir::Value *GetLayerInput(caffe::Layer<float> *layer);
  std::vector<mlir::Value *> GetLayerInputs(caffe::Layer<float> *layer);

  void ConvertLayers(mlir::Block *block, caffe::Net<float> &net);

  void convertBatchNormLayer(mlir::Block *block, caffe::Layer<float> *layer);
  void convertConcatLayer(mlir::Block *block, caffe::Layer<float> *layer);
  void convertConvolutionLayer(mlir::Block *block, caffe::Layer<float> *layer,
                               bool is_deconv = false);
  void convertCropLayer(mlir::Block *block, caffe::Layer<float> *layer);
  void convertDetectionOutputLayer(mlir::Block *block, caffe::Layer<float> *layer);
  void convertDropoutLayer(mlir::Block *block, caffe::Layer<float> *layer);
  void convertDummyDataLayer(mlir::Block *block, caffe::Layer<float> *layer);
  void convertEltwiseLayer(mlir::Block *block, caffe::Layer<float> *layer);
  void convertFlattenLayer(mlir::Block *block, caffe::Layer<float> *layer);
  void convertInnerProductLayer(mlir::Block *block, caffe::Layer<float> *layer);
  void convertInputLayer(mlir::Block *block, caffe::Layer<float> *layer);
  void convertNormalizeLayer(mlir::Block *block, caffe::Layer<float> *layer);
  void convertPermuteLayer(mlir::Block *block, caffe::Layer<float> *layer);
  void convertPoolingLayer(mlir::Block *block, caffe::Layer<float> *layer);
  void convertPowerLayer(mlir::Block *block, caffe::Layer<float> *layer);
  void convertPReLULayer(mlir::Block *block, caffe::Layer<float> *layer);
  void convertPriorBoxLayer(mlir::Block *block, caffe::Layer<float> *layer);
  // void convertReductionLayer(mlir::Block *block, caffe::Layer<float> *layer);
  void convertReLULayer(mlir::Block *block, caffe::Layer<float> *layer);
  void convertReshapeLayer(mlir::Block *block, caffe::Layer<float> *layer);
  void convertRetinaFaceDetectionLayer(mlir::Block *block, caffe::Layer<float> *layer);
  void convertScaleLayer(mlir::Block *block, caffe::Layer<float> *layer);
  void convertShuffleChannelLayer(mlir::Block *block, caffe::Layer<float> *layer);
  void convertSigmoidLayer(mlir::Block *block, caffe::Layer<float> *layer);
  void convertSliceLayer(mlir::Block *block, caffe::Layer<float> *layer);
  void convertSoftmaxLayer(mlir::Block *block, caffe::Layer<float> *layer);
  void convertSplitLayer(mlir::Block *block, caffe::Layer<float> *layer);
  void convertTanHLayer(mlir::Block *block, caffe::Layer<float> *layer);
  void convertUpsampleLayer(mlir::Block *block, caffe::Layer<float> *layer);
  void convertYoloDetectionLayer(mlir::Block *block, caffe::Layer<float> *layer);
  mlir::Value *insertTransposeLayer(mlir::Block *block, mlir::Value *opd,
                                    const std::string &name);
  void insertPreprocessLayer(mlir::Block *block, mlir::Value *opd,
                             const std::string &name);

  void doPreprocess(mlir::Block *block, mlir::Value *input);
  void doPreSwapChannel(mlir::Block *block, std::string name, const std::vector<int> &order);
  void doPreScale(mlir::Block *block, std::string name, float scale);
  void doPreMean(mlir::Block *block, std::string name, const std::vector<int> &order);

  mlir::ModuleOp module_;
  mlir::Builder builder_;
  mlir::Type elementType_;
  mlir::Value *lastResult_;
  std::map<std::string, mlir::Value *> tensor_map_;
  mlir::Value *weightFileVar_;
};

static void printCaffeLayerParam(const caffe::Layer<float> *layer) {
  auto layer_param = layer->layer_param();
  llvm::errs() << llvm::format(">> [%-12s] %s\n", layer->type(),
                               layer_param.name().c_str());
  for (int i = 0; i <= layer_param.bottom_size() - 1; ++i) {
    llvm::errs() << "btm: " << layer_param.bottom(i) << "\n";
  }
  for (int i = 0; i <= layer_param.top_size() - 1; ++i) {
    llvm::errs() << "top: " << layer_param.top(i) << "\n";
  }
}

static void printCaffeNetAllLayer(const caffe::Net<float> &net) {
  for (size_t i = 0; i <= net.layers().size() - 1; ++i) {
    auto layer = net.layers()[i].get();
    printCaffeLayerParam(layer);
  }
}

#define calcConv2DSpatialOutput(_i_, _k_, _s_, _p_, _d_)                                 \
  (((_i_) + 2 * (_p_) - (_d_) * ((_k_)-1) - 1) / (_s_) + 1)

#define calcDeConv2DSpatialOutput(_i_, _k_, _s_, _p_, _d_)                               \
  ((_s_) * (((_i_)) - 1) + (_d_) * ((_k_)-1) - 2 * (_p_) + 1)

mlir::Type CaffeImporter::GetTypeFromCaffeShape(const std::vector<int> shape,
                                                mlir::Type elementType, bool transpose) {
  std::vector<int64_t> shape_int64(shape.begin(), shape.end());
  if (transpose) {
    assert(shape.size() == 4);
    shape_int64[1] = shape[2];
    shape_int64[2] = shape[3];
    shape_int64[3] = shape[1];
  }
  return RankedTensorType::get(llvm::ArrayRef<int64_t>(shape_int64), elementType);
}

void CaffeImporter::ParseNetInputOutput(caffe::Net<float> &net,
                                        std::map<std::string, mlir::Type> &inputs,
                                        std::map<std::string, mlir::Type> &outputs) {
  for (int i = 0; i <= net.num_inputs() - 1; ++i) {
    int index = net.input_blob_indices()[i];
    LLVM_DEBUG(llvm::errs() << "net input [" << i << "] - [" << index << "] : "
                            << ", blob: " << net.blob_names()[index]
                            << ", shape: " << net.input_blobs()[i]->shape_string()
                            << ", layer: " << net.layer_names()[index] << "\n";);
    std::vector<int> input_shape = net.input_blobs()[i]->shape();
    if (clStaticBatchsize > 0) {
      input_shape[0] = clStaticBatchsize;
    } else {
      // set batch to dynamic
      input_shape[0] = -1;
    }

    bool transpose = false;
    if (i == 0 && clAddPreprocess && clPreprocessTranspose) {
      transpose = true;
    }
    inputs[net.blob_names()[index]] =
        GetTypeFromCaffeShape(input_shape, elementType_, transpose);
  }

  for (int i = 0; i <= net.num_outputs() - 1; ++i) {
    int index = net.output_blob_indices()[i];

    LLVM_DEBUG(llvm::errs() << "net output[" << i << "] - [" << index << "] : "
                            << ", blob: " << net.blob_names()[index]
                            << ", shape: " << net.output_blobs()[i]->shape_string()
                            << ", layer: " << net.layer_names()[index] << "\n";);
    std::vector<int> output_shape = net.output_blobs()[i]->shape();
    if (clStaticBatchsize > 0) {
      output_shape[0] = clStaticBatchsize;
    } else {
      // set batch to dynamic
      output_shape[0] = -1;
    }

    ///
    /// fixup `DetectionOutput` output shape
    /// not use dynamic shape here for bbox num
    /// determine the shape by parsing the `keep_top_k` field of the layer
    ///
    auto layer = net.layer_by_name(net.blob_names()[index]);
    if (layer.get() != nullptr) {
      if (strcmp(layer->type(), "DetectionOutput") == 0) {
        auto layer_param = layer->layer_param();
        auto detection_output_param = layer_param.detection_output_param();
        output_shape[2] = detection_output_param.keep_top_k();
      }
    }
    outputs[net.blob_names()[index]] = GetTypeFromCaffeShape(output_shape, elementType_);
  }
}

mlir::Block *
CaffeImporter::CreateOneBlockFunction(std::map<std::string, mlir::Type> &inputs,
                                      std::map<std::string, mlir::Type> &outputs) {
  std::vector<mlir::Type> arguments;
  for (auto e : inputs)
    arguments.push_back(e.second);
  std::vector<mlir::Type> returns;
  for (auto e : outputs)
    returns.push_back(e.second);
  auto fnType = builder_.getFunctionType(llvm::ArrayRef<mlir::Type>{arguments},
                                         llvm::ArrayRef<mlir::Type>{returns});
  auto fn = FuncOp::create(builder_.getUnknownLoc(), "tpu_func", fnType);
  module_.push_back(fn);
  auto *block = fn.addEntryBlock();
  return block;
}

void CaffeImporter::AddWeightFileOp(mlir::Block *block,
                                    const llvm::StringRef &weightFilename) {
  auto weight_type = MemRefType::get({0x80000000}, elementType_);
  auto weight_attr = builder_.getStringAttr(weightFilename);
  auto op = OpBuilder(block).create<tpu::WeightFileOp>(builder_.getUnknownLoc(),
                                                       weight_type, weight_attr);
  // auto op = OpBuilder(block).create<tpu::WeightFileOp>(
  //    builder_.getUnknownLoc(), mlir::NoneType(), weight_attr);
  weightFileVar_ = op.getResult();
}

TensorFile *CaffeImporter::GetWeightFile() {
  auto wfOp = cast<tpu::WeightFileOp>(weightFileVar_->getDefiningOp());
  return wfOp.get();
}

mlir::Value *CaffeImporter::AddLoadWeightOp(mlir::Block *block, std::string name,
                                            TensorType &type) {
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder_.getNamedAttr("name", builder_.getStringAttr(name)));
  return OpBuilder(block).create<tpu::LoadWeightOp>(builder_.getUnknownLoc(), type,
                                                    ArrayRef<Value *>{weightFileVar_},
                                                    ArrayRef<NamedAttribute>{attrs});
}

void CaffeImporter::AddReturnOp(mlir::Block *block,
                                std::map<std::string, mlir::Type> &outputs) {
  std::vector<mlir::Value *> returns;
  for (auto e : outputs) {
    auto it = tensor_map_.find(e.first);
    assert(it != tensor_map_.end());
    mlir::Value *var = it->second;
    returns.push_back(var);
  }
  OpBuilder(block).create<ReturnOp>(builder_.getUnknownLoc(),
                                    llvm::ArrayRef<mlir::Value *>{returns});
}

mlir::Value *CaffeImporter::GetLayerInput(caffe::Layer<float> *layer) {
  auto layer_param = layer->layer_param();
  auto it = tensor_map_.find(layer_param.bottom(0));
  assert(it != tensor_map_.end());
  mlir::Value *input = it->second;
  DEBUG_WITH_TYPE(DEBUG_TYPE "_VERBOSE", input->getType().dump(););
  return input;
}

std::vector<mlir::Value *> CaffeImporter::GetLayerInputs(caffe::Layer<float> *layer) {
  auto layer_param = layer->layer_param();
  std::vector<mlir::Value *> inputs;
  for (int i = 0; i < layer_param.bottom_size(); ++i) {
    auto it = tensor_map_.find(layer_param.bottom(i));
    assert(it != tensor_map_.end());
    inputs.push_back(it->second);
    DEBUG_WITH_TYPE(DEBUG_TYPE "_VERBOSE", it->second->getType().dump(););
  }
  return inputs;
}

void CaffeImporter::ConvertLayers(mlir::Block *block, caffe::Net<float> &net) {
  for (size_t i = 0; i <= net.layers().size() - 1; ++i) {
    auto layer = net.layers()[i].get();
    LLVM_DEBUG(printCaffeLayerParam(layer););

    if (strcmp(layer->type(), "Input") == 0) {
      convertInputLayer(block, layer);
    } else if (strcmp(layer->type(), "Split") == 0) {
      convertSplitLayer(block, layer);
    } else if ((strcmp(layer->type(), "Convolution") == 0) ||
               (strcmp(layer->type(), "ConvolutionDepthwise") == 0)) {
      convertConvolutionLayer(block, layer);
    } else if (strcmp(layer->type(), "InnerProduct") == 0) {
      convertInnerProductLayer(block, layer);
    } else if (strcmp(layer->type(), "Pooling") == 0) {
      convertPoolingLayer(block, layer);
    } else if (strcmp(layer->type(), "BatchNorm") == 0) {
      convertBatchNormLayer(block, layer);
    } else if (strcmp(layer->type(), "Scale") == 0) {
      convertScaleLayer(block, layer);
    } else if (strcmp(layer->type(), "ReLU") == 0) {
      convertReLULayer(block, layer);
    } else if (strcmp(layer->type(), "PReLU") == 0) {
      convertPReLULayer(block, layer);
    } else if (strcmp(layer->type(), "Eltwise") == 0) {
      convertEltwiseLayer(block, layer);
    } else if (strcmp(layer->type(), "Upsample") == 0) {
      convertUpsampleLayer(block, layer);
    } else if (strcmp(layer->type(), "Softmax") == 0) {
      convertSoftmaxLayer(block, layer);
    } else if (strcmp(layer->type(), "Concat") == 0) {
      convertConcatLayer(block, layer);
    } else if (strcmp(layer->type(), "Dropout") == 0) {
      convertDropoutLayer(block, layer);
    } else if (strcmp(layer->type(), "DummyData") == 0) {
      convertDummyDataLayer(block, layer);
    } else if (strcmp(layer->type(), "Crop") == 0) {
      convertCropLayer(block, layer);
    } else if (strcmp(layer->type(), "Sigmoid") == 0) {
      convertSigmoidLayer(block, layer);
    } else if (strcmp(layer->type(), "Flatten") == 0) {
      convertFlattenLayer(block, layer);
    } else if (strcmp(layer->type(), "Slice") == 0) {
      convertSliceLayer(block, layer);
    } else if (strcmp(layer->type(), "Reshape") == 0) {
      convertReshapeLayer(block, layer);
    } else if (strcmp(layer->type(), "Permute") == 0) {
      convertPermuteLayer(block, layer);
    } else if (strcmp(layer->type(), "Normalize") == 0) {
      convertNormalizeLayer(block, layer);
    } else if (strcmp(layer->type(), "TanH") == 0) {
      convertTanHLayer(block, layer);
    } else if (strcmp(layer->type(), "Deconvolution") == 0) {
      convertConvolutionLayer(block, layer, true);
    } else if (strcmp(layer->type(), "PriorBox") == 0) {
      convertPriorBoxLayer(block, layer);
    } else if (strcmp(layer->type(), "DetectionOutput") == 0) {
      convertDetectionOutputLayer(block, layer);
    } else if (strcmp(layer->type(), "Power") == 0) {
      convertPowerLayer(block, layer);
    } else if (strcmp(layer->type(), "ShuffleChannel") == 0) {
      convertShuffleChannelLayer(block, layer);
    } else if (strcmp(layer->type(), "RetinaFaceDetection") == 0) {
      convertRetinaFaceDetectionLayer(block, layer);
    } else if (strcmp(layer->type(), "YoloDetection") == 0) {
      convertYoloDetectionLayer(block, layer);
    } else {
      llvm::errs() << "    UNKNOWN : " << layer->type() << "\n";
      assert(false);
    }
  }
}

void CaffeImporter::convertBatchNormLayer(mlir::Block *block,
                                          caffe::Layer<float> *layer) {
  mlir::Value *input_var = GetLayerInput(layer);

  auto layer_param = layer->layer_param();

  float epsilon = 1e-5;
  if (layer_param.has_batch_norm_param())
    epsilon = layer_param.batch_norm_param().eps();

  int64_t n, c, h, w;
  llvm::ArrayRef<int64_t> input_var_shape =
      input_var->getType().dyn_cast<mlir::TensorType>().getShape();

  assert(input_var_shape.size() == 4 || input_var_shape.size() == 2);

  n = input_var_shape[0];
  c = input_var_shape[1];
  LLVM_DEBUG(llvm::errs() << "  N: " << n << ", C: " << c;);
  if (input_var_shape.size() == 4) {
    h = input_var_shape[2];
    w = input_var_shape[3];
    LLVM_DEBUG(llvm::errs() << ", IH*IW: " << h << " * " << w;);
  }
  LLVM_DEBUG(llvm::errs() << "\n";);

  std::vector<Value *> operands;
  operands.push_back(input_var);

  // - blobs_[2] holds the scale, which is one scalar data
  // - blobs_[0] holds the mean
  // - blobs_[1] holds the variance
  assert(layer->blobs().size() == 3);

  auto mean_name = layer->layer_param().name() + "_0";
  auto mean_type = RankedTensorType::get({c}, elementType_);
  GetWeightFile()->addTensor(mean_name, layer->blobs()[0].get()->cpu_data(), mean_type);
  operands.push_back(AddLoadWeightOp(block, mean_name, mean_type));

  auto variance_name = layer->layer_param().name() + "_1";
  auto variance_type = RankedTensorType::get({c}, elementType_);
  GetWeightFile()->addTensor(variance_name, layer->blobs()[1].get()->cpu_data(),
                             variance_type);
  operands.push_back(AddLoadWeightOp(block, variance_name, variance_type));

  auto scale_name = layer->layer_param().name() + "_2";
  auto scale_type = RankedTensorType::get({1}, elementType_);
  GetWeightFile()->addTensor(scale_name, layer->blobs()[2].get()->cpu_data(), scale_type);
  operands.push_back(AddLoadWeightOp(block, scale_name, scale_type));

  // auto result_type = RankedTensorType::get({n, c, h, w}, elementType_);
  auto result_type = RankedTensorType::get(input_var_shape, elementType_);

  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      builder_.getNamedAttr("name", builder_.getStringAttr(layer_param.name())));
  attrs.push_back(
      builder_.getNamedAttr("variance_epsilon", builder_.getF32FloatAttr(epsilon)));
  auto op = OpBuilder(block).create<tpu::BatchNormOp>(
      builder_.getUnknownLoc(), result_type, ArrayRef<Value *>{operands},
      ArrayRef<NamedAttribute>{attrs});
  auto result_var = op.getResult();

  tensor_map_[layer_param.top(0)] = result_var;
}

void CaffeImporter::convertConcatLayer(mlir::Block *block, caffe::Layer<float> *layer) {
  std::vector<mlir::Value *> input_vars = GetLayerInputs(layer);

  auto layer_param = layer->layer_param();
  auto concat_param = layer_param.concat_param();
  int axis = concat_param.axis();
  int64_t n = 0, c = 0, h = 0, w = 0;
  int64_t concat_axis_dim = 0;

  if (input_vars.size() == 1) {
    // special case for YOLOv3 caffe model, which has only one input
    // remove that node
    // llvm::errs() << "WARNING: concat layer has only one input\n";
    tensor_map_[layer_param.top(0)] = input_vars[0];
    return;
  }

  RankedTensorType result_type = nullptr;
  llvm::ArrayRef<int64_t> input_shape =
      input_vars[0]->getType().dyn_cast<mlir::TensorType>().getShape();

  if (input_shape.size() == 4) {
    for (uint32_t i = 0; i < input_vars.size(); i++) {

      input_shape = input_vars[i]->getType().dyn_cast<mlir::TensorType>().getShape();

      n = input_shape[0];
      c = input_shape[1];
      h = input_shape[2];
      w = input_shape[3];
      concat_axis_dim += input_shape[axis];
      LLVM_DEBUG(llvm::errs() << " var: " << i << "  N: " << n << ", C: " << c
                              << ", IH*IW: " << h << " * " << w << "\n";);
    }

    switch (axis) {
      case 0:
        n = concat_axis_dim;
        break;
      case 1:
        c = concat_axis_dim;
        break;
      case 2:
        h = concat_axis_dim;
        break;
      case 3:
        w = concat_axis_dim;
        break;
      default:
        assert(0);
    }

    LLVM_DEBUG(llvm::errs() << " axis: " << input_vars.size() << "  N: " << n
                            << ", C: " << c << ", IH*IW: " << h << " * " << w << "\n";);

    // construct OP
    result_type = RankedTensorType::get({n, c, h, w}, elementType_);

  } else if (input_shape.size() == 2) {

    for (uint32_t i = 0; i < input_vars.size(); i++) {
      input_shape = input_vars[i]->getType().dyn_cast<mlir::TensorType>().getShape();

      assert(input_shape.size() == 2);

      h = input_shape[0];
      w = input_shape[1];
      concat_axis_dim += input_shape[axis];
      LLVM_DEBUG(llvm::errs() << " var: " << i << ", IH*IW: " << h << " * " << w
                              << "\n";);
    }

    switch (axis) {
      case 0:
        h = concat_axis_dim;
        break;
      case 1:
        w = concat_axis_dim;
        break;
      default:
        assert(0);
    }

    LLVM_DEBUG(llvm::errs() << " axis: " << input_vars.size() << ", IH*IW: " << h << " * "
                            << w << "\n";);
    // construct OP
    result_type = RankedTensorType::get({h, w}, elementType_);

  } else if (input_shape.size() == 3) {

    for (uint32_t i = 0; i < input_vars.size(); i++) {
      input_shape = input_vars[i]->getType().dyn_cast<mlir::TensorType>().getShape();

      assert(input_shape.size() == 3);

      c = input_shape[0];
      h = input_shape[1];
      w = input_shape[2];
      concat_axis_dim += input_shape[axis];
      LLVM_DEBUG(llvm::errs() << " var: " << i << ", IH*IW: " << h << " * " << w
                              << "\n";);
    }

    switch (axis) {
      case 0:
        c = concat_axis_dim;
        break;
      case 1:
        h = concat_axis_dim;
        break;
      case 2:
        w = concat_axis_dim;
        break;
      default:
        assert(0);
    }

    LLVM_DEBUG(llvm::errs() << " axis: " << axis << ", C: " << c << ", OH*OW: " << h
                            << " * " << w << "\n";);

    // construct OP
    result_type = RankedTensorType::get({c, h, w}, elementType_);

  } else {
    assert(0);
  }

  std::vector<Value *> operands;
  for (unsigned i = 0; i < input_vars.size(); i++) {
    operands.push_back(input_vars[i]);
  }
  auto NoneOp = OpBuilder(block).create<tpu::NoneOp>(builder_.getUnknownLoc(),
                                                     builder_.getNoneType());
  operands.push_back(NoneOp.getResult()); // quant_scale
  operands.push_back(NoneOp.getResult()); // quant_zeropoint
  operands.push_back(NoneOp.getResult()); // quant_rshift
  operands.push_back(NoneOp.getResult()); // quant_multiplier

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder_.getNamedAttr("axis", builder_.getI32IntegerAttr(axis)));
  attrs.push_back(
      builder_.getNamedAttr("name", builder_.getStringAttr(layer_param.name())));
  attrs.push_back(builder_.getNamedAttr("quant", getDefaultQuantParam(builder_)));
  auto op = OpBuilder(block).create<tpu::ConcatOp>(builder_.getUnknownLoc(), result_type,
                                                   ArrayRef<Value *>{operands},
                                                   ArrayRef<NamedAttribute>{attrs});
  auto result_var = op.getResult();

  tensor_map_[layer_param.top(0)] = result_var;
}

void CaffeImporter::convertConvolutionLayer(mlir::Block *block,
                                            caffe::Layer<float> *layer, bool is_deconv) {
  mlir::Value *input_var = GetLayerInput(layer);

  auto layer_param = layer->layer_param();
  assert(layer_param.has_convolution_param());
  auto p = layer_param.convolution_param();
  int64_t n, ic, oc, g;
  std::vector<int64_t> kernel(2), stride(2), padding(2), dilation(2);
  std::vector<int64_t> ifmap(2),
      ofmap(2); // spatial dims only (height and width)

  bool with_bias = p.bias_term();
  oc = p.num_output();
  if (strcmp(layer->type(), "ConvolutionDepthwise") == 0) {
    g = oc;
  } else {
    g = p.has_group() ? p.group() : 1;
  }

  kernel[0] = p.has_kernel_h()
                  ? p.kernel_h()
                  : p.kernel_size_size() > 1 ? p.kernel_size(1) : p.kernel_size(0);
  kernel[1] = p.has_kernel_w() ? p.kernel_w() : p.kernel_size(0);
  stride[0] =
      p.has_stride_h()
          ? p.stride_h()
          : p.stride_size() > 1 ? p.stride(1) : p.stride_size() > 0 ? p.stride(0) : 1;
  stride[1] = p.has_stride_w() ? p.stride_w() : p.stride_size() > 0 ? p.stride(0) : 1;
  padding[0] = p.has_pad_h()
                   ? p.pad_h()
                   : p.pad_size() > 1 ? p.pad(1) : p.pad_size() > 0 ? p.pad(0) : 0;
  padding[1] = p.has_pad_w() ? p.pad_w() : p.pad_size() > 0 ? p.pad(0) : 0;
  dilation[0] =
      p.dilation_size() > 1 ? p.dilation(1) : p.dilation_size() > 0 ? p.dilation(0) : 1;
  dilation[1] = p.dilation_size() > 0 ? p.dilation(0) : 1;

  // get input shape from input var
  llvm::ArrayRef<int64_t> input_shape =
      input_var->getType().dyn_cast<mlir::TensorType>().getShape();
  assert(input_shape.size() == 4);
  n = input_shape[0];
  ic = input_shape[1];
  ifmap[0] = input_shape[2];
  ifmap[1] = input_shape[3];
  // get output shape from inference
  if (!is_deconv) {
    ofmap[0] =
        calcConv2DSpatialOutput(ifmap[0], kernel[0], stride[0], padding[0], dilation[0]);
    ofmap[1] =
        calcConv2DSpatialOutput(ifmap[1], kernel[1], stride[1], padding[1], dilation[1]);
  } else {
    ofmap[0] = calcDeConv2DSpatialOutput(ifmap[0], kernel[0], stride[0], padding[0],
                                         dilation[0]);
    ofmap[1] = calcDeConv2DSpatialOutput(ifmap[1], kernel[1], stride[1], padding[1],
                                         dilation[1]);
  }
  bool is_dw = false;
  if (g == oc) {
    is_dw = true;
  }

  LLVM_DEBUG(llvm::errs() << "  N: " << n << ", IC: " << ic << ", IH*IW: " << ifmap[0]
                          << " * " << ifmap[1] << ", OC: " << oc
                          << ", OH*OW: " << ofmap[0] << " * " << ofmap[1] << "\n";
             llvm::errs() << "  with_bias: " << with_bias << ", K: " << kernel[0] << " * "
                          << kernel[1] << ", S: " << stride[0] << " * " << stride[1]
                          << ", P: " << padding[0] << " * " << padding[1]
                          << ", D: " << dilation[0] << " * " << dilation[1]
                          << ", group: " << g << "\n";);

  std::vector<Value *> operands;
  operands.push_back(input_var);
  auto NoneOp = OpBuilder(block).create<tpu::NoneOp>(builder_.getUnknownLoc(),
                                                     builder_.getNoneType());

  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  auto filter_name = layer->layer_param().name() + "_0";
  TensorType filter_type;
  if (g != 1) {
    filter_type =
        RankedTensorType::get({g, oc / g, ic / g, kernel[0], kernel[1]}, elementType_);
  } else {
    filter_type = RankedTensorType::get({oc, ic, kernel[0], kernel[1]}, elementType_);
  }
  GetWeightFile()->addTensor(filter_name, layer->blobs()[0].get()->cpu_data(),
                             filter_type);
  operands.push_back(AddLoadWeightOp(block, filter_name, filter_type));
  if (with_bias) {
    auto bias_name = layer->layer_param().name() + "_1";
    auto bias_type = RankedTensorType::get({oc}, elementType_);
    GetWeightFile()->addTensor(bias_name, layer->blobs()[1].get()->cpu_data(), bias_type);
    operands.push_back(AddLoadWeightOp(block, bias_name, bias_type));
  } else {
    operands.push_back(NoneOp.getResult());
  }
  operands.push_back(NoneOp.getResult()); // quant_scale
  operands.push_back(NoneOp.getResult()); // quant_zeropoint
  operands.push_back(NoneOp.getResult()); // quant_rshift
  operands.push_back(NoneOp.getResult()); // quant_multiplier

  // construct OP
  auto result_type = RankedTensorType::get({n, oc, ofmap[0], ofmap[1]}, elementType_);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      builder_.getNamedAttr("name", builder_.getStringAttr(layer_param.name())));
  attrs.push_back(builder_.getNamedAttr(
      "param",
      tpu::ConvParam::get(
          builder_.getI32IntegerAttr(stride[0]), builder_.getI32IntegerAttr(stride[1]),
          (padding[0] || padding[1]) ? builder_.getStringAttr("SAME")
                                     : builder_.getStringAttr("VALID"),
          builder_.getI32IntegerAttr(dilation[0]),
          builder_.getI32IntegerAttr(dilation[1]), builder_.getI32IntegerAttr(g),
          builder_.getBoolAttr(is_dw), builder_.getBoolAttr(with_bias),
          builder_.getBoolAttr(false), builder_.getContext())));
  attrs.push_back(builder_.getNamedAttr("quant", getDefaultQuantParam(builder_)));
  Value *result_var;
  if (!is_deconv) {
    auto op = OpBuilder(block).create<tpu::Conv2DOp>(
        builder_.getUnknownLoc(), result_type, ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    result_var = op.getResult();
  } else {
    auto op = OpBuilder(block).create<tpu::DeConv2DOp>(
        builder_.getUnknownLoc(), result_type, ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    result_var = op.getResult();
  }
  tensor_map_[layer_param.top(0)] = result_var;
}

void CaffeImporter::convertCropLayer(mlir::Block *block, caffe::Layer<float> *layer) {
  std::vector<mlir::Value *> input_vars = GetLayerInputs(layer);

  assert(input_vars.size() == 2 && "Crop expected two input blobs");

  auto layer_param = layer->layer_param();
  auto crop_param = layer_param.crop_param();

  // get input shape from input vars
  llvm::ArrayRef<int64_t> input_shape =
      input_vars[0]->getType().dyn_cast<mlir::TensorType>().getShape();
  llvm::ArrayRef<int64_t> _crop_shape =
      input_vars[1]->getType().dyn_cast<mlir::TensorType>().getShape();

  int input_dim = input_shape.size();
  int axis_index = crop_param.axis();
  int start_axis = axis_index;
  int offset_size = crop_param.offset_size();
  if (offset_size > 1) {
    // the number of crop values specified must be equal to the number
    // of dimensions following axis
    assert((offset_size + axis_index <= input_dim) &&
           " number of offset values specified must be equal to the number "
           "ofdimensions following axis.");
  }

  LLVM_DEBUG(llvm::errs() << "\n  Crop\n"
                          << "    bottom: " << input_shape[0] << ", " << input_shape[1]
                          << ", " << input_shape[2] << ", " << input_shape[3] << "\n"
                          << "    crop: " << _crop_shape[0] << ", " << _crop_shape[1]
                          << ", " << _crop_shape[2] << ", " << _crop_shape[3] << "\n"
                          << "    start_axis " << start_axis << ", offset_size() "
                          << crop_param.offset_size() << "\n";);

  std::vector<int> output_shape(input_dim);
  std::vector<int> crop_shape(input_dim);
  std::vector<int> crop_offset(input_dim);
  crop_shape.assign(_crop_shape.begin(), _crop_shape.end());
  // Determine crop offsets and the new shape post-crop
  for (int i = 0; i < input_dim; ++i) {
    int offset = 0;
    int new_size = input_shape[i];
    if (i >= start_axis) {
      new_size = crop_shape[i];
      if (crop_param.offset_size() == 1) {
        // If only one offset is given, all crops have the same offset.
        offset = crop_param.offset(0);
      } else if (crop_param.offset_size() > 1) {
        // For several offsets, the number of offsets must be equal to the
        // number of dimensions to crop, that is dimensions after the axis.
        offset = crop_param.offset(i - start_axis);
      }
    }

    LLVM_DEBUG(llvm::errs() << "    [" << i << "] crop_offset=" << offset
                            << ", new_size=" << new_size << "\n";);

    output_shape[i] = new_size;
    crop_offset[i] = offset;
  }

  // consruct OP
  std::vector<Value *> operands;
  operands.push_back(input_vars[0]);

  Operation *shapeOp = input_vars[1]->getDefiningOp();
  if (isa<tpu::DummyOp>(shapeOp)) {
    // erase the dummy
    shapeOp->erase();
  }

  auto result_type = RankedTensorType::get(
      {output_shape[0], output_shape[1], output_shape[2], output_shape[3]}, elementType_);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder_.getNamedAttr(
      "crop_shape", builder_.getI32ArrayAttr(ArrayRef<int32_t>({crop_shape}))));
  attrs.push_back(builder_.getNamedAttr(
      "crop_offset", builder_.getI32ArrayAttr(ArrayRef<int32_t>({crop_offset}))));
  attrs.push_back(
      builder_.getNamedAttr("name", builder_.getStringAttr(layer_param.name())));
  attrs.push_back(builder_.getNamedAttr("quant", getDefaultQuantParam(builder_)));
  // we only accept first input to IR, second input shape will be attribute.
  auto op = OpBuilder(block).create<tpu::CropOp>(builder_.getUnknownLoc(), result_type,
                                                 ArrayRef<Value *>{operands},
                                                 ArrayRef<NamedAttribute>{attrs});
  auto result_var = op.getResult();
  tensor_map_[layer_param.top(0)] = result_var;
}

void CaffeImporter::convertDetectionOutputLayer(mlir::Block *block,
                                                caffe::Layer<float> *layer) {

  std::vector<mlir::Value *> input_vars = GetLayerInputs(layer);

  auto layer_param = layer->layer_param();
  auto detection_output_param = layer_param.detection_output_param();
  std::string code_type;

  if (detection_output_param.code_type() == caffe::PriorBoxParameter_CodeType_CORNER) {
    code_type = "CORNER";
  } else if (detection_output_param.code_type() ==
             caffe::PriorBoxParameter_CodeType_CENTER_SIZE) {
    code_type = "CENTER_SIZE";
  } else if (detection_output_param.code_type() ==
             caffe::PriorBoxParameter_CodeType_CORNER_SIZE) {
    code_type = "CORNER_SIZE";
  }

  // llvm::ArrayRef<int64_t> input_shape =
  //    input_vars[0]->getType().dyn_cast<mlir::TensorType>().getShape();

  std::vector<NamedAttribute> attrs;

  attrs.push_back(builder_.getNamedAttr(
      "num_classes", builder_.getI32IntegerAttr(detection_output_param.num_classes())));
  attrs.push_back(builder_.getNamedAttr(
      "share_location", builder_.getBoolAttr(detection_output_param.share_location())));
  attrs.push_back(builder_.getNamedAttr(
      "background_label_id",
      builder_.getI32IntegerAttr(detection_output_param.background_label_id())));

  attrs.push_back(builder_.getNamedAttr(
      "nms_threshold",
      builder_.getF32FloatAttr(detection_output_param.nms_param().nms_threshold())));
  attrs.push_back(builder_.getNamedAttr(
      "top_k", builder_.getI32IntegerAttr(detection_output_param.nms_param().top_k())));
  attrs.push_back(builder_.getNamedAttr("code_type", builder_.getStringAttr(code_type)));
  attrs.push_back(builder_.getNamedAttr(
      "keep_top_k", builder_.getI32IntegerAttr(detection_output_param.keep_top_k())));
  attrs.push_back(builder_.getNamedAttr(
      "confidence_threshold",
      builder_.getF32FloatAttr(detection_output_param.confidence_threshold())));
  attrs.push_back(
      builder_.getNamedAttr("name", builder_.getStringAttr(layer_param.name())));

  assert(1.0 == detection_output_param.nms_param().eta());
  assert(false == detection_output_param.variance_encoded_in_target());
  // construct OP
  auto result_type =
      RankedTensorType::get({1, 1, detection_output_param.keep_top_k(), 7}, elementType_);

  auto reshape_op = OpBuilder(block).create<tpu::DetectionOutputOp>(
      builder_.getUnknownLoc(), result_type, ArrayRef<Value *>{input_vars},
      ArrayRef<NamedAttribute>{attrs});

  auto result_var = reshape_op.getResult();
  tensor_map_[layer_param.top(0)] = result_var;
}

void CaffeImporter::convertDropoutLayer(mlir::Block *block, caffe::Layer<float> *layer) {
  mlir::Value *input_var = GetLayerInput(layer);

  auto layer_param = layer->layer_param();
  tensor_map_[layer_param.top(0)] = input_var;
}

void CaffeImporter::convertDummyDataLayer(mlir::Block *block,
                                          caffe::Layer<float> *layer) {
  auto layer_param = layer->layer_param();
  auto dummy_data_param = layer_param.dummy_data_param();
  if (dummy_data_param.shape_size() < 1) {
    assert(0 && "dummy data op no define dim");
  }
  auto dummy_shape = dummy_data_param.shape(0);

  int n = dummy_shape.dim(0);
  int c = dummy_shape.dim(1);
  int h = dummy_shape.dim(2);
  int w = dummy_shape.dim(3);
  LLVM_DEBUG(llvm::errs() << "DummyData  N: " << n << ", C: " << c << ", IH*IW: " << h
                          << " * " << w << "\n";);

  // construct OP
  auto result_type = RankedTensorType::get({n, c, h, w}, elementType_);
  auto op = OpBuilder(block).create<tpu::DummyOp>(builder_.getUnknownLoc(), result_type,
                                                  ArrayRef<Value *>{},
                                                  ArrayRef<NamedAttribute>{});
  auto result_var = op.getResult();
  tensor_map_[layer_param.top(0)] = result_var;
}

void CaffeImporter::convertEltwiseLayer(mlir::Block *block, caffe::Layer<float> *layer) {
  std::vector<mlir::Value *> input_vars = GetLayerInputs(layer);

  auto layer_param = layer->layer_param();
  auto eltwise_param = layer_param.eltwise_param();
  std::string method;
  assert(eltwise_param.coeff_size() == 0);
  if (eltwise_param.operation() == caffe::EltwiseParameter_EltwiseOp_SUM) {
    method = "SUM";
  } else if (eltwise_param.operation() == caffe::EltwiseParameter_EltwiseOp_PROD) {
    method = "PROD";
  } else if (eltwise_param.operation() ==
             caffe::EltwiseParameter_EltwiseOp_EltwiseOp_MAX) {
    method = "MAX";
  } else {
    assert(0 && "eltwise only support, SUM, PROD, MAX now");
  }
  int64_t n, c, h, w;
  llvm::ArrayRef<int64_t> input_shape =
      input_vars[0]->getType().dyn_cast<mlir::TensorType>().getShape();
  assert(input_shape.size() == 4);
  n = input_shape[0];
  c = input_shape[1];
  h = input_shape[2];
  w = input_shape[3];

  LLVM_DEBUG(llvm::errs() << "  N: " << n << ", C: " << c << ", IH*IW: " << h << " * "
                          << w << "\n";);

  // construct OP
  std::vector<Value *> operands;
  for (unsigned i = 0; i < input_vars.size(); i++) {
    operands.push_back(input_vars[i]);
  }
  auto NoneOp = OpBuilder(block).create<tpu::NoneOp>(builder_.getUnknownLoc(),
                                                     builder_.getNoneType());
  operands.push_back(NoneOp.getResult());
  operands.push_back(NoneOp.getResult());
  operands.push_back(NoneOp.getResult());
  operands.push_back(NoneOp.getResult());

  auto result_type = RankedTensorType::get({n, c, h, w}, elementType_);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      builder_.getNamedAttr("name", builder_.getStringAttr(layer_param.name())));
  attrs.push_back(builder_.getNamedAttr("quant", getDefaultQuantParam(builder_)));
  Value *result_var = nullptr;
  if (method == "SUM") {
    auto op = OpBuilder(block).create<tpu::EltwiseAddOp>(
        builder_.getUnknownLoc(), result_type, ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    result_var = op.getResult();
  } else if (method == "PROD") {
    auto op = OpBuilder(block).create<tpu::EltwiseMulOp>(
        builder_.getUnknownLoc(), result_type, ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    result_var = op.getResult();
  } else if (method == "MAX") {
    auto op = OpBuilder(block).create<tpu::EltwiseMaxOp>(
        builder_.getUnknownLoc(), result_type, ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    result_var = op.getResult();
  } else {
    assert(false);
  }

  tensor_map_[layer_param.top(0)] = result_var;
}

void CaffeImporter::convertFlattenLayer(mlir::Block *block, caffe::Layer<float> *layer) {
  mlir::Value *input_var = GetLayerInput(layer);

  auto layer_param = layer->layer_param();

  int64_t n, c, h, w;
  llvm::ArrayRef<int64_t> input_shape =
      input_var->getType().dyn_cast<mlir::TensorType>().getShape();
  // assert(input_shape.size() == 4);
  RankedTensorType result_type = nullptr;

  if (input_shape.size() == 4) {
    n = input_shape[0];
    c = input_shape[1];
    h = input_shape[2];
    w = input_shape[3];

    LLVM_DEBUG(llvm::errs() << "  N: " << n << ", C: " << c << ", IH*IW: " << h << " * "
                            << w << "\n";);

    // construct OP
    result_type = RankedTensorType::get({n, c * h * w}, elementType_);

  } else if (input_shape.size() == 3) { // for ssd mbox_conf_flatten layer
    c = input_shape[0];
    h = input_shape[1];
    w = input_shape[2];

    LLVM_DEBUG(llvm::errs() << "C: " << c << "IH*IW: " << h << " * " << w << "\n";);

    // construct OP
    result_type = RankedTensorType::get({c, h * w}, elementType_);

  } else {
    assert(0);
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      builder_.getNamedAttr("name", builder_.getStringAttr(layer_param.name())));
  // attrs.push_back(builder_.getNamedAttr("quant",
  // getDefaultQuantParam(builder_)));
  auto reshape_op = OpBuilder(block).create<tpu::ReshapeOp>(
      builder_.getUnknownLoc(), result_type, ArrayRef<Value *>{input_var},
      ArrayRef<NamedAttribute>{attrs});
  auto result_var = reshape_op.getResult();
  tensor_map_[layer_param.top(0)] = result_var;
}

void CaffeImporter::convertInnerProductLayer(mlir::Block *block,
                                             caffe::Layer<float> *layer) {
  mlir::Value *input_var = GetLayerInput(layer);

  auto layer_param = layer->layer_param();
  auto p = layer_param.inner_product_param();
  bool with_bias = p.bias_term();
  bool with_transpose = p.transpose();
  // M is the batch_size, K is input number, N is output number
  // (M, K) * (K, N) => (M, N)
  int64_t M, K, N;
  // N is the output num
  N = p.num_output();

  // get input shape from input var
  llvm::ArrayRef<int64_t> input_shape =
      input_var->getType().dyn_cast<mlir::TensorType>().getShape();
  bool reshape_first = false;
  if (input_shape.size() == 2) {
    M = input_shape[0];
    K = input_shape[1];
  } else {
    reshape_first = true;
    M = input_shape[0];
    K = 1;
    for (size_t i = 1; i <= input_shape.size() - 1; ++i) {
      K *= input_shape[i];
    }
  }
  // not support transpose for now
  assert(!with_transpose);

  LLVM_DEBUG(llvm::errs() << "  M: " << M << ", K: " << K << ", N: " << N
                          << ", with_bias: " << with_bias
                          << ", with_transpose: " << with_transpose << "\n";);

  mlir::Value *fc_input_var = input_var;
  // construct reshape OP
  if (reshape_first) {
    auto fc_input_type = RankedTensorType::get({M, K}, elementType_);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(builder_.getNamedAttr(
        "name", builder_.getStringAttr(layer_param.name() + "_reshape")));
    // attrs.push_back(builder_.getNamedAttr("quant",
    // getDefaultQuantParam(builder_)));
    auto reshape_op = OpBuilder(block).create<tpu::ReshapeOp>(
        builder_.getUnknownLoc(), fc_input_type, ArrayRef<Value *>{input_var},
        ArrayRef<NamedAttribute>{attrs});
    fc_input_var = reshape_op.getResult();
  }

  std::vector<Value *> operands;
  operands.push_back(fc_input_var);
  auto NoneOp = OpBuilder(block).create<tpu::NoneOp>(builder_.getUnknownLoc(),
                                                     builder_.getNoneType());

  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  auto filter_name = layer->layer_param().name() + "_0";
  auto filter_type = RankedTensorType::get({N, K}, elementType_);
  GetWeightFile()->addTensor(filter_name, layer->blobs()[0].get()->cpu_data(),
                             filter_type);
  operands.push_back(AddLoadWeightOp(block, filter_name, filter_type));
  if (with_bias) {
    auto bias_name = layer->layer_param().name() + "_1";
    auto bias_type = RankedTensorType::get({N}, elementType_);
    GetWeightFile()->addTensor(bias_name, layer->blobs()[1].get()->cpu_data(), bias_type);
    operands.push_back(AddLoadWeightOp(block, bias_name, bias_type));
  } else {
    operands.push_back(NoneOp.getResult());
  }
  operands.push_back(NoneOp.getResult()); // quant_scale
  operands.push_back(NoneOp.getResult()); // quant_zeropoint
  operands.push_back(NoneOp.getResult()); // quant_rshift
  operands.push_back(NoneOp.getResult()); // quant_multiplier

  // construct OP
  auto result_type = RankedTensorType::get({M, N}, elementType_);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder_.getNamedAttr("with_bias", builder_.getBoolAttr(with_bias)));
  attrs.push_back(
      builder_.getNamedAttr("name", builder_.getStringAttr(layer_param.name())));
  attrs.push_back(builder_.getNamedAttr("quant", getDefaultQuantParam(builder_)));
  auto op = OpBuilder(block).create<tpu::FullyConnectedOp>(
      builder_.getUnknownLoc(), result_type, ArrayRef<Value *>{operands},
      ArrayRef<NamedAttribute>{attrs});
  auto result_var = op.getResult();

  tensor_map_[layer_param.top(0)] = result_var;
}

void CaffeImporter::convertInputLayer(mlir::Block *block, caffe::Layer<float> *layer) {
  auto layer_param = layer->layer_param();
  assert(layer_param.bottom_size() == 0 && layer_param.top_size() == 1);

  std::vector<Value *> operands;
  operands.push_back(block->getArgument(0));

  auto result_type = block->getArgument(0)->getType();
  std::vector<NamedAttribute> attrs;
  // note input is a inserted layer, we should use top blob name rather than
  // layer_name
  std::string name = clAddPreprocess ? (std::string)layer_param.top(0) + "_raw"
                                     : (std::string)layer_param.top(0);
  attrs.push_back(builder_.getNamedAttr("name", builder_.getStringAttr(name)));
  attrs.push_back(builder_.getNamedAttr("quant", getDefaultQuantParam(builder_)));
  auto op = OpBuilder(block).create<tpu::InputOp>(builder_.getUnknownLoc(), result_type,
                                                  ArrayRef<Value *>{operands},
                                                  ArrayRef<NamedAttribute>{attrs});
  auto result_var = op.getResult();

  if (!clResolvePreprocess) {
    tensor_map_[name] = result_var;
    if (clAddPreprocess) {
      auto opd = result_var;
      if (clPreprocessTranspose) {
        opd = insertTransposeLayer(block, opd, layer_param.top(0));
      }
      insertPreprocessLayer(block, opd, layer_param.top(0));
    }
  } else {
    doPreprocess(block, result_var);
  }
}

void CaffeImporter::convertNormalizeLayer(mlir::Block *block,
                                          caffe::Layer<float> *layer) {

  std::vector<mlir::Value *> input_vars = GetLayerInputs(layer);
  auto input_var = input_vars[0];

  auto layer_param = layer->layer_param();
  auto norm_param = layer_param.norm_param();
  bool across_spatial = norm_param.across_spatial();
  bool channel_shared = norm_param.channel_shared();

  // implement for ssd case first
  assert(!across_spatial);

  std::vector<Value *> operands;
  operands.push_back(input_var);

  int64_t n, c, ih, iw, oh, ow;

  llvm::ArrayRef<int64_t> input_shape =
      input_var->getType().dyn_cast<mlir::TensorType>().getShape();
  assert(input_shape.size() == 4);
  n = input_shape[0];
  c = input_shape[1];
  ih = input_shape[2];
  iw = input_shape[3];
  oh = ih;
  ow = iw;

  LLVM_DEBUG(llvm::errs() << "  N: " << n << ", C: " << c << ", IH*IW: " << ih << " * "
                          << iw << ", OH*OW: " << oh << " * " << ow << "\n";);

  auto scale_name = layer->layer_param().name();
  auto scale_type = RankedTensorType::get({1, c}, elementType_);

  if (channel_shared) {
    assert(layer->blobs()[0].get()->count() == 1);
    std::vector<float> scale_input(c, layer->blobs()[0].get()->cpu_data()[0]);
    GetWeightFile()->addTensor(scale_name, scale_input.data(), scale_type);
  } else {
    assert(layer->blobs()[0].get()->count() == c);
    GetWeightFile()->addTensor(scale_name, layer->blobs()[0].get()->cpu_data(),
                               scale_type);
  }
  operands.push_back(AddLoadWeightOp(block, scale_name, scale_type));

  // construct Normalize OP
  auto result_type = RankedTensorType::get({n, c, oh, ow}, elementType_);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      builder_.getNamedAttr("across_spatial", builder_.getBoolAttr(across_spatial)));
  attrs.push_back(
      builder_.getNamedAttr("channel_shared", builder_.getBoolAttr(channel_shared)));
  attrs.push_back(
      builder_.getNamedAttr("name", builder_.getStringAttr(layer_param.name())));

  auto op = OpBuilder(block).create<tpu::NormalizeOp>(
      builder_.getUnknownLoc(), result_type, ArrayRef<Value *>{operands},
      ArrayRef<NamedAttribute>{attrs});

  auto result_var = op.getResult();
  tensor_map_[layer_param.top(0)] = result_var;
}

void CaffeImporter::convertPermuteLayer(mlir::Block *block, caffe::Layer<float> *layer) {
  std::vector<mlir::Value *> input_vars = GetLayerInputs(layer);

  auto layer_param = layer->layer_param();
  auto permute_param = layer_param.permute_param();
  llvm::ArrayRef<int64_t> input_shape =
      input_vars[0]->getType().dyn_cast<mlir::TensorType>().getShape();

  assert(permute_param.order_size() == 4);

  int64_t in, ic, ih, iw, on, oc, oh, ow;

  in = input_shape[0];
  ic = input_shape[1];
  ih = input_shape[2];
  iw = input_shape[3];

  on = input_shape[permute_param.order(0)];
  oc = input_shape[permute_param.order(1)];
  oh = input_shape[permute_param.order(2)];
  ow = input_shape[permute_param.order(3)];

  LLVM_DEBUG(llvm::errs() << "  IN: " << in << ", IC: " << ic << ", IH*IW: " << ih
                          << " * " << iw << "  ON: " << on << ", OC: " << oc
                          << ", OH*OW: " << oh << " * " << ow << "\n";);

  // construct OP
  auto result_type = RankedTensorType::get({on, oc, oh, ow}, elementType_);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      builder_.getNamedAttr("name", builder_.getStringAttr(layer_param.name())));
  attrs.push_back(builder_.getNamedAttr(
      "order0", builder_.getI32IntegerAttr(permute_param.order(0))));
  attrs.push_back(builder_.getNamedAttr("quant", getDefaultQuantParam(builder_)));
  attrs.push_back(builder_.getNamedAttr(
      "order1", builder_.getI32IntegerAttr(permute_param.order(1))));
  attrs.push_back(builder_.getNamedAttr(
      "order2", builder_.getI32IntegerAttr(permute_param.order(2))));
  attrs.push_back(builder_.getNamedAttr(
      "order3", builder_.getI32IntegerAttr(permute_param.order(3))));
  auto op = OpBuilder(block).create<tpu::PermuteOp>(builder_.getUnknownLoc(), result_type,
                                                    ArrayRef<Value *>{input_vars},
                                                    ArrayRef<NamedAttribute>{attrs});
  auto result_var = op.getResult();

  tensor_map_[layer_param.top(0)] = result_var;
}

void CaffeImporter::convertPoolingLayer(mlir::Block *block, caffe::Layer<float> *layer) {
  mlir::Value *input_var = GetLayerInput(layer);

  auto layer_param = layer->layer_param();
  assert(layer_param.has_pooling_param());
  auto p = layer_param.pooling_param();

  bool is_average_pooling;
  bool is_global_pooling;
  int64_t n, c;
  std::vector<int64_t> kernel(2), stride(2), padding(2);
  std::vector<int64_t> ifmap(2),
      ofmap(2); // spatial dims only (height and width)

  if (p.pool() == caffe::PoolingParameter_PoolMethod_AVE) {
    is_average_pooling = true;
  } else if (p.pool() == caffe::PoolingParameter_PoolMethod_MAX) {
    is_average_pooling = false;
  } else {
    assert(false && "Invalid pool type");
  }
  is_global_pooling = p.global_pooling();

  // get input shape from input var
  llvm::ArrayRef<int64_t> input_shape =
      input_var->getType().dyn_cast<mlir::TensorType>().getShape();
  assert(input_shape.size() == 4);
  n = input_shape[0];
  c = input_shape[1];
  ifmap[0] = input_shape[2];
  ifmap[1] = input_shape[3];

  if (is_global_pooling) {
    kernel[0] = ifmap[0];
    kernel[1] = ifmap[1];
  } else {
    kernel[0] = p.has_kernel_h() ? p.kernel_h() : p.kernel_size();
    kernel[1] = p.has_kernel_w() ? p.kernel_w() : p.kernel_size();
  }
  stride[0] = p.has_stride_h() ? p.stride_h() : p.has_stride() ? p.stride() : 1;
  stride[1] = p.has_stride_w() ? p.stride_w() : p.has_stride() ? p.stride() : 1;
  padding[0] = p.has_pad_h() ? p.pad_h() : p.has_pad() ? p.pad() : 0;
  padding[1] = p.has_pad_w() ? p.pad_w() : p.has_pad() ? p.pad() : 0;

  // when (ih - kh) % sh != 0, asymetric padding are needed
  // and ceiling/floor mode are different
  // eg.1 resnet50 112x112 -> 56x56, k=3x3, s=2x2, ph=0, pw=0, ceil_mode
  // eg.2 resnet50 7x7 -> 1x1, k=7x7, s=1x1, ph=0, pw=0, ceil_mode
  // eg.3 retinaface 300x300 -> 151x151, k=3x3, s=2x2, ph=1, pw=1, ceil_mode
  //   => pad_top = 1, pad_bottom = 2, pad_left = 1, pad_right = 2
  // eg.4 300x300 -> 150x150, k=3x3, s=2x2, ph=1, pw=1, floor_mode
  //   => pad_top = 1, pad_bottom = 1, pad_left = 1, pad_right = 1

  std::vector<int64_t> padding_tl(2), padding_br(2);
  bool ceil_mode = p.ceil_mode();
  bool round_mode = p.round_mode();
  //  if round_mode is not default value, use round_mode as ceil_mode
  if (round_mode == caffe::PoolingParameter_RoundMode_FLOOR)
    ceil_mode = false;

  padding_tl[0] = padding[0];
  padding_tl[1] = padding[1];
  padding_br[0] = padding[0];
  padding_br[1] = padding[1];

  for (size_t i = 0; i < 2; ++i) {
    if (ceil_mode)
      ofmap[i] =
          (static_cast<int>(ceil(
               static_cast<float>(ifmap[i] + 2 * padding[i] - kernel[i]) / stride[i])) +
           1);
    else
      ofmap[i] =
          (static_cast<int>(floor(
               static_cast<float>(ifmap[i] + 2 * padding[i] - kernel[i]) / stride[i])) +
           1);

    int remain_pixel = (ifmap[i] + 2 * padding[i] - kernel[i]) % stride[i];
    if (remain_pixel > 0) {
      if (ceil_mode)
        padding_br[i] += (stride[i] - remain_pixel);
      else
        padding_br[i] -= remain_pixel;
    }
  }

  if (is_global_pooling) {
    assert((padding[0] == 0) && (padding[1] == 0));
    assert((stride[0] == 1) && (stride[1] == 1));
    assert((ofmap[0] == 1) && (ofmap[1] == 1));
  }

  LLVM_DEBUG(llvm::errs() << "  N: " << n << ", C: " << c << ", IH*IW: " << ifmap[0]
                          << " * " << ifmap[1] << ", OH*OW: " << ofmap[0] << " * "
                          << ofmap[1]
                          << ", type: " << (is_average_pooling ? "AVG" : "MAX") << "\n";
             llvm::errs() << "  K: " << kernel[0] << " * " << kernel[1]
                          << ", S: " << stride[0] << " * " << stride[1]
                          << ", P_TL: " << padding_tl[0] << " * " << padding_tl[1]
                          << ", P_BR: " << padding_br[0] << " * " << padding_br[1]
                          << ", global_pooling: " << is_global_pooling << "\n";);

  // construct OP
  std::vector<Value *> operands;
  operands.push_back(input_var);
  auto NoneOp = OpBuilder(block).create<tpu::NoneOp>(builder_.getUnknownLoc(),
                                                     builder_.getNoneType());

  auto result_type = RankedTensorType::get({n, c, ofmap[0], ofmap[1]}, elementType_);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      builder_.getNamedAttr("name", builder_.getStringAttr(layer_param.name())));
  attrs.push_back(builder_.getNamedAttr(
      "param",
      tpu::PoolParam::get(builder_.getI32IntegerAttr(kernel[0]),     // kernel_h
                          builder_.getI32IntegerAttr(kernel[1]),     // kernel_w
                          builder_.getI32IntegerAttr(padding_tl[0]), // padding_t
                          builder_.getI32IntegerAttr(padding_br[0]), // padding_b
                          builder_.getI32IntegerAttr(padding_tl[1]), // padding_l
                          builder_.getI32IntegerAttr(padding_br[1]), // padding_r
                          builder_.getI32IntegerAttr(stride[0]),     // stride_h
                          builder_.getI32IntegerAttr(stride[1]),     // stride_w
                          builder_.getBoolAttr(false),               // do_relu
                          builder_.getContext())));
  attrs.push_back(builder_.getNamedAttr("quant", getDefaultQuantParam(builder_)));

  Value *result_var;
  if (is_average_pooling) {
    operands.push_back(NoneOp.getResult()); // quant_scale
    operands.push_back(NoneOp.getResult()); // quant_zeropoint
    operands.push_back(NoneOp.getResult()); // quant_rshift
    operands.push_back(NoneOp.getResult()); // quant_multiplier
    auto op = OpBuilder(block).create<tpu::PoolAvg2DOp>(
        builder_.getUnknownLoc(), result_type, ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    result_var = op.getResult();
  } else {
    auto op = OpBuilder(block).create<tpu::PoolMax2DOp>(
        builder_.getUnknownLoc(), result_type, ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    result_var = op.getResult();
  }

  tensor_map_[layer_param.top(0)] = result_var;
}

void CaffeImporter::convertPowerLayer(mlir::Block *block, caffe::Layer<float> *layer) {
  mlir::Value *input_var = GetLayerInput(layer);

  auto layer_param = layer->layer_param();
  auto power = layer_param.power_param().power();
  auto scale = layer_param.power_param().scale();
  auto shift = layer_param.power_param().shift();

  std::vector<Value *> operands;
  operands.push_back(input_var);
  // FIXME: it could remove once power_param = 1 and scale_param = 1 and
  // shift_param = 0
  // FIXME: deal with power = 0 or scale = 0
  if (shift == 0 && power == 1 && scale == 1) {
    return;
  }

  // int64_t n, c, h, w;
  llvm::ArrayRef<int64_t> input_shape =
      input_var->getType().dyn_cast<mlir::TensorType>().getShape();
  // assert(input_shape.size() == 4);
  int64_t nchw[4];
  for (uint64_t i = 0; i < input_shape.size(); i++) {
    nchw[i] = input_shape[i];
  }

  LLVM_DEBUG(llvm::errs() << "  N: " << nchw[0] << ", C: " << nchw[1]
                          << ", IH*IW: " << nchw[2] << " * " << nchw[3] << "\n";);

#if 0
  // we leverage depthwise to calculat a*x + b,
  // one shot by channel, we should reserve weight
  // for extend scale/shift from 1 dimension to <1, NUP_NUM, 1, 1>
  // FIXME: not harcode
  int channel = 32;
  int tbl_size = channel;
  auto table_type_scale = RankedTensorType::get({1, channel, 1, 1}, elementType_);
  std::vector<float> dataVec_fp32;
  dataVec_fp32.reserve(tbl_size);
  auto filter_name = layer->layer_param().name()+"_scale";
  GetWeightFile()->addTensor(filter_name, &dataVec_fp32, table_type_scale);
  operands.push_back(AddLoadWeightOp(block, filter_name, table_type_scale));

  // we just allocate 1 batch cuz
  // `AssignWeightAddress.cpp` auto seperate high/low part into int8 buffer
  tbl_size = 1 * channel;
  auto table_type_shift = RankedTensorType::get({1, channel, 1, 1}, elementType_);
  dataVec_fp32.reserve(tbl_size);
  filter_name = layer->layer_param().name()+"_shift";
  GetWeightFile()->addTensor(filter_name, &dataVec_fp32, table_type_shift);
  operands.push_back(AddLoadWeightOp(block, filter_name, table_type_shift));
#endif

  // construct OP
  auto result_type = RankedTensorType::get(input_shape, elementType_);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder_.getNamedAttr("power", builder_.getF32FloatAttr(power)));
  attrs.push_back(builder_.getNamedAttr("scale", builder_.getF32FloatAttr(scale)));
  attrs.push_back(builder_.getNamedAttr("shift", builder_.getF32FloatAttr(shift)));

  attrs.push_back(
      builder_.getNamedAttr("name", builder_.getStringAttr(layer_param.name())));
  attrs.push_back(builder_.getNamedAttr("quant", getDefaultQuantParam(builder_)));

  auto op = OpBuilder(block).create<tpu::PowerOp>(builder_.getUnknownLoc(), result_type,
                                                  ArrayRef<Value *>{operands},
                                                  ArrayRef<NamedAttribute>{attrs});
  auto result_var = op.getResult();

  tensor_map_[layer_param.top(0)] = result_var;
}

void CaffeImporter::convertPReLULayer(mlir::Block *block, caffe::Layer<float> *layer) {
  mlir::Value *input_var = GetLayerInput(layer);

  auto layer_param = layer->layer_param();
  auto prelu_param = layer_param.prelu_param();

  int64_t n, c, h, w;
  llvm::ArrayRef<int64_t> input_shape =
      input_var->getType().dyn_cast<mlir::TensorType>().getShape();
  assert(input_shape.size() == 4);
  n = input_shape[0];
  c = input_shape[1];
  h = input_shape[2];
  w = input_shape[3];

  // int batch_size = layer->blobs()[0].get()->num();
  // int channels = layer->blobs()[0].get()->channels();
  // int height = layer->blobs()[0].get()->height();
  // int width = layer->blobs()[0].get()->width();
  std::vector<Value *> operands;
  operands.push_back(input_var);

  // - blobs_[0] holds the negative_slope
  auto negative_slope_name = layer->layer_param().name() + "_0";
  auto negative_slope_type = RankedTensorType::get({1, c, 1, 1}, elementType_);
  GetWeightFile()->addTensor(negative_slope_name, layer->blobs()[0].get()->cpu_data(),
                             negative_slope_type);
  operands.push_back(AddLoadWeightOp(block, negative_slope_name, negative_slope_type));

  auto NoneOp = OpBuilder(block).create<tpu::NoneOp>(builder_.getUnknownLoc(),
                                                     builder_.getNoneType());
  for (int i = 0; i < 8; i++) {
    operands.push_back(
        NoneOp.getResult()); // quant: scale/zp/rshift/muliplier, pos and neg
  }
  // construct OP
  auto result_type = RankedTensorType::get({n, c, h, w}, elementType_);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      builder_.getNamedAttr("name", builder_.getStringAttr(layer_param.name())));
  attrs.push_back(builder_.getNamedAttr("quant", getDefaultQuantParam(builder_)));
  auto op = OpBuilder(block).create<tpu::PReluOp>(builder_.getUnknownLoc(), result_type,
                                                  ArrayRef<Value *>{operands},
                                                  ArrayRef<NamedAttribute>{attrs});
  auto result_var = op.getResult();

  tensor_map_[layer_param.top(0)] = result_var;
}

void CaffeImporter::convertPriorBoxLayer(mlir::Block *block, caffe::Layer<float> *layer) {

  std::vector<mlir::Value *> input_vars = GetLayerInputs(layer);

  auto layer_param = layer->layer_param();
  auto prior_box_param = layer_param.prior_box_param();

  int64_t h, w;
  llvm::ArrayRef<int64_t> input_shape =
      input_vars[0]->getType().dyn_cast<mlir::TensorType>().getShape();

  assert(prior_box_param.max_size_size() == 1 && prior_box_param.min_size_size() == 1 &&
         prior_box_param.aspect_ratio_size() <= 2);

  h = input_shape[2];
  w = input_shape[3];

  std::vector<Value *> operands;
  operands.push_back(input_vars[0]);
  operands.push_back(input_vars[1]);

  std::vector<NamedAttribute> attrs;

  attrs.push_back(builder_.getNamedAttr(
      "min_size", builder_.getF32FloatAttr(prior_box_param.min_size(0))));
  attrs.push_back(builder_.getNamedAttr(
      "min_size_size", builder_.getI32IntegerAttr(prior_box_param.min_size_size())));
  attrs.push_back(builder_.getNamedAttr(
      "max_size", builder_.getF32FloatAttr(prior_box_param.max_size(0))));
  attrs.push_back(builder_.getNamedAttr(
      "max_size_size", builder_.getI32IntegerAttr(prior_box_param.max_size_size())));
  attrs.push_back(builder_.getNamedAttr(
      "aspect_ratio0", builder_.getF32FloatAttr(prior_box_param.aspect_ratio(0))));

  if (prior_box_param.aspect_ratio_size() == 2) {
    attrs.push_back(builder_.getNamedAttr(
        "aspect_ratio1", builder_.getF32FloatAttr(prior_box_param.aspect_ratio(1))));
  }

  attrs.push_back(builder_.getNamedAttr(
      "aspect_ratios_size",
      builder_.getI32IntegerAttr(prior_box_param.aspect_ratio_size())));
  attrs.push_back(
      builder_.getNamedAttr("flip", builder_.getBoolAttr(prior_box_param.flip())));
  attrs.push_back(
      builder_.getNamedAttr("clip", builder_.getBoolAttr(prior_box_param.clip())));
  attrs.push_back(builder_.getNamedAttr(
      "variance0", builder_.getF32FloatAttr(prior_box_param.variance(0))));
  attrs.push_back(builder_.getNamedAttr(
      "variance1", builder_.getF32FloatAttr(prior_box_param.variance(1))));
  attrs.push_back(builder_.getNamedAttr(
      "variance2", builder_.getF32FloatAttr(prior_box_param.variance(2))));
  attrs.push_back(builder_.getNamedAttr(
      "variance3", builder_.getF32FloatAttr(prior_box_param.variance(3))));
  attrs.push_back(
      builder_.getNamedAttr("step", builder_.getF32FloatAttr(prior_box_param.step())));
  attrs.push_back(
      builder_.getNamedAttr("name", builder_.getStringAttr(layer_param.name())));
  attrs.push_back(builder_.getNamedAttr(
      "offset", builder_.getF32FloatAttr(prior_box_param.offset())));

  std::vector<float> min_sizes_;
  std::vector<float> max_sizes_;
  std::vector<float> aspect_ratios_;
  bool flip_ = true;
  int num_priors_ = 0;
  // bool clip_=false;
  std::vector<float> variance_;

  for (int i = 0; i < prior_box_param.min_size_size(); ++i) {
    min_sizes_.push_back(prior_box_param.min_size(i));
    assert(min_sizes_.back() > 0 && "min_size must be positive.");
  }
  aspect_ratios_.clear();
  aspect_ratios_.push_back(1.);
  flip_ = prior_box_param.flip();
  for (int i = 0; i < prior_box_param.aspect_ratio_size(); ++i) {
    float ar = prior_box_param.aspect_ratio(i);
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
        aspect_ratios_.push_back(1. / ar);
      }
    }
  }

  num_priors_ = aspect_ratios_.size() * min_sizes_.size();
  if (prior_box_param.max_size_size() > 0) {
    CHECK_EQ(prior_box_param.min_size_size(), prior_box_param.max_size_size());
    for (int i = 0; i < prior_box_param.max_size_size(); ++i) {
      max_sizes_.push_back(prior_box_param.max_size(i));
      assert(max_sizes_[i] > min_sizes_[i] &&
             ("max_size must be greater than min_size."));
      num_priors_ += 1;
    }
  }

  // construct OP
  auto result_type =
      RankedTensorType::get({1, 2, (h * w * num_priors_ * 4)}, elementType_);

  auto reshape_op = OpBuilder(block).create<tpu::PriorBoxOp>(
      builder_.getUnknownLoc(), result_type, ArrayRef<Value *>{operands},
      ArrayRef<NamedAttribute>{attrs});

  auto result_var = reshape_op.getResult();
  tensor_map_[layer_param.top(0)] = result_var;
}

void CaffeImporter::convertReLULayer(mlir::Block *block, caffe::Layer<float> *layer) {
  mlir::Value *input_var = GetLayerInput(layer);

  auto layer_param = layer->layer_param();
  auto relu_param = layer_param.relu_param();
  float negative_slope = 0.0f;
  if (relu_param.has_negative_slope()) {
    negative_slope = relu_param.negative_slope();
  }

  int64_t n, c, h, w;
  llvm::ArrayRef<int64_t> input_shape =
      input_var->getType().dyn_cast<mlir::TensorType>().getShape();
  RankedTensorType result_type = nullptr;

  if (input_shape.size() == 4) {
    n = input_shape[0];
    c = input_shape[1];
    h = input_shape[2];
    w = input_shape[3];
    result_type = RankedTensorType::get({n, c, h, w}, elementType_);
  } else if (input_shape.size() == 2) {
    n = input_shape[0];
    c = input_shape[1];
    h = 1;
    w = 1;
    result_type = RankedTensorType::get({n, c}, elementType_);
  } else {
    assert(input_shape.size() == 4 || input_shape.size() == 2);
  }
  LLVM_DEBUG(llvm::errs() << "  N: " << n << ", C: " << c << ", IH*IW: " << h << " * "
                          << w << "\n";);

  // construct OP
  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      builder_.getNamedAttr("name", builder_.getStringAttr(layer_param.name())));
  attrs.push_back(builder_.getNamedAttr("quant", getDefaultQuantParam(builder_)));

  Value *result_var = nullptr;
  if (negative_slope == 0.0f) {
    auto op = OpBuilder(block).create<tpu::ReluOp>(builder_.getUnknownLoc(), result_type,
                                                   ArrayRef<Value *>{input_var},
                                                   ArrayRef<NamedAttribute>{attrs});
    result_var = op.getResult();
  } else {
    std::vector<Value *> operands;
    operands.push_back(input_var);
    auto NoneOp = OpBuilder(block).create<tpu::NoneOp>(builder_.getUnknownLoc(),
                                                       builder_.getNoneType());
    for (int i = 0; i < 8; i++) {
      operands.push_back(
          NoneOp.getResult()); // quant: scale/zp/rshift/muliplier, pos and neg
    }
    assert(negative_slope > 0.0f && negative_slope < 1.0f);
    attrs.push_back(builder_.getNamedAttr("negative_slope",
                                          builder_.getF32FloatAttr(negative_slope)));
    auto op = OpBuilder(block).create<tpu::LeakyReluOp>(
        builder_.getUnknownLoc(), result_type, ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    result_var = op.getResult();
  }

  tensor_map_[layer_param.top(0)] = result_var;
}

void CaffeImporter::convertReshapeLayer(mlir::Block *block, caffe::Layer<float> *layer) {
  mlir::Value *input_var = GetLayerInput(layer);

  auto layer_param = layer->layer_param();

  llvm::ArrayRef<int64_t> input_shape =
      input_var->getType().dyn_cast<mlir::TensorType>().getShape();
  RankedTensorType result_type = nullptr;

  if (input_shape.size() == 4) {

    const int input_start_axis = layer_param.reshape_param().axis();
    const int num_axes = layer_param.reshape_param().num_axes();
    const int start_axis = (input_start_axis >= 0)
                               ? input_start_axis
                               : input_shape.size() + input_start_axis + 1;

    assert(start_axis >= 0);
    assert(start_axis <= (int)input_shape.size());
    assert(num_axes >= -1);
    const int end_axis = (num_axes == -1) ? input_shape.size() : (start_axis + num_axes);
    assert(end_axis <= (int)input_shape.size());

    const int num_axes_replaced = end_axis - start_axis;
    const int num_axes_retained = input_shape.size() - num_axes_replaced;
    auto top_blob_shape = layer_param.reshape_param().shape();
    const int num_new_axes = top_blob_shape.dim_size();

    std::vector<int> copy_axes;
    int inferred_axis = -1;
    int constant_count = 1;
    for (int i = 0; i < num_new_axes; ++i) {
      const int top_dim = top_blob_shape.dim(i);
      if (top_dim == 0) {
        copy_axes.push_back(i);
      } else if (top_dim == -1) {
        assert(inferred_axis == -1);
        inferred_axis = i;
      } else {
        constant_count *= top_dim;
      }
    }

    std::vector<int64_t> top_shape(num_axes_retained + num_new_axes);
    int top_shape_index = 0;
    for (int i = 0; i < start_axis; ++i) {
      top_shape[top_shape_index++] = input_shape[i];
    }
    for (int i = 0; i < num_new_axes; ++i) {
      top_shape[top_shape_index++] = top_blob_shape.dim(i);
    }
    for (unsigned i = end_axis; i < input_shape.size(); ++i) {
      top_shape[top_shape_index++] = input_shape[i];
    }
    assert(top_shape_index == (int)top_shape.size());
    for (unsigned i = 0; i < copy_axes.size(); ++i) {
      const int copy_axis_index = copy_axes[i];
      assert((int)input_shape.size() > start_axis + copy_axis_index);
      top_shape[start_axis + copy_axis_index] = input_shape[start_axis + copy_axis_index];
    }

    if (inferred_axis >= 0) {
      // A -1 dim was specified; infer the correct dimension by computing the
      // product of the other dimensions.
      int explicit_count = constant_count;
      for (int i = 0; i < start_axis; i++) {
        explicit_count *= input_shape[i];
      }
      for (unsigned i = end_axis; i < input_shape.size(); i++) {
        explicit_count *= input_shape[i];
      }
      for (unsigned i = 0; i < copy_axes.size(); ++i) {
        const int copy_axis_index = copy_axes[i];
        explicit_count *= top_shape[start_axis + copy_axis_index];
      }
      int64_t input_count =
          input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];
      assert(0 == input_count % explicit_count);
      const int inferred_dim = input_count / explicit_count;
      top_shape[start_axis + inferred_axis] = inferred_dim;
    }

    // construct OP
    result_type = RankedTensorType::get(ArrayRef<int64_t>{top_shape}, elementType_);

  } else if (input_shape.size() == 2) {

    assert((layer_param.reshape_param().shape().dim_size() == 3) &&
           "only support input shape size is 2 && output shape size is 3 case ");
    auto size = std::accumulate(std::begin(input_shape), std::end(input_shape), 1,
                                std::multiplies<>());
    std::vector<int64_t> output_shape(3);
    int inference_dim = 0;
    for (int i = 0; i < layer_param.reshape_param().shape().dim_size(); i++) {
      int dim_value = layer_param.reshape_param().shape().dim(i);
      if (dim_value == 0) {
        output_shape[i] = input_shape[i];
        size /= output_shape[i];
      } else if (dim_value == -1) {
        inference_dim = i;

      } else {
        output_shape[i] = layer_param.reshape_param().shape().dim(i);
        size /= layer_param.reshape_param().shape().dim(i);
      }
    }
    output_shape[inference_dim] = size;

    result_type = RankedTensorType::get(
        {output_shape[0], output_shape[1], output_shape[2]}, elementType_);
    LLVM_DEBUG(llvm::errs() << "  C: " << output_shape[0] << ", H: " << output_shape[1]
                            << ", W: " << output_shape[2] << "\n";);

  } else {
    assert(input_shape.size() == 4 || input_shape.size() == 2);
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      builder_.getNamedAttr("name", builder_.getStringAttr(layer_param.name())));
  // attrs.push_back(builder_.getNamedAttr("quant",
  // getDefaultQuantParam(builder_)));
  auto reshape_op = OpBuilder(block).create<tpu::ReshapeOp>(
      builder_.getUnknownLoc(), result_type, ArrayRef<Value *>{input_var},
      ArrayRef<NamedAttribute>{attrs});
  auto result_var = reshape_op.getResult();
  tensor_map_[layer_param.top(0)] = result_var;
}

void CaffeImporter::convertRetinaFaceDetectionLayer(mlir::Block *block,
                                                    caffe::Layer<float> *layer) {
  auto layer_param = layer->layer_param();
  auto retinaface_detection_param = layer_param.retinaface_detection_param();
  float nms_threshold = retinaface_detection_param.nms_threshold();
  float confidence_threshold = retinaface_detection_param.confidence_threshold();
  int keep_topk = retinaface_detection_param.keep_topk();

  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      builder_.getNamedAttr("name", builder_.getStringAttr(layer_param.name())));
  attrs.push_back(
      builder_.getNamedAttr("nms_threshold", builder_.getF32FloatAttr(nms_threshold)));
  attrs.push_back(builder_.getNamedAttr("confidence_threshold",
                                        builder_.getF32FloatAttr(confidence_threshold)));
  attrs.push_back(
      builder_.getNamedAttr("keep_topk", builder_.getI32IntegerAttr(keep_topk)));
  std::vector<mlir::Value *> input_vars = GetLayerInputs(layer);

  auto result_type = RankedTensorType::get({1, 1, keep_topk, 15}, elementType_);

  auto retina_op = OpBuilder(block).create<tpu::RetinaFaceDetectionOp>(
      builder_.getUnknownLoc(), result_type, ArrayRef<Value *>{input_vars},
      ArrayRef<NamedAttribute>{attrs});
  auto result_var = retina_op.getResult();
  tensor_map_[layer_param.top(0)] = result_var;
}

void CaffeImporter::convertScaleLayer(mlir::Block *block, caffe::Layer<float> *layer) {
  std::vector<mlir::Value *> input_vars = GetLayerInputs(layer);

  auto layer_param = layer->layer_param();
  assert(layer_param.has_scale_param());
  auto scale_param = layer_param.scale_param();
  bool with_bias = scale_param.bias_term();
  int64_t n, c, h, w;

  auto input_var = input_vars[0];
  llvm::ArrayRef<int64_t> input_var_shape =
      input_var->getType().dyn_cast<mlir::TensorType>().getShape();

  assert(input_var_shape.size() == 4 || input_var_shape.size() == 2);

  n = input_var_shape[0];
  c = input_var_shape[1];
  LLVM_DEBUG(llvm::errs() << "  N: " << n << ", C: " << c;);
  if (input_var_shape.size() == 4) {
    h = input_var_shape[2];
    w = input_var_shape[3];
    LLVM_DEBUG(llvm::errs() << ", IH*IW: " << h << " * " << w;);
  }
  LLVM_DEBUG(llvm::errs() << "\n";);

  std::vector<Value *> operands;
  operands.push_back(input_var);
  auto NoneOp = OpBuilder(block).create<tpu::NoneOp>(builder_.getUnknownLoc(),
                                                     builder_.getNoneType());

  Value *result_var;
  if (input_vars.size() == 2) {
    // two bottom input
    operands.push_back(input_vars[1]);
    operands.push_back(NoneOp.getResult()); // quant_scale
    operands.push_back(NoneOp.getResult()); // quant_zeropoint
    operands.push_back(NoneOp.getResult()); // quant_rshift
    operands.push_back(NoneOp.getResult()); // quant_multiplier

    auto result_type = RankedTensorType::get(input_var_shape, elementType_);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        builder_.getNamedAttr("name", builder_.getStringAttr(layer_param.name())));
    attrs.push_back(builder_.getNamedAttr("quant", getDefaultQuantParam(builder_)));
    auto op = OpBuilder(block).create<tpu::BroadcastMulOp>(
        builder_.getUnknownLoc(), result_type, ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    result_var = op.getResult();
  } else {
    // - blobs_[0] holds the scale
    // - blobs_[1] holds the biases (optional)
    auto scale_name = layer->layer_param().name() + "_0";
    auto scale_type = RankedTensorType::get({c}, elementType_);
    GetWeightFile()->addTensor(scale_name, layer->blobs()[0].get()->cpu_data(),
                               scale_type);
    operands.push_back(AddLoadWeightOp(block, scale_name, scale_type));
    if (with_bias) {
      auto bias_name = layer->layer_param().name() + "_1";
      auto bias_type = RankedTensorType::get({c}, elementType_);
      GetWeightFile()->addTensor(bias_name, layer->blobs()[1].get()->cpu_data(),
                                 bias_type);
      operands.push_back(AddLoadWeightOp(block, bias_name, bias_type));
    } else {
      operands.push_back(NoneOp.getResult());
    }

    auto result_type = RankedTensorType::get(input_var_shape, elementType_);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        builder_.getNamedAttr("name", builder_.getStringAttr(layer_param.name())));
    auto op = OpBuilder(block).create<tpu::ScaleOp>(builder_.getUnknownLoc(), result_type,
                                                    ArrayRef<Value *>{operands},
                                                    ArrayRef<NamedAttribute>{attrs});
    result_var = op.getResult();
  }

  tensor_map_[layer_param.top(0)] = result_var;
}

void CaffeImporter::convertShuffleChannelLayer(mlir::Block *block,
                                               caffe::Layer<float> *layer) {
  auto layer_param = layer->layer_param();
  assert(layer_param.has_shuffle_channel_param());
  auto p = layer_param.shuffle_channel_param();
  auto group = p.group();
  mlir::Value *input_var = GetLayerInput(layer);
  llvm::ArrayRef<int64_t> input_shape =
      input_var->getType().dyn_cast<mlir::TensorType>().getShape();
  assert(input_shape.size() >= 2);

  std::vector<Value *> operands;
  operands.push_back(input_var);

  auto result_type = RankedTensorType::get(input_shape, elementType_);

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder_.getNamedAttr("group", builder_.getI32IntegerAttr(group)));
  attrs.push_back(
      builder_.getNamedAttr("name", builder_.getStringAttr(layer_param.name())));
  attrs.push_back(builder_.getNamedAttr("quant", getDefaultQuantParam(builder_)));

  auto op = OpBuilder(block).create<tpu::ShuffleChannelOp>(
      builder_.getUnknownLoc(), result_type, ArrayRef<Value *>{operands},
      ArrayRef<NamedAttribute>{attrs});
  auto result_var = op.getResult();

  tensor_map_[layer_param.top(0)] = result_var;
}

void CaffeImporter::convertSigmoidLayer(mlir::Block *block, caffe::Layer<float> *layer) {
  mlir::Value *input_var = GetLayerInput(layer);

  auto layer_param = layer->layer_param();

  int64_t n, c, h, w;
  llvm::ArrayRef<int64_t> input_shape =
      input_var->getType().dyn_cast<mlir::TensorType>().getShape();
  assert(input_shape.size() == 4);
  n = input_shape[0];
  c = input_shape[1];
  h = input_shape[2];
  w = input_shape[3];

  LLVM_DEBUG(llvm::errs() << "  N: " << n << ", C: " << c << ", IH*IW: " << h << " * "
                          << w << "\n";);
  std::vector<Value *> operands;
  operands.push_back(input_var);
  // construct OP
  auto result_type = RankedTensorType::get({n, c, h, w}, elementType_);

  auto NoneOp = OpBuilder(block).create<tpu::NoneOp>(builder_.getUnknownLoc(),
                                                     builder_.getNoneType());
  operands.push_back(NoneOp.getResult()); // quant_table
  operands.push_back(NoneOp.getResult()); // quant_table

  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      builder_.getNamedAttr("name", builder_.getStringAttr(layer_param.name())));
  attrs.push_back(builder_.getNamedAttr("has_table", builder_.getBoolAttr(false)));
  attrs.push_back(builder_.getNamedAttr("quant", getDefaultQuantParam(builder_)));
  auto op = OpBuilder(block).create<tpu::SigmoidOp>(builder_.getUnknownLoc(), result_type,
                                                    ArrayRef<Value *>{operands},
                                                    ArrayRef<NamedAttribute>{attrs});
  auto result_var = op.getResult();

  tensor_map_[layer_param.top(0)] = result_var;
}

void CaffeImporter::convertSliceLayer(mlir::Block *block, caffe::Layer<float> *layer) {
  mlir::Value *input_var = GetLayerInput(layer);

  auto layer_param = layer->layer_param();
  auto slice_param = layer_param.slice_param();
  int axis = slice_param.axis();
  int top_size = layer_param.top_size();

  llvm::ArrayRef<int64_t> input_shape =
      input_var->getType().dyn_cast<mlir::TensorType>().getShape();
  assert(input_shape.size() == 4);

  const int bottom_slice_axis = input_shape[axis];
  std::vector<int> slices;
  if (slice_param.slice_point_size() != 0) {
    assert(slice_param.slice_point_size() == top_size - 1);
    assert(top_size < bottom_slice_axis);
    uint32_t prev = 0;
    for (int i = 0; i < slice_param.slice_point_size(); ++i) {
      assert(slice_param.slice_point(i) > prev);
      slices.push_back(slice_param.slice_point(i) - prev);
      prev = slice_param.slice_point(i);
    }
    slices.push_back(bottom_slice_axis - prev);
  } else {
    assert(bottom_slice_axis % top_size == 0);
    for (int i = 0; i < top_size; i++) {
      slices.push_back(bottom_slice_axis / top_size);
    }
  }

  // construct OP
  int offset = 0;
  for (int i = 0; i < top_size; i++) {
    int64_t n = 0, c = 0, h = 0, w = 0;
    switch (axis) {
      case 1:
        n = input_shape[0];
        c = slices[i];
        h = input_shape[2];
        w = input_shape[3];
        break;
      default:
        llvm::errs() << "Only support channel slice for now."
                     << "\n";
        assert(false);
    }

    auto result_type = RankedTensorType::get({n, c, h, w}, elementType_);

    std::vector<NamedAttribute> attrs;
    attrs.push_back(builder_.getNamedAttr("axis", builder_.getI32IntegerAttr(axis)));
    attrs.push_back(builder_.getNamedAttr("offset", builder_.getI32IntegerAttr(offset)));
    attrs.push_back(builder_.getNamedAttr(
        "name", builder_.getStringAttr(layer_param.name() + "_" + std::to_string(i))));
    attrs.push_back(builder_.getNamedAttr("quant", getDefaultQuantParam(builder_)));
    auto op = OpBuilder(block).create<tpu::SliceOp>(builder_.getUnknownLoc(), result_type,
                                                    ArrayRef<Value *>{input_var},
                                                    ArrayRef<NamedAttribute>{attrs});
    auto result_var = op.getResult();

    tensor_map_[layer_param.top(i)] = result_var;
    offset += c;
  }
}

void CaffeImporter::convertSoftmaxLayer(mlir::Block *block, caffe::Layer<float> *layer) {
  mlir::Value *input_var = GetLayerInput(layer);

  auto layer_param = layer->layer_param();
  int axis = 1;
  if (layer_param.has_softmax_param()) {
    axis = layer_param.softmax_param().axis();
  }

  llvm::ArrayRef<int64_t> input_shape =
      input_var->getType().dyn_cast<mlir::TensorType>().getShape();

  for (size_t i = 0; i < input_shape.size(); ++i) {
    LLVM_DEBUG(llvm::errs() << "input_shape[" << i << "] = " << input_shape[i] << ", ");
  }
  LLVM_DEBUG(llvm::errs() << "\n");

  // construct OP
  auto result_type = input_var->getType();
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder_.getNamedAttr("axis", builder_.getI32IntegerAttr(axis)));
  attrs.push_back(
      builder_.getNamedAttr("name", builder_.getStringAttr(layer_param.name())));
  auto op = OpBuilder(block).create<tpu::SoftmaxOp>(builder_.getUnknownLoc(), result_type,
                                                    ArrayRef<Value *>{input_var},
                                                    ArrayRef<NamedAttribute>{attrs});

  auto result_var = op.getResult();
  tensor_map_[layer_param.top(0)] = result_var;
}

void CaffeImporter::convertSplitLayer(mlir::Block *block, caffe::Layer<float> *layer) {
  mlir::Value *input_var = GetLayerInput(layer);
  // simply bypass, register top and bottom blobs to the same tensor
  auto layer_param = layer->layer_param();
  int top_size = layer_param.top_size();

  for (int i = 0; i < top_size; i++)
    tensor_map_[layer_param.top(i)] = input_var;
}

void CaffeImporter::convertTanHLayer(mlir::Block *block, caffe::Layer<float> *layer) {
  mlir::Value *input_var = GetLayerInput(layer);

  auto layer_param = layer->layer_param();

  int64_t n, c, h, w;
  llvm::ArrayRef<int64_t> input_shape =
      input_var->getType().dyn_cast<mlir::TensorType>().getShape();
  assert(input_shape.size() == 4);
  n = input_shape[0];
  c = input_shape[1];
  h = input_shape[2];
  w = input_shape[3];

  LLVM_DEBUG(llvm::errs() << "  N: " << n << ", C: " << c << ", IH*IW: " << h << " * "
                          << w << "\n";);

  std::vector<Value *> operands;
  operands.push_back(input_var);

#if 0
  // add y0 / slope table
  // FIXME: not hard code
  int channel = 32;
  int table_h = 32;
  int table_w = 8;
  int table_hw = table_h * table_w;

  // 32 for hard code, # of channel
  // table shape hw is 8,32 - hw define
  auto table_type = RankedTensorType::get({1, channel, table_h, table_w}, elementType_);

  int tbl_size = channel * table_hw;
  std::vector<float> dataVec_fp32;
  dataVec_fp32.reserve(tbl_size);

  // reserve dummy weight and assign in opt
  auto filter_name = layer->layer_param().name()+"_y0";
  GetWeightFile()->addTensor(filter_name, &dataVec_fp32, table_type);
  operands.push_back(AddLoadWeightOp(block, filter_name, table_type));

  filter_name = layer->layer_param().name()+"_slope";
  GetWeightFile()->addTensor(filter_name, &dataVec_fp32, table_type);
  operands.push_back(AddLoadWeightOp(block, filter_name, table_type));
#endif

  // construct OP
  auto result_type = RankedTensorType::get({n, c, h, w}, elementType_);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      builder_.getNamedAttr("name", builder_.getStringAttr(layer_param.name())));
  attrs.push_back(builder_.getNamedAttr("quant", getDefaultQuantParam(builder_)));

  auto op = OpBuilder(block).create<tpu::TanHOp>(builder_.getUnknownLoc(), result_type,
                                                 ArrayRef<Value *>{operands},
                                                 ArrayRef<NamedAttribute>{attrs});
  auto result_var = op.getResult();

  tensor_map_[layer_param.top(0)] = result_var;
}

void CaffeImporter::convertUpsampleLayer(mlir::Block *block, caffe::Layer<float> *layer) {
  std::vector<mlir::Value *> input_vars = GetLayerInputs(layer);

  auto layer_param = layer->layer_param();
  auto upsample_param = layer_param.upsample_param();
  unsigned scale = upsample_param.scale();
  assert(scale == 2);

  int64_t n, c, ih, iw, oh, ow;
  llvm::ArrayRef<int64_t> input_shape =
      input_vars[0]->getType().dyn_cast<mlir::TensorType>().getShape();
  assert(input_shape.size() == 4);
  n = input_shape[0];
  c = input_shape[1];
  ih = input_shape[2];
  iw = input_shape[3];
  oh = ih * scale;
  ow = iw * scale;

  LLVM_DEBUG(llvm::errs() << "  N: " << n << ", C: " << c << ", IH*IW: " << ih << " * "
                          << iw << ", OH*OW: " << oh << " * " << ow << "\n";);

  // construct OP
  auto result_type = RankedTensorType::get({n, c, oh, ow}, elementType_);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder_.getNamedAttr("scale", builder_.getI32IntegerAttr(scale)));
  attrs.push_back(
      builder_.getNamedAttr("name", builder_.getStringAttr(layer_param.name())));
  attrs.push_back(builder_.getNamedAttr("quant", getDefaultQuantParam(builder_)));
  auto op = OpBuilder(block).create<tpu::UpsampleOp>(
      builder_.getUnknownLoc(), result_type, ArrayRef<Value *>{input_vars},
      ArrayRef<NamedAttribute>{attrs});
  auto result_var = op.getResult();

  tensor_map_[layer_param.top(0)] = result_var;
}

void CaffeImporter::convertYoloDetectionLayer(mlir::Block *block,
                                                    caffe::Layer<float> *layer) {
  auto layer_param = layer->layer_param();
  auto yolo_detection_param = layer_param.yolo_detection_param();
  int net_input_h = yolo_detection_param.net_input_h();
  int net_input_w = yolo_detection_param.net_input_w();
  float nms_threshold = yolo_detection_param.nms_threshold();
  float obj_threshold = yolo_detection_param.obj_threshold();
  int keep_topk = yolo_detection_param.keep_topk();

  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      builder_.getNamedAttr("name", builder_.getStringAttr(layer_param.name())));
  attrs.push_back(
      builder_.getNamedAttr("net_input_h", builder_.getI32IntegerAttr(net_input_h)));
  attrs.push_back(
      builder_.getNamedAttr("net_input_w", builder_.getI32IntegerAttr(net_input_w)));
  attrs.push_back(
      builder_.getNamedAttr("nms_threshold", builder_.getF32FloatAttr(nms_threshold)));
  attrs.push_back(
      builder_.getNamedAttr("obj_threshold", builder_.getF32FloatAttr(obj_threshold)));
  attrs.push_back(
      builder_.getNamedAttr("keep_topk", builder_.getI32IntegerAttr(keep_topk)));
  std::vector<mlir::Value *> input_vars = GetLayerInputs(layer);

  auto result_type = RankedTensorType::get({1, 1, keep_topk, 6}, elementType_);

  auto yolo_op = OpBuilder(block).create<tpu::YoloDetectionOp>(
      builder_.getUnknownLoc(), result_type, ArrayRef<Value *>{input_vars},
      ArrayRef<NamedAttribute>{attrs});
  auto result_var = yolo_op.getResult();
  tensor_map_[layer_param.top(0)] = result_var;
}

mlir::Value *CaffeImporter::insertTransposeLayer(mlir::Block *block, mlir::Value *opd,
                                                 const std::string &name) {
  std::vector<Value *> operands;
  operands.push_back(opd);

  llvm::ArrayRef<int64_t> opd_shape =
      opd->getType().dyn_cast<mlir::TensorType>().getShape();
  assert(opd_shape.size() == 4);
  // transpose shape from hwc to chw
  std::vector<int64_t> result_shape{opd_shape[0], opd_shape[3], opd_shape[1],
                                    opd_shape[2]};
  auto result_type =
      RankedTensorType::get(llvm::ArrayRef<int64_t>(result_shape), elementType_);

  std::vector<NamedAttribute> attrs;
  std::string my_name = name + "_trans";
  attrs.push_back(builder_.getNamedAttr("name", builder_.getStringAttr(my_name)));
  attrs.push_back(builder_.getNamedAttr("quant", getDefaultQuantParam(builder_)));
  auto op = OpBuilder(block).create<tpu::TransposeOp>(
      builder_.getUnknownLoc(), result_type, ArrayRef<Value *>{operands},
      ArrayRef<NamedAttribute>{attrs});
  auto result_var = op.getResult();

  tensor_map_[my_name] = result_var;
  return result_var;
}

void CaffeImporter::insertPreprocessLayer(mlir::Block *block, mlir::Value *opd,
                                          const std::string &name) {
  std::vector<Value *> operands;
  operands.push_back(opd);

  llvm::ArrayRef<int64_t> input_var_shape =
      opd->getType().dyn_cast<mlir::TensorType>().getShape();
  auto result_type = RankedTensorType::get(input_var_shape, elementType_);

  std::vector<float> mean;
  std::vector<int32_t> color_order;
  if (!clPreprocessMean.empty()) {
    SmallVector<StringRef, 3> sub_strs;
    StringRef(clPreprocessMean).split(sub_strs, ",");
    for (auto &str : sub_strs) {
      double val;
      assert(!str.getAsDouble(val));
      mean.push_back((float)val);
    }
  }
  if (!clPreprocessColorOrder.empty()) {
    SmallVector<StringRef, 3> sub_strs;
    StringRef(clPreprocessColorOrder).split(sub_strs, ",");
    for (auto &str : sub_strs) {
      int32_t val;
      assert(!str.getAsInteger(10, val));
      color_order.push_back(val);
    }
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder_.getNamedAttr("name", builder_.getStringAttr(name)));
  attrs.push_back(builder_.getNamedAttr("quant", getDefaultQuantParam(builder_)));
  attrs.push_back(
      builder_.getNamedAttr("scale", builder_.getF32FloatAttr(clPreprocessScale)));
  attrs.push_back(
      builder_.getNamedAttr("raw_scale", builder_.getF32FloatAttr(clPreprocessRawScale)));
  if (mean.size())
    attrs.push_back(
        builder_.getNamedAttr("mean", builder_.getF32ArrayAttr(ArrayRef<float>({mean}))));
  if (color_order.size())
    attrs.push_back(builder_.getNamedAttr(
        "color_order", builder_.getI32ArrayAttr(ArrayRef<int32_t>({color_order}))));
  auto op = OpBuilder(block).create<tpu::PreprocessOp>(
      builder_.getUnknownLoc(), result_type, ArrayRef<Value *>{operands},
      ArrayRef<NamedAttribute>{attrs});
  auto result_var = op.getResult();

  tensor_map_[name] = result_var;
}

LogicalResult CaffeImporter::Import(const llvm::StringRef inputFilename,
                                    llvm::StringRef caffemodelFilename) {
  google::SetCommandLineOption("GLOG_minloglevel", "2");

  caffe::Net<float> net(inputFilename, caffe::TEST);
  net.CopyTrainedLayersFrom(caffemodelFilename);
  DEBUG_WITH_TYPE(DEBUG_TYPE "_VERBOSE", printCaffeNetAllLayer(net););

  auto weightFileName =
      TensorFile::generateName(llvm::sys::path::stem(caffemodelFilename), 0);

  elementType_ = mlir::FloatType::getF32(builder_.getContext());
  std::map<std::string, mlir::Type> net_inputs;
  std::map<std::string, mlir::Type> net_outputs;
  ParseNetInputOutput(net, net_inputs, net_outputs);

  mlir::Block *block = CreateOneBlockFunction(net_inputs, net_outputs);
  AddWeightFileOp(block, weightFileName);
  ConvertLayers(block, net);
  AddReturnOp(block, net_outputs);

  return success();
}

// add preprocess ops
void CaffeImporter::doPreSwapChannel(mlir::Block *block, std::string name, const std::vector<int> &order) {
  if (order.empty()) {
    return;
  }
  llvm::ArrayRef<int64_t> input_shape =
      lastResult_->getType().dyn_cast<mlir::TensorType>().getShape();
  std::vector<Value *> operands;
  operands.push_back(lastResult_);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder_.getNamedAttr("name", builder_.getStringAttr(name)));
  attrs.push_back(builder_.getNamedAttr("quant", getDefaultQuantParam(builder_)));
  attrs.push_back(builder_.getNamedAttr("channel_order", builder_.getI32ArrayAttr(ArrayRef<int32_t>({order}))));
  auto result_type = RankedTensorType::get(input_shape, elementType_);
  auto op = OpBuilder(block).create<tpu::SwapChannelOp>(
      builder_.getUnknownLoc(), result_type, ArrayRef<Value *>{operands},
      ArrayRef<NamedAttribute>{attrs});
  lastResult_ = op.getResult();
}

void CaffeImporter::doPreScale(mlir::Block *block, std::string name, float scale) {
  if (scale == 1.0f) {
    return;
  }
  llvm::ArrayRef<int64_t> input_shape =
      lastResult_->getType().dyn_cast<mlir::TensorType>().getShape();
  std::vector<Value *> operands;
  operands.push_back(lastResult_);
  int64_t channel = input_shape[1];
  auto scale_type = RankedTensorType::get({channel}, elementType_);
  std::vector<float> scale_v(channel, scale);
  GetWeightFile()->addTensor(name + "_0", scale_v.data(), scale_type);
  operands.push_back(AddLoadWeightOp(block, name + "_0", scale_type));
  std::vector<float> bias(channel, 0.0f);
  auto bias_type = RankedTensorType::get({channel}, elementType_);
  GetWeightFile()->addTensor(name + "_1", bias.data(), bias_type);
  operands.push_back(AddLoadWeightOp(block, name + "_1", bias_type));
  auto result_type = RankedTensorType::get(input_shape, elementType_);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder_.getNamedAttr("name", builder_.getStringAttr(name)));
  attrs.push_back(builder_.getNamedAttr("quant", getDefaultQuantParam(builder_)));
  auto op = OpBuilder(block).create<tpu::ScaleOp>(builder_.getUnknownLoc(), result_type,
                                                  ArrayRef<Value *>{operands},
                                                  ArrayRef<NamedAttribute>{attrs});
  lastResult_ = op.getResult();
}

void CaffeImporter::doPreMean(mlir::Block *block, std::string name, const std::vector<int> &order) {
  if (clMean.empty()) {
    return;
  }
  llvm::ArrayRef<int64_t> input_shape =
      lastResult_->getType().dyn_cast<mlir::TensorType>().getShape();
  assert(input_shape.size() == 4);
  int64_t c = input_shape[1];
  assert(c == 3);

  // input
  std::vector<Value *> operands;
  operands.push_back(lastResult_);

  // mean
  float mean_value[3];
  SmallVector<StringRef, 3> sub_strs;
  StringRef(clMean).split(sub_strs, ",");
  int index = 0;
  for (auto &str : sub_strs) {
      double val;
      assert(!str.getAsDouble(val));
      mean_value[index++] = (float)val;
  }
  if (order.empty() == false) {
    assert((int)order.size() == c);
    float val[3];
    memcpy(val, mean_value, 3 * sizeof(float));
    for (int i = 0; i < c; i ++) {
      mean_value[i] = val[order[i]];
    }
  }
  auto data_type = RankedTensorType::get({c}, elementType_);
  GetWeightFile()->addTensor(name + "_0", mean_value, data_type);
  operands.push_back(AddLoadWeightOp(block, name + "_0", data_type));

  // variance
  float variance_value[3] = {1.0f, 1.0f, 1.0f};
  GetWeightFile()->addTensor(name + "_1", variance_value, data_type);
  operands.push_back(AddLoadWeightOp(block, name + "_1", data_type));

  // scale
  float scale_value[1] = {1.0f};
  auto scale_type = RankedTensorType::get({1}, elementType_);
  GetWeightFile()->addTensor(name + "_2", scale_value, scale_type);
  operands.push_back(AddLoadWeightOp(block, name + "_2", scale_type));

  auto result_type = RankedTensorType::get(input_shape, elementType_);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder_.getNamedAttr("name", builder_.getStringAttr(name)));
  attrs.push_back(
      builder_.getNamedAttr("variance_epsilon", builder_.getF32FloatAttr(0.0f)));
  auto op = OpBuilder(block).create<tpu::BatchNormOp>(
      builder_.getUnknownLoc(), result_type, ArrayRef<Value *>{operands},
      ArrayRef<NamedAttribute>{attrs});
  lastResult_ = op.getResult();
}

void CaffeImporter::doPreprocess(mlir::Block *block, mlir::Value *input) {
  lastResult_ = input;
  std::vector<int32_t> order;
  if (!clSwapChannel.empty()) {
    SmallVector<StringRef, 3> sub_strs;
    StringRef(clSwapChannel).split(sub_strs, ",");
    for (auto &str : sub_strs) {
      int32_t val;
      assert(!str.getAsInteger(10, val));
      order.push_back(val);
    }
    int index = 0;
    int order_size = (int)order.size();
    for (; index < order_size; index++) {
      assert(order[index] <order_size);
      if ( index != order[index]) {
        break;
      }
    }
    // same order , needn't swap
    if (index ==order_size) {
      order.clear();
    }
  }
  doPreScale(block, "pre_raw_scale", clRawScale.getValue());
  doPreMean(block, "pre_mean", order);
  doPreScale(block, "pre_scale", clScale.getValue());
  doPreSwapChannel(block, "pre_swap_channel", order);
  auto inputOp = llvm::cast<tpu::InputOp>(input->getDefiningOp());
  tensor_map_[inputOp.name()] = lastResult_;
}

// ====================translate=======================================

// Translate CaffeModel and returns a module in TPU Dialect.
static OwningModuleRef caffeToMlirTranslate(llvm::SourceMgr &sourceMgr,
                                            llvm::StringRef caffemodelFilename,
                                            MLIRContext *context) {
  mlir::OwningModuleRef module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context));

  // we didn't parse caffe prototxt by ourselves, we pass it to caffe
  // however caffe take filename as input, therefore we save the source
  // to a tmp file, the file will be automatically deleted.
  std::string tmpFile = "./tmp.txt";
  std::string errorMessage;
  auto tmp = openOutputFile(tmpFile, &errorMessage);
  if (!tmp) {
    llvm::errs() << errorMessage << "\n";
    return nullptr;
  }
  const llvm::MemoryBuffer *buffer = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
  tmp->os().write(buffer->getBufferStart(), buffer->getBufferSize());
  tmp->os().flush();

  CaffeImporter importer(module.get());
  auto status = importer.Import(tmpFile, caffemodelFilename);
  if (failed(status)) {
    mlir::emitError(mlir::UnknownLoc::get(context));
  }
  assert(succeeded(status));
  return module;
}

static TranslateToMLIRRegistration
    registration("caffe-to-mlir", [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
      return caffeToMlirTranslate(sourceMgr, clCaffeModelFilename, context);
    });
