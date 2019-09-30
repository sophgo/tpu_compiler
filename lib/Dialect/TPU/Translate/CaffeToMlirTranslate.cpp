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

// Importer that takes an Caffe model and imports it as an MLIR module in the TPU
// dialect.
class CaffeImporter {
public:
  explicit CaffeImporter(mlir::ModuleOp module)
      : module_(module),
        builder_(module.getContext()) {}

  // Import the Caffe model file into the MLIR Module.
  LogicalResult Import(const llvm::StringRef inputFilename,
      llvm::StringRef caffemodelFilename);

private:
  mlir::Type GetTypeFromCaffeShape(
      const std::vector<int> shape, mlir::Type elementType);

  void ParseNetInputOutput(caffe::Net<float> &net,
      std::map<std::string, mlir::Type> &inputs,
      std::map<std::string, mlir::Type> &outputs);

  mlir::Block* CreateOneBlockFunction(
      std::map<std::string, mlir::Type> &inputs,
      std::map<std::string, mlir::Type> &outputs);

  void AddLoadFileOp(mlir::Block *block,
      const llvm::StringRef weightFilename);

  void ConvertLayers(mlir::Block *block, caffe::Net<float> &net);

  void AddReturnOp(mlir::Block *block,
      std::map<std::string, mlir::Type> &outputs);

  mlir::Value* AddLoadWeightOp(mlir::Block *block,
      std::string name, TensorType &type);

  mlir::Value* GetLayerInput(caffe::Layer<float> *layer);
  std::vector<mlir::Value *> GetLayerInputs(caffe::Layer<float> *layer);

  void convertInputLayer(mlir::Block *block, caffe::Layer<float> *layer);
  void convertSplitLayer(mlir::Block *block, caffe::Layer<float> *layer);
  void convertConvolutionLayer(mlir::Block *block, caffe::Layer<float> *layer);
  void convertInnerProductLayer(mlir::Block *block, caffe::Layer<float> *layer);
  void convertPoolingLayer(mlir::Block *block, caffe::Layer<float> *layer);
  void convertBatchNormLayer(mlir::Block *block, caffe::Layer<float> *layer);
  void convertScaleLayer(mlir::Block *block, caffe::Layer<float> *layer);
  void convertReLULayer(mlir::Block *block, caffe::Layer<float> *layer);
  void convertEltwiseLayer(mlir::Block *block, caffe::Layer<float> *layer);
  void convertSoftmaxLayer(mlir::Block *block, caffe::Layer<float> *layer);

  mlir::ModuleOp module_;
  mlir::Builder builder_;
  std::unique_ptr<TensorFile> weightFile_;
  mlir::Type elementType_;
  std::map<std::string, mlir::Value *> tensor_map_;
  mlir::Value *weightFileVar_;
};

static void printCaffeLayerParam(const caffe::Layer<float>* layer) {
  auto layer_param = layer->layer_param();
  llvm::errs() << llvm::format(">> [%-12s] %s\n", layer->type(), layer_param.name().c_str());
  for (int i = 0; i <= layer_param.bottom_size() - 1; ++i) {
    llvm::errs() << "btm: " << layer_param.bottom(i) << "\n";
  }
  for (int i = 0; i <= layer_param.top_size() - 1; ++i) {
    llvm::errs() << "top: " << layer_param.top(i) << "\n";
  }
}

static void printCaffeNetAllLayer(const caffe::Net<float>& net) {
  for (size_t i = 0; i <= net.layers().size() - 1; ++i) {
    auto layer = net.layers()[i].get();
    printCaffeLayerParam(layer);
  }
}

#define calcConv2DSpatialOutput(_i_, _k_, _s_, _p_, _d_) \
    (((_i_) + 2 * (_p_) - (_d_) * ((_k_) - 1) - 1) / (_s_) + 1)

mlir::Type CaffeImporter::GetTypeFromCaffeShape(
    const std::vector<int> shape, mlir::Type elementType) {
  std::vector<int64_t> shape_int64(shape.begin(), shape.end());
  llvm::ArrayRef<int64_t> mlir_shape(shape_int64);
  return builder_.getTensorType(mlir_shape, elementType);
}

void CaffeImporter::ParseNetInputOutput(caffe::Net<float> &net,
    std::map<std::string, mlir::Type> &inputs,
    std::map<std::string, mlir::Type> &outputs) {
  for (int i = 0; i <= net.num_inputs() - 1; ++i) {
    int index = net.input_blob_indices()[i];
    LLVM_DEBUG(
      llvm::errs()
          << "net input [" << i << "] - [" << index << "] : "
          << ", blob: " << net.blob_names()[index]
          << ", shape: " << net.input_blobs()[i]->shape_string()
          << ", layer: " << net.layer_names()[index]
          << "\n";
    );
    inputs[net.blob_names()[index]] = GetTypeFromCaffeShape(
        net.input_blobs()[i]->shape(), elementType_);
  }
  for (int i = 0; i <= net.num_outputs() - 1; ++i) {
    int index = net.output_blob_indices()[i];
    LLVM_DEBUG(
      llvm::errs()
          << "net output[" << i << "] - [" << index << "] : "
          << ", blob: " << net.blob_names()[index]
          << ", shape: " << net.output_blobs()[i]->shape_string()
          << ", layer: " << net.layer_names()[index]
          << "\n";
    );
    outputs[net.blob_names()[index]] = GetTypeFromCaffeShape(
        net.output_blobs()[i]->shape(), elementType_);
  }
}

mlir::Block* CaffeImporter::CreateOneBlockFunction(
    std::map<std::string, mlir::Type> &inputs,
    std::map<std::string, mlir::Type> &outputs) {
  std::vector<mlir::Type> arguments;
  for(auto e : inputs)
    arguments.push_back(e.second);
  std::vector<mlir::Type> returns;
  for(auto e : outputs)
    returns.push_back(e.second);
  auto fnType = builder_.getFunctionType(llvm::ArrayRef<mlir::Type>{arguments},
      llvm::ArrayRef<mlir::Type>{returns});
  auto fn = FuncOp::create(builder_.getUnknownLoc(), "tpu_func", fnType);
  module_.push_back(fn);
  auto *block = fn.addEntryBlock();
  return block;
}

void CaffeImporter::AddLoadFileOp(mlir::Block *block,
    const llvm::StringRef weightFilename) {
  auto weight_type = builder_.getMemRefType({0x80000000}, elementType_);
  auto weight_attr = builder_.getStringAttr(weightFilename);
  weightFileVar_ = OpBuilder(block).create<tpu::LoadFileOp>(
      builder_.getUnknownLoc(), weight_type, weight_attr);
}

void CaffeImporter::ConvertLayers(mlir::Block *block,
    caffe::Net<float> &net) {
  for (size_t i = 0; i <= net.layers().size() - 1; ++i) {
    auto layer = net.layers()[i].get();
    LLVM_DEBUG(printCaffeLayerParam(layer););

    if (strcmp(layer->type(), "Input") == 0) {
      convertInputLayer(block, layer);
    } else if (strcmp(layer->type(), "Split") == 0) {
      convertSplitLayer(block, layer);
    } else if (strcmp(layer->type(), "Convolution") == 0) {
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
    } else if (strcmp(layer->type(), "Eltwise") == 0) {
      convertEltwiseLayer(block, layer);
    } else if (strcmp(layer->type(), "Softmax") == 0) {
      convertSoftmaxLayer(block, layer);
    } else {
      llvm::errs() << "    UNKNOWN" << "\n";
      assert(false);
    }
  }
}

void CaffeImporter::AddReturnOp(mlir::Block *block,
    std::map<std::string, mlir::Type> &outputs) {
  std::vector<mlir::Value *> returns;
  for(auto e : outputs) {
    auto it = tensor_map_.find(e.first);
    assert(it != tensor_map_.end());
	  mlir::Value *var = it->second;
    returns.push_back(var);
  }
  OpBuilder(block).create<ReturnOp>(builder_.getUnknownLoc(),
      llvm::ArrayRef<mlir::Value *>{returns});
}

mlir::Value* CaffeImporter::AddLoadWeightOp(mlir::Block *block,
    std::string name, TensorType &type) {
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder_.getNamedAttr("name", builder_.getStringAttr(name)));
  return OpBuilder(block).create<tpu::LoadWeightOp>(builder_.getUnknownLoc(),
      type, ArrayRef<Value *>{weightFileVar_}, ArrayRef<NamedAttribute>{attrs});
}

mlir::Value* CaffeImporter::GetLayerInput(caffe::Layer<float> *layer) {
  auto layer_param = layer->layer_param();
  assert(layer_param.bottom_size() == 1);
  auto it = tensor_map_.find(layer_param.bottom(0));
  assert(it != tensor_map_.end());
  mlir::Value *input = it->second;
  DEBUG_WITH_TYPE(DEBUG_TYPE"_VERBOSE", input->getType().dump(););
  return input;
}

std::vector<mlir::Value *> CaffeImporter::GetLayerInputs(caffe::Layer<float> *layer) {
  auto layer_param = layer->layer_param();
  std::vector<mlir::Value *> inputs;
  for (int i = 0; i < layer_param.bottom_size(); ++i) {
    auto it = tensor_map_.find(layer_param.bottom(i));
    assert(it != tensor_map_.end());
    inputs.push_back(it->second);
    DEBUG_WITH_TYPE(DEBUG_TYPE"_VERBOSE", it->second->getType().dump(););
  }
  return inputs;
}

void CaffeImporter::convertInputLayer(mlir::Block *block,
    caffe::Layer<float> *layer) {
  auto layer_param = layer->layer_param();
  assert(layer_param.bottom_size() == 0 && layer_param.top_size() == 1);

  tensor_map_[layer_param.top(0)] = block->getArgument(0);
}

void CaffeImporter::convertSplitLayer(mlir::Block *block,
    caffe::Layer<float> *layer) {
  mlir::Value *input_var = GetLayerInput(layer);
  // simply bypass, register top and bottom blobs to the same tensor
  auto layer_param = layer->layer_param();
  tensor_map_[layer_param.top(0)] = input_var;
  tensor_map_[layer_param.top(1)] = input_var;
}

void CaffeImporter::convertConvolutionLayer(mlir::Block *block,
    caffe::Layer<float> *layer) {
  mlir::Value *input_var = GetLayerInput(layer);

  auto layer_param = layer->layer_param();
  assert(layer_param.has_convolution_param());
  auto p = layer_param.convolution_param();
  int64_t n, ic, oc, group;
  std::vector<int64_t> kernel(2), stride(2), padding(2), dilation(2);
  std::vector<int64_t> ifmap(2), ofmap(2); // spatial dims only (height and width)

  bool with_bias = p.bias_term();
  oc = p.num_output();
  group  = p.has_group()? p.group() : 1;
  kernel[0] = p.has_kernel_h() ? p.kernel_h() : p.kernel_size_size() > 1 ? p.kernel_size(1) : p.kernel_size(0);
  kernel[1] = p.has_kernel_w() ? p.kernel_w() : p.kernel_size(0);
  stride[0] = p.has_stride_h() ? p.stride_h() : p.stride_size() > 1 ? p.stride(1) : p.stride_size() > 0 ? p.stride(0) : 1;
  stride[1] = p.has_stride_w() ? p.stride_w() : p.stride_size() > 0 ? p.stride(0) : 1;
  padding[0]  = p.has_pad_h() ? p.pad_h() : p.pad_size() > 1 ? p.pad(1) : p.pad_size() > 0 ? p.pad(0) : 0;
  padding[1]  = p.has_pad_w() ? p.pad_w() : p.pad_size() > 0 ? p.pad(0) : 0;
  dilation[0] = p.dilation_size() > 1 ? p.dilation(1) : p.dilation_size() > 0 ? p.dilation(0) : 1;
  dilation[1] = p.dilation_size() > 0 ? p.dilation(0) : 1;

  // TODO: don't support group for now
  assert(group == 1);
  assert( (dilation[0] == 1) && (dilation[1] == 1) );

  // get input shape from input var
  llvm::ArrayRef<int64_t> input_shape =
      input_var->getType().dyn_cast<mlir::TensorType>().getShape();
  assert(input_shape.size() == 4);
  n = input_shape[0];
  ic = input_shape[1];
  ifmap[0] = input_shape[2];
  ifmap[1] = input_shape[3];
  // get output shape from inference
  ofmap[0] = calcConv2DSpatialOutput(ifmap[0], kernel[0], stride[0], padding[0], dilation[0]);
  ofmap[1] = calcConv2DSpatialOutput(ifmap[1], kernel[1], stride[1], padding[1], dilation[1]);

  LLVM_DEBUG(
    llvm::errs()
        << "  N: " << n
        << ", IC: " << ic
        << ", IH*IW: " << ifmap[0] << " * " << ifmap[1]
        << ", OC: " << oc
        << ", OH*OW: " << ofmap[0] << " * " << ofmap[1]
        << "\n";
    llvm::errs()
        << "  with_bias: " << with_bias
        << ", K: " << kernel[0]   << " * " << kernel[1]
        << ", S: " << stride[0]   << " * " << stride[1]
        << ", P: " << padding[0]  << " * " << padding[1]
        << ", D: " << dilation[0] << " * " << dilation[1]
        << ", group: " << group
        << "\n";
  );

  std::vector<Value *> operands;
  operands.push_back(input_var);

  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  auto filter_name = layer->layer_param().name()+"_0";
  auto filter_type = builder_.getTensorType({oc, ic, kernel[0], kernel[1]}, elementType_);
  weightFile_->addTensor(filter_name, layer->blobs()[0].get()->cpu_data(), filter_type);
  operands.push_back(AddLoadWeightOp(block, filter_name, filter_type));
  if (with_bias) {
    auto bias_name = layer->layer_param().name()+"_1";
    auto bias_type = builder_.getTensorType({oc}, elementType_);
    weightFile_->addTensor(bias_name, layer->blobs()[1].get()->cpu_data(), bias_type);
    operands.push_back(AddLoadWeightOp(block, bias_name, bias_type));
  }

  // construct OP
  auto result_type = builder_.getTensorType({n, oc, ofmap[0], ofmap[1]}, elementType_);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder_.getNamedAttr("dilation_h_factor", builder_.getI32IntegerAttr(dilation[0])));
  attrs.push_back(builder_.getNamedAttr("dilation_w_factor", builder_.getI32IntegerAttr(dilation[1])));
  //attrs.push_back(builder.getNamedAttr("fused_activation_function", builder.getStringAttr("NONE")));
  attrs.push_back(builder_.getNamedAttr("padding", (padding[0] || padding[1])
                  ? builder_.getStringAttr("SAME") : builder_.getStringAttr("VALID")));
  attrs.push_back(builder_.getNamedAttr("stride_h", builder_.getI32IntegerAttr(stride[0])));
  attrs.push_back(builder_.getNamedAttr("stride_w", builder_.getI32IntegerAttr(stride[1])));
  attrs.push_back(builder_.getNamedAttr("name", builder_.getStringAttr(layer_param.name())));
  auto op = OpBuilder(block).create<tpu::Conv2DOp>(
      builder_.getUnknownLoc(), result_type,
      ArrayRef<Value *>{operands}, ArrayRef<NamedAttribute>{attrs});
  auto result_var = op.getResult();

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

  LLVM_DEBUG(
    llvm::errs()
        << "  M: " << M
        << ", K: " << K
        << ", N: " << N
        << ", with_bias: " << with_bias
        << ", with_transpose: " << with_transpose
        << "\n";
  );

  mlir::Value *fc_input_var = input_var;
  // construct reshape OP
  if (reshape_first) {
    auto fc_input_type = builder_.getTensorType({M, K}, elementType_);
    auto reshape_op = OpBuilder(block).create<tpu::ReshapeOp>(
        builder_.getUnknownLoc(), fc_input_type, input_var);
    fc_input_var = reshape_op.getResult();
  }

  std::vector<Value *> operands;
  operands.push_back(fc_input_var);

  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  auto filter_name = layer->layer_param().name()+"_0";
  auto filter_type = builder_.getTensorType({N, K}, elementType_);
  weightFile_->addTensor(filter_name, layer->blobs()[0].get()->cpu_data(), filter_type);
  operands.push_back(AddLoadWeightOp(block, filter_name, filter_type));
  if (with_bias) {
    auto bias_name = layer->layer_param().name()+"_1";
    auto bias_type = builder_.getTensorType({N}, elementType_);
    weightFile_->addTensor(bias_name, layer->blobs()[1].get()->cpu_data(), bias_type);
    operands.push_back(AddLoadWeightOp(block, bias_name, bias_type));
  }

  // construct OP
  auto result_type = builder_.getTensorType({M, N}, elementType_);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder_.getNamedAttr("name", builder_.getStringAttr(layer_param.name())));
  auto op = OpBuilder(block).create<tpu::FullyConnectedOp>(
      builder_.getUnknownLoc(), result_type,
      ArrayRef<Value *>{operands}, ArrayRef<NamedAttribute>{attrs});
  auto result_var = op.getResult();

  tensor_map_[layer_param.top(0)] = result_var;
}

void CaffeImporter::convertPoolingLayer(mlir::Block *block,
    caffe::Layer<float> *layer) {
  mlir::Value *input_var = GetLayerInput(layer);

  auto layer_param = layer->layer_param();
  assert(layer_param.has_pooling_param());
  auto p = layer_param.pooling_param();

  bool is_average_pooling;
  bool is_global_pooling;
  int64_t n, c;
  std::vector<int64_t> kernel(2), stride(2), padding(2);
  std::vector<int64_t> ifmap(2), ofmap(2);  // spatial dims only (height and width)

  if (p.pool() == caffe::PoolingParameter_PoolMethod_AVE) {
    is_average_pooling = true;
  } else if (p.pool() == caffe::PoolingParameter_PoolMethod_MAX) {
    is_average_pooling = false;
  } else {
    assert(false && "Invalid pool type");
  }
  is_global_pooling = p.global_pooling();

  // get input shape from input var
  llvm::ArrayRef<int64_t> input_shape
      = input_var->getType().dyn_cast<mlir::TensorType>().getShape();
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
  stride[0]  = p.has_stride_h() ? p.stride_h() : p.has_stride() ? p.stride() : 1;
  stride[1]  = p.has_stride_w() ? p.stride_w() : p.has_stride() ? p.stride() : 1;
  padding[0] = p.has_pad_h() ? p.pad_h() : p.has_pad() ? p.pad() : 0;
  padding[1] = p.has_pad_w() ? p.pad_w() : p.has_pad() ? p.pad() : 0;
  //
  // Fix caffe pooling padding
  //
  //  pooled_height_ = static_cast<int>(ceil(static_cast<float>(
  //      height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
  //  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
  //      width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
  //
  ofmap[0] = (static_cast<int>(ceil(static_cast<float>(
        ifmap[0] + 2 * padding[0] - kernel[0]) / stride[0])) + 1);
  ofmap[1] = (static_cast<int>(ceil(static_cast<float>(
        ifmap[1] + 2 * padding[1] - kernel[1]) / stride[1])) + 1);

  if (is_global_pooling) {
    assert( (padding[0] == 0) && (padding[1] == 0) );
    assert( (stride[0] == 1) && (stride[1] == 1) );
    assert( (ofmap[0] == 1) && (ofmap[1] == 1) );
  }

  LLVM_DEBUG(
    llvm::errs()
        << "  N: " << n
        << ", C: " << c
        << ", IH*IW: " << ifmap[0] << " * " << ifmap[1]
        << ", OH*OW: " << ofmap[0] << " * " << ofmap[1]
        << ", type: " << (is_average_pooling ? "AVG" : "MAX")
        << "\n";
    llvm::errs()
        << "  K: " << kernel[0] << " * " << kernel[1]
        << ", S: " << stride[0] << " * " << stride[1]
        << ", P: " << padding[0] << " * " << padding[1]
        << ", global_pooling: " << is_global_pooling
        << "\n";
  );

  // construct OP
  auto result_type = builder_.getTensorType({n, c, ofmap[0], ofmap[1]}, elementType_);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder_.getNamedAttr("filter_height", builder_.getI32IntegerAttr(kernel[0])));
  attrs.push_back(builder_.getNamedAttr("filter_width", builder_.getI32IntegerAttr(kernel[1])));
  attrs.push_back(builder_.getNamedAttr("padding",
      (padding[0] || padding[1]) ? builder_.getStringAttr("SAME") : builder_.getStringAttr("VALID")));
  attrs.push_back(builder_.getNamedAttr("stride_h", builder_.getI32IntegerAttr(stride[0])));
  attrs.push_back(builder_.getNamedAttr("stride_w", builder_.getI32IntegerAttr(stride[1])));
  attrs.push_back(builder_.getNamedAttr("fused_activation_function", builder_.getStringAttr("NONE")));
  attrs.push_back(builder_.getNamedAttr("name", builder_.getStringAttr(layer_param.name())));
  mlir::Value* result_var;
  if (is_average_pooling) {
    auto op = OpBuilder(block).create<tpu::AveragePool2DOp>(
        builder_.getUnknownLoc(), result_type, ArrayRef<Value *>{input_var},
        ArrayRef<NamedAttribute>{attrs});
    result_var = op.getResult();
  } else {
    auto op = OpBuilder(block).create<tpu::MaxPool2DOp>(
        builder_.getUnknownLoc(), result_type, ArrayRef<Value *>{input_var},
        ArrayRef<NamedAttribute>{attrs});
    result_var = op.getResult();
  }

  tensor_map_[layer_param.top(0)] = result_var;
}

void CaffeImporter::convertBatchNormLayer(mlir::Block *block,
    caffe::Layer<float> *layer) {
  mlir::Value *input_var = GetLayerInput(layer);

  auto layer_param = layer->layer_param();
  assert(layer_param.has_batch_norm_param());
  auto batch_norm_param = layer_param.batch_norm_param();
  //float epsilon = batch_norm_param.eps();

  int64_t n, c, h, w;
  llvm::ArrayRef<int64_t> input_var_shape =
      input_var->getType().dyn_cast<mlir::TensorType>().getShape();
  assert(input_var_shape.size() == 4);
  n = input_var_shape[0];
  c = input_var_shape[1];
  h = input_var_shape[2];
  w = input_var_shape[3];

  LLVM_DEBUG(
    llvm::errs()
        << "  N: " << n
        << ", C: " << c
        << ", IH*IW: " << h << " * " << w
        << "\n";
  );

  std::vector<Value *> operands;
  operands.push_back(input_var);

  // - blobs_[2] holds the scale, which is one scalar data
  // - blobs_[0] holds the mean
  // - blobs_[1] holds the variance
  assert(layer->blobs().size() == 3);

  auto mean_name = layer->layer_param().name()+"_0";
  auto mean_type = builder_.getTensorType({c}, elementType_);
  weightFile_->addTensor(mean_name, layer->blobs()[0].get()->cpu_data(), mean_type);
  operands.push_back(AddLoadWeightOp(block, mean_name, mean_type));

  auto variance_name = layer->layer_param().name()+"_1";
  auto variance_type = builder_.getTensorType({c}, elementType_);
  weightFile_->addTensor(variance_name, layer->blobs()[1].get()->cpu_data(), variance_type);
  operands.push_back(AddLoadWeightOp(block, variance_name, variance_type));

  auto scale_name = layer->layer_param().name()+"_2";
  auto scale_type = builder_.getTensorType({1}, elementType_);
  weightFile_->addTensor(scale_name, layer->blobs()[2].get()->cpu_data(), scale_type);
  operands.push_back(AddLoadWeightOp(block, scale_name, scale_type));

  // construct OP
  auto result_type = builder_.getTensorType({n, c, h, w}, elementType_);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder_.getNamedAttr("name", builder_.getStringAttr(layer_param.name())));
  auto op = OpBuilder(block).create<tpu::BatchNormOp>(
      builder_.getUnknownLoc(), result_type,
      ArrayRef<Value *>{operands}, ArrayRef<NamedAttribute>{attrs});
  auto result_var = op.getResult();

  tensor_map_[layer_param.top(0)] = result_var;
}

void CaffeImporter::convertScaleLayer(mlir::Block *block,
    caffe::Layer<float> *layer) {
  mlir::Value *input_var = GetLayerInput(layer);

  auto layer_param = layer->layer_param();
  assert(layer_param.has_scale_param());
  auto scale_param = layer_param.scale_param();
  bool with_bias = scale_param.bias_term();

  int64_t n, c, h, w;
  llvm::ArrayRef<int64_t> input_var_shape =
      input_var->getType().dyn_cast<mlir::TensorType>().getShape();
  assert(input_var_shape.size() == 4);
  n = input_var_shape[0];
  c = input_var_shape[1];
  h = input_var_shape[2];
  w = input_var_shape[3];

  LLVM_DEBUG(
    llvm::errs()
        << "  N: " << n
        << ", C: " << c
        << ", IH*IW: " << h << " * " << w
        << "\n";
  );

  std::vector<Value *> operands;
  operands.push_back(input_var);

  // - blobs_[0] holds the scale
  // - blobs_[1] holds the biases (optional)
  auto scale_name = layer->layer_param().name()+"_0";
  auto scale_type = builder_.getTensorType({c}, elementType_);
  weightFile_->addTensor(scale_name, layer->blobs()[0].get()->cpu_data(), scale_type);
  operands.push_back(AddLoadWeightOp(block, scale_name, scale_type));
  if (with_bias) {
    auto bias_name = layer->layer_param().name()+"_1";
    auto bias_type = builder_.getTensorType({c}, elementType_);
    weightFile_->addTensor(bias_name, layer->blobs()[1].get()->cpu_data(), bias_type);
    operands.push_back(AddLoadWeightOp(block, bias_name, bias_type));
  }

  // construct OP
  auto result_type = builder_.getTensorType({n, c, h, w}, elementType_);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder_.getNamedAttr("name", builder_.getStringAttr(layer_param.name())));
  auto op = OpBuilder(block).create<tpu::ScaleOp>(
      builder_.getUnknownLoc(), result_type,
      ArrayRef<Value *>{operands}, ArrayRef<NamedAttribute>{attrs});
  auto result_var = op.getResult();

  tensor_map_[layer_param.top(0)] = result_var;
}

void CaffeImporter::convertReLULayer(mlir::Block *block,
    caffe::Layer<float> *layer) {
  mlir::Value *input_var = GetLayerInput(layer);

  auto layer_param = layer->layer_param();
  auto relu_param = layer_param.relu_param();
  float negative_slope = relu_param.negative_slope();

  int64_t n, c, h, w;
  llvm::ArrayRef<int64_t> input_shape = input_var->getType().dyn_cast<mlir::TensorType>().getShape();
  assert(input_shape.size() == 4);
  n = input_shape[0];
  c = input_shape[1];
  h = input_shape[2];
  w = input_shape[3];

  LLVM_DEBUG(
    llvm::errs()
        << "  N: " << n
        << ", C: " << c
        << ", IH*IW: " << h << " * " << w
        << "\n";
  );

  // construct OP
  auto result_type = builder_.getTensorType({n, c, h, w}, elementType_);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder_.getNamedAttr("negative_slope", builder_.getF32FloatAttr(negative_slope)));
  attrs.push_back(builder_.getNamedAttr("name", builder_.getStringAttr(layer_param.name())));
  auto op = OpBuilder(block).create<tpu::ReluOp>(
      builder_.getUnknownLoc(), result_type,
      ArrayRef<Value *>{input_var}, ArrayRef<NamedAttribute>{attrs});
  auto result_var = op.getResult();

  tensor_map_[layer_param.top(0)] = result_var;
}

void CaffeImporter::convertEltwiseLayer(mlir::Block *block,
    caffe::Layer<float> *layer) {
  std::vector<mlir::Value *> input_vars = GetLayerInputs(layer);

  auto layer_param = layer->layer_param();
  auto eltwise_param = layer_param.eltwise_param();
  assert(eltwise_param.coeff_size() == 0);
  assert(eltwise_param.operation() == caffe::EltwiseParameter_EltwiseOp_SUM);

  int64_t n, c, h, w;
  llvm::ArrayRef<int64_t> input_shape =
      input_vars[0]->getType().dyn_cast<mlir::TensorType>().getShape();
  assert(input_shape.size() == 4);
  n = input_shape[0];
  c = input_shape[1];
  h = input_shape[2];
  w = input_shape[3];

  LLVM_DEBUG(
    llvm::errs()
        << "  N: " << n
        << ", C: " << c
        << ", IH*IW: " << h << " * " << w
        << "\n";
  );

  // construct OP
  auto result_type = builder_.getTensorType({n, c, h, w}, elementType_);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder_.getNamedAttr("name", builder_.getStringAttr(layer_param.name())));
  auto op = OpBuilder(block).create<tpu::EltwiseOp>(
      builder_.getUnknownLoc(), result_type,
      ArrayRef<Value *>{input_vars}, ArrayRef<NamedAttribute>{attrs});
  auto result_var = op.getResult();

  tensor_map_[layer_param.top(0)] = result_var;
}

void CaffeImporter::convertSoftmaxLayer(mlir::Block *block,
    caffe::Layer<float> *layer) {
  mlir::Value *input_var = GetLayerInput(layer);

  auto layer_param = layer->layer_param();
  int64_t n, c;
  llvm::ArrayRef<int64_t> input_var_shape =
      input_var->getType().dyn_cast<mlir::TensorType>().getShape();
  assert(input_var_shape.size() == 2);
  n = input_var_shape[0];
  c = input_var_shape[1];

  LLVM_DEBUG(
    llvm::errs()
        << "  N: " << n
        << ", C: " << c
        << "\n";
  );

  // construct OP
  auto result_type = builder_.getTensorType({n, c}, elementType_);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder_.getNamedAttr("name", builder_.getStringAttr(layer_param.name())));
  auto op = OpBuilder(block).create<tpu::SoftmaxOp>(
      builder_.getUnknownLoc(), result_type,
      ArrayRef<Value *>{input_var}, ArrayRef<NamedAttribute>{attrs});
  auto result_var = op.getResult();

  tensor_map_[layer_param.top(0)] = result_var;
}

LogicalResult CaffeImporter::Import(const llvm::StringRef inputFilename,
    llvm::StringRef caffemodelFilename) {
  caffe::Net<float> net(inputFilename, caffe::TEST);
  net.CopyTrainedLayersFrom(caffemodelFilename);
  DEBUG_WITH_TYPE(DEBUG_TYPE"_VERBOSE", printCaffeNetAllLayer(net););

  auto weightFilename = llvm::sys::path::stem(caffemodelFilename).str() + ".npz";
  weightFile_ = openOutputTensorFile(weightFilename);

  elementType_ = mlir::FloatType::getF32(builder_.getContext());
  std::map<std::string, mlir::Type> net_inputs;
  std::map<std::string, mlir::Type> net_outputs;
  ParseNetInputOutput(net, net_inputs, net_outputs);

  mlir::Block *block = CreateOneBlockFunction(net_inputs, net_outputs);

  AddLoadFileOp(block, weightFilename);
  ConvertLayers(block, net);
  AddReturnOp(block, net_outputs);

  return success();
}

// Translate CaffeModel in the file named as `inputFilename` and returns a
// module in TPU Dialect.
static OwningModuleRef caffeToMlirTranslate(llvm::StringRef inputFilename,
    llvm::StringRef caffemodelFilename, MLIRContext *context) {
  mlir::OwningModuleRef module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(context));
  CaffeImporter importer(module.get());
  auto status = importer.Import(inputFilename, caffemodelFilename);
  if (failed(status)) {
    mlir::emitError(mlir::UnknownLoc::get(context));
  }
  assert(succeeded(status));
  return module;
}

static llvm::cl::OptionCategory clOptionsCategory("caffe translate options");

static llvm::cl::opt<std::string> clCaffeModelFilename(
    "caffe-model",
    llvm::cl::desc("Specify the caffemodel filename"),
    llvm::cl::cat(clOptionsCategory));

static TranslateToMLIRRegistration
    registration("caffe-to-mlir",
                 [](StringRef inputFilename, MLIRContext *context) {
                   return caffeToMlirTranslate(inputFilename,
                       clCaffeModelFilename, context);
                 });
