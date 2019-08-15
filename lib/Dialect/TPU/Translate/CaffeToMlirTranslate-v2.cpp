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
#include "mlir/StandardOps/Ops.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Translation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"

#include "caffe/caffe.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/util/signal_handler.h"

#include <cstring>
#include <map>
#include <string>
#include <vector>

#define DEBUG_TYPE "caffe-to-mlir-v2"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//
static mlir::Type getMlirTypeFromCaffeShape(Builder builder,
    const std::vector<int> shape, mlir::Type elementType) {
  std::vector<int64_t> shape_int64(shape.begin(), shape.end());
  llvm::ArrayRef<int64_t> mlir_shape(shape_int64);
  auto mlir_type = builder.getTensorType(mlir_shape, elementType);
  return mlir_type;
}

static void printCaffeLayerParam(const caffe::Layer<float>* layer) {
  auto layer_param = layer->layer_param();
  std::cout << std::left << std::setfill(' ')
      << ">> [" << std::setw(12) << layer->type() << std::setw(0) << "] "
      << layer_param.name() << "\n";
  for (int i = 0; i <= layer_param.bottom_size() - 1; ++i) {
    std::cout << "btm: " << layer_param.bottom(i) << "\n";
  }
  for (int i = 0; i <= layer_param.top_size() - 1; ++i) {
    std::cout << "top: " << layer_param.top(i) << "\n";
  }
}

static void printCaffeNetAllLayer(const caffe::Net<float>& net) {
  for (size_t i = 0; i <= net.layers().size() - 1; ++i) {
    auto layer = net.layers()[i].get();
    //auto layer_param = layer->layer_param();
    //std::cout << std::left << std::setfill(' ')
    //    << ">> [" << std::setw(12) << layer->type() << std::setw(0) << "] "
    //    << layer_param.name() << "\n";
    printCaffeLayerParam(layer);
  }
}

//===----------------------------------------------------------------------===//
// Create Operation
//===----------------------------------------------------------------------===//
static mlir::Value *addConv2dOpInBlockFromCaffe(Builder builder, Block *block,
    mlir::Type elementType, mlir::Value *input_var,
    const caffe::ConvolutionParameter& conv_param) {
  //auto conv_param = layer_param.convolution_param();
  std::vector<int64_t> k, s, d, p;
  int64_t oc = conv_param.num_output();
  const int num_spatial_axes = 2;
  if (conv_param.has_kernel_h() && conv_param.has_kernel_w()) {
    assert(conv_param.kernel_size_size() == 0);
    k.push_back(conv_param.kernel_h());
    k.push_back(conv_param.kernel_w());
  } else {
    const int num_kernel_dims = conv_param.kernel_size_size();
    for (int i = 0; i < num_spatial_axes; ++i) {
      k.push_back(conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i));
    }
  }
  if (conv_param.has_stride_h() && conv_param.has_stride_w()) {
    assert(conv_param.stride_size() == 0);
    s.push_back(conv_param.stride_h());
    s.push_back(conv_param.stride_w());
  } else {
    const int num_stride_dims = conv_param.stride_size();
    for (int i = 0; i < num_spatial_axes; ++i) {
      s.push_back(conv_param.stride((num_stride_dims == 1) ? 0 : i));
    }
  }
  if (conv_param.has_pad_h() && conv_param.has_pad_w()) {
    assert(conv_param.pad_size() == 0);
    p.push_back(conv_param.pad_h());
    p.push_back(conv_param.pad_w());
  } else {
    const int num_pad_dims = conv_param.pad_size();
    const int kDefaultPad = 0;
    for (int i = 0; i < num_spatial_axes; ++i) {
      p.push_back((num_pad_dims == 0) ? kDefaultPad :
          conv_param.pad((num_pad_dims == 1) ? 0 : i));
    }
  }
  const int num_dilation_dims = conv_param.dilation_size();
  const int kDefaultDilation = 1;
  for (int i = 0; i < num_spatial_axes; ++i) {
    d.push_back((num_dilation_dims == 0) ? kDefaultDilation :
        conv_param.dilation((num_dilation_dims == 1) ? 0 : i));
  }

  std::cout << "bias: " << conv_param.bias_term()
      << ", OC: " << oc
      << ", K: " << k[0] << " * " << k[1]
      << ", S: " << s[0] << " * " << s[1]
      << ", P: " << p[0] << " * " << p[1]
      << ", D: " << d[0] << " * " << d[1]
      << "\n";

  // current input shape
  input_var->getType().dump();
  std::cout << "\n";
  int64_t n, ic;
  std::vector<int64_t> ifmap;
  llvm::ArrayRef<int64_t> ifmap_shape =
      input_var->getType().dyn_cast<mlir::TensorType>().getShape();
  assert(ifmap_shape.size() == 4);
  n = ifmap_shape[0];
  ic = ifmap_shape[1];
  ifmap.push_back(ifmap_shape[2]);
  ifmap.push_back(ifmap_shape[3]);

  std::cout << "N: " << n
      << ", IC: " << ic
      << ", IH*IW: " << ifmap[0] << " * " << ifmap[1]
      << "\n";

  // inferred output shape
  std::vector<int64_t> ofmap;
  // does not support dilation for now
  assert(d[0] == 1 && d[1] == 1);
  ofmap.push_back((ifmap[0] - k[0] + 2 * p[0]) / s[0] + 1);
  ofmap.push_back((ifmap[1] - k[1] + 2 * p[1]) / s[1] + 1);

  std::cout
      << "OH*OW: " << ofmap[0] << " * " << ofmap[1]
      << "\n";

  // construct OP
  auto filter_type = builder.getTensorType({oc, ic, k[0], k[1]}, elementType);
  auto filter_attr = builder.getZeroAttr(filter_type);
  auto filter = OpBuilder(block).create<ConstantOp>(builder.getUnknownLoc(),
      filter_type, filter_attr);
  auto bias_type = builder.getTensorType({oc}, elementType);
  auto bias_attr = builder.getZeroAttr(bias_type);
  auto bias = OpBuilder(block).create<ConstantOp>(builder.getUnknownLoc(),
      bias_type, bias_attr);
  auto result_type = builder.getTensorType({n, oc, ofmap[0], ofmap[1]}, elementType);
  auto op = OpBuilder(block).create<tpu::Conv2DOp>(
      builder.getUnknownLoc(), result_type, input_var, filter, bias,
      /*dilation_h_factor=*/builder.getI32IntegerAttr(d[0]),
      /*dilation_w_factor=*/builder.getI32IntegerAttr(d[1]),
      /*fused_activation_function=*/builder.getStringAttr("NONE"),
      /*padding=*/builder.getStringAttr("SAME"),
      /*stride_h=*/builder.getI32IntegerAttr(s[0]),
      /*stride_w=*/builder.getI32IntegerAttr(s[1]));
  auto result_var = op.getResult();
  return result_var;
}

static mlir::Value *addReshapeOpInBlock(Builder builder, Block *block,
    mlir::Value *input_value, mlir::Type output_type) {
  auto op = OpBuilder(block).create<tpu::ReshapeOp>(
        builder.getUnknownLoc(), output_type, input_value);
  auto result = op.getResult();
  return result;
}

// Adds a one-block function named as `tpu_module` to `module` and returns the
// block. The created block will be terminated by `std.return`.
static Block *createOneBlockFunction(Builder builder, ModuleOp module,
    ArrayRef<mlir::Type> arguments, ArrayRef<mlir::Type> returns) {
  auto fnType = builder.getFunctionType(arguments, returns);
  auto fn = FuncOp::create(builder.getUnknownLoc(), "tpu_func", fnType);
  module.push_back(fn);

  //fn.addEntryBlock();
  //auto *block = &fn.front();
  /// auto &block = *fn.addEntryBlock();
  auto *block = fn.addEntryBlock();
  return block;
}

// Translate CaffeModel in the file named as `inputFilename` and returns a
// module in TPU Dialect.
static OwningModuleRef caffeToMlirTranslate(llvm::StringRef inputFilename,
                                  MLIRContext *context) {
  // builder and module
  Builder builder(context);
  OwningModuleRef module(ModuleOp::create(
      FileLineColLoc::get(inputFilename, /*line=*/0, /*column=*/0, context)));
  mlir::Type elementType = mlir::FloatType::getF32(builder.getContext());

  // init caffe net
  caffe::NetParameter param;
  caffe::ReadNetParamsFromTextFileOrDie(inputFilename, &param);
  param.mutable_state()->set_phase(caffe::TEST);
  caffe::Net<float> net(param);

  // dump all layers
  LLVM_DEBUG(printCaffeNetAllLayer(net););

  // create a map for mapping blob_name and a mlir tensor value
  std::map<std::string, mlir::Value *> tensor_map;

  // find caffe model inputs and outputs
  std::vector<mlir::Type> net_input_type_vec;
  for (int i = 0; i <= net.num_inputs() - 1; ++i) {
    int index = net.input_blob_indices()[i];
    std::cout << "net input [" << i << "] - [" << index << "] : "
        << ", blob: " << net.blob_names()[index]
        << ", shape: " << net.input_blobs()[i]->shape_string()
        << ", layer: " << net.layer_names()[index]
        << "\n";
    net_input_type_vec.push_back(getMlirTypeFromCaffeShape(builder,
        net.input_blobs()[i]->shape(), elementType));
  }
  std::vector<mlir::Type> net_output_type_vec;
  for (int i = 0; i <= net.num_outputs() - 1; ++i) {
    int index = net.output_blob_indices()[i];
    std::cout << "net output[" << i << "] - [" << index << "] : "
        << "blob: " << net.blob_names()[index]
        << ", shape: " << net.output_blobs()[i]->shape_string()
        << ", layer: " << net.layer_names()[index]
        << "\n";
    net_output_type_vec.push_back(getMlirTypeFromCaffeShape(builder,
        net.output_blobs()[i]->shape(), elementType));
  }

  // create Function Op with arguments and returns type
  llvm::ArrayRef<mlir::Type> func_arg_type(net_input_type_vec);
  llvm::ArrayRef<mlir::Type> func_ret_type(net_output_type_vec);
  Block *block = createOneBlockFunction(builder, module.get(),
      func_arg_type, func_ret_type);

  // we can only handle one input for now, get the function input variable
  assert(func_arg_type.size() == 1);
  mlir::Value *func_arg0_var = block->getArgument(0);

  // convert layers
  for (size_t i = 0; i <= net.layers().size() - 1; ++i) {
    auto layer = net.layers()[i].get();
    auto layer_param = layer->layer_param();
    printCaffeLayerParam(layer);

    if (strcmp(layer->type(), "Input") == 0) {
      assert(layer_param.bottom_size() == 0 && layer_param.top_size() == 1);
      tensor_map[layer_param.top(0)] = func_arg0_var;
    } else if (strcmp(layer->type(), "Convolution") == 0) {
      assert(layer_param.bottom_size() == 1 && layer_param.top_size() == 1);
      mlir::Value *input_var = tensor_map.find(layer_param.bottom(0))->second;
      assert(input_var);
      assert(layer_param.has_convolution_param());
      mlir::Value *result_var = addConv2dOpInBlockFromCaffe(builder, block,
          elementType, input_var, layer_param.convolution_param());
      tensor_map[layer_param.top(0)] = result_var;
    } else if (strcmp(layer->type(), "BatchNorm") == 0) {
      assert(layer_param.bottom_size() == 1 && layer_param.top_size() == 1);
      mlir::Value *input = tensor_map.find(layer_param.bottom(0))->second;
      assert(input);

      // bypass for now
      tensor_map[layer_param.top(0)] = input;
    } else if (strcmp(layer->type(), "Scale") == 0) {
      assert(layer_param.bottom_size() == 1 && layer_param.top_size() == 1);
      mlir::Value *input = tensor_map.find(layer_param.bottom(0))->second;
      assert(input);

      // bypass for now
      tensor_map[layer_param.top(0)] = input;
    } else if (strcmp(layer->type(), "ReLU") == 0) {
      assert(layer_param.bottom_size() == 1 && layer_param.top_size() == 1);
      mlir::Value *input = tensor_map.find(layer_param.bottom(0))->second;
      assert(input);

      // bypass for now
      tensor_map[layer_param.top(0)] = input;
    } else if (strcmp(layer->type(), "Eltwise") == 0) {
      assert(layer_param.bottom_size() == 2 && layer_param.top_size() == 1);
      mlir::Value *input_1 = tensor_map.find(layer_param.bottom(0))->second;
      assert(input_1);
      mlir::Value *input_2 = tensor_map.find(layer_param.bottom(0))->second;
      assert(input_2);

      // bypass for now
      tensor_map[layer_param.top(0)] = input_1;
    } else if (strcmp(layer->type(), "Pooling") == 0) {
      assert(layer_param.bottom_size() == 1 && layer_param.top_size() == 1);
      mlir::Value *input = tensor_map.find(layer_param.bottom(0))->second;
      assert(input);

      // bypass for now
      tensor_map[layer_param.top(0)] = input;
    } else if (strcmp(layer->type(), "InnerProduct") == 0) {
      assert(layer_param.bottom_size() == 1 && layer_param.top_size() == 1);
      mlir::Value *input = tensor_map.find(layer_param.bottom(0))->second;
      assert(input);

      // bypass for now
      tensor_map[layer_param.top(0)] = input;
    } else if (strcmp(layer->type(), "Split") == 0) {
      assert(layer_param.bottom_size() == 1 && layer_param.top_size() == 2);
      mlir::Value *input = tensor_map.find(layer_param.bottom(0))->second;
      assert(input);

      // bypass
      // by registering blob_name to the same mlir tensor
      tensor_map[layer_param.top(0)] = input;
      tensor_map[layer_param.top(1)] = input;
    } else if (strcmp(layer->type(), "Softmax") == 0) {
      std::cout << "    SKIP" << "\n";
      assert(layer_param.bottom_size() == 1 && layer_param.top_size() == 1);
      std::cout << "btm: " << layer_param.bottom(0) << "\n";
      std::cout << "top: " << layer_param.top(0) << "\n";
      mlir::Value *input = tensor_map.find(layer_param.bottom(0))->second;
      assert(input);

      // bypass for now
      tensor_map[layer_param.top(0)] = input;
    } else {
      std::cout << "    UNKNOWN" << "\n";
      assert(false);
    }
  }

  // add fake input and reshape
  // support only one output for now
  assert(func_ret_type.size() == 1);
  auto reshape_output = addReshapeOpInBlock(builder, block, func_arg0_var,
      func_ret_type[0]);

  // 4. return Op
  llvm::ArrayRef<mlir::Value *> func_ret_var = {reshape_output};
  OpBuilder(block).create<ReturnOp>(builder.getUnknownLoc(), func_ret_var);

  return module;
}

static TranslateToMLIRRegistration
    registration("caffe-to-mlir-v2",
                 [](StringRef inputFilename, MLIRContext *context) {
                   return caffeToMlirTranslate(inputFilename, context);
                 });
