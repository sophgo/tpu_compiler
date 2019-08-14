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

using namespace mlir;

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
    ArrayRef<mlir::Type> inputs, ArrayRef<mlir::Type> outputs) {
  auto fnType = builder.getFunctionType(inputs, outputs);
  auto fn = FuncOp::create(builder.getUnknownLoc(), "tpu_func", fnType);
  module.push_back(fn);

  //fn.addEntryBlock();
  //auto *block = &fn.front();
  /// auto &block = *fn.addEntryBlock();
  auto *block = fn.addEntryBlock();
  return block;
}

static mlir::Type mlirTypeFromCaffeShape(Builder builder,
    const std::vector<int> shape, mlir::Type elementType) {
  std::vector<int64_t> shape_int64(shape.begin(), shape.end());
  llvm::ArrayRef<int64_t> mlir_shape(shape_int64);
  auto mlir_type = builder.getTensorType(mlir_shape, elementType);
  return mlir_type;
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
  for (size_t i = 0; i <= net.layers().size() - 1; ++i) {
    //LOG(INFO) << "> [" << std::left << std::setw(12) << std::setfill(' ')
    //    << net.layers()[i]->type() << std::setw(0) << "] "
    //    << net.layers()[i]->layer_param().name();
    auto layer = net.layers()[i];
    auto layer_param = layer->layer_param();
    std::cout << "> [" << std::left << std::setw(12) << std::setfill(' ') << layer->type()
        << std::setw(0) << "] " << layer_param.name() << "\n";
  }

  // 1. find caffe model input and output
  std::vector<mlir::Type> input_type_vec;
  for (int i = 0; i <= net.num_inputs() - 1; ++i) {
    int index = net.input_blob_indices()[i];
    std::cout << "input [" << i << "] - [" << index << "] : "
        << ", blob: " << net.blob_names()[index]
        << ", shape: " << net.input_blobs()[i]->shape_string()
        << ", layer: " << net.layer_names()[index]
        << "\n";
    //std::vector<int> shape = net.input_blobs()[i]->shape();
    //std::vector<int64_t> shape_int64(shape.begin(), shape.end());
    //llvm::ArrayRef<int64_t> input_shape(shape_int64);
    //auto input_type = builder.getTensorType(input_shape, elementType);
    //input_type_vec.push_back(input_type);
    input_type_vec.push_back(mlirTypeFromCaffeShape(builder,
        net.input_blobs()[i]->shape(), elementType));
  }
  std::vector<mlir::Type> output_type_vec;
  for (int i = 0; i <= net.num_outputs() - 1; ++i) {
    int index = net.output_blob_indices()[i];
    std::cout << "output[" << i << "] - [" << index << "] : "
        << "blob: " << net.blob_names()[index]
        << ", shape: " << net.output_blobs()[i]->shape_string()
        << ", layer: " << net.layer_names()[index]
        << "\n";
    output_type_vec.push_back(mlirTypeFromCaffeShape(builder,
        net.output_blobs()[i]->shape(), elementType));
  }

  // 2. create Function Op with input and output
  llvm::ArrayRef<mlir::Type> inputs(input_type_vec);
  llvm::ArrayRef<mlir::Type> outputs(output_type_vec);
  Block *block = createOneBlockFunction(builder, module.get(), inputs, outputs);

  // 3. convert layers
  for (size_t i = 0; i <= net.layers().size() - 1; ++i) {
    auto layer = net.layers()[i];
    auto layer_param = layer->layer_param();
    std::cout << ">> [" << std::left << std::setw(12) << std::setfill(' ') << layer->type()
        << std::setw(0) << "] " << layer_param.name() << "\n";
    if (strcmp(layer->type(), "Input") == 0) {
      std::cout << "    SKIP" << "\n";
    } else if (strcmp(layer->type(), "Convolution") == 0) {

    } else if (strcmp(layer->type(), "BatchNorm") == 0) {

    } else if (strcmp(layer->type(), "Scale") == 0) {

    } else if (strcmp(layer->type(), "ReLU") == 0) {

    } else if (strcmp(layer->type(), "Eltwise") == 0) {

    } else if (strcmp(layer->type(), "Pooling") == 0) {

    } else if (strcmp(layer->type(), "InnerProduct") == 0) {

    } else if (strcmp(layer->type(), "Split") == 0) {

    } else if (strcmp(layer->type(), "Softmax") == 0) {
      std::cout << "    SKIP" << "\n";
    } else {
      std::cout << "    UNKNOWN" << "\n";
      assert(false);
    }
  }

  // add fake input and reshape
  mlir::Value *input = block->getArgument(0);
  auto output = addReshapeOpInBlock(builder, block, input, output_type_vec[0]);

  // 4. return Op
  llvm::ArrayRef<mlir::Value *> results = {output};
  OpBuilder(block).create<ReturnOp>(builder.getUnknownLoc(), results);

  return module;
}

static TranslateToMLIRRegistration
    registration("caffe-to-mlir-v2",
                 [](StringRef inputFilename, MLIRContext *context) {
                   return caffeToMlirTranslate(inputFilename, context);
                 });
