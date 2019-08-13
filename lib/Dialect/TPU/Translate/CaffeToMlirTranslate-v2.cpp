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

// Adds a one-block function named as `tpu_module` to `module` and returns the
// block. The created block will be terminated by `std.return`.
static Block *createOneBlockFunction(Builder builder, ModuleOp module) {
  auto fnType = builder.getFunctionType(/*inputs=*/{}, /*results=*/{});
  auto fn = FuncOp::create(builder.getUnknownLoc(), "tpu_module", fnType);
  module.push_back(fn);

  //fn.addEntryBlock();
  //auto *block = &fn.front();
  /// auto &block = *fn.addEntryBlock();
  auto *block = fn.addEntryBlock();

  mlir::Type elementType = mlir::FloatType::getF32(builder.getContext());
  auto result_type = builder.getTensorType({1, 16, 28, 28}, elementType);
  auto input_type = builder.getTensorType({1, 3, 28, 28}, elementType);
  auto input_attr = builder.getZeroAttr(input_type);
  auto input = OpBuilder(block).create<ConstantOp>(builder.getUnknownLoc(), input_type, input_attr);
  auto filter_type = builder.getTensorType({16, 3, 3, 3}, elementType);
  auto filter_attr = builder.getZeroAttr(filter_type);
  auto filter = OpBuilder(block).create<ConstantOp>(builder.getUnknownLoc(), filter_type, filter_attr);
  auto bias_type = builder.getTensorType({16}, elementType);
  auto bias_attr = builder.getZeroAttr(bias_type);
  auto bias = OpBuilder(block).create<ConstantOp>(builder.getUnknownLoc(), bias_type, bias_attr);
  OpBuilder(block).create<tpu::Conv2DOp>(
        builder.getUnknownLoc(), result_type, input, filter, bias,
        /*dilation_h_factor=*/builder.getI32IntegerAttr(1),
        /*dilation_w_factor=*/builder.getI32IntegerAttr(1),
        /*fused_activation_function=*/builder.getStringAttr("NONE"),
        /*padding=*/builder.getStringAttr("SAME"),
        /*stride_h=*/builder.getI32IntegerAttr(1),
        /*stride_w=*/builder.getI32IntegerAttr(1));

  OpBuilder(block).create<ReturnOp>(builder.getUnknownLoc());

  return block;
}

// Translate CaffeModel in the file named as `inputFilename` and returns a
// module in TPU Dialect.
static OwningModuleRef caffeToMlirTranslate(llvm::StringRef inputFilename,
                                  MLIRContext *context) {
  Builder builder(context);

  //std::string errorMessage;
  //auto file = openInputFile(inputFilename, &errorMessage);
  //if (!file) {
  //  emitError(UnknownLoc::get(context), errorMessage);
  //  return {};
  //}
  caffe::NetParameter param1;
  caffe::ReadNetParamsFromTextFileOrDie(inputFilename, &param1);
  param1.mutable_state()->set_phase(caffe::TEST);
  caffe::Net<float> net1(param1);
  for (int i = 0; i <= net1.layers().size() - 1; ++i) {
    //LOG(INFO) << "> [" << std::left << std::setw(12) << std::setfill(' ') << net1.layers()[i]->type()
    //    << std::setw(0) << "] " << net1.layers()[i]->layer_param().name();
    auto layer = net1.layers()[i];
    auto layer_param = layer->layer_param();
    std::cout << "> [" << std::left << std::setw(12) << std::setfill(' ') << layer->type()
        << std::setw(0) << "] " << layer_param.name() << "\n";
  }

  // wrapping the converted TPU ModuleOp inside a MLIR module.
  OwningModuleRef module(ModuleOp::create(
      FileLineColLoc::get(inputFilename, /*line=*/0, /*column=*/0, context)));
  Block *block = createOneBlockFunction(builder, module.get());
  //block->push_front(tpuModule->getOperation());

  return module;
}

static TranslateToMLIRRegistration
    registration("caffe-to-mlir-v2",
                 [](StringRef inputFilename, MLIRContext *context) {
                   return caffeToMlirTranslate(inputFilename, context);
                 });
