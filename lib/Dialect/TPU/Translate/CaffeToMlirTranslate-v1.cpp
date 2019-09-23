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
#include "mlir/Translation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace mlir;

static mlir::Value *addConv2dOpInBlock(Builder builder, Block *block,
    mlir::Type elementType, mlir::Value *input, int64_t n,
    int64_t ic, int64_t ih, int64_t iw, int64_t oc, int64_t oh, int64_t ow,
    int64_t kh, int64_t kw, int64_t sh, int64_t sw, int64_t dh, int64_t dw) {
  auto filter_type = builder.getTensorType({oc, ic, kh, kw}, elementType);
  auto filter_attr = builder.getZeroAttr(filter_type);
  auto filter = OpBuilder(block).create<ConstantOp>(builder.getUnknownLoc(),
      filter_type, filter_attr);
  auto bias_type = builder.getTensorType({oc}, elementType);
  auto bias_attr = builder.getZeroAttr(bias_type);
  auto bias = OpBuilder(block).create<ConstantOp>(builder.getUnknownLoc(),
      bias_type, bias_attr);
  auto result_type = builder.getTensorType({n, oc, oh, ow}, elementType);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("dilation_h_factor", builder.getI32IntegerAttr(dh)));
  attrs.push_back(builder.getNamedAttr("dilation_w_factor", builder.getI32IntegerAttr(dw)));
  attrs.push_back(builder.getNamedAttr("fused_activation_function", builder.getStringAttr("NONE")));
  attrs.push_back(builder.getNamedAttr("padding", builder.getStringAttr("SAME")));
  attrs.push_back(builder.getNamedAttr("stride_h", builder.getI32IntegerAttr(sh)));
  attrs.push_back(builder.getNamedAttr("stride_w", builder.getI32IntegerAttr(sw)));
  auto op = OpBuilder(block).create<tpu::Conv2DOp>(
      builder.getUnknownLoc(), result_type,
      ArrayRef<Value *>{input, filter, bias},
      ArrayRef<NamedAttribute>{attrs});
  auto result = op.getResult();
  return result;
}

static mlir::Value *addReluOpInBlock(Builder builder, Block *block,
    mlir::Type elementType, mlir::Value *input, int64_t n,
    int64_t c, int64_t h, int64_t w) {
  auto result_type = builder.getTensorType({n, c, h, w}, elementType);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("negative_slope", builder.getF32FloatAttr(1.0f)));
  auto op = OpBuilder(block).create<tpu::ReluOp>(
      builder.getUnknownLoc(), result_type, ArrayRef<Value *>{input},
      ArrayRef<NamedAttribute>{attrs});
  auto result = op.getResult();
  return result;
}

static mlir::Value *addAveragePool2DOpInBlock(Builder builder, Block *block,
    mlir::Type elementType, mlir::Value *input, int64_t n,
    int64_t c, int64_t ih, int64_t iw, int64_t oh, int64_t ow,
    int64_t kh, int64_t kw, int64_t sh, int64_t sw) {
  auto result_type = builder.getTensorType({n, c, oh, ow}, elementType);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("filter_height", builder.getI32IntegerAttr(kh)));
  attrs.push_back(builder.getNamedAttr("filter_width", builder.getI32IntegerAttr(kw)));
  attrs.push_back(builder.getNamedAttr("padding", builder.getStringAttr("VALID")));
  attrs.push_back(builder.getNamedAttr("stride_h", builder.getI32IntegerAttr(sh)));
  attrs.push_back(builder.getNamedAttr("stride_w", builder.getI32IntegerAttr(sw)));
  attrs.push_back(builder.getNamedAttr("fused_activation_function", builder.getStringAttr("NONE")));
  auto op = OpBuilder(block).create<tpu::AveragePool2DOp>(
        builder.getUnknownLoc(), result_type, ArrayRef<Value *>{input},
        ArrayRef<NamedAttribute>{attrs});
  auto result = op.getResult();
  return result;
}

static mlir::Value *addReshapeOpInBlock(Builder builder, Block *block,
    mlir::Type elementType, mlir::Value *input, llvm::ArrayRef<int64_t> shape) {
  auto result_type = builder.getTensorType(shape, elementType);
  auto op = OpBuilder(block).create<tpu::ReshapeOp>(
        builder.getUnknownLoc(), result_type, input);
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

// Translate CaffeModel in the file named as `inputFilename` and returns a
// module in TPU Dialect.
static OwningModuleRef caffeToMlirTranslate(llvm::StringRef inputFilename,
                                  MLIRContext *context) {
  Builder builder(context);

  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    emitError(UnknownLoc::get(context), errorMessage);
    return {};
  }

  // wrapping the converted TPU ModuleOp inside a MLIR module.
  OwningModuleRef module(ModuleOp::create(
      FileLineColLoc::get(inputFilename, /*line=*/0, /*column=*/0, context)));

  mlir::Type elementType = mlir::FloatType::getF32(builder.getContext());

  auto input_type = builder.getTensorType({1, 1, 28, 28}, elementType);
  auto output_type = builder.getTensorType({1, 10}, elementType);
  llvm::ArrayRef<mlir::Type> inputs = {input_type};
  llvm::ArrayRef<mlir::Type> outputs = {output_type};
  Block *block = createOneBlockFunction(builder, module.get(), inputs, outputs);

  //
  // construct a mnist cnn
  //
  //mlir::Value *input = fn.getArgument(0);
  mlir::Value *input = block->getArgument(0);
  auto c1 = addConv2dOpInBlock(builder, block, elementType, input,
      /*I_NCHW*/1, 1, 28, 28, /*O_CHW*/16, 14, 14, /*k/s/d*/3, 3, 2, 2, 1, 1);
  auto r1 = addReluOpInBlock(builder, block, elementType, c1,
      1, 16, 14, 14);
  auto c2 = addConv2dOpInBlock(builder, block, elementType, r1,
      /*I_NCHW*/1, 16, 14, 14, /*O_CHW*/16, 7, 7, /*k/s/d*/3, 3, 2, 2, 1, 1);
  auto r2 = addReluOpInBlock(builder, block, elementType, c2,
      1, 16, 7, 7);
  auto c3 = addConv2dOpInBlock(builder, block, elementType, r2,
      /*I_NCHW*/1, 16, 7, 7, /*O_CHW*/10, 4, 4, /*k/s/d*/3, 3, 2, 2, 1, 1);
  auto r3 = addReluOpInBlock(builder, block, elementType, c3,
      1, 10, 4, 4);
  auto avg = addAveragePool2DOpInBlock(builder, block, elementType, r3,
      /*I_NCHW*/1, 10, 4, 4, /*O_HW*/1, 1, /*k/s*/4, 4, 1, 1);
  auto output = addReshapeOpInBlock(builder, block, elementType, avg,
      llvm::ArrayRef<int64_t>({1, 10}));

  llvm::ArrayRef<mlir::Value *> results = {output};
  OpBuilder(block).create<ReturnOp>(builder.getUnknownLoc(), results);

  return module;
}

static TranslateToMLIRRegistration
    registration("caffe-to-mlir-v1",
                 [](StringRef inputFilename, MLIRContext *context) {
                   return caffeToMlirTranslate(inputFilename, context);
                 });
