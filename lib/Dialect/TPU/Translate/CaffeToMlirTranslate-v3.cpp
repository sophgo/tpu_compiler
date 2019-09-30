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
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/Format.h"
#include "mlir/Support/LogicalResult.h"

#include "caffe/caffe.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/util/signal_handler.h"

#include <iostream>
#include <cstring>
#include <numeric>
#include <map>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <fcntl.h>
#include "google/protobuf/message.h"
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include "bmnet/common_calibration.pb.h"

#define DEBUG_TYPE "caffe-to-mlir-v3"

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

//===----------------------------------------------------------------------===//
// Create Operations from Caffe Proto, and extract weight
//===----------------------------------------------------------------------===//
static mlir::Value *addConv2dOpInBlockFromCaffe(Builder builder, Block *block,
    mlir::Type elementType, mlir::Value *input_var, caffe::Layer<float> *layer,
    mlir::Value *weight_var, llvm::raw_fd_ostream *weight_os = NULL,
    mlir::TensorFile *weightTensorFile = NULL) {
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
  padding[0] = p.has_pad_h() ? p.pad_h() : p.pad_size() > 1 ? p.pad(1) : p.pad_size() > 0 ? p.pad(0) : 0;
  padding[1] = p.has_pad_w() ? p.pad_w() : p.pad_size() > 0 ? p.pad(0) : 0;
  dilation[0] = p.dilation_size() > 1 ? p.dilation(1) : p.dilation_size() > 0 ? p.dilation(0) : 1;
  dilation[1] = p.dilation_size() > 0 ? p.dilation(0) : 1;

  // TODO: don't support group for now
  assert(group == 1);
  assert( (dilation[0] == 1) && (dilation[1] == 1) );

  // get input shape from input var
  DEBUG_WITH_TYPE(DEBUG_TYPE"_VERBOSE", input_var->getType().dump(););
  llvm::ArrayRef<int64_t> input_shape = input_var->getType().dyn_cast<mlir::TensorType>().getShape();
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

  size_t pos_filter, pos_bias;
  if (weight_os) {
    // - blobs_[0] holds the filter weights
    // - blobs_[1] holds the biases (optional)
    assert(layer->blobs().size() == 1 || layer->blobs().size() == 2);

    // filter weights
    const caffe::Blob<float> *blob_filter = layer->blobs()[0].get();
    llvm::errs() << "  filter shape " << blob_filter->shape_string() << "\n";
    std::vector<int> filter_shape = blob_filter->shape();
    assert(filter_shape.size() == 4);
    assert(filter_shape[0] == oc);
    assert(filter_shape[1] == ic);
    assert(filter_shape[2] == kernel[0]);
    assert(filter_shape[3] == kernel[1]);
    pos_filter = weight_os->tell();
    weight_os->write(reinterpret_cast<const char*>(blob_filter->cpu_data()),
        blob_filter->count() * sizeof(float));
    llvm::errs() << "  filter: " << llvm::format_hex(pos_filter, 10)
                 << " --> " << llvm::format_hex(weight_os->tell(), 10) << "\n";

    // bias
    if (with_bias) {
      assert(layer->blobs().size() == 2);
      const caffe::Blob<float> *blob_bias = layer->blobs()[1].get();
      llvm::errs() << "  bias shape " << blob_bias->shape_string() << "\n";
      std::vector<int> bias_shape = blob_bias->shape();
      assert(bias_shape.size() == 1);
      assert(bias_shape[0] == oc);
      pos_bias = weight_os->tell();
      weight_os->write(reinterpret_cast<const char*>(blob_bias->cpu_data()),
          blob_bias->count() * sizeof(float));
      llvm::errs() << "  bias: " << llvm::format_hex(pos_bias, 10)
                   << " --> " << llvm::format_hex(weight_os->tell(), 10)
                   << "\n";
    } else {
      assert(layer->blobs().size() == 1);
    }
  }

  // construct OP
  std::vector<Value *> operands;
  operands.push_back(input_var);

  auto filter_tensorname = layer->layer_param().name()+"_0";
  auto filter_type = builder.getTensorType({oc, ic, kernel[0], kernel[1]}, elementType);
  std::vector<NamedAttribute> filter_attrs;
  filter_attrs.push_back(builder.getNamedAttr("offset", builder.getI64IntegerAttr(pos_filter)));
  filter_attrs.push_back(builder.getNamedAttr("name", builder.getStringAttr(filter_tensorname)));
  auto filter = OpBuilder(block).create<tpu::LoadWeightOp>(
      builder.getUnknownLoc(), filter_type,
      ArrayRef<Value *>{weight_var}, ArrayRef<NamedAttribute>{filter_attrs});
  operands.push_back(filter);

  const caffe::Blob<float> *blob_filter = layer->blobs()[0].get();
  weightTensorFile->addTensor(filter_tensorname, blob_filter->cpu_data(),
      filter_type);

  if (with_bias) {
    auto bias_type = builder.getTensorType({oc}, elementType);
    std::vector<NamedAttribute> bias_attrs;
    bias_attrs.push_back(builder.getNamedAttr("offset", builder.getI64IntegerAttr(pos_bias)));
    auto bias = OpBuilder(block).create<tpu::LoadWeightOp>(
        builder.getUnknownLoc(), bias_type,
        ArrayRef<Value *>{weight_var}, ArrayRef<NamedAttribute>{bias_attrs});
    operands.push_back(bias);
  }
  auto result_type = builder.getTensorType({n, oc, ofmap[0], ofmap[1]}, elementType);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("dilation_h_factor", builder.getI32IntegerAttr(dilation[0])));
  attrs.push_back(builder.getNamedAttr("dilation_w_factor", builder.getI32IntegerAttr(dilation[1])));
  //attrs.push_back(builder.getNamedAttr("fused_activation_function", builder.getStringAttr("NONE")));
  attrs.push_back(builder.getNamedAttr("padding", (padding[0] || padding[1])
                  ? builder.getStringAttr("SAME") : builder.getStringAttr("VALID")));
  attrs.push_back(builder.getNamedAttr("stride_h", builder.getI32IntegerAttr(stride[0])));
  attrs.push_back(builder.getNamedAttr("stride_w", builder.getI32IntegerAttr(stride[1])));
  auto op = OpBuilder(block).create<tpu::Conv2DOp>(
      builder.getUnknownLoc(), result_type,
      ArrayRef<Value *>{operands}, ArrayRef<NamedAttribute>{attrs});
  auto result_var = op.getResult();
  return result_var;
}

static mlir::Value *addBatchNormOpInBlockFromCaffe(Builder builder, Block *block,
    mlir::Type elementType, mlir::Value *input_var,
    caffe::Layer<float> *layer,
    mlir::Value *weight_var,
    llvm::raw_fd_ostream *weight_os = NULL) {
  auto layer_param = layer->layer_param();
  assert(layer_param.has_batch_norm_param());
  auto batch_norm_param = layer_param.batch_norm_param();
  //float epsilon = batch_norm_param.eps();

  int64_t n, c, h, w;
  // get input shape from input vars
  DEBUG_WITH_TYPE(DEBUG_TYPE"_VERBOSE", input_var->getType().dump(););
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

  size_t pos_mean, pos_variance, pos_scale;
  if (weight_os) {
    // - blobs_[2] holds the scale, which is one scalar data
    // - blobs_[0] holds the mean
    // - blobs_[1] holds the variance
    assert(layer->blobs().size() == 3);

    // mean weights
    const caffe::Blob<float> *blob_mean = layer->blobs()[0].get();
    llvm::errs() << "  mean shape " << blob_mean->shape_string() << "\n";
    std::vector<int> mean_shape = blob_mean->shape();
    assert(mean_shape.size() == 1);
    assert(mean_shape[0] == c);
    pos_mean = weight_os->tell();
    weight_os->write(reinterpret_cast<const char*>(blob_mean->cpu_data()),
        blob_mean->count() * sizeof(float));
    llvm::errs() << "  mean: " << llvm::format_hex(pos_mean, 10)
                 << " --> " << llvm::format_hex(weight_os->tell(), 10) << "\n";

    // variance weights
    const caffe::Blob<float> *blob_variance = layer->blobs()[1].get();
    llvm::errs() << "  variance shape " << blob_variance->shape_string() << "\n";
    std::vector<int> variance_shape = blob_variance->shape();
    assert(variance_shape.size() == 1);
    assert(variance_shape[0] == c);
    pos_variance = weight_os->tell();
    weight_os->write(reinterpret_cast<const char*>(blob_variance->cpu_data()),
        blob_variance->count() * sizeof(float));
    llvm::errs() << "  variance: " << llvm::format_hex(pos_variance, 10)
                 << " --> " << llvm::format_hex(weight_os->tell(), 10) << "\n";

    // scale is also a learnable param
    const caffe::Blob<float> *blob_scale = layer->blobs()[2].get();
    llvm::errs() << "  scale shape " << blob_scale->shape_string() << "\n";
    std::vector<int> scale_shape = blob_scale->shape();
    assert(scale_shape.size() == 1);
    assert(scale_shape[0] == 1);
    float scale_value = *((const float *)blob_scale->cpu_data());
    llvm::errs() << "  scale: " << scale_value << "\n";
    pos_scale = weight_os->tell();
    weight_os->write(reinterpret_cast<const char*>(blob_scale->cpu_data()),
        blob_scale->count() * sizeof(float));
    llvm::errs() << "  scale: " << llvm::format_hex(pos_scale, 10)
                 << " --> " << llvm::format_hex(weight_os->tell(), 10) << "\n";
  }

  // construct OP
  auto mean_type = builder.getTensorType({c}, elementType);
  std::vector<NamedAttribute> mean_attrs;
  mean_attrs.push_back(builder.getNamedAttr("offset", builder.getI64IntegerAttr(pos_mean)));
  auto mean = OpBuilder(block).create<tpu::LoadWeightOp>(
      builder.getUnknownLoc(), mean_type,
      ArrayRef<Value *>{weight_var}, ArrayRef<NamedAttribute>{mean_attrs});
  auto variance_type = builder.getTensorType({c}, elementType);
  std::vector<NamedAttribute> variance_attrs;
  variance_attrs.push_back(builder.getNamedAttr("offset", builder.getI64IntegerAttr(pos_variance)));
  auto variance = OpBuilder(block).create<tpu::LoadWeightOp>(
      builder.getUnknownLoc(), variance_type,
      ArrayRef<Value *>{weight_var}, ArrayRef<NamedAttribute>{variance_attrs});
  auto scale_type = builder.getTensorType({1}, elementType);
  std::vector<NamedAttribute> scale_attrs;
  scale_attrs.push_back(builder.getNamedAttr("offset", builder.getI64IntegerAttr(pos_scale)));
  auto scale = OpBuilder(block).create<tpu::LoadWeightOp>(
      builder.getUnknownLoc(), scale_type,
      ArrayRef<Value *>{weight_var}, ArrayRef<NamedAttribute>{scale_attrs});
  auto result_type = builder.getTensorType({n, c, h, w}, elementType);
  auto op = OpBuilder(block).create<tpu::BatchNormOp>(
      builder.getUnknownLoc(), result_type,
      ArrayRef<Value *>{input_var, mean, variance, scale},
      ArrayRef<NamedAttribute>{});
  auto result_var = op.getResult();
  return result_var;
}

static mlir::Value *addScaleOpInBlockFromCaffe(Builder builder, Block *block,
    mlir::Type elementType, mlir::Value *input_var,
    caffe::Layer<float> *layer,
    mlir::Value *weight_var,
    llvm::raw_fd_ostream *weight_os = NULL) {
  auto layer_param = layer->layer_param();
  assert(layer_param.has_scale_param());
  auto scale_param = layer_param.scale_param();
  bool with_bias = scale_param.bias_term();

  int64_t n, c, h, w;
  // get input shape from input vars
  DEBUG_WITH_TYPE(DEBUG_TYPE"_VERBOSE", input_var->getType().dump(););
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

  size_t pos_scale, pos_bias;
  if (weight_os) {
    // - blobs_[0] holds the scale
    // - blobs_[1] holds the biases (optional)
    // sometimes, offset is hold in a aublayer (bias layer) blob_[0]
    assert(layer->blobs().size() == 1 || layer->blobs().size() == 2);

    // scale weights
    const caffe::Blob<float> *blob_scale = layer->blobs()[0].get();
    llvm::errs() << "  scale shape " << blob_scale->shape_string() << "\n";
    std::vector<int> scale_shape = blob_scale->shape();
    assert(scale_shape.size() == 1);
    assert(scale_shape[0] == c);
    pos_scale = weight_os->tell();
    weight_os->write(reinterpret_cast<const char*>(blob_scale->cpu_data()),
        blob_scale->count() * sizeof(float));
    llvm::errs() << "  scale: " << llvm::format_hex(pos_scale, 10)
                 << " --> " << llvm::format_hex(weight_os->tell(), 10) << "\n";

    // bias
    if (with_bias) {
      assert(layer->blobs().size() == 2);
      const caffe::Blob<float> *blob_bias = layer->blobs()[1].get();
      llvm::errs() << "  bias shape " << blob_bias->shape_string() << "\n";
      std::vector<int> bias_shape = blob_bias->shape();
      assert(bias_shape.size() == 1);
      assert(bias_shape[0] == c);
      pos_bias = weight_os->tell();
      weight_os->write(reinterpret_cast<const char*>(blob_bias->cpu_data()),
          blob_bias->count() * sizeof(float));
      llvm::errs() << "  bias: " << llvm::format_hex(pos_bias, 10)
                   << " --> " << llvm::format_hex(weight_os->tell(), 10)
                   << "\n";
    } else {
      assert(layer->blobs().size() == 1);
    }
  }

  // construct OP
  std::vector<Value *> operands;
  operands.push_back(input_var);
  auto scale_type = builder.getTensorType({c}, elementType);
  std::vector<NamedAttribute> scale_attrs;
  scale_attrs.push_back(builder.getNamedAttr("offset", builder.getI64IntegerAttr(pos_scale)));
  auto scale = OpBuilder(block).create<tpu::LoadWeightOp>(
      builder.getUnknownLoc(), scale_type,
      ArrayRef<Value *>{weight_var}, ArrayRef<NamedAttribute>{scale_attrs});
  operands.push_back(scale);
  if (with_bias) {
    auto bias_type = builder.getTensorType({c}, elementType);
    std::vector<NamedAttribute> bias_attrs;
    bias_attrs.push_back(builder.getNamedAttr("offset", builder.getI64IntegerAttr(pos_bias)));
    auto bias = OpBuilder(block).create<tpu::LoadWeightOp>(
        builder.getUnknownLoc(), bias_type,
        ArrayRef<Value *>{weight_var}, ArrayRef<NamedAttribute>{bias_attrs});
    operands.push_back(bias);
  }
  auto result_type = builder.getTensorType({n, c, h, w}, elementType);
  auto op = OpBuilder(block).create<tpu::ScaleOp>(
      builder.getUnknownLoc(), result_type, ArrayRef<Value *>{operands},
      ArrayRef<NamedAttribute>{});
  auto result_var = op.getResult();
  return result_var;
}

static mlir::Value *addReluOpInBlockFromCaffe(Builder builder, Block *block,
    mlir::Type elementType, mlir::Value *input_var,
    caffe::Layer<float> *layer) {
  auto layer_param = layer->layer_param();
  //assert(layer_param.has_relu_param());
  auto relu_param = layer_param.relu_param();
  float negative_slope = relu_param.negative_slope();

  int64_t n, c, h, w;
  // get input shape from input vars
  DEBUG_WITH_TYPE(DEBUG_TYPE"_VERBOSE", input_var->getType().dump(););
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
  auto result_type = builder.getTensorType({n, c, h, w}, elementType);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("negative_slope", builder.getF32FloatAttr(negative_slope)));
  auto op = OpBuilder(block).create<tpu::ReluOp>(
      builder.getUnknownLoc(), result_type, ArrayRef<Value *>{input_var},
      ArrayRef<NamedAttribute>{attrs});
  auto result_var = op.getResult();
  return result_var;
}

static mlir::Value *addSoftmaxOpInBlockFromCaffe(Builder builder, Block *block,
    mlir::Type elementType, mlir::Value *input_var,
    caffe::Layer<float> *layer) {
  auto layer_param = layer->layer_param();

  int64_t n, c;
  // get input shape from input vars
  DEBUG_WITH_TYPE(DEBUG_TYPE"_VERBOSE", input_var->getType().dump(););
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
  auto result_type = builder.getTensorType({n, c}, elementType);
  auto op = OpBuilder(block).create<tpu::SoftmaxOp>(
      builder.getUnknownLoc(), result_type, ArrayRef<Value *>{input_var},
      ArrayRef<NamedAttribute>{});
  auto result_var = op.getResult();
  return result_var;
}

static mlir::Value *addEltwiseOpInBlockFromCaffe(Builder builder, Block *block,
    mlir::Type elementType, mlir::Value *input_1_var, mlir::Value *input_2_var,
    caffe::Layer<float> *layer) {
  auto layer_param = layer->layer_param();
  //assert(layer_param.has_eltwise_param());
  auto eltwise_param = layer_param.eltwise_param();
  assert(eltwise_param.coeff_size() == 0);
  assert(eltwise_param.operation() == caffe::EltwiseParameter_EltwiseOp_SUM);

  int64_t n, c, h, w;
  // get input shape from input vars
  DEBUG_WITH_TYPE(DEBUG_TYPE"_VERBOSE", input_1_var->getType().dump(););
  llvm::ArrayRef<int64_t> input_1_shape =
      input_1_var->getType().dyn_cast<mlir::TensorType>().getShape();
  assert(input_1_shape.size() == 4);
  n = input_1_shape[0];
  c = input_1_shape[1];
  h = input_1_shape[2];
  w = input_1_shape[3];

  DEBUG_WITH_TYPE(DEBUG_TYPE"_VERBOSE", input_2_var->getType().dump(););
  llvm::ArrayRef<int64_t> input_2_shape =
      input_2_var->getType().dyn_cast<mlir::TensorType>().getShape();
  assert(input_2_shape == input_1_shape);

  LLVM_DEBUG(
    llvm::errs()
        << "  N: " << n
        << ", C: " << c
        << ", IH*IW: " << h << " * " << w
        << "\n";
  );

  // construct OP
  auto result_type = builder.getTensorType({n, c, h, w}, elementType);
  auto op = OpBuilder(block).create<tpu::EltwiseOp>(
      builder.getUnknownLoc(), result_type,
      ArrayRef<Value *>{input_1_var, input_2_var},
      ArrayRef<NamedAttribute>{});
  auto result_var = op.getResult();
  return result_var;
}

static mlir::Value *addPoolingOpInBlockFromCaffe(Builder builder, Block *block,
    mlir::Type elementType, mlir::Value *input_var,
    caffe::Layer<float> *layer) {
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
  DEBUG_WITH_TYPE(DEBUG_TYPE"_VERBOSE", input_var->getType().dump(););
  llvm::ArrayRef<int64_t> input_shape = input_var->getType().dyn_cast<mlir::TensorType>().getShape();
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
  auto result_type = builder.getTensorType({n, c, ofmap[0], ofmap[1]}, elementType);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("filter_height", builder.getI32IntegerAttr(kernel[0])));
  attrs.push_back(builder.getNamedAttr("filter_width", builder.getI32IntegerAttr(kernel[1])));
  attrs.push_back(builder.getNamedAttr("padding",
      (padding[0] || padding[1]) ? builder.getStringAttr("SAME") : builder.getStringAttr("VALID")));
  attrs.push_back(builder.getNamedAttr("stride_h", builder.getI32IntegerAttr(stride[0])));
  attrs.push_back(builder.getNamedAttr("stride_w", builder.getI32IntegerAttr(stride[1])));
  attrs.push_back(builder.getNamedAttr("fused_activation_function", builder.getStringAttr("NONE")));

  if (is_average_pooling) {
    auto op = OpBuilder(block).create<tpu::AveragePool2DOp>(
        builder.getUnknownLoc(), result_type, ArrayRef<Value *>{input_var},
        ArrayRef<NamedAttribute>{attrs});
    auto result_var = op.getResult();
    return result_var;
  } else {
    auto op = OpBuilder(block).create<tpu::MaxPool2DOp>(
        builder.getUnknownLoc(), result_type, ArrayRef<Value *>{input_var},
        ArrayRef<NamedAttribute>{attrs});
    auto result_var = op.getResult();
    return result_var;
  }
}

static mlir::Value *addFullyConnectedOpInBlockFromCaffe(Builder builder, Block *block,
    mlir::Type elementType, mlir::Value *input_var,
    caffe::Layer<float> *layer,
    mlir::Value *weight_var,
    llvm::raw_fd_ostream *weight_os = NULL) {
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
  DEBUG_WITH_TYPE(DEBUG_TYPE"_VERBOSE", input_var->getType().dump());
  llvm::ArrayRef<int64_t> input_shape = input_var->getType().dyn_cast<mlir::TensorType>().getShape();
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
    auto fc_input_type = builder.getTensorType({M, K}, elementType);
    auto reshape_op = OpBuilder(block).create<tpu::ReshapeOp>(
        builder.getUnknownLoc(), fc_input_type, input_var);
    fc_input_var = reshape_op.getResult();
  }

  size_t pos_filter, pos_bias;
  if (weight_os) {
    // - blobs_[0] holds the filter weights
    // - blobs_[1] holds the biases (optional)
    assert(layer->blobs().size() == 1 || layer->blobs().size() == 2);

    // filter weights
    const caffe::Blob<float> *blob_filter = layer->blobs()[0].get();
    llvm::errs() << "  filter shape " << blob_filter->shape_string() << "\n";
    std::vector<int> filter_shape = blob_filter->shape();
    assert(filter_shape.size() == 2);
    assert(filter_shape[0] == N);
    assert(filter_shape[1] == K);
    pos_filter = weight_os->tell();
    weight_os->write(reinterpret_cast<const char*>(blob_filter->cpu_data()),
        blob_filter->count() * sizeof(float));
    llvm::errs() << "  filter: " << llvm::format_hex(pos_filter, 10)
                 << " --> " << llvm::format_hex(weight_os->tell(), 10) << "\n";

    // bias
    if (with_bias) {
      assert(layer->blobs().size() == 2);
      const caffe::Blob<float> *blob_bias = layer->blobs()[1].get();
      llvm::errs() << "  bias shape " << blob_bias->shape_string() << "\n";
      std::vector<int> bias_shape = blob_bias->shape();
      assert(bias_shape.size() == 1);
      assert(bias_shape[0] == N);
      pos_bias = weight_os->tell();
      weight_os->write(reinterpret_cast<const char*>(blob_bias->cpu_data()),
          blob_bias->count() * sizeof(float));
      llvm::errs() << "  bias: " << llvm::format_hex(pos_bias, 10)
                   << " --> " << llvm::format_hex(weight_os->tell(), 10)
                   << "\n";
    } else {
      assert(layer->blobs().size() == 1);
    }
  }

  // construct the fully_connected OP
  std::vector<Value *> operands;
  operands.push_back(fc_input_var);
  auto filter_type = builder.getTensorType({N, K}, elementType);
  std::vector<NamedAttribute> filter_attrs;
  filter_attrs.push_back(builder.getNamedAttr("offset", builder.getI64IntegerAttr(pos_filter)));
  auto filter = OpBuilder(block).create<tpu::LoadWeightOp>(
      builder.getUnknownLoc(), filter_type,
      ArrayRef<Value *>{weight_var}, ArrayRef<NamedAttribute>{filter_attrs});
  operands.push_back(filter);
  if (with_bias) {
    auto bias_type = builder.getTensorType({N}, elementType);
    std::vector<NamedAttribute> bias_attrs;
    bias_attrs.push_back(builder.getNamedAttr("offset", builder.getI64IntegerAttr(pos_bias)));
    auto bias = OpBuilder(block).create<tpu::LoadWeightOp>(
        builder.getUnknownLoc(), bias_type,
        ArrayRef<Value *>{weight_var}, ArrayRef<NamedAttribute>{bias_attrs});
    operands.push_back(bias);
  }
  auto result_type = builder.getTensorType({M, N}, elementType);
  auto op = OpBuilder(block).create<tpu::FullyConnectedOp>(
        builder.getUnknownLoc(), result_type,
        ArrayRef<Value *>{operands}, ArrayRef<NamedAttribute>{});
  auto result_var = op.getResult();
  return result_var;
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

using google::protobuf::io::FileInputStream;
using google::protobuf::Message;

static bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  assert((fd != -1) && "File not found");
  FileInputStream* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

// Translate CaffeModel in the file named as `inputFilename` and returns a
// module in TPU Dialect.
static OwningModuleRef caffeToMlirTranslate(llvm::StringRef inputFilename,
    llvm::StringRef caffemodelFilename, llvm::StringRef quantTableFilename,
    uint weightAlign, MLIRContext *context) {
  // extract quant table
  NetCalibrationParameter calib_param;
  bool ret = ReadProtoFromTextFile(quantTableFilename.str().c_str(), &calib_param);
  assert(ret == true);
  //map<string, float> blob_threshold;
  for (int i = 0; i < calib_param.layer_size(); i++) {
    const LayerCalibrationParameter *layer_param = &calib_param.layer(i);
    llvm::errs()
        << "layer " << layer_param->name()
        << ", threshold_y size: " << layer_param->threshold_y_size()
        << ", threshold_y[0]: " << (layer_param->threshold_y_size() ? layer_param->threshold_y(0) : 0)
        << ", right_shift_width: " << (layer_param->has_right_shift_width() ? layer_param->right_shift_width() : 0)
        << ", threshold_x_quantized size: " << layer_param->threshold_x_quantized_size()
        << ", threshold_x_quantized[0]: " << (layer_param->threshold_x_quantized_size() ? layer_param->threshold_x_quantized(0) : 0)
        << ", fusion_skipped: " << (layer_param->has_fusion_skipped() ? layer_param->fusion_skipped() : 0)
        << "\n";
  }

  // builder and module
  Builder builder(context);
  OwningModuleRef module(ModuleOp::create(
      FileLineColLoc::get(inputFilename, /*line=*/0, /*column=*/0, context)));
  mlir::Type elementType = mlir::FloatType::getF32(builder.getContext());

  // init caffe net
  //caffe::NetParameter param;
  //caffe::ReadNetParamsFromTextFileOrDie(inputFilename, &param);
  //param.mutable_state()->set_phase(caffe::TEST);
  //caffe::Net<float> net(param);
  caffe::Net<float> net(inputFilename, caffe::TEST);
  net.CopyTrainedLayersFrom(caffemodelFilename);

  auto weightFilename = llvm::sys::path::stem(caffemodelFilename).str() + ".weight";
  //std::vector<int> v(100) ; // vector with 100 ints.
  //std::iota (std::begin(v), std::end(v), 0);

  std::error_code ec;
  llvm::raw_fd_ostream weight_os(weightFilename, ec);
  //weight_os.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(int));
  //llvm::errs() << "file pos " << weight_os.tell() << "\n";

  //std::ofstream os;
  //os.open(weightFilename.c_str(), std::ios::out | std::ios::binary);
  //os.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(int));
  //os.close();

  //auto weight_file = openOutputFile(weightFilename);
  //assert(weight_file);
  //weight_file->os().write(reinterpret_cast<char *>(v.data()), v.size() * sizeof(int));
  //llvm::errs() << "file pos " << weight_file->os().tell() << "\n";
  //weight_file->keep();

  // dump all layers
  DEBUG_WITH_TYPE(DEBUG_TYPE"_VERBOSE", printCaffeNetAllLayer(net););

  auto weightTensorFilename = llvm::sys::path::stem(caffemodelFilename).str() + ".npz";
  auto weightTensorFile = openOutputTensorFile(weightTensorFilename);

  // find caffe model inputs and outputs
  std::vector<mlir::Type> net_input_type_vec;
  std::vector<std::string> net_input_name_vec;
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
    net_input_type_vec.push_back(getMlirTypeFromCaffeShape(builder,
        net.input_blobs()[i]->shape(), elementType));
    net_input_name_vec.push_back(net.blob_names()[index]);
  }
  std::vector<mlir::Type> net_output_type_vec;
  std::vector<std::string> net_output_name_vec;
  for (int i = 0; i <= net.num_outputs() - 1; ++i) {
    int index = net.output_blob_indices()[i];
    LLVM_DEBUG(
      llvm::errs()
          << "net output[" << i << "] - [" << index << "] : "
          << "blob: " << net.blob_names()[index]
          << ", shape: " << net.output_blobs()[i]->shape_string()
          << ", layer: " << net.layer_names()[index]
          << "\n";
    );
    net_output_type_vec.push_back(getMlirTypeFromCaffeShape(builder,
        net.output_blobs()[i]->shape(), elementType));
    net_output_name_vec.push_back(net.blob_names()[index]);
  }

  // create Function Op with arguments and returns type
  llvm::ArrayRef<mlir::Type> func_arg_type(net_input_type_vec);
  llvm::ArrayRef<mlir::Type> func_ret_type(net_output_type_vec);
  Block *block = createOneBlockFunction(builder, module.get(),
      func_arg_type, func_ret_type);

  // we can only handle one input for now, get the function input variable
  assert(func_arg_type.size() == 1);
  mlir::Value *func_arg0_var = block->getArgument(0);

  auto weight_type = builder.getMemRefType({0x80000000}, elementType);
  auto weight_attr = builder.getStringAttr(weightFilename);
  auto weight_var = OpBuilder(block).create<tpu::LoadFileOp>(
      builder.getUnknownLoc(), weight_type, weight_attr);

  // create a map for mapping blob_name and a mlir tensor value
  std::map<std::string, mlir::Value *> tensor_map;

  // convert layers
  for (size_t i = 0; i <= net.layers().size() - 1; ++i) {
    auto layer = net.layers()[i].get();
    auto layer_param = layer->layer_param();
    printCaffeLayerParam(layer);

    if (strcmp(layer->type(), "Input") == 0) {
      assert(layer_param.bottom_size() == 0 && layer_param.top_size() == 1);
      tensor_map[layer_param.top(0)] = func_arg0_var;
      // sanity check
      assert(net_input_name_vec.size() == 1);
      assert(net_input_name_vec[0].compare(layer_param.top(0)) == 0);

    } else if (strcmp(layer->type(), "Split") == 0) {
      assert(layer_param.bottom_size() == 1 && layer_param.top_size() == 2);
      mlir::Value *input_var = tensor_map.find(layer_param.bottom(0))->second;
      assert(input_var);
      // bypass, by registering top and bottom blob_name to the same mlir tensor
      tensor_map[layer_param.top(0)] = input_var;
      tensor_map[layer_param.top(1)] = input_var;

    } else if (strcmp(layer->type(), "Convolution") == 0) {
      assert(layer_param.bottom_size() == 1 && layer_param.top_size() == 1);
      mlir::Value *input_var = tensor_map.find(layer_param.bottom(0))->second;
      assert(input_var);
      mlir::Value *result_var = addConv2dOpInBlockFromCaffe(builder, block,
          elementType, input_var, layer, weight_var, &weight_os, weightTensorFile.get());
      tensor_map[layer_param.top(0)] = result_var;

    } else if (strcmp(layer->type(), "BatchNorm") == 0) {
      assert(layer_param.bottom_size() == 1 && layer_param.top_size() == 1);
      mlir::Value *input_var = tensor_map.find(layer_param.bottom(0))->second;
      assert(input_var);
      mlir::Value *result_var = addBatchNormOpInBlockFromCaffe(builder, block,
          elementType, input_var, layer, weight_var, &weight_os);
      tensor_map[layer_param.top(0)] = result_var;

    } else if (strcmp(layer->type(), "Scale") == 0) {
      assert(layer_param.bottom_size() == 1 && layer_param.top_size() == 1);
      mlir::Value *input_var = tensor_map.find(layer_param.bottom(0))->second;
      assert(input_var);
      mlir::Value *result_var = addScaleOpInBlockFromCaffe(builder, block,
          elementType, input_var, layer, weight_var, &weight_os);
      tensor_map[layer_param.top(0)] = result_var;

    } else if (strcmp(layer->type(), "ReLU") == 0) {
      assert(layer_param.bottom_size() == 1 && layer_param.top_size() == 1);
      mlir::Value *input_var = tensor_map.find(layer_param.bottom(0))->second;
      assert(input_var);
      mlir::Value *result_var = addReluOpInBlockFromCaffe(builder, block,
          elementType, input_var, layer);
      tensor_map[layer_param.top(0)] = result_var;

    } else if (strcmp(layer->type(), "Eltwise") == 0) {
      assert(layer_param.bottom_size() == 2 && layer_param.top_size() == 1);
      mlir::Value *input_1_var = tensor_map.find(layer_param.bottom(0))->second;
      mlir::Value *input_2_var = tensor_map.find(layer_param.bottom(1))->second;
      assert(input_1_var && input_2_var);
      mlir::Value *result_var = addEltwiseOpInBlockFromCaffe(builder, block,
          elementType, input_1_var, input_2_var, layer);
      tensor_map[layer_param.top(0)] = result_var;

    } else if (strcmp(layer->type(), "Pooling") == 0) {
      assert(layer_param.bottom_size() == 1 && layer_param.top_size() == 1);
      mlir::Value *input_var = tensor_map.find(layer_param.bottom(0))->second;
      assert(input_var);
      mlir::Value *result_var = addPoolingOpInBlockFromCaffe(builder, block,
          elementType, input_var, layer);
      tensor_map[layer_param.top(0)] = result_var;

    } else if (strcmp(layer->type(), "InnerProduct") == 0) {
      assert(layer_param.has_inner_product_param());
      assert(layer_param.bottom_size() == 1 && layer_param.top_size() == 1);
      mlir::Value *input_var = tensor_map.find(layer_param.bottom(0))->second;
      assert(input_var);
      mlir::Value *result_var = addFullyConnectedOpInBlockFromCaffe(builder, block,
          elementType, input_var, layer, weight_var, &weight_os);
      tensor_map[layer_param.top(0)] = result_var;

    } else if (strcmp(layer->type(), "Softmax") == 0) {
      llvm::errs() << "    SKIP" << "\n";
      assert(layer_param.bottom_size() == 1 && layer_param.top_size() == 1);
      mlir::Value *input_var = tensor_map.find(layer_param.bottom(0))->second;
      assert(input_var);
      mlir::Value *result_var = addSoftmaxOpInBlockFromCaffe(builder, block,
          elementType, input_var, layer);
      tensor_map[layer_param.top(0)] = result_var;

    } else {
      llvm::errs() << "    UNKNOWN" << "\n";
      assert(false);
    }
  }

  // find the result by looking up tensor_map for the networt output blob_name
  // support only one output for now
  assert(func_ret_type.size() == 1);
  assert(net_output_name_vec.size() == 1);
  mlir::Value *func_ret0_var = tensor_map.find(net_output_name_vec[0])->second;
  assert(func_ret0_var);

  // 4. return Op
  llvm::ArrayRef<mlir::Value *> func_ret_var = {func_ret0_var};
  OpBuilder(block).create<ReturnOp>(builder.getUnknownLoc(), func_ret_var);

  // handle weight
  llvm::errs() << caffemodelFilename << ", align " << weightAlign << "\n";
  llvm::errs() << weightFilename << "\n";
  return module;
}

static llvm::cl::OptionCategory clOptionsCategory("caffe int8 translate options");

static llvm::cl::opt<std::string> clCaffeModelFilename(
    "caffe-model-int8",
    llvm::cl::desc("Specify the caffemodel filename"),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<std::string> clQuantTableFilename(
    "quant-table",
    llvm::cl::desc("Specify the quantization table filename"),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<uint> clWeightAlign(
    "weight-align-int8",
    llvm::cl::desc("Specify the alignment for each weight"),
    llvm::cl::init(32), llvm::cl::cat(clOptionsCategory));

static TranslateToMLIRRegistration
    registration("caffe-to-mlir-v3",
                 [](StringRef inputFilename, MLIRContext *context) {
                   return caffeToMlirTranslate(inputFilename, clCaffeModelFilename,
                       clQuantTableFilename, clWeightAlign, context);
                 });
