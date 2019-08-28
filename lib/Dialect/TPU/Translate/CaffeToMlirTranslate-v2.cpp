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
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
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

#define calcConv2DSpatialOutput(_i_, _k_, _s_, _p_, _d_) \
    (((_i_) + 2 * (_p_) - (_d_) * ((_k_) - 1) - 1) / (_s_) + 1)

// because there is no base class for ConvolutionParameter, PoolingParameter
// use macro to handle Kernel, Stride, Pad, and Dilation
// unfortunately, PoolingParameter has slightly different structions
// in the end, we have to implement separate set of macros for Pooling
#define getKernelSizeFromCaffeParam(_k_, _param_) \
do { \
  const int _num_spatial_axes_ = 2; \
  if (_param_.has_kernel_h() && _param_.has_kernel_w()) { \
    assert(_param_.kernel_size_size() == 0); \
    _k_.push_back(_param_.kernel_h()); \
    _k_.push_back(_param_.kernel_w()); \
  } else { \
    const int _num_kernel_dims_ = _param_.kernel_size_size(); \
    for (int i = 0; i < _num_spatial_axes_; ++i) { \
      _k_.push_back(_param_.kernel_size((_num_kernel_dims_ == 1) ? 0 : i)); \
    } \
  } \
} while(0)

#define getStrideFromCaffeParam(_s_, _param_) \
do { \
  const int _num_spatial_axes_ = 2; \
  if (_param_.has_stride_h() && _param_.has_stride_w()) { \
    assert(_param_.stride_size() == 0); \
    _s_.push_back(_param_.stride_h()); \
    _s_.push_back(_param_.stride_w()); \
  } else { \
    const int _num_stride_dims_ = _param_.stride_size(); \
    for (int i = 0; i < _num_spatial_axes_; ++i) { \
      _s_.push_back(_param_.stride((_num_stride_dims_ == 1) ? 0 : i)); \
    } \
  } \
} while(0)

#define getPadFromCaffeParam(_p_, _param_) \
do { \
  const int _num_spatial_axes_ = 2; \
  if (_param_.has_pad_h() && _param_.has_pad_w()) { \
    assert(_param_.pad_size() == 0); \
    _p_.push_back(_param_.pad_h()); \
    _p_.push_back(_param_.pad_w()); \
  } else { \
    const int _num_pad_dims_ = _param_.pad_size(); \
    const int kDefaultPad = 0; \
    for (int i = 0; i < _num_spatial_axes_; ++i) { \
      _p_.push_back((_num_pad_dims_ == 0) ? kDefaultPad : \
          _param_.pad((_num_pad_dims_ == 1) ? 0 : i)); \
    } \
  } \
} while(0)

#define getDilationFromCaffeParam(_d_, _param_) \
do { \
  const int _num_spatial_axes_ = 2; \
  const int _num_dilation_dims_ = _param_.dilation_size(); \
  const int kDefaultDilation = 1; \
  for (int i = 0; i < _num_spatial_axes_; ++i) { \
    _d_.push_back((_num_dilation_dims_ == 0) ? kDefaultDilation : \
        _param_.dilation((_num_dilation_dims_ == 1) ? 0 : i)); \
  } \
} while(0)

// unfortunately, PoolingParameter has slightly different structions
#define getKernelSizeFromCaffeParam_Pooling(_k_, _param_) \
do { \
  if (_param_.has_kernel_h() && _param_.has_kernel_w()) { \
    assert(!_param_.has_kernel_size()); \
    _k_.push_back(_param_.kernel_h()); \
    _k_.push_back(_param_.kernel_w()); \
  } else { \
    assert(_param_.has_kernel_size()); \
    _k_.push_back(_param_.kernel_size()); \
    _k_.push_back(_param_.kernel_size()); \
  } \
} while(0)

#define getStrideFromCaffeParam_Pooling(_s_, _param_) \
do { \
  if (_param_.has_stride_h() && _param_.has_stride_w()) { \
    assert(!_param_.has_stride()); \
    _s_.push_back(_param_.stride_h()); \
    _s_.push_back(_param_.stride_w()); \
  } else { \
    assert(_param_.has_stride()); \
    _s_.push_back(_param_.stride()); \
    _s_.push_back(_param_.stride()); \
  } \
} while(0)

#define getPadFromCaffeParam_Pooling(_p_, _param_) \
do { \
  if (_param_.has_pad_h() && _param_.has_pad_w()) { \
    assert(!_param_.has_pad() == 0); \
    _p_.push_back(_param_.pad_h()); \
    _p_.push_back(_param_.pad_w()); \
  } else { \
    _p_.push_back(_param_.pad()); \
    _p_.push_back(_param_.pad()); \
  } \
} while(0)

//===----------------------------------------------------------------------===//
// Create Operations from Caffe Proto, and extract weight
//===----------------------------------------------------------------------===//
static mlir::Value *addConv2dOpInBlockFromCaffe(Builder builder, Block *block,
    mlir::Type elementType, mlir::Value *input_var,
    caffe::Layer<float> *layer,
    llvm::raw_fd_ostream *weight_os = NULL) {
  auto layer_param = layer->layer_param();
  auto conv_param = layer_param.convolution_param();
  bool with_bias = conv_param.bias_term();
  int64_t n, ic, oc, group;
  std::vector<int64_t> k, s, p, d;
  std::vector<int64_t> ifmap, ofmap;  // spatial dims only (height and width)

  getKernelSizeFromCaffeParam(k, conv_param);
  getStrideFromCaffeParam(s, conv_param);
  getPadFromCaffeParam(p, conv_param);
  getDilationFromCaffeParam(d, conv_param);
  oc = conv_param.num_output();
  group = conv_param.group();
  // TODO: don't support group for now
  assert(group == 1);

  // get input shape from input var
  LLVM_DEBUG(input_var->getType().dump(););
  llvm::ArrayRef<int64_t> input_var_shape =
      input_var->getType().dyn_cast<mlir::TensorType>().getShape();
  assert(input_var_shape.size() == 4);
  n = input_var_shape[0];
  ic = input_var_shape[1];
  ifmap.push_back(input_var_shape[2]);
  ifmap.push_back(input_var_shape[3]);

  // get ofmap shape by inference
  // does not support dilation for now
  ofmap.push_back(calcConv2DSpatialOutput(ifmap[0], k[0], s[0], p[0], d[0]));
  ofmap.push_back(calcConv2DSpatialOutput(ifmap[1], k[1], s[1], p[1], d[1]));

  llvm::errs()
      << "  N: " << n
      << ", IC: " << ic
      << ", IH*IW: " << ifmap[0] << " * " << ifmap[1]
      << ", OC: " << oc
      << ", OH*OW: " << ofmap[0] << " * " << ofmap[1]
      << "\n";

  llvm::errs()
      << "  with_bias: " << with_bias
      << ", K: " << k[0] << " * " << k[1]
      << ", S: " << s[0] << " * " << s[1]
      << ", P: " << p[0] << " * " << p[1]
      << ", D: " << d[0] << " * " << d[1]
      << ", group: " << group
      << "\n";

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
    assert(filter_shape[2] == k[0]);
    assert(filter_shape[3] == k[1]);
    pos_filter = weight_os->tell();
    weight_os->write(reinterpret_cast<const char*>(blob_filter->cpu_data()), blob_filter->count() * sizeof(float));
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
      weight_os->write(reinterpret_cast<const char*>(blob_bias->cpu_data()), blob_bias->count() * sizeof(float));
      llvm::errs() << "  bias: " << llvm::format_hex(pos_bias, 10)
                   << " --> " << llvm::format_hex(weight_os->tell(), 10) << "\n";
    } else {
      assert(layer->blobs().size() == 1);
    }
  }

  // construct OP
  auto filter_type = builder.getTensorType({oc, ic, k[0], k[1]}, elementType);
  auto filter_attr = builder.getZeroAttr(filter_type);
  auto filter = OpBuilder(block).create<ConstantOp>(builder.getUnknownLoc(),
      filter_type, filter_attr);
  #if 0 // TODO: don't know how to handle optional operand
  mlir::Value *bias = nullptr;
  if (with_bias) {
    auto bias_type = builder.getTensorType({oc}, elementType);
    auto bias_attr = builder.getZeroAttr(bias_type);
    bias = OpBuilder(block).create<ConstantOp>(builder.getUnknownLoc(),
        bias_type, bias_attr);
  }
  #else // # if 0
  auto bias_type = builder.getTensorType({oc}, elementType);
  auto bias_attr = builder.getZeroAttr(bias_type);
  auto bias = OpBuilder(block).create<ConstantOp>(builder.getUnknownLoc(),
      bias_type, bias_attr);
  #endif // # if 0
  auto result_type = builder.getTensorType({n, oc, ofmap[0], ofmap[1]}, elementType);
  auto op = OpBuilder(block).create<tpu::Conv2DOp>(
      builder.getUnknownLoc(), result_type, input_var, filter, bias,
      /*dilation_h_factor=*/builder.getI32IntegerAttr(d[0]),
      /*dilation_w_factor=*/builder.getI32IntegerAttr(d[1]),
      /*fused_activation_function=*/builder.getStringAttr("NONE"),
      /*padding=*/(p[0] || p[1]) ? builder.getStringAttr("SAME") : builder.getStringAttr("VALID"),
      /*stride_h=*/builder.getI32IntegerAttr(s[0]),
      /*stride_w=*/builder.getI32IntegerAttr(s[1]));
  auto result_var = op.getResult();
  return result_var;
}

static mlir::Value *addPoolingOpInBlockFromCaffe(Builder builder, Block *block,
    mlir::Type elementType, mlir::Value *input_var,
    const caffe::PoolingParameter& pooling_param) {
  //auto pooling_param = layer_param.pooling_param();

  bool is_average_pooling;
  bool is_global_pooling;
  int64_t n, c;
  std::vector<int64_t> k, s, p;
  std::vector<int64_t> ifmap, ofmap;  // spatial dims only (height and width)

  if (pooling_param.pool() == caffe::PoolingParameter_PoolMethod_AVE) {
    is_average_pooling = true;
  } else if (pooling_param.pool() == caffe::PoolingParameter_PoolMethod_MAX) {
    is_average_pooling = false;
  } else {
    assert(false);
  }

  // get input shape from input var
  LLVM_DEBUG(input_var->getType().dump(););
  llvm::ArrayRef<int64_t> input_var_shape =
      input_var->getType().dyn_cast<mlir::TensorType>().getShape();
  assert(input_var_shape.size() == 4);
  n = input_var_shape[0];
  c = input_var_shape[1];
  ifmap.push_back(input_var_shape[2]);
  ifmap.push_back(input_var_shape[3]);

  is_global_pooling = pooling_param.global_pooling();
  if (is_global_pooling) {
    k.push_back(ifmap[0]);
    k.push_back(ifmap[1]);
  } else {
    getKernelSizeFromCaffeParam_Pooling(k, pooling_param);
  }
  getStrideFromCaffeParam_Pooling(s, pooling_param);
  getPadFromCaffeParam_Pooling(p, pooling_param);

  // get ofmap shape by inference
  ofmap.push_back((ifmap[0] - k[0] + 2 * p[0]) / s[0] + 1);
  ofmap.push_back((ifmap[1] - k[1] + 2 * p[1]) / s[1] + 1);
  if (is_global_pooling) {
    assert(p[0] == 0 && p[1] == 0 && s[0] == 1 && s[1] == 1);
    assert(ofmap[0] == 1 && ofmap[1] == 1);
  }

  llvm::errs()
      << "  N: " << n
      << ", C: " << c
      << ", IH*IW: " << ifmap[0] << " * " << ifmap[1]
      << ", OH*OW: " << ofmap[0] << " * " << ofmap[1]
      << ", type: " << (is_average_pooling ? "AVG" : "MAX")
      << "\n";

  llvm::errs()
      << "  K: " << k[0] << " * " << k[1]
      << ", S: " << s[0] << " * " << s[1]
      << ", P: " << p[0] << " * " << p[1]
      << ", global_pooling: " << is_global_pooling
      << "\n";

  // construct OP
  auto result_type = builder.getTensorType({n, c, ofmap[0], ofmap[1]}, elementType);
  if (is_average_pooling) {
    auto op = OpBuilder(block).create<tpu::AveragePool2DOp>(
        builder.getUnknownLoc(), result_type, input_var,
        /*filter_height=*/builder.getI32IntegerAttr(k[0]),
        /*filter_width=*/builder.getI32IntegerAttr(k[1]),
        /*padding=*/(p[0] || p[1]) ? builder.getStringAttr("SAME") : builder.getStringAttr("VALID"),
        /*stride_h=*/builder.getI32IntegerAttr(s[0]),
        /*stride_w=*/builder.getI32IntegerAttr(s[1]),
        /*fused_activation_function=*/builder.getStringAttr("NONE"));
    auto result_var = op.getResult();
    return result_var;
  } else {
    auto op = OpBuilder(block).create<tpu::MaxPool2DOp>(
        builder.getUnknownLoc(), result_type, input_var,
        /*filter_height=*/builder.getI32IntegerAttr(k[0]),
        /*filter_width=*/builder.getI32IntegerAttr(k[1]),
        /*padding=*/(p[0] || p[1]) ? builder.getStringAttr("SAME") : builder.getStringAttr("VALID"),
        /*stride_h=*/builder.getI32IntegerAttr(s[0]),
        /*stride_w=*/builder.getI32IntegerAttr(s[1]),
        /*fused_activation_function=*/builder.getStringAttr("NONE"));
    auto result_var = op.getResult();
    return result_var;
  }
}

static mlir::Value *addFullyConnectedOpInBlockFromCaffe(Builder builder, Block *block,
    mlir::Type elementType, mlir::Value *input_var,
    const caffe::InnerProductParameter& fc_param) {
  //auto fc_param = layer_param.inner_product_param();

  bool with_bias, with_transpose;
  // M is the batch_size, K is input number, N is output number
  // (M, K) * (K, N) => (M, N)
  int64_t M, K, N;

  with_bias = fc_param.bias_term();
  with_transpose = fc_param.transpose();
  // N is the output num
  N = fc_param.num_output();

  // get input shape from input var
  LLVM_DEBUG(input_var->getType().dump(););
  llvm::ArrayRef<int64_t> input_var_shape =
      input_var->getType().dyn_cast<mlir::TensorType>().getShape();

  bool reshape_first = false;
  if (input_var_shape.size() == 2) {
    M = input_var_shape[0];
    K = input_var_shape[1];
  } else {
    reshape_first = true;
    M = input_var_shape[0];
    K = 1;
    for (size_t i = 1; i <= input_var_shape.size() - 1; ++i) {
      K *= input_var_shape[i];
    }
  }

  llvm::errs()
      << "  M: " << M
      << ", K: " << K
      << ", N: " << N
      << ", with_bias: " << with_bias
      << ", with_transpose: " << with_transpose
      << "\n";

  // not support transpose for now
  assert(!with_transpose);

  mlir::Value *fc_input_var = input_var;
  // construct reshape OP
  if (reshape_first) {
    auto fc_input_type = builder.getTensorType({M, K}, elementType);
    auto reshape_op = OpBuilder(block).create<tpu::ReshapeOp>(
        builder.getUnknownLoc(), fc_input_type, input_var);
    fc_input_var = reshape_op.getResult();
  }

  // construct the fully_connected OP
  auto weight_type = builder.getTensorType({K, N}, elementType);
  auto weight_attr = builder.getZeroAttr(weight_type);
  auto weight = OpBuilder(block).create<ConstantOp>(builder.getUnknownLoc(),
      weight_type, weight_attr);
  #if 0 // TODO: don't how to handle optional operand
  mlir::Value *bias = nullptr;
  if (with_bias) {
    auto bias_type = builder.getTensorType({N}, elementType);
    auto bias_attr = builder.getZeroAttr(bias_type);
    bias = OpBuilder(block).create<ConstantOp>(builder.getUnknownLoc(),
        bias_type, bias_attr);
  }
  #else // # if 0
  auto bias_type = builder.getTensorType({N}, elementType);
  auto bias_attr = builder.getZeroAttr(bias_type);
  auto bias = OpBuilder(block).create<ConstantOp>(builder.getUnknownLoc(),
      bias_type, bias_attr);
  #endif // # if 0
  auto result_type = builder.getTensorType({M, N}, elementType);
  auto op = OpBuilder(block).create<tpu::FullyConnectedOp>(
        builder.getUnknownLoc(), result_type, fc_input_var, weight, bias,
        /*fused_activation_function=*/builder.getStringAttr("NONE"));
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

// Translate CaffeModel in the file named as `inputFilename` and returns a
// module in TPU Dialect.
static OwningModuleRef caffeToMlirTranslate(llvm::StringRef inputFilename,
    llvm::StringRef caffemodelFilename, uint weightAlign, MLIRContext *context) {
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
  std::vector<int> v(100) ; // vector with 100 ints.
  std::iota (std::begin(v), std::end(v), 0);

  std::error_code ec;
  llvm::raw_fd_ostream weight_os(weightFilename, ec);
  weight_os.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(int));
  llvm::errs() << "file pos " << weight_os.tell() << "\n";

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
  LLVM_DEBUG(printCaffeNetAllLayer(net););

  // create a map for mapping blob_name and a mlir tensor value
  std::map<std::string, mlir::Value *> tensor_map;

  // find caffe model inputs and outputs
  std::vector<mlir::Type> net_input_type_vec;
  std::vector<std::string> net_input_name_vec;
  for (int i = 0; i <= net.num_inputs() - 1; ++i) {
    int index = net.input_blob_indices()[i];
    llvm::errs() << "net input [" << i << "] - [" << index << "] : "
        << ", blob: " << net.blob_names()[index]
        << ", shape: " << net.input_blobs()[i]->shape_string()
        << ", layer: " << net.layer_names()[index]
        << "\n";
    net_input_type_vec.push_back(getMlirTypeFromCaffeShape(builder,
        net.input_blobs()[i]->shape(), elementType));
    net_input_name_vec.push_back(net.blob_names()[index]);
  }
  std::vector<mlir::Type> net_output_type_vec;
  std::vector<std::string> net_output_name_vec;
  for (int i = 0; i <= net.num_outputs() - 1; ++i) {
    int index = net.output_blob_indices()[i];
    llvm::errs() << "net output[" << i << "] - [" << index << "] : "
        << "blob: " << net.blob_names()[index]
        << ", shape: " << net.output_blobs()[i]->shape_string()
        << ", layer: " << net.layer_names()[index]
        << "\n";
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
      assert(layer_param.has_convolution_param());
      assert(layer_param.bottom_size() == 1 && layer_param.top_size() == 1);
      mlir::Value *input_var = tensor_map.find(layer_param.bottom(0))->second;
      assert(input_var);
      mlir::Value *result_var = addConv2dOpInBlockFromCaffe(builder, block,
          elementType, input_var, layer, &weight_os);
      tensor_map[layer_param.top(0)] = result_var;

    } else if (strcmp(layer->type(), "BatchNorm") == 0) {
      assert(layer_param.bottom_size() == 1 && layer_param.top_size() == 1);
      mlir::Value *input_var = tensor_map.find(layer_param.bottom(0))->second;
      assert(input_var);
      // TODO: bypass for now
      tensor_map[layer_param.top(0)] = input_var;

    } else if (strcmp(layer->type(), "Scale") == 0) {
      assert(layer_param.bottom_size() == 1 && layer_param.top_size() == 1);
      mlir::Value *input_var = tensor_map.find(layer_param.bottom(0))->second;
      assert(input_var);
      // TODO: bypass for now
      tensor_map[layer_param.top(0)] = input_var;

    } else if (strcmp(layer->type(), "ReLU") == 0) {
      assert(layer_param.bottom_size() == 1 && layer_param.top_size() == 1);
      mlir::Value *input_var = tensor_map.find(layer_param.bottom(0))->second;
      assert(input_var);
      // TODO: bypass for now
      tensor_map[layer_param.top(0)] = input_var;

    } else if (strcmp(layer->type(), "Eltwise") == 0) {
      assert(layer_param.bottom_size() == 2 && layer_param.top_size() == 1);
      mlir::Value *input_1_var = tensor_map.find(layer_param.bottom(0))->second;
      mlir::Value *input_2_var = tensor_map.find(layer_param.bottom(0))->second;
      assert(input_1_var && input_2_var);
      // TODO: bypass for now, use input_1_var
      tensor_map[layer_param.top(0)] = input_1_var;

    } else if (strcmp(layer->type(), "Pooling") == 0) {
      assert(layer_param.has_pooling_param());
      assert(layer_param.bottom_size() == 1 && layer_param.top_size() == 1);
      mlir::Value *input_var = tensor_map.find(layer_param.bottom(0))->second;
      assert(input_var);
      mlir::Value *result_var = addPoolingOpInBlockFromCaffe(builder, block,
          elementType, input_var, layer_param.pooling_param());
      tensor_map[layer_param.top(0)] = result_var;

    } else if (strcmp(layer->type(), "InnerProduct") == 0) {
      assert(layer_param.has_inner_product_param());
      assert(layer_param.bottom_size() == 1 && layer_param.top_size() == 1);
      mlir::Value *input_var = tensor_map.find(layer_param.bottom(0))->second;
      assert(input_var);
      mlir::Value *result_var = addFullyConnectedOpInBlockFromCaffe(builder, block,
          elementType, input_var, layer_param.inner_product_param());
      tensor_map[layer_param.top(0)] = result_var;

    } else if (strcmp(layer->type(), "Softmax") == 0) {
      llvm::errs() << "    SKIP" << "\n";
      assert(layer_param.bottom_size() == 1 && layer_param.top_size() == 1);
      mlir::Value *input_var = tensor_map.find(layer_param.bottom(0))->second;
      assert(input_var);
      // TODO: bypass
      tensor_map[layer_param.top(0)] = input_var;

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

static llvm::cl::OptionCategory clOptionsCategory("caffe translate options");

static llvm::cl::opt<std::string> clCaffeModelFilename(
    "caffe-model",
    llvm::cl::desc("Specify the caffemodel filename"),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<uint> clWeightAlign(
    "weight-align",
    llvm::cl::desc("Specify the alignment for each weight"),
    llvm::cl::init(32), llvm::cl::cat(clOptionsCategory));

static TranslateToMLIRRegistration
    registration("caffe-to-mlir-v2",
                 [](StringRef inputFilename, MLIRContext *context) {
                   return caffeToMlirTranslate(inputFilename, clCaffeModelFilename,
                       clWeightAlign, context);
                 });
