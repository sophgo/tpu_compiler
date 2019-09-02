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
#include "mlir/Dialect/TPU/Interpreter.h"
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

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MemoryBuffer.h"

#include <numeric>
#include <functional>

#define USE_MKLDNN

#ifdef USE_MKLDNN
#include <assert.h>

#include <chrono>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "mkldnn.hpp"

using namespace mkldnn;

using namespace std;

//static memory::dim product(const memory::dims &dims) {
//    return std::accumulate(dims.begin(), dims.end(), (memory::dim)1,
//            std::multiplies<memory::dim>());
//}

static int mkldnn_conv(float *input, float *weight, float *bias,
    float *output, int n, int ic, int ih, int iw, int oc, int oh, int ow,
    int kh, int kw, int sh, int sw, int ph, int pw) {
  using tag = memory::format_tag;
  using dt = memory::data_type;

  engine eng(engine::kind::cpu, 0);
  stream s(eng);

  std::vector<primitive> net;
  std::vector<std::unordered_map<int, memory>> net_args;

  const memory::dim batch = n;
  memory::dims conv_src_tz = { batch, ic, ih, iw };
  memory::dims conv_weights_tz = { oc, ic, kh, kw };
  memory::dims conv_bias_tz = { oc };
  memory::dims conv_dst_tz = { batch, oc, oh, ow };
  memory::dims conv_strides = { sh, sw };
  memory::dims conv_padding = { ph, pw };

  //std::vector<float> conv_src(product(conv_src_tz));
  //std::vector<float> conv_dst(product(conv_dst_tz));
  //std::vector<float> conv_weights(product(conv_weights_tz));
  //std::vector<float> conv_bias(product(conv_bias_tz));

  if (!bias) {
    auto zero_bias = new std::vector<float>(oc, 0.0f);
    bias = zero_bias->data();
  }

  // memory
  auto user_conv_src_memory = memory(
      { { conv_src_tz }, dt::f32, tag::nchw }, eng, input);
  auto user_conv_weights_memory = memory(
      { { conv_weights_tz }, dt::f32, tag::oihw }, eng, weight);
  auto user_conv_bias_memory = memory(
      { { conv_bias_tz }, dt::f32, tag::x }, eng, bias);
  auto user_conv_dst_memory = memory(
      { { conv_dst_tz }, dt::f32, tag::nchw }, eng, output);

  // md
  auto conv_src_md = memory::desc({ conv_src_tz }, dt::f32, tag::any);
  auto conv_bias_md = memory::desc({ conv_bias_tz }, dt::f32, tag::any);
  auto conv_weights_md
      = memory::desc({ conv_weights_tz }, dt::f32, tag::any);
  auto conv_dst_md = memory::desc({ conv_dst_tz }, dt::f32, tag::any);

  // conv desc
  auto conv_desc = convolution_forward::desc(prop_kind::forward_inference,
      algorithm::convolution_direct, conv_src_md, conv_weights_md, conv_bias_md,
      conv_dst_md, conv_strides, conv_padding, conv_padding);
  auto conv_prim_desc = convolution_forward::primitive_desc(conv_desc, eng);

  // do reorder if needed
  auto conv_src_memory = user_conv_src_memory;
  if (conv_prim_desc.src_desc() != user_conv_src_memory.get_desc()) {
    conv_src_memory = memory(conv_prim_desc.src_desc(), eng);
    net.push_back(reorder(user_conv_src_memory, conv_src_memory));
    net_args.push_back({ { MKLDNN_ARG_FROM, user_conv_src_memory },
        { MKLDNN_ARG_TO, conv_src_memory } });
  }
  auto conv_weights_memory = user_conv_weights_memory;
  if (conv_prim_desc.weights_desc() != user_conv_weights_memory.get_desc()) {
    conv_weights_memory = memory(conv_prim_desc.weights_desc(), eng);
    reorder(user_conv_weights_memory, conv_weights_memory)
        .execute(s, user_conv_weights_memory, conv_weights_memory);
  }
  auto conv_bias_memory = user_conv_bias_memory;

  auto conv_dst_memory = memory(conv_prim_desc.dst_desc(), eng);

  net.push_back(convolution_forward(conv_prim_desc));
  net_args.push_back({ { MKLDNN_ARG_SRC, conv_src_memory },
      { MKLDNN_ARG_WEIGHTS, conv_weights_memory },
      { MKLDNN_ARG_BIAS, conv_bias_memory },
      { MKLDNN_ARG_DST, conv_dst_memory } });

  // reorder or copy the output
  if (conv_dst_memory != user_conv_dst_memory) {
    net.push_back(reorder(conv_dst_memory, user_conv_dst_memory));
    net_args.push_back({ { MKLDNN_ARG_FROM, conv_dst_memory },
        { MKLDNN_ARG_TO, user_conv_dst_memory } });
  }

  // run
  assert(net.size() == net_args.size() && "something is missing");
  for (size_t i = 0; i < net.size(); ++i)
      net.at(i).execute(s, net_args.at(i));

  s.wait();

  return 0;
}

static int mkldnn_pool(float *input, float *output,
    int n, int c, int ih, int iw, int oh, int ow,
    int kh, int kw, int sh, int sw, int ph, int pw,
    bool is_avg) {
  using tag = memory::format_tag;
  using dt = memory::data_type;

  engine eng(engine::kind::cpu, 0);
  stream s(eng);

  std::vector<primitive> net;
  std::vector<std::unordered_map<int, memory>> net_args;

  const memory::dim batch = n;
  memory::dims src_tz = { batch, c, ih, iw };
  memory::dims dst_tz = { batch, c, oh, ow };
  memory::dims kernel = { kh, kw };
  memory::dims strides = { sh, sw };
  memory::dims padding = { ph, pw };

  // memory
  auto user_src_memory = memory(
      { { src_tz }, dt::f32, tag::nchw }, eng, input);
  auto user_dst_memory = memory(
      { { dst_tz }, dt::f32, tag::nchw }, eng, output);

  // md
  //auto src_md = memory::desc({ src_tz }, dt::f32, tag::any);
  //auto dst_md = memory::desc({ dst_tz }, dt::f32, tag::any);

  // pool desc
  auto pool_desc = pooling_forward::desc(prop_kind::forward_inference,
      is_avg ? algorithm::pooling_avg : algorithm::pooling_max,
      user_src_memory.get_desc(), user_dst_memory.get_desc(),
      strides, kernel, padding, padding);
  auto prim_desc = pooling_forward::primitive_desc(pool_desc, eng);

  // do reorder if needed
  auto src_memory = user_src_memory;
  if (prim_desc.src_desc() != user_src_memory.get_desc()) {
    src_memory = memory(prim_desc.src_desc(), eng);
    net.push_back(reorder(user_src_memory, src_memory));
    net_args.push_back({ { MKLDNN_ARG_FROM, user_src_memory },
        { MKLDNN_ARG_TO, src_memory } });
  }

  auto dst_memory = memory(prim_desc.dst_desc(), eng);

  net.push_back(pooling_forward(prim_desc));
  net_args.push_back({ { MKLDNN_ARG_SRC, src_memory },
      { MKLDNN_ARG_DST, dst_memory } });

  // reorder or copy the output
  if (dst_memory != user_dst_memory) {
    net.push_back(reorder(dst_memory, user_dst_memory));
    net_args.push_back({ { MKLDNN_ARG_FROM, dst_memory },
        { MKLDNN_ARG_TO, user_dst_memory } });
  }

  // run
  assert(net.size() == net_args.size() && "something is missing");
  for (size_t i = 0; i < net.size(); ++i)
      net.at(i).execute(s, net_args.at(i));

  s.wait();

  return 0;
}

static int mkldnn_ip(float *input, float *weight, float *bias,
    float *output, int m, int k, int n, bool transpose) {
  using tag = memory::format_tag;
  using dt = memory::data_type;

  engine eng(engine::kind::cpu, 0);
  stream s(eng);

  std::vector<primitive> net;
  std::vector<std::unordered_map<int, memory>> net_args;

  memory::dims src_tz = { m, k };
  memory::dims weights_tz = { n, k };
  memory::dims bias_tz = { n };
  memory::dims dst_tz = { m, n };

  if (!bias) {
    auto zero_bias = new std::vector<float>(n, 0.0f);
    bias = zero_bias->data();
  }

  // memory
  auto user_src_memory = memory(
      { { src_tz }, dt::f32, tag::nc }, eng, input);
  auto user_weights_memory = memory(
      { { weights_tz }, dt::f32, tag::oi }, eng, weight);
  auto user_bias_memory = memory(
      { { bias_tz }, dt::f32, tag::x }, eng, bias);
  auto user_dst_memory = memory(
      { { dst_tz }, dt::f32, tag::nc }, eng, output);

  // md
  auto src_md = memory::desc({ src_tz }, dt::f32, tag::any);
  auto weights_md = memory::desc({ weights_tz }, dt::f32, tag::any);
  auto bias_md = memory::desc({ bias_tz }, dt::f32, tag::any);
  auto dst_md = memory::desc({ dst_tz }, dt::f32, tag::any);

  // fc desc
  auto fc_desc = inner_product_forward::desc(prop_kind::forward_inference,
      src_md, weights_md, bias_md, dst_md);
  auto fc_prim_desc = inner_product_forward::primitive_desc(fc_desc, eng);

  // do reorder if needed
  auto src_memory = user_src_memory;
  if (fc_prim_desc.src_desc() != user_src_memory.get_desc()) {
    src_memory = memory(fc_prim_desc.src_desc(), eng);
    net.push_back(reorder(user_src_memory, src_memory));
    net_args.push_back({ { MKLDNN_ARG_FROM, user_src_memory },
        { MKLDNN_ARG_TO, src_memory } });
  }
  auto weights_memory = user_weights_memory;
  if (fc_prim_desc.weights_desc() != user_weights_memory.get_desc()) {
    weights_memory = memory(fc_prim_desc.weights_desc(), eng);
    reorder(user_weights_memory, weights_memory)
        .execute(s, user_weights_memory, weights_memory);
  }
  auto bias_memory = user_bias_memory;

  auto dst_memory = memory(fc_prim_desc.dst_desc(), eng);

  net.push_back(inner_product_forward(fc_prim_desc));
  net_args.push_back({ { MKLDNN_ARG_SRC, src_memory },
      { MKLDNN_ARG_WEIGHTS, weights_memory },
      { MKLDNN_ARG_BIAS, bias_memory },
      { MKLDNN_ARG_DST, dst_memory } });

  // reorder or copy the output
  if (dst_memory != user_dst_memory) {
    net.push_back(reorder(dst_memory, user_dst_memory));
    net_args.push_back({ { MKLDNN_ARG_FROM, dst_memory },
        { MKLDNN_ARG_TO, user_dst_memory } });
  }

  // run
  assert(net.size() == net_args.size() && "something is missing");
  for (size_t i = 0; i < net.size(); ++i)
      net.at(i).execute(s, net_args.at(i));

  s.wait();

  return 0;
}
#endif // #ifdef USE_MKLDNN

#define calcConv2DSpatialOutput(_i_, _k_, _s_, _p_, _d_) \
    (((_i_) + 2 * (_p_) - (_d_) * ((_k_) - 1) - 1) / (_s_) + 1)

static int64_t findPadForSamePadding(int64_t i, int64_t o, int64_t k, int64_t s, int64_t d) {
  //llvm::errs() << "i: " << i << ", o: " << o << ", k: " << k << ", s: " << s << ", d: " << d << "\n";
  if (k == 1) {
    return 0;
  }
  for (int64_t p = 1; p <= k - 1; ++p) {
    if (calcConv2DSpatialOutput(i, k, s, p, d) == o) {
      return p;
    }
  }
  assert(false);
  return 0;
}

namespace mlir {

LogicalResult ModuleInterpreter::runOperation(Operation &opInst) {
  // #include "mlir/Dialect/LLVMIR/LLVMConversions.inc"
  if (auto loadFileOp = dyn_cast<tpu::LoadFileOp>(opInst)) {
    llvm::errs() << "LoadFileOp" << "\n";
    auto filename = loadFileOp.getAttrOfType<StringAttr>("filename").getValue();
    llvm::errs() << "  filename " << filename << "\n";
    weight_is = std::make_unique<std::ifstream>(filename.str(),
        std::ios::in | std::ios::binary);

    return success();
  }
  if (auto loadWeightOp = dyn_cast<tpu::LoadWeightOp>(opInst)) {
    llvm::errs() << "LoadWeightOp" << "\n";
    auto offset = loadWeightOp.offset().getLimitedValue();
    llvm::errs() << "  offset " << offset << "\n";
    auto result = loadWeightOp.getResult();
    llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());;
    auto weight_data = std::make_unique<std::vector<float> >(size);

    weight_is.get()->seekg(offset, std::ios::beg);
    weight_is.get()->read((char*)weight_data.get()->data(), size * sizeof(float));

    valueMapping[result] = std::move(weight_data);

    return success();
  }
  if (auto op = dyn_cast<tpu::Conv2DOp>(opInst)) {
    llvm::errs() << "Conv2DOp" << "\n";
    //op.dump();
    assert(op.getNumOperands() == 3);
    std::vector<std::vector<float> *> operand_tensors;
    unsigned int operandIdx = 0;
    for (auto *operand : op.getOperands()) {
      llvm::errs() << "  operand[" << operandIdx << "] "; operand->getType().dump(); llvm::errs() << "\n";
      // find operand in valueMapping
      auto it = valueMapping.find(operand);
      if (it == valueMapping.end()) {
        llvm::errs() << "    didn't find\n";
        assert(0);
      } else {
        llvm::errs() << "    found in map\n";
        auto vec = it->second.get();
        if (vec) {
          llvm::errs() << "      vec size = " << vec->size() << "\n";
        } else {
          llvm::errs() << "      vec is nullptr\n";
        }
        operand_tensors.push_back(vec);
      }
      operandIdx++;
    }

    auto result = op.getResult();
    llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() == 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());;
    auto result_tensor = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here
    int n, ic, ih, iw, oc, oh, ow, kh, kw, sh, sw, ph, pw, dh, dw;
    dh = op.dilation_h_factor().getLimitedValue();  // APInt, use .getLimitedValue(); to get uint65_t
    dw = op.dilation_w_factor().getLimitedValue();
    sh = op.stride_h().getLimitedValue();
    sw = op.stride_w().getLimitedValue();
    auto input_type = op.input()->getType().cast<TensorType>();
    std::vector<int64_t> i_s(input_type.getShape());
    auto output_type = op.output()->getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());
    auto filter_type = op.filter()->getType().cast<TensorType>();
    std::vector<int64_t> f_s(filter_type.getShape());
    assert((i_s[0] == o_s[0]) && "input N not equal to output N");
    n = i_s[0];
    ih = i_s[2];
    iw = i_s[3];
    oc = f_s[0];
    ic = f_s[1];
    kh = f_s[2];
    kw = f_s[3];
    oh = o_s[2];
    ow = o_s[3];
    auto padding_attr = op.getAttrOfType<StringAttr>("padding");
    if (padding_attr.getValue() == "SAME") {
      ph = findPadForSamePadding(ih, oh, kh, sh, dh);
      pw = findPadForSamePadding(iw, ow, kw, sw, dw);
    } else if (padding_attr.getValue() == "VALID") {
      ph = 0;
      pw = 0;
    } else {
      assert(false);
    }
    float *mkldnn_input = (float *)operand_tensors[0]->data();
    float *mkldnn_weight = (float *)operand_tensors[1]->data();
    float *mkldnn_bias = nullptr;
    if (operand_tensors[2]) {
      mkldnn_bias = (float *)operand_tensors[2]->data();
    }
    float *mkldnn_output = (float *)result_tensor.get()->data();
    int mkldnn_ret = mkldnn_conv(mkldnn_input, mkldnn_weight, mkldnn_bias, mkldnn_output,
        n, ic, ih, iw, oc, oh, ow, kh, kw, sh, sw, ph, pw);
    assert(mkldnn_ret == 0);
    //dump_data_float_abs("mkldnn_output", mkldnn_output, n, oc, oh, ow);
    // TODO: End of compute, need refactor

    valueMapping[result] = std::move(result_tensor);

    return success();
  }
  if (auto op = dyn_cast<tpu::AveragePool2DOp>(opInst)) {
    llvm::errs() << "AveragePool2DOp" << "\n";
    //op.dump();
    //assert(op.getNumOperands() == 1);
    std::vector<std::vector<float> *> operand_tensors;
    {
      auto operand = op.getOperand();
      llvm::errs() << "  operand[0] "; operand->getType().dump(); llvm::errs() << "\n";
      // find operand in valueMapping
      auto it = valueMapping.find(operand);
      if (it == valueMapping.end()) {
        llvm::errs() << "    didn't find\n";
        assert(0);
      } else {
        llvm::errs() << "    found in map\n";
        auto vec = it->second.get();
        if (vec) {
          llvm::errs() << "      vec size = " << vec->size() << "\n";
        } else {
          llvm::errs() << "      vec is nullptr\n";
        }
        operand_tensors.push_back(vec);
      }
    }
    auto result = op.getResult();
    llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());;
    auto result_tensor = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here
    int n, c, ih, iw, oh, ow, kh, kw, sh, sw, ph, pw;
    kh = op.filter_height().getLimitedValue();
    kw = op.filter_width().getLimitedValue();
    sh = op.stride_h().getLimitedValue();
    sw = op.stride_w().getLimitedValue();
    auto input_type = op.input()->getType().cast<TensorType>();
    std::vector<int64_t> i_s(input_type.getShape());
    auto output_type = op.output()->getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());
    assert((i_s[0] == o_s[0]) && "input N not equal to output N");
    assert((i_s[1] == o_s[1]) && "input C not equal to output C");
    n = i_s[0];
    c = i_s[1];
    ih = i_s[2];
    iw = i_s[3];
    oh = o_s[2];
    ow = o_s[3];
    auto padding_attr = op.getAttrOfType<StringAttr>("padding");
    if (padding_attr.getValue() == "SAME") {
      ph = findPadForSamePadding(ih, oh, kh, sh, 1);
      pw = findPadForSamePadding(iw, ow, kw, sw, 1);
    } else if (padding_attr.getValue() == "VALID") {
      ph = 0;
      pw = 0;
    } else {
      assert(false);
    }
    float *mkldnn_input = (float *)operand_tensors[0]->data();
    float *mkldnn_output = (float *)result_tensor.get()->data();
    int mkldnn_ret = mkldnn_pool(mkldnn_input, mkldnn_output,
        n, c, ih, iw, oh, ow, kh, kw, sh, sw, ph, pw, true);
    assert(mkldnn_ret == 0);
    //dump_data_float_abs("mkldnn_output", mkldnn_output, n, c, oh, ow);
    // TODO: End of compute, need refactor

    valueMapping[result] = std::move(result_tensor);

    return success();
  }
  if (auto op = dyn_cast<tpu::MaxPool2DOp>(opInst)) {
    llvm::errs() << "MaxPool2DOp" << "\n";
    //op.dump();
    //assert(op.getNumOperands() == 1);
    std::vector<std::vector<float> *> operand_tensors;
    {
      auto operand = op.getOperand();
      llvm::errs() << "  operand[0] "; operand->getType().dump(); llvm::errs() << "\n";
      // find operand in valueMapping
      auto it = valueMapping.find(operand);
      if (it == valueMapping.end()) {
        llvm::errs() << "    didn't find\n";
        assert(0);
      } else {
        llvm::errs() << "    found in map\n";
        auto vec = it->second.get();
        if (vec) {
          llvm::errs() << "      vec size = " << vec->size() << "\n";
        } else {
          llvm::errs() << "      vec is nullptr\n";
        }
        operand_tensors.push_back(vec);
      }
    }
    auto result = op.getResult();
    llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());;
    auto result_tensor = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here
    int n, c, ih, iw, oh, ow, kh, kw, sh, sw, ph, pw;
    kh = op.filter_height().getLimitedValue();
    kw = op.filter_width().getLimitedValue();
    sh = op.stride_h().getLimitedValue();
    sw = op.stride_w().getLimitedValue();
    auto input_type = op.input()->getType().cast<TensorType>();
    std::vector<int64_t> i_s(input_type.getShape());
    auto output_type = op.output()->getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());
    assert((i_s[0] == o_s[0]) && "input N not equal to output N");
    assert((i_s[1] == o_s[1]) && "input C not equal to output C");
    n = i_s[0];
    c = i_s[1];
    ih = i_s[2];
    iw = i_s[3];
    oh = o_s[2];
    ow = o_s[3];
    auto padding_attr = op.getAttrOfType<StringAttr>("padding");
    if (padding_attr.getValue() == "SAME") {
      ph = findPadForSamePadding(ih, oh, kh, sh, 1);
      pw = findPadForSamePadding(iw, ow, kw, sw, 1);
    } else if (padding_attr.getValue() == "VALID") {
      ph = 0;
      pw = 0;
    } else {
      assert(false);
    }
    float *mkldnn_input = (float *)operand_tensors[0]->data();
    float *mkldnn_output = (float *)result_tensor.get()->data();
    int mkldnn_ret = mkldnn_pool(mkldnn_input, mkldnn_output,
        n, c, ih, iw, oh, ow, kh, kw, sh, sw, ph, pw, false);
    assert(mkldnn_ret == 0);
    //dump_data_float_abs("mkldnn_output", mkldnn_output, n, c, oh, ow);
    // TODO: End of compute, need refactor

    valueMapping[result] = std::move(result_tensor);

    return success();
  }
  if (auto op = dyn_cast<tpu::FullyConnectedOp>(opInst)) {
    llvm::errs() << "FullyConnectedOp" << "\n";
    //op.dump();
    assert(op.getNumOperands() == 3);
    std::vector<std::vector<float> *> operand_tensors;
    unsigned int operandIdx = 0;
    for (auto *operand : op.getOperands()) {
      llvm::errs() << "  operand[" << operandIdx << "] "; operand->getType().dump(); llvm::errs() << "\n";
      // find operand in valueMapping
      auto it = valueMapping.find(operand);
      if (it == valueMapping.end()) {
        llvm::errs() << "    didn't find\n";
        assert(0);
      } else {
        llvm::errs() << "    found in map\n";
        auto vec = it->second.get();
        if (vec) {
          llvm::errs() << "      vec size = " << vec->size() << "\n";
        } else {
          llvm::errs() << "      vec is nullptr\n";
        }
        operand_tensors.push_back(vec);
      }
      operandIdx++;
    }

    auto result = op.getResult();
    llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() == 2);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());;
    auto result_tensor = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here
    int m, k, n;
    bool transpose = false;
    auto input_type = op.input()->getType().cast<TensorType>();
    std::vector<int64_t> i_s(input_type.getShape());
    auto output_type = op.output()->getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());
    auto filter_type = op.filter()->getType().cast<TensorType>();
    std::vector<int64_t> f_s(filter_type.getShape());
    assert((i_s[0] == o_s[0]) && "input M not equal to output M");
    m = i_s[0];
    // assuming transpose is false
    assert((i_s[1] == f_s[1]) && "input K not equal to filter K");
    k = i_s[1];
    assert((f_s[0] == o_s[1]) && "filter N not equal to output N");
    n = o_s[1];

    float *mkldnn_input = (float *)operand_tensors[0]->data();
    float *mkldnn_weight = (float *)operand_tensors[1]->data();
    float *mkldnn_bias = nullptr;
    if (operand_tensors[2]) {
      mkldnn_bias = (float *)operand_tensors[2]->data();
    }
    float *mkldnn_output = (float *)result_tensor.get()->data();
    int mkldnn_ret = mkldnn_ip(mkldnn_input, mkldnn_weight, mkldnn_bias,
        mkldnn_output, m, k, n, transpose);
    assert(mkldnn_ret == 0);
    //dump_data_float_abs("mkldnn_output", mkldnn_output, 1, 1, m, n);
    // TODO: End of compute, need refactor

    valueMapping[result] = std::move(result_tensor);

    return success();
  }
  if (auto op = dyn_cast<tpu::ReluOp>(opInst)) {
    llvm::errs() << "ReluOp" << "\n";
    //op.dump();
    {
      auto operand = op.getOperand();
      llvm::errs() << "  operand[0] "; operand->getType().dump(); llvm::errs() << "\n";
      // find operand in valueMapping
      auto it = valueMapping.find(operand);
      if (it == valueMapping.end()) {
        llvm::errs() << "    didn't find\n";
        assert(0);
      } else {
        llvm::errs() << "    found in map\n";
        auto vec = it->second.get();
        if (vec) {
          llvm::errs() << "      vec size = " << vec->size() << "\n";
        } else {
          llvm::errs() << "      vec is nullptr\n";
        }
      }
    }
    auto result = op.getResult();
    llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());;
    auto result_data = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here
    // TODO: End of compute, need refactor

    valueMapping[result] = std::move(result_data);
    return success();
  }
  if (auto op = dyn_cast<tpu::BatchNormOp>(opInst)) {
    llvm::errs() << "BatchNormOp" << "\n";
    //op.dump();
    unsigned int operandIdx = 0;
    for (auto *operand : op.getOperands()) {
      llvm::errs() << "  operand[" << operandIdx << "] "; operand->getType().dump(); llvm::errs() << "\n";
      // find operand in valueMapping
      auto it = valueMapping.find(operand);
      if (it == valueMapping.end()) {
        llvm::errs() << "    didn't find\n";
        assert(0);
      } else {
        llvm::errs() << "    found in map\n";
        auto vec = it->second.get();
        if (vec) {
          llvm::errs() << "      vec size = " << vec->size() << "\n";
        } else {
          llvm::errs() << "      vec is nullptr\n";
        }
      }
      operandIdx++;
    }
    auto result = op.getResult();
    llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());;
    auto result_data = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here
    // TODO: End of compute, need refactor

    valueMapping[result] = std::move(result_data);
    return success();
  }
  if (auto op = dyn_cast<tpu::ScaleOp>(opInst)) {
    llvm::errs() << "ScaleOp" << "\n";
    //op.dump();
    unsigned int operandIdx = 0;
    for (auto *operand : op.getOperands()) {
      llvm::errs() << "  operand[" << operandIdx << "] "; operand->getType().dump(); llvm::errs() << "\n";
      // find operand in valueMapping
      auto it = valueMapping.find(operand);
      if (it == valueMapping.end()) {
        llvm::errs() << "    didn't find\n";
        assert(0);
      } else {
        llvm::errs() << "    found in map\n";
        auto vec = it->second.get();
        if (vec) {
          llvm::errs() << "      vec size = " << vec->size() << "\n";
        } else {
          llvm::errs() << "      vec is nullptr\n";
        }
      }
      operandIdx++;
    }
    auto result = op.getResult();
    llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());;
    auto result_data = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here
    // TODO: End of compute, need refactor

    valueMapping[result] = std::move(result_data);
    return success();
  }
  if (auto op = dyn_cast<tpu::EltwiseOp>(opInst)) {
    llvm::errs() << "ScaleOp" << "\n";
    //op.dump();
    unsigned int operandIdx = 0;
    for (auto *operand : op.getOperands()) {
      llvm::errs() << "  operand[" << operandIdx << "] "; operand->getType().dump(); llvm::errs() << "\n";
      // find operand in valueMapping
      auto it = valueMapping.find(operand);
      if (it == valueMapping.end()) {
        llvm::errs() << "    didn't find\n";
        assert(0);
      } else {
        llvm::errs() << "    found in map\n";
        auto vec = it->second.get();
        if (vec) {
          llvm::errs() << "      vec size = " << vec->size() << "\n";
        } else {
          llvm::errs() << "      vec is nullptr\n";
        }
      }
      operandIdx++;
    }
    auto result = op.getResult();
    llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());;
    auto result_data = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here
    // TODO: End of compute, need refactor

    valueMapping[result] = std::move(result_data);
    return success();
  }
  if (auto op = dyn_cast<tpu::ReshapeOp>(opInst)) {
    llvm::errs() << "ReshapeOp" << "\n";
    //op.dump();
    {
      auto operand = op.getOperand();
      llvm::errs() << "  operand[0] "; operand->getType().dump(); llvm::errs() << "\n";
      // find operand in valueMapping
      auto it = valueMapping.find(operand);
      if (it == valueMapping.end()) {
        llvm::errs() << "    didn't find\n";
        assert(0);
      } else {
        llvm::errs() << "    found in map\n";
        auto vec = it->second.get();
        if (vec) {
          llvm::errs() << "      vec size = " << vec->size() << "\n";
        } else {
          llvm::errs() << "      vec is nullptr\n";
        }
      }
    }
    auto result = op.getResult();
    llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());;
    auto result_data = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here
    // TODO: End of compute, need refactor

    valueMapping[result] = std::move(result_data);

    return success();
  }

  if (auto op = dyn_cast<ConstantOp>(opInst)) {
    llvm::errs() << "ConstantOp" << "\n";
    //op.dump();
    // TODO: use specific Op for null operand
    // only support zero constant for now
    // TODO: check isZero

    // it it safe to ignore, put null pointer to the valueMapping
    auto result = op.getResult();
    valueMapping[result] = std::move(nullptr);

    return success();
  }

  if (auto op = dyn_cast<ReturnOp>(opInst)) {
    llvm::errs() << "ReturnOp" << "\n";
    //op.dump();
    std::vector<float> *return_vec;
    unsigned int operandIdx = 0;
    for (auto *operand : op.getOperands()) {
      llvm::errs() << "  operand[" << operandIdx << "] "; operand->getType().dump(); llvm::errs() << "\n";
      // find operand in valueMapping
      auto it = valueMapping.find(operand);
      if (it == valueMapping.end()) {
        llvm::errs() << "    didn't find\n";
        assert(0);
      } else {
        llvm::errs() << "    found in map\n";
        return_vec = it->second.get();
        llvm::errs() << "      vec size = " << return_vec->size() << "\n";
      }
      operandIdx++;
    }

    //copy the value into outputs
    assert(outputs.size() == 1);
    outputs[0]->swap(*return_vec);

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
  llvm::errs() << "func " << func.getName() << "\n";
  // Clear the value mappings, it is only relevant within one function.
  valueMapping.clear();

  // Add function arguments to the value remapping table.
  unsigned int argIdx = 0;
  assert(inputs.size() == 1);
  for (auto arg : func.getArguments()) {
    llvm::errs() << "arg " << argIdx << ": ";
    arg->getType().dump();
    llvm::errs() << "\n";

    // copy the inputs[0] into a unique_ptr pointed vector
    // TODO: pass input as unique_ptr directly
    auto input = std::make_unique<std::vector<float> >();
    input.get()->swap(*inputs[0]);
    valueMapping[arg] = std::move(input);
    argIdx++;
  }
  assert(argIdx == 1);

  // Then, convert blocks one by one.
  for (Block &bb : func.getBlocks()) {
    if (failed(runBlock(bb)))
      return failure();
  }

  return success();
}

LogicalResult ModuleInterpreter::runFunctions() {
  for (FuncOp function : mlirModule.getOps<FuncOp>()) {
    llvm::errs() << "run " << function.getName() << "\n";

    if (!function.getName().equals("tpu_func")) {
      //continue;
      assert(0);
    }
    if (failed(runOneFunction(function)))
      return failure();
  }

  return success();
}

LogicalResult runTpuModule(ModuleOp m,
    std::vector<std::vector<float> *> &inputs,
    std::vector<std::vector<float> *> &outputs) {
  return ModuleInterpreter::runModule<>(m, inputs, outputs);
}

} // namespace mlir
