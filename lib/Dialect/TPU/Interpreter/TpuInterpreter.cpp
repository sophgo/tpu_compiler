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
#include "mlir/Support/TensorFile.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Path.h"
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

#define DUMP_FLAG

using namespace mkldnn;

using namespace std;

#ifdef DUMP_FLAG
static size_t write_bianry_file(std::string filename, const char *data,
    size_t size = 0) {
  std::ofstream os;
  os.open(filename.c_str(), std::ios::out | std::ios::binary);
  llvm::errs() << "write " << size << " bytes to " << filename << "\n";
  os.write(data, size);
  os.close();
  return size;
}
#endif // DUMP_FLAG

static int mkldnn_conv(float *input, float *weight, float *bias,
    float *output, int n, int ic, int ih, int iw, int oc, int oh, int ow,
    int kh, int kw, int sh, int sw, int ph, int pw) {
  if (!bias) {
    auto zero_bias = new std::vector<float>(oc, 0.0f);
    bias = zero_bias->data();
  }

#ifdef DUMP_FLAG
  static int conv_idx = 0;
  std::string prefix = std::string("conv") + std::to_string(conv_idx);
  if (conv_idx < 2) {
    write_bianry_file(prefix + std::string("_in.bin"),
        (const char *)input, n * ic * ih * iw * sizeof(float));
    write_bianry_file(prefix + std::string("_filter.bin"),
        (const char *)weight, oc * ic * kh * kw * sizeof(float));
    write_bianry_file(prefix + std::string("_bias.bin"),
        (const char *)bias, oc * sizeof(float));
  }
#endif // DUMP_FLAG
  llvm::errs() << "  k: (" << kh << "*" << kw << "), "
               << "s: (" << sh << "*" << sw << "), "
               << "p: (" << ph << "*" << pw << ")" << "\n";

  using tag = memory::format_tag;
  using dt = memory::data_type;

  engine eng(engine::kind::cpu, 0);
  stream s(eng);

  std::vector<primitive> net;
  std::vector<std::unordered_map<int, memory>> net_args;

  const memory::dim batch = n;
  memory::dims src_tz = { batch, ic, ih, iw };
  memory::dims weights_tz = { oc, ic, kh, kw };
  memory::dims bias_tz = { oc };
  memory::dims dst_tz = { batch, oc, oh, ow };
  memory::dims strides = { sh, sw };
  memory::dims padding = { ph, pw };

  // memory
  auto user_src_memory = memory(
      { { src_tz }, dt::f32, tag::nchw }, eng, input);
  auto user_weights_memory = memory(
      { { weights_tz }, dt::f32, tag::oihw }, eng, weight);
  auto user_bias_memory = memory(
      { { bias_tz }, dt::f32, tag::x }, eng, bias);
  auto user_dst_memory = memory(
      { { dst_tz }, dt::f32, tag::nchw }, eng, output);

  // md
  auto src_md     = memory::desc({ src_tz }, dt::f32, tag::any);
  auto weights_md = memory::desc({ weights_tz }, dt::f32, tag::any);
  auto bias_md    = memory::desc({ bias_tz }, dt::f32, tag::any);
  auto dst_md     = memory::desc({ dst_tz }, dt::f32, tag::any);

  // conv desc
  auto conv_desc = convolution_forward::desc(prop_kind::forward_inference,
      algorithm::convolution_direct, src_md, weights_md, bias_md, dst_md,
      strides, padding, padding);
  auto conv_prim_desc = convolution_forward::primitive_desc(conv_desc, eng);

  // do reorder if needed
  auto src_memory = user_src_memory;
  if (conv_prim_desc.src_desc() != user_src_memory.get_desc()) {
    src_memory = memory(conv_prim_desc.src_desc(), eng);
    net.push_back(reorder(user_src_memory, src_memory));
    net_args.push_back({ { MKLDNN_ARG_FROM, user_src_memory },
        { MKLDNN_ARG_TO, src_memory } });
  }
  auto weights_memory = user_weights_memory;
  if (conv_prim_desc.weights_desc() != user_weights_memory.get_desc()) {
    weights_memory = memory(conv_prim_desc.weights_desc(), eng);
    reorder(user_weights_memory, weights_memory)
        .execute(s, user_weights_memory, weights_memory);
  }
  auto bias_memory = user_bias_memory;

  auto dst_memory = memory(conv_prim_desc.dst_desc(), eng);

  net.push_back(convolution_forward(conv_prim_desc));
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

#ifdef DUMP_FLAG
  if (conv_idx < 2) {
    write_bianry_file(prefix + std::string("_out.bin"),
        (const char *)output, n * oc * oh * ow * sizeof(float));
  }
  conv_idx ++;
#endif // DUMP_FLAG

  return 0;
}

static int mkldnn_pool(float *input, float *output,
    int n, int c, int ih, int iw, int oh, int ow,
    int kh, int kw, int sh, int sw, int ph, int pw,
    bool is_avg) {
  int p_t = ph;
  int p_b = ph;
  int p_l = pw;
  int p_r = pw;
  // Fix padding
  if ( (ih - kh) % sh ) {
    assert(sh == 2);
    assert(oh == static_cast<int>(ceil(static_cast<float>(
        ih + 2 * ph - kh) / sh)) + 1);
    // caffe will pass ph == 0 (padding == "SAME") here
    // by passing ph == 0, caffe actually means
    // p_top = 0, p_bottom = 1
    // if ph == 1 is passed (padding == "SAME")
    // we handle it with the opposite of caffe, i.e.
    // p_top = 1, p_bottom = 0
    if (ph == 0) {
      p_b = 1;
    } else {
      assert(ph == 1);
      p_b = 0;
    }
    assert(ph == 0);  // put a reminder here, just in case we met the case
  }
  if ( (iw - kw) % sw ) {
    assert(sw == 2);
    assert(ow == static_cast<int>(ceil(static_cast<float>(
        iw + 2 * pw - kw) / sw)) + 1);
    // caffe will pass pw == 0 (padding == "SAME") here
    // by passing pw == 0, caffe actually means
    // p_left = 0, p_right = 1
    // if pw == 1 is passed (padding == "SAME")
    // we handle it with the opposite of caffe, i.e.
    // p_left = 1, p_right = 0
    if (pw == 0) {
      p_r = 1;
    } else {
      assert(pw == 1);
      p_r = 0;
    }
    assert(pw == 0);  // put a reminder here, just in case we met the case
  }

#ifdef DUMP_FLAG
  static int dump_idx = 0;
  std::string prefix = std::string("pool") + std::to_string(dump_idx);
  if (dump_idx == 0) {
    write_bianry_file(prefix + std::string("_in.bin"),
        (const char *)input, n * c * ih * iw * sizeof(float));
  }
#endif // DUMP_FLAG
  llvm::errs() << "  k: (" << kh << "*" << kw << "), "
               << "s: (" << sh << "*" << sw << "), "
               << "p: (" << p_t << "-" << p_b
               << "*" << p_l << "-" << p_r << ")" << "\n";

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
  memory::dims padding_t_l = { p_t, p_l };
  memory::dims padding_b_r = { p_b, p_r };

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
      strides, kernel, padding_t_l, padding_b_r);
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

#ifdef DUMP_FLAG
  if (dump_idx == 0) {
    write_bianry_file(prefix + std::string("_out.bin"),
        (const char *)output, n * c * oh * ow * sizeof(float));
  }
  dump_idx ++;
#endif // DUMP_FLAG

  return 0;
}

static int mkldnn_ip(float *input, float *weight, float *bias,
    float *output, int m, int k, int n, bool transpose) {
  if (!bias) {
    auto zero_bias = new std::vector<float>(n, 0.0f);
    bias = zero_bias->data();
  }

#ifdef DUMP_FLAG
  static int dump_idx = 0;
  std::string prefix = std::string("ip") + std::to_string(dump_idx);
  if (dump_idx == 0) {
    write_bianry_file(prefix + std::string("_in.bin"),
        (const char *)input, m * k * sizeof(float));
  }
#endif // DUMP_FLAG

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

#ifdef DUMP_FLAG
  if (dump_idx == 0) {
    write_bianry_file(prefix + std::string("_out.bin"),
        (const char *)output, m * n * sizeof(float));
  }
  dump_idx ++;
#endif // DUMP_FLAG

  return 0;
}
#endif // #ifdef USE_MKLDNN

static int my_relu(float *input, float *output,
    int n, int c, int h, int w, float negative_slope) {
#ifdef DUMP_FLAG
  static int dump_idx = 0;
  std::string prefix = std::string("relu") + std::to_string(dump_idx);
  if (dump_idx < 4) {
    write_bianry_file(prefix + std::string("_in.bin"),
        (const char *)input, n * c * h * w * sizeof(float));
  }
#endif // DUMP_FLAG
  llvm::errs() << "  n: " << n << ", c: " << c
               << ", h: " << h << ", w: " << w << "\n";

  for (int i = 0; i < n * c * h * w; ++i) {
    if (input[i] >= 0) {
      output[i] = input[i];
    } else {
      output[i] = negative_slope * input[i];
    }
  }
#ifdef DUMP_FLAG
  if (dump_idx < 4) {
    write_bianry_file(prefix + std::string("_out.bin"),
        (const char *)output, n * c * h * w * sizeof(float));
  }
  dump_idx ++;
#endif // DUMP_FLAG
  return 0;
}

// Y = (X-mean(X))/(sqrt(var(X)+eps))
static int my_bn(float *input, float *mean, float *variance, float *scale,
    float *output, int n, int c, int h, int w) {
  float eps = 1.0e-5;
  float scale_factor = 1 / scale[0];
  for (int i = 0; i < c; ++i) {
    mean[i] = mean[i] * scale_factor;
    variance[i] = variance[i] * scale_factor;
  }
  for (int ni = 0; ni < n; ++ni) {
    for (int ci = 0; ci < c; ++ci) {
      for (int i = 0; i < h * w; ++i) {
        auto x = input[ni * c * h * w + ci * h * w + i] - mean[ci];
        auto d = sqrt(variance[ci] + eps);
        output[ni * c * h * w + ci * h * w + i] = x / d;
      }
    }
  }
  return 0;
}

static int my_scale(float *input, float *scale, float *bias,
    float *output, int n, int c, int h, int w) {
#ifdef DUMP_FLAG
  static int dump_idx = 0;
  std::string prefix = std::string("scale") + std::to_string(dump_idx);
  if (dump_idx == 1 || dump_idx == 4) {
    write_bianry_file(prefix + std::string("_in.bin"),
        (const char *)input, n * c * h * w * sizeof(float));
  }
#endif // DUMP_FLAG

  for (int ni = 0; ni < n; ++ni) {
    for (int ci = 0; ci < c; ++ci) {
      for (int i = 0; i < h * w; ++i) {
        auto x = input[ni * c * h * w + ci * h * w + i];
        auto y = x * scale[ci];
        if (bias) {
          y += bias[ci];
        }
        output[ni * c * h * w + ci * h * w + i] = y;
      }
    }
  }

#ifdef DUMP_FLAG
  if (dump_idx == 1 || dump_idx == 4) {
    write_bianry_file(prefix + std::string("_out.bin"),
        (const char *)output, n * c * h * w * sizeof(float));
  }
  dump_idx ++;
#endif // DUMP_FLAG
  return 0;
}

#include <math.h>

static int my_softmax(float *input, float *output, int n, int c) {
#ifdef DUMP_FLAG
  static int dump_idx = 0;
  std::string prefix = std::string("softmax") + std::to_string(dump_idx);
  if (dump_idx == 0) {
    write_bianry_file(prefix + std::string("_in.bin"),
        (const char *)input, n * c * sizeof(float));
  }
#endif // DUMP_FLAG

  // find max and subtract the max to avoid numerical issues
  float max_input = input[0];
  for (int i = 0; i < n * c; ++i) {
    if (input[i] > max_input)
      max_input = input[i];
  }
  // do softmax
  float *ex = (float *)malloc(c * sizeof(float));
  for (int ni = 0; ni < n; ++ni) {
    float sum_of_ex = 0.0f;
    for (int ci = 0; ci < c; ++ci) {
      int i = ni * c + ci;
      float x = input[i] - max_input;
      ex[ci] = exp(x);
      sum_of_ex += ex[ci];
    }
    for (int ci = 0; ci < c; ++ci) {
      int i = ni * c + ci;
      output[i] = ex[ci] / sum_of_ex;
    }
  }
  free(ex);

#ifdef DUMP_FLAG
  if (dump_idx == 0) {
    write_bianry_file(prefix + std::string("_out.bin"),
        (const char *)output, n * c * sizeof(float));
  }
  dump_idx ++;
#endif // DUMP_FLAG
  return 0;
}

static int my_eltwise(float *input_1, float *input_2, float *output,
    int n, int c, int h, int w, int op) {
#ifdef DUMP_FLAG
  static int dump_idx = 0;
  std::string prefix = std::string("eltwise") + std::to_string(dump_idx);
  if (dump_idx < 4) {
    write_bianry_file(prefix + std::string("_1_in.bin"),
        (const char *)input_1, n * c * h * w * sizeof(float));
    write_bianry_file(prefix + std::string("_2_in.bin"),
        (const char *)input_2, n * c * h * w * sizeof(float));
  }
#endif // DUMP_FLAG

  for (int i = 0; i < n * c * h * w; ++i) {
    switch (op) {
    case 0: //caffe::EltwiseParameter_EltwiseOp_PROD:
      output[i] = input_1[i] * input_2[i];
      break;
    case 1: //caffe::EltwiseParameter_EltwiseOp_SUM:
      output[i] = input_1[i] + input_2[i];
      break;
    case 2: //caffe::EltwiseParameter_EltwiseOp_MAX:
      output[i] = input_1[i] > input_2[i] ? input_1[i] : input_2[i];
      break;
    default:
      assert(0);
    }
  }
#ifdef DUMP_FLAG
  if (dump_idx < 4) {
    write_bianry_file(prefix + std::string("_out.bin"),
        (const char *)output, n * c * h * w * sizeof(float));
  }
  dump_idx ++;
#endif // DUMP_FLAG

  return 0;
}

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
    auto filename_tensorfile = llvm::sys::path::stem(filename).str() + ".npz";
    weight_file = openInputTensorFile(filename_tensorfile);

    return success();
  }
  if (auto loadWeightOp = dyn_cast<tpu::LoadWeightOp>(opInst)) {
    llvm::errs() << "LoadWeightOp" << "\n";
    assert(loadWeightOp.offset().hasValue());
    auto offset = loadWeightOp.offset().getValue().getLimitedValue();
    llvm::errs() << "  offset " << offset << "\n";
    auto result = loadWeightOp.getResult();
    llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto weight_data = std::make_unique<std::vector<float> >(size);

    weight_is.get()->seekg(offset, std::ios::beg);
    weight_is.get()->read((char*)weight_data.get()->data(), size * sizeof(float));

    //valueMapping[result] = std::move(weight_data);

    if (loadWeightOp.name().hasValue()) {
      auto tensor_name = loadWeightOp.name().getValue();
      llvm::errs() << "  tensor_name " << tensor_name << "\n";
      auto type = result->getType().cast<TensorType>();
      auto tensor = weight_file->readTensor<float>(tensor_name, type);
      valueMapping[result] = std::move(tensor);
    } else {
      valueMapping[result] = std::move(weight_data);
    }
    return success();
  }
  if (auto op = dyn_cast<tpu::Conv2DOp>(opInst)) {
    llvm::errs() << "Conv2DOp" << "\n";
    //op.dump();
    //assert(op.getNumOperands() == 3);
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
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
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
    if (operand_tensors.size() > 2) {
      assert(operand_tensors.size() == 3);
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
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
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
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
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
    //assert(op.getNumOperands() == 3);
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
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
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
    if (operand_tensors.size() > 2) {
      assert(operand_tensors.size() == 3);
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
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto result_tensor = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here
    int n, c, h, w;
    float negative_slope = op.negative_slope().convertToFloat();
    llvm::errs() << "  negative_slope " << negative_slope << "\n";
    auto input_type = op.x()->getType().cast<TensorType>();
    std::vector<int64_t> i_s(input_type.getShape());
    auto output_type = op.y()->getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());
    assert((i_s == o_s) && "input shape not equal to output shape");
    n = i_s[0];
    c = i_s[1];
    h = i_s[2];
    w = i_s[3];
    float *input = (float *)operand_tensors[0]->data();
    float *output = (float *)result_tensor.get()->data();
    int ret = my_relu(input, output, n, c, h, w, negative_slope);
    assert(ret == 0);
    //dump_data_float_abs("mkldnn_output", mkldnn_output, n, c, oh, ow);
    // TODO: End of compute, need refactor

    valueMapping[result] = std::move(result_tensor);
    return success();
  }
  if (auto op = dyn_cast<tpu::SoftmaxOp>(opInst)) {
    llvm::errs() << "SoftmaxOp" << "\n";
    //op.dump();
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
    assert(shape.size() == 2);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto result_tensor = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here
    int n, c;
    auto input_type = op.x()->getType().cast<TensorType>();
    std::vector<int64_t> i_s(input_type.getShape());
    auto output_type = op.y()->getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());
    assert((i_s == o_s) && "input shape not equal to output shape");
    n = i_s[0];
    c = i_s[1];
    float *input = (float *)operand_tensors[0]->data();
    float *output = (float *)result_tensor.get()->data();
    int ret = my_softmax(input, output, n, c);
    assert(ret == 0);
    //dump_data_float_abs("mkldnn_output", mkldnn_output, n, c, oh, ow);
    // TODO: End of compute, need refactor

    valueMapping[result] = std::move(result_tensor);
    return success();
  }
  if (auto op = dyn_cast<tpu::BatchNormOp>(opInst)) {
    llvm::errs() << "BatchNormOp" << "\n";
    //op.dump();
    assert(op.getNumOperands() == 4);
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
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto result_tensor = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here
    int n, c, h, w;
    auto input_type = op.x()->getType().cast<TensorType>();
    std::vector<int64_t> i_s(input_type.getShape());
    auto output_type = op.y()->getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());
    assert((i_s == o_s) && "input shape not equal to output shape");
    n = i_s[0];
    c = i_s[1];
    h = i_s[2];
    w = i_s[3];
    float *input = (float *)operand_tensors[0]->data();
    float *mean = (float *)operand_tensors[1]->data();
    float *variance = (float *)operand_tensors[2]->data();
    float *scale = (float *)operand_tensors[3]->data();
    float *output = (float *)result_tensor.get()->data();
    int ret = my_bn(input, mean, variance, scale, output, n, c, h, w);
    assert(ret == 0);
    //dump_data_float_abs("mkldnn_output", mkldnn_output, n, c, oh, ow);
    // TODO: End of compute, need refactor

    valueMapping[result] = std::move(result_tensor);
    return success();
  }
  if (auto op = dyn_cast<tpu::ScaleOp>(opInst)) {
    llvm::errs() << "ScaleOp" << "\n";
    //op.dump();
    //assert(op.getNumOperands() == 3);
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
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto result_tensor = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here
    int n, c, h, w;
    auto input_type = op.x()->getType().cast<TensorType>();
    std::vector<int64_t> i_s(input_type.getShape());
    auto output_type = op.y()->getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());
    assert((i_s == o_s) && "input shape not equal to output shape");
    n = i_s[0];
    c = i_s[1];
    h = i_s[2];
    w = i_s[3];
    float *input = (float *)operand_tensors[0]->data();
    float *scale = (float *)operand_tensors[1]->data();
    float *bias = nullptr;
    if (operand_tensors.size() > 2) {
      assert(operand_tensors.size() == 3);
      bias = (float *)operand_tensors[2]->data();
    }
    float *output = (float *)result_tensor.get()->data();
    int ret = my_scale(input, scale, bias, output, n, c, h, w);
    assert(ret == 0);
    //dump_data_float_abs("mkldnn_output", mkldnn_output, n, c, oh, ow);
    // TODO: End of compute, need refactor

    valueMapping[result] = std::move(result_tensor);
    return success();
  }
  if (auto op = dyn_cast<tpu::EltwiseOp>(opInst)) {
    llvm::errs() << "EltwiseOp" << "\n";
    //op.dump();
    assert(op.getNumOperands() == 2);
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
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto result_tensor = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here
    int n, c, h, w;
    auto input_1_type = op.x1()->getType().cast<TensorType>();
    std::vector<int64_t> i1_s(input_1_type.getShape());
    auto input_2_type = op.x2()->getType().cast<TensorType>();
    std::vector<int64_t> i2_s(input_2_type.getShape());
    auto output_type = op.y()->getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());
    assert((i1_s == i2_s) && "two input shapes not equal");
    assert((i1_s == o_s) && "input shape not equal to output shape");
    n = o_s[0];
    c = o_s[1];
    h = o_s[2];
    w = o_s[3];
    float *input_1 = (float *)operand_tensors[0]->data();
    float *input_2 = (float *)operand_tensors[1]->data();
    float *output = (float *)result_tensor.get()->data();
    int ret = my_eltwise(input_1, input_2, output, n, c, h, w, 1);
    assert(ret == 0);
    //dump_data_float_abs("mkldnn_output", mkldnn_output, n, c, oh, ow);
    // TODO: End of compute, need refactor

    valueMapping[result] = std::move(result_tensor);
    return success();
  }
  if (auto op = dyn_cast<tpu::ReshapeOp>(opInst)) {
    llvm::errs() << "ReshapeOp" << "\n";
    //op.dump();
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
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto result_tensor = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here
    auto input_type = op.input()->getType().cast<TensorType>();
    std::vector<int64_t> i_s(input_type.getShape());
    auto output_type = op.output()->getType().cast<TensorType>();
    std::vector<int64_t> o_s(output_type.getShape());
    auto i_size = std::accumulate(std::begin(i_s), std::end(i_s), 1, std::multiplies<>());
    auto o_size = std::accumulate(std::begin(o_s), std::end(o_s), 1, std::multiplies<>());
    assert((i_size == o_size) && "input size not equal to output size");

    // use copy for now
    result_tensor.get()->swap(*operand_tensors[0]);
    // TODO: End of compute, need refactor

    valueMapping[result] = std::move(result_tensor);

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
