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
#include "mlir/Dialect/TPU/QuantizationUtils.h"
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
#include "llvm/Support/CommandLine.h"

#include <numeric>
#include <functional>

#define DEBUG_TYPE "interpreter"

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

//#define DUMP_FLAG
//#define QUANT_DEQUANT_EVERY_LAYER

using namespace mkldnn;

using namespace std;


static llvm::cl::OptionCategory clOptionsCategory("interpreter options");

static llvm::cl::opt<std::string> clAllTensorFilename(
    "dump-all-tensor",
    llvm::cl::desc("dump all tensor into a npz file"),
    llvm::cl::init("-"),
    llvm::cl::cat(clOptionsCategory));

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
  LLVM_DEBUG(
    llvm::errs() << "  k: (" << kh << "*" << kw << "), "
                 << "s: (" << sh << "*" << sw << "), "
                 << "p: (" << ph << "*" << pw << ")" << "\n";
  );

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
  LLVM_DEBUG(
    llvm::errs() << "  k: (" << kh << "*" << kw << "), "
                 << "s: (" << sh << "*" << sw << "), "
                 << "p: (" << p_t << "-" << p_b
                 << "*" << p_l << "-" << p_r << ")" << "\n";
  );

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
  LLVM_DEBUG(
    llvm::errs() << "  n: " << n << ", c: " << c
                 << ", h: " << h << ", w: " << w << "\n";
  );

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

static inline int8_t rshiftAndSaturate(float v, uint32_t rshift) {
  float q_f = v / (1 << rshift);
  #if 0
  // away_from_zero
  int q_i = (q_f >= 0) ? (int)ceil(q_f) : (int)floor(q_f);
  #else
  int q_i = (int)roundf(q_f);
  #endif
  //assert( (q <= 127) && (q >= -128) );
  DEBUG_WITH_TYPE(DEBUG_TYPE"_WARNING",
    if ( (q_i > 127) || (q_i < -128) ) {
      llvm::errs() << "  element exceeds limits [-128, 127] : "
                   << v << " -> " << q_i << "\n";
    }
  );
  if ( q_i > 127 )
    q_i = 127;
  if ( q_i < -128 )
    q_i = -128;

  return (int8_t)q_i;
}

static inline int8_t divideMultiplierAndSaturate(float v, float multiplier) {
  float q_f = v / multiplier;
  #if 0
  // away_from_zero
  int q_i = (q_f >= 0) ? (int)ceil(q_f) : (int)floor(q_f);
  #else
  int q_i = (int)roundf(q_f);
  #endif
  //assert( (q <= 127) && (q >= -128) );
  DEBUG_WITH_TYPE(DEBUG_TYPE"_WARNING",
    if ( (q_i > 127) || (q_i < -128) ) {
      llvm::errs() << "  element exceeds limits [-128, 127] : "
                   << v << " -> " << q_i << "\n";
    }
  );
  if ( q_i > 127 )
    q_i = 127;
  if ( q_i < -128 )
    q_i = -128;

  return (int8_t)q_i;
}

static uint32_t findRShiftFromScale(float scale) {
  // scale = numerator / (1 << rshift)
  // find a rshift put the numerator in range (64, 127)
  assert(scale < 128);
  for (uint32_t rshift = 0; rshift < 32; ++rshift) {
    if ( (scale * (1 << rshift)) >= 64 )
      return rshift;
  }
  assert(false);
  return 31;
}

namespace mlir {

static llvm::StringRef getOpName(Operation *op) {
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::InputOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::Conv2DOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::FullyConnectedOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::Pool2DOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::BatchNormOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::ScaleOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::ReluOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::EltwiseOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::SoftmaxOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::ReshapeOp>(op)) {
    return cast_op.name().getValue();
  }
  llvm::errs() << op->getName();
  assert(false);
  return "not_found";
}

LogicalResult ModuleInterpreter::runOperation(Operation &opInst) {
  // #include "mlir/Dialect/LLVMIR/LLVMConversions.inc"
  if (auto loadFileOp = dyn_cast<tpu::LoadFileOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "LoadFileOp" << "\n";);
    auto filename = loadFileOp.getAttrOfType<StringAttr>("filename").getValue();
    LLVM_DEBUG(llvm::errs() << "  filename " << filename << "\n";);
    weight_is = std::make_unique<std::ifstream>(filename.str(),
        std::ios::in | std::ios::binary);
    auto filename_tensorfile = llvm::sys::path::stem(filename).str() + ".npz";
    weight_file = openInputTensorFile(filename_tensorfile);

    return success();
  }
  if (auto loadWeightOp = dyn_cast<tpu::LoadWeightOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "LoadWeightOp" << "\n";);

    auto result = loadWeightOp.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    if (loadWeightOp.name().hasValue()) {
      auto tensor_name = loadWeightOp.name().getValue();
      LLVM_DEBUG(llvm::errs() << "  tensor_name " << tensor_name << "\n";);
      auto type = result->getType().cast<TensorType>();
      auto tensor = weight_file->readTensor<float>(tensor_name, type);

      valueMapping[result] = std::move(tensor);
    } else {
      assert(loadWeightOp.offset().hasValue());
      auto offset = loadWeightOp.offset().getValue().getLimitedValue();
      LLVM_DEBUG(llvm::errs() << "  offset " << offset << "\n";);
      std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
      assert(shape.size() <= 4);
      auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
      auto weight_data = std::make_unique<std::vector<float> >(size);

      weight_is.get()->seekg(offset, std::ios::beg);
      weight_is.get()->read((char*)weight_data.get()->data(), size * sizeof(float));

      valueMapping[result] = std::move(weight_data);
    }
    return success();
  }
  if (auto op = dyn_cast<tpu::InputOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "InputOp" << "\n";);
    //op.dump();
    //assert(op.getNumOperands() == 3);
    std::vector<std::vector<float> *> operand_tensors;
    unsigned int operandIdx = 0;
    {
      auto operand = op.getOperand();
      LLVM_DEBUG(llvm::errs() << "  operand[" << operandIdx << "] "; operand->getType().dump(); llvm::errs() << "\n";);
      // find operand in valueMapping
      auto it = valueMapping.find(operand);
      if (it == valueMapping.end()) {
        llvm::errs() << "    didn't find\n";
        assert(0);
      } else {
        LLVM_DEBUG(llvm::errs() << "    found in map\n";);
        auto vec = it->second.get();
        if (vec) {
          LLVM_DEBUG(llvm::errs() << "      vec size = " << vec->size() << "\n";);
        } else {
          llvm::errs() << "      vec is nullptr\n";
        }
        operand_tensors.push_back(vec);
      }
      operandIdx++;
    }

    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() == 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto result_tensor = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here
    // use copy for now
    result_tensor->swap(*operand_tensors[0]);

    // do quantize on input data
    if (0) {
      float *output = (float *)result_tensor.get()->data();
      float threshold_y = op.threshold_y().getValue().convertToFloat();
      LLVM_DEBUG(llvm::errs() << "  input data quantize, threshold_y = "
                   << std::to_string(threshold_y) << "\n";);
      for (int i = 0; i < size; ++i) {
        output[i] = output[i] * 128.0 / threshold_y;
      }
    }

    // TODO: End of compute, need refactor

    valueMapping[result] = std::move(result_tensor);

    return success();
  }
  if (auto op = dyn_cast<tpu::Conv2DOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "Conv2DOp" << "\n";);
    //op.dump();
    //assert(op.getNumOperands() == 3);
    std::vector<std::vector<float> *> operand_tensors;
    unsigned int operandIdx = 0;
    for (auto *operand : op.getOperands()) {
      LLVM_DEBUG(llvm::errs() << "  operand[" << operandIdx << "] "; operand->getType().dump(); llvm::errs() << "\n";);
      // find operand in valueMapping
      auto it = valueMapping.find(operand);
      if (it == valueMapping.end()) {
        llvm::errs() << "    didn't find\n";
        assert(0);
      } else {
        LLVM_DEBUG(llvm::errs() << "    found in map\n";);
        auto vec = it->second.get();
        if (vec) {
          LLVM_DEBUG(llvm::errs() << "      vec size = " << vec->size() << "\n";);
        } else {
          llvm::errs() << "      vec is nullptr\n";
        }
        operand_tensors.push_back(vec);
      }
      operandIdx++;
    }

    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
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
    if (op.padding() == "SAME") {
      ph = findPadForSamePadding(ih, oh, kh, sh, dh);
      pw = findPadForSamePadding(iw, ow, kw, sw, dw);
    } else if (op.padding() == "VALID") {
      ph = 0;
      pw = 0;
    } else {
      assert(false);
    }
    float *mkldnn_input = (float *)operand_tensors[0]->data();
    float *mkldnn_weight = (float *)operand_tensors[1]->data();
    float *mkldnn_bias = nullptr;
    float *rshift = nullptr;
    float *multiplier = nullptr;
    if (op.quant() == "NONE") {
      if (operand_tensors.size() > 2) {
        assert(operand_tensors.size() == 3);
        mkldnn_bias = (float *)operand_tensors[2]->data();
      }
    } else if (op.quant() == "INT8" || op.quant() == "INT8_PER_CHANNEL") {
      if (operand_tensors.size() > 3) {
        assert(operand_tensors.size() == 4);
        mkldnn_bias = (float *)operand_tensors[2]->data();
        rshift = (float *)operand_tensors[3]->data();
      } else {
        assert(operand_tensors.size() == 3);
        rshift = (float *)operand_tensors[2]->data();
      }
    } else if (op.quant() == "INT8_MULTIPLIER") {
      if (operand_tensors.size() > 3) {
        assert(operand_tensors.size() == 4);
        mkldnn_bias = (float *)operand_tensors[2]->data();
        multiplier = (float *)operand_tensors[3]->data();
      } else {
        assert(operand_tensors.size() == 3);
        multiplier = (float *)operand_tensors[2]->data();
      }
    } else {
      assert(false);
    }

#ifdef QUANT_DEQUANT_EVERY_LAYER
    // do quantize on input
    // remove this when the network is full int8, and passed legalization
    // copy the input first
    std::vector<float> input_copy(*operand_tensors[0]);
    mkldnn_input = input_copy.data();
    if (op.quant() == "INT8" || op.quant() == "INT8_PER_CHANNEL"
        || op.quant() == "INT8_MULTIPLIER") {
      float threshold_x;
      auto status = getPreviousOpThreshold(op, &threshold_x);
      LLVM_DEBUG(llvm::errs() << "  conv input quantize, threshold_x = " << std::to_string(threshold_x) << "\n";);
      assert(succeeded(status));
      for (size_t i = 0; i < operand_tensors[0]->size(); ++i) {
        mkldnn_input[i] = mkldnn_input[i] * 128.0 / threshold_x;
      }
    }
#endif

    float *mkldnn_output = (float *)result_tensor.get()->data();
    int mkldnn_ret = mkldnn_conv(mkldnn_input, mkldnn_weight, mkldnn_bias, mkldnn_output,
        n, ic, ih, iw, oc, oh, ow, kh, kw, sh, sw, ph, pw);
    assert(mkldnn_ret == 0);
    //dump_data_float_abs("mkldnn_output", mkldnn_output, n, oc, oh, ow);

    // rshift and saturate on output
    if (op.quant() == "INT8") {
      assert(rshift);
      for (int i = 0; i < size; ++i) {
        mkldnn_output[i] = (float)rshiftAndSaturate(mkldnn_output[i], (uint32_t)rshift[0]);
      }
    } else if (op.quant() == "INT8_PER_CHANNEL") {
      assert(rshift);
      int inner_size = size / oc;
      for (int i = 0; i < oc; ++i) {
        for (int j = 0; j < inner_size; ++j) {
          mkldnn_output[i * inner_size + j] =
              (float)rshiftAndSaturate(mkldnn_output[i * inner_size + j],
                                       (uint32_t)rshift[i]);
        }
      }
    } else if (op.quant() == "INT8_MULTIPLIER") {
      assert(multiplier);
      int inner_size = size / oc;
      for (int i = 0; i < oc; ++i) {
        for (int j = 0; j < inner_size; ++j) {
          mkldnn_output[i * inner_size + j] =
              (float)divideMultiplierAndSaturate(mkldnn_output[i * inner_size + j],
                                       multiplier[i]);
        }
      }
    }

#ifdef QUANT_DEQUANT_EVERY_LAYER
    // do dequantize on output
    // remove this when the network is full int8, and passed legalization
    if (op.quant() == "INT8" || op.quant() == "INT8_PER_CHANNEL"
        || op.quant() == "INT8_MULTIPLIER") {
      float threshold_y = op.threshold_y().getValue().convertToFloat();
      LLVM_DEBUG(llvm::errs() << "  conv output dequantize, threshold_y = "
                   << std::to_string(threshold_y) << "\n";);
      for (int i = 0; i < size; ++i) {
        mkldnn_output[i] = mkldnn_output[i] * threshold_y / 128.0;
      }
    }
#endif

    // TODO: End of compute, need refactor

    valueMapping[result] = std::move(result_tensor);

    return success();
  }
  if (auto op = dyn_cast<tpu::Pool2DOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "Pool2DOp" << "\n";);
    //op.dump();
    //assert(op.getNumOperands() == 1);
    std::vector<std::vector<float> *> operand_tensors;
    {
      auto operand = op.getOperand();
      LLVM_DEBUG(llvm::errs() << "  operand[0] "; operand->getType().dump(); llvm::errs() << "\n";);
      // find operand in valueMapping
      auto it = valueMapping.find(operand);
      if (it == valueMapping.end()) {
        llvm::errs() << "    didn't find\n";
        assert(0);
      } else {
        LLVM_DEBUG(llvm::errs() << "    found in map\n";);
        auto vec = it->second.get();
        if (vec) {
          LLVM_DEBUG(llvm::errs() << "      vec size = " << vec->size() << "\n";);
        } else {
          llvm::errs() << "      vec is nullptr\n";
        }
        operand_tensors.push_back(vec);
      }
    }
    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto result_tensor = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here
    auto pool_method = op.getAttrOfType<StringAttr>("pool");
    bool is_average_pool;
    if (pool_method.getValue() == "AVE") {
      is_average_pool = true;
    } else if (pool_method.getValue() == "MAX") {
      is_average_pool = false;
    } else {
      assert(false);
    }
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

    // for INT8, get threshold_x and make copy of input first
    std::vector<float> input_copy;
    float threshold_x;
    float threshold_y;
    if (op.quant() == "INT8" && is_average_pool) {
      // make copy
      std::vector<float> &src_vec = *operand_tensors[0];
      std::copy(src_vec.begin(), src_vec.end(), back_inserter(input_copy));
      mkldnn_input = input_copy.data();

      // get threshold_x
      auto status = getPreviousOpThreshold(op, &threshold_x);
      assert(succeeded(status));
      // get threshold_y
      threshold_y = op.threshold_y().getValue().convertToFloat();
    }

#ifdef QUANT_DEQUANT_EVERY_LAYER
    // do quantize on input
    // remove this when the network is full int8, and passed legalization
    if (op.quant() == "INT8" && is_average_pool) {
      for (size_t i = 0; i < operand_tensors[0]->size(); ++i) {
        mkldnn_input[i] = mkldnn_input[i] * 128.0 / threshold_x;
      }
    }
#endif

    float *mkldnn_output = (float *)result_tensor.get()->data();
    int mkldnn_ret = mkldnn_pool(mkldnn_input, mkldnn_output,
        n, c, ih, iw, oh, ow, kh, kw, sh, sw, ph, pw, is_average_pool);
    assert(mkldnn_ret == 0);
    //dump_data_float_abs("mkldnn_output", mkldnn_output, n, c, oh, ow);

    // do quantize for average pooling, max poolings are bypassed
    if (op.quant() == "INT8" && is_average_pool) {
      // determine multiplier and rshift according to threshold_x
      // scale = threshold_x / threshold_y
      // scale will be implemented by hardware as
      // scale = multiplier / (1 << rshift)
      // find a rshift, that put max(multiplier) into range (64, 127)
      uint32_t rshift;
      int8_t multiplier;
      rshift = findRShiftFromScale(threshold_x / threshold_y);
      float scale = threshold_x / threshold_y;
      multiplier = (int8_t)(scale * (1 << rshift));

      // apply multiplier
      for (int i = 0; i < size; ++i) {
        mkldnn_output[i] = mkldnn_output[i] * multiplier;
      }

      // rshift and saturate on output
      for (int i = 0; i < size; ++i) {
        mkldnn_output[i] = (float)rshiftAndSaturate(mkldnn_output[i], (uint32_t)rshift);
      }
    }

#ifdef QUANT_DEQUANT_EVERY_LAYER
    // do dequantize on output
    // remove this when the network is full int8, and passed legalization
    if (op.quant() == "INT8" && is_average_pool) {
      LLVM_DEBUG(llvm::errs() << "  avg pool output dequantize, threshold_y = "
                 << std::to_string(threshold_y) << "\n";);
      for (int i = 0; i < size; ++i) {
        mkldnn_output[i] = mkldnn_output[i] * threshold_y / 128.0;
      }
    }
#endif
    // TODO: End of compute, need refactor

    valueMapping[result] = std::move(result_tensor);

    return success();
  }
  if (auto op = dyn_cast<tpu::FullyConnectedOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "FullyConnectedOp" << "\n";);
    //op.dump();
    //assert(op.getNumOperands() == 3);
    std::vector<std::vector<float> *> operand_tensors;
    unsigned int operandIdx = 0;
    for (auto *operand : op.getOperands()) {
      LLVM_DEBUG(llvm::errs() << "  operand[" << operandIdx << "] "; operand->getType().dump(); llvm::errs() << "\n";);
      // find operand in valueMapping
      auto it = valueMapping.find(operand);
      if (it == valueMapping.end()) {
        llvm::errs() << "    didn't find\n";
        assert(0);
      } else {
        LLVM_DEBUG(llvm::errs() << "    found in map\n";);
        auto vec = it->second.get();
        if (vec) {
          LLVM_DEBUG(llvm::errs() << "      vec size = " << vec->size() << "\n";);
        } else {
          llvm::errs() << "      vec is nullptr\n";
        }
        operand_tensors.push_back(vec);
      }
      operandIdx++;
    }

    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
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
    float *rshift = nullptr;
    if (op.quant() == "NONE") {
      if (operand_tensors.size() > 2) {
        assert(operand_tensors.size() == 3);
        mkldnn_bias = (float *)operand_tensors[2]->data();
      }
    } else if (op.quant() == "INT8") {
      if (operand_tensors.size() > 3) {
        assert(operand_tensors.size() == 4);
        mkldnn_bias = (float *)operand_tensors[2]->data();
        rshift = (float *)operand_tensors[3]->data();
      } else {
        assert(operand_tensors.size() == 3);
        rshift = (float *)operand_tensors[2]->data();
      }
    } else {
      assert(false);
    }

#ifdef QUANT_DEQUANT_EVERY_LAYER
    // do quantize on input
    // remove this when the network is full int8, and passed legalization
    // copy the input first
    std::vector<float> input_copy(*operand_tensors[0]);
    mkldnn_input = input_copy.data();
    if (op.quant() == "INT8") {
      float threshold_x;
      auto status = getPreviousOpThreshold(op, &threshold_x);
      LLVM_DEBUG(llvm::errs() << "  fc input quantize, threshold_x = " << std::to_string(threshold_x) << "\n";);
      assert(succeeded(status));
      for (size_t i = 0; i < operand_tensors[0]->size(); ++i) {
        mkldnn_input[i] = mkldnn_input[i] * 128.0 / threshold_x;
      }
    }
#endif

    float *mkldnn_output = (float *)result_tensor.get()->data();
    int mkldnn_ret = mkldnn_ip(mkldnn_input, mkldnn_weight, mkldnn_bias,
        mkldnn_output, m, k, n, transpose);
    assert(mkldnn_ret == 0);
    //dump_data_float_abs("mkldnn_output", mkldnn_output, 1, 1, m, n);

    // rshift and saturate on output
    if (op.quant() == "INT8") {
      assert(rshift);
      for (int i = 0; i < size; ++i) {
        mkldnn_output[i] = (float)rshiftAndSaturate(mkldnn_output[i], (uint32_t)rshift[0]);
      }
    }

#ifdef QUANT_DEQUANT_EVERY_LAYER
    // do dequantize on output
    // remove this when the network is full int8, and passed legalization
    if (op.quant() == "INT8") {
      float threshold_y = op.threshold_y().getValue().convertToFloat();
      LLVM_DEBUG(llvm::errs() << "  fc output dequantize, threshold_y = "
                   << std::to_string(threshold_y) << "\n";);
      for (int i = 0; i < size; ++i) {
        mkldnn_output[i] = mkldnn_output[i] * threshold_y / 128.0;
      }
    }
#endif

    // TODO: End of compute, need refactor

    valueMapping[result] = std::move(result_tensor);

    return success();
  }
  if (auto op = dyn_cast<tpu::ReluOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "ReluOp" << "\n";);
    //op.dump();
    std::vector<std::vector<float> *> operand_tensors;
    {
      auto operand = op.getOperand();
      LLVM_DEBUG(llvm::errs() << "  operand[0] "; operand->getType().dump(); llvm::errs() << "\n";);
      // find operand in valueMapping
      auto it = valueMapping.find(operand);
      if (it == valueMapping.end()) {
        llvm::errs() << "    didn't find\n";
        assert(0);
      } else {
        LLVM_DEBUG(llvm::errs() << "    found in map\n";);
        auto vec = it->second.get();
        if (vec) {
          LLVM_DEBUG(llvm::errs() << "      vec size = " << vec->size() << "\n";);
        } else {
          llvm::errs() << "      vec is nullptr\n";
        }
        operand_tensors.push_back(vec);
      }
    }

    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto result_tensor = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here
    int n, c, h, w;
    float negative_slope = op.negative_slope().convertToFloat();
    LLVM_DEBUG(llvm::errs() << "  negative_slope " << negative_slope << "\n";);
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
    LLVM_DEBUG(llvm::errs() << "SoftmaxOp" << "\n";);
    //op.dump();
    std::vector<std::vector<float> *> operand_tensors;
    {
      auto operand = op.getOperand();
      LLVM_DEBUG(llvm::errs() << "  operand[0] "; operand->getType().dump(); llvm::errs() << "\n";);
      // find operand in valueMapping
      auto it = valueMapping.find(operand);
      if (it == valueMapping.end()) {
        llvm::errs() << "    didn't find\n";
        assert(0);
      } else {
        LLVM_DEBUG(llvm::errs() << "    found in map\n";);
        auto vec = it->second.get();
        if (vec) {
          LLVM_DEBUG(llvm::errs() << "      vec size = " << vec->size() << "\n";);
        } else {
          llvm::errs() << "      vec is nullptr\n";
        }
        operand_tensors.push_back(vec);
      }
    }

    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
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

    // do dequantization
    if (0) {
      float threshold_x;
      auto status = getPreviousOpThreshold(op, &threshold_x);
      LLVM_DEBUG(llvm::errs() << "  softmax dequantize, threshold_x = " << std::to_string(threshold_x) << "\n";);
      assert(succeeded(status));
      for (size_t i = 0; i < operand_tensors[0]->size(); ++i) {
        input[i] = input[i] * threshold_x / 128.0;
      }
    }

    float *output = (float *)result_tensor.get()->data();
    int ret = my_softmax(input, output, n, c);
    assert(ret == 0);
    //dump_data_float_abs("mkldnn_output", mkldnn_output, n, c, oh, ow);
    // TODO: End of compute, need refactor

    valueMapping[result] = std::move(result_tensor);
    return success();
  }
  if (auto op = dyn_cast<tpu::BatchNormOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "BatchNormOp" << "\n";);
    //op.dump();
    assert(op.getNumOperands() == 4);
    std::vector<std::vector<float> *> operand_tensors;
    unsigned int operandIdx = 0;
    for (auto *operand : op.getOperands()) {
      LLVM_DEBUG(llvm::errs() << "  operand[" << operandIdx << "] "; operand->getType().dump(); llvm::errs() << "\n";);
      // find operand in valueMapping
      auto it = valueMapping.find(operand);
      if (it == valueMapping.end()) {
        llvm::errs() << "    didn't find\n";
        assert(0);
      } else {
        LLVM_DEBUG(llvm::errs() << "    found in map\n";);
        auto vec = it->second.get();
        if (vec) {
          LLVM_DEBUG(llvm::errs() << "      vec size = " << vec->size() << "\n";);
        } else {
          llvm::errs() << "      vec is nullptr\n";
        }
        operand_tensors.push_back(vec);
      }
      operandIdx++;
    }

    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
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
    LLVM_DEBUG(llvm::errs() << "ScaleOp" << "\n";);
    //op.dump();
    //assert(op.getNumOperands() == 3);
    std::vector<std::vector<float> *> operand_tensors;
    unsigned int operandIdx = 0;
    for (auto *operand : op.getOperands()) {
      LLVM_DEBUG(llvm::errs() << "  operand[" << operandIdx << "] "; operand->getType().dump(); llvm::errs() << "\n";);
      // find operand in valueMapping
      auto it = valueMapping.find(operand);
      if (it == valueMapping.end()) {
        llvm::errs() << "    didn't find\n";
        assert(0);
      } else {
        LLVM_DEBUG(llvm::errs() << "    found in map\n";);
        auto vec = it->second.get();
        if (vec) {
          LLVM_DEBUG(llvm::errs() << "      vec size = " << vec->size() << "\n";);
        } else {
          llvm::errs() << "      vec is nullptr\n";
        }
        operand_tensors.push_back(vec);
      }
      operandIdx++;
    }

    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
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
    LLVM_DEBUG(llvm::errs() << "EltwiseOp" << "\n";);
    //op.dump();
    assert(op.getNumOperands() == 2);
    std::vector<std::vector<float> *> operand_tensors;
    unsigned int operandIdx = 0;
    for (auto *operand : op.getOperands()) {
      LLVM_DEBUG(llvm::errs() << "  operand[" << operandIdx << "] "; operand->getType().dump(); llvm::errs() << "\n";);
      // find operand in valueMapping
      auto it = valueMapping.find(operand);
      if (it == valueMapping.end()) {
        llvm::errs() << "    didn't find\n";
        assert(0);
      } else {
        LLVM_DEBUG(llvm::errs() << "    found in map\n";);
        auto vec = it->second.get();
        if (vec) {
          LLVM_DEBUG(llvm::errs() << "      vec size = " << vec->size() << "\n";);
        } else {
          llvm::errs() << "      vec is nullptr\n";
        }
        operand_tensors.push_back(vec);
      }
      operandIdx++;
    }

    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    assert(shape.size() <= 4);
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto result_tensor = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here
#define MAX_ELTWISE_INPUT (2)
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
    float *input[MAX_ELTWISE_INPUT];
    for (int index = 0; index < MAX_ELTWISE_INPUT; ++index) {
      input[index] = (float *)operand_tensors[index]->data();
    }
    float *output = (float *)result_tensor.get()->data();

    // for INT8, get threshold_x and make copy of input first
    std::vector<float> input_copy[MAX_ELTWISE_INPUT];
    std::vector<float> threshold_x(MAX_ELTWISE_INPUT);
    float threshold_y;
    if (op.quant() == "INT8") {
      for (int index = 0; index < MAX_ELTWISE_INPUT; ++index) {
        // make copy
        std::vector<float> &src_vec = *operand_tensors[index];
        std::copy(src_vec.begin(), src_vec.end(), back_inserter(input_copy[index]));
        input[index] = input_copy[index].data();

        // get threshold_x
        auto status = getPreviousOpThreshold(op, &threshold_x[index], index);
        assert(succeeded(status));
      }
      // get threshold_y
      threshold_y = op.threshold_y().getValue().convertToFloat();
    }

#ifdef QUANT_DEQUANT_EVERY_LAYER
    // do quantize on input
    // remove this when the network is full int8, and passed legalization
    if (op.quant() == "INT8") {
      for (int index = 0; index < MAX_ELTWISE_INPUT; ++index) {
        for (size_t i = 0; i < operand_tensors[index]->size(); ++i) {
          input[index][i] = input[index][i] * 128.0 / threshold_x[index];
        }
      }
    }
#endif

    // determine multiplier and rshift according each threshold_x
    // scale[i] = threshold_x[i] / threshold_y
    // each scale will be implemented by hardware as
    // scale[i] = multiplier / (1 << rshift)
    // find a rshift, that put max(multiplier) into range (64, 127)
    uint32_t rshift;
    int8_t multiplier[MAX_ELTWISE_INPUT];
    if (op.quant() == "INT8") {
      // determine rshift for all inputs, and multiplier for each input
      // use max threshold_x to find rshift first
      float max_threshold_x = *std::max_element(
          std::begin(threshold_x), std::end(threshold_x));
      rshift = findRShiftFromScale(max_threshold_x / threshold_y);
      for (int index = 0; index < 2; ++index) {
        float scale = threshold_x[index] / threshold_y;
        multiplier[index] = (int8_t)(scale * (1 << rshift));
      }
    }

    // apply multiplier
    if (op.quant() == "INT8") {
      for (int index = 0; index < MAX_ELTWISE_INPUT; ++index) {
        for (size_t i = 0; i < operand_tensors[index]->size(); ++i) {
          input[index][i] = input[index][i] * multiplier[index];
        }
      }
    }

    int ret = my_eltwise(input[0], input[1], output, n, c, h, w, 1);
    assert(ret == 0);
    //dump_data_float_abs("output", mkldnn_output, n, c, oh, ow);

    // rshift and saturate on output
    if (op.quant() == "INT8") {
      //assert(rshift);
      for (int i = 0; i < size; ++i) {
        output[i] = (float)rshiftAndSaturate(output[i], (uint32_t)rshift);
      }
    }

#ifdef QUANT_DEQUANT_EVERY_LAYER
    // do dequantize on output
    // remove this when the network is full int8, and passed legalization
    if (op.quant() == "INT8") {
      LLVM_DEBUG(llvm::errs() << "  fc output dequantize, threshold_y = "
                   << std::to_string(threshold_y) << "\n";);
      for (int i = 0; i < size; ++i) {
        output[i] = output[i] * threshold_y / 128.0;
      }
    }
#endif
    // TODO: End of compute, need refactor

    valueMapping[result] = std::move(result_tensor);
    return success();
  }
  if (auto op = dyn_cast<tpu::ReshapeOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "ReshapeOp" << "\n";);
    //op.dump();
    std::vector<std::vector<float> *> operand_tensors;
    {
      auto operand = op.getOperand();
      LLVM_DEBUG(llvm::errs() << "  operand[0] "; operand->getType().dump(); llvm::errs() << "\n";);
      // find operand in valueMapping
      auto it = valueMapping.find(operand);
      if (it == valueMapping.end()) {
        llvm::errs() << "    didn't find\n";
        assert(0);
      } else {
        LLVM_DEBUG(llvm::errs() << "    found in map\n";);
        auto vec = it->second.get();
        if (vec) {
          LLVM_DEBUG(llvm::errs() << "      vec size = " << vec->size() << "\n";);
        } else {
          llvm::errs() << "      vec is nullptr\n";
        }
        operand_tensors.push_back(vec);
      }
    }

    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
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
  if (auto op = dyn_cast<tpu::QuantizationOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "QuantizationOp" << "\n";);
    //op.dump();
    //assert(op.getNumOperands() == 3);
    std::vector<std::vector<float> *> operand_tensors;
    unsigned int operandIdx = 0;
    {
      auto operand = op.getOperand();
      LLVM_DEBUG(llvm::errs() << "  operand[" << operandIdx << "] "; operand->getType().dump(); llvm::errs() << "\n";);
      // find operand in valueMapping
      auto it = valueMapping.find(operand);
      if (it == valueMapping.end()) {
        llvm::errs() << "    didn't find\n";
        assert(0);
      } else {
        LLVM_DEBUG(llvm::errs() << "    found in map\n";);
        auto vec = it->second.get();
        if (vec) {
          LLVM_DEBUG(llvm::errs() << "      vec size = " << vec->size() << "\n";);
        } else {
          llvm::errs() << "      vec is nullptr\n";
        }
        operand_tensors.push_back(vec);
      }
      operandIdx++;
    }

    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto result_tensor = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here
    if (op.quant() == "INT8") {
      float *input = (float *)operand_tensors[0]->data();
      float *output = (float *)result_tensor.get()->data();
      float threshold = op.threshold().getValue().convertToFloat();
      LLVM_DEBUG(llvm::errs() << "  quantization, threshold = "
                   << std::to_string(threshold) << "\n";);
      for (int i = 0; i < size; ++i) {
        output[i] = input[i] * 128.0 / threshold;
      }
    }
    // TODO: End of compute, need refactor

    valueMapping[result] = std::move(result_tensor);

    return success();
  }
  if (auto op = dyn_cast<tpu::DequantizationOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "DequantizationOp" << "\n";);
    //op.dump();
    //assert(op.getNumOperands() == 3);
    std::vector<std::vector<float> *> operand_tensors;
    unsigned int operandIdx = 0;
    {
      auto operand = op.getOperand();
      LLVM_DEBUG(llvm::errs() << "  operand[" << operandIdx << "] "; operand->getType().dump(); llvm::errs() << "\n";);
      // find operand in valueMapping
      auto it = valueMapping.find(operand);
      if (it == valueMapping.end()) {
        llvm::errs() << "    didn't find\n";
        assert(0);
      } else {
        LLVM_DEBUG(llvm::errs() << "    found in map\n";);
        auto vec = it->second.get();
        if (vec) {
          LLVM_DEBUG(llvm::errs() << "      vec size = " << vec->size() << "\n";);
        } else {
          llvm::errs() << "      vec is nullptr\n";
        }
        operand_tensors.push_back(vec);
      }
      operandIdx++;
    }

    auto result = op.getResult();
    LLVM_DEBUG(llvm::errs() << "  result "; result->getType().dump(); llvm::errs() << "\n";);
    std::vector<int64_t> shape = result->getType().cast<TensorType>().getShape();
    auto size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
    auto result_tensor = std::make_unique<std::vector<float> >(size);

    // TODO: do the actual compute here
    if (op.quant() == "INT8") {
      float *input = (float *)operand_tensors[0]->data();
      float *output = (float *)result_tensor.get()->data();
      float threshold = op.threshold().getValue().convertToFloat();
      LLVM_DEBUG(llvm::errs() << "  quantization, threshold = "
                   << std::to_string(threshold) << "\n";);
      for (int i = 0; i < size; ++i) {
        output[i] = input[i] * threshold / 128.0;
      }
    }
    // TODO: End of compute, need refactor

    valueMapping[result] = std::move(result_tensor);

    return success();
  }

  if (auto op = dyn_cast<ConstantOp>(opInst)) {
    LLVM_DEBUG(llvm::errs() << "ConstantOp" << "\n";);
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
    LLVM_DEBUG(llvm::errs() << "ReturnOp" << "\n";);
    //op.dump();
    std::vector<float> *return_vec;
    unsigned int operandIdx = 0;
    for (auto *operand : op.getOperands()) {
      LLVM_DEBUG(llvm::errs() << "  operand[" << operandIdx << "] "; operand->getType().dump(); llvm::errs() << "\n";);
      // find operand in valueMapping
      auto it = valueMapping.find(operand);
      if (it == valueMapping.end()) {
        llvm::errs() << "    didn't find\n";
        assert(0);
      } else {
        LLVM_DEBUG(llvm::errs() << "    found in map\n";);
        return_vec = it->second.get();
        LLVM_DEBUG(llvm::errs() << "      vec size = " << return_vec->size() << "\n";);
      }
      operandIdx++;
    }

    //copy the value into outputs_
    assert(outputs_.size() == 1);
    outputs_[0]->swap(*return_vec);

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
  LLVM_DEBUG(llvm::errs() << "func " << func.getName() << "\n";);
  // Clear the value mappings, it is only relevant within one function.
  valueMapping.clear();

  // Add function arguments to the value remapping table.
  unsigned int argIdx = 0;
  assert(inputs_.size() == 1);
  for (auto arg : func.getArguments()) {
    LLVM_DEBUG(
      llvm::errs() << "arg " << argIdx << ": ";
      arg->getType().dump();
      llvm::errs() << "\n";
    );

    // copy the inputs_[0] into a unique_ptr pointed vector
    // TODO: pass input as unique_ptr directly
    auto input = std::make_unique<std::vector<float> >();
    input->swap(*inputs_[0]);
    valueMapping[arg] = std::move(input);
    argIdx++;
  }
  assert(argIdx == 1);

  // Then, run blocks one by one.
  for (Block &bb : func.getBlocks()) {
    if (failed(runBlock(bb)))
      return failure();
  }

  if (clAllTensorFilename != "-") {
    // dump all values
    LLVM_DEBUG(llvm::errs() << "valueMapping size " << valueMapping.size() << "\n";);
    auto TensorOut = openOutputTensorFile(clAllTensorFilename);
    for (auto it = valueMapping.begin(); it != valueMapping.end(); it++ ) {
      auto op = it->first->getDefiningOp();
      if (!op) {
        //it->first->dump();
        continue;
      }
      LLVM_DEBUG(llvm::errs() << op->getName() << " : " << getOpName(op) << "\n";);
      auto vec = it->second.get();
      assert(vec);
      auto type = it->first->getType().dyn_cast<mlir::TensorType>();
      LLVM_DEBUG(llvm::errs() << "  vec size = " << vec->size() << "\n";);
      TensorOut->addTensor(getOpName(op), vec, type);
    }
    TensorOut->keep();
  }

  return success();
}

LogicalResult ModuleInterpreter::runFunctions() {
  for (FuncOp function : mlirModule.getOps<FuncOp>()) {
    LLVM_DEBUG(llvm::errs() << "run " << function.getName() << "\n";);

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
