#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <numeric>
#include <functional>
#include <assert.h>
#include <chrono>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "mkldnn.hpp"
#include <math.h>

#define DEBUG_TYPE "native-cpu"

//#define DUMP_FLAG

using namespace mkldnn;

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

int mkldnn_conv(float *input, float *weight, float *bias,
    float *output, int n, int ic, int ih, int iw, int oc, int oh, int ow,
    int kh, int kw, int sh, int sw, int dh, int dw, int ph, int pw, int g) {
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
        (const char *)weight, g * (oc/g) * (ic/g) * kh * kw * sizeof(float));
    write_bianry_file(prefix + std::string("_bias.bin"),
        (const char *)bias, oc * sizeof(float));
  }
#endif // DUMP_FLAG
  LLVM_DEBUG(
    llvm::errs() << "  k: (" << kh << "*" << kw << "), "
                 << "s: (" << sh << "*" << sw << "), "
                 << "p: (" << ph << "*" << pw << "), "
                 << "g: " << g << "\n";
  );

  using tag = memory::format_tag;
  using dt = memory::data_type;

  engine eng(engine::kind::cpu, 0);
  stream s(eng);

  std::vector<primitive> net;
  std::vector<std::unordered_map<int, memory>> net_args;

  const memory::dim batch = n;
  memory::dims src_tz = { batch, ic, ih, iw };
  memory::dims weights_tz = (g != 1) ? memory::dims{g, oc/g, ic/g, kh, kw}
                                    : memory::dims{oc, ic, kh, kw};
  memory::dims bias_tz = { oc };
  memory::dims dst_tz = { batch, oc, oh, ow };
  memory::dims strides = { sh, sw };
  int ph_t = ph;
  int pw_l = pw;
  int ph_b = ph;
  int pw_r = pw;
  memory::dims padding_l = { ph_t, pw_l };
  memory::dims padding_r = { ph_b, pw_r };
  memory::dims dilation = { dh-1, dw-1 }; // mkldnn dialtion is different with caffe

  // memory
  auto user_src_memory = memory(
      { { src_tz }, dt::f32, tag::nchw }, eng, input);
  auto user_weights_memory = (g != 1)
      ? memory({ { weights_tz }, dt::f32, tag::goihw }, eng, weight)
      : memory({ { weights_tz }, dt::f32, tag::oihw }, eng, weight);
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
      strides, dilation, padding_l, padding_r);
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

int mkldnn_deconv(float *input, float *weight, float *bias,
    float *output, int n, int ic, int ih, int iw, int oc, int oh, int ow,
    int kh, int kw, int sh, int sw, int ph, int pw, int g) {
  if (!bias) {
    auto zero_bias = new std::vector<float>(oc, 0.0f);
    bias = zero_bias->data();
  }

  // TODO - padding
  assert(ph == 0);
  assert(pw == 0);
  int ph_t = ph;
  int pw_l = pw;
  int ph_b = ph;
  int pw_r = pw;

  LLVM_DEBUG(
    llvm::errs() << "  i: (" << ih << "*" << iw << "), "
                 << "  o: (" << oh << "*" << ow << "), "
                 << "  k: (" << kh << "*" << kw << "), "
                 << "s: (" << sh << "*" << sw << "), "
                 << "p: (" << ph << "*" << pw << "), "
                 << "g: " << g << "\n";
  );

  using tag = memory::format_tag;
  using dt = memory::data_type;

  engine eng(engine::kind::cpu, 0);
  stream s(eng);

  std::vector<primitive> net;
  std::vector<std::unordered_map<int, memory>> net_args;

  const memory::dim batch = n;
  memory::dims src_tz = { batch, ic, ih, iw };
  memory::dims weights_tz = (g != 1) ? memory::dims{g, oc/g, ic/g, kh, kw}
                                    : memory::dims{oc, ic, kh, kw};
  memory::dims bias_tz = { oc };
  memory::dims dst_tz = { batch, oc, oh, ow };
  memory::dims strides = { sh, sw };
  memory::dims padding_l = { ph_t, pw_l };
  memory::dims padding_r = { ph_b, pw_r };

  // memory
  auto user_src_memory = memory(
      { { src_tz }, dt::f32, tag::nchw }, eng, input);
  auto user_weights_memory = (g != 1)
      ? memory({ { weights_tz }, dt::f32, tag::goihw }, eng, weight)
      : memory({ { weights_tz }, dt::f32, tag::oihw }, eng, weight);
  auto user_bias_memory = memory(
      { { bias_tz }, dt::f32, tag::x }, eng, bias);
  auto user_dst_memory = memory(
      { { dst_tz }, dt::f32, tag::nchw }, eng, output);

  // md
  auto src_md     = memory::desc({ src_tz }, dt::f32, tag::any);
  auto weights_md = memory::desc({ weights_tz }, dt::f32, tag::any);
  auto bias_md    = memory::desc({ bias_tz }, dt::f32, tag::any);
  auto dst_md     = memory::desc({ dst_tz }, dt::f32, tag::any);

  // deconv desc
  auto deconv_desc = deconvolution_forward::desc(prop_kind::forward_inference,
      algorithm::deconvolution_direct, src_md, weights_md, bias_md, dst_md,
      strides, padding_l, padding_r);

  auto deconv_prim_desc = deconvolution_forward::primitive_desc(deconv_desc, eng);

  // do reorder if needed
  auto src_memory = user_src_memory;
  if (deconv_prim_desc.src_desc() != user_src_memory.get_desc()) {
    src_memory = memory(deconv_prim_desc.src_desc(), eng);
    net.push_back(reorder(user_src_memory, src_memory));
    net_args.push_back({ { MKLDNN_ARG_FROM, user_src_memory },
        { MKLDNN_ARG_TO, src_memory } });
  }
  auto weights_memory = user_weights_memory;
  if (deconv_prim_desc.weights_desc() != user_weights_memory.get_desc()) {
    weights_memory = memory(deconv_prim_desc.weights_desc(), eng);
    reorder(user_weights_memory, weights_memory)
        .execute(s, user_weights_memory, weights_memory);
  }
  auto bias_memory = user_bias_memory;

  auto dst_memory = memory(deconv_prim_desc.dst_desc(), eng);

  net.push_back(deconvolution_forward(deconv_prim_desc));
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

int mkldnn_pool(float *input, float *output,
    int n, int c, int ih, int iw, int oh, int ow,
    int kh, int kw, int sh, int sw, int pt, int pb, int pl, int pr,
    bool is_avg) {
  LLVM_DEBUG(
    llvm::errs() << "mkldnn_pool: "<< "  i: (" << ih << "*" << iw << "), "
              << "o: (" << oh << "*" << ow << "), "
              << "k: (" << kh << "*" << kw << "), "
              << "s: (" << sh << ", " << sw << "), "
              << "p: (" << pt << ", " << pb  << ", " << pl << ", "  << pr << "), " << "\n";
  );

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
                 << "p: (" << pt << "-" << pb
                 << "*" << pl << "-" << pr << ")" << "\n";
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
  memory::dims padding_t_l = { pt, pl };
  memory::dims padding_b_r = { pb, pr };

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
      is_avg ? algorithm::pooling_avg_include_padding : algorithm::pooling_max,
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

int mkldnn_ip(float *input, float *weight, float *bias,
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

int my_relu(float *input, float *output,
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

int my_prelu(float *input, float *output, int n, int c, int h, int w,
            float *negative_slope) {

  LLVM_DEBUG(llvm::errs() << "  n: " << n << ", c: " << c << ", h: " << h
                          << ", w: " << w << "\n";);

  for(int batch = 0; batch < n; ++batch){
    for(int channel = 0 ; channel < c; ++channel){
        int index = batch * c * w * h + channel * w * h;
        for(int i = 0; i < w * h; ++i){
          if (input[index + i] > 0) {
            output[index + i] = input[index + i];
          } else {
            output[index + i] = negative_slope[channel] * input[index + i];
          }

        }
    }
  }

  return 0;
}

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 0.5 * tanh(0.5 * x) + 0.5;
}
int my_sigmoid(float *input, float *output, int n, int c, int h, int w) {
  LLVM_DEBUG(llvm::errs() << "  n: " << n << ", c: " << c << ", h: " << h
                          << ", w: " << w << "\n";);

  for (int i = 0; i < n * c * h * w; ++i) {
      output[i] = sigmoid(input[i]);
  }

  return 0;
}

// Y = (X-mean(X))/(sqrt(var(X)+eps))
int my_bn(float *input, float *mean, float *variance, float *scale,
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
        if (fabs(variance[ci]) <= eps && fabs(mean[ci]) <= 1e-8
            && fabs(input[ni * c * h * w + ci * h * w + i]) >= 1.0e-4
            && fabs(output[ni * c * h * w + ci * h * w + i]) >= 1.0e-2) {
          llvm::errs() << "WARNING: BN: var too small, i=" << i
                       << ", v=" << std::to_string(variance[ci])
                       << ", m=" << std::to_string(mean[ci])
                       << "\n               "
                       << ", i=" << std::to_string(input[ni * c * h * w + ci * h * w + i])
                       << ", x=" << std::to_string(x)
                       << ", d=" << std::to_string(d)
                       << ", o=" << std::to_string(output[ni * c * h * w + ci * h * w + i])
                       << "\n";
          //assert(0);
        }
      }
    }
  }
  return 0;
}

int my_scale(float *input, float *scale, float *bias,
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

int my_upsample(float *input, float *output,
    int n, int c, int ih, int iw, int scale) {
  int h = ih * scale;
  int w = iw * scale;
  for (int ni = 0; ni < n; ni++) {
    for (int ci = 0; ci < c; ci++) {
      for (int hi = 0; hi < h; hi++) {
        for (int wi = 0; wi < w; wi++) {
          int nwi = wi/scale;
          int nhi = hi/scale;
          int out_idx = (((ni * c + ci) * h) + hi) * w + wi;
          int in_idx = (((ni * c + ci) * (h / scale)) + nhi) * (w / scale) + nwi;
          output[out_idx] = input[in_idx];
        }
      }
    }
  }
  return 0;
}

int my_softmax2D(float *input, float *output, int n, int c) {
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

int my_softmax4D(float *input, float *output, int axis, const std::vector<int64_t>& shape) {
  int iter = 0;
  float* max_val = new float[shape[axis]];
  for (int i = 0; i <shape[axis]; ++i)
    max_val[i] = 0;

  // Only support axis == 1 so far, which means calculate softmax along C
  assert(axis == 1);
  for (int N = 0; N < shape[0]; ++N) {
    for (int H = 0; H < shape[2]; ++H) {
      for (int W = 0; W < shape[3]; ++W) {

        // find max and subtract the max to avoid numerical issues
        float max_val = 0;
        for (int C = 0; C < shape[1]; ++C) {
          iter = (N * shape[1] * shape[2] * shape[3]) 
            + (C * shape[2] * shape[3]) + (H * shape[3]) + W;

          max_val = std::max(input[iter], max_val);
        }

        // find softmax divisor
        float *ex = new float[shape[1]];
        float sum_of_ex = 0.0f;
        for (int C = 0; C < shape[1]; ++C) {
          iter = (N * shape[1] * shape[2] * shape[3]) 
            + (C * shape[2] * shape[3]) + (H * shape[3]) + W;

          float x = input[iter] - max_val;
          ex[C] = exp(x);
          sum_of_ex += ex[C];
        }

        // calculate softmax
        for (int C = 0; C < shape[1]; ++C) {
          iter = (N * shape[1] * shape[2] * shape[3]) 
            + (C * shape[2] * shape[3]) + (H * shape[3]) + W;

          output[iter] = ex[C] / sum_of_ex;
        }
        delete[] ex;
      }
    }
  }
  return 0;
}

int my_crop(float *input, float *output, long int *shape1, long int *shape2, long int *top_shape,
            int cur_dim, int *offsets, int *indices) {
  // for loop if dim is not last
  if (cur_dim + 1 < 4) {
    for (int i = 0; i < top_shape[cur_dim]; ++i) {
      indices[cur_dim] = i;
      my_crop(input, output, shape1, shape2, top_shape, cur_dim + 1, offsets,
              indices);
    }
  } else {
    std::vector<int> ind_red(cur_dim, 0);
    std::vector<int> ind_off(cur_dim + 1, 0);

    for (int j = 0; j < cur_dim; ++j) {
      ind_red[j] = indices[j];

      ind_off[j] = indices[j] + offsets[j];
    }
    ind_off[cur_dim] = offsets[cur_dim];

    int c = shape1[1];
    int h = shape1[2];
    int w = shape1[3];

    int c2 = shape2[1];
    int h2 = shape2[2];
    int w2 = shape2[3];

    int btm_offset =
        ((ind_off[0] * c + ind_off[1]) * h + ind_off[2]) * w + ind_off[3];

    int top_offset =
        ((ind_red[0] * c2 + ind_red[1]) * h2 + ind_red[2]) * w2;
    int offset = shape2[cur_dim] ;
    std::copy(input + btm_offset, input + btm_offset + offset,
              output + top_offset);
  }
  return 0;
}

int my_tanh(float *input, float *output,
    int n, int c, int h, int w) {
#ifdef DUMP_FLAG
  static int dump_idx = 0;
  std::string prefix = std::string("tanh") + std::to_string(dump_idx);
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
    output[i] = tanh(input[i]);
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
int my_eltwise(float *input_1, float *input_2, float *output, int n, int c,
               int h, int w, int op) {
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

int my_permute(float *input, float *output, const int input_shape_size,
    int in, int ic, int ih, int iw,
    int on, int oc, int oh, int ow,
    int order0, int order1, int order2, int order3) {

  /* This algorthem is referred to caffe permute layer */
  int count = in*ic*ih*iw;
  LLVM_DEBUG(llvm::errs() << "my_permute"<<"\n";);
  LLVM_DEBUG(llvm::errs() << "shap_size = "<<input_shape_size<<"\n";);
  std::vector<int> old_steps;
  std::vector<int> new_steps;
  std::vector<int> orders;


  old_steps.push_back(ic*ih*iw);
  old_steps.push_back(ih*iw);
  old_steps.push_back(iw);
  old_steps.push_back(1);

  new_steps.push_back(oc*oh*ow);
  new_steps.push_back(oh*ow);
  new_steps.push_back(ow);
  new_steps.push_back(1);

  orders.push_back(order0);
  orders.push_back(order1);
  orders.push_back(order2);
  orders.push_back(order3);

  for (int i = 0; i < count; ++i) {
    int old_idx = 0;
    int idx = i;
    for (int j = 0; j < input_shape_size; ++j) {
      int order = orders[j];
      old_idx += (idx / new_steps[j]) * old_steps[order];
      idx %= new_steps[j];
    }
     output[i] = input[old_idx];
  }
    return 0 ;
}

int my_normalize(float *input, float *output, bool across_spatial,
    int n, int c, int h, int w){

  float eps = 1.0e-5;
  if(!across_spatial){ // only ssd case currently
    auto spatial_dim = h*w;
    auto norm_ = c*h*w;
      for(int ni=0;ni<n;ni++)
        for(int i=0;i<spatial_dim;i++){
          auto value = 0;
          for(int ci=0;ci<c;ci++){
            value += pow(input[ni*norm_+ci*spatial_dim+i],2);
          }

          for(int ci=0;ci<c;ci++){
            output[ni*norm_+ci*spatial_dim+i] = input[ni*norm_+ci*spatial_dim+i]/sqrt(value + eps);
          }
        }
  }else{
    assert(0);
  }
  return 0;
}

int my_slice(float *input, float *output, int axis,
  std::vector<int64_t> input_shape, std::vector<int64_t> output_shape) {
  const int bottom_slice_axis = input_shape[axis];
  const int top_slice_axis = output_shape[axis];

  int num_slices = 1;
  for (int i = 0; i < axis; i++) {
    num_slices *= input_shape[i];
  }

  int slice_size = 1;
  for (uint32_t i = axis + 1; i < input_shape.size(); i++) {
    slice_size *= input_shape[i];
  }

  for (int n = 0; n < num_slices; ++n) {
    const int top_offset = n * top_slice_axis * slice_size;
    const int bottom_offset =
        (n * bottom_slice_axis) * slice_size;
    memcpy(output + top_offset, input + bottom_offset, sizeof(float) * top_slice_axis * slice_size);
  }

  return 0;
}

int my_power(float *input, float *output,
    int n, int c, int h, int w, float scale, float shift, float power) {
#ifdef DUMP_FLAG
  static int dump_idx = 0;
  std::string prefix = std::string("power") + std::to_string(dump_idx);
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
    output[i] = pow(scale * input[i] + shift, power);
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
