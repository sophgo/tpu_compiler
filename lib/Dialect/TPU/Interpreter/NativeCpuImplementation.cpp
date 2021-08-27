#include "tpuc/MachineInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"
#include "tpuc/NativeCpuImplementation.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FormatVariadic.h"

#include <numeric>
#include <functional>
#include <assert.h>
#include <chrono>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <climits>
#include <functional>

#include "mkldnn.hpp"
#include <math.h>

#define DEBUG_TYPE "native-cpu"
using namespace mkldnn;

//#define DUMP_FLAG


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
    int kh, int kw, int sh, int sw, int dh, int dw, int pt, int pb, int pl, int pr, int g, int pad_value) {
  std::shared_ptr<std::vector<float>> zero_bias = NULL;
  if (!bias) {
    zero_bias = std::make_shared<std::vector<float>>(oc, 0.0f);
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
                 << "pt: " << pt << " pb: " << pb << " pl: " << pl << " pr: " << pr
                 << " g: " << g << "\n";
    llvm::errs() << "n:" << n << " c: " << ic << " h:" << ih << " w:" << iw << "\n"
                << " oc: " << oc << " oh:" << oh << " ow:" << ow << "\n";
    llvm::errs() << "pad value: " << pad_value << "\n";
  );

  using tag = memory::format_tag;
  using dt = memory::data_type;

  engine eng(engine::kind::cpu, 0);
  stream s(eng);

  std::vector<primitive> net;
  std::vector<std::unordered_map<int, memory>> net_args;

  std::vector<float> input_after_pad;
  // mkldnn not support non zero padding
  // we handle it.
  if (pad_value != 0) {
    if (pt != 0 || pl != 0 || pb != 0 || pr != 0) {
      input_after_pad.resize(n * ic * (ih + pt + pb) * (iw + pl + pr));
      std::vector<int> pads = {0, 0, pt, pl, 0, 0, pb, pr};
      std::vector<int64_t> input_shape = {n, ic, ih, iw};
      my_pad_constant(input, input_after_pad.data(), input_shape, pads,
                      pad_value);
      input = input_after_pad.data();
      ih = ih + pt + pb;
      iw = iw + pl + pr;
      pt = 0;
      pb = 0;
      pl = 0;
      pr = 0;
    }
  }

  const memory::dim batch = n;
  memory::dims src_tz = { batch, ic, ih, iw };
  memory::dims weights_tz = (g != 1) ? memory::dims{g, oc/g, ic/g, kh, kw}
                                    : memory::dims{oc, ic, kh, kw};
  memory::dims bias_tz = { oc };
  memory::dims dst_tz = { batch, oc, oh, ow };
  memory::dims strides = { sh, sw };

  memory::dims padding_l = { pt, pl };
  memory::dims padding_r = { pb, pr };
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

// copy from caffe_cpu_interp2
void my_interp(const int channels,
    const float *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
    float *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2) {
  bool packed = false;

  assert(x1 >= 0 && y1 >= 0 && height1 > 0 && width1 > 0 && x2 >= 0 && y2 >= 0 && height2 > 0 && width2 > 0);
  assert(Width1 >= width1 + x1 && Height1 >= height1 + y1 && Width2 >= width2 + x2 && Height2 >= height2 + y2);

  // special case: just copy
  if (height1 == height2 && width1 == width2) {
    for (int h2 = 0; h2 < height2; ++h2) {
      const int h1 = h2;
      for (int w2 = 0; w2 < width2; ++w2) {
        const int w1 = w2;
        if (packed) {
          const float* pos1 = &data1[channels * ((y1 + h1) * Width1 + (x1 + w1))];
          float* pos2 = &data2[channels * ((y2 + h2) * Width2 + (x2 + w2))];
          for (int c = 0; c < channels; ++c) {
            pos2[0] = pos1[0];
            pos1++;
            pos2++;
          }
        }
        else {
          const float* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
          float* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
          for (int c = 0; c < channels; ++c) {
            pos2[0] = pos1[0];
            pos1 += Width1 * Height1;
            pos2 += Width2 * Height2;
          }
        }
      }
    }
    return;
  }
  const float rheight = (height2 > 1) ? static_cast<float>(height1 - 1) / (height2 - 1) : 0.f;
  const float rwidth = (width2 > 1) ? static_cast<float>(width1 - 1) / (width2 - 1) : 0.f;
  for (int h2 = 0; h2 < height2; ++h2) {
    const float h1r = rheight * h2;
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const float h1lambda = h1r - h1;
    const float h0lambda = float(1.) - h1lambda;
    for (int w2 = 0; w2 < width2; ++w2) {
      const float w1r = rwidth * w2;
      const int w1 = w1r;
      const int w1p = (w1 < width1 - 1) ? 1 : 0;
      const float w1lambda = w1r - w1;
      const float w0lambda = float(1.) - w1lambda;
      if (packed) {
        const float* pos1 = &data1[channels * ((y1 + h1) * Width1 + (x1 + w1))];
        float* pos2 = &data2[channels * ((y2 + h2) * Width2 + (x2 + w2))];
        for (int c = 0; c < channels; ++c) {
          pos2[0] =
            h0lambda * (w0lambda * pos1[0]            + w1lambda * pos1[channels * w1p]) +
            h1lambda * (w0lambda * pos1[channels * h1p * Width1] + w1lambda * pos1[channels * (h1p * Width1 + w1p)]);
          pos1++;
          pos2++;
        }
      }
      else {
        const float* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
        float* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
        for (int c = 0; c < channels; ++c) {
          pos2[0] =
            h0lambda * (w0lambda * pos1[0]            + w1lambda * pos1[w1p]) +
            h1lambda * (w0lambda * pos1[h1p * Width1] + w1lambda * pos1[h1p * Width1 + w1p]);
          pos1 += Width1 * Height1;
          pos2 += Width2 * Height2;
        }
      }
    }
  }
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

int my_pixelshuffle(float *input, float *output, int in, int ic, int ih, int iw,
                    int on, int oc, int oh, int ow, int upscale_factor, bool dcr_mode){
  int i_index = 0, o_index = 0, new_c = 0, new_h = 0, new_w = 0, r = upscale_factor;
  LLVM_DEBUG(llvm::errs() << "  in: " << in << ", ic: " << ic << ", ih: " << ih
                          << ", iw: " << iw << "\n";);
  LLVM_DEBUG(llvm::errs() << "  on: " << on << ", oc: " << oc << ", oh: " << oh
                          << ", ow: " << ow << "\n";);
  if(dcr_mode){
    for (int n = 0; n < in; n++) {
      for (int c = 0; c < ic; c++) {
        for (int h = 0; h < ih; h++) {
          for (int w = 0; w < iw; w++) {
            new_c = c % oc;
            new_h = h * r + static_cast<int>(floor((c / oc) / r));
            new_w = w * r + ((c / oc) % r);
            o_index =
                n * (oc * oh * ow) + new_c * (oh * ow) + new_h * ow + new_w;
            output[o_index] = input[i_index];
            i_index++;
          }
        }
      }
    }
  } else {
    for (int n = 0; n < in; n++) {
      for (int c = 0; c < ic; c++) {
        for (int h = 0; h < ih; h++) {
          for (int w = 0; w < iw; w++) {
            new_c = static_cast<int>(floor(c / (r * r)));
            new_h = h * r + (static_cast<int>(floor(c / r))) % r;
            new_w = w * r + (c % (r * r)) % r;
            o_index =
                n * (oc * oh * ow) + new_c * (oh * ow) + new_h * ow + new_w;
            output[o_index] = input[i_index];
            i_index++;
          }
        }
      }
    }
  }
  return 0;
}

int my_upsample(float *input, float *output, int n, int c, int ih, int iw,
                    int scale_h, int scale_w) {
  int h = ih * scale_h;
  int w = iw * scale_w;
  for (int ni = 0; ni < n; ni++) {
    for (int ci = 0; ci < c; ci++) {
      for (int hi = 0; hi < h; hi++) {
        for (int wi = 0; wi < w; wi++) {
          int nwi = wi/scale_w;
          int nhi = hi/scale_h;
          int out_idx = (((ni * c + ci) * h) + hi) * w + wi;
          int in_idx = (((ni * c + ci) * (h / scale_h)) + nhi) * (w / scale_w) + nwi;
          output[out_idx] = input[in_idx];
        }
      }
    }
  }
  return 0;
}

inline int crop_offset(const std::vector<int>& indices,long int *shape) {
  int offset = 0;
  for (int i = 0; i < 4; ++i) {
    offset *= shape[i];
    if ((int)indices.size() > i) {
      offset += indices[i];
    }
  }
  return offset;
}

int calc_dilute_hw (int h, int ins_h, int ins_h_l, int pad_h_b, int pad_h_t) {
  return (h - 1) * (ins_h + 1) + ins_h_l +
    1 + pad_h_t + pad_h_b;
}

void my_dilateActivation (float* input, float* output,
    int pad_h_t, int pad_h_b,
    int ins_h,   int ins_h_l,
    int pad_w_l, int pad_w_r,
    int ins_w,   int ins_w_l,
    int n, int c, int h, int w, int fill_constant) {
  int oh = calc_dilute_hw(h, ins_h, ins_h_l, pad_h_t, pad_h_b);
  int ow = calc_dilute_hw(w, ins_w, ins_w_l, pad_w_l, pad_w_r);
  assert(!ins_h_l && !ins_w_l);
  for (int in = 0; in < n; in++) {
    for (int ic = 0; ic < c; ic++) {
      for (int _oh = 0; _oh < oh; _oh++) {
        for (int _ow = 0; _ow < ow; _ow++) {
          int out_idx = (((in * c + ic) * oh) + _oh) * ow + _ow;
          int in_nc = (in * c + ic) * h * w;
          output[out_idx] = fill_constant; //dilate
          if (_ow % (ins_w+1) == 0 && _oh % (ins_h+1) == 0) {
            output[out_idx] = input[in_nc + (_oh / (ins_h+1)) * w + _ow / (ins_w+1)];
          }
        }
      }
    }
  }
}

int my_permute(float *input, float *output, int in, int ic, int ih, int iw,
               int order0, int order1, int order2, int order3) {
  int shape[4] = {in, ic, ih, iw};
  for (int n = 0; n < in; n++) {
    for (int c = 0; c < ic; c++) {
      for (int h = 0; h < ih; h++) {
        for (int w = 0; w < iw; w++) {
          int cur[4] = {n, c, h, w};
          int in_idx = w + h * iw + c * ih * iw + n * ic * ih * iw;
          int out_idx =
              cur[order3] + cur[order2] * shape[order3] +
              cur[order1] * shape[order3] * shape[order2] +
              cur[order0] * shape[order3] * shape[order2] * shape[order1];
          output[out_idx] = input[in_idx];
        }
      }
    }
  }
  return 0;
}


static float softplus_activate(float x) {
  return logf(expf(x) + 1);
}

static inline float tanh_activate (float x) {
  return (2 / (1 + expf(-2 * x)) - 1);
}

float my_mish_activate(float x_val) {
  return x_val * tanh_activate(softplus_activate(x_val));
}

// input shape (in_n, in_c, in_h, in_w)
// output_shape (out_n, out_c, out_h, out_w)
// pads (x0_begin, x1_begin, x2_begin, x3_begin, x0_end, x1_end, x2_end, x3_end)
//
// on = x0_begin + x0_end + in
// oc = x1_begin + x1_end + ic
// oh = x2_begin + x2_end + ih
// ow = x3_begin + x3_end + iw
int my_pad_constant(float *input, float *output,
                     std::vector<int64_t> &input_shape,
                     std::vector<int> &pads, float const_val) {
  int in = input_shape[0];
  int ic = input_shape[1];
  int ih = input_shape[2];
  int iw = input_shape[3];
  // int on = pads[0] + pads[4] + in;
  int oc = pads[1] + pads[5] + ic;
  int oh = pads[2] + pads[6] + ih;
  int ow = pads[3] + pads[7] + iw;

  int in_offset = 0;
  int out_offset = 0;
  auto pad_n_begin_size = pads[0] * oc * oh * ow;
  for (int in_idx = 0; in_idx < in ; in_idx++) {
    in_offset = in_idx * ic * ih * iw;
    auto pad_c_begin_size = pads[1] * oh * ow;
    out_offset = pad_n_begin_size + pad_c_begin_size +
                                    in_idx * oc * oh * ow;
    for (int ic_idx = 0; ic_idx < ic ; ic_idx++) {
      auto in_ic_offset = in_offset + ic_idx * ih * iw;
      auto out_oc_offset = out_offset + ic_idx * oh * ow;

      // padding h_top and h_bottom;
      int pad_h_size = oh * iw;
      std::vector<float> out_pad_h(pad_h_size, const_val);

      int pad_top_offset = pads[2] * iw;
      memcpy(out_pad_h.data() + pad_top_offset, input + in_ic_offset,
                                                ih * iw * sizeof(int));

      if ((pads[3] != 0) || (pads[7] != 0)) {
        int pad_hw_size = oh * ow;
        std::vector<float> out_pad_hw(pad_hw_size, const_val);

        for (int i = 0; i < oh; i++) {
          int offset = i * ow + pads[3];
          memcpy(out_pad_hw.data() + offset, out_pad_h.data() + i * iw,
                                             iw * sizeof(int));
        }
        memcpy(output + out_oc_offset, out_pad_hw.data(), pad_hw_size * sizeof(int));
      } else {
        memcpy(output + out_oc_offset, out_pad_h.data(), pad_h_size * sizeof(int));
      }
    }
  }
  return 0;
}

static void get_strides_from_shapes5d(
    int strides[5], const int shapes[5], int ws)
{
  strides[5 - 1] = ws;
  for (int i = 5 - 2; i >= 0; i--)
    strides[i] = shapes[i + 1] * strides[i + 1];
}

static int get_tensor5d_offset(
    int poss[5], const int strides[5])
{
  int offset = 0;
  for (int i = 0; i < 5; i++)
    offset += poss[i] * strides[i];

  return offset;
}

// input (n, ic, id, ih, iw)
// output (n, oc, od, oh, ow)
// weight (oc, ic, kd, kh, kw), pytorch
void conv3d_float_ref(float *input, float *weight, float *bias, float *output,
  int batch, int input_c, int input_d, int input_h, int input_w,
  int output_c, int output_d, int output_h, int output_w,
  int kernel_d, int kernel_h, int kernel_w,
  int stride_d, int stride_h, int stride_w,
  int dilation_d, int dilation_h, int dilation_w,
  int pad_d0, int pad_top, int pad_bottom,
  int pad_d1, int pad_left, int pad_right) {
  (void)pad_bottom;
  (void)pad_d1;
  (void)pad_right;

  int input_shapes[5] = {batch, input_c, input_d, input_h, input_w};
  int output_shapes[5] = {batch, output_c, output_d, output_h, output_w};

  //int kernel_shapes[5] = {output_c, kernel_d, kernel_h, kernel_w, input_c};
  int kernel_shapes[5] = {output_c, input_c, kernel_d, kernel_h, kernel_w};

  int input_strides[5];
  int output_strides[5];
  int kernel_strides[5];

  // input/output shape (n, c, d, h, w)
  get_strides_from_shapes5d(input_strides, input_shapes, sizeof(float));
  get_strides_from_shapes5d(output_strides, output_shapes, sizeof(float));

  // kernel shape (oc, ic, kd, kh, kw), pytorch
  get_strides_from_shapes5d(kernel_strides, kernel_shapes, sizeof(float));

  for (int i = 0; i < batch; ++i) {
    for (int oc = 0; oc < output_c; oc++) {
      for (int oz = 0; oz < output_d; oz++) {
        for (int oy = 0; oy < output_h; ++oy) {
          for (int ox = 0; ox < output_w; ++ox) {
            for (int ic = 0; ic < input_c; ++ic) {
              for (int kz = 0; kz < kernel_d; ++kz) {
                const int iz = oz * stride_d + kz * dilation_d - pad_d0;
                if (iz < input_d) {
                  for (int ky = 0; ky < kernel_h; ++ky) {
                    const int iy = oy * stride_h + ky * dilation_h - pad_top;
                    if (iy < input_h) {
                      for (int kx = 0; kx < kernel_w; ++kx) {
                        const int ix = ox * stride_w + kx * dilation_w - pad_left;
                        if (ix < input_w) {
                          int input_poss[5] = {i, ic, iz, iy, ix};
                          int input_offset = get_tensor5d_offset(input_poss, input_strides)
                                             / input_strides[5 - 1];

                          // pytorch (oc=1, ic=1, kd=1, kh=3, kw=3)
                          int kernel_poss[5] = {oc, ic, kz, ky, kx};

                          int kernel_offset =
                            get_tensor5d_offset(kernel_poss, kernel_strides)
                                / kernel_strides[5 - 1];

                          int output_poss[5] = {i, oc, oz, oy, ox};
                          int output_offset =
                            get_tensor5d_offset(output_poss, output_strides)
                                / output_strides[5 - 1];

                          output[output_offset] +=
                            input[input_offset] * weight[kernel_offset];
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  if (bias) {
    for (int i = 0; i < batch; ++i) {
      for (int oy = 0; oy < output_h; ++oy) {
        for (int ox = 0; ox < output_w; ++ox) {
          for (int oc = 0; oc < output_c; ++oc) {
            for (int od = 0; od < output_d; ++od) {
              int output_poss[5] = {i, oc, od, oy, ox};
              int output_offset =
                  get_tensor5d_offset(output_poss, output_strides)
                      / output_strides[5 - 1];
              output[output_offset] += bias[oc];
            }
          }
        }
      }
    }
  }
}

// input (n, c, id, ih, iw)
// weight (kd, kh, kw), stride (sd, sh, sw)
// output (n, c, od, oh, ow)
void pool3d_float_ref(float *input, float *output,
    int input_n, int input_c, int input_d, int input_h, int input_w,
    int output_d, int output_h, int output_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d0, int pad_d1,
    int pad_top, int pad_bot, int pad_left, int pad_right)
{
  (void)pad_d1;
  (void)pad_bot;
  (void)pad_right;

  int input_shapes[5] = {input_n, input_c, input_d, input_h, input_w};
  int input_strides[5];

  int output_shapes[5] = {input_n, input_c, output_d, output_h, output_w};
  int output_strides[5];

  // logical stride, in unit of float
  get_strides_from_shapes5d(input_strides, input_shapes, 1);
  get_strides_from_shapes5d(output_strides, output_shapes, 1);

  for (int i = 0; i < input_n; i++) {
    for (int c = 0; c < input_c; c++) {
      for (int oz = 0; oz < output_d; oz++) {
        for (int oy = 0; oy < output_h; oy++) {
          for (int ox = 0; ox < output_w; ox++) {
            float max_value = -std::numeric_limits<float>::infinity();

            for (int pd = 0; pd < kernel_d; pd++) {
              int iz = oz * stride_d + pd - pad_d0;
              for (int py = 0; py < kernel_h; py++) {
                int iy = oy * stride_h + py - pad_top;
                for (int px = 0; px < kernel_w; px++) {
                  int ix = ox * stride_w + px - pad_left;
                  if (iz < input_d && iy < input_h && ix < input_w) {
                    int poss[5] = {i, c, iz, iy, ix};
                    int input_offset = get_tensor5d_offset(poss, input_strides);
                    max_value = std::fmax(max_value, input[input_offset]);
                  }
                }
              }
            }

            int output_poss[5] = {i, c, oz, oy, ox};
            int output_offset = get_tensor5d_offset(output_poss, output_strides);
            output[output_offset] = max_value;
          }
        }
      }
    }
  }
}
