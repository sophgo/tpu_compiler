#include "mlir/Dialect/TPU/MachineInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"

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

// align cmodel
#include "bmkernel/bm1880v2/1880v2_fp_convert.h"

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
    int kh, int kw, int sh, int sw, int dh, int dw, int pt, int pb, int pl, int pr, int g) {
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
                 << "pt:" << pt << " pb:" << pb << "pl: " << pl << " pr:" << pr
                 << "g: " << g << "\n";
    llvm::errs() << "n:" << n << " c: " << ic << " h:" << ih << " w:" << iw << "\n"
                << " oc: " << oc << " oh:" << oh << " ow:" << ow << "\n"
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
  int ph_t = pt;
  int pw_l = pl;
  int ph_b = pb;
  int pw_r = pr;
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
    int kh, int kw, int sh, int sw, int pt, int pb, int pl, int pr, int g) {
  if (!bias) {
    auto zero_bias = new std::vector<float>(oc, 0.0f);
    bias = zero_bias->data();
  }

  // TODO - padding
  // assert(pt == 0);
  // assert(pl == 0);
  int ph_t = pt;
  int pw_l = pl;
  int ph_b = pb;
  int pw_r = pr;

  LLVM_DEBUG(
    llvm::errs() << "  i: (" << ih << "*" << iw << "), "
                 << "  o: (" << oh << "*" << ow << "), "
                 << "  k: (" << kh << "*" << kw << "), "
                 << "s: (" << sh << "*" << sw << "), "
                 << "pt: " << pt << " pb: " << pb << " pl: " << pl << " pr: " << pr
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
    bool is_avg, bool count_include_pad) {
  LLVM_DEBUG(llvm::errs() << "mkldnn_pool: "
                          << "  i: (" << ih << "*" << iw << "), "
                          << "o: (" << oh << "*" << ow << "), "
                          << "k: (" << kh << "*" << kw << "), "
                          << "s: (" << sh << ", " << sw << "), "
                          << "p: (" << pt << ", " << pb << ", " << pl << ", "
                          << pr << "), count_include_pad"
                          << ": " << count_include_pad << "\n";);

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
  auto pool_avg_algo = count_include_pad
                           ? algorithm::pooling_avg_include_padding
                           : algorithm::pooling_avg_exclude_padding;
  // pool desc
  auto pool_desc = pooling_forward::desc(
      prop_kind::forward_inference,
      is_avg ? pool_avg_algo : algorithm::pooling_max,
      user_src_memory.get_desc(), user_dst_memory.get_desc(), strides, kernel,
      padding_t_l, padding_b_r);
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

// int my_exp(float *input, float *output, int n, int c, int h, int w, bool is_bf16) {
//   LLVM_DEBUG(
//     llvm::errs() << "  n: " << n << ", c: " << c
//                  << ", h: " << h << ", w: " << w << "\n";
//   );

//   for (int i = 0; i < n * c * h * w; ++i) {
//     output[i] = exp(input[i]);
//   }

//   return 0;
// }

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 0.5 * tanh(0.5 * x) + 0.5;
}

int my_gru(float *input, float *output,
           float *weight, float *recurrence, float *bias, float *initial_h,
           int seq_len, int batch_size, int input_size, int hidden_size,
           bool b_bidirectional, bool b_linear_before_reset) {
  assert(b_bidirectional == false);
  assert(b_linear_before_reset == true);
  assert(batch_size == 1);
  assert(initial_h);
  // TODO: optimize gru implementation, ex: use mkldnn
  // weight: Concatenation of weight matrix for update, reset, and hidden gates. shape = [num_directions, 3*hidden_size, input_size]
  // recurrence: Concatenation of recurrence weight matrix for update, reset, and hidden gates
  // bias: Concatenation of Wb[update, reset, hidden gates] and Rb[update, reset, hidden gates], shape = [num_directions, 6*hidden_size]
  // initial_h: [num_directions, batch_size, hidden_size]

  // int num_directions = b_bidirectional ? 2 : 1;
  // int gate_weight_size = hidden_size * input_size;
  float* prev_hidden_state = initial_h; // ht
  float* update_gate = new float[hidden_size]; // zt
  float* reset_gate = new float[hidden_size]; // rt
  float* hidden_gate = new float[hidden_size]; // ht

  for (int t = 0; t < seq_len; ++t) {
    // zt = sigmoid(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
    // rt = sigmoid(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
    // ht = tanh(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset != 0
    // Wzrh: hidden_size * input_size
    // Rzrh: hidden_size * hidden_size
    // Xt: seq_len * batch_size * input_size
    float* xt = input + (t * input_size);

    for (int i = 0; i < hidden_size; ++i) {
      update_gate[i] = 0;
      reset_gate[i] = 0;
      hidden_gate[i] = 0;
    }

    for (int i = 0; i < hidden_size; ++i) {
      float* wz = weight + i * input_size;
      float* wr = weight + (hidden_size + i) * input_size;
      float* wh = weight + (2 * hidden_size + i) * input_size;
      float* rz = recurrence + i * hidden_size;
      float* rr = recurrence + (hidden_size + i) * hidden_size;

      for (int j = 0; j < input_size; ++j) {
        update_gate[i] += wz[j] * xt[j];
        reset_gate[i] += wr[j] * xt[j];
        hidden_gate[i] += wh[j] * xt[j];
      }

      for (int j = 0; j < hidden_size; ++j) {
        update_gate[i] += rz[j] * prev_hidden_state[j];
        reset_gate[i] += rr[j] * prev_hidden_state[j];
      }

      update_gate[i] = sigmoid(update_gate[i] + bias[i] + bias[3 * hidden_size + i]);
      reset_gate[i] = sigmoid(reset_gate[i] + bias[hidden_size + i] + bias[4 * hidden_size + i]);
      hidden_gate[i] += bias[2 * hidden_size + i];
    }

    // second part of hidden gate
    // (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset != 0
    for (int i = 0; i < hidden_size; ++i) {
      float* rh = recurrence + (2 * hidden_size + i) * hidden_size;
      float hidden_gate_acc = 0;

      for (int j = 0; j < hidden_size; ++j)
        hidden_gate_acc += rh[j] * prev_hidden_state[j];

      hidden_gate_acc += bias[5 * hidden_size + i];
      hidden_gate[i] = tanh(hidden_gate[i] + reset_gate[i] * hidden_gate_acc);
    }

    // Ht = (1 - zt) (.) ht + zt (.) Ht-1
    float* hidden_state = output + t * hidden_size;
    for (int i = 0; i < hidden_size; ++i)
      hidden_state[i] = ((1 - update_gate[i]) * hidden_gate[i]) + (update_gate[i] * prev_hidden_state[i]);

    prev_hidden_state = hidden_state;
  }
  delete[] update_gate;
  delete[] reset_gate;
  delete[] hidden_gate;
  return 0;
}

int my_lstm(float *input, float *output,
           float *weight, float *recurrence, float *bias, float *initial_h, float *initial_c,
           int seq_len, int batch_size, int input_size, int hidden_size,
           bool b_bidirectional) {
  assert(b_bidirectional == false);
  assert(batch_size == 1);
  // refer to the implementation of onnx backend
  // https://github.com/onnx/onnx/blob/v1.7.0/onnx/backend/test/case/node/lstm.py#L65-L86
  int num_direction = b_bidirectional ? 2 : 1;
  float* prev_hidden_state = initial_h; // ht_minus_1
  float* prev_cell_state = initial_c; // Ct_minus_1
  float* cell_state= output + (seq_len * num_direction * batch_size * hidden_size); // Ct

  int iofc_size = hidden_size * 4;
  float* wx_plus_b_plus_rx_plus_b = new float[iofc_size];
  float* input_gate = wx_plus_b_plus_rx_plus_b; // it
  float* output_gate = wx_plus_b_plus_rx_plus_b + hidden_size; // ot
  float* forget_gate = wx_plus_b_plus_rx_plus_b + hidden_size * 2; // ft
  float* cell_gate = wx_plus_b_plus_rx_plus_b + hidden_size * 3; // ct
  float* wb = bias;
  float* rb = bias + iofc_size;

  for (int t = 0; t < seq_len; ++t) {
    // it = sigmoid(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
    // ot = sigmoid(Xt*(Wo^T) + Ht-1*(Ro^T) + Wbo + Rbo)
    // ft = sigmoid(Xt*(Wf^T) + Ht-1*(Rf^T) + Wbf + Rbf)
    // ct = tanh(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
    // Ct = ft (.) Ct-1 + it (.) ct
    // Ht = ot (.) tanh(Ct)
    // Wiofc: 4 * hidden_size * input_size
    // Riofc: 4 * hidden_size * hidden_size
    // Xt: seq_len * batch_size * input_size
    float* xt = input + t * input_size;

    for (int i = 0; i < iofc_size; ++i) {
      float* w = weight + i * input_size;
      float* r = recurrence + i * hidden_size;

      wx_plus_b_plus_rx_plus_b[i] = wb[i] + rb[i];

      for (int j = 0; j < input_size; ++j)
        wx_plus_b_plus_rx_plus_b[i] += w[j] * xt[j];

      for (int j = 0; j < hidden_size; ++j)
        wx_plus_b_plus_rx_plus_b[i] += r[j] * prev_hidden_state[j];
    }

    float* hidden_state= output + (t * num_direction * batch_size * hidden_size); // ht

    for (int i = 0; i < hidden_size; ++i) {
      input_gate[i] = sigmoid(input_gate[i]);
      output_gate[i] = sigmoid(output_gate[i]);
      forget_gate[i] = sigmoid(forget_gate[i]);
      cell_gate[i] = tanh(cell_gate[i]);
      cell_state[i] = forget_gate[i] * prev_cell_state[i] + input_gate[i] * cell_gate[i];
      hidden_state[i] = output_gate[i] * tanh(cell_state[i]);
    }

    prev_hidden_state = hidden_state;
    prev_cell_state = cell_state;
  }

  delete[] wx_plus_b_plus_rx_plus_b;

  return 0;
}

int my_abs(float *input, float *output, int n, int c, int h, int w) {
#ifdef DUMP_FLAG
  static int dump_idx = 0;
  std::string prefix = std::string("abs") + std::to_string(dump_idx);
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
    output[i] = fabs(input[i]);
    //llvm::errs() << "  ["<<i<<"] s:" << input[i] << ", out:" << output[i] << "\n";
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

int my_avg_pooling(float *input, float *output, int n, int c, int ih, int iw,
                   int oh, int ow, int kh, int kw, int sh, int sw, int pt,
                   int pb, int pl, int pr) {
  // Todo: my case only has global average, if your model has other case,
  //       plz add and test
  assert(kh == ih && kw == iw); //global average
  for (int in = 0; in < n; ++in) {
    for (int i = 0; i < c; ++i) {
      float val = 0;
      for (int j = 0; j < kw * kh; ++j) {
        val += input[in * c * kh * kw +i * kh * kw + j];
      }
      output[in * c + i] = val;
    }
  }
  return 0;
}

int my_relu(float *input, float *output, int n, int c, int h, int w,
                float negative_slope) {
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

// align cmodel
extern const int BF16_TABLE_START;
extern const int BF16_TABLE_END;
void gen_bf16_table(int start, int end, int table_hw, float *table,
                           double (*activate_func)(double)) {
  int half = table_hw / 2;
  int table_idx = 0;
  int range = abs(end - start);
  float interval = (float)range / (float)table_hw;
  double x_value;
  double y_value;

  // Set idx [0 , 127] fp32 and bf16 data
  for (int i = 0; i < half; i++) {
    x_value = (start + end) / 2 + i * interval;
    y_value = activate_func(x_value);
    table[table_idx] = y_value;
    table_idx++;
  }

  // set idx 129 to 256, 2's complment
  for (int i = 0; i < half; i++) {
    x_value = start + i * interval;
    y_value = activate_func(x_value);
    table[table_idx] = y_value;
    table_idx++;
  }
}

void gen_bf16_slope_table(int start, int end, int table_hw,
                                         float *table,
                                         float *slope_table, double (*activate_func)(double)) {
  // int range = abs(end - start);
  // float interval = (float)range / (float)table_hw;
  int half = table_hw / 2;
  for (int i = 0; i < table_hw; ++i) {
    double x0 = table[i];
    double x1 = table[i + 1];
    double delta = 1.0;

    bool isSameBf16Value = false;
    unsigned int intX0 = convert_fp32_bf16(x0);
    unsigned int intX1 = convert_fp32_bf16(x1);
    if(intX0 == intX1) {
      isSameBf16Value = true;
    }

    float slope;
    if(isSameBf16Value || (i == half - 1) || (i == half)) {
      //Sean : DONNOT allow same bf16 value with non-zero slope
      //Sean : DONNOT allow extrapolation method
      slope = 0;
    } else {
      if (i > half) {
        x0 = table[i];
        x1 = table[i - 1];
        delta = -1.0;
      }
      slope = (x1 - x0) / delta;
    }
    slope_table[i] = slope;
  }
}

// <! gen reciprocal f(x) = 1/x
static double _gen_reciprocal(int base, int p) {
  // y = x ^ -1
  double f = (double) (pow(base, -1 * p));
  return f;
}


void bf16_gen_reciprocal(int start, int end, int table_hw, uint16_t *table_data) {

  int exp_start = start;
  int half = table_hw / 2;
  uint64_t idx = 0;

  // prepare channel 0
  // double s = 0.0;
  // 0^-1 is invalid, use positive/negtive max value: 0x7F7F / 0xFF7F
  table_data[idx] = 0x7F7F; //<! convert to 0x7F7F

  idx++;

  // > 0, exp from 0 -62 -61 ..  62  63
  for (int i = 0; i < half - 1; i++) {
    int shift = (exp_start + i);
    float exp = shift;

    double s = _gen_reciprocal(2, exp);
    //FloatToBFloat16((float*)&s,&table_data[idx],(size_t)1);
    table_data[idx] = convert_fp32_bf16(s);
    idx++;
  }

  table_data[idx] = 0xFF7F; //<! convert to 0x7F7F
  idx++;

  // < 0, exp from 0 -62 -61 ..  62  63
  for (int i = 0; i < half - 1; i++) {
    int shift = (exp_start + i);
    float exp = shift;

    double s = -1 * _gen_reciprocal(2, exp);
    //table_data[idx] = convert_fp32_bf16(s);
    //FloatToBFloat16((float*)&s,&table_data[idx],(size_t)1);
    table_data[idx] = convert_fp32_bf16(s);
    idx++;
  }
}

void bf16_gen_reciprocal_mantissa(int start, int end, int table_hw, uint16_t *table_mantissa) {
  int half = table_hw/2;

  int idx = 0;
  double d;
  for (int i = 0; i < half; i++) {
    d = 1 + i * 1 / 128.0;
    d = (double) pow(d, -1);
    table_mantissa[128+idx] = convert_fp32_bf16(d);
    //13=2^3x1.625=(2^2)x(2^1x1.625)
    table_mantissa[idx] = convert_fp32_bf16(d);
    idx++;
  }
}


// <! gen invert sqrt
static double _gen_sqrt(int base, int p) {
  // y = x ^ 0.5
  double f = (double) (pow(base, p * 0.5));
  return f;
}

void bf16_gen_sqrt(int start, int table_hw, uint16_t *table_data) {
  //<! 32*8 table, duplicate `channel` times;

  int half = table_hw / 2;
  uint64_t idx = 0;
  assert(half == 128);

  // prepare channel 0
  float s = 0.0;
  table_data[idx] = convert_fp32_bf16(s);
  idx++;

  // > 0, exp from 0 -62 -61 ..  62  63
  for (int i = 0; i < half; i++) {
    int shift = (start + i);
    bool is_odd = (shift % 2);
    float exp = shift;
    if (is_odd) {
      exp = exp - 1;
    }

    double s = _gen_sqrt(2, exp);
    //table_data[idx] = convert_fp32_bf16(s);
    //FloatToBFloat16((float*)&s,&table_data[idx],(size_t)1);
    table_data[idx] = convert_fp32_bf16(s);
    idx++;
  }
}

void bf16_gen_sqrt_mantissa(int table_hw, uint16_t* table_mantissa) {

  uint32_t half = table_hw  / 2;
  assert(half == 128);

  int idx = 0;
  double d;
  for (uint32_t i = 0; i < half; i++) {
    d = 1 + i * 1 / 128.0;
    d = (double) pow(d, 0.5);
    table_mantissa[128+idx] = convert_fp32_bf16(d);
    LLVM_DEBUG(llvm::errs() <<","<< "table_mantissa["<<idx+128<<"] = " <<table_mantissa[128+idx];);

    d = 2 * (1 + i * 1 / 128.0);

    d = (double) pow(d, 0.5);
    table_mantissa[idx] = convert_fp32_bf16(d);
    LLVM_DEBUG(llvm::errs() <<","<< "table_mantissa["<<idx<<"] = " <<table_mantissa[idx];);
    idx++;
  }
}

// gen power exp table
void bf16_gen_power_exp_table(uint16_t *table_data, float beta,
                              int start, int table_hw) {
  int exp_start = start;
  int half = table_hw/2;
  uint64_t idx = 0;

  table_data[idx] = 0x1; // power(0)
  idx++;

  // > 0, exp from -62 -61 ..  62  63
  for (int i = 0; i < half - 1; i++) {
    int shift = (exp_start + i);
    bool is_odd = (shift % 2);
    float exp = shift;
    if (is_odd) {
      exp = exp - 1;
    }

    double s = (double)(pow(2, (exp*(-beta))));
    //FloatToBFloat16((float*)&s,&table_data[idx],(size_t)1);
    table_data[idx] = convert_fp32_bf16(s);
    idx++;
  }

  table_data[idx] = 1; // power(-0)
  idx++;

  // < 0, exp from 0 -62 -61 ..  62  63
  for (int i = 0; i < half - 1; i++) {
    int shift = (exp_start + i);
    bool is_odd = (shift % 2);
    float exp = shift;
    if (is_odd) {
      exp = exp - 1;
    }

    double s = -1 * (double)(pow(2, (exp*(-beta))));
    table_data[idx] = convert_fp32_bf16(s);
    idx++;
  }
}

void bf16_gen_power_mantissa_table(uint16_t* table_mantissa, float beta,
                                   int table_hw) {
  int half = table_hw / 2;
  assert(half == 128);

  int idx = 0;
  double d;
  for (int i = 0; i < half; i++) {
    d = 1 + i * 1 / 128.0;
    d = (double) pow(d, (0-beta));
    table_mantissa[128+idx] = convert_fp32_bf16(d);
    LLVM_DEBUG(llvm::errs() <<","<< "table_mantissa["<<idx+128
                            <<"] = " <<table_mantissa[128+idx];);

    //13=2^3x1.625=(2^2)x(2^1x1.625)
    d = 2 * (1 + i * 1 / 128.0);
    d = (double) pow(d, (0-beta));
    table_mantissa[idx] = convert_fp32_bf16(d);
    LLVM_DEBUG(llvm::errs() <<","<< "table_mantissa["<<idx
                            <<"] = " <<table_mantissa[idx];);
    idx++;
  }
}

int my_lut_interpolation(float *input, float *output, int n, int c, int h, int w,
                         bool is_bf16, double (*activate_func)(double),
                         float thresh_min, float thresh_max, bool isExpFunc) {
  LLVM_DEBUG(llvm::errs() << "  n: " << n << ", c: " << c << ", h: " << h
                          << ", w: " << w << "\n";);
  int npu_num = 32; //<! 1880v2 hardcode

  //<! 1880v2 hw bf16 config
  int table_h = 32;
  int table_w = 8;
  std::vector<float> y0_table;
  std::vector<float> y0_slope_table; // use in bf16
  int table_hw = table_h * table_w;
  int tbl_shape = npu_num * table_hw;
  y0_table.resize(tbl_shape);
  y0_slope_table.resize(tbl_shape);
  std::vector<float> y0_fp32_table(table_hw);
  std::vector<float> y0_fp32_slope_table(table_hw);
  std::vector<uint16_t> y0_bf16_table(table_hw);
  std::vector<uint16_t> y0_bf16_slope_table(table_hw);

  gen_bf16_table(thresh_min, thresh_max, table_hw,
      y0_fp32_table.data(), activate_func);

  gen_bf16_slope_table(thresh_min, thresh_max, table_hw,
      y0_fp32_table.data(), y0_fp32_slope_table.data(),
      activate_func);

  if(isExpFunc) {
    // Make lut exp(x) = 0 when x <= -15
    y0_fp32_table[128] = 0;
  }

  for (int i = 0; i < table_hw; i++) {
    // convert fp32 to bf16
    y0_bf16_table.data()[i] = convert_fp32_bf16(y0_fp32_table.data()[i]);
    y0_bf16_slope_table.data()[i] = convert_fp32_bf16(y0_fp32_slope_table.data()[i]);
  }

  int shape_size =  n * c * h * w;
  int scale = 256 / (thresh_max - thresh_min); // quant from interger index range from 16(-8~8)->256(lut index size)

  // rounding
  scale = convert_bf16_fp32(convert_fp32_bf16(scale));
  float offset = (float)(thresh_max + thresh_min) / 2;
  for (int i = 0; i < shape_size; ++i) {
    if (!is_bf16) {
      output[i] = activate_func(input[i]);
    }
    else {
      float reOffset_input = convert_bf16_fp32(convert_fp32_bf16(input[i])) - offset;
      float rescale_input = convert_bf16_fp32(convert_fp32_bf16(reOffset_input)) * scale;
      uint16_t rescale_input_bf16 = convert_fp32_bf16(rescale_input);

      // get interger part to get table index and x0
      int rescale_input_i8 = _convert_bf16_s8(rescale_input_bf16, /*int8_rnd_mode=*/1);

      // get delta x (x - x0)
      float delta_x = rescale_input - rescale_input_i8;

      // get slope
      uint16_t slope = y0_bf16_slope_table[rescale_input_i8 & 0xff];

      // base y0 = f(x0)
      uint16_t base = y0_bf16_table[rescale_input_i8 & 0xff];

      // result = y0 + delta * slope
      float r = convert_bf16_fp32(base) + delta_x * convert_bf16_fp32(slope);
      output[i] = convert_bf16_fp32(convert_fp32_bf16(r));
    }
  }

  return 0;
}

int my_sigmoid(float *input, float *output, int n, int c, int h, int w, bool is_bf16) {
  double (*activate_func)(double);
  activate_func = sigmoid;
  my_lut_interpolation(input, output, n, c, h, w, is_bf16, activate_func, BF16_TABLE_START, BF16_TABLE_END, false);
  return 0;
}

int my_exp(float *input, float *output, int n, int c, int h, int w, bool is_bf16) {
  double (*activate_func)(double);
  activate_func = exp;
  const float threshMin = -15;
  const float threshMax = 1;
  my_lut_interpolation(input, output, n, c, h, w, is_bf16, activate_func, threshMin, threshMax, true);
  return 0;
}

int my_reciprocal(float *input, float *output, int n, int c, int h, int w, bool is_bf16) {
  const int expStart = -62;
  const int expEnd = 63;

  //<! 1880v2 hw bf16 config
  int table_h = 32;
  int table_w = 8;
  int table_hw = table_h * table_w;
  std::vector<uint16_t> table_data_lut_bf16(table_hw);
  std::vector<uint16_t> table_data_mantissa_lut_bf16(table_hw);
  bf16_gen_reciprocal(expStart, expEnd, table_hw, table_data_lut_bf16.data());
  bf16_gen_reciprocal_mantissa(expStart, expEnd, table_hw, table_data_mantissa_lut_bf16.data());
  int shape_size =  n * c * h * w;
  for (int i = 0; i < shape_size; i++) {
    if (!is_bf16) {
      output[i] = 1 / input[i];
    }
    else {
      uint16_t bf16InputValue = convert_fp32_bf16(input[i]);
      int exponentIndex;
      if (input[i] == 0) {
        exponentIndex = 0;
      }
      else if (input[i] >= 0) {
        exponentIndex = floor(log2(input[i]));
        exponentIndex = exponentIndex + 62 + 1; // 62 means start with 2^-62, index from 1
      }
      else {
        exponentIndex = floor(log2(-1 * input[i]));
        exponentIndex = exponentIndex + 62 + 129; // 62 means start with 2^-62, index from 129
      }
      float exponentFloatValue = convert_bf16_fp32(table_data_lut_bf16[exponentIndex]);
      float mantissaFloatValue = convert_bf16_fp32(table_data_mantissa_lut_bf16[bf16InputValue & 0xff]);
      output[i] = convert_bf16_fp32(convert_fp32_bf16(exponentFloatValue * mantissaFloatValue));
    }
  }

  return 0;
}

// \y0_bf16_slope_table and \y0_bf16_table occupy sizeof(float) and its content quanted as bf16 layout
static void hw_lut(float *input, float *output,
    int n, int c, int h, int w,
    float* y0_bf16_table, float* y0_bf16_slope_table,
    double (*activate_func)(double)) {
  int shape_size =  n * c * h * w;
  float scale = 256 / (BF16_TABLE_END - BF16_TABLE_START); // quant from interger index range from 16(-8~8)->256(lut index size)

  // rounding
  scale = convert_bf16_fp32(convert_fp32_bf16(scale));
  //std::for_each(std::execution::par, input.begin(), input.end(), [&](const size_t& i) {
  for (int i = 0; i < shape_size; ++i) {
    float rescale_input = convert_bf16_fp32(convert_fp32_bf16(input[i])) * scale;
    uint16_t rescale_input_bf16 = convert_fp32_bf16(rescale_input);

    // get interger part to get table index and x0
    int rescale_input_i8 = _convert_bf16_s8(rescale_input_bf16, /*int8_rnd_mode=*/1);

    // get delta x (x - x0)
    float delta_x = rescale_input - rescale_input_i8;

    // get slope
    uint16_t slope = y0_bf16_slope_table[rescale_input_i8 & 0xff];

    // base y0 = f(x0)
    uint16_t base = y0_bf16_table[rescale_input_i8 & 0xff];

    // result = y0 + delta * slope
    float r = convert_bf16_fp32(base) + delta_x * convert_bf16_fp32(slope);
    output[i] = convert_bf16_fp32(convert_fp32_bf16(r));
  }
  //});

}

int my_sigmoid(float *input, float *output, int n, int c, int h, int w,
    float* y0_bf16_table, float* y0_bf16_slope_table, bool is_bf16) {
  LLVM_DEBUG(llvm::errs() << "  n: " << n << ", c: " << c << ", h: " << h
                          << ", w: " << w << "\n";);

  if (!is_bf16) {
    int shape_size =  n * c * h * w;
    for (int i = 0; i < shape_size; ++i) {
      output[i] = sigmoid(input[i]);
    }
  }
  else {
    hw_lut(input, output, n, c, h, w, y0_bf16_table, y0_bf16_slope_table, sigmoid);
  }

  return 0;
}

// Y = (X-mean(X))/(sqrt(var(X)+variance_epsilon))
int my_bn(float *input, float *mean, float *variance, float *scale, float variance_epsilon,
    float *output, int n, int c, int h, int w) {
  float scale_factor = 1 / scale[0];
  for (int i = 0; i < c; ++i) {
    mean[i] = mean[i] * scale_factor;
    variance[i] = variance[i] * scale_factor;
  }
  for (int ni = 0; ni < n; ++ni) {
    for (int ci = 0; ci < c; ++ci) {
      for (int i = 0; i < h * w; ++i) {
        auto x = input[ni * c * h * w + ci * h * w + i] - mean[ci];
        auto d = sqrt(variance[ci] + variance_epsilon);
        output[ni * c * h * w + ci * h * w + i] = x / d;
        if (fabs(variance[ci]) <= variance_epsilon && fabs(mean[ci]) <= 1e-8
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
  for (int i = 0; i < c; ++i) {
    mean[i] = mean[i] * scale[0];
    variance[i] = variance[i] * scale[0];
  }
  return 0;
}

int my_pspnet_bn(float *input, float *slope, float *bias, float* mean, float* variance,
    float *output, float variance_epsilon, bool frozen, int n, int c, int h, int w) {
  if (frozen) {
    float scale[1] = {1};
    my_bn(input, mean, variance, scale, variance_epsilon,
        output, n, c, h, w);

    for (int ni = 0; ni < n; ++ni) {
      for (int ci = 0; ci < c; ++ci) {
        for (int i = 0; i < h * w; ++i) {
          auto x = output[ni * c * h * w + ci * h * w + i] * slope[ni*c + ci];
          output[ni * c * h * w + ci * h * w + i] = x + bias[ni*c + ci];
        }
      }
    }
  }
  else {
    llvm_unreachable("unsupported setting");
  }

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


template <typename Dtype>
static void array_mul(const int N, const Dtype *a, Dtype *b, Dtype *y) {
  for (int i = 0; i < N; i++) {
    y[i] = a[i] * b[i];
  }
}

template <typename Dtype>
static void array_axpy(const int N, const Dtype alpha, const Dtype *X,
                       Dtype *Y) {
  for (int i = 0; i < N; i++) {
    Y[i] = X[i] * alpha + Y[i];
  }
}

template <typename Dtype>
static void array_ax(const int N, const Dtype *X, const Dtype alpha, Dtype *Y) {
  for (int i = 0; i < N; i++) {
    Y[i] = X[i] * alpha;
  }
}

template <typename Dtype>
static void array_add(const int N, const Dtype *X, const Dtype alpha,
                      Dtype *Y) {
  for (int i = 0; i < N; i++) {
    Y[i] = X[i] + alpha;
  }
}

template <typename Dtype>
static void array_powx(const int N, const Dtype *a, const Dtype b, Dtype *y) {
  for (int i = 0; i < N; i++) {
    y[i] = std::pow(a[i], b);
  }
}

// lrn step one
int my_lrn_one(float *input, float *output, int n, int c, int h, int w,
               unsigned int local_size, float alpha) {
  int count = n * c * h * w;
  array_mul(count, input, input, output);
  array_ax(count, output, alpha / local_size, output);
  return 0;
}

// lrn step two
int my_lrn_two(float *input, float *output, int n, int c, int h, int w,
               unsigned int local_size) {
  int count = n * c * h * w;
  // start with the constant value
  for (int i = 0; i < count; ++i) {
    output[i] = 0;
  }
  int batch_size = c * h * w;
  int frame_size = h * w;
  int pre_pad = (local_size - 1) / 2;
  std::vector<float> padded_square((c + local_size - 1) * h * w, 0.0f);
  float *padded_square_data = padded_square.data();
  // go through the images
  for (int index_n = 0; index_n < n; ++index_n) {
    float *in_data = input + index_n * batch_size;
    float *out_data = output + index_n * batch_size;
    memcpy(padded_square_data + pre_pad * frame_size, in_data,
           batch_size * sizeof(float));
    for (uint32_t index_c = 0; index_c < local_size; ++index_c) {
      array_axpy(frame_size, 1.0f, padded_square_data + index_c * frame_size,
                 out_data);
    }
    for (int index_c = 1; index_c < c; ++index_c) {
      // copy previous scale
      memcpy(out_data + index_c * frame_size,
             out_data + (index_c - 1) * frame_size, frame_size * sizeof(float));
      // add head
      array_axpy(frame_size, 1.0f,
                 padded_square_data + (index_c + local_size - 1) * frame_size,
                 out_data + index_c * frame_size);
      // subtract tail
      array_axpy(frame_size, -1.0f,
                 padded_square_data + (index_c - 1) * frame_size,
                 out_data + index_c * frame_size);
    }
  }
  return 0;
}

// lrn step three
int my_lrn_three(float *input, float *output, int n, int c, int h, int w,
                 float beta, float k) {
  int count = n * c * h * w;
  array_add(count, input, k, output);
  array_powx(count, output, -beta, output);
  return 0;
}

// lrn step main
int my_lrn_main(float *input, float *scale, float *output, int n, int c, int h,
                int w) {
  int count = n * c * h * w;
  array_mul(count, scale, input, output);
  return 0;
}

int my_lrn_int8(float *input, float *output, int n, int c, int h, int w,
                unsigned int local_size, float *sqr_lut, float *power_lut,
                int sq_right_shift, int lrn_right_shift, int quant0,
                int quant1) {
  int count = n * c * h * w;
  int pre_pad = (local_size - 1) / 2;
  int padded_c = c + local_size - 1;
  int batch_size = c * h * w;
  int frame_size = h * w;
  std::vector<float> padded_square(padded_c * h * w, 0.0f);
  std::vector<float> scale(count, 0.0f);
  float *padded_square_data = padded_square.data();
  float *scale_data = scale.data();
  for (int i = 0; i < count; i++) {
    output[i] = 0;
  }
  float *square_data = padded_square_data + pre_pad * frame_size;
  for (int index_n = 0; index_n < n; index_n++) {
    float *in_ndata = input + index_n * batch_size;
    float *scale_ndata = scale_data + index_n * batch_size;
    for (int i = 0; i < batch_size; i++) {
      square_data[i] = sqr_lut[(uint8_t)in_ndata[i]];
    }
    for (uint32_t index_c = 0; index_c < local_size; ++index_c) {
      array_axpy(frame_size, (float)quant0,
                 padded_square_data + index_c * frame_size, scale_ndata);
    }
    for (int index_c = 1; index_c < c; ++index_c) {
      // copy previous scale
      memcpy(scale_ndata + index_c * frame_size,
             scale_ndata + (index_c - 1) * frame_size,
             frame_size * sizeof(float));
      // add head
      array_axpy(frame_size, (float)quant0,
                 padded_square_data + (index_c + local_size - 1) * frame_size,
                 scale_ndata + index_c * frame_size);
      // subtract tail
      array_axpy(frame_size, (float)-quant0,
                 padded_square_data + (index_c - 1) * frame_size,
                 scale_ndata + index_c * frame_size);
    }
  }
  float sq_scale = 1.0f / (1 << sq_right_shift);
  float lrn_scale = 1.0f / (1 << lrn_right_shift);
  for (int i = 0; i < count; i++) {
    scale_data[i] = std::floor(scale_data[i] * sq_scale + 0.5f);
    if (scale_data[i] < 0.0f) {
      scale_data[i] = 0.0f;
    } else if (scale_data[i] > 255.0f) {
      scale_data[i] = 255.0;
    }
    output[i] = power_lut[(uint8_t)scale_data[i]];
    output[i] *= input[i] * quant1 * lrn_scale;
    output[i] = std::floor(output[i] + 0.5f);
    if (output[i] < -128.0f) {
      output[i] = -128.0f;
    } else if (output[i] > 127.0f) {
      output[i] = 127.0f;
    }
  }
  return 0;
}

// shuffle channel
int my_shuffle_channel(float *input, float *output, unsigned int group, int n, int c,  int frame_size) {
    LLVM_DEBUG(llvm::errs() << "  n: " << n << ", c: " << c << ",  g: " << group
                          << ", f: " << frame_size << "\n";);
    const int batch_length = frame_size * c;
    int group_column = int(c/ group);
    if (c % group != 0) {
      llvm::errs() << "Error: Wrong group size, c=" << c << ", group =" << group;
      assert(0);
    }

    for(int i = 0; i < n; ++i)
    {
      float * p_in = input + i * batch_length;
      float * p_out = output + i * batch_length;
      for (uint32_t j = 0; j < group; ++j) // 2
      {
          for(int k = 0; k < group_column ; ++k) // 3
          {
              float* p_i = p_in + (j * group_column + k ) * frame_size;
              float* p_o = p_out + (k * group + j ) * frame_size;

              memcpy((void*)p_o, (void*)p_i, frame_size * sizeof(float) );
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

// swap channel
int my_swap_channel(float *input, float *output, int n, int c, int h, int w, int * order) {
  LLVM_DEBUG(llvm::errs() << "  n: " << n << ", c: " << c << ",  h: " << h
                          << ", w: " << w << "\n";);
  int frame_size = h * w;
  int batch_length = c * h * w;

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < c; j++) {
      float *p_in = input + i * batch_length + frame_size * order[j];
      float *p_out = output + i * batch_length + frame_size * j ;
      memcpy((void *)p_out, (void *)p_in, frame_size * sizeof(float));
    }
  }
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

int my_clip(float *input, float *output, int in, int ic, int ih, int iw,
                    int on, int oc, int oh, int ow, float min, float max) {
  int i_index = 0, o_index = 0;
  LLVM_DEBUG(llvm::errs() << "  in: " << in << ", ic: " << ic << ", ih: " << ih
                          << ", iw: " << iw << "\n";);
  LLVM_DEBUG(llvm::errs() << "  on: " << on << ", oc: " << oc << ", oh: " << oh
                          << ", ow: " << ow << "\n";);
  for (int n = 0; n < in; n++) {
    for (int c = 0; c < ic; c++) {
      for (int h = 0; h < ih; h++) {
        for (int w = 0; w < iw; w++) {
          // copy from caffe
          //top_data[i] = std::max(min, std::min(bottom_data[i], max));
          output[o_index] = std::max(min, std::min(input[i_index], max));
          o_index++;
          i_index++;
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

int my_softmax2D(float *input, float *output, int n, int c, bool is_bf16) {
#ifdef DUMP_FLAG
  static int dump_idx = 0;
  std::string prefix = std::string("softmax") + std::to_string(dump_idx);
  if (dump_idx == 0) {
    write_bianry_file(prefix + std::string("_in.bin"),
        (const char *)input, n * c * sizeof(float));
  }
#endif // DUMP_FLAG

  // find max and subtract the max to avoid numerical issues
  // float max_input = input[0];
  float *max_input = (float *)malloc(n * sizeof(float));
  float *ex = (float *)malloc(n * c * sizeof(float));
  float *tmp = (float *)malloc(n * c * sizeof(float));
  //init
  for (int index = 0; index < n; index++)
      max_input[index] = std::numeric_limits<float>::lowest();

  for (int ni = 0; ni < n; ++ni) {
    // find out max value
    for (int ci = 0; ci < c; ++ci) {
      int i = ni * c + ci;
      if (input[i] > max_input[ni])
        max_input[ni] = input[i];
    }

    // sub max value
    for (int ci = 0; ci < c; ++ci) {
      int i = ni * c + ci;
      tmp[i] = input[i] - max_input[ni];
    }
  }

  // do exp
  my_exp(tmp, ex, n, c, 1, 1, is_bf16);


  for (int ni = 0; ni < n; ++ni) {
    float sum_of_ex = 0.0f;
    for (int ci = 0; ci < c; ++ci) {
      int i = ni * c + ci;
      sum_of_ex += ex[i];
    }
    float reciprocalValue;
    my_reciprocal(&sum_of_ex, &reciprocalValue, 1, 1, 1, 1, is_bf16);
    for (int ci = 0; ci < c; ++ci) {
      int i = ni * c + ci;
      output[i] = ex[i] * reciprocalValue;
    }
  }
  free(ex);
  free(tmp);
  free(max_input);
#ifdef DUMP_FLAG
  if (dump_idx == 0) {
    write_bianry_file(prefix + std::string("_out.bin"),
        (const char *)output, n * c * sizeof(float));
  }
  dump_idx ++;
#endif // DUMP_FLAG
  return 0;
}


int my_softmax4D(float *input, float *output, int axis, const std::vector<int64_t>& shape, bool is_bf16) {
  int iter = 0;
  // Only support axis == 1 so far, which means calculate softmax along C
  assert(axis == 1);
  for (int N = 0; N < shape[0]; ++N) {
    for (int H = 0; H < shape[2]; ++H) {
      for (int W = 0; W < shape[3]; ++W) {

        // find max and subtract the max to avoid numerical issues
        float max_val = std::numeric_limits<float>::lowest();
        for (int C = 0; C < shape[1]; ++C) {
          iter = (N * shape[1] * shape[2] * shape[3])
            + (C * shape[2] * shape[3]) + (H * shape[3]) + W;

          max_val = std::max(input[iter], max_val);
        }

        // find softmax divisor
        float *ex = new float[shape[1]];
        float *tmp = new float[shape[1]];
        for(int C = 0; C < shape[1]; ++C) {
          iter = (N * shape[1] * shape[2] * shape[3])
            + (C * shape[2] * shape[3]) + (H * shape[3]) + W;
          tmp[C] = input[iter] - max_val;
        }
        // do exp
        my_exp(tmp, ex, 1, shape[1], 1, 1, is_bf16);
        float sum_of_ex = 0.0f;
        for (int C = 0; C < shape[1]; ++C) {
          sum_of_ex += ex[C];
        }
        float reciprocalValue;
        my_reciprocal(&sum_of_ex, &reciprocalValue, 1, 1, 1, 1, is_bf16);

        // calculate softmax
        for (int C = 0; C < shape[1]; ++C) {
          iter = (N * shape[1] * shape[2] * shape[3])
            + (C * shape[2] * shape[3]) + (H * shape[3]) + W;

          output[iter] = ex[C] * reciprocalValue;
        }
        delete[] ex;
        delete[] tmp;
      }
    }
  }
  return 0;
}

int my_softmax3D(float *input, float *output, int axis, const std::vector<int64_t>& shape, bool is_bf16) {
  assert(shape.size() == 3);
  int c = shape[0];
  int h = shape[1];
  int w = shape[2];
  //just for axis = 2 now
  assert(axis == 2);

  auto tmp_resultT = std::make_unique<std::vector<float> >(w);
  float *tmp = (float *)tmp_resultT.get()->data();

  for(int ci = 0; ci < c; ci++) {
    for(int hi = 0; hi < h; hi++) {
      for(int wi = 0; wi < w; wi++) {
        tmp[wi] = input[ci * w * h + hi * w + wi];
      }

      int ret = my_softmax2D(tmp, tmp, 1, w, is_bf16);
      assert(ret == 0);
      for(int wi = 0; wi < w; wi++) {
        output[ci * w * h + hi * w + wi] = tmp[wi];
      }
    }  //end for hi
  } //end for ci
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

int my_crop(float *input, float *output, long int *input_shape, long int *output_shape,
            int cur_dim, int *offsets, int *indices) {
  // for loop if dim is not last
  if (cur_dim + 1 < 4) {
    for (int i = 0; i < output_shape[cur_dim]; ++i) {
      indices[cur_dim] = i;
      my_crop(input, output, input_shape, output_shape, cur_dim + 1, offsets,
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

    memcpy(output + crop_offset(ind_red, output_shape),
           input + crop_offset(ind_off, input_shape),
           sizeof(float) * output_shape[cur_dim]);
  }
  return 0;
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

int my_interptile(float *input, float *output, int n, int c, int h, int w,
    int _ih, int _iw) {
  /**
   * input:   output: (5x5)
   * (3x3)
   * 0 1 2    0.0 1.0 0.0 1.0 1.0 2.0 1.0 2.0 2.0 2.0
   * 3 4 5 -> 3.0 4.0 3.0 4.0 4.0 5.0 4.0 5.0 5.0 5.0
   * 6 7 8    0.0 1.0 0.0 1.0 1.0 2.0 1.0 2.0 2.0 2.0
   *          3.0 4.0 3.0 4.0 4.0 5.0 4.0 5.0 5.0 5.0
   *          3.0 4.0 3.0 4.0 4.0 5.0 4.0 5.0 5.0 5.0
   *          6.0 7.0 6.0 7.0 7.0 8.0 7.0 8.0 8.0 8.0
   *          3.0 4.0 3.0 4.0 4.0 5.0 4.0 5.0 5.0 5.0
   *          6.0 7.0 6.0 7.0 7.0 8.0 7.0 8.0 8.0 8.0
   *          6.0 7.0 6.0 7.0 7.0 8.0 7.0 8.0 8.0 8.0
   *          6.0 7.0 6.0 7.0 7.0 8.0 7.0 8.0 8.0 8.0
   */
  // for last one
  int scale_h = (h - 1) / (_ih - 1);
  int scale_w = (w - 1) / (_iw - 1);
  h = h * 2;
  w = w * 2;
  for (int in = 0; in < n; in++) {
    for (int ic = 0; ic < c; ic++) {
      for (int ih = 0; ih < h; ih++) {
        for (int iw = 0; iw < w; iw++) {
          int nc = (in * c + ic) * _ih * _iw;
          int out_idx = (((in * c + ic) * h) + ih) * w + iw;

          int riw = iw;
          int rih = ih;
          if (riw >= (int)(scale_w * (_iw-1) * 2)) {
            // last w
            riw = (int)(scale_w * (_iw-1) * 2) - 1;
          }
          if (rih >= (int)(scale_h * (_ih-1) * 2)) {
            // last h
            rih = (int)(scale_h * (_ih-1) * 2) - 1;
          }

          output[out_idx] = input[nc + ((rih / (scale_h * 2) + rih % 2) * _iw) + riw / (scale_w * 2) + riw % 2];
        }
      }
    }
  }
  return 0;
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

// mish, copy from caffe
inline float tanh_activate (float x) {
  return (2 / (1 + expf(-2 * x)) - 1);
}

inline float softplus_activate (float x, float threshold) {
  if (x > threshold) return x;                // too large
  else if (x < -threshold) return expf(x);    // too small
  return logf(expf(x) + 1);
}

float my_mish_caffe(float x_val, float mish_threshold) {
  return x_val * tanh_activate(softplus_activate(x_val, mish_threshold));
}

static float my_mish_wrapper_threshold;
double my_mish_wrapper(double x_val) {
  return my_mish_caffe(x_val, my_mish_wrapper_threshold);
}

int my_mish(float *input, float *output, int n, int c, int h, int w, bool is_bf16, float mish_threshold) {
  LLVM_DEBUG(llvm::errs() << "  n: " << n << ", c: " << c << ", h: " << h
                          << ", w: " << w << "\n";);
  int npu_num = MInfo::lane_num;

  //<! 1880v2 hw bf16 config
  int table_h = 32;
  int table_w = 8;
  std::vector<float> y0_table;
  std::vector<float> y0_slope_table; // use in bf16
  int table_hw = table_h * table_w;
  int tbl_shape = npu_num * table_hw;
  y0_table.resize(tbl_shape);
  y0_slope_table.resize(tbl_shape);
  std::vector<float> y0_fp32_table(table_hw);
  std::vector<float> y0_fp32_slope_table(table_hw);
  std::vector<uint16_t> y0_bf16_table(table_hw);
  std::vector<uint16_t> y0_bf16_slope_table(table_hw);

  // use function pointer
  double (*activate_func)(double);
  my_mish_wrapper_threshold = mish_threshold;
  activate_func = my_mish_wrapper;

  gen_bf16_table(BF16_TABLE_START, BF16_TABLE_END, table_hw,
      y0_fp32_table.data(), activate_func);

  gen_bf16_slope_table(BF16_TABLE_START, BF16_TABLE_END, table_hw,
      y0_fp32_table.data(), y0_fp32_slope_table.data(),
      activate_func);

  for (int i = 0; i < table_hw; i++) {
    // convert fp32 to bf16
    y0_bf16_table.data()[i] = convert_fp32_bf16(y0_fp32_table.data()[i]);
    y0_bf16_slope_table.data()[i] = convert_fp32_bf16(y0_fp32_slope_table.data()[i]);
  }

  int shape_size =  n * c * h * w;
  int scale = 256 / (BF16_TABLE_END - BF16_TABLE_START); // quant from interger index range from 16(-8~8)->256(lut index size)

  // rounding
  scale = convert_bf16_fp32(convert_fp32_bf16(scale));

  for (int i = 0; i < shape_size; ++i) {
    if (!is_bf16) {
      output[i] = my_mish_caffe(input[i], mish_threshold);
    }
    else {
      float rescale_input = convert_bf16_fp32(convert_fp32_bf16(input[i])) * scale;
      uint16_t rescale_input_bf16 = convert_fp32_bf16(rescale_input);

      // get interger part to get table index and x0
      int rescale_input_i8 = _convert_bf16_s8(rescale_input_bf16, /*int8_rnd_mode=*/1);

      // get delta x (x - x0)
      float delta_x = rescale_input - rescale_input_i8;

      // get slope
      uint16_t slope = y0_bf16_slope_table[rescale_input_i8 & 0xff];

      // base y0 = f(x0)
      uint16_t base = y0_bf16_table[rescale_input_i8 & 0xff];

      // result = y0 + delta * slope
      float r = convert_bf16_fp32(base) + delta_x * convert_bf16_fp32(slope);
      output[i] = convert_bf16_fp32(convert_fp32_bf16(r));
    }
  }

  return 0;
}
int my_normalize(float *input,float *scale, float *output,
    bool across_spatial,bool channel_shared,
    int n, int c, int h, int w){
  float eps=1.0e-5;
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
              output[ni*norm_+ci*spatial_dim+i] = (input[ni*norm_+ci*spatial_dim+i]/sqrt(value + eps))*scale[ci];
          }
        }
  }else{
    assert(0);
  }
  return 0;
}

int my_slice(float *input, float *output, int axis, int offset,
    std::vector<int64_t> input_shape, std::vector<int64_t> output_shape) {
  int osz = 1;
  for (int i = 0; i < axis; i++) {
    osz *= input_shape[i];
  }
  int isz = 1;
  for (unsigned i = axis + 1; i < input_shape.size(); i++) {
    isz *= input_shape[i];
  }
  int axis_total_size = input_shape[axis];
  int axis_slice_size = output_shape[axis];

  for (int n = 0; n < osz; ++n) {
    int output_offset = n * axis_slice_size * isz;
    int input_offset = n * axis_total_size * isz + offset * isz;
    memcpy(output + output_offset, input + input_offset,
           sizeof(float) * axis_slice_size * isz);
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

int my_preprocess(float *input, float *output,
                  int n, int c, int h, int w,
                  const std::vector<int>& channel_order,
                  const std::vector<float>&mean,
                  const std::vector<float>&std,
                  float raw_scale, float input_scale) {
  int csz = h * w;
  int isz = c * h * w;
  int count = n * c * h * w;
  float *p = input;
  float *q = output;

  if (channel_order.size()) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < c; j++) {
        memcpy(q + channel_order[j] * csz,
               p + j * csz, csz * sizeof(float));
      }
      p += isz;
      q += isz;
    }
    p = q = output;
  }

  for (int i = 0; i < count; i++) {
    float val = *p++;
    if (raw_scale != 0) {
      val *= (raw_scale / 255);
    }
    if (mean.size()) {
      val -= mean[(i / csz) % c];
    }
    if (std.size()) {
      val /= std[(i / csz) % c];
    }
    if (input_scale != 0) {
      val *= input_scale;
    }
    *q++ = val;
  }
  return 0;
}

int my_transpose(float *input, float *output, int n, int c, int h, int w) {
  int csz = h * w;
  int isz = c * csz;

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < isz; j++) {
      int x_idx = i * isz + j;
      int y_idx = i * isz + (j % c) * csz + j / c;
      output[y_idx] = input[x_idx];
    }
  }
  return 0;
}

int my_reorg(float *input, float *output, uint32_t stride, int n, int c, int h, int w) {
  int out_c = c / (stride * stride);
  int out_w = w * stride;
  int out_h = h * stride;
  for (int b = 0; b < n; b++) {
    for (int k = 0; k < c; k++) {
      for (int j = 0; j < h; j++) {
        for (int i = 0; i < w; i++) {
          int in_index = i + w * (j + h * (k + c * b));
          int c2 = k % out_c;
          int offset = k / out_c;
          int w2 = i * stride + offset % stride;
          int h2 = j * stride + offset / stride;
          int out_index = w2 + out_w * (h2 + out_h * (c2 + out_c * b));
          output[in_index] = input[out_index];
        }
      }
    }
  }
  return 0;
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

int my_roipooling(float *data, float *rois, float *output, int pooled_h, int pooled_w,
                  float spatial_scale, int batch, int num_rois, int channel, int height, int width) {
  for (int b = 0; b < batch; ++b) {
    auto batched_rois = rois + b * num_rois * 5;
    auto batched_output = output + b * num_rois * channel * pooled_h * pooled_w;
    for (int n = 0; n < num_rois; ++n) {
      int roi_batch_ind = batched_rois[0];
      int roi_start_w = std::round(batched_rois[1] * spatial_scale);
      int roi_start_h = std::round(batched_rois[2] * spatial_scale);
      int roi_end_w = std::round(batched_rois[3] * spatial_scale);
      int roi_end_h = std::round(batched_rois[4] * spatial_scale);
      assert(roi_batch_ind < batch);

      int roi_height = std::max(roi_end_h - roi_start_h + 1, 1);
      int roi_width = std::max(roi_end_w - roi_start_w + 1, 1);
      const float bin_size_h = static_cast<float>(roi_height)
                              / static_cast<float>(pooled_h);
      const float bin_size_w = static_cast<float>(roi_width)
                              / static_cast<float>(pooled_w);

      float* batch_data = data + roi_batch_ind * channel * height * width;

      for (int c = 0; c < channel; ++c) {
        for (int ph = 0; ph < pooled_h; ++ph) {
          for (int pw = 0; pw < pooled_w; ++pw) {
            // Compute pooling region for this output unit:
            //  start (included) = floor(ph * roi_height / pooled_height_)
            //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
            int hstart = static_cast<int>(std::floor(static_cast<float>(ph)
                                                * bin_size_h));
            int wstart = static_cast<int>(std::floor(static_cast<float>(pw)
                                                * bin_size_w));
            int hend = static_cast<int>(std::ceil(static_cast<float>(ph + 1)
                                            * bin_size_h));
            int wend = static_cast<int>(std::ceil(static_cast<float>(pw + 1)
                                            * bin_size_w));

            hstart = std::min(std::max(hstart + roi_start_h, 0), height);
            hend = std::min(std::max(hend + roi_start_h, 0), height);
            wstart = std::min(std::max(wstart + roi_start_w, 0), width);
            wend = std::min(std::max(wend + roi_start_w, 0), width);

            bool is_empty = (hend <= hstart) || (wend <= wstart);

            const int pool_index = ph * pooled_w + pw;
            if (is_empty) {
              batched_output[pool_index] = 0;
            }

            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int index = h * width + w;
                if (batch_data[index] > batched_output[pool_index]) {
                  batched_output[pool_index] = batch_data[index];
                }
              }
            }
          }
        }
        batch_data += height * width;
        batched_output += pooled_h * pooled_w;
      }
      batched_rois += 5;
    }
  }
  return 0;
}

inline int count(std::vector<int64_t> &shape, int start_axis, int end_axis) {
    int64_t count = 1;
    for (int i = start_axis; i < end_axis; ++i) {
      count *= shape[i];
    }
    return count;
}

int my_reduce_mean(float *input, float *output,
                     std::vector<int64_t> &org_input_shape,
                     std::vector<int> &axes) {
  assert(axes.size() > 0);
  auto input_shape = org_input_shape;
  int size = count(input_shape, 0, input_shape.size());
  std::vector<float> tmp (size, 0);
  float* _output = tmp.data();

  for (int i = 0; i < (int)axes.size(); i++) {
    int dim = input_shape.size();
    int axis = axes[i];
    assert(dim > axis);

    int inner = count(input_shape, axis + 1, input_shape.size());
    int next_inner = inner * input_shape[axis];
    int outer = count(input_shape, 0, axis);

    for (int i = 0; i < outer; i++) {
      std::vector<float> inner_sum (inner, 0);
      for (int s = 0; s < input_shape[axis]; s++) {
        for (int j = 0; j < inner; j++) {
          inner_sum[j] += input[i * next_inner + s * inner + j];
        }
      }

      // mean
      for (int j = 0; j < inner; j++) {
        _output[i * inner + j]  = inner_sum[j] / input_shape[axis];
      }
    }

    input_shape[axis] = 1;
    input = _output;
  }

  // export
  size = count(input_shape, 0, input_shape.size());
  std::copy(_output, _output + size, output);

  return 0;
}

int my_reduce_mean_int8(float *input, float *output,
                        std::vector<int64_t> &org_input_shape,
                        std::vector<int> &axes,
                        int avg_const, int rshift) {
  assert(axes.size() > 0);
  auto input_shape = org_input_shape;
  int size = count(input_shape, 0, input_shape.size());
  std::vector<int> tmp (size, 0);
  int* _output = tmp.data();
  std::vector<int> tmp2 (size, 0);
  int* _input = tmp2.data();

  // Convert integer format
  for (int i = 0; i < size; i++)
    _input[i] = (int)input[i];

  for (int i = 0; i < (int)axes.size(); i++) {
    int dim = input_shape.size();
    int axis = axes[i];
    assert(dim > axis);

    int inner = count(input_shape, axis + 1, input_shape.size());
    int next_inner = inner * input_shape[axis];
    int outer = count(input_shape, 0, axis);

    llvm::outs() << "  [" << i << "] inner " << inner
                 << ", outer " << outer << "\n";

    for (int i = 0; i < outer; i++) {
      std::vector<int> inner_sum (inner, 0);
      for (int s = 0; s < input_shape[axis]; s++) {
        for (int j = 0; j < inner; j++) {
          inner_sum[j] += _input[i * next_inner + s * inner + j] * avg_const;
        }
      }

      // quantization down
      for (int j = 0; j < inner; j++) {
        int val = inner_sum[j];
        val >>= rshift - 1;
        val += 1; // round up
        val >>= 1;
        val = std::max(val, -128);
        val = std::min(val, 127);
        _output[i * inner + j] = val;
      }

    }

    input_shape[axis] = 1;
    _input = _output;
  }


  // Store float format
  size = count(input_shape, 0, input_shape.size());
  for (int i = 0; i < size; i++)
    output[i] = (float)_output[i];

  return 0;
}

int my_reduce_max(float *input, float *output,
                     std::vector<int64_t> &input_shape,
                     std::vector<int> &axes) {
  assert(axes.size() > 0);
  int axis = axes[0];
  // only support one axis, if has two axis, should be continous
  int total = count(input_shape, 0, input_shape.size());
  int n = count(input_shape, 0, axis);
  int c = input_shape[axis];
  int hw = total / (n*c);

  for (int nidx = 0; nidx < n; nidx++) {
    for (int inner_idx = 0; inner_idx < hw; inner_idx++) {
      for (int cidx = 0; cidx < c; cidx++) {
        float tmp = input[nidx * c * hw + cidx * hw + inner_idx];
        if (cidx == 0)
          output[nidx * hw + inner_idx] = tmp;
        output[nidx * hw + inner_idx] = std::max(tmp, output[nidx * hw + inner_idx]);
      }
    }
  }
  return 0;
}

int my_tile(float *input, float *output, std::vector<int64_t> &input_shape,
            std::vector<int64_t> &output_shape, std::vector<int32_t> &resp) {
  int axis = 0;
  int tiles = 0;
  auto iter = resp.begin();
  for (int i = 0; iter != resp.end(); ++iter, ++i) {
    if (*iter != 1) {
      axis = i;
      tiles = *iter;
      break;
    }
  }

  int outer_count = std::accumulate(input_shape.begin(), input_shape.begin() + axis,
                        1, std::multiplies<int>());
  int inner_count = std::accumulate(input_shape.begin() + axis, input_shape.end(),
                        1, std::multiplies<int>());

  llvm::errs() << "axis: " << axis << ", " << tiles << "\n";
  llvm::errs() << "tile input_shape: (" << input_shape[0] << ", " << input_shape[1]
                           << ", " << input_shape[2] << ", " << input_shape[3] << ")" << "\n";
  int input_offset = 0;
  int output_offset = 0;
  for (int out = 0; out < outer_count; ++out) {
    for (int t = 0; t < tiles; ++t) {
      auto start = input + input_offset;
      auto end = start + inner_count;
      std::copy(start, end, output + output_offset);
      output_offset += inner_count;
    }
    input_offset += inner_count;
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
