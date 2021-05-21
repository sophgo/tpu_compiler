#include "tpuc/Interpreter/cpu/pooling.hpp"
#include "mkldnn.h"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Interpreter/cpu/conv.hpp"
#include "tpuc/Interpreter/cpu/slice.hpp"
#include "tpuc/MachineInfo.h"
#include "tpuc/ModuleInterpreter.h"
#include "tpuc/Interpreter/cpu/activation.hpp"

namespace mlir {

PoolingOpKernel::PoolingOpKernel(Operation &op, value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  tpu::PoolParam pool_param;
  bool is_avg;
  if (isa<tpu::PoolAvg2DOp>(op)) {
    auto poolavgOp = cast<tpu::PoolAvg2DOp>(op);
    pool_method = POOL_METHOD::AVG;
    pool_param = poolavgOp.param();
    is_avg = true;
  } else if (isa<tpu::PoolMax2DOp>(op)) {
    auto poolmaxOp = cast<tpu::PoolMax2DOp>(op);
    pool_method = POOL_METHOD::MAX;
    pool_param = poolmaxOp.param();
    is_avg = false;
  }

  this->input_shape = getTensorShape(op.getOperand(0));
  this->is_asymmetric = isOpQuantAsymmetric(&op);
  if (!is_asymmetric && pad_value != 0) {
    llvm::errs() << "pad value:" << pad_value << "\n";
    llvm_unreachable("symmetric pad is zero");
  }

  auto input_value = op.getOperand(0);
  auto result = op.getResult(0);
  parsePoolParam(pool_param, input_value, result, n, c, ih, iw, oh, ow, kh,
                 kw, sh, sw, pt, pb, pl, pr, pad_value, is_global, do_relu,
                 count_include_pad);
  is_asymmetric = isOpQuantAsymmetric(&op);

  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
  using tag = mkldnn::memory::format_tag;
  using dt = mkldnn::memory::data_type;

  this->mkl_eng = mkldnn::engine(mkldnn::engine::kind::cpu, 0);
  this->mkl_stream = mkldnn::stream(mkl_eng);

  if (datatype == DataType::INT8 && pool_method == POOL_METHOD::AVG) {
    // in int8 average pool case:
    // SyQy = Avg(SxQx)
    // Qy = 1/Sy *  Sx * Qxi * 1 / (kh * kw)
    // mkldnn pool can not simulate this case,
    // we use detphwise conv,  use mutlipiler and rshift to handle 1 / (kh* kw)
    // clean mkldn and reset
    auto quant_rshift = this->opdTensors[3];
    auto quant_multiplier = this->opdTensors[4];
    if (!quant_rshift) {
      llvm_unreachable("quant_rshift is null!");
    }
    if (!quant_multiplier) {
      llvm_unreachable("quant_multiplier is null!");
    }
    this->rshift = quant_rshift->at(0);
    this->multiplier = quant_multiplier->at(0);
    set_i8_avg_mkldnn();
  } else {
    mkldnn::memory::dims mkl_src_shape = {n, c, ih, iw};
    mkldnn::memory::dims mkl_dst_shape = {n, c, oh, ow};
    mkldnn::memory::dims mkl_strides = {sh, sw};
    mkldnn::memory::dims mkl_kernel = {kh, kw};
    mkldnn::memory::dims mkl_padding_tl = {pt, pl};
    mkldnn::memory::dims mkl_padding_br = {pb, pr};

    mkldnn::memory mkl_src_memory = mkldnn::memory(
        {{mkl_src_shape}, dt::f32, tag::nchw}, mkl_eng, input_data->data());

    mkldnn::memory mkl_dst_memory = mkldnn::memory(
        {{mkl_dst_shape}, dt::f32, tag::nchw}, mkl_eng, output_data->data());

    auto pool_avg_algo = count_include_pad
                             ? mkldnn::algorithm::pooling_avg_include_padding
                             : mkldnn::algorithm::pooling_avg_exclude_padding;

    // pool desc
    auto pool_desc = mkldnn::pooling_forward::desc(
        mkldnn::prop_kind::forward_inference,
        is_avg ? pool_avg_algo : mkldnn::algorithm::pooling_max,
        mkl_src_memory.get_desc(), mkl_dst_memory.get_desc(), mkl_strides,
        mkl_kernel, mkl_padding_tl, mkl_padding_br);

    auto prim_desc =
        mkldnn::pooling_forward::primitive_desc(pool_desc, mkl_eng);

    // do reorder if needed
    auto src_memory = mkl_src_memory;
    if (prim_desc.src_desc() != mkl_src_memory.get_desc()) {
      src_memory = mkldnn::memory(prim_desc.src_desc(), mkl_eng);
      mkl_net.push_back(mkldnn::reorder(mkl_src_memory, src_memory));
      mkl_net_args.push_back(
          {{MKLDNN_ARG_FROM, mkl_src_memory}, {MKLDNN_ARG_TO, src_memory}});
    }

    auto dst_memory = mkldnn::memory(prim_desc.dst_desc(), mkl_eng);

    mkl_net.push_back(mkldnn::pooling_forward(prim_desc));
    mkl_net_args.push_back(
        {{MKLDNN_ARG_SRC, src_memory}, {MKLDNN_ARG_DST, dst_memory}});

    // reorder or copy the output
    if (dst_memory != mkl_dst_memory) {
      mkl_net.push_back(mkldnn::reorder(dst_memory, mkl_dst_memory));
      mkl_net_args.push_back(
          {{MKLDNN_ARG_FROM, dst_memory}, {MKLDNN_ARG_TO, mkl_dst_memory}});
    }
  }
  assert(mkl_net.size() == mkl_net_args.size() && "something is missing");
}

void PoolingOpKernel::set_i8_avg_mkldnn() {

  mkldnn::memory::dims mkl_src_shape = {n, c, ih, iw};
  mkldnn::memory::dims mkl_filter_shape = mkldnn::memory::dims{c, 1, 1, kh, kw};
  mkldnn::memory::dims mkl_bias_shape = {c};
  mkldnn::memory::dims mkl_dst_shape = {n, c, oh, ow};
  mkldnn::memory::dims mkl_strides = {sh, sw};

  mkldnn::memory::dims mkl_padding_l = {pt, pl};
  mkldnn::memory::dims mkl_padding_r = {pb, pr};
  mkldnn::memory::dims mkl_dilation = {0, 0};

  using tag = mkldnn::memory::format_tag;
  using dt = mkldnn::memory::data_type;

  size_t filter_size = c * kh * kw;

  filter_data = std::make_shared<std::vector<float>>(filter_size, 1);
  // set mkldnn memory
  mkldnn::memory mkl_src_memory = mkldnn::memory(
      {{mkl_src_shape}, dt::f32, tag::nchw}, mkl_eng, input_data->data());

  mkldnn::memory mkl_filter_memory = mkldnn::memory(
      {{mkl_filter_shape}, dt::f32, tag::goihw}, mkl_eng, filter_data->data());

  zero_bias = std::make_shared<std::vector<float>>(c, 0.0f);

  mkldnn::memory mkl_bias_memory = mkldnn::memory(
      {{mkl_bias_shape}, dt::f32, tag::x}, mkl_eng, zero_bias->data());
  mkldnn::memory mkl_dst_memory = mkldnn::memory(
      {{mkl_dst_shape}, dt::f32, tag::nchw}, mkl_eng, output_data->data());

  mkldnn::memory::desc src_md =
      mkldnn::memory::desc({mkl_src_shape}, dt::f32, tag::any);

  mkldnn::memory::desc filter_md =
      mkldnn::memory::desc({mkl_filter_shape}, dt::f32, tag::any);
  mkldnn::memory::desc bias_md =
      mkldnn::memory::desc({mkl_bias_shape}, dt::f32, tag::any);
  mkldnn::memory::desc dst_md =
      mkldnn::memory::desc({mkl_dst_shape}, dt::f32, tag::any);

  mkldnn::convolution_forward::desc conv_desc =
      mkldnn::convolution_forward::desc(
          mkldnn::prop_kind::forward_inference,
          mkldnn::algorithm::convolution_direct, src_md, filter_md, bias_md,
          dst_md, mkl_strides, mkl_dilation, mkl_padding_l, mkl_padding_r);
  mkldnn::convolution_forward::primitive_desc conv_prim_desc =
      mkldnn::convolution_forward::primitive_desc(conv_desc, mkl_eng);

  // do reorder if needed
  auto prim_src_memory = mkl_src_memory;
  if (conv_prim_desc.src_desc() != mkl_src_memory.get_desc()) {
    prim_src_memory = mkldnn::memory(conv_prim_desc.src_desc(), mkl_eng);
    mkl_net.push_back(mkldnn::reorder(mkl_src_memory, prim_src_memory));
    mkl_net_args.push_back(
        {{MKLDNN_ARG_FROM, mkl_src_memory}, {MKLDNN_ARG_TO, prim_src_memory}});
  }
  auto prim_filter_memory = mkl_filter_memory;
  if (conv_prim_desc.weights_desc() != mkl_filter_memory.get_desc()) {
    prim_filter_memory = mkldnn::memory(conv_prim_desc.weights_desc(), mkl_eng);
    mkldnn::reorder(mkl_filter_memory, prim_filter_memory)
        .execute(mkl_stream, mkl_filter_memory, prim_filter_memory);
  }
  auto prim_dst_memory = mkldnn::memory(conv_prim_desc.dst_desc(), mkl_eng);

  mkl_net.push_back(mkldnn::convolution_forward(conv_prim_desc));
  mkl_net_args.push_back({{MKLDNN_ARG_SRC, prim_src_memory},
                          {MKLDNN_ARG_WEIGHTS, prim_filter_memory},
                          {MKLDNN_ARG_BIAS, mkl_bias_memory},
                          {MKLDNN_ARG_DST, prim_dst_memory}});

  // reorder or copy the output
  if (prim_dst_memory != mkl_dst_memory) {
    mkl_net.push_back(mkldnn::reorder(prim_dst_memory, mkl_dst_memory));
    mkl_net_args.push_back(
        {{MKLDNN_ARG_FROM, prim_dst_memory}, {MKLDNN_ARG_TO, mkl_dst_memory}});
  }
}

void PoolingOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " Pool op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

void PoolingOpKernel::fp32_invoke() {
  for (size_t i = 0; i < mkl_net.size(); ++i) {
    mkl_net.at(i).execute(mkl_stream, mkl_net_args.at(i));
  }
  mkl_stream.wait();
}

void PoolingOpKernel::i8_avg_invoke() {
  for (size_t i = 0; i < mkl_net.size(); ++i) {
    mkl_net.at(i).execute(mkl_stream, mkl_net_args.at(i));
  }
  mkl_stream.wait();
  size_t output_size = output_data->size();
#pragma omp parallel for schedule(static, omp_schedule(output_size))
  for (size_t i = 0; i < output_size; ++i) {
    output_data->at(i) = (float)applyMultiplierAndRShiftAndSaturateInt8(
        output_data->at(i), (uint32_t)rshift, (uint32_t)multiplier, false);
  }
}

void PoolingOpKernel::invoke() {

  int ih = this->input_shape.at(2);
  int iw = this->input_shape.at(3);
  int size = output_data->size();
  if (datatype == DataType::FP32) {
    fp32_invoke();
  } else if (datatype == DataType::INT8) {
    if (pool_method == POOL_METHOD::AVG) {
      int lmem_size = MInfo::lmem_per_lane;
      if ((ih * iw) > ((lmem_size - size) / 2) && kh == ih && kw == iw) {
        // In hardware limitation, we can not put avg pool with large kernel
        // if avg pool ih * iw > local memory, in our hardware
        // need to split it then sum
        std::vector<int> h_slices;
        int h_slice_size = (int)(((lmem_size - size) / iw) / 2);
        int total_h = ih;
        while (total_h > 0) {
          if (total_h > h_slice_size) {
            total_h -= h_slice_size;
            h_slices.push_back(h_slice_size);
          } else {
            h_slices.push_back(total_h);
            break;
          }
        }
        int offset = 0;
        std::vector<float> output_data_(size, 0);
        for (auto &s : h_slices) {
          int filter_shape = c * s * kw;
          int g = c;
          int oc = c;
          int dh = 1, dw = 1;
          int input_slice_size = n * c * s * kw;
          std::vector<float> conv_filter(filter_shape, 1);
          std::vector<float> input_slice(input_slice_size);
          std::vector<float> output_tmp_data(size);
          std::vector<int64_t> tmp_shape = {n, c, s, iw};
          slice(input_data->data(), input_slice.data(), 2, offset, input_shape,
                tmp_shape);

          once_mkldnn_conv(input_slice.data(), conv_filter.data(), NULL,
                           output_tmp_data.data(), n, c, s, iw, oc, 1, 1, s, kw,
                           sh, sw, dh, dw, pt, pb, pl, pr, g, 0);
          offset += s;
          for (int64_t i = 0; i < size; ++i) {
            float sum = output_tmp_data[i];
            output_tmp_data[i] = (float)applyMultiplierAndRShiftAndSaturateInt8(
                sum, (uint32_t)rshift, (uint32_t)multiplier, false);
            output_data_[i] += output_tmp_data[i];
          }
        }
        for (int64_t i = 0; i < size; ++i) {
          output_data->at(i) = output_data_[i];
        }
      } else {
        i8_avg_invoke();
      }
    } else {
      // int8 max pool same with fp32 max pool
      fp32_invoke();
    }
  } else {
    fp32_invoke();
    clean16bitmantissa(output_data->data(), output_data->data(),
                       output_data->size());
  }
}
std::vector<float> PoolingOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}
void PoolingOpKernel::dump() {
  std::string pm = pool_method == POOL_METHOD::AVG ? "Average" : "Max";

  OpKernel::dump();
  llvm::outs() << "\tMethod:" << pm << "\n";

  llvm::outs() << "\tStrides: " << sh << "*" << sw << "\n";
  llvm::outs() << "\tPadding: "
               << "top: " << pt << ", buttom: " << pb << ", left: " << pl
               << ", right: " << pr << "\n";
  if (this->datatype == DataType::INT8 && pool_method == POOL_METHOD::AVG) {
    llvm::outs() << "\tRSHIFT: " << rshift << "\n";
    llvm::outs() << "\tMULTIPLIER: " << multiplier << "\n";
  }
}
} // namespace mlir