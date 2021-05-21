#include "tpuc/Interpreter/cpu/fullyconnected.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Interpreter/cpu/activation.hpp"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {
FullyConnectedOpKernel::FullyConnectedOpKernel(Operation &op,
                                               value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  auto fcOp = cast<tpu::FullyConnectedOp>(op);
  parseFullyConnectedParam(fcOp.input(), fcOp.output(), fcOp.filter(), m, k, n);
  // get tensors
  input_data = this->opdTensors[0];
  filter_data = this->opdTensors[1];
  bias_data = this->opdTensors[2];
  output_data = this->resTensor;

  this->do_relu = fcOp.do_relu();

  // set mkldnn
  this->mkl_eng = mkldnn::engine(mkldnn::engine::kind::cpu, 0);
  this->mkl_stream = mkldnn::stream(mkl_eng);

  if (datatype == DataType::INT8) {
    auto quant_rshift = this->opdTensors[5];
    auto quant_multiplier = this->opdTensors[6];
    if (!quant_rshift) {
      llvm_unreachable("quant_rshift is null!");
    }
    if (!quant_multiplier) {
      llvm_unreachable("quant_multiplier is null!");
    }
    rshift = quant_rshift->at(0);
    multiplier = quant_multiplier->at(0);
  }

  using tag = mkldnn::memory::format_tag;
  using dt = mkldnn::memory::data_type;

  mkldnn::memory::dims mkl_src_shape = {m, k};
  mkldnn::memory::dims mkl_filter_shape = {n, k};
  mkldnn::memory::dims mkl_bias_shape = {n};
  mkldnn::memory::dims mkl_dst_shape = {m, n};
  // set mkldnn memory
  mkldnn::memory mkl_src_memory = mkldnn::memory(
      {{mkl_src_shape}, dt::f32, tag::nc}, mkl_eng, input_data->data());

  mkldnn::memory mkl_filter_memory = mkldnn::memory(
      {{mkl_filter_shape}, dt::f32, tag::oi}, mkl_eng, filter_data->data());
  if (!bias_data) {
    auto zero_bias = std::make_shared<std::vector<float>>(n, 0.0f);
    bias_data = zero_bias;
  }
  mkldnn::memory mkl_bias_memory = mkldnn::memory(
      {{mkl_bias_shape}, dt::f32, tag::x}, mkl_eng, bias_data->data());
  mkldnn::memory mkl_dst_memory = mkldnn::memory(
      {{mkl_dst_shape}, dt::f32, tag::nc}, mkl_eng, output_data->data());

  mkldnn::memory::desc src_md =
      mkldnn::memory::desc({mkl_src_shape}, dt::f32, tag::any);

  mkldnn::memory::desc filter_md =
      mkldnn::memory::desc({mkl_filter_shape}, dt::f32, tag::any);
  mkldnn::memory::desc bias_md =
      mkldnn::memory::desc({mkl_bias_shape}, dt::f32, tag::any);
  mkldnn::memory::desc dst_md =
      mkldnn::memory::desc({mkl_dst_shape}, dt::f32, tag::any);

  // fc desc
  auto fc_desc = mkldnn::inner_product_forward::desc(
      mkldnn::prop_kind::forward_inference, src_md, filter_md, bias_md, dst_md);

  auto fc_prim_desc =
      mkldnn::inner_product_forward::primitive_desc(fc_desc, mkl_eng);

  // do reorder if needed
  auto src_memory = mkl_src_memory;
  if (fc_prim_desc.src_desc() != mkl_src_memory.get_desc()) {
    src_memory = mkldnn::memory(fc_prim_desc.src_desc(), mkl_eng);
    mkl_net.push_back(mkldnn::reorder(mkl_src_memory, src_memory));
    mkl_net_args.push_back(
        {{MKLDNN_ARG_FROM, mkl_src_memory}, {MKLDNN_ARG_TO, src_memory}});
  }
  auto weights_memory = mkl_filter_memory;
  if (fc_prim_desc.weights_desc() != mkl_filter_memory.get_desc()) {
    weights_memory = mkldnn::memory(fc_prim_desc.weights_desc(), mkl_eng);
    mkldnn::reorder(mkl_filter_memory, weights_memory)
        .execute(mkl_stream, mkl_filter_memory, weights_memory);
  }
  auto bias_memory = mkl_bias_memory;

  auto dst_memory = mkldnn::memory(fc_prim_desc.dst_desc(), mkl_eng);

  mkl_net.push_back(mkldnn::inner_product_forward(fc_prim_desc));
  mkl_net_args.push_back({{MKLDNN_ARG_SRC, src_memory},
                          {MKLDNN_ARG_WEIGHTS, weights_memory},
                          {MKLDNN_ARG_BIAS, bias_memory},
                          {MKLDNN_ARG_DST, dst_memory}});

  // reorder or copy the output
  if (dst_memory != mkl_dst_memory) {
    mkl_net.push_back(mkldnn::reorder(dst_memory, mkl_dst_memory));
    mkl_net_args.push_back(
        {{MKLDNN_ARG_FROM, dst_memory}, {MKLDNN_ARG_TO, mkl_dst_memory}});
  }
  assert(mkl_net.size() == mkl_net_args.size() && "something is missing");
} // namespace mlir

void FullyConnectedOpKernel::invoke() {
  for (size_t i = 0; i < mkl_net.size(); ++i) {
    mkl_net.at(i).execute(mkl_stream, mkl_net_args.at(i));
  }
  mkl_stream.wait();

  if (do_relu) {
    relu(output_data->data(), output_data->data(), output_data->size());
  }

  int output_size = output_data->size();
  if (datatype == DataType::INT8) {
#pragma omp parallel for schedule(static, omp_schedule(output_size))
    for (size_t i = 0; i < output_data->size(); ++i) {
      output_data->at(i) = (float)applyMultiplierAndRShiftAndSaturateInt8(
          output_data->at(i), (uint32_t)rshift, (uint32_t)multiplier, true);
    }
  }
  if (datatype == DataType::BF16) {
    clean16bitmantissa(output_data->data(), output_data->data(),
                       output_data->size());
  }
}

} // namespace mlir