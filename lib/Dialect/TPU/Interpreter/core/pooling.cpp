#include "tpuc/Interpreter/cpu/pooling.hpp"
#include "mkldnn.h"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"
namespace mlir {
PoolingOpKernel::PoolingOpKernel(Operation &op, value_map_t &valueMapping) {
  tpu::PoolParam pool_param;
  Value mlir_input_value;
  Value result;
  std::string name;
  bool is_avg;
  if (isa<tpu::PoolAvg2DOp>(op)) {
    auto poolavgOp = cast<tpu::PoolAvg2DOp>(op);
    pool_method = POOL_METHOD::AVG;
    mlir_input_value = poolavgOp.input();
    result = poolavgOp.getResult();
    pool_param = poolavgOp.param();
    name = poolavgOp.name().str();
    is_avg = true;
  } else if (isa<tpu::PoolMax2DOp>(op)) {
    auto poolmaxOp = cast<tpu::PoolMax2DOp>(op);
    pool_method = POOL_METHOD::MAX;
    mlir_input_value = poolmaxOp.input();
    result = poolmaxOp.getResult();
    pool_param = poolmaxOp.param();
    name = poolmaxOp.name().str();
    is_avg = false;
  }
  llvm::outs() << " Pool op: [" << name << "]\n";
  this->name = name;
  this->op_type = op.getName().getStringRef().str();
  set_datatype(getOpQuant(&op).str());

  auto opTensors = getOperandTensors(&op, valueMapping);

  auto size = getTensorSize(result);
  llvm::outs() << "    =>required memory size: [" << size << "]\n";
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  parsePoolParam(pool_param, mlir_input_value, result, n, c, ih, iw, oh, ow, kh,
                 kw, sh, sw, pt, pb, pl, pr, pad_value, is_global, do_relu,
                 count_include_pad);
  is_asymmetric = isOpQuantAsymmetric(&op);

  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

  // get tensors
  input_data = opTensors[0];
  output_data = resultTensor;
  using tag = mkldnn::memory::format_tag;
  using dt = mkldnn::memory::data_type;

  this->mkl_eng = mkldnn::engine(mkldnn::engine::kind::cpu, 0);
  this->mkl_stream = mkldnn::stream(mkl_eng);

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

  // auto src_md = mkldnn::memory::desc({mkl_src_shape}, dt::f32, tag::any);
  // auto dst_md = mkldnn::memory::desc({mkl_dst_shape}, dt::f32, tag::any);

  // pool desc
  auto pool_desc = mkldnn::pooling_forward::desc(
      mkldnn::prop_kind::forward_inference,
      is_avg ? pool_avg_algo : mkldnn::algorithm::pooling_max,
      mkl_src_memory.get_desc(), mkl_dst_memory.get_desc(), mkl_strides,
      mkl_kernel, mkl_padding_tl, mkl_padding_br);

  auto prim_desc = mkldnn::pooling_forward::primitive_desc(pool_desc, mkl_eng);

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
  assert(mkl_net.size() == mkl_net_args.size() && "something is missing");
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
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
void PoolingOpKernel::invoke() {
  for (size_t i = 0; i < mkl_net.size(); ++i) {
    mkl_net.at(i).execute(mkl_stream, mkl_net_args.at(i));
  }
  mkl_stream.wait();
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
}
} // namespace mlir