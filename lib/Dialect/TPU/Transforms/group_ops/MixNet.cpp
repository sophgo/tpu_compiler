/*
 * Copyright (C) Bitmain Technologies Inc.
 * All Rights Reserved.
 */
#include "MixNet.hpp"
#include "mlir/Analysis/Dominance.h"

#define DEBUG_TYPE "mixnet"

namespace mlir {
// set bottom
void MixOp::add_bottom_name(string bottom_name) {
  operands_.push_back(bottom_name);
}

// set top
void MixOp::add_top_name(string top_name) {
  results_.push_back(top_name);
}

MixNet::MixNet(NetGraph * net_graph, FuncOp * fn, MLIRContext * context) {
  net_graph_ = net_graph;
  fn_ = fn;
  context_ = context;
  fn->walk([&](Operation *op) {
    if (isa<tpu::WeightFileOp>(op)) {
      weightFileOp_ = op;
    }
  });
}

Value* MixNet::get_op_from_name(string name) {
  if (name_op_map_.find(name)!= name_op_map_.end()) {
    return (Value *)name_op_map_[name];
  } else {
    LLVM_DEBUG(llvm::errs() << "Cannot find op name " << name << " in MixNet.\n";);
    assert(0);
  }
}

void MixNet::add_opd_to_list(string op_name, Value * opd, bool b_generated) {
  pair <map<string, Value *>::iterator, bool> ptr;
  ptr = name_op_map_.insert(std::pair<string, Value*>(op_name, opd));
  if (b_generated)
    parallel_list_.push_back(opd->getDefiningOp());
  if (!ptr.second) {
    LLVM_DEBUG(llvm::errs() << "Value aleady inserted in op_name map, " << op_name << "\n";);
  }
  LLVM_DEBUG(llvm::errs() << "Add op_name " << op_name << "\n";);
}

void MixNet::parallel_start() {
  parallel_list_.clear();
}

void MixNet::parallel_end() {
  int op_size = parallel_list_.size();
  if (op_size < 2)
    return;
  for(int i = 0; i < op_size; i++) {
    Operation * cur_op = parallel_list_[i];
    if ( i == 0 ) {
      auto op = llvm::dyn_cast<tpu::TpuTLOpCodegenInterface>(cur_op);
      auto tmp = llvm::dyn_cast<tpu::TpuOpCommonInterface>(cur_op);
      LLVM_DEBUG(llvm::errs() << "add parallel enable for inst: " << getOpName(tmp) << "\n";);
      op.setEnableParallel(true);
    }

    if ( i == (op_size - 1)) {
      auto op = llvm::dyn_cast<tpu::TpuTLOpCodegenInterface>(cur_op);
      auto tmp = llvm::dyn_cast<tpu::TpuOpCommonInterface>(cur_op);
      LLVM_DEBUG(llvm::errs() << "add parallel disable for inst: " << getOpName(tmp) << "\n";);
      op.setDisableParallel(true);
    }
  }
}

void MixNet::set_net_in_tensor(int tensor_id) { this->net_in_tensors_.push_back(tensor_id); }

void MixNet::set_net_out_tensor(int tensor_id) { this->net_out_tensors_.push_back(tensor_id); }

static const std::string _get_postfix_name(int group_idx, int n_loop, int h_loop) {
  const std::string name = std::string("_") + std::to_string(group_idx) + "_" +
                           std::to_string(n_loop) + "_" + std::to_string(h_loop);
  return name;
}



// add group start layer
void MixNet::add_group_start_ops(int group_idx, Group* group, Operation *op,
                                 int n_secs, int h_secs) {
  Builder builder_(context_);
  set<int> in_neuron_tensors = group->get_group_in_neuron_tensors();

  for (auto tid : in_neuron_tensors) {
   int from_layer = net_graph_->get_tensor_from_layer(tid);
    const ImLayer * im_layer = net_graph_->get_layer_by_id(from_layer);
    assert(im_layer->op()->getNumResults() == 1);
    string name = top_name(im_layer->op(),0);
    Value * in_op = im_layer->op()->getResult(0);
    add_opd_to_list(name, in_op, false);
  }
}

void MixNet::add_group_end_ops(int group_idx, Group* group, int n_secs, int h_secs) {
  Builder builder_(context_);
  vector<int> out_neuron_tensors = group->get_group_out_tensors();

  for (auto tid : out_neuron_tensors) {
    int from_layer = net_graph_->get_tensor_from_layer(tid);
    const ImLayer * im_layer = net_graph_->get_layer_by_id(from_layer);
    Operation * old_op = im_layer->op();
    string old_name = top_name(old_op,0);
    Value * old_op_r = old_op->getResult(0);
    vector<Value *> operands;
    gaddr_t start_gaddr = 0xFFFFFFFF;
    for (int i = 0; i < n_secs; i++) {
      for (int j = 0; j < h_secs; j++) {
        std::string new_name = old_name + _get_postfix_name(group_idx, i, j);
        new_name += "_st";
        Value * new_opd = get_op_from_name(new_name);
        auto store_op = dyn_cast<tpu::TL_LG_StoreOp>(new_opd->getDefiningOp());
        gaddr_t op_gaddr = store_op.gaddr()->getLimitedValue();
        if (op_gaddr < start_gaddr)
          start_gaddr = op_gaddr;
        operands.push_back(new_opd);
      }
    }
    std::vector<NamedAttribute> attrs;
    attrs.push_back(builder_.getNamedAttr("name", builder_.getStringAttr(old_name)));
    attrs.push_back(builder_.getNamedAttr("gaddr", builder_.getI64IntegerAttr(start_gaddr)));
    auto join_op = OpBuilder(get_start_op()).create<tpu::TL_LG_JoinOp>(get_start_op()->getLoc(), old_op_r->getType(),
        ArrayRef<Value *>{operands}, ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(old_name, join_op.getResult(), false);

    // replace the next group's usage to this new generated op
    old_op->replaceAllUsesWith(join_op);

  }
}

void MixNet::add_tl_layer(int group_idx, int layer_id, net_timestep* time_step, int timestep_idx,
                          bool is_h_split, int n_loop, int h_loop) {
  const ImLayer* im_layer = net_graph_->get_layer_by_id(layer_id);
  const vector<int>& in_tensors = net_graph_->get_in_tensors_of_layer(layer_id);
  const vector<int>& out_tensors = net_graph_->get_out_tensors_of_layer(layer_id);
  std::string postfix = _get_postfix_name(group_idx, n_loop, h_loop);

  MixOp * mix_op = new MixOp(this, layer_id);
  mix_op->set_name(im_layer->name() + postfix);

  for (u32 i = 0; i < in_tensors.size(); i++) {
    Tensor* in_tensor = net_graph_->get_tensor_by_id(in_tensors[i]);
    const string& name = in_tensor->name();
    if (in_tensor->type() != TENSOR_NEURON && in_tensor->type() != TENSOR_NEURON_WINOGRAD &&
        in_tensor->type() != TENSOR_MATRIX) {
      // coeff not do slice, no need postfix
      mix_op->add_bottom_name(name);
    } else {
      mix_op->add_bottom_name(name + postfix);
    }
  }

  for (u32 i = 0; i < out_tensors.size(); i++) {
    Tensor* out_tensor = net_graph_->get_tensor_by_id(out_tensors[i]);
    const string& name = out_tensor->name();
    mix_op->add_top_name(name + postfix);
  }

  for (u32 i = 0; i < out_tensors.size(); ++i) {
    mem_buffer_key_t key = {timestep_idx, out_tensors[i], false};
    const mem_buffer_value_t* value = time_step->get_mem_buffer_value(&key);
    net_graph_->set_tensor_local_offest(out_tensors[i], value->local_mem_offset);
  }

  IR_TYPE layer_type = im_layer->type();

  switch (layer_type) {
    case IR_CONVOLUTION:
      mix_op->set_type("tl_convolution");
      _add_tl_convolution_param(mix_op, in_tensors, out_tensors, time_step,
                                timestep_idx, is_h_split);
      break;
    case IR_ELTWISE:
      mix_op->set_type("tl_eltwise");
      _add_tl_eltwise_param(mix_op, in_tensors, out_tensors, time_step,
                            timestep_idx, is_h_split);
      break;
    case IR_POOLING:
      mix_op->set_type("tl_pooling");
      _add_tl_pooling_param(mix_op, in_tensors, out_tensors, time_step, timestep_idx,
                            is_h_split);
      break;
  //   case IR_DECONVOLUTION:
  //     layer->set_type("tl_deconvolution");
  //     _add_tl_deconvolution_param(layer_id, layer, im_layer, in_tensors, out_tensors, time_step,
  //                                 timestep_idx, is_h_split);
  //     break;


  //   case IR_LRN:
  //     layer->set_type("tl_lrn");
  //     _add_tl_lrn_param(layer_id, layer, im_layer, in_tensors, out_tensors, time_step, timestep_idx,
  //                       is_h_split);
  //     break;
  //   case IR_BATCHNORM:
  //     layer->set_type("tl_batchnorm");
  //     _add_tl_batchnorm_param(layer, im_layer, in_tensors, out_tensors, time_step, timestep_idx,
  //                             is_h_split);
  //     break;
  //   case IR_SCALE:
  //     layer->set_type("tl_scale");
  //     _add_tl_scale_param(layer, im_layer, in_tensors, out_tensors, time_step, timestep_idx,
  //                         is_h_split);
  //     break;
  //   case IR_MAC:
  //     layer->set_type("tl_mac");
  //     _add_tl_mac_param(layer, im_layer, in_tensors, out_tensors, time_step, timestep_idx,
  //                       is_h_split);
  //     break;

  //   case IR_UPSAMPLE:
  //     layer->set_type("tl_upsample");
  //     _add_tl_upsample_param(layer, im_layer, in_tensors, out_tensors, time_step, timestep_idx,
  //                            is_h_split);
  //     break;
  //   case IR_PRELU:
  //   case IR_RELU:
  //   case IR_ACTIVATION:
  //     layer->set_type("tl_activation");
  //     _add_tl_activation_param(layer, im_layer, in_tensors, out_tensors, time_step, timestep_idx,
  //                              is_h_split);
  //     break;
  //   case IR_SHUFFLECHANNEL:
  //     layer->set_type("tl_shufflechannel");
  //     _add_tl_shuffle_channel_param(layer, im_layer, in_tensors, out_tensors, time_step,
  //                                   timestep_idx, is_h_split);
  //     break;
  //   case IR_ARITHMETIC:
  //     layer->set_type("tl_arithmetic");
  //     this->_add_tl_arithmetic_param(layer, im_layer, in_tensors, out_tensors, time_step,
  //                                    timestep_idx, is_h_split);
  //     break;
  //   case IR_INNERPRODUCT:
  //     layer->set_type("tl_innerproduct");
  //     _add_tl_innerproduct_param(layer, im_layer, in_tensors, out_tensors, time_step, timestep_idx,
  //                                is_h_split);
  //     break;
  //   case IR_QUANTIZATION:
  //     layer->set_type("tl_quantization");
  //     _add_tl_quantization_param(layer, im_layer, in_tensors, out_tensors, time_step, timestep_idx,
  //                                is_h_split);
  //     break;
    default:
      cout << "unknown layer type:" << layer_type << endl;
      //exit(1);
  }
}

void MixNet::_add_tl_convolution_param(MixOp* mix_op,
                                       const vector<int>& in_tensors,
                                       const vector<int>& out_tensors, net_timestep* time_step,
                                       int timestep_idx, bool is_h_split) {
  const ImLayer* im_layer = net_graph_->get_layer_by_id(mix_op->get_layer_id());
  auto old_op = dyn_cast<tpu::Conv2DOp>(im_layer->op());
  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, ph, pw, dh, dw;
  bool is_deconv = isa<tpu::DeConv2DOp>(old_op.getOperation());

  parseConvParam(old_op.param(), is_deconv, old_op.input(), old_op.output(), old_op.filter(),
                  n, ic, ih, iw, oc, oh, ow, g,
                  kh, kw, sh, sw, ph, pw, dh, dw, is_dw, with_bias, do_relu);

  auto old_input_type = old_op.input()->getType().cast<RankedTensorType>();
  Tensor* in_tensor = im_layer->in_tensors[0].get();

  int real_h_idx, real_h_slice;
  int top_pad_h, bottom_pad_h;
  int left_pad_w, right_pad_w;
  int h_end;
  int bottom_dim[4];
  int top_dim[4];

  net_graph_->get_tensor_dim(in_tensors[0], bottom_dim);
  net_graph_->get_tensor_dim(out_tensors[0], top_dim);

  bottom_dim[0] = in_tensor->n_slice;
  bottom_dim[2] = in_tensor->h_slice;
  top_pad_h = ph;
  bottom_pad_h = ph;
  left_pad_w = pw;
  right_pad_w = pw;

  if (is_h_split) {
    if (in_tensor->h_idx > 0) {
      real_h_idx = in_tensor->h_idx;
      top_pad_h = 0;
    } else {
      real_h_idx = 0;
      top_pad_h = 0 - in_tensor->h_idx;
    }
    h_end = in_tensor->h_idx + in_tensor->h_slice;
    if (h_end > in_tensor->h()) {
      real_h_slice = in_tensor->h() - real_h_idx;
      bottom_pad_h = h_end - in_tensor->h();
    } else {
      real_h_slice = h_end - real_h_idx;
      bottom_pad_h = 0;
    }
    bottom_dim[2] = real_h_slice;
  }

  const Tensor* out_tensor = net_graph_->get_tensor_by_id(out_tensors[0]);
  top_dim[0] = out_tensor->n_slice;
  top_dim[2] = out_tensor->h_slice;

  if (is_h_split) {
    if (out_tensor->h_idx > 0) {
      real_h_idx = out_tensor->h_idx;
    } else {
      real_h_idx = 0;
    }
    h_end = out_tensor->h_idx + out_tensor->h_slice;
    if (h_end > out_tensor->h()) {
      real_h_slice = out_tensor->h() - real_h_idx;
    } else {
      real_h_slice = h_end - real_h_idx;
    }
    top_dim[2] = real_h_slice;
  }

  u32 input_laddr = (net_graph_->get_tensor_local_offset(in_tensors[0]));
  u32 weight_laddr = (net_graph_->get_tensor_local_offset(in_tensors[1]));
  u32 output_laddr = (net_graph_->get_tensor_local_offset(out_tensors[0]));
  int bias_laddr = net_graph_->get_tensor_local_offset(in_tensors[2]);


  RankedTensorType input_type = RankedTensorType::get(
                          {bottom_dim[0], bottom_dim[1],
                           bottom_dim[2], bottom_dim[3]},
                           old_input_type.getElementType());

  RankedTensorType output_type = RankedTensorType::get(
                          {top_dim[0], top_dim[1],
                           top_dim[2], top_dim[3]},
                           old_input_type.getElementType());

  // setup parameter
  vector<NamedAttribute> attrs;
  Builder builder_(context_);
  attrs.push_back(builder_.getNamedAttr("name", builder_.getStringAttr(mix_op->name())));
  attrs.push_back(builder_.getNamedAttr("param",
    tpu::ConvParam::get(
      builder_.getI32IntegerAttr(sh),
      builder_.getI32IntegerAttr(sw),
      builder_.getStringAttr("VALID"),
      builder_.getI32IntegerAttr(dh),
      builder_.getI32IntegerAttr(dw),
      builder_.getI32IntegerAttr(g),
      builder_.getBoolAttr(is_dw),
      builder_.getBoolAttr(with_bias),
      builder_.getBoolAttr(do_relu),
      builder_.getContext())));
  attrs.push_back(builder_.getNamedAttr("la_input", builder_.getI32IntegerAttr(input_laddr)));
  attrs.push_back(builder_.getNamedAttr("la_output", builder_.getI32IntegerAttr(output_laddr)));
  attrs.push_back(builder_.getNamedAttr("la_filter", builder_.getI32IntegerAttr(weight_laddr)));
  attrs.push_back(builder_.getNamedAttr("la_bias", builder_.getI32IntegerAttr(bias_laddr)));
  attrs.push_back(builder_.getNamedAttr("la_working", builder_.getI32IntegerAttr(0)));
  attrs.push_back(builder_.getNamedAttr("layer_id", old_op.layer_idAttr()));
  attrs.push_back(builder_.getNamedAttr("pad_top_h", builder_.getI32IntegerAttr(top_pad_h)));
  attrs.push_back(builder_.getNamedAttr("pad_bottom_h", builder_.getI32IntegerAttr(bottom_pad_h)));
  attrs.push_back(builder_.getNamedAttr("pad_left_w", builder_.getI32IntegerAttr(left_pad_w)));
  attrs.push_back(builder_.getNamedAttr("pad_right_w", builder_.getI32IntegerAttr(right_pad_w)));

  // setup input operation
  vector<Value *> operands;
  Operation * input_op = get_op_from_name(mix_op->bottom_name(0))->getDefiningOp();
  input_op->getResult(0)->setType(input_type);
  operands.push_back(input_op->getResult(0));

  // setup filter operation
  Operation * filter_op = get_op_from_name(mix_op->bottom_name(1))->getDefiningOp();
  operands.push_back(filter_op->getResult(0));
  // setup bias operation
  Operation * bias_op = get_op_from_name(mix_op->bottom_name(2))->getDefiningOp();
  operands.push_back(bias_op->getResult(0));

  // build tl_conv operation
  auto op = OpBuilder(get_start_op()).create<tpu::TL_LG_Conv2DOp>(get_start_op()->getLoc(),
            output_type, ArrayRef<Value *>{operands}, ArrayRef<NamedAttribute>{attrs});
  add_opd_to_list(mix_op->name(), op.getResult(), true);
}


void MixNet::_add_tl_eltwise_param(MixOp* mix_op,
                                   const vector<int>& in_tensors, const vector<int>& out_tensors,
                                   net_timestep* time_step, int timestep_idx, bool is_h_split) {
  const ImLayer* im_layer = net_graph_->get_layer_by_id(mix_op->get_layer_id());
  const Tensor* in_tensor = im_layer->in_tensors[0].get();
  auto old_op = dyn_cast<tpu::EltwiseAddOp>(im_layer->op());
  auto old_input_type = old_op.getOperand(0)->getType().cast<RankedTensorType>();
  bool do_relu = old_op.do_relu();
  int nInputs = old_op.getNumOperands();
  int newNInputs = nInputs - 4;
  int bottom_dim[4];

  u64 working_laddr = 0;
  u64 output_laddr = 0;
  vector<int32_t> input_laddr;
  net_graph_->get_tensor_dim(in_tensors[0], bottom_dim);

  bottom_dim[0] = in_tensor->n_slice;
  bottom_dim[2] = in_tensor->h_slice;

  mem_buffer_key_t key = {timestep_idx, im_layer->imm_tensors[0].get()->id(), true};
  const mem_buffer_value_t* imm = time_step->get_mem_buffer_value(&key);
  working_laddr = (imm->local_mem_offset);

  assert(nInputs == 6);
  // input0, input2, rshift, multiplier, only support two inputs
  for (int i = 0; i < newNInputs; i++) {
    input_laddr.push_back(net_graph_->get_tensor_local_offset(in_tensors[i]));
  }

  output_laddr = (net_graph_->get_tensor_local_offset(out_tensors[0]));

  // build eltwise op
  vector<NamedAttribute> attrs;
  Builder builder_(context_);
  attrs.push_back(builder_.getNamedAttr("name", builder_.getStringAttr(mix_op->name())));
  attrs.push_back(builder_.getNamedAttr("la_input", builder_.getI32ArrayAttr(ArrayRef<int32_t>({input_laddr}))));
  attrs.push_back(builder_.getNamedAttr("la_output", builder_.getI32IntegerAttr(output_laddr)));
  attrs.push_back(builder_.getNamedAttr("la_working", builder_.getI32IntegerAttr(working_laddr)));
  attrs.push_back(builder_.getNamedAttr("do_relu", builder_.getBoolAttr(do_relu)));
  attrs.push_back(builder_.getNamedAttr("layer_id", old_op.layer_idAttr()));

  // setup input/output type
  RankedTensorType input_type = RankedTensorType::get(
                          {bottom_dim[0], bottom_dim[1],
                           bottom_dim[2], bottom_dim[3]},
                           old_input_type.getElementType());

  RankedTensorType output_type = RankedTensorType::get(
                          {bottom_dim[0], bottom_dim[1],
                           bottom_dim[2], bottom_dim[3]},
                           old_input_type.getElementType());


  TensorFile *wTF = getWeightTensorFile(im_layer->op());
  Value * quant_rshift = old_op.getOperand(nInputs - 2);
  Value * quant_multiplier = old_op.getOperand(nInputs - 1);
  auto rshift = readWeightTensor<float>(quant_rshift, wTF);
  assert(rshift->size() == 1);
  attrs.push_back(builder_.getNamedAttr("rshift",
      builder_.getI8IntegerAttr(static_cast<int8_t>(rshift->at(0)))));

  // m_i8_inputs
  auto multiplier = readWeightTensor<float>(quant_multiplier, wTF);
  std::vector<int32_t> m_i8_inputs_array(newNInputs);
  for (int i = 0; i < newNInputs; ++i) {
    m_i8_inputs_array[i] = static_cast<int32_t>(multiplier->at(i));
  }
  attrs.push_back(builder_.getNamedAttr("m_i8_inputs",
      builder_.getI32ArrayAttr(ArrayRef<int32_t>({m_i8_inputs_array}))));

  // setup input operation
  vector<Value *> operands;
  for( u32 i = 0; i < in_tensors.size(); i++) {
    Operation * input_op = get_op_from_name(mix_op->bottom_name(i))->getDefiningOp();
    input_op->getResult(0)->setType(input_type);
    operands.push_back(input_op->getResult(0));
  }

  // build eltwiseadd operation
  auto op = OpBuilder(get_start_op()).create<tpu::TL_LG_EltwiseAddOp>(get_start_op()->getLoc(),
            output_type, ArrayRef<Value *>{operands}, ArrayRef<NamedAttribute>{attrs});

  add_opd_to_list(mix_op->name(), op.getResult(), true);
}

void MixNet::_add_tl_pooling_param(MixOp * mix_op,
                                   const vector<int>& in_tensors, const vector<int>& out_tensors,
                                   net_timestep* time_step, int timestep_idx, bool is_h_split) {
  const ImLayer* im_layer = net_graph_->get_layer_by_id(mix_op->get_layer_id());
  const Tensor* in_tensor = net_graph_->get_tensor_by_id(in_tensors[0]);
  const Tensor* out_tensor = net_graph_->get_tensor_by_id(out_tensors[0]);
  bool is_avg = isa<tpu::PoolAvg2DOp>(im_layer->op());
  auto old_op = (im_layer->op());
  auto old_input_type = old_op->getOperand(0)->getType().cast<RankedTensorType>();
  int nInputs = old_op->getNumOperands();
  vector<NamedAttribute> attrs;
  // parse param
  bool is_global, do_relu;
  int n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr;
  if (is_avg) {
    auto tmp_op = dyn_cast<tpu::PoolAvg2DOp>(im_layer->op());
    parsePoolParam(tmp_op.param(), tmp_op.input(), tmp_op.output(),
                  n, c, ih, iw, oh, ow,
                  kh, kw, sh, sw, pt, pb, pl, pr,
                  is_global, do_relu);

  } else {
    auto tmp_op = dyn_cast<tpu::PoolMax2DOp>(im_layer->op());
    parsePoolParam(tmp_op.param(), tmp_op.input(), tmp_op.output(),
                  n, c, ih, iw, oh, ow,
                  kh, kw, sh, sw, pt, pb, pl, pr,
                  is_global, do_relu);
  }


  int bottom_dim[4];
  int top_dim[4];
  net_graph_->get_tensor_dim(in_tensors[0], bottom_dim);
  net_graph_->get_tensor_dim(out_tensors[0], top_dim);

  bottom_dim[0] = in_tensor->n_slice;
  bottom_dim[2] = in_tensor->h_slice;

  top_dim[0] = out_tensor->n_slice;
  top_dim[2] = out_tensor->h_slice;

  int top_pad_h = pt;
  int bottom_pad_h = pb;
  int left_pad_w = pl;
  int right_pad_w = pr;

  if (sh * (top_dim[2] - 1) + kh > bottom_dim[2] + top_pad_h + bottom_pad_h) {
    bottom_pad_h = sh * (top_dim[2] - 1) + kh - bottom_dim[2] - pt;
  }

  if (sw * (top_dim[3] - 1) + kw > bottom_dim[3] + left_pad_w + right_pad_w) {
    right_pad_w = sw * (top_dim[3] - 1) + kw - bottom_dim[3] - pl;
  }

  if (is_h_split) {
    int real_h_idx, real_h_slice;

    // bottom
    if (in_tensor->h_idx > 0) {
      real_h_idx = in_tensor->h_idx;
      top_pad_h = 0;
    } else {
      real_h_idx = 0;
      top_pad_h = 0 - in_tensor->h_idx;
    }
    int h_end = in_tensor->h_idx + in_tensor->h_slice;
    if (h_end > in_tensor->h()) {
      real_h_slice = in_tensor->h() - real_h_idx;
      bottom_pad_h = h_end - in_tensor->h();
    } else {
      real_h_slice = h_end - real_h_idx;
      bottom_pad_h = 0;
    }
    bottom_dim[2] = real_h_slice;

    // top
    if (out_tensor->h_idx > 0) {
      real_h_idx = out_tensor->h_idx;
    } else {
      real_h_idx = 0;
    }
    h_end = out_tensor->h_idx + out_tensor->h_slice;
    if (h_end > out_tensor->h()) {
      real_h_slice = out_tensor->h() - real_h_idx;
    } else {
      real_h_slice = h_end - real_h_idx;
    }
    top_dim[2] = real_h_slice;
  }

  u64 input_laddr = (net_graph_->get_tensor_local_offset(in_tensors[0]));
  u64 output_laddr = (net_graph_->get_tensor_local_offset(out_tensors[0]));

   // build pooling op
  Builder builder_(context_);
  attrs.push_back(builder_.getNamedAttr("name", builder_.getStringAttr(mix_op->name())));
  attrs.push_back(builder_.getNamedAttr("param",
    tpu::PoolParam::get(
      builder_.getI32IntegerAttr(kh),
      builder_.getI32IntegerAttr(kw),
      builder_.getI32IntegerAttr(pt),
      builder_.getI32IntegerAttr(pb),
      builder_.getI32IntegerAttr(pl),
      builder_.getI32IntegerAttr(pr),
      builder_.getI32IntegerAttr(sh),
      builder_.getI32IntegerAttr(sw),
      builder_.getBoolAttr(do_relu),
      builder_.getContext())));
  attrs.push_back(builder_.getNamedAttr("la_input", builder_.getI32IntegerAttr(input_laddr)));
  attrs.push_back(builder_.getNamedAttr("la_output", builder_.getI32IntegerAttr(output_laddr)));
  attrs.push_back(builder_.getNamedAttr("layer_id", builder_.getI32IntegerAttr(getOpLayerId(old_op))));

  if (is_avg) {
    // add rshift and multiplier
    TensorFile *wTF = getWeightTensorFile(im_layer->op());
    Value * quant_rshift = old_op->getOperand(nInputs - 2);
    Value * quant_multiplier = old_op->getOperand(nInputs - 1);
    auto rshift = readWeightTensor<float>(quant_rshift, wTF);
    assert(rshift->size() == 1);
    attrs.push_back(builder_.getNamedAttr("rshift",
        builder_.getI8IntegerAttr(static_cast<int8_t>(rshift->at(0)))));

    // m_i8_inputs
    auto multiplier = readWeightTensor<float>(quant_multiplier, wTF);
    int32_t m_i8;
    // only 2 input
    m_i8 = static_cast<int32_t>(multiplier->at(0));
    attrs.push_back(builder_.getNamedAttr("m_i8",
        builder_.getI32IntegerAttr((m_i8))));
  }

  // setup input/output type
  RankedTensorType input_type = RankedTensorType::get(
                          {bottom_dim[0], bottom_dim[1],
                           bottom_dim[2], bottom_dim[3]},
                           old_input_type.getElementType());

  RankedTensorType output_type = RankedTensorType::get(
                          {top_dim[0], top_dim[1],
                           top_dim[2], top_dim[3]},
                           old_input_type.getElementType());

  // setup input operation
  vector<Value *> operands;
  for( u32 i = 0; i < in_tensors.size(); i++) {
    Operation * input_op = get_op_from_name(mix_op->bottom_name(i))->getDefiningOp();
    input_op->getResult(0)->setType(input_type);
    operands.push_back(input_op->getResult(0));
  }

  // build pooling operation
  if (is_avg) {
    auto op = OpBuilder(get_start_op()).create<tpu::TL_LG_INT8_PoolAvg2DOp>(get_start_op()->getLoc(),
              output_type, ArrayRef<Value *>{operands}, ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(mix_op->name(), op.getResult(), true);
   } else {
    auto op = OpBuilder(get_start_op()).create<tpu::TL_LG_INT8_PoolMax2DOp>(get_start_op()->getLoc(),
              output_type, ArrayRef<Value *>{operands}, ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(mix_op->name(), op.getResult(), true);
   }
}

void MixNet::add_transport_param(const TENSOR_STEP& tensor,
                                                           net_timestep* time_step,
                                                           int timestep_idx) {
  int tensor_id = tensor.first;
  if (tensor.second == TIMESTEP_LOAD) {
    _add_load_param(tensor_id, time_step, timestep_idx);
  }
  else if (tensor.second == TIMESTEP_DDR_TO_TSM) {
    assert(0);
  } else if (tensor.second == TIMESTEP_TSM_TO_LMEM) {
    assert(0);
  } else if (tensor.second == TIMESTEP_LMEM_TO_TSM) {
    assert(false && "not support now");
    exit(-1);
  } else if (tensor.second == TIMESTEP_TSM_TO_DDR) {
    assert(false && "not support now");
    exit(-1);
  } else if (tensor.second == TIMESTEP_STORE) {
    _add_store_param(tensor_id, time_step, timestep_idx);
  }
}

void MixNet::_add_load_param(int tensor_id,
                             net_timestep* time_step, int timestep_idx) {
  return _add_load_param_bm1880v2(tensor_id, time_step, timestep_idx);
}

void MixNet::_add_load_param_bm1880v2(int tensor_id,
                                    net_timestep* time_step, int timestep_idx) {
  int tensor_dim[4];
  int local_shape[4], global_shape[4];
  u64 laddr = 0, gaddr = 0;
  bool aligned = false;
  bool transpose = false;
  string direction = "S2L";
  string tensor_type_str = "CONV_COEFF";
  int dtype = NEURON;
  string name;
  vector<NamedAttribute> attrs;
  Builder builder_(context_);

  const tensor_type_t tensor_type = net_graph_->get_tensor_type(tensor_id);
  Tensor* tensor = net_graph_->get_tensor_by_id(tensor_id);
  string storage = tensor->storage();
  net_graph_->get_tensor_dim(tensor_id, tensor_dim);

  name = tensor->name();
  Value * src_opd = weightFileOp_->getResult(0);
  gaddr = net_graph_->get_tensor_global_mem(tensor_id);

  if (tensor_type == TENSOR_COEFF) {
    laddr = net_graph_->get_tensor_local_offset(tensor_id);

    // to match mlir requirement for conv weight, shape is
    // (oc, ic, kh, kw)
    local_shape[0] = tensor_dim[1];
    local_shape[1] = tensor_dim[0];
    local_shape[2] = tensor_dim[2];
    local_shape[3] = tensor_dim[3];

    global_shape[0] = tensor_dim[1];
    global_shape[1] = tensor_dim[0];
    global_shape[2] = tensor_dim[2];
    global_shape[3] = tensor_dim[3];

    aligned = (false);
    transpose = (false);
    tensor_type_str = "CONV_COEFF";
    dtype = COEFF;
  } else if (tensor_type == TENSOR_BIAS) {
    laddr = net_graph_->get_tensor_local_offset(tensor_id);

    local_shape[0] = (tensor_dim[0]);
    local_shape[1] = (tensor_dim[1]);
    local_shape[2] = (tensor_dim[2]);
    local_shape[3] = (tensor_dim[3]);

    global_shape[0] = (tensor_dim[0]);
    global_shape[1] = (tensor_dim[1]);
    global_shape[2] = (tensor_dim[2]);
    global_shape[3] = (tensor_dim[3]);

    aligned = (false);
    transpose = (false);
    tensor_type_str = "BIAS";
    dtype = COEFF;
  } else if (tensor_type == TENSOR_DEPTHCONV_OPD1) {
    laddr = net_graph_->get_tensor_local_offset(tensor_id);

    local_shape[0] = (tensor_dim[0]);
    local_shape[1] = (tensor_dim[1]);
    local_shape[2] = (tensor_dim[2]);
    local_shape[3] = (tensor_dim[3]);

    global_shape[0] = (tensor_dim[0]);
    global_shape[1] = (tensor_dim[1]);
    global_shape[2] = (tensor_dim[2]);
    global_shape[3] = (tensor_dim[3]);

    aligned = (true);
    transpose = (false);
    tensor_type_str = "CONV_DEPTH_OPD1";
    dtype = COEFF;
  } else {
    int n_idx = tensor->n_idx;
    int n_slice = tensor->n_slice;
    int h_idx = tensor->h_idx;
    int h_slice = tensor->h_slice;
    int h_end = h_idx + h_slice;
    h_idx = h_idx > 0 ? h_idx : 0;
    h_slice = h_end > tensor_dim[2] ? (tensor_dim[2] - h_idx) : (h_end - h_idx);
    src_opd = get_op_from_name(name);
    name = name + _get_postfix_name(tensor->get_group_id(), tensor->get_n_loop(), tensor->get_h_loop());

    mem_buffer_key_t key = {timestep_idx, tensor_id, false};
    const mem_buffer_value_t* value = time_step->get_mem_buffer_value(&key);
    laddr = value->local_mem_offset;

    gaddr += (n_idx * tensor_dim[1] * tensor_dim[2] * tensor_dim[3] + h_idx * tensor_dim[3]) *
              tensor->unit_size();
    llvm::errs() << name << ":         n_idx/h_idx = " << n_idx << "/" << h_idx << " tensor_dim: ( "
                 << tensor_dim[0] << ", " << tensor_dim[1] << "," << tensor_dim[2]
                 << ", " << tensor_dim[3] << ") << gaddr:" << gaddr << "\n";

    local_shape[0] = (n_slice);
    local_shape[1] = (tensor_dim[1]);
    local_shape[2] = (h_slice);
    local_shape[3] = (tensor_dim[3]);

    global_shape[0] = (tensor_dim[0]);
    global_shape[1] = (tensor_dim[1]);
    global_shape[2] = (tensor_dim[2]);
    global_shape[3] = (tensor_dim[3]);


    if (tensor_type == TENSOR_NEURON || tensor_type == TENSOR_NEURON_WINOGRAD) {
      aligned = (true);
    } else {  // TENSOR_COEFF_NEURON
      if (tensor_type != TENSOR_NEURON_AS_COEFF) {
        dtype = COEFF;
      }
      aligned = (false);
    }
    transpose = (false);
    net_graph_->set_tensor_local_offest(tensor_id, laddr);
    tensor_type_str = "NEURON";
    attrs.push_back(builder_.getNamedAttr("gaddr", builder_.getI64IntegerAttr(gaddr)));
  }

  // build tl_load instruction
  attrs.push_back(builder_.getNamedAttr("name", builder_.getStringAttr(name)));
  attrs.push_back(builder_.getNamedAttr("ls_direction", builder_.getStringAttr(direction)));
  attrs.push_back(builder_.getNamedAttr("laddr", builder_.getI64IntegerAttr(laddr)));
  attrs.push_back(builder_.getNamedAttr("align", builder_.getBoolAttr(aligned)));
  attrs.push_back(builder_.getNamedAttr("transpose", builder_.getBoolAttr(transpose)));
  attrs.push_back(builder_.getNamedAttr("tensor_type", builder_.getStringAttr(tensor_type_str)));
  attrs.push_back(builder_.getNamedAttr("storage", builder_.getStringAttr(storage)));

  // setup input operation
  vector<Value *> operands;
  operands.push_back(src_opd);
  RankedTensorType output_type = RankedTensorType::get(
                          {local_shape[0], local_shape[1],
                           local_shape[2], local_shape[3]},
                           getElementType(context_, tensor->unit_size()));


  // build tl_load operation
  if (dtype == COEFF) {
    auto op = OpBuilder(get_start_op()).create<tpu::TL_LG_LoadCoeffOp>(get_start_op()->getLoc(),
            output_type, ArrayRef<Value *>{operands}, ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(name, op.getResult(), true);
  } else {
    auto op = OpBuilder(get_start_op()).create<tpu::TL_LG_LoadNeuronOp>(get_start_op()->getLoc(),
              output_type, ArrayRef<Value *>{operands}, ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(name, op.getResult(), true);
  }
}

// do not support concat optimization
void MixNet::_add_store_param(int tensor_id, net_timestep * time_step, int timestep_idx) {
  int tensor_dim[4];
  int local_shape[4], global_shape[4];
  Tensor* tensor = net_graph_->get_tensor_by_id(tensor_id);
  const vector<int>& dst_layers = net_graph_->get_tensor_to_layer(tensor_id);
  u32 laddr = net_graph_->get_tensor_local_offset(tensor_id);
  u64 gaddr = net_graph_->get_tensor_global_mem(tensor_id);
  bool aligned = true;
  bool transpose = false;
  string direction = "L2S";

  net_graph_->get_tensor_dim(tensor_id, tensor_dim);

  int n_idx = tensor->n_idx;
  int n_slice = tensor->n_slice;
  int h_idx = tensor->h_idx;
  int h_slice = tensor->h_slice;
  int h_end = h_idx + h_slice;
  h_idx = h_idx > 0 ? h_idx : 0;
  h_slice = h_end > tensor_dim[2] ? (tensor_dim[2] - h_idx) : (h_end - h_idx);

  string tensor_name = tensor->name();
  tensor_name += _get_postfix_name(tensor->get_group_id(), tensor->get_n_loop(), tensor->get_h_loop());
  Value *src_opd = get_op_from_name(tensor_name);
  string store_op_name = tensor_name + "_st";

  local_shape[0] = (n_slice);
  local_shape[1] = (tensor_dim[1]);
  local_shape[2] = (h_slice);
  local_shape[3] = (tensor_dim[3]);

  global_shape[0] = (tensor_dim[0]);
  global_shape[1] = (tensor_dim[1]);
  global_shape[2] = (tensor_dim[2]);
  global_shape[3] = (tensor_dim[3]);

  gaddr += (n_idx * tensor_dim[1] * tensor_dim[2] * tensor_dim[3] + h_idx * tensor_dim[3]) *
            tensor->unit_size();

  vector<NamedAttribute> attrs;
  Builder builder_(context_);
  attrs.push_back(builder_.getNamedAttr("name", builder_.getStringAttr(store_op_name)));
  attrs.push_back(builder_.getNamedAttr("ls_direction", builder_.getStringAttr(direction)));
  attrs.push_back(builder_.getNamedAttr("laddr", builder_.getI64IntegerAttr(laddr)));
  attrs.push_back(builder_.getNamedAttr("gaddr", builder_.getI64IntegerAttr(gaddr)));
  attrs.push_back(builder_.getNamedAttr("align", builder_.getBoolAttr(aligned)));
  attrs.push_back(builder_.getNamedAttr("transpose", builder_.getBoolAttr(transpose)));

  // setup input operation
  vector<Value *> operands;
  operands.push_back(src_opd);
  RankedTensorType output_type = RankedTensorType::get(
                          {global_shape[0], global_shape[1],
                           global_shape[2], global_shape[3]},
                           getElementType(context_, tensor->unit_size()));


  // build tl_load operation
  auto op = OpBuilder(get_start_op()).create<tpu::TL_LG_StoreOp>(get_start_op()->getLoc(),
          output_type, ArrayRef<Value *>{operands}, ArrayRef<NamedAttribute>{attrs});
  add_opd_to_list(store_op_name, op.getResult(), true);

}

}
