/*
 * Copyright (C) Bitmain Technologies Inc.
 * All Rights Reserved.
 */
#include "MixNet.hpp"

#define DEBUG_TYPE "group_ops"

namespace mlir {

static Type getStorageType(MLIRContext *context, std::string storage) {
  Builder builder(context);
  if (storage == "INT8" ||
      storage == "UINT8") {
    return builder.getIntegerType(8);
  } else if (storage == "INT16" ||
             storage == "UINT16") {
    return builder.getIntegerType(16);
  } else if (storage == "BF16") {
    return builder.getBF16Type();
  } else if (storage == "FP32") {
    return builder.getF32Type();
  } else if (storage == "UINT32" ||
             storage == "INT32") {
    return builder.getIntegerType(32);
  } else {
    assert(!"Not supported storage type.\n");
  }
}

void MixOp::add_bottom_name(std::string bottom_name) {
  operands_.push_back(bottom_name);
}

void MixOp::add_top_name(std::string top_name) {
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

Value MixNet::get_op_from_name(std::string name) {
  if (name_op_map_.find(name)!= name_op_map_.end()) {
    return (Value)name_op_map_[name];
  } else {
    LLVM_DEBUG(llvm::errs() << "Cannot find op name " << name << " in MixNet.\n";);
    assert(0);
  }
}

void MixNet::add_opd_to_list(std::string op_name, Value opd, bool b_generated) {
  std::pair <std::map<std::string, Value>::iterator, bool> ptr;
  ptr = name_op_map_.insert(std::pair<std::string, Value>(op_name, opd));
  if (b_generated)
    parallel_list_.push_back(opd.getDefiningOp());
  // if (!ptr.second) {
  //   LLVM_DEBUG(llvm::errs() << "Value aleady inserted in op_name map, " << op_name << "\n";);
  // }
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
      op.setEnableParallel(true);
    }

    if ( i == (op_size - 1)) {
      auto op = llvm::dyn_cast<tpu::TpuTLOpCodegenInterface>(cur_op);
      op.setDisableParallel(true);
    }
  }
}

void MixNet::set_net_in_tensor(int tensor_id) {
  this->net_in_tensors_.push_back(tensor_id);
}

void MixNet::set_net_out_tensor(int tensor_id) {
  this->net_out_tensors_.push_back(tensor_id);
}

static const std::string _get_postfix_name(int group_idx,
                                          int n_loop,
                                          int h_loop) {
  const std::string name = std::string("_") +
                           std::to_string(group_idx) + "_" +
                           std::to_string(n_loop) + "_" +
                           std::to_string(h_loop);
  return name;
}


// add group start layer
void MixNet::add_group_start_ops(int group_idx, Group* group,
                                 Operation *op,
                                 int n_secs, int h_secs) {
  Builder builder_(context_);
  std::set<int> in_neuron_tensors = group->get_group_in_neuron_tensors();

  for (auto tid : in_neuron_tensors) {
   int from_layer = net_graph_->get_tensor_from_layer(tid);
    const ImLayer * im_layer = net_graph_->get_layer_by_id(from_layer);
    assert(im_layer->op()->getNumResults() == 1);
    std::string name = top_name(im_layer->op(),0).str();
    Value in_op = im_layer->op()->getResult(0);
    add_opd_to_list(name, in_op, false);
  }
}

void MixNet::add_group_end_ops(int group_idx, Group* group, int n_secs, int h_secs) {
  Builder builder_(context_);
  std::vector<int> out_neuron_tensors = group->get_group_out_tensors();

  for (auto tid : out_neuron_tensors) {
    int from_layer = net_graph_->get_tensor_from_layer(tid);
    const ImLayer * im_layer = net_graph_->get_layer_by_id(from_layer);
    Operation * old_op = im_layer->op();
    std::string old_name = top_name(old_op,0).str();
    Value old_op_r = old_op->getResult(0);
    std::vector<Value> operands;
    for (int i = 0; i < n_secs; i++) {
      for (int j = 0; j < h_secs; j++) {
        std::string new_name = old_name + _get_postfix_name(group_idx, i, j);
        new_name += "_st";
        Value new_opd = get_op_from_name(new_name);
        operands.push_back(new_opd);
      }
    }
    std::vector<NamedAttribute> attrs;
    attrs.push_back(builder_.getNamedAttr("name",
                    builder_.getStringAttr(old_name)));
    auto join_op = OpBuilder(get_start_op()).create<tpu::TL_LG_JoinOp>(
                             get_start_op()->getLoc(), old_op_r.getType(),
                             ArrayRef<Value>{operands},
                             ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(old_name, join_op.getResult(), false);

    // replace the next group's usage to this new generated op
    old_op->replaceAllUsesWith(join_op);

  }
}

void MixNet::add_tl_layer(int group_idx, int layer_id, net_timestep* time_step, int timestep_idx,
                          bool is_h_split, int n_loop, int h_loop) {
  const ImLayer* im_layer = net_graph_->get_layer_by_id(layer_id);
  const std::vector<int>& in_tensors = net_graph_->get_in_tensors_of_layer(layer_id);
  const std::vector<int>& out_tensors = net_graph_->get_out_tensors_of_layer(layer_id);
  std::string postfix = _get_postfix_name(group_idx, n_loop, h_loop);

  MixOp * mix_op = new MixOp(this, layer_id);
  mix_op->set_name(im_layer->name() + postfix);

  for (uint32_t i = 0; i < in_tensors.size(); i++) {
    Tensor* in_tensor = net_graph_->get_tensor_by_id(in_tensors[i]);
    const std::string& name = in_tensor->name();
    if (in_tensor->type() != TENSOR_NEURON &&
        in_tensor->type() != TENSOR_NEURON_WINOGRAD &&
        in_tensor->type() != TENSOR_MATRIX) {
      // coeff not do slice, no need postfix
      mix_op->add_bottom_name(name);
    } else {
      mix_op->add_bottom_name(name + postfix);
    }
  }

  for (uint32_t i = 0; i < out_tensors.size(); i++) {
    Tensor* out_tensor = net_graph_->get_tensor_by_id(out_tensors[i]);
    const std::string& name = out_tensor->name();
    mix_op->add_top_name(name + postfix);
  }

  for (uint32_t i = 0; i < out_tensors.size(); ++i) {
    mem_buffer_key_t key = {timestep_idx, out_tensors[i], false};
    const mem_buffer_value_t* value = time_step->get_mem_buffer_value(&key);
    net_graph_->set_tensor_local_offest(out_tensors[i], value->local_mem_offset);
  }

  IR_TYPE layer_type = im_layer->type();

  switch (layer_type) {
    case IR_ABS:
      mix_op->set_type("tl_abs");
      _add_tl_abs_op(mix_op, in_tensors, out_tensors, time_step,
                      timestep_idx, is_h_split);
     break;
    case IR_CONVOLUTION:
      mix_op->set_type("tl_convolution");
      _add_tl_convolution_op(mix_op, in_tensors, out_tensors,
                             time_step, timestep_idx, is_h_split);
      break;
    case IR_DECONVOLUTION:
      mix_op->set_type("tl_deconvolution");
      _add_tl_deconvolution_op(mix_op, in_tensors, out_tensors,
                               time_step, timestep_idx, is_h_split);
      break;
    case IR_ELTWISE:
      mix_op->set_type("tl_eltwise");
      _add_tl_eltwise_op(mix_op, in_tensors, out_tensors,
                          time_step, timestep_idx, is_h_split);
      break;
    case IR_POOLING:
      mix_op->set_type("tl_pooling");
      _add_tl_pooling_op(mix_op, in_tensors, out_tensors,
                          time_step, timestep_idx, is_h_split);
      break;
    case IR_LRN:
      mix_op->set_type("tl_lrn");
      _add_tl_lrn_op(mix_op, in_tensors, out_tensors,
                      time_step, timestep_idx, is_h_split);
      break;
    case IR_BROADCAST_MUL:
      mix_op->set_type("tl_broadcast_mul");
      _add_tl_broadcast_mul_op(mix_op, in_tensors, out_tensors,
                               time_step, timestep_idx, is_h_split);
      break;
    case IR_ACTIVATION:
      mix_op->set_type("tl_activation");
      _add_tl_activation_op(mix_op, in_tensors, out_tensors,
                            time_step, timestep_idx, is_h_split);
      break;
    case IR_UPSAMPLE:
      mix_op->set_type("tl_upsample");
      _add_tl_upsample_op(mix_op, in_tensors, out_tensors, time_step,
                          timestep_idx, is_h_split);
      break;
    case IR_LEAKY_RELU:
      mix_op->set_type("tl_leaky_relu");
      _add_tl_leaky_relu_op(mix_op, in_tensors, out_tensors, time_step,
                            timestep_idx, is_h_split);
      break;
    case IR_PRELU:
      mix_op->set_type("tl_leaky_relu");
      _add_tl_prelu_op(mix_op, in_tensors, out_tensors, time_step,
                       timestep_idx, is_h_split);
      break;
    case IR_CONCAT:
      mix_op->set_type("tl_concat");
      _add_tl_concat_op(mix_op, in_tensors, out_tensors, time_step,
                        timestep_idx, is_h_split);
      break;
    case IR_PAD:
      mix_op->set_type("tl_pad");
      _add_tl_pad_op(mix_op, in_tensors, out_tensors, time_step,
                     timestep_idx, is_h_split);
      break;
    case IR_CROP:
      mix_op->set_type("tl_crop");
      _add_tl_crop_op(mix_op, in_tensors, out_tensors, time_step,
                      timestep_idx, is_h_split);
     break;
    case IR_RELU:
      mix_op->set_type("tl_relu");
      _add_tl_relu_op(mix_op, in_tensors, out_tensors, time_step,
                      timestep_idx, is_h_split);
     break;
    case IR_QUANT:
      mix_op->set_type("tl_quant");
      _add_tl_quant_op(mix_op, in_tensors, out_tensors, time_step,
                        timestep_idx, is_h_split);
      break;
    case IR_ZERO_MASK:
      mix_op->set_type("tl_zero_mask");
      _add_tl_zero_mask_op(mix_op, in_tensors, out_tensors, time_step,
                            timestep_idx, is_h_split);
      break;
    case IR_SLICE:
      mix_op->set_type("tl_slice");
      _add_tl_slice_op(mix_op, in_tensors, out_tensors, time_step,
                      timestep_idx, is_h_split);
     break;

    default:
      llvm::errs() << "unknown layer type:" << layer_type << "\n";
  }
}

static void add_fused_leaky_attrs(Builder &builder, Operation * op,
                             std::vector<NamedAttribute> & attrs) {
  if (auto conv_op = dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(op)) {
    if (conv_op.negative_slope().hasValue())
      attrs.push_back(builder.getNamedAttr("negative_slope",
                      conv_op.negative_slopeAttr()));
    if (conv_op.rshift_pos().hasValue())
      attrs.push_back(builder.getNamedAttr("rshift_pos",
                      conv_op.rshift_posAttr()));
    if (conv_op.m_i8_pos().hasValue())
      attrs.push_back(builder.getNamedAttr("m_i8_pos",
                      conv_op.m_i8_posAttr()));
    if (conv_op.rshift_neg().hasValue())
      attrs.push_back(builder.getNamedAttr("rshift_neg",
                      conv_op.rshift_negAttr()));
    if (conv_op.m_i8_neg().hasValue())
      attrs.push_back(builder.getNamedAttr("m_i8_neg",
                      conv_op.m_i8_negAttr()));
    attrs.push_back(builder.getNamedAttr("do_leaky_relu",
                    builder.getBoolAttr(true)));
  } else if (auto conv_op = dyn_cast<tpu::TG_BF16_Conv2DOp>(op)) {
    if (conv_op.negative_slope().hasValue())
      attrs.push_back(builder.getNamedAttr("negative_slope",
                      conv_op.negative_slopeAttr()));
    if (conv_op.rshift_pos().hasValue())
      attrs.push_back(builder.getNamedAttr("rshift_pos",
                      conv_op.rshift_posAttr()));
    if (conv_op.m_i8_pos().hasValue())
      attrs.push_back(builder.getNamedAttr("m_i8_pos",
                      conv_op.m_i8_posAttr()));
    if (conv_op.rshift_neg().hasValue())
      attrs.push_back(builder.getNamedAttr("rshift_neg",
                      conv_op.rshift_negAttr()));
    if (conv_op.m_i8_neg().hasValue())
      attrs.push_back(builder.getNamedAttr("m_i8_neg",
                      conv_op.m_i8_negAttr()));
    attrs.push_back(builder.getNamedAttr("do_leaky_relu",
                    builder.getBoolAttr(true)));
  }

}

void MixNet::_add_tl_abs_op(MixOp * mix_op,
                             const std::vector<int>& in_tensors,
                             const std::vector<int>& out_tensors,
                             net_timestep* time_step,
                             int timestep_idx,
                             bool is_h_split) {
  const ImLayer* im_layer = net_graph_->get_layer_by_id(mix_op->get_layer_id());
  Operation *op = im_layer->op();
  auto opd0 = op->getOperand(0);
  auto old_input_type = opd0.getType().cast<RankedTensorType>();

  int bottom_dim[4];
  int top_dim[4];

  net_graph_->get_tl_tensor_dim(in_tensors[0], bottom_dim, is_h_split);
  net_graph_->get_tl_tensor_dim(out_tensors[0], top_dim, is_h_split);

  std::string name = mix_op->name();
  uint32_t la_input = net_graph_->get_tensor_local_offset(in_tensors[0]);
  uint32_t la_output = net_graph_->get_tensor_local_offset(out_tensors[0]);

  Builder builder_(context_);
  std::vector<NamedAttribute> attrs;

  attrs.push_back(builder_.getNamedAttr("name",
                           builder_.getStringAttr(name)));
  attrs.push_back(builder_.getNamedAttr("align",
                           builder_.getBoolAttr(true)));
  attrs.push_back(builder_.getNamedAttr("la_input",
                           builder_.getI32IntegerAttr(la_input)));
  attrs.push_back(builder_.getNamedAttr("la_output",
                           builder_.getI32IntegerAttr(la_output)));

  RankedTensorType output_type = RankedTensorType::get(
                          {top_dim[0], top_dim[1],
                           top_dim[2], top_dim[3]},
                           old_input_type.getElementType());

   // setup input operation
  std::vector<Value> operands;
  Operation * input_op =
                    get_op_from_name(mix_op->bottom_name(0)).getDefiningOp();
  input_op->getResult(0).setType(output_type);
  operands.push_back(input_op->getResult(0));

  // build tl_abs operation
  if (isa<tpu::TG_INT8_AbsOp>(op)) {
    auto tl_op = OpBuilder(get_start_op()).create<tpu::TL_LG_INT8_AbsOp>(
                        get_start_op()->getLoc(), output_type,
                        ArrayRef<Value>{operands},
                        ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(mix_op->name(), tl_op.getResult(), true);
  } else if (isa<tpu::TG_BF16_AbsOp>(op)) {
    auto tl_op = OpBuilder(get_start_op()).create<tpu::TL_LG_BF16_AbsOp>(
                        get_start_op()->getLoc(), output_type,
                        ArrayRef<Value>{operands},
                        ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(mix_op->name(), tl_op.getResult(), true);
  }
}

void MixNet::_add_tl_convolution_op(MixOp* mix_op,
                                    const std::vector<int>& in_tensors,
                                    const std::vector<int>& out_tensors,
                                    net_timestep* time_step,
                                    int timestep_idx, bool is_h_split) {
  const ImLayer* im_layer = net_graph_->get_layer_by_id(mix_op->get_layer_id());
  Operation * op = im_layer->op();
  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw;
  int sh, sw, pt, pb, pl, pr, dh, dw, pad_value;
  bool do_ic_align = false;
  bool do_leaky_relu = false;
  bool bInt8ConvOp = isa<tpu::TG_INT8_PC_Conv2DOp>(op);

  getConvParam(op, n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, pt, pb, pl, pr,
               dh, dw, is_dw, with_bias, do_relu, do_ic_align, do_leaky_relu,
               pad_value);

  bool has_bias_op = (bInt8ConvOp || (!bInt8ConvOp && with_bias));
  auto old_input_type = op->getOperand(0).getType().cast<RankedTensorType>();

  int bottom_dim[4];
  int top_dim[4];

  int pad_h[2];
  int pad_w[2];

  pad_h[0] = pt;
  pad_h[1] = pb;
  pad_w[0] = pl;
  pad_w[1] = pr;

  net_graph_->get_tl_tensor_dim_pads(in_tensors[0], bottom_dim, pad_h, is_h_split);
  net_graph_->get_tl_tensor_dim(out_tensors[0], top_dim, is_h_split);

  uint32_t input_laddr = (net_graph_->get_tensor_local_offset(in_tensors[0]));
  uint32_t weight_laddr = (net_graph_->get_tensor_local_offset(in_tensors[1]));
  uint32_t output_laddr = (net_graph_->get_tensor_local_offset(out_tensors[0]));
  int bias_laddr = 0;
  if (has_bias_op)
    bias_laddr = net_graph_->get_tensor_local_offset(in_tensors[2]);
  int working_laddr = 0;


  RankedTensorType input_type = RankedTensorType::get(
                          {bottom_dim[0], bottom_dim[1],
                           bottom_dim[2], bottom_dim[3]},
                           old_input_type.getElementType());

  RankedTensorType output_type = RankedTensorType::get(
                          {top_dim[0], top_dim[1],
                           top_dim[2], top_dim[3]},
                           old_input_type.getElementType());

  // setup parameter
  std::vector<NamedAttribute> attrs;
  Builder builder_(context_);
  attrs.push_back(builder_.getNamedAttr("name",
                          builder_.getStringAttr(mix_op->name())));
  attrs.push_back(builder_.getNamedAttr("la_input",
                           builder_.getI32IntegerAttr(input_laddr)));
  attrs.push_back(builder_.getNamedAttr("la_output",
                           builder_.getI32IntegerAttr(output_laddr)));
  attrs.push_back(builder_.getNamedAttr("la_filter",
                           builder_.getI32IntegerAttr(weight_laddr)));
  attrs.push_back(builder_.getNamedAttr("la_bias",
                           builder_.getI32IntegerAttr(bias_laddr)));
  attrs.push_back(builder_.getNamedAttr("pad_top_h",
                           builder_.getI32IntegerAttr(pad_h[0])));
  attrs.push_back(builder_.getNamedAttr("pad_bottom_h",
                           builder_.getI32IntegerAttr(pad_h[1])));
  attrs.push_back(builder_.getNamedAttr("pad_left_w",
                           builder_.getI32IntegerAttr(pad_w[0])));
  attrs.push_back(builder_.getNamedAttr("pad_right_w",
                           builder_.getI32IntegerAttr(pad_w[1])));
  attrs.push_back(builder_.getNamedAttr("param",
    tpu::ConvParam::get(
      builder_.getI32IntegerAttr(sh),
      builder_.getI32IntegerAttr(sw),
      builder_.getStringAttr("VALID"),
      builder_.getI32IntegerAttr(dh),
      builder_.getI32IntegerAttr(dw),
      builder_.getI32IntegerAttr(0), // pd_t
      builder_.getI32IntegerAttr(0), // pd_b
      builder_.getI32IntegerAttr(0), // pd_l
      builder_.getI32IntegerAttr(0), // pd_r
      builder_.getI32IntegerAttr(g),
      builder_.getBoolAttr(is_dw),
      builder_.getBoolAttr(with_bias),
      builder_.getBoolAttr(do_relu),
      builder_.getI32ArrayAttr(ArrayRef<int32_t>({})), // [0]ins_w/[1]ins_h
      builder_.getI32IntegerAttr(0), //pad_value
      builder_.getContext())));

  if(do_ic_align){
      attrs.push_back(builder_.getNamedAttr("do_ic_alignment",
                           builder_.getBoolAttr(do_ic_align)));
  }
  if(do_leaky_relu) {
    add_fused_leaky_attrs(builder_, op, attrs);
    mem_buffer_key_t key = {timestep_idx, im_layer->imm_tensors[0].get()->id(), true};
    const mem_buffer_value_t* imm = time_step->get_mem_buffer_value(&key);
    working_laddr = (imm->local_mem_offset);
  }

  attrs.push_back(builder_.getNamedAttr("la_working",
                           builder_.getI32IntegerAttr(working_laddr)));
  // setup input operation
  std::vector<Value> operands;
  Operation * input_op =
    get_op_from_name(mix_op->bottom_name(0)).getDefiningOp();
  input_op->getResult(0).setType(input_type);
  operands.push_back(input_op->getResult(0));

  // setup filter operation
  Operation * filter_op =
    get_op_from_name(mix_op->bottom_name(1)).getDefiningOp();
  operands.push_back(filter_op->getResult(0));
  // setup bias operation
  if (has_bias_op) {
    Operation * bias_op =
      get_op_from_name(mix_op->bottom_name(2)).getDefiningOp();
    operands.push_back(bias_op->getResult(0));
  } else {
    auto none_op = OpBuilder(get_start_op()).create<tpu::NoneOp>(
                            builder_.getUnknownLoc(), builder_.getNoneType());
    operands.push_back(none_op.getResult());
  }

  // build tl_conv operation
  if (isa<tpu::TG_INT8_PC_Conv2DOp>(op)) {
    auto tl_op = OpBuilder(get_start_op()).create<tpu::TL_LG_INT8_Conv2DOp>(
                        get_start_op()->getLoc(), output_type,
                        ArrayRef<Value>{operands},
                        ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(mix_op->name(), tl_op.getResult(), true);
  } else if (isa<tpu::TG_BF16_Conv2DOp>(op)) {
    auto tl_op = OpBuilder(get_start_op()).create<tpu::TL_LG_BF16_Conv2DOp>(
                    get_start_op()->getLoc(), output_type,
                    ArrayRef<Value>{operands},
                    ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(mix_op->name(), tl_op.getResult(), true);
  }

}

void MixNet::_add_tl_deconvolution_op(MixOp* mix_op,
                                      const std::vector<int>& in_tensors,
                                      const std::vector<int>& out_tensors, net_timestep* time_step,
                                      int timestep_idx, bool is_h_split) {
  const ImLayer* im_layer = net_graph_->get_layer_by_id(mix_op->get_layer_id());
  Operation * op = im_layer->op();
  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw;
  int sh, sw, pt, pb, pl, pr, dh, dw, pad_value;
  bool do_ic_align = false;
  bool do_leaky_relu = false;
  bool bInt8ConvOp = isa<tpu::TG_INT8_PC_DeConv2DOp>(op);

  getConvParam(op, n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, pt, pb, pl, pr,
               dh, dw, is_dw, with_bias, do_relu, do_ic_align, do_leaky_relu,
               pad_value);

  bool has_bias_op = (bInt8ConvOp || (!bInt8ConvOp && with_bias));
  auto old_input_type = op->getOperand(0).getType().cast<RankedTensorType>();
  Tensor* in_tensor = im_layer->in_tensors[0].get();

  int real_h_idx, real_h_slice;
  int pad_h_top, pad_h_bottom;
  int pad_w_left, pad_w_right;
  int h_end;
  int bottom_dim[4];
  int top_dim[4];

  net_graph_->get_tensor_dim(in_tensors[0], bottom_dim);
  net_graph_->get_tensor_dim(out_tensors[0], top_dim);

  bottom_dim[0] = in_tensor->n_slice;
  bottom_dim[2] = in_tensor->h_slice;
  pad_h_top = pt;
  pad_h_bottom = pb;
  pad_w_left = pl;
  pad_w_right = pr;

  real_h_slice = in_tensor->h_slice;
  real_h_idx = in_tensor->h_idx;
  if (is_h_split) {
    if (in_tensor->h_idx > 0) {
      real_h_idx = in_tensor->h_idx;
    } else {
      real_h_idx = 0;
    }
    h_end = in_tensor->h_idx + in_tensor->h_slice;
    if (h_end > in_tensor->h()) {
      real_h_slice = in_tensor->h() - real_h_idx;
    } else {
      real_h_slice = h_end - real_h_idx;
    }
    bottom_dim[2] = real_h_slice;
  }

  const Tensor* out_tensor = net_graph_->get_tensor_by_id(out_tensors[0]);
  top_dim[0] = out_tensor->n_slice;
  top_dim[2] = out_tensor->h_slice;

  real_h_slice = out_tensor->h_slice;
  real_h_idx = out_tensor->h_idx;
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
  int kh_ext = (kh - 1) * dh + 1;
  int kw_ext = (kw - 1) * dw + 1;
  int ins_last_w = (ow + pad_w_left + pad_w_right - kw_ext) % sw;
  int height_insert0 = (ih - 1) * sh + 1;
  pad_h_top = kh_ext - pad_h_top - 1;
  pad_h_bottom = kh_ext - pad_h_bottom - 1;
  int o_ht = real_h_idx;
  int o_hb = real_h_idx + real_h_slice;
  int if_pad_h_t = o_ht;
  int if_pad_h_b = o_hb + kh_ext - 1;
  int if_insert_h_t = 0;
  int pad_h_t = 0;
  if(if_pad_h_t < pad_h_top){
    pad_h_t = pad_h_top - if_pad_h_t;
  }else{
    if_insert_h_t = if_pad_h_t - pad_h_top;
  }
  int if_insert_h_b = height_insert0;
  int pad_h_b = 0;
  if( (if_pad_h_b - pad_h_bottom) < height_insert0 ){
    if_insert_h_b = if_pad_h_b - pad_h_bottom;
  }else{
    pad_h_b = if_pad_h_b - height_insert0 - pad_h_bottom;
  }
  int hinsert0_t = if_insert_h_t % sh == 0 ? 0:
      (sh - if_insert_h_t % sh);
  int hinsert0_b = (if_insert_h_b + sh - 1) % sh;

  pad_h_top = pad_h_t + hinsert0_t;
  pad_h_bottom = pad_h_b + hinsert0_b;
  pad_w_left = kw_ext - pad_w_left - 1;
  pad_w_right = kw_ext - pad_w_right - 1 + ins_last_w;
  int ins_h = sh - 1;
  int ins_last_h = 0;
  int ins_w = sw - 1;
  uint32_t input_laddr = (net_graph_->get_tensor_local_offset(in_tensors[0]));
  uint32_t weight_laddr = (net_graph_->get_tensor_local_offset(in_tensors[1]));
  uint32_t output_laddr = (net_graph_->get_tensor_local_offset(out_tensors[0]));
  int bias_laddr = 0;
  if (has_bias_op)
    bias_laddr = net_graph_->get_tensor_local_offset(in_tensors[2]);

  // hw limitation once set ins_w / ins_h that input w/h should > 1
  if (ins_h && bottom_dim[2] == 1) {
    ins_last_h += ins_h;
    ins_h = 0;
    if (pad_h_top) {
      ins_last_h = 0; // included in pad_h_top
    }
  }
  if (ins_w && bottom_dim[3] == 1) {
    ins_last_w += ins_w;
    ins_w = 0;
    if (pad_w_left) {
      // TODO: need to verify
      ins_last_w = 0; // included in pad_w_left
    }
  }


  RankedTensorType input_type = RankedTensorType::get(
                          {bottom_dim[0], bottom_dim[1],
                           bottom_dim[2], bottom_dim[3]},
                           old_input_type.getElementType());

  RankedTensorType output_type = RankedTensorType::get(
                          {top_dim[0], top_dim[1],
                           top_dim[2], top_dim[3]},
                           old_input_type.getElementType());

  // setup parameter
  std::vector<NamedAttribute> attrs;
  Builder builder_(context_);
  attrs.push_back(builder_.getNamedAttr("name", builder_.getStringAttr(mix_op->name())));
  attrs.push_back(builder_.getNamedAttr("param",
    tpu::ConvParam::get(
      builder_.getI32IntegerAttr(sh),
      builder_.getI32IntegerAttr(sw),
      builder_.getStringAttr("VALID"),
      builder_.getI32IntegerAttr(dh),
      builder_.getI32IntegerAttr(dw),
      builder_.getI32IntegerAttr(0), // pd_t
      builder_.getI32IntegerAttr(0), // pd_b
      builder_.getI32IntegerAttr(0), // pd_l
      builder_.getI32IntegerAttr(0), // pd_r
      builder_.getI32IntegerAttr(g),
      builder_.getBoolAttr(is_dw),
      builder_.getBoolAttr(with_bias),
      builder_.getBoolAttr(do_relu),
      builder_.getI32ArrayAttr(ArrayRef<int32_t>({})), // [0]ins_w/[1]ins_h
      builder_.getI32IntegerAttr(0), //pad_value
      builder_.getContext())));
  attrs.push_back(builder_.getNamedAttr("la_input",
                           builder_.getI32IntegerAttr(input_laddr)));
  attrs.push_back(builder_.getNamedAttr("la_output",
                           builder_.getI32IntegerAttr(output_laddr)));
  attrs.push_back(builder_.getNamedAttr("la_filter",
                           builder_.getI32IntegerAttr(weight_laddr)));
  attrs.push_back(builder_.getNamedAttr("la_bias",
                           builder_.getI32IntegerAttr(bias_laddr)));
  attrs.push_back(builder_.getNamedAttr("la_working",
                           builder_.getI32IntegerAttr(0)));
  attrs.push_back(builder_.getNamedAttr("ins_h",
                           builder_.getI32IntegerAttr(ins_h)));
  attrs.push_back(builder_.getNamedAttr("ins_last_h",
                           builder_.getI32IntegerAttr(ins_last_h)));
  attrs.push_back(builder_.getNamedAttr("ins_w",
                           builder_.getI32IntegerAttr(ins_w)));
  attrs.push_back(builder_.getNamedAttr("ins_last_w",
                           builder_.getI32IntegerAttr(ins_last_w)));
  attrs.push_back(builder_.getNamedAttr("pad_top_h",
                           builder_.getI32IntegerAttr(pad_h_top)));
  attrs.push_back(builder_.getNamedAttr("pad_bottom_h",
                           builder_.getI32IntegerAttr(pad_h_bottom)));
  attrs.push_back(builder_.getNamedAttr("pad_left_w",
                           builder_.getI32IntegerAttr(pad_w_left)));
  attrs.push_back(builder_.getNamedAttr("pad_right_w",
                           builder_.getI32IntegerAttr(pad_w_right)));
  if(do_ic_align){
      attrs.push_back(builder_.getNamedAttr("do_ic_alignment",
                               builder_.getBoolAttr(do_ic_align)));
  }

  // setup input operation
  std::vector<Value> operands;
  Operation * input_op =
    get_op_from_name(mix_op->bottom_name(0)).getDefiningOp();
  input_op->getResult(0).setType(input_type);
  operands.push_back(input_op->getResult(0));

  // setup filter operation
  Operation * filter_op =
    get_op_from_name(mix_op->bottom_name(1)).getDefiningOp();
  operands.push_back(filter_op->getResult(0));
  // setup bias operation
  if (has_bias_op) {
    Operation * bias_op =
      get_op_from_name(mix_op->bottom_name(2)).getDefiningOp();
    operands.push_back(bias_op->getResult(0));
  } else {
    auto none_op = OpBuilder(get_start_op()).create<tpu::NoneOp>(
                            builder_.getUnknownLoc(), builder_.getNoneType());
    operands.push_back(none_op.getResult());
  }

  // build tl_deconv operation
  if (isa<tpu::TG_INT8_PC_DeConv2DOp>(op)) {
    auto tl_op = OpBuilder(get_start_op()).create<tpu::TL_LG_INT8_DeConv2DOp>(
                        get_start_op()->getLoc(), output_type,
                        ArrayRef<Value>{operands},
                        ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(mix_op->name(), tl_op.getResult(), true);
  } else if (isa<tpu::TG_BF16_DeConv2DOp>(op)) {
    auto tl_op = OpBuilder(get_start_op()).create<tpu::TL_LG_BF16_DeConv2DOp>(
                        get_start_op()->getLoc(), output_type,
                        ArrayRef<Value>{operands},
                        ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(mix_op->name(), tl_op.getResult(), true);
  }
}


void MixNet::_add_tl_eltwise_op(MixOp* mix_op,
                                const std::vector<int>& in_tensors,
                                const std::vector<int>& out_tensors,
                                net_timestep* time_step,
                                int timestep_idx, bool is_h_split) {
  int id = mix_op->get_layer_id();
  const ImLayer* im_layer = net_graph_->get_layer_by_id(id);
  if (isa<tpu::TG_INT8_EltwiseAddOp>(im_layer->op()) ||
      isa<tpu::TG_BF16_EltwiseAddOp>(im_layer->op())) {
    _add_tl_eltwise_add_op(mix_op, in_tensors, out_tensors,
                           time_step, timestep_idx, is_h_split);
  } else if (isa<tpu::TG_INT8_EltwiseMulOp>(im_layer->op()) ||
             isa<tpu::TG_BF16_EltwiseMulOp>(im_layer->op())) {
    _add_tl_eltwise_mul_op(mix_op, in_tensors, out_tensors,
                           time_step, timestep_idx, is_h_split);
  }
}


void MixNet::_add_tl_eltwise_add_op(MixOp* mix_op,
                                const std::vector<int>& in_tensors,
                                const std::vector<int>& out_tensors,
                                net_timestep* time_step,
                                int timestep_idx, bool is_h_split) {
  const ImLayer* im_layer = net_graph_->get_layer_by_id(mix_op->get_layer_id());
  const Tensor* in_tensor = im_layer->in_tensors[0].get();
  Operation *op = im_layer->op();
  auto old_input_type =
       op->getOperand(0).getType().cast<RankedTensorType>();
  bool do_relu = false;
  getEltwiseReluParam(op, do_relu);
  int nInputs = op->getNumOperands();
  int bottom_dim[4];
  int top_dim[4];

  uint64_t working_laddr = 0;
  uint64_t la_output = 0;
  std::vector<int32_t> la_input;
  net_graph_->get_tensor_dim(in_tensors[0], bottom_dim);

  bottom_dim[0] = in_tensor->n_slice;
  bottom_dim[2] = in_tensor->h_slice;

  mem_buffer_key_t key = {timestep_idx, im_layer->imm_tensors[0].get()->id(), true};
  const mem_buffer_value_t* imm = time_step->get_mem_buffer_value(&key);
  working_laddr = (imm->local_mem_offset);

  assert(nInputs == 2);
  // input0, input1
  for (int i = 0; i < nInputs; i++) {
    la_input.push_back(net_graph_->get_tensor_local_offset(in_tensors[i]));
  }

  la_output = (net_graph_->get_tensor_local_offset(out_tensors[0]));

  // build eltwise op
  std::vector<NamedAttribute> attrs;
  Builder builder_(context_);
  attrs.push_back(builder_.getNamedAttr("name",
                           builder_.getStringAttr(mix_op->name())));
  attrs.push_back(builder_.getNamedAttr("la_input",
                  builder_.getI32ArrayAttr(ArrayRef<int32_t>({la_input}))));
  attrs.push_back(builder_.getNamedAttr("la_output",
                           builder_.getI32IntegerAttr(la_output)));
  attrs.push_back(builder_.getNamedAttr("la_working",
                           builder_.getI32IntegerAttr(working_laddr)));
  attrs.push_back(builder_.getNamedAttr("do_relu",
                           builder_.getBoolAttr(do_relu)));

  top_dim[2] = bottom_dim[2];
  top_dim[3] = bottom_dim[3];
  bool do_early_stride = false;
  int h_stride, w_stride;
  getEltwiseAddParam(op, do_early_stride, h_stride, w_stride);
  if (do_early_stride) {
    attrs.push_back(builder_.getNamedAttr("do_early_stride",
                             builder_.getBoolAttr(do_early_stride)));
    attrs.push_back(builder_.getNamedAttr("early_stride_h",
                             builder_.getI32IntegerAttr(h_stride)));
    attrs.push_back(builder_.getNamedAttr("early_stride_w",
                             builder_.getI32IntegerAttr(w_stride)));

    top_dim[2] = bottom_dim[2] / h_stride;
    top_dim[3] = bottom_dim[3] / w_stride;
  }

  if (auto add_op = dyn_cast<tpu::TG_INT8_EltwiseAddOp>(op)) {
    attrs.push_back(builder_.getNamedAttr("rshift", add_op.rshiftAttr()));
    attrs.push_back(builder_.getNamedAttr("m_i8", add_op.m_i8_inputsAttr()));
  } else if (auto add_op = dyn_cast<tpu::TG_BF16_EltwiseAddOp>(op)) {
    attrs.push_back(builder_.getNamedAttr("coeff", add_op.coeffAttr()));
  }

  // setup input/output type
  RankedTensorType input_type =
    RankedTensorType::get({
      bottom_dim[0], bottom_dim[1],
      bottom_dim[2], bottom_dim[3] },
      old_input_type.getElementType());

  RankedTensorType output_type =
    RankedTensorType::get({
      bottom_dim[0], bottom_dim[1],
      top_dim[2],    top_dim[3] },
      old_input_type.getElementType());

  // setup input operation
  std::vector<Value> operands;
  for( int i = 0; i < nInputs; i++) {
    Operation * input_op =
      get_op_from_name(mix_op->bottom_name(i)).getDefiningOp();
    input_op->getResult(0).setType(input_type);
    operands.push_back(input_op->getResult(0));
  }

  // build eltwiseadd operation
  if (isa<tpu::TG_INT8_EltwiseAddOp>(op)) {
    auto tl_op = OpBuilder(get_start_op()).create<tpu::TL_LG_INT8_EltwiseAddOp>(
                        get_start_op()->getLoc(), output_type,
                        ArrayRef<Value>{operands},
                        ArrayRef<NamedAttribute>{attrs});

    add_opd_to_list(mix_op->name(), tl_op.getResult(), true);
  } else if (isa<tpu::TG_BF16_EltwiseAddOp>(op)) {
    auto tl_op = OpBuilder(get_start_op()).create<tpu::TL_LG_BF16_EltwiseAddOp>(
                        get_start_op()->getLoc(), output_type,
                        ArrayRef<Value>{operands},
                        ArrayRef<NamedAttribute>{attrs});

    add_opd_to_list(mix_op->name(), tl_op.getResult(), true);
  }
}


void MixNet::_add_tl_eltwise_mul_op(MixOp* mix_op,
                                const std::vector<int>& in_tensors,
                                const std::vector<int>& out_tensors,
                                net_timestep* time_step,
                                int timestep_idx, bool is_h_split) {
  const ImLayer* im_layer = net_graph_->get_layer_by_id(mix_op->get_layer_id());
  const Tensor* in_tensor = im_layer->in_tensors[0].get();
  Operation *op  = im_layer->op();
  auto old_input_type =
       op->getOperand(0).getType().cast<RankedTensorType>();
  bool do_relu = false;
  getEltwiseReluParam(op, do_relu);
  int nInputs = op->getNumOperands();
  int bottom_dim[4];
  uint64_t working_laddr = 0;
  uint64_t la_output = 0;
  std::vector<int32_t> la_input;
  net_graph_->get_tensor_dim(in_tensors[0], bottom_dim);

  bottom_dim[0] = in_tensor->n_slice;
  bottom_dim[2] = in_tensor->h_slice;

  assert(nInputs == 2);
  // input0, input1
  for (int i = 0; i < nInputs; i++) {
    la_input.push_back(net_graph_->get_tensor_local_offset(in_tensors[i]));
  }

  la_output = (net_graph_->get_tensor_local_offset(out_tensors[0]));

  // build eltwise op
  std::vector<NamedAttribute> attrs;
  Builder builder_(context_);
  attrs.push_back(builder_.getNamedAttr("name",
                           builder_.getStringAttr(mix_op->name())));
  attrs.push_back(builder_.getNamedAttr("la_input",
                  builder_.getI32ArrayAttr(ArrayRef<int32_t>({la_input}))));
  attrs.push_back(builder_.getNamedAttr("la_output",
                           builder_.getI32IntegerAttr(la_output)));
  attrs.push_back(builder_.getNamedAttr("la_working",
                           builder_.getI32IntegerAttr(working_laddr)));
  attrs.push_back(builder_.getNamedAttr("do_relu",
                           builder_.getBoolAttr(do_relu)));
  if (auto add_op = dyn_cast<tpu::TG_INT8_EltwiseMulOp>(op)) {
    attrs.push_back(builder_.getNamedAttr("rshift", add_op.rshiftAttr()));
    attrs.push_back(builder_.getNamedAttr("m_i32_output", add_op.m_i32_outputAttr()));
  }

  // setup input/output type
  RankedTensorType input_type =
    RankedTensorType::get({
      bottom_dim[0], bottom_dim[1],
      bottom_dim[2], bottom_dim[3] },
      old_input_type.getElementType());

  RankedTensorType output_type =
    RankedTensorType::get({
      bottom_dim[0], bottom_dim[1],
      bottom_dim[2], bottom_dim[3] },
      old_input_type.getElementType());

  // setup input operation
  std::vector<Value> operands;
  for( int i = 0; i < nInputs; i++) {
    Operation * input_op =
      get_op_from_name(mix_op->bottom_name(i)).getDefiningOp();
    input_op->getResult(0).setType(input_type);
    operands.push_back(input_op->getResult(0));
  }

  // build eltwise_mul operation
  if (isa<tpu::TG_INT8_EltwiseMulOp>(op)) {
    auto tl_op = OpBuilder(get_start_op()).create<tpu::TL_LG_INT8_EltwiseMulOp>(
                        get_start_op()->getLoc(), output_type,
                        ArrayRef<Value>{operands},
                        ArrayRef<NamedAttribute>{attrs});

    add_opd_to_list(mix_op->name(), tl_op.getResult(), true);
  } else if (isa<tpu::TG_BF16_EltwiseMulOp>(op)) {
    auto tl_op = OpBuilder(get_start_op()).create<tpu::TL_LG_BF16_EltwiseMulOp>(
                        get_start_op()->getLoc(), output_type,
                        ArrayRef<Value>{operands},
                        ArrayRef<NamedAttribute>{attrs});

    add_opd_to_list(mix_op->name(), tl_op.getResult(), true);
  }
}

void MixNet::_add_tl_pooling_op(MixOp * mix_op,
                                const std::vector<int>& in_tensors,
                                const std::vector<int>& out_tensors,
                                net_timestep* time_step,
                                int timestep_idx,
                                bool is_h_split) {
  const ImLayer* im_layer = net_graph_->get_layer_by_id(mix_op->get_layer_id());
  const Tensor* in_tensor = net_graph_->get_tensor_by_id(in_tensors[0]);
  const Tensor* out_tensor = net_graph_->get_tensor_by_id(out_tensors[0]);
  Operation *op = im_layer->op();
  auto old_input_type = op->getOperand(0).getType().cast<RankedTensorType>();
  Builder builder_(context_);
  std::vector<NamedAttribute> attrs;
  // parse param
  bool is_global, do_relu, count_include_pad;
  int n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  getPoolingParam(op,
                  n, c, ih, iw, oh, ow,
                  kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
                  is_global, do_relu, count_include_pad);

  int bottom_dim[4];
  int top_dim[4];
  int pad_h[2];
  int pad_w[2];

  pad_h[0] = pt;
  pad_h[1] = pb;
  pad_w[0] = pl;
  pad_w[1] = pr;

  net_graph_->get_tensor_dim(in_tensors[0], bottom_dim);
  net_graph_->get_tensor_dim(out_tensors[0], top_dim);

  bottom_dim[0] = in_tensor->n_slice;
  bottom_dim[2] = in_tensor->h_slice;

  top_dim[0] = out_tensor->n_slice;
  top_dim[2] = out_tensor->h_slice;

  if (sh * (top_dim[2] - 1) + kh > bottom_dim[2] + pad_h[0] + pad_h[1]) {
    pad_h[0] = sh * (top_dim[2] - 1) + kh - bottom_dim[2] - pt;
  }

  if (sw * (top_dim[3] - 1) + kw > bottom_dim[3] + pad_w[0] + pad_w[1]) {
    pad_w[0] = sw * (top_dim[3] - 1) + kw - bottom_dim[3] - pl;
  }

  net_graph_->get_tl_tensor_dim_pads(in_tensors[0], bottom_dim, pad_h, is_h_split);
  net_graph_->get_tl_tensor_dim(out_tensors[0], top_dim, is_h_split);

  uint64_t la_input = (net_graph_->get_tensor_local_offset(in_tensors[0]));
  uint64_t la_output = (net_graph_->get_tensor_local_offset(out_tensors[0]));

   // build pooling op
  attrs.push_back(builder_.getNamedAttr("name",
                           builder_.getStringAttr(mix_op->name())));
  attrs.push_back(builder_.getNamedAttr("param",
    tpu::PoolParam::get(
      builder_.getI32IntegerAttr(kh),
      builder_.getI32IntegerAttr(kw),
      builder_.getI32IntegerAttr(pad_h[0]),
      builder_.getI32IntegerAttr(pad_h[1]),
      builder_.getI32IntegerAttr(pad_w[0]),
      builder_.getI32IntegerAttr(pad_w[1]),
      builder_.getI32IntegerAttr(pad_value),
      builder_.getI32IntegerAttr(sh),
      builder_.getI32IntegerAttr(sw),
      builder_.getBoolAttr(do_relu),
      builder_.getBoolAttr(count_include_pad),
      builder_.getContext())));
  attrs.push_back(builder_.getNamedAttr("la_input",
                           builder_.getI32IntegerAttr(la_input)));
  attrs.push_back(builder_.getNamedAttr("la_output",
                           builder_.getI32IntegerAttr(la_output)));

  if(auto tmp_op = dyn_cast<tpu::TG_INT8_PoolAvg2DOp>(im_layer->op())) {
    attrs.push_back(builder_.getNamedAttr("rshift", tmp_op.rshiftAttr()));
    attrs.push_back(builder_.getNamedAttr("m_i8", tmp_op.m_i8Attr()));
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
  std::vector<Value> operands;
  for( uint32_t i = 0; i < in_tensors.size(); i++) {
    Operation * input_op = get_op_from_name(mix_op->bottom_name(i)).getDefiningOp();
    input_op->getResult(0).setType(input_type);
    operands.push_back(input_op->getResult(0));
  }

  // build pooling operation
  if (isa<tpu::TG_INT8_PoolAvg2DOp>(op)) {
    auto tl_op = OpBuilder(get_start_op()).create<tpu::TL_LG_INT8_PoolAvg2DOp>(
                        get_start_op()->getLoc(), output_type,
                        ArrayRef<Value>{operands},
                        ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(mix_op->name(), tl_op.getResult(), true);
  } else if(isa<tpu::TG_INT8_PoolMax2DOp>(op)){
    auto tl_op = OpBuilder(get_start_op()).create<tpu::TL_LG_INT8_PoolMax2DOp>(
                        get_start_op()->getLoc(), output_type,
                        ArrayRef<Value>{operands},
                        ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(mix_op->name(), tl_op.getResult(), true);
  } else if (isa<tpu::TG_BF16_PoolAvg2DOp>(op)) {
    auto tl_op = OpBuilder(get_start_op()).create<tpu::TL_LG_BF16_PoolAvg2DOp>(
                        get_start_op()->getLoc(), output_type,
                        ArrayRef<Value>{operands},
                        ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(mix_op->name(), tl_op.getResult(), true);
  } else if (isa<tpu::TG_BF16_PoolMax2DOp>(op)) {
    auto tl_op = OpBuilder(get_start_op()).create<tpu::TL_LG_BF16_PoolMax2DOp>(
                        get_start_op()->getLoc(), output_type,
                        ArrayRef<Value>{operands},
                        ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(mix_op->name(), tl_op.getResult(), true);
  }
}

void MixNet::_add_tl_broadcast_mul_op(
                              MixOp * mix_op,
                              const std::vector<int>& in_tensors,
                              const std::vector<int>& out_tensors,
                              net_timestep* time_step,
                              int timestep_idx, bool is_h_split) {
  int bottom_dim[4];
  const ImLayer* im_layer =
      net_graph_->get_layer_by_id(mix_op->get_layer_id());
  Operation* op = im_layer->op();
  auto op_input_type =
    op->getOperand(0).getType().cast<RankedTensorType>();
  bool bInt8Op = isa<tpu::TG_INT8_BroadcastMulOp>(op);

  Tensor* in_tensor = net_graph_->get_tensor_by_id(in_tensors[0]);
  net_graph_->get_tensor_dim(in_tensors[0], bottom_dim);
  bottom_dim[0] = in_tensor->n_slice;
  bottom_dim[2] = in_tensor->h_slice;

  std::string name = mix_op->name();
  uint32_t la_input = net_graph_->get_tensor_local_offset(in_tensors[0]);
  uint32_t la_scale = net_graph_->get_tensor_local_offset(in_tensors[1]);
  uint32_t la_output = net_graph_->get_tensor_local_offset(out_tensors[0]);

  RankedTensorType input_type =
          RankedTensorType::get({
                  bottom_dim[0], bottom_dim[1],
                  bottom_dim[2], bottom_dim[3]},
                  op_input_type.getElementType());

  // setup parameter
  std::vector<NamedAttribute> attrs;
  Builder builder_(context_);
  attrs.push_back(builder_.getNamedAttr("name",
                           builder_.getStringAttr(mix_op->name())));
  attrs.push_back(builder_.getNamedAttr("la_input",
                           builder_.getI32IntegerAttr(la_input)));
  attrs.push_back(builder_.getNamedAttr("la_output",
                           builder_.getI32IntegerAttr(la_output)));
  attrs.push_back(builder_.getNamedAttr("la_scale",
                           builder_.getI32IntegerAttr(la_scale)));
  if (bInt8Op) {
    uint32_t la_bias = net_graph_->get_tensor_local_offset(in_tensors[2]);
    attrs.push_back(builder_.getNamedAttr("la_bias",
                            builder_.getI32IntegerAttr(la_bias)));
  } else {
    attrs.push_back(builder_.getNamedAttr("la_bias",
                            builder_.getI32IntegerAttr(0)));
  }

  bool do_relu = false;
  if(auto tmp_op = dyn_cast<tpu::TG_INT8_BroadcastMulOp>(op)) {
    do_relu = tmp_op.param().do_relu().getValue();
  } else if (auto tmp_op = dyn_cast<tpu::TG_BF16_BroadcastMulOp>(op)) {
    do_relu = tmp_op.param().do_relu().getValue();
  }
  attrs.push_back(builder_.getNamedAttr("do_relu",
                           builder_.getBoolAttr(do_relu)));
  // setup input operation
  std::vector<Value> operands;
  Operation * input_op =
    get_op_from_name(mix_op->bottom_name(0)).getDefiningOp();
  input_op->getResult(0).setType(input_type);
  operands.push_back(input_op->getResult(0));

  // setup filter operation
  Operation * scale_op =
    get_op_from_name(mix_op->bottom_name(1)).getDefiningOp();
  operands.push_back(scale_op->getResult(0));
  // setup bias operation
  if (bInt8Op) {
    Operation * bias_op =
      get_op_from_name(mix_op->bottom_name(2)).getDefiningOp();
    operands.push_back(bias_op->getResult(0));
  } else {
    auto none_op = OpBuilder(get_start_op()).create<tpu::NoneOp>(
                            builder_.getUnknownLoc(), builder_.getNoneType());
    operands.push_back(none_op.getResult());
  }

  // build tl_broadcast operation
  if (isa<tpu::TG_INT8_BroadcastMulOp>(op)) {
    auto tl_op = OpBuilder(get_start_op()).create<tpu::TL_LG_INT8_BroadcastMulOp>(
                        get_start_op()->getLoc(), input_type,
                        ArrayRef<Value>{operands},
                        ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(mix_op->name(), tl_op.getResult(), true);
  } else if (isa<tpu::TG_BF16_BroadcastMulOp>(op)) {
    auto tl_op = OpBuilder(get_start_op()).create<tpu::TL_LG_BF16_BroadcastMulOp>(
                        get_start_op()->getLoc(), input_type,
                        ArrayRef<Value>{operands},
                        ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(mix_op->name(), tl_op.getResult(), true);
  }
}

void MixNet::_add_tl_activation_op(MixOp * mix_op,
                                  const std::vector<int>& in_tensors,
                                  const std::vector<int>& out_tensors,
                                  net_timestep* time_step,
                                  int timestep_idx, bool is_h_split) {
  int bottom_dim[4];
  const ImLayer* im_layer =
      net_graph_->get_layer_by_id(mix_op->get_layer_id());
  RankedTensorType old_input_type;
  int is_int8 = 1;
  int lut_nr = 1; // one lut for int8
  Type bf16Type;
  std::string method;
  Builder builder_(context_);
  std::vector<NamedAttribute> attrs;
  float table_thresh_min;
  float table_thresh_max;
  bool added_offset;

  if (auto old_op = dyn_cast<tpu::TG_INT8_LutOp>(im_layer->op())) {
    old_input_type =
      old_op.getOperand(0).getType().cast<RankedTensorType>();
    table_thresh_min = old_op.min_range().convertToFloat();
    table_thresh_max = old_op.max_range().convertToFloat();
    added_offset = old_op.added_offset();
  }
  else if (auto old_op = dyn_cast<tpu::TG_BF16_LutOp>(im_layer->op())) {
    old_input_type =
      old_op.getResult().getType().cast<RankedTensorType>();
    is_int8 = 0;
    lut_nr = 2; // y0 + mantissa
    bf16Type = FloatType::getBF16(builder_.getContext()); // for td define
    table_thresh_min = old_op.min_range().convertToFloat();
    table_thresh_max = old_op.max_range().convertToFloat();
    added_offset = old_op.added_offset();
    attrs.push_back(builder_.getNamedAttr("method", old_op.methodAttr()));
  }
  else {
    llvm_unreachable("unsupported type, it should be TG_INT8_LutOp/TG_BF16_LutOp");
  }

  Tensor* in_tensor = net_graph_->get_tensor_by_id(in_tensors[0]);
  net_graph_->get_tensor_dim(in_tensors[0], bottom_dim);
  bottom_dim[0] = in_tensor->n_slice;
  bottom_dim[2] = in_tensor->h_slice;

  std::string name = mix_op->name();
  uint32_t la_input = net_graph_->get_tensor_local_offset(in_tensors[0]);
  uint32_t la_output = net_graph_->get_tensor_local_offset(out_tensors[0]);
  uint32_t la_y_table = net_graph_->get_tensor_local_offset(in_tensors[1]);
  uint32_t la_slope_lut = 0, la_working = 0;
  if (!is_int8) {
    // order by \ImLayer.cpp
    la_slope_lut = net_graph_->get_tensor_local_offset(in_tensors[2]);

    mem_buffer_key_t key = {timestep_idx, im_layer->imm_tensors[0].get()->id(), true};
    const mem_buffer_value_t* imm = time_step->get_mem_buffer_value(&key);
    la_working = imm->local_mem_offset;
  }

  // attrs
  attrs.push_back(builder_.getNamedAttr("name",
                           builder_.getStringAttr(name)));
  attrs.push_back(builder_.getNamedAttr("la_input",
                           builder_.getI32IntegerAttr(la_input)));
  attrs.push_back(builder_.getNamedAttr("la_output",
                           builder_.getI32IntegerAttr(la_output)));
  attrs.push_back(builder_.getNamedAttr("la_slope_lut",
                           builder_.getI32IntegerAttr(la_slope_lut)));
  attrs.push_back(builder_.getNamedAttr("la_y_table",
                           builder_.getI32IntegerAttr(la_y_table)));
  attrs.push_back(builder_.getNamedAttr("la_working",
                           builder_.getI32IntegerAttr(la_working)));

  attrs.push_back(builder_.getNamedAttr("max_range",
                           builder_.getF32FloatAttr(table_thresh_max)));
  attrs.push_back(builder_.getNamedAttr("min_range",
                           builder_.getF32FloatAttr(table_thresh_min)));
  attrs.push_back(builder_.getNamedAttr("added_offset",
                           builder_.getBoolAttr(added_offset)));

  // setup input/output type
  RankedTensorType input_type = RankedTensorType::get(
                          {bottom_dim[0], bottom_dim[1],
                           bottom_dim[2], bottom_dim[3]},
                           old_input_type.getElementType());

  RankedTensorType output_type = RankedTensorType::get(
                          {bottom_dim[0], bottom_dim[1],
                           bottom_dim[2], bottom_dim[3]},
                           old_input_type.getElementType());
  // setup operands
  std::vector<Value> operands;
  Operation * input_op =
    get_op_from_name(mix_op->bottom_name(0)).getDefiningOp();
  input_op->getResult(0).setType(input_type);
  operands.push_back(input_op->getResult(0));

  for (int32_t i = 0; i < lut_nr ; i++) {
    // + 1 means shift after 0(input)
    input_op = get_op_from_name(mix_op->bottom_name(i + 1)).getDefiningOp();
    if (!is_int8) {
      auto shape = input_op->getResult(0).getType().cast<TensorType>().getShape();
      auto type = RankedTensorType::get(shape, bf16Type);
      input_op->getResult(0).setType(type);
    }
    operands.push_back(input_op->getResult(0));
  }

  if (is_int8) {
    auto NoneOp = OpBuilder(get_start_op()).create<tpu::NoneOp>(builder_.getUnknownLoc(),
                  builder_.getNoneType());
    operands.push_back(NoneOp.getResult());
    auto op = OpBuilder(get_start_op()).create<tpu::TL_LG_INT8_LutOp>(
                        get_start_op()->getLoc(), output_type,
                        ArrayRef<Value>{operands},
                        ArrayRef<NamedAttribute>{attrs});

    add_opd_to_list(mix_op->name(), op.getResult(), true);
  } else {
    auto op = OpBuilder(get_start_op()).create<tpu::TL_LG_BF16_LutOp>(
                    get_start_op()->getLoc(), output_type,
                    ArrayRef<Value>{operands},
                    ArrayRef<NamedAttribute>{attrs});

    add_opd_to_list(mix_op->name(), op.getResult(), true);
  }
}

void MixNet::_add_tl_quant_op(MixOp *mix_op, const std::vector<int> &in_tensors,
                              const std::vector<int> &out_tensors,
                              net_timestep *time_step, int timestep_idx,
                              bool is_h_split) {
  int bottom_dim[4];
  float const_scale = 1.0;
  int la_working = 0;
  bool bExtraInput = false;
  StringRef from, to;
  const ImLayer *im_layer = net_graph_->get_layer_by_id(mix_op->get_layer_id());

  // it MUST quant op
  RankedTensorType old_input_type, old_output_type;

  auto quantOp = cast<tpu::TG_QuantOp>(im_layer->op());
  old_input_type = quantOp.getOperand().getType().cast<RankedTensorType>();
  old_output_type = quantOp.getResult().getType().cast<RankedTensorType>();
  from = quantOp.from();
  to = quantOp.to();

  Tensor *in_tensor = net_graph_->get_tensor_by_id(in_tensors[0]);
  net_graph_->get_tensor_dim(in_tensors[0], bottom_dim);
  bottom_dim[0] = in_tensor->n_slice;
  bottom_dim[2] = in_tensor->h_slice;

  std::string name = mix_op->name();
  uint32_t la_input = net_graph_->get_tensor_local_offset(in_tensors[0]);
  uint32_t la_output = net_graph_->get_tensor_local_offset(out_tensors[0]);

  if (((from == "INT8" || from == "UINT8") && to == "BF16") ||
      (from == "BF16" && to == "INT8")) {
    // quant
    const_scale = quantOp.scale().convertToFloat();
    if (from == "BF16" && to == "INT8") {
      if (im_layer->imm_tensors.size()) {
        mem_buffer_key_t key = {timestep_idx, im_layer->imm_tensors[0].get()->id(), true};
        const mem_buffer_value_t* imm = time_step->get_mem_buffer_value(&key);
        la_working = (imm->local_mem_offset);
        bExtraInput = true;
      }
    }
  }

  // attrs
  Builder builder_(context_);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder_.getNamedAttr("name", builder_.getStringAttr(name)));
  attrs.push_back(
      builder_.getNamedAttr("la_input", builder_.getI32IntegerAttr(la_input)));
  attrs.push_back(builder_.getNamedAttr("la_output",
                                        builder_.getI32IntegerAttr(la_output)));
  attrs.push_back(builder_.getNamedAttr("la_working",
                                        builder_.getI32IntegerAttr(la_working)));
  attrs.push_back(builder_.getNamedAttr("from", builder_.getStringAttr(from)));
  attrs.push_back(builder_.getNamedAttr("to", builder_.getStringAttr(to)));
  attrs.push_back(builder_.getNamedAttr("const_scale",
                                        builder_.getF32FloatAttr(const_scale)));
  attrs.push_back(builder_.getNamedAttr("bExtraInput",
                                        builder_.getBoolAttr(bExtraInput)));

  // setup input/output type
  RankedTensorType input_type = RankedTensorType::get(
      {bottom_dim[0], bottom_dim[1], bottom_dim[2], bottom_dim[3]},
      old_input_type.getElementType());

  RankedTensorType output_type = RankedTensorType::get(
      {bottom_dim[0], bottom_dim[1], bottom_dim[2], bottom_dim[3]},
      old_output_type.getElementType());

  // setup operands
  std::vector<Value> operands;

  // only one input
  Operation *input_op =
      get_op_from_name(mix_op->bottom_name(0)).getDefiningOp();
  input_op->getResult(0).setType(input_type);
  operands.push_back(input_op->getResult(0));

  auto op = OpBuilder(get_start_op()).create<tpu::TL_LG_QuantOp>(
              get_start_op()->getLoc(), output_type,
              ArrayRef<Value>{operands},
              ArrayRef<NamedAttribute>{attrs});

  add_opd_to_list(mix_op->name(), op.getResult(), true);
}

void MixNet::_add_tl_lrn_op(MixOp * mix_op,
                              const std::vector<int>& in_tensors,
                              const std::vector<int>& out_tensors,
                              net_timestep* time_step,
                              int timestep_idx, bool is_h_split) {
  int bottom_dim[4];
  const ImLayer* im_layer =
      net_graph_->get_layer_by_id(mix_op->get_layer_id());
  Operation* op = im_layer->op();
  auto op_input_type =
    op->getOperand(0).getType().cast<RankedTensorType>();

  Tensor* in_tensor = net_graph_->get_tensor_by_id(in_tensors[0]);
  net_graph_->get_tensor_dim(in_tensors[0], bottom_dim);
  bottom_dim[0] = in_tensor->n_slice;
  bottom_dim[2] = in_tensor->h_slice;

  mem_buffer_key_t key =
        {timestep_idx, im_layer->imm_tensors[0].get()->id(), true};
  const mem_buffer_value_t* imm = time_step->get_mem_buffer_value(&key);

  std::string name = mix_op->name();
  uint32_t la_input = net_graph_->get_tensor_local_offset(in_tensors[0]);
  uint32_t la_output = net_graph_->get_tensor_local_offset(out_tensors[0]);
  uint32_t la_sqrt = net_graph_->get_tensor_local_offset(in_tensors[1]);
  uint32_t la_power = net_graph_->get_tensor_local_offset(in_tensors[2]);
  uint32_t la_working = imm->local_mem_offset;

  uint32_t local_size;
  int sum_rshift, lrn_rshift, quant_data0, quant_data1;
  float alpha, k;
  getLrnParam(op, local_size, sum_rshift, lrn_rshift,
              quant_data0, quant_data1, alpha, k);
  // attrs
  Builder builder_(context_);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder_.getNamedAttr("name",
                           builder_.getStringAttr(name)));
  attrs.push_back(builder_.getNamedAttr("la_input",
                           builder_.getI32IntegerAttr(la_input)));
  attrs.push_back(builder_.getNamedAttr("la_output",
                           builder_.getI32IntegerAttr(la_output)));
  attrs.push_back(builder_.getNamedAttr("la_sqrt",
                           builder_.getI32IntegerAttr(la_sqrt)));
  attrs.push_back(builder_.getNamedAttr("la_power",
                           builder_.getI32IntegerAttr(la_power)));
  attrs.push_back(builder_.getNamedAttr("la_working",
                           builder_.getI32IntegerAttr(la_working)));
  // attrs.push_back(builder_.getNamedAttr("norm_region",
  //                          builder_.getI32IntegerAttr(norm_region)));
  attrs.push_back(builder_.getNamedAttr("local_size",
                          builder_.getI32IntegerAttr(local_size)));
  attrs.push_back(builder_.getNamedAttr("sum_rshift",
                          builder_.getI32IntegerAttr(sum_rshift)));
  attrs.push_back(builder_.getNamedAttr("lrn_rshift",
                          builder_.getI32IntegerAttr(lrn_rshift)));
  attrs.push_back(builder_.getNamedAttr("quant_data0",
                          builder_.getI32IntegerAttr(quant_data0)));
  attrs.push_back(builder_.getNamedAttr("quant_data1",
                          builder_.getI32IntegerAttr(quant_data1)));
  attrs.push_back(builder_.getNamedAttr("alpha",
                          builder_.getF32FloatAttr(alpha)));
  attrs.push_back(builder_.getNamedAttr("k",
                          builder_.getF32FloatAttr(k)));

  // setup input/output type
  RankedTensorType input_type = RankedTensorType::get(
                          {bottom_dim[0], bottom_dim[1],
                           bottom_dim[2], bottom_dim[3]},
                           op_input_type.getElementType());

  RankedTensorType output_type = RankedTensorType::get(
                          {bottom_dim[0], bottom_dim[1],
                           bottom_dim[2], bottom_dim[3]},
                           op_input_type.getElementType());

  // setup operands
  std::vector<Value> operands;

  for( uint32_t i = 0; i < 3; i++) {
    Operation * input_op =
      get_op_from_name(mix_op->bottom_name(i)).getDefiningOp();
    if ( i == 0)
      input_op->getResult(0).setType(input_type);
    operands.push_back(input_op->getResult(0));
  }

  if (isa<tpu::TG_INT8_LrnOp>(op)) {
    auto op = OpBuilder(get_start_op()).create<tpu::TL_LG_INT8_LrnOp>(
                        get_start_op()->getLoc(), output_type,
                        ArrayRef<Value>{operands},
                        ArrayRef<NamedAttribute>{attrs});

    add_opd_to_list(mix_op->name(), op.getResult(), true);
  } else if (isa<tpu::TG_BF16_LrnOp>(op)) {
    auto op = OpBuilder(get_start_op()).create<tpu::TL_LG_BF16_LrnOp>(
                    get_start_op()->getLoc(), output_type,
                    ArrayRef<Value>{operands},
                    ArrayRef<NamedAttribute>{attrs});

    add_opd_to_list(mix_op->name(), op.getResult(), true);
  }

}


void MixNet::add_transport_op(int group_idx,
                              const TENSOR_STEP& tensor,
                              net_timestep* time_step,
                              int timestep_idx) {
  int tensor_id = tensor.first;
  if (tensor.second == TIMESTEP_LOAD) {
    _add_load_op(group_idx, tensor_id, time_step, timestep_idx);
  } else if (tensor.second == TIMESTEP_STORE) {
    _add_store_op(group_idx, tensor_id, time_step, timestep_idx);
  } else {
    assert(false && "not support now");
    exit(-1);
  }
}

void MixNet::_add_load_op(int group_idx,
                          int tensor_id,
                          net_timestep* time_step,
                          int timestep_idx) {
  int tensor_dim[4];
  int local_shape[4];
  uint64_t laddr = 0;
  bool aligned = false;
  bool transpose = false;
  int32_t offset = 0;
  std::string tensor_type_str = "CONV_COEFF";
  int dtype = NEURON;
  std::string name;
  std::vector<NamedAttribute> attrs;
  Builder builder_(context_);

  const tensor_type_t tensor_type = net_graph_->get_tensor_type(tensor_id);
  Tensor* tensor = net_graph_->get_tensor_by_id(tensor_id);
  std::string storage = tensor->storage();
  net_graph_->get_tensor_dim(tensor_id, tensor_dim);

  name = tensor->name();
  Value src_opd = weightFileOp_->getResult(0);

  if (tensor_type == TENSOR_COEFF || tensor_type == TENSOR_COEFF_LUT) {
    laddr = net_graph_->get_tensor_local_offset(tensor_id);

    if (tensor_type == TENSOR_COEFF_LUT) {
      // lut case, no need to reshape
      local_shape[0] = (tensor_dim[0]);
      local_shape[1] = (tensor_dim[1]);
      local_shape[2] = (tensor_dim[2]);
      local_shape[3] = (tensor_dim[3]);
    }
    else {
      // to match mlir requirement for conv weight, shape is
      // (oc, ic, kh, kw)
      local_shape[0] = tensor_dim[1];
      local_shape[1] = tensor_dim[0];
      local_shape[2] = tensor_dim[2];
      local_shape[3] = tensor_dim[3];
    }

    aligned = (false);
    transpose = (false);
    tensor_type_str = "CONV_COEFF";
    if (tensor_type == TENSOR_COEFF_LUT) {
      tensor_type_str = "LUT_COEFF";
    }
    dtype = COEFF;
    attrs.push_back(builder_.getNamedAttr("storage", builder_.getStringAttr(storage)));
  } else if (tensor_type == TENSOR_BIAS) {
    laddr = net_graph_->get_tensor_local_offset(tensor_id);

    local_shape[0] = (tensor_dim[0]);
    local_shape[1] = (tensor_dim[1]);
    local_shape[2] = (tensor_dim[2]);
    local_shape[3] = (tensor_dim[3]);

    aligned = (false);
    transpose = (false);
    tensor_type_str = "BIAS";
    dtype = COEFF;
    attrs.push_back(builder_.getNamedAttr("storage", builder_.getStringAttr(storage)));
  } else if (tensor_type == TENSOR_DEPTHCONV_OPD1) {
    laddr = net_graph_->get_tensor_local_offset(tensor_id);

    local_shape[0] = (tensor_dim[0]);
    local_shape[1] = (tensor_dim[1]);
    local_shape[2] = (tensor_dim[2]);
    local_shape[3] = (tensor_dim[3]);

    aligned = (true);
    transpose = (false);
    tensor_type_str = "CONV_DEPTH_OPD1";
    dtype = COEFF;
    attrs.push_back(builder_.getNamedAttr("storage", builder_.getStringAttr(storage)));
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
    offset = (n_idx * tensor_dim[1] * tensor_dim[2] * tensor_dim[3] + h_idx * tensor_dim[3]) *
              tensor->unit_size();

    local_shape[0] = (n_slice);
    local_shape[1] = (tensor_dim[1]);
    local_shape[2] = (h_slice);
    local_shape[3] = (tensor_dim[3]);

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
    attrs.push_back(builder_.getNamedAttr("offset", builder_.getI64IntegerAttr(offset)));
  }

  // build tl_load instruction
  attrs.push_back(builder_.getNamedAttr("name", builder_.getStringAttr(name)));
  attrs.push_back(builder_.getNamedAttr("laddr", builder_.getI64IntegerAttr(laddr)));
  attrs.push_back(builder_.getNamedAttr("align", builder_.getBoolAttr(aligned)));
  attrs.push_back(builder_.getNamedAttr("transpose", builder_.getBoolAttr(transpose)));
  attrs.push_back(builder_.getNamedAttr("tensor_type", builder_.getStringAttr(tensor_type_str)));

  // setup input operation
  std::vector<Value> operands;
  operands.push_back(src_opd);
  RankedTensorType output_type = RankedTensorType::get(
                          {local_shape[0], local_shape[1],
                           local_shape[2], local_shape[3]},
                           getStorageType(context_, storage));


  // build tl_load operation
  if (dtype == COEFF) {
    auto op = OpBuilder(get_start_op()).create<tpu::TL_LG_LoadCoeffOp>(get_start_op()->getLoc(),
            output_type, ArrayRef<Value>{operands}, ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(name, op.getResult(), true);
  } else {
    auto op = OpBuilder(get_start_op()).create<tpu::TL_LG_LoadNeuronOp>(get_start_op()->getLoc(),
              output_type, ArrayRef<Value>{operands}, ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(name, op.getResult(), true);
  }
}

// do not support concat optimization
void MixNet::_add_store_op(int group_idx, int tensor_id, net_timestep * time_step, int timestep_idx) {
  int tensor_dim[4];
  int global_shape[4];
  Tensor* tensor = net_graph_->get_tensor_by_id(tensor_id);
  uint32_t laddr = net_graph_->get_tensor_local_offset(tensor_id);
  bool aligned = true;
  bool transpose = false;

  net_graph_->get_tensor_dim(tensor_id, tensor_dim);

  int n_idx = tensor->n_idx;
  int h_idx = tensor->h_idx;
  int h_slice = tensor->h_slice;
  int h_end = h_idx + h_slice;
  h_idx = h_idx > 0 ? h_idx : 0;
  h_slice = h_end > tensor_dim[2] ? (tensor_dim[2] - h_idx) : (h_end - h_idx);

  std::string tensor_name = tensor->name();
  tensor_name += _get_postfix_name(tensor->get_group_id(), tensor->get_n_loop(), tensor->get_h_loop());
  Value src_opd = get_op_from_name(tensor_name);
  std::string store_op_name = tensor_name + "_st";

  global_shape[0] = (tensor_dim[0]);
  global_shape[1] = (tensor_dim[1]);
  global_shape[2] = (tensor_dim[2]);
  global_shape[3] = (tensor_dim[3]);

  int32_t offset = (n_idx * tensor_dim[1] * tensor_dim[2] * tensor_dim[3] + h_idx * tensor_dim[3]) *
                   tensor->unit_size();

  std::vector<NamedAttribute> attrs;
  Builder builder_(context_);
  attrs.push_back(builder_.getNamedAttr("name", builder_.getStringAttr(store_op_name)));
  attrs.push_back(builder_.getNamedAttr("offset", builder_.getI64IntegerAttr(offset)));
  attrs.push_back(builder_.getNamedAttr("laddr", builder_.getI64IntegerAttr(laddr)));
  attrs.push_back(builder_.getNamedAttr("align", builder_.getBoolAttr(aligned)));
  attrs.push_back(builder_.getNamedAttr("transpose", builder_.getBoolAttr(transpose)));

  // setup input operation
  std::vector<Value> operands;
  operands.push_back(src_opd);
  Type _input_type = src_opd.getType().cast<RankedTensorType>().getElementType();
  RankedTensorType output_type = RankedTensorType::get(
                          {global_shape[0], global_shape[1],
                           global_shape[2], global_shape[3]},
                           _input_type);


  // build tl_load operation
  auto op = OpBuilder(get_start_op()).create<tpu::TL_LG_StoreOp>(get_start_op()->getLoc(),
          output_type, ArrayRef<Value>{operands}, ArrayRef<NamedAttribute>{attrs});
  add_opd_to_list(store_op_name, op.getResult(), true);

}

void MixNet::_add_tl_upsample_op(MixOp * mix_op,
                                  const std::vector<int>& in_tensors,
                                  const std::vector<int>& out_tensors,
                                  net_timestep* time_step,
                                  int timestep_idx,
                                  bool is_h_split) {
  const ImLayer* im_layer = net_graph_->get_layer_by_id(mix_op->get_layer_id());
  const Tensor* in_tensor = net_graph_->get_tensor_by_id(in_tensors[0]);
  const Tensor* out_tensor = net_graph_->get_tensor_by_id(out_tensors[0]);
  Operation *op = im_layer->op();
  auto old_input_type = op->getOperand(0).getType().cast<RankedTensorType>();
  int scale_h = 1;
  int scale_w = 1;
  getUpsampleParam(op, scale_h, scale_w);

  int bottom_dim[4];
  int top_dim[4];

  net_graph_->get_tensor_dim(in_tensors[0], bottom_dim);
  net_graph_->get_tensor_dim(out_tensors[0], top_dim);
  bottom_dim[0] = in_tensor->n_slice;
  bottom_dim[2] = in_tensor->h_slice;

  top_dim[0] = out_tensor->n_slice;
  top_dim[2] = out_tensor->h_slice;

  if (is_h_split) {
    int real_h_slice = 0;
    int real_h_idx = 0;

    // bottom
    if (in_tensor->h_idx > 0) {
      real_h_idx = in_tensor->h_idx;
    } else {
      real_h_idx = 0;
    }
    int h_end = in_tensor->h_idx + in_tensor->h_slice;
    if (h_end > in_tensor->h()) {
      real_h_slice = in_tensor->h() - real_h_idx;
    } else {
      real_h_slice = h_end - real_h_idx;
    }
    bottom_dim[2] = real_h_slice;
    top_dim[2] = bottom_dim[2] * scale_h;
  }

  std::string name = mix_op->name();
  uint32_t la_input = net_graph_->get_tensor_local_offset(in_tensors[0]);
  uint32_t la_output = net_graph_->get_tensor_local_offset(out_tensors[0]);

  Builder builder_(context_);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder_.getNamedAttr("name",
                           builder_.getStringAttr(name)));
  attrs.push_back(builder_.getNamedAttr("la_input",
                           builder_.getI32IntegerAttr(la_input)));
  attrs.push_back(builder_.getNamedAttr("la_output",
                           builder_.getI32IntegerAttr(la_output)));
  attrs.push_back(builder_.getNamedAttr("scale_h",
                           builder_.getI32IntegerAttr(scale_h)));
  attrs.push_back(builder_.getNamedAttr("scale_w",
                           builder_.getI32IntegerAttr(scale_w)));



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
  std::vector<Value> operands;
  Operation * input_op =
                    get_op_from_name(mix_op->bottom_name(0)).getDefiningOp();
  input_op->getResult(0).setType(input_type);
  operands.push_back(input_op->getResult(0));

  // build tl_upsample operation
  if (isa<tpu::TG_INT8_UpsampleOp>(op)) {
    auto tl_op = OpBuilder(get_start_op()).create<tpu::TL_LG_INT8_UpsampleOp>(
                        get_start_op()->getLoc(), output_type,
                        ArrayRef<Value>{operands},
                        ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(mix_op->name(), tl_op.getResult(), true);
  } else if (isa<tpu::TG_BF16_UpsampleOp>(op)) {
    auto tl_op = OpBuilder(get_start_op()).create<tpu::TL_LG_BF16_UpsampleOp>(
                        get_start_op()->getLoc(), output_type,
                        ArrayRef<Value>{operands},
                        ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(mix_op->name(), tl_op.getResult(), true);
  }
}

static void add_leaky_attrs(Builder &builder,
                            Operation * op,
                            std::vector<NamedAttribute> & attrs) {
  if (auto leaky_op = dyn_cast<tpu::TG_INT8_LeakyReluOp>(op)) {
    if (leaky_op.rshift_pos().hasValue()) {
      attrs.push_back(builder.getNamedAttr("rshift_pos",
                                            leaky_op.rshift_posAttr()));
      attrs.push_back(builder.getNamedAttr("m_i8_pos",
                                            leaky_op.m_i8_posAttr()));
    }
    attrs.push_back(builder.getNamedAttr("rshift_neg",
                                          leaky_op.rshift_negAttr()));
    attrs.push_back(builder.getNamedAttr("m_i8_neg",
                                          leaky_op.m_i8_negAttr()));
    attrs.push_back(builder.getNamedAttr("negative_slope",
                                          leaky_op.negative_slopeAttr()));
  } else if (auto leaky_op = dyn_cast<tpu::TG_BF16_LeakyReluOp>(op)) {
    attrs.push_back(builder.getNamedAttr("negative_slope",
                                          leaky_op.negative_slopeAttr()));
  }
}

void MixNet::_add_tl_leaky_relu_op(MixOp * mix_op,
                                   const std::vector<int>& in_tensors,
                                   const std::vector<int>& out_tensors,
                                   net_timestep* time_step,
                                   int timestep_idx,
                                   bool is_h_split) {
  const ImLayer* im_layer = net_graph_->get_layer_by_id(mix_op->get_layer_id());
  const Tensor* in_tensor = net_graph_->get_tensor_by_id(in_tensors[0]);
  const Tensor* out_tensor = net_graph_->get_tensor_by_id(out_tensors[0]);
  Operation* op = im_layer->op();
  auto old_input_type = op->getOperand(0).getType().cast<RankedTensorType>();
  Builder builder_(context_);
  std::vector<NamedAttribute> attrs;
  int bottom_dim[4];
  int top_dim[4];

  net_graph_->get_tensor_dim(in_tensors[0], bottom_dim);
  net_graph_->get_tensor_dim(out_tensors[0], top_dim);
  bottom_dim[0] = in_tensor->n_slice;
  bottom_dim[2] = in_tensor->h_slice;

  top_dim[0] = out_tensor->n_slice;
  top_dim[2] = out_tensor->h_slice;

  std::string name = mix_op->name();
  uint32_t la_input = net_graph_->get_tensor_local_offset(in_tensors[0]);
  uint32_t la_output = net_graph_->get_tensor_local_offset(out_tensors[0]);

  attrs.push_back(builder_.getNamedAttr("name",
                           builder_.getStringAttr(name)));
  attrs.push_back(builder_.getNamedAttr("la_input",
                           builder_.getI32IntegerAttr(la_input)));
  attrs.push_back(builder_.getNamedAttr("la_output",
                           builder_.getI32IntegerAttr(la_output)));

  add_leaky_attrs(builder_, op, attrs);

  // setup input/output type
  RankedTensorType input_type = RankedTensorType::get(
                          { bottom_dim[0], bottom_dim[1],
                            bottom_dim[2], bottom_dim[3]},
                            old_input_type.getElementType());

  RankedTensorType output_type = RankedTensorType::get(
                          { top_dim[0], top_dim[1],
                            top_dim[2], top_dim[3]},
                            old_input_type.getElementType());
  // setup input operation
  std::vector<Value> operands;
  Operation * input_op =
                    get_op_from_name(mix_op->bottom_name(0)).getDefiningOp();
  input_op->getResult(0).setType(input_type);
  operands.push_back(input_op->getResult(0));

  // build tl_leaky operation
  if (isa<tpu::TG_INT8_LeakyReluOp>(op)) {
    auto tl_op = OpBuilder(get_start_op()).create<tpu::TL_LG_INT8_LeakyReluOp>(
                        get_start_op()->getLoc(), output_type,
                        ArrayRef<Value>{operands},
                        ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(mix_op->name(), tl_op.getResult(), true);
  } else if (isa<tpu::TG_BF16_LeakyReluOp>(op)) {
    auto tl_op = OpBuilder(get_start_op()).create<tpu::TL_LG_BF16_LeakyReluOp>(
                        get_start_op()->getLoc(), output_type,
                        ArrayRef<Value>{operands},
                        ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(mix_op->name(), tl_op.getResult(), true);
  }
}

void MixNet::_add_tl_prelu_op(MixOp * mix_op,
                              const std::vector<int>& in_tensors,
                              const std::vector<int>& out_tensors,
                              net_timestep* time_step,
                              int timestep_idx,
                              bool is_h_split) {
  const ImLayer* im_layer = net_graph_->get_layer_by_id(mix_op->get_layer_id());
  const Tensor* in_tensor = net_graph_->get_tensor_by_id(in_tensors[0]);
  const Tensor* out_tensor = net_graph_->get_tensor_by_id(out_tensors[0]);
  Operation *op = im_layer->op();
  auto opd0 = op->getOperand(0);
  auto old_input_type = opd0.getType().cast<RankedTensorType>();
  Builder builder_(context_);
  std::vector<NamedAttribute> attrs;

  int bottom_dim[4];
  int top_dim[4];

  net_graph_->get_tensor_dim(in_tensors[0], bottom_dim);
  net_graph_->get_tensor_dim(out_tensors[0], top_dim);
  bottom_dim[0] = in_tensor->n_slice;
  bottom_dim[2] = in_tensor->h_slice;

  top_dim[0] = out_tensor->n_slice;
  top_dim[2] = out_tensor->h_slice;

  std::string name = mix_op->name();
  uint32_t la_input = net_graph_->get_tensor_local_offset(in_tensors[0]);
  uint32_t la_slope = net_graph_->get_tensor_local_offset(in_tensors[1]);
  uint32_t la_output = net_graph_->get_tensor_local_offset(out_tensors[0]);

  attrs.push_back(builder_.getNamedAttr("name",
                           builder_.getStringAttr(name)));
  attrs.push_back(builder_.getNamedAttr("la_input",
                           builder_.getI32IntegerAttr(la_input)));
  attrs.push_back(builder_.getNamedAttr("la_output",
                           builder_.getI32IntegerAttr(la_output)));
  attrs.push_back(builder_.getNamedAttr("la_slope",
                           builder_.getI32IntegerAttr(la_slope)));

  if (auto prelu_op = dyn_cast<tpu::TG_INT8_PReluOp>(op)) {
    if (prelu_op.rshift_pos().hasValue()) {
      attrs.push_back(builder_.getNamedAttr("r_i8_pos", prelu_op.rshift_posAttr()));
      attrs.push_back(builder_.getNamedAttr("m_i8_pos", prelu_op.m_i8_posAttr()));
    }
    attrs.push_back(builder_.getNamedAttr("r_i8_neg", prelu_op.rshift_negAttr()));
  }

  // setup input/output type
  RankedTensorType input_type = RankedTensorType::get(
                          { bottom_dim[0], bottom_dim[1],
                            bottom_dim[2], bottom_dim[3]},
                            old_input_type.getElementType());

  RankedTensorType output_type = RankedTensorType::get(
                          { top_dim[0], top_dim[1],
                            top_dim[2], top_dim[3]},
                            old_input_type.getElementType());
  // setup input operation
  std::vector<Value> operands;
  Operation * input_op =
                    get_op_from_name(mix_op->bottom_name(0)).getDefiningOp();
  input_op->getResult(0).setType(input_type);
  operands.push_back(input_op->getResult(0));

  Operation * slope_op =
                    get_op_from_name(mix_op->bottom_name(1)).getDefiningOp();
  operands.push_back(slope_op->getResult(0));

  // build tl_prelu operation
  if (isa<tpu::TG_INT8_PReluOp>(op)) {
    auto tl_op = OpBuilder(get_start_op()).create<tpu::TL_LG_INT8_PReluOp>(
                        get_start_op()->getLoc(), output_type,
                        ArrayRef<Value>{operands},
                        ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(mix_op->name(), tl_op.getResult(), true);
  } else if (isa<tpu::TG_BF16_PReluOp>(op)) {
     auto tl_op = OpBuilder(get_start_op()).create<tpu::TL_LG_BF16_PReluOp>(
                        get_start_op()->getLoc(), output_type,
                        ArrayRef<Value>{operands},
                        ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(mix_op->name(), tl_op.getResult(), true);
  }
}

void MixNet::_add_tl_concat_op(MixOp * mix_op,
                               const std::vector<int>& in_tensors,
                               const std::vector<int>& out_tensors,
                               net_timestep* time_step,
                               int timestep_idx, bool is_h_split) {
  int bottom_dim[4];
  int top_dim[4];
  const ImLayer* im_layer =
      net_graph_->get_layer_by_id(mix_op->get_layer_id());
  Operation *op = im_layer->op();
  auto old_input_type =
    op->getOperand(0).getType().cast<RankedTensorType>();
  int op_num = op->getNumOperands();

  std::vector<int32_t> la_input(op_num);
  std::vector<int32_t> input_dim_c(op_num);
  for (int i = 0; i < op_num; i++) {
    net_graph_->get_tensor_dim(in_tensors[i], bottom_dim);
    input_dim_c[i] = bottom_dim[1];
    la_input[i] = net_graph_->get_tensor_local_offset(in_tensors[i]);
  }

  Tensor* in_tensor = net_graph_->get_tensor_by_id(in_tensors[0]);
  net_graph_->get_tensor_dim(in_tensors[0], bottom_dim);
  Tensor* out_tensor = net_graph_->get_tensor_by_id(out_tensors[0]);
  net_graph_->get_tensor_dim(out_tensors[0], top_dim);
  bottom_dim[0] = in_tensor->n_slice;
  bottom_dim[2] = in_tensor->h_slice;
  top_dim[0] = out_tensor->n_slice;
  top_dim[2] = out_tensor->h_slice;

  std::string name = mix_op->name();
  uint32_t la_output = net_graph_->get_tensor_local_offset(out_tensors[0]);

  // attrs
  int axis = 0;
  bool do_relu = false;
  getConcatParam(op, axis, do_relu);

  Builder builder_(context_);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder_.getNamedAttr("name",
                           builder_.getStringAttr(name)));
  attrs.push_back(builder_.getNamedAttr("la_input",
                           builder_.getI32ArrayAttr(ArrayRef<int32_t>({la_input}))));
  attrs.push_back(builder_.getNamedAttr("la_output",
                           builder_.getI32IntegerAttr(la_output)));
  attrs.push_back(builder_.getNamedAttr("la_working",
                           builder_.getI32IntegerAttr(0)));
  attrs.push_back(builder_.getNamedAttr("do_relu",
                           builder_.getBoolAttr(do_relu)));
  attrs.push_back(builder_.getNamedAttr("axis",
                           builder_.getI32IntegerAttr(axis)));
  // std::set quant value
  if (auto tmp = dyn_cast<tpu::TG_INT8_ConcatOp>(op)) {
    if (tmp.rshift().hasValue()) {
      attrs.push_back(builder_.getNamedAttr("r_i8", tmp.rshiftAttr()));
    }
    if (tmp.m_i8_inputs().hasValue()) {
      attrs.push_back(builder_.getNamedAttr("m_i8", tmp.m_i8_inputsAttr()));
    }
  }

  // setup operands
  std::vector<Value> operands;
  for (int32_t i = 0; i < op_num; i++) {
    Operation * input_op =
      get_op_from_name(mix_op->bottom_name(i)).getDefiningOp();

    RankedTensorType input_type = RankedTensorType::get(
                          { bottom_dim[0], input_dim_c[i],
                            bottom_dim[2], bottom_dim[3]},
                            old_input_type.getElementType());
    input_op->getResult(0).setType(input_type);
    operands.push_back(input_op->getResult(0));
  }

  RankedTensorType output_type = RankedTensorType::get(
                          { top_dim[0], top_dim[1],
                            top_dim[2], top_dim[3]},
                            old_input_type.getElementType());

  if (isa<tpu::TG_INT8_ConcatOp>(op)) {
    auto tl_op = OpBuilder(get_start_op()).create<tpu::TL_LG_INT8_ConcatOp>(
                        get_start_op()->getLoc(), output_type,
                        ArrayRef<Value>{operands},
                        ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(mix_op->name(), tl_op.getResult(), true);
  } else if (isa<tpu::TG_BF16_ConcatOp>(op)) {
    auto tl_op = OpBuilder(get_start_op()).create<tpu::TL_LG_BF16_ConcatOp>(
                        get_start_op()->getLoc(), output_type,
                        ArrayRef<Value>{operands},
                        ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(mix_op->name(), tl_op.getResult(), true);
  }
}

void MixNet::_add_tl_pad_op(MixOp * mix_op,
                            const std::vector<int>& in_tensors,
                            const std::vector<int>& out_tensors,
                            net_timestep* time_step,
                            int timestep_idx,
                            bool is_h_split) {
  const ImLayer* im_layer = net_graph_->get_layer_by_id(mix_op->get_layer_id());
  const Tensor* in_tensor = net_graph_->get_tensor_by_id(in_tensors[0]);
  const Tensor* out_tensor = net_graph_->get_tensor_by_id(out_tensors[0]);
  Operation *op = im_layer->op();
  auto opd0 = op->getOperand(0);
  auto old_input_type = opd0.getType().cast<RankedTensorType>();

  std::vector<int32_t> pads;
  float const_val;
  if (auto pad_op = dyn_cast<tpu::TG_INT8_PadOp>(op)) {
    const_val = pad_op.const_val().convertToFloat();
    arrayAttrToVector(pad_op.pads().getValue(), pads);
  } else if (auto pad_op = dyn_cast<tpu::TG_BF16_PadOp>(op)) {
    const_val = pad_op.const_val().convertToFloat();
    arrayAttrToVector(pad_op.pads().getValue(), pads);
  }

  int bottom_dim[4];
  int top_dim[4];
  net_graph_->get_tensor_dim(in_tensors[0], bottom_dim);
  net_graph_->get_tensor_dim(out_tensors[0], top_dim);
  bottom_dim[0] = in_tensor->n_slice;
  bottom_dim[2] = in_tensor->h_slice;

  top_dim[0] = out_tensor->n_slice;
  top_dim[2] = out_tensor->h_slice;

  if (is_h_split) {
    int real_h_slice = 0;
    int real_h_idx = 0;

    // bottom
    if (in_tensor->h_idx > 0) {
      real_h_idx = in_tensor->h_idx;
    } else {
      real_h_idx = 0;
    }
    int h_end = in_tensor->h_idx + in_tensor->h_slice;
    if (h_end >= in_tensor->h()) {
      real_h_slice = in_tensor->h() - real_h_idx;
      pads[2] = 0; // pad_top = 0;
      pads[6] = h_end - in_tensor->h(); // pad_bottom
    } else {
      real_h_slice = h_end - real_h_idx;
      // slice is not enough, need pads
      int real_out_h_slice = (out_tensor->h_idx < 0) ?
                        (out_tensor->h_slice + out_tensor->h_idx) :
                        out_tensor->h_slice;
      pads[2] = real_out_h_slice - real_h_slice;
      pads[6] = 0;
    }
    if (in_tensor->h_idx < 0)
      pads[2] = -in_tensor->h_idx;
    bottom_dim[2] = real_h_slice;
  }

  std::string name = mix_op->name();
  uint32_t la_input = net_graph_->get_tensor_local_offset(in_tensors[0]);
  uint32_t la_output = net_graph_->get_tensor_local_offset(out_tensors[0]);

  Builder builder_(context_);
  std::vector<NamedAttribute> attrs;
  SmallVector<Attribute, 8> padsAttr;

  for (unsigned int i = 0; i < pads.size(); i++) {
    auto padAttr = builder_.getI32IntegerAttr(pads[i]);
    padsAttr.push_back(padAttr);
  }

  attrs.push_back(builder_.getNamedAttr("name",
                           builder_.getStringAttr(name)));
  attrs.push_back(builder_.getNamedAttr("align",
                           builder_.getBoolAttr(true)));
  attrs.push_back(builder_.getNamedAttr("la_input",
                           builder_.getI32IntegerAttr(la_input)));
  attrs.push_back(builder_.getNamedAttr("la_output",
                           builder_.getI32IntegerAttr(la_output)));
  attrs.push_back(builder_.getNamedAttr("const_val",
                           builder_.getF32FloatAttr(const_val)));
  attrs.push_back(builder_.getNamedAttr("pads",
                           builder_.getArrayAttr(padsAttr)));

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
  std::vector<Value> operands;
  Operation * input_op =
                    get_op_from_name(mix_op->bottom_name(0)).getDefiningOp();
  input_op->getResult(0).setType(input_type);
  operands.push_back(input_op->getResult(0));

  // build tl_pad operation
  if (isa<tpu::TG_INT8_PadOp>(op)) {
    auto tl_op = OpBuilder(get_start_op()).create<tpu::TL_LG_INT8_PadOp>(
                        get_start_op()->getLoc(), output_type,
                        ArrayRef<Value>{operands},
                        ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(mix_op->name(), tl_op.getResult(), true);
  } else if (isa<tpu::TG_BF16_PadOp>(op)) {
    auto tl_op = OpBuilder(get_start_op()).create<tpu::TL_LG_BF16_PadOp>(
                        get_start_op()->getLoc(), output_type,
                        ArrayRef<Value>{operands},
                        ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(mix_op->name(), tl_op.getResult(), true);
  }
}

void MixNet::_add_tl_crop_op(MixOp * mix_op,
                             const std::vector<int>& in_tensors,
                             const std::vector<int>& out_tensors,
                             net_timestep* time_step,
                             int timestep_idx,
                             bool is_h_split) {
  const ImLayer* im_layer = net_graph_->get_layer_by_id(mix_op->get_layer_id());
  const Tensor* in_tensor = net_graph_->get_tensor_by_id(in_tensors[0]);
  const Tensor* out_tensor = net_graph_->get_tensor_by_id(out_tensors[0]);
  Operation* op = im_layer->op();
  auto opd0 = op->getOperand(0);
  auto old_input_type = opd0.getType().cast<RankedTensorType>();

  std::vector<int32_t> crop_offsets;
  if (auto crop_op = dyn_cast<tpu::TG_INT8_CropOp>(op))
    arrayAttrToVector(crop_op.crop_offset().getValue(), crop_offsets);
  else if(auto crop_op = dyn_cast<tpu::TG_BF16_CropOp>(op))
    arrayAttrToVector(crop_op.crop_offset().getValue(), crop_offsets);

  int bottom_dim[4];
  int top_dim[4];

  net_graph_->get_tensor_dim(in_tensors[0], bottom_dim);
  net_graph_->get_tensor_dim(out_tensors[0], top_dim);
  bottom_dim[0] = in_tensor->n_slice;
  bottom_dim[2] = in_tensor->h_slice;
  top_dim[0] = out_tensor->n_slice;
  top_dim[2] = out_tensor->h_slice;

  if (is_h_split) {
    int real_h_slice = 0;
    int real_h_idx = 0;

    // bottom
    if (in_tensor->h_idx > 0) {
      real_h_idx = in_tensor->h_idx;
    } else {
      real_h_idx = 0;
    }
    int h_end = in_tensor->h_idx + in_tensor->h_slice;
    if (h_end >= in_tensor->h()) {
      real_h_slice = in_tensor->h() - real_h_idx;
      crop_offsets[2] = 0;
    } else {
      real_h_slice = h_end - real_h_idx;
      if (in_tensor->h_idx != 0)
        crop_offsets[2] = 0;
    }
    bottom_dim[2] = real_h_slice;
  }

  assert(bottom_dim[2] > crop_offsets[2]);

  std::string name = mix_op->name();
  uint32_t la_input = net_graph_->get_tensor_local_offset(in_tensors[0]);
  uint32_t la_output = net_graph_->get_tensor_local_offset(out_tensors[0]);

  Builder builder_(context_);
  std::vector<NamedAttribute> attrs;
  SmallVector<Attribute, 4> crop_offsetsAttr;

  for (unsigned int i = 0; i < crop_offsets.size(); i++) {
    auto offsetAttr = builder_.getI32IntegerAttr(crop_offsets[i]);
    crop_offsetsAttr.push_back(offsetAttr);
  }

  attrs.push_back(builder_.getNamedAttr("name",
                           builder_.getStringAttr(name)));
  attrs.push_back(builder_.getNamedAttr("align",
                           builder_.getBoolAttr(true)));
  attrs.push_back(builder_.getNamedAttr("la_input",
                           builder_.getI32IntegerAttr(la_input)));
  attrs.push_back(builder_.getNamedAttr("la_output",
                           builder_.getI32IntegerAttr(la_output)));
  attrs.push_back(builder_.getNamedAttr("crop_offsets",
                           builder_.getArrayAttr(crop_offsetsAttr)));

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
  std::vector<Value> operands;
  Operation * input_op =
                    get_op_from_name(mix_op->bottom_name(0)).getDefiningOp();
  input_op->getResult(0).setType(input_type);
  operands.push_back(input_op->getResult(0));

  // build tl_crop operation
  if (isa<tpu::TG_INT8_CropOp>(op)) {
    auto tl_op = OpBuilder(get_start_op()).create<tpu::TL_LG_INT8_CropOp>(
                        get_start_op()->getLoc(), output_type,
                        ArrayRef<Value>{operands},
                        ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(mix_op->name(), tl_op.getResult(), true);
  } else if (isa<tpu::TG_BF16_CropOp>(op)) {
    auto tl_op = OpBuilder(get_start_op()).create<tpu::TL_LG_BF16_CropOp>(
                        get_start_op()->getLoc(), output_type,
                        ArrayRef<Value>{operands},
                        ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(mix_op->name(), tl_op.getResult(), true);
  }
}

void MixNet::_add_tl_relu_op(MixOp * mix_op,
                             const std::vector<int>& in_tensors,
                             const std::vector<int>& out_tensors,
                             net_timestep* time_step,
                             int timestep_idx,
                             bool is_h_split) {
  const ImLayer* im_layer = net_graph_->get_layer_by_id(mix_op->get_layer_id());
  Operation *op = im_layer->op();
  auto opd0 = op->getOperand(0);
  auto old_input_type = opd0.getType().cast<RankedTensorType>();

  int bottom_dim[4];
  int top_dim[4];

  net_graph_->get_tl_tensor_dim(in_tensors[0], bottom_dim, is_h_split);
  net_graph_->get_tl_tensor_dim(out_tensors[0], top_dim, is_h_split);

  std::string name = mix_op->name();
  uint32_t la_input = net_graph_->get_tensor_local_offset(in_tensors[0]);
  uint32_t la_output = net_graph_->get_tensor_local_offset(out_tensors[0]);

  Builder builder_(context_);
  std::vector<NamedAttribute> attrs;

  attrs.push_back(builder_.getNamedAttr("name",
                           builder_.getStringAttr(name)));
  attrs.push_back(builder_.getNamedAttr("align",
                           builder_.getBoolAttr(true)));
  attrs.push_back(builder_.getNamedAttr("la_input",
                           builder_.getI32IntegerAttr(la_input)));
  attrs.push_back(builder_.getNamedAttr("la_output",
                           builder_.getI32IntegerAttr(la_output)));

  RankedTensorType output_type = RankedTensorType::get(
                          {top_dim[0], top_dim[1],
                           top_dim[2], top_dim[3]},
                           old_input_type.getElementType());

   // setup input operation
  std::vector<Value> operands;
  Operation * input_op =
                    get_op_from_name(mix_op->bottom_name(0)).getDefiningOp();
  input_op->getResult(0).setType(output_type);
  operands.push_back(input_op->getResult(0));

  // build tl_relu operation
  if (isa<tpu::TG_INT8_ReluOp>(op)) {
    auto tl_op = OpBuilder(get_start_op()).create<tpu::TL_LG_INT8_ReluOp>(
                        get_start_op()->getLoc(), output_type,
                        ArrayRef<Value>{operands},
                        ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(mix_op->name(), tl_op.getResult(), true);
  } else if (isa<tpu::TG_BF16_ReluOp>(op)) {
    auto tl_op = OpBuilder(get_start_op()).create<tpu::TL_LG_BF16_ReluOp>(
                        get_start_op()->getLoc(), output_type,
                        ArrayRef<Value>{operands},
                        ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(mix_op->name(), tl_op.getResult(), true);
  }
}

void MixNet::_add_tl_zero_mask_op(MixOp * mix_op,
                                   const std::vector<int>& in_tensors,
                                   const std::vector<int>& out_tensors,
                                   net_timestep* time_step,
                                   int timestep_idx,
                                   bool is_h_split) {
  const ImLayer* im_layer = net_graph_->get_layer_by_id(mix_op->get_layer_id());
  const Tensor* in_tensor = net_graph_->get_tensor_by_id(in_tensors[0]);
  const Tensor* out_tensor = net_graph_->get_tensor_by_id(out_tensors[0]);
  Operation* op = im_layer->op();
  auto old_input_type = op->getOperand(0).getType().cast<RankedTensorType>();
  Builder builder_(context_);
  std::vector<NamedAttribute> attrs;
  int bottom_dim[4];
  int top_dim[4];

  net_graph_->get_tensor_dim(in_tensors[0], bottom_dim);
  net_graph_->get_tensor_dim(out_tensors[0], top_dim);
  bottom_dim[0] = in_tensor->n_slice;
  bottom_dim[2] = in_tensor->h_slice;

  top_dim[0] = out_tensor->n_slice;
  top_dim[2] = out_tensor->h_slice;

  std::string name = mix_op->name();
  uint32_t la_input = net_graph_->get_tensor_local_offset(in_tensors[0]);
  uint32_t la_output = net_graph_->get_tensor_local_offset(out_tensors[0]);


  attrs.push_back(builder_.getNamedAttr("name",
                           builder_.getStringAttr(name)));
  attrs.push_back(builder_.getNamedAttr("la_input",
                           builder_.getI32IntegerAttr(la_input)));
  attrs.push_back(builder_.getNamedAttr("la_output",
                           builder_.getI32IntegerAttr(la_output)));

  // setup input/output type
  RankedTensorType input_type = RankedTensorType::get(
                          { bottom_dim[0], bottom_dim[1],
                            bottom_dim[2], bottom_dim[3]},
                            old_input_type.getElementType());

  RankedTensorType output_type = RankedTensorType::get(
                          { top_dim[0], top_dim[1],
                            top_dim[2], top_dim[3]},
                            old_input_type.getElementType());
  // setup input operation
  std::vector<Value> operands;
  Operation * input_op =
                    get_op_from_name(mix_op->bottom_name(0)).getDefiningOp();
  input_op->getResult(0).setType(input_type);
  operands.push_back(input_op->getResult(0));

  // build tl_zero_mask operation
  if (isa<tpu::TG_INT8_ZeroMaskOp>(op)) {
    mem_buffer_key_t key = {timestep_idx, im_layer->imm_tensors[0].get()->id(), true};
    const mem_buffer_value_t* imm = time_step->get_mem_buffer_value(&key);
    uint32_t la_working = imm->local_mem_offset;
    attrs.push_back(builder_.getNamedAttr("la_working",
                           builder_.getI32IntegerAttr(la_working)));
    auto tl_op = OpBuilder(get_start_op()).create<tpu::TL_LG_INT8_ZeroMaskOp>(
                        get_start_op()->getLoc(), output_type,
                        ArrayRef<Value>{operands},
                        ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(mix_op->name(), tl_op.getResult(), true);
  } else if (isa<tpu::TG_BF16_ZeroMaskOp>(op)) {
    attrs.push_back(builder_.getNamedAttr("la_working",
                           builder_.getI32IntegerAttr(0)));
    auto tl_op = OpBuilder(get_start_op()).create<tpu::TL_LG_BF16_ZeroMaskOp>(
                        get_start_op()->getLoc(), output_type,
                        ArrayRef<Value>{operands},
                        ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(mix_op->name(), tl_op.getResult(), true);
  }
}

void MixNet::_add_tl_slice_op(MixOp * mix_op,
                             const std::vector<int>& in_tensors,
                             const std::vector<int>& out_tensors,
                             net_timestep* time_step,
                             int timestep_idx,
                             bool is_h_split) {
  const ImLayer* im_layer = net_graph_->get_layer_by_id(mix_op->get_layer_id());
  Operation* op = im_layer->op();
  auto opd0 = op->getOperand(0);
  auto old_input_type = opd0.getType().cast<RankedTensorType>();

  int offset = 0;
  int axis = 0;
  if (auto slice_op = dyn_cast<tpu::TG_INT8_SliceOp>(op)) {
    offset = slice_op.offset();
    axis = slice_op.axis();
  } else if(auto slice_op = dyn_cast<tpu::TG_BF16_SliceOp>(op)) {
    offset = slice_op.offset();
    axis = slice_op.axis();
  }

  int bottom_dim[4];
  int top_dim[4];

  net_graph_->get_tl_tensor_dim(in_tensors[0], bottom_dim, is_h_split);
  net_graph_->get_tl_tensor_dim(out_tensors[0], top_dim, is_h_split);

  std::string name = mix_op->name();
  uint32_t la_input = net_graph_->get_tensor_local_offset(in_tensors[0]);
  uint32_t la_output = net_graph_->get_tensor_local_offset(out_tensors[0]);

  Builder builder_(context_);
  std::vector<NamedAttribute> attrs;

  attrs.push_back(builder_.getNamedAttr("name",
                           builder_.getStringAttr(name)));
  attrs.push_back(builder_.getNamedAttr("align",
                           builder_.getBoolAttr(true)));
  attrs.push_back(builder_.getNamedAttr("la_input",
                           builder_.getI32IntegerAttr(la_input)));
  attrs.push_back(builder_.getNamedAttr("la_output",
                           builder_.getI32IntegerAttr(la_output)));
  attrs.push_back(builder_.getNamedAttr("offset",
                           builder_.getI32IntegerAttr(offset)));
  attrs.push_back(builder_.getNamedAttr("axis",
                           builder_.getI32IntegerAttr(axis)));

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
  std::vector<Value> operands;
  Operation * input_op =
                    get_op_from_name(mix_op->bottom_name(0)).getDefiningOp();
  input_op->getResult(0).setType(input_type);
  operands.push_back(input_op->getResult(0));

  // build tl_slice operation
  if (isa<tpu::TG_INT8_SliceOp>(op)) {
    auto tl_op = OpBuilder(get_start_op()).create<tpu::TL_LG_INT8_SliceOp>(
                        get_start_op()->getLoc(), output_type,
                        ArrayRef<Value>{operands},
                        ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(mix_op->name(), tl_op.getResult(), true);
  } else if (isa<tpu::TG_BF16_SliceOp>(op)) {
    auto tl_op = OpBuilder(get_start_op()).create<tpu::TL_LG_BF16_SliceOp>(
                        get_start_op()->getLoc(), output_type,
                        ArrayRef<Value>{operands},
                        ArrayRef<NamedAttribute>{attrs});
    add_opd_to_list(mix_op->name(), tl_op.getResult(), true);
  }
}
}
