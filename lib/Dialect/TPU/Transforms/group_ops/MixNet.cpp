/*
 * Copyright (C) Bitmain Technologies Inc.
 * All Rights Reserved.
 */
#include "MixNet.hpp"

namespace mlir {

void MixNet::set_net_in_tensor(int tensor_id) { this->net_in_tensors_.push_back(tensor_id); }

void MixNet::set_net_out_tensor(int tensor_id) { this->net_out_tensors_.push_back(tensor_id); }

// void MixNet::add_start_layer(const std::string input_name, const int data_type_size) {
//   LayerParameter* layer = out_net_.add_layer();
//   int dim[4];
//   // only support one input tensor of network.
//   int tensor_id = this->net_in_tensors_[0];
//   net_graph_->get_tensor_dim(tensor_id, dim);

//   layer->set_name("start");
//   layer->set_type("start");
//   layer->add_top(input_name);
//   layer->set_id(0);
//   CHECK(out_net_.layer_size() == 1) << "Cannot exist any layer before start.";;

//   BlobShape* output_shape = layer->add_output_shape();
//   output_shape->add_dim(dim[0]);
//   output_shape->add_dim(dim[1]);
//   output_shape->add_dim(dim[2]);
//   output_shape->add_dim(dim[3]);
//   output_shape->set_data_type_size(data_type_size);
//   layer->add_global_output(0);

//   StartParameter* start = layer->mutable_start_param();
//   start->set_input_offset(0);
// }

// void MixNet::add_end_layer(u64 neuron_size) {
//   LayerParameter* layer = out_net_.add_layer();

//   layer->set_name("end");
//   layer->set_type("end");
//   layer->set_id(out_net_.layer_size());

//   int tensor_id = this->net_out_tensors_[0];
//   Tensor* bottom = net_graph_->get_tensor_by_id(tensor_id);
//   const int(&dim)[4] = bottom->dims();

//   layer->add_bottom(bottom->name());

//   BlobShape* input_shape = layer->add_input_shape();
//   input_shape->add_dim(dim[0]);
//   if (dim[1]) {
//     input_shape->add_dim(dim[1]);
//   }
//   if (dim[2]) {
//     input_shape->add_dim(dim[2]);
//   }
//   if (dim[3]) {
//     input_shape->add_dim(dim[3]);
//   }

//   input_shape->set_data_type_size(bottom->unit_size());

//   u64 global_offset = net_graph_->get_tensor_global_mem(tensor_id);
//   u64 tensor_size = static_cast<u64>(bottom->unit_size()) * (dim[0] ? dim[0] : 1) *
//                     (dim[1] ? dim[1] : 1) * (dim[2] ? dim[2] : 1) * (dim[3] ? dim[3] : 1);

//   EndParameter* end = layer->mutable_end_param();
//   end->set_output_offset(global_offset);
//   end->set_output_size(tensor_size);
//   end->set_total_neuron_size(neuron_size);
// }

template<typename OpTy>
struct Default2TGPattern : public RewritePattern {
  Default2TGPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    auto tpuOp = llvm::dyn_cast<tpu::TpuOpLowerInterface>(op);
    if (!tpuOp) {
      return matchFailure();
    }
    llvm::errs() << "default tg pattern.\n";
    // auto newValue = tpuOp.convertToTG();
    // if (!newValue) {
    //   return matchFailure();
    // }
    // rewriter.replaceOp(op, {newValue});
    return matchSuccess();
  }
};

void MixNet::add_tg_layer(int layer_id) {
  const ImLayer* im_layer = net_graph_->get_layer_by_id(layer_id);

  Operation * op = im_layer->op();

  if (isa<tpu::Conv2DOp>(op)) {

    // auto builder = Builder(op->getContext());
    // std::vector<Value *> operands;
    // for (auto * v : op->getOperands()) {
    //   operands.push_back(v);
    // }

    // auto attrs = op->getAttrList();
    // attrs.set(Identifier::get(StringRef("name"),op->getContext()), builder.getStringAttr(StringRef("hello")));
    //attrs.push_back(builder.getNamedAttr("gaddr", builder.getIntergerAttr(0xFFFF0000)));
    // auto new_op = OpBuilder(op).create<tpu::Conv2DOp>(op->getLoc(),
    //                 op->getResult(0)->getType(),
    //                 ArrayRef<Value *>{operands}, ArrayRef<NamedAttribute>{attrs});
    // auto new_op = OpBuilder(op).create<tpu::Conv2DOp>(op->getLoc(),
    //                 op->getResult(0)->getType(),
    //                 ArrayRef<Value *>{operands}, attrs.getAttrs());

  }

    // OwningRewritePatternList patterns_pack;
    // patterns_pack.insert<
    //     Default2TGPattern<tpu::Conv2DOp>
    // >(op->getContext());
    // applyPatternsGreedily(*fn_, patterns_pack);

}

// void MixNet::add_tg_layer(int layer_id) {
//   const ImLayer* im_layer = net_graph_->get_layer_by_id(layer_id);

//   LayerParameter* layer = out_net_.add_layer();
//   layer->CopyFrom(*(im_layer->op()));

//   layer->set_id(out_net_.layer_size());
//   layer->clear_global_input();
//   layer->clear_global_output();

//   for (int i = 0; i < layer->bottom_size(); i++) {
//     for (auto& tensor : im_layer->in_tensors) {
//       if (tensor->name() == layer->bottom(i)) {
//         layer->add_global_input(tensor->gaddr);
//         break;
//       }
//     }
//   }

//   for (int i = 0; i < layer->top_size(); i++) {
//     for (auto& tensor : im_layer->out_tensors) {
//       if (tensor->name() == layer->top(i)) {
//         layer->add_global_output(tensor->gaddr);
//         break;
//       }
//     }
//   }

//   if (im_layer->type() == IR_CONCAT) {
//     if (im_layer->op()->tg_concat_param().need_quantize_num() == 0) {
//       ImConcat* concat_layer = (ImConcat*)im_layer;
//       for (auto idx : concat_layer->ignored_bottoms) {
//         layer->mutable_tg_concat_param()->add_ignored_bottom(idx);
//       }
//     }
//   }
// }

// static const std::string _get_postfix_name(int group_idx, int n_loop, int h_loop) {
//   const std::string name = std::string("_") + std::to_string(group_idx) + "_" +
//                            std::to_string(n_loop) + "_" + std::to_string(h_loop);
//   return name;
// }

// void MixNet::add_group_start_layer(int group_idx, Group* cluster, int n_secs, int h_secs) {
//   LayerParameter* layer = out_net_.add_layer();
//   std::string layer_name = std::string("group") + std::to_string(group_idx);
//   layer->set_name(layer_name);
//   layer->set_type("tl_group");
//   layer->set_id(out_net_.layer_size());

//   set<int> in_neuron_tensors = cluster->get_cluster_in_neuron_tensors();
//   if (in_neuron_tensors.empty()) {
//     const LayerParameter &start_layer = out_net_.layer(0);
//     CHECK(start_layer.type() == "start") << "The first layer is not start layer.";
//     CHECK(start_layer.top_size() == 1) << "Start layer has more than one output.";
//     layer->add_bottom(start_layer.top(0));
//     layer->add_global_input(0);
//     for (int i = 0; i < n_secs; i++) {
//       for (int j = 0; j < h_secs; j++) {
//         layer->add_top(start_layer.top(0) + _get_postfix_name(group_idx, i, j));
//       }
//     }
//   } else {
//     for (auto tid : in_neuron_tensors) {
//       Tensor* tensor = net_graph_->get_tensor_by_id(tid);
//       std::string name = tensor->name();
//       layer->add_bottom(name);
//       layer->add_global_input(tensor->gaddr);
//       for (int i = 0; i < n_secs; i++) {
//         for (int j = 0; j < h_secs; j++) {
//           layer->add_top(name + _get_postfix_name(group_idx, i, j));
//         }
//       }
//     }
//   }
// }

// void MixNet::add_group_end_layer(int group_idx, Group* cluster, int n_secs, int h_secs) {
//   LayerParameter* layer = out_net_.add_layer();
//   TLGroupParameter* param = layer->mutable_tl_group_param();
//   std::string layer_name = std::string("groupend") + std::to_string(group_idx);
//   layer->set_name(layer_name);
//   layer->set_type("tl_group_end");
//   layer->set_id(out_net_.layer_size());

//   vector<int> out_tensors = cluster->get_cluster_out_tensors();
//   for (auto tid : out_tensors) {
//     Tensor* tensor = net_graph_->get_tensor_by_id(tid);
//     if (tensor->type() != TENSOR_NEURON && tensor->type() != TENSOR_NEURON_WINOGRAD &&
//         tensor->type() != TENSOR_MATRIX) {
//       continue;
//     }

//     string tensor_name = tensor->name();
//     int from_layer = net_graph_->get_tensor_from_layer(tid);
//     const ImLayer* im_layer = net_graph_->get_layer_by_id(from_layer);

//     layer->add_top(tensor_name);
//     layer->add_global_output(tensor->gaddr);
//     param->add_from_layer(im_layer->name());

//     BlobShape* shape = layer->add_output_shape();
//     int n = tensor->n();
//     int c = tensor->c();
//     int h = tensor->h();
//     int w = tensor->w();

//     shape->set_data_type_size(tensor->unit_size());
//     shape->add_dim(n);
//     if (c) {
//       shape->add_dim(c);
//     }
//     if (h) {
//       shape->add_dim(h);
//     }
//     if (w) {
//       shape->add_dim(w);
//     }

//     for (int i = 0; i < n_secs; i++) {
//       for (int j = 0; j < h_secs; j++) {
//         layer->add_bottom(tensor_name + _get_postfix_name(group_idx, i, j));
//       }
//     }
//   }
// }

// void MixNet::add_tl_layer(int group_idx, int layer_id, net_timestep* time_step, int timestep_idx,
//                           bool is_h_split, int n_loop, int h_loop) {
//   const ImLayer* im_layer = net_graph_->get_layer_by_id(layer_id);
//   const vector<int>& in_tensors = net_graph_->get_in_tensors_of_layer(layer_id);
//   const vector<int>& out_tensors = net_graph_->get_out_tensors_of_layer(layer_id);
//   const std::string postfix = _get_postfix_name(group_idx, n_loop, h_loop);

//   LayerParameter* layer = out_net_.add_layer();
//   layer->set_name(im_layer->name() + postfix);
//   if (CHIP_IS_BM188X) {
//     layer->set_calib_id(im_layer->name());
//   }
//   layer->set_id(out_net_.layer_size());

//   while (!suspend_transports_.empty()) {
//     TLTransportParameter* transport = layer->add_tl_transport_param();
//     TLTransportParameter* suspend = suspend_transports_.back();
//     transport->CopyFrom(*suspend);
//     delete suspend;
//     suspend_transports_.pop_back();
//   }

//   for (int i = 0; i < in_tensors.size(); i++) {
//     Tensor* in_tensor = net_graph_->get_tensor_by_id(in_tensors[i]);
//     if (in_tensor->type() != TENSOR_NEURON && in_tensor->type() != TENSOR_NEURON_WINOGRAD &&
//         in_tensor->type() != TENSOR_MATRIX) {
//       continue;
//     }

//     const string& name = in_tensor->name();
//     layer->add_bottom(name + postfix);
//   }

//   for (int i = 0; i < out_tensors.size(); i++) {
//     Tensor* out_tensor = net_graph_->get_tensor_by_id(out_tensors[i]);
//     const string& name = out_tensor->name();
//     layer->add_top(name + postfix);
//   }

//   for (int i = 0; i < out_tensors.size(); ++i) {
//     mem_buffer_key_t key = {timestep_idx, out_tensors[i], false};
//     const mem_buffer_value_t* value = time_step->get_mem_buffer_value(&key);
//     net_graph_->set_tensor_local_offest(out_tensors[i], value->local_mem_offset);
//   }

//   IR_TYPE layer_type = im_layer->type();

//   switch (layer_type) {
//     case IR_CONVOLUTION:
//       layer->set_type("tl_convolution");
//       _add_tl_convolution_param(layer_id, layer, im_layer, in_tensors, out_tensors, time_step,
//                                 timestep_idx, is_h_split);
//       break;
//     case IR_DECONVOLUTION:
//       layer->set_type("tl_deconvolution");
//       _add_tl_deconvolution_param(layer_id, layer, im_layer, in_tensors, out_tensors, time_step,
//                                   timestep_idx, is_h_split);
//       break;
//     case IR_POOLING:
//       layer->set_type("tl_pooling");
//       _add_tl_pooling_param(layer, im_layer, in_tensors, out_tensors, time_step, timestep_idx,
//                             is_h_split);
//       break;
//     case IR_LRN:
//       layer->set_type("tl_lrn");
//       _add_tl_lrn_param(layer_id, layer, im_layer, in_tensors, out_tensors, time_step, timestep_idx,
//                         is_h_split);
//       break;
//     case IR_BATCHNORM:
//       layer->set_type("tl_batchnorm");
//       _add_tl_batchnorm_param(layer, im_layer, in_tensors, out_tensors, time_step, timestep_idx,
//                               is_h_split);
//       break;
//     case IR_SCALE:
//       layer->set_type("tl_scale");
//       _add_tl_scale_param(layer, im_layer, in_tensors, out_tensors, time_step, timestep_idx,
//                           is_h_split);
//       break;
//     case IR_MAC:
//       layer->set_type("tl_mac");
//       _add_tl_mac_param(layer, im_layer, in_tensors, out_tensors, time_step, timestep_idx,
//                         is_h_split);
//       break;
//     case IR_ELTWISE:
//       layer->set_type("tl_eltwise");
//       _add_tl_eltwise_param(layer_id, layer, im_layer, in_tensors, out_tensors, time_step,
//                             timestep_idx, is_h_split);
//       break;
//     case IR_UPSAMPLE:
//       layer->set_type("tl_upsample");
//       _add_tl_upsample_param(layer, im_layer, in_tensors, out_tensors, time_step, timestep_idx,
//                              is_h_split);
//       break;
//     case IR_PRELU:
//     case IR_RELU:
//     case IR_ACTIVATION:
//       layer->set_type("tl_activation");
//       _add_tl_activation_param(layer, im_layer, in_tensors, out_tensors, time_step, timestep_idx,
//                                is_h_split);
//       break;
//     case IR_SHUFFLECHANNEL:
//       layer->set_type("tl_shufflechannel");
//       _add_tl_shuffle_channel_param(layer, im_layer, in_tensors, out_tensors, time_step,
//                                     timestep_idx, is_h_split);
//       break;
//     case IR_ARITHMETIC:
//       layer->set_type("tl_arithmetic");
//       this->_add_tl_arithmetic_param(layer, im_layer, in_tensors, out_tensors, time_step,
//                                      timestep_idx, is_h_split);
//       break;
//     case IR_INNERPRODUCT:
//       layer->set_type("tl_innerproduct");
//       _add_tl_innerproduct_param(layer, im_layer, in_tensors, out_tensors, time_step, timestep_idx,
//                                  is_h_split);
//       break;
//     case IR_QUANTIZATION:
//       layer->set_type("tl_quantization");
//       _add_tl_quantization_param(layer, im_layer, in_tensors, out_tensors, time_step, timestep_idx,
//                                  is_h_split);
//       break;
//     default:
//       cout << "unknown layer type:" << layer_type << endl;
//       exit(1);
//   }
// }

// void MixNet::_add_tl_convolution_param(int layer_id, LayerParameter* layer, const ImLayer* im_layer,
//                                        const vector<int>& in_tensors,
//                                        const vector<int>& out_tensors, net_timestep* time_step,
//                                        int timestep_idx, bool is_h_split) {
//   const TGConvolutionParameter& param = im_layer->op()->tg_convolution_param();
//   const ImLayer* ir = net_graph_->get_layer_by_id(layer_id);
//   Tensor* in_tensor = ir->in_tensors[0].get();

//   int real_h_idx, real_h_slice;
//   int top_pad_h, bottom_pad_h;
//   int left_pad_w, right_pad_w;
//   int h_end;
//   int bottom_dim[4];
//   int top_dim[4];

//   net_graph_->get_tensor_dim(in_tensors[0], bottom_dim);
//   net_graph_->get_tensor_dim(out_tensors[0], top_dim);

//   bottom_dim[0] = in_tensor->n_slice;
//   bottom_dim[2] = in_tensor->h_slice;
//   top_pad_h = param.pad(0);
//   bottom_pad_h = param.pad(0);
//   left_pad_w = param.pad(1);
//   right_pad_w = param.pad(1);

//   if (param.pad_size() == 4) {
//     top_pad_h = param.pad(0);
//     bottom_pad_h = param.pad(1);
//     left_pad_w = param.pad(2);
//     right_pad_w = param.pad(3);
//   }

//   if (is_h_split) {
//     if (in_tensor->h_idx > 0) {
//       real_h_idx = in_tensor->h_idx;
//       top_pad_h = 0;
//     } else {
//       real_h_idx = 0;
//       top_pad_h = 0 - in_tensor->h_idx;
//     }
//     h_end = in_tensor->h_idx + in_tensor->h_slice;
//     if (h_end > in_tensor->h()) {
//       real_h_slice = in_tensor->h() - real_h_idx;
//       bottom_pad_h = h_end - in_tensor->h();
//     } else {
//       real_h_slice = h_end - real_h_idx;
//       bottom_pad_h = 0;
//     }
//     bottom_dim[2] = real_h_slice;
//   }

//   const Tensor* out_tensor = net_graph_->get_tensor_by_id(out_tensors[0]);
//   top_dim[0] = out_tensor->n_slice;
//   top_dim[2] = out_tensor->h_slice;

//   if (is_h_split) {
//     if (out_tensor->h_idx > 0) {
//       real_h_idx = out_tensor->h_idx;
//     } else {
//       real_h_idx = 0;
//     }
//     h_end = out_tensor->h_idx + out_tensor->h_slice;
//     if (h_end > out_tensor->h()) {
//       real_h_slice = out_tensor->h() - real_h_idx;
//     } else {
//       real_h_slice = h_end - real_h_idx;
//     }
//     top_dim[2] = real_h_slice;
//   }

//   TLConvolutionParameter* out_param = layer->mutable_tl_convolution_param();
//   out_param->set_input(net_graph_->get_tensor_local_offset(in_tensors[0]));
//   out_param->set_weight(net_graph_->get_tensor_local_offset(in_tensors[1]));
//   // Should be has_bias
//   // FIXME: Need to align with im_layers.cpp
//   if (in_tensors.size() > 2) {
//     out_param->set_bias_term(param.bias_term());
//     if (CHIP_IS_BM188X && param.group() > 1) {
//       if (param.use_winograd()) {
//         for (int i = 0; i < param.group(); i++) {
//           out_param->add_group_weight(net_graph_->get_tensor_local_offset(in_tensors[i + 1]));
//           out_param->add_group_bias(
//               net_graph_->get_tensor_local_offset(in_tensors[i + param.group() + 1]));
//         }
//       } else if (param.group() == out_tensor->c() && param.group() == in_tensor->c() && param.group() != 1) {
//         LOG(WARNING) << "Bias tensor for depthwise layer found!";
//         // Backend only gets bias, not add_group_bias, store both
//         out_param->set_bias(net_graph_->get_tensor_local_offset(in_tensors[2]));
//         out_param->add_group_bias(net_graph_->get_tensor_local_offset(in_tensors[2]));
//       } else if (in_tensors.size() == param.group() + 2) {
//         LOG(WARNING) << "Multiple bias tensors found!";
//         for (int i = 0; i < param.group(); i++) {
//           out_param->add_group_bias(net_graph_->get_tensor_local_offset(in_tensors[i + 2]));
//         }
//       } else {
//         // If you see this message, go to im_layers.cpp:186 (has_bias) for more info.
//         LOG(FATAL) << "Error nums of bias tensors from optimizer.";
//       }
//     } else {
//       out_param->set_bias(net_graph_->get_tensor_local_offset(in_tensors[2]));
//     }
//   }

//   //<! NOTICE: it must sync with im_layers and currently place the last one in in_tensors
//   if (FlagInst::get()->flagOpt(hwflags::OPT::SUPPORT_BIAS_INT32) && CHIP_IS_BM188X) {
//     out_param->set_global_per_channel(net_graph_->get_tensor_local_offset(in_tensors[in_tensors.size()-1]));
//   }

//   out_param->set_output(net_graph_->get_tensor_local_offset(out_tensors[0]));
//   out_param->set_h_slice_skip_first(out_tensor->h_slice_skip_first);
//   out_param->set_h_slice_skip_last(out_tensor->h_slice_skip_last);

//   assert(layer->input_shape_size() == 0);
//   BlobShape* input_shape = layer->add_input_shape();
//   input_shape->add_dim(bottom_dim[0]);
//   input_shape->add_dim(bottom_dim[1]);
//   input_shape->add_dim(bottom_dim[2]);
//   input_shape->add_dim(bottom_dim[3]);

//   assert(layer->output_shape_size() == 0);
//   BlobShape* output_shape = layer->add_output_shape();
//   output_shape->add_dim(top_dim[0]);
//   output_shape->add_dim(top_dim[1]);
//   output_shape->add_dim(top_dim[2]);
//   output_shape->add_dim(top_dim[3]);

//   out_param->set_group(param.group());
//   out_param->set_use_winograd(param.use_winograd());
//   out_param->add_kernel_size(net_graph_->get_tensor_height(in_tensors[1]));
//   out_param->add_kernel_size(net_graph_->get_tensor_width(in_tensors[1]));
//   out_param->add_dilation(param.dilation(0));
//   out_param->add_dilation(param.dilation(1));
//   out_param->add_pad(top_pad_h);
//   out_param->add_pad(bottom_pad_h);
//   out_param->add_pad(left_pad_w);
//   out_param->add_pad(right_pad_w);
//   out_param->add_stride(param.stride(0));
//   out_param->add_stride(param.stride(1));
//   out_param->set_result_add(param.result_add());
//   out_param->set_if_relu(param.do_activation());
//   if (param.activation_arg_size() > 0) {
//     out_param->set_relu_slope(param.activation_arg(0));
//     if (param.activation_arg(0) != 0.0f) {
//       mem_buffer_key_t key = {timestep_idx, ir->imm_tensors[0].get()->id(), true};
//       const mem_buffer_value_t* imm = time_step->get_mem_buffer_value(&key);
//       out_param->set_working(imm->local_mem_offset);
//     }
//   }

//   if (param.do_activation()) {
//     TGConvolutionParameter* _out_param = layer->mutable_tg_convolution_param();
//     _out_param->set_activation(param.activation());
//     for (u32 i = 0; i < param.activation_arg_size(); i++) {
//       _out_param->add_activation_arg(param.activation_arg(i));
//     }
//   }
// }

// void MixNet::_add_tl_deconvolution_param(int layer_id, LayerParameter* layer,
//                                          const ImLayer* im_layer, const vector<int>& in_tensors,
//                                          const vector<int>& out_tensors, net_timestep* time_step,
//                                          int timestep_idx, bool is_h_split) {
//   const TGConvolutionParameter& param = im_layer->op()->tg_convolution_param();
//   Tensor* in_tensor = net_graph_->get_tensor_by_id(in_tensors[0]);

//   int real_h_idx, real_h_slice;
//   int h_end;
//   int bottom_dim[4];
//   int top_dim[4];

//   net_graph_->get_tensor_dim(in_tensors[0], bottom_dim);
//   net_graph_->get_tensor_dim(out_tensors[0], top_dim);

//   bottom_dim[0] = in_tensor->n_slice;
//   bottom_dim[2] = in_tensor->h_slice;
//   int pad_h_top = param.pad(0);
//   int pad_h_bottom = param.pad(0);
//   int pad_w_left = param.pad(1);
//   int pad_w_right = param.pad(1);

//   if (param.pad_size() == 4) {
//     pad_h_top = param.pad(0);
//     pad_h_bottom = param.pad(1);
//     pad_w_left = param.pad(2);
//     pad_w_right = param.pad(3);
//   }

//   if (is_h_split) {
//     if (in_tensor->h_idx > 0) {
//       real_h_idx = in_tensor->h_idx;
//     } else {
//       real_h_idx = 0;
//     }

//     h_end = in_tensor->h_idx + in_tensor->h_slice;
//     if (h_end > in_tensor->h()) {
//       real_h_slice = in_tensor->h() - real_h_idx;
//     } else {
//       real_h_slice = h_end - real_h_idx;
//     }
//     bottom_dim[2] = real_h_slice;
//   }

//   const Tensor* out_tensor = net_graph_->get_tensor_by_id(out_tensors[0]);
//   top_dim[0] = out_tensor->n_slice;
//   top_dim[2] = out_tensor->h_slice;

//   real_h_slice = top_dim[2];
//   real_h_idx = out_tensor->h_idx;
//   if (is_h_split) {
//     if (out_tensor->h_idx > 0) {
//       real_h_idx = out_tensor->h_idx;
//     } else {
//       real_h_idx = 0;
//     }
//     h_end = out_tensor->h_idx + out_tensor->h_slice;
//     if (h_end > out_tensor->h()) {
//       real_h_slice = out_tensor->h() - real_h_idx;
//     } else {
//       real_h_slice = h_end - real_h_idx;
//     }
//     top_dim[2] = real_h_slice;
//   }

//   int kh = param.kernel_size(0);
//   int kw = param.kernel_size(1);
//   int dh = param.dilation(0);
//   int dw = param.dilation(1);
//   int stride_h = param.stride(0);
//   int stride_w = param.stride(1);
//   int kh_ext = (kh - 1) * dh + 1;
//   int kw_ext = (kw - 1) * dw + 1;
//   int input_h = in_tensor->h();
//   int output_w = out_tensor->w();
//   int ins_w_last = (output_w + pad_w_left + pad_w_right - kw_ext) % stride_w;
//   int height_insert0 = (input_h - 1) * stride_h + 1;
//   pad_h_top = kh_ext - pad_h_top - 1;
//   pad_h_bottom = kh_ext - pad_h_bottom - 1;
//   int o_ht = real_h_idx;
//   int o_hb = real_h_idx + real_h_slice;
//   int if_pad_h_t = o_ht;
//   int if_pad_h_b = o_hb + kh_ext - 1;
//   int if_insert_h_t = 0;
//   int pad_h_t = 0;
//   if (if_pad_h_t < pad_h_top) {
//     pad_h_t = pad_h_top - if_pad_h_t;
//   } else {
//     if_insert_h_t = if_pad_h_t - pad_h_top;
//   }

//   int if_insert_h_b = height_insert0;
//   int pad_h_b = 0;

//   if ((if_pad_h_b - pad_h_bottom) < height_insert0) {
//     if_insert_h_b = if_pad_h_b - pad_h_bottom;
//   } else {
//     pad_h_b = if_pad_h_b - height_insert0 - pad_h_bottom;
//   }
//   int hinsert0_t = (if_insert_h_t % stride_h == 0) ? 0 : (stride_h - if_insert_h_t % stride_h);
//   int hinsert0_b = (if_insert_h_b + stride_h - 1) % stride_h;

//   int up_pad_h = pad_h_t + hinsert0_t;
//   int down_pad_h = pad_h_b + hinsert0_b;
//   int left_pad_w = kw_ext - pad_w_left - 1;
//   int right_pad_w = kw_ext - pad_w_right - 1 + ins_w_last;
//   int ins_h = stride_h - 1;
//   int ins_w = stride_w - 1;

//   TLConvolutionParameter* out_param = layer->mutable_tl_convolution_param();
//   out_param->set_input(net_graph_->get_tensor_local_offset(in_tensors[0]));
//   out_param->set_weight(net_graph_->get_tensor_local_offset(in_tensors[1]));
//   if (in_tensors.size() > 2) {
//     out_param->set_bias_term(true);
//     if (CHIP_IS_BM188X && param.group() > 1) {
//       for (int i = 0; i < param.group(); i++) {
//         out_param->add_group_bias(net_graph_->get_tensor_local_offset(in_tensors[i + 2]));
//       }
//     } else {
//       out_param->set_bias(net_graph_->get_tensor_local_offset(in_tensors[2]));
//     }
//   }

//   //<! NOTICE: it must sync with im_layers and currently place the last one in in_tensors
//   if (FlagInst::get()->flagOpt(hwflags::OPT::SUPPORT_BIAS_INT32) && CHIP_IS_BM188X) {
//     out_param->set_global_per_channel(net_graph_->get_tensor_local_offset(in_tensors[in_tensors.size()-1]));
//   }

//   out_param->set_output(net_graph_->get_tensor_local_offset(out_tensors[0]));

//   assert(layer->input_shape_size() == 0);
//   BlobShape* input_shape = layer->add_input_shape();
//   input_shape->add_dim(bottom_dim[0]);
//   input_shape->add_dim(bottom_dim[1]);
//   input_shape->add_dim(bottom_dim[2]);
//   input_shape->add_dim(bottom_dim[3]);

//   assert(layer->output_shape_size() == 0);
//   BlobShape* output_shape = layer->add_output_shape();
//   output_shape->add_dim(top_dim[0]);
//   output_shape->add_dim(top_dim[1]);
//   output_shape->add_dim(top_dim[2]);
//   output_shape->add_dim(top_dim[3]);

//   out_param->set_group(param.group());
//   out_param->set_use_winograd(param.use_winograd());
//   out_param->add_kernel_size(net_graph_->get_tensor_height(in_tensors[1]));
//   out_param->add_kernel_size(net_graph_->get_tensor_width(in_tensors[1]));
//   out_param->add_dilation(param.dilation(0));
//   out_param->add_dilation(param.dilation(1));
//   out_param->add_pad(up_pad_h);
//   out_param->add_pad(down_pad_h);
//   out_param->add_pad(left_pad_w);
//   out_param->add_pad(right_pad_w);
//   out_param->add_ins(ins_h);
//   out_param->add_ins(0);
//   out_param->add_ins(ins_w);
//   out_param->add_ins(0);
//   out_param->add_stride(1);
//   out_param->add_stride(1);
//   out_param->set_result_add(param.result_add());
//   out_param->set_if_relu(param.do_activation());
// }

// void MixNet::_add_tl_pooling_param(LayerParameter* layer, const ImLayer* im_layer,
//                                    const vector<int>& in_tensors, const vector<int>& out_tensors,
//                                    net_timestep* time_step, int timestep_idx, bool is_h_split) {
//   const TGPoolingParameter& param = im_layer->op()->tg_pooling_param();
//   const Tensor* in_tensor = net_graph_->get_tensor_by_id(in_tensors[0]);
//   const Tensor* out_tensor = net_graph_->get_tensor_by_id(out_tensors[0]);

//   int bottom_dim[4];
//   int top_dim[4];
//   net_graph_->get_tensor_dim(in_tensors[0], bottom_dim);
//   net_graph_->get_tensor_dim(out_tensors[0], top_dim);

//   bottom_dim[0] = in_tensor->n_slice;
//   bottom_dim[2] = in_tensor->h_slice;

//   top_dim[0] = out_tensor->n_slice;
//   top_dim[2] = out_tensor->h_slice;

//   int top_pad_h = param.pad(0);
//   int bottom_pad_h = param.pad(0);
//   int left_pad_w = param.pad(1);
//   int right_pad_w = param.pad(1);

//   if (param.pad_size() == 4) {
//     top_pad_h = param.pad(0);
//     bottom_pad_h = param.pad(1);
//     left_pad_w = param.pad(2);
//     right_pad_w = param.pad(3);
//   }

//   int sh = param.stride(0);
//   int sw = param.stride(1);
//   int kh = param.kernel_size(0);
//   int kw = param.kernel_size(1);

//   if (sh * (top_dim[2] - 1) + kh > bottom_dim[2] + top_pad_h + bottom_pad_h) {
//     bottom_pad_h = sh * (top_dim[2] - 1) + kh - bottom_dim[2] - param.pad(0);
//   }

//   if (sw * (top_dim[3] - 1) + kw > bottom_dim[3] + left_pad_w + right_pad_w) {
//     right_pad_w = sw * (top_dim[3] - 1) + kw - bottom_dim[3] - param.pad(1);
//   }

//   if (is_h_split) {
//     int real_h_idx, real_h_slice;

//     // bottom
//     if (in_tensor->h_idx > 0) {
//       real_h_idx = in_tensor->h_idx;
//       top_pad_h = 0;
//     } else {
//       real_h_idx = 0;
//       top_pad_h = 0 - in_tensor->h_idx;
//     }
//     int h_end = in_tensor->h_idx + in_tensor->h_slice;
//     if (h_end > in_tensor->h()) {
//       real_h_slice = in_tensor->h() - real_h_idx;
//       bottom_pad_h = h_end - in_tensor->h();
//     } else {
//       real_h_slice = h_end - real_h_idx;
//       bottom_pad_h = 0;
//     }
//     bottom_dim[2] = real_h_slice;

//     // top
//     if (out_tensor->h_idx > 0) {
//       real_h_idx = out_tensor->h_idx;
//     } else {
//       real_h_idx = 0;
//     }
//     h_end = out_tensor->h_idx + out_tensor->h_slice;
//     if (h_end > out_tensor->h()) {
//       real_h_slice = out_tensor->h() - real_h_idx;
//     } else {
//       real_h_slice = h_end - real_h_idx;
//     }
//     top_dim[2] = real_h_slice;
//   }

//   TLPoolingParameter* out_param = layer->mutable_tl_pooling_param();
//   out_param->set_input(net_graph_->get_tensor_local_offset(in_tensors[0]));
//   out_param->set_output(net_graph_->get_tensor_local_offset(out_tensors[0]));

//   assert(layer->input_shape_size() == 0);
//   BlobShape* input_shape = layer->add_input_shape();
//   input_shape->add_dim(bottom_dim[0]);
//   input_shape->add_dim(bottom_dim[1]);
//   input_shape->add_dim(bottom_dim[2]);
//   input_shape->add_dim(bottom_dim[3]);

//   assert(layer->output_shape_size() == 0);
//   BlobShape* output_shape = layer->add_output_shape();
//   output_shape->add_dim(top_dim[0]);
//   output_shape->add_dim(top_dim[1]);
//   output_shape->add_dim(top_dim[2]);
//   output_shape->add_dim(top_dim[3]);

//   out_param->add_kernel_size(kh);
//   out_param->add_kernel_size(kw);
//   out_param->add_pad(top_pad_h);
//   out_param->add_pad(bottom_pad_h);
//   out_param->add_pad(left_pad_w);
//   out_param->add_pad(right_pad_w);
//   out_param->add_stride(sh);
//   out_param->add_stride(sw);
//   out_param->set_avg_pool(param.pool());
//   out_param->set_if_relu(param.do_activation());
// }

// void MixNet::_add_tl_upsample_param(LayerParameter* layer, const ImLayer* im_layer,
//                                     const vector<int>& in_tensors, const vector<int>& out_tensors,
//                                     net_timestep* time_step, int timestep_idx, bool is_h_split) {
//   const TGUpsampleParameter& param = im_layer->op()->tg_upsample_param();
//   const Tensor* in_tensor = net_graph_->get_tensor_by_id(in_tensors[0]);
//   const Tensor* out_tensor = net_graph_->get_tensor_by_id(out_tensors[0]);

//   int bottom_dim[4];
//   int top_dim[4];
//   net_graph_->get_tensor_dim(in_tensors[0], bottom_dim);
//   net_graph_->get_tensor_dim(out_tensors[0], top_dim);

//   bottom_dim[0] = in_tensor->n_slice;
//   bottom_dim[2] = in_tensor->h_slice;

//   top_dim[0] = out_tensor->n_slice;
//   top_dim[2] = out_tensor->h_slice;

//   int size = param.size(0);

//   if (is_h_split) {
//     int real_h_slice = 0;
//     int real_h_idx = 0;

//     // bottom
//     if (in_tensor->h_idx > 0) {
//       real_h_idx = in_tensor->h_idx;
//     } else {
//       real_h_idx = 0;
//     }
//     int h_end = in_tensor->h_idx + in_tensor->h_slice;
//     if (h_end > in_tensor->h()) {
//       real_h_slice = in_tensor->h() - real_h_idx;
//     } else {
//       real_h_slice = h_end - real_h_idx;
//     }
//     bottom_dim[2] = real_h_slice;

//     top_dim[2] = bottom_dim[2] * size;
//   }

//   TLUpsampleParameter* out_param = layer->mutable_tl_upsample_param();
//   out_param->set_input(net_graph_->get_tensor_local_offset(in_tensors[0]));
//   out_param->set_output(net_graph_->get_tensor_local_offset(out_tensors[0]));

//   assert(layer->input_shape_size() == 0);
//   BlobShape* input_shape = layer->add_input_shape();
//   input_shape->add_dim(bottom_dim[0]);
//   input_shape->add_dim(bottom_dim[1]);
//   input_shape->add_dim(bottom_dim[2]);
//   input_shape->add_dim(bottom_dim[3]);

//   assert(layer->output_shape_size() == 0);
//   BlobShape* output_shape = layer->add_output_shape();
//   output_shape->add_dim(top_dim[0]);
//   output_shape->add_dim(top_dim[1]);
//   output_shape->add_dim(top_dim[2]);
//   output_shape->add_dim(top_dim[3]);

//   out_param->add_size(size);
// }

// void MixNet::_add_tl_lrn_param(int layer_id, LayerParameter* layer, const ImLayer* im_layer,
//                                const vector<int>& in_tensors, const vector<int>& out_tensors,
//                                net_timestep* time_step, int timestep_idx, bool is_h_split) {
//   const TGLRNParameter& param = im_layer->op()->tg_lrn_param();
//   const ImLayer* ir = net_graph_->get_layer_by_id(layer_id);
//   const Tensor* in_tensor = ir->in_tensors[0].get();

//   mem_buffer_key_t key = {timestep_idx, ir->imm_tensors[0].get()->id(), true};
//   const mem_buffer_value_t* imm = time_step->get_mem_buffer_value(&key);

//   int bottom_dim[4];
//   net_graph_->get_tensor_dim(in_tensors[0], bottom_dim);

//   bottom_dim[0] = in_tensor->n_slice;
//   bottom_dim[2] = in_tensor->h_slice;

//   TLLrnParameter* out_param = layer->mutable_tl_lrn_param();
//   out_param->set_input(net_graph_->get_tensor_local_offset(in_tensors[0]));
//   if (CHIP_IS_BM188X) {
//     out_param->set_sqr_lut_weight(net_graph_->get_tensor_local_offset(in_tensors[1]));
//     out_param->set_power_lut_weight(net_graph_->get_tensor_local_offset(in_tensors[2]));
//   }
//   out_param->set_output(net_graph_->get_tensor_local_offset(out_tensors[0]));

//   assert(layer->input_shape_size() == 0);
//   BlobShape* input_shape = layer->add_input_shape();
//   input_shape->add_dim(bottom_dim[0]);
//   input_shape->add_dim(bottom_dim[1]);
//   input_shape->add_dim(bottom_dim[2]);
//   input_shape->add_dim(bottom_dim[3]);

//   assert(layer->output_shape_size() == 0);
//   BlobShape* output_shape = layer->add_output_shape();
//   output_shape->add_dim(bottom_dim[0]);
//   output_shape->add_dim(bottom_dim[1]);
//   output_shape->add_dim(bottom_dim[2]);
//   output_shape->add_dim(bottom_dim[3]);

//   out_param->set_working(imm->local_mem_offset);
//   out_param->set_alpha(param.alpha());
//   out_param->set_local_size(param.local_size());
//   out_param->set_beta(param.beta());
//   out_param->set_k(param.k());
//   out_param->set_norm_region(static_cast<TLLrnParameter_NormRegion>(param.norm_region()));
// }

// void MixNet::_add_tl_batchnorm_param(LayerParameter* layer, const ImLayer* im_layer,
//                                      const vector<int>& in_tensors, const vector<int>& out_tensors,
//                                      net_timestep* time_step, int timestep_idx, bool is_h_split) {
//   const TGBatchNormParameter& param = im_layer->op()->tg_batchnorm_param();
//   const Tensor* in_tensor = net_graph_->get_tensor_by_id(in_tensors[0]);

//   int bottom_dim[4];
//   net_graph_->get_tensor_dim(in_tensors[0], bottom_dim);

//   bottom_dim[0] = in_tensor->n_slice;
//   bottom_dim[2] = in_tensor->h_slice;

//   TLBatchNormParameter* out_param = layer->mutable_tl_batchnorm_param();
//   out_param->set_input(net_graph_->get_tensor_local_offset(in_tensors[0]));
//   out_param->set_output(net_graph_->get_tensor_local_offset(out_tensors[0]));

//   assert(layer->input_shape_size() == 0);
//   BlobShape* input_shape = layer->add_input_shape();
//   input_shape->add_dim(bottom_dim[0]);
//   input_shape->add_dim(bottom_dim[1]);
//   input_shape->add_dim(bottom_dim[2]);
//   input_shape->add_dim(bottom_dim[3]);

//   assert(layer->output_shape_size() == 0);
//   BlobShape* output_shape = layer->add_output_shape();
//   output_shape->add_dim(bottom_dim[0]);
//   output_shape->add_dim(bottom_dim[1]);
//   output_shape->add_dim(bottom_dim[2]);
//   output_shape->add_dim(bottom_dim[3]);

//   out_param->set_mean(net_graph_->get_tensor_local_offset(in_tensors[1]));
//   out_param->set_variance(net_graph_->get_tensor_local_offset(in_tensors[2]));
//   out_param->set_eps(param.eps());
//   out_param->set_scale_ma(param.global_fraction());
// }

// void MixNet::_add_tl_shuffle_channel_param(LayerParameter* layer, const ImLayer* im_layer,
//                                            const vector<int>& in_tensors,
//                                            const vector<int>& out_tensors, net_timestep* time_step,
//                                            int timestep_idx, bool is_h_split) {
//   const TGShuffleChannelParameter& param = im_layer->op()->tg_shuffle_channel_param();
//   const Tensor* in_tensor = net_graph_->get_tensor_by_id(in_tensors[0]);

//   int bottom_dim[4];
//   net_graph_->get_tensor_dim(in_tensors[0], bottom_dim);

//   bottom_dim[0] = in_tensor->n_slice;
//   bottom_dim[2] = in_tensor->h_slice;

//   TLShuffleChannelParameter* out_param = layer->mutable_tl_shuffle_channel_param();
//   out_param->set_input(net_graph_->get_tensor_local_offset(in_tensors[0]));
//   out_param->set_output(net_graph_->get_tensor_local_offset(out_tensors[0]));

//   assert(layer->input_shape_size() == 0);
//   BlobShape* input_shape = layer->add_input_shape();
//   input_shape->add_dim(bottom_dim[0]);
//   input_shape->add_dim(bottom_dim[1]);
//   input_shape->add_dim(bottom_dim[2]);
//   input_shape->add_dim(bottom_dim[3]);

//   assert(layer->output_shape_size() == 0);
//   BlobShape* output_shape = layer->add_output_shape();
//   output_shape->add_dim(bottom_dim[0]);
//   output_shape->add_dim(bottom_dim[1]);
//   output_shape->add_dim(bottom_dim[2]);
//   output_shape->add_dim(bottom_dim[3]);

//   out_param->set_group(param.group());
// }

// void MixNet::_add_tl_innerproduct_param(LayerParameter* layer, const ImLayer* im_layer,
//                                         const vector<int>& in_tensors,
//                                         const vector<int>& out_tensors, net_timestep* time_step,
//                                         int timestep_idx, bool is_h_split) {
//   const TGInnerProductParameter& param = im_layer->op()->tg_inner_product_param();
//   const Tensor* in_tensor = net_graph_->get_tensor_by_id(in_tensors[0]);

//   int bottom_dim[4];
//   int top_dim[4];

//   net_graph_->get_tensor_dim(in_tensors[0], bottom_dim);
//   net_graph_->get_tensor_dim(out_tensors[0], top_dim);

//   TLInnerProductParameter* out_param = layer->mutable_tl_inner_product_param();
//   out_param->set_input(net_graph_->get_tensor_local_offset(in_tensors[0]));
//   out_param->set_output(net_graph_->get_tensor_local_offset(out_tensors[0]));
//   out_param->set_weight(net_graph_->get_tensor_local_offset(in_tensors[1]));

//   if (in_tensors.size() > 2) {
//     out_param->set_bias_term(true);
//     out_param->set_bias(net_graph_->get_tensor_local_offset(in_tensors[2]));
//   }

//   out_param->set_output(net_graph_->get_tensor_local_offset(out_tensors[0]));
//   BlobShape* input_shape = layer->add_input_shape();
//   input_shape->add_dim(bottom_dim[0]);
//   input_shape->add_dim(bottom_dim[1]);
//   input_shape->add_dim(bottom_dim[2]);
//   input_shape->add_dim(bottom_dim[3]);

//   BlobShape* output_shape = layer->add_output_shape();
//   output_shape->add_dim(top_dim[0]);
//   output_shape->add_dim(top_dim[1]);
//   output_shape->add_dim(top_dim[2]);
//   output_shape->add_dim(top_dim[3]);

//   if (param.weight_transpose()) {
//     out_param->set_transpose(true);
//   }
//   if (param.has_result_add()) {
//     out_param->set_result_add(true);
//   }
//   if (param.has_activation()) {
//     out_param->set_do_activation(true);
//     out_param->set_activation(param.activation());
//   }
// }

// void MixNet::_add_tl_scale_param(LayerParameter* layer, const ImLayer* im_layer,
//                                  const vector<int>& in_tensors, const vector<int>& out_tensors,
//                                  net_timestep* time_step, int timestep_idx, bool is_h_split) {
//   const TGScaleParameter& param = im_layer->op()->tg_scale_param();
//   const Tensor* in_tensor = net_graph_->get_tensor_by_id(in_tensors[0]);

//   int bottom_dim[4];

//   net_graph_->get_tensor_dim(in_tensors[0], bottom_dim);
//   bottom_dim[0] = in_tensor->n_slice;
//   bottom_dim[2] = in_tensor->h_slice;

//   TLScaleParameter* out_param = layer->mutable_tl_scale_param();
//   out_param->set_input(net_graph_->get_tensor_local_offset(in_tensors[0]));
//   out_param->set_output(net_graph_->get_tensor_local_offset(out_tensors[0]));

//   assert(layer->input_shape_size() == 0);
//   BlobShape* input_shape = layer->add_input_shape();
//   input_shape->add_dim(bottom_dim[0]);
//   input_shape->add_dim(bottom_dim[1]);
//   input_shape->add_dim(bottom_dim[2]);
//   input_shape->add_dim(bottom_dim[3]);

//   assert(layer->output_shape_size() == 0);
//   BlobShape* output_shape = layer->add_output_shape();
//   output_shape->add_dim(bottom_dim[0]);
//   output_shape->add_dim(bottom_dim[1]);
//   output_shape->add_dim(bottom_dim[2]);
//   output_shape->add_dim(bottom_dim[3]);

//   out_param->set_is_scale_const(param.is_scale_const());
//   if (!param.is_scale_const()) {
//     out_param->set_scale_dim(param.scale_dim());
//     out_param->set_scale(net_graph_->get_tensor_local_offset(in_tensors[1]));
//   } else {
//     out_param->set_const_scale(param.const_scale());
//   }

//   if (in_tensors.size() > 2) {
//     out_param->set_bias_term(true);
//     out_param->set_bias(net_graph_->get_tensor_local_offset(in_tensors[2]));
//   }

//   out_param->set_if_relu(param.do_activation());
//   if (param.activation_arg_size() > 0) {
//     out_param->set_relu_slope(param.activation_arg(0));
//   }
// }

// void MixNet::_add_tl_mac_param(LayerParameter* layer, const ImLayer* im_layer,
//                                const vector<int>& in_tensors, const vector<int>& out_tensors,
//                                net_timestep* time_step, int timestep_idx, bool is_h_split) {
//   const TGMacParameter& param = im_layer->op()->tg_mac_param();
//   const Tensor* in_tensor = net_graph_->get_tensor_by_id(in_tensors[0]);

//   int bottom_dim[4];

//   net_graph_->get_tensor_dim(in_tensors[0], bottom_dim);
//   bottom_dim[0] = in_tensor->n_slice;
//   bottom_dim[2] = in_tensor->h_slice;

//   TLMacParameter* out_param = layer->mutable_tl_mac_param();
//   out_param->set_input(net_graph_->get_tensor_local_offset(in_tensors[0]));
//   out_param->set_output(net_graph_->get_tensor_local_offset(out_tensors[0]));

//   assert(layer->input_shape_size() == 0);
//   BlobShape* input_shape = layer->add_input_shape();
//   input_shape->add_dim(bottom_dim[0]);
//   input_shape->add_dim(bottom_dim[1]);
//   input_shape->add_dim(bottom_dim[2]);
//   input_shape->add_dim(bottom_dim[3]);

//   assert(layer->output_shape_size() == 0);
//   BlobShape* output_shape = layer->add_output_shape();
//   output_shape->add_dim(bottom_dim[0]);
//   output_shape->add_dim(bottom_dim[1]);
//   output_shape->add_dim(bottom_dim[2]);
//   output_shape->add_dim(bottom_dim[3]);

//   out_param->set_scale(net_graph_->get_tensor_local_offset(in_tensors[1]));
//   out_param->set_bias(net_graph_->get_tensor_local_offset(in_tensors[2]));

//   out_param->set_if_relu(param.do_activation());
//   if (param.activation_arg_size() > 0) {
//     out_param->set_relu_slope(param.activation_arg(0));
//   }
// }

// void MixNet::_add_tl_eltwise_param(int layer_id, LayerParameter* layer, const ImLayer* im_layer,
//                                    const vector<int>& in_tensors, const vector<int>& out_tensors,
//                                    net_timestep* time_step, int timestep_idx, bool is_h_split) {
//   const TGEltwiseParameter& param = im_layer->op()->tg_eltwise_param();
//   const ImLayer* ir = net_graph_->get_layer_by_id(layer_id);
//   const Tensor* in_tensor = ir->in_tensors[0].get();

//   int bottom_dim[4];
//   net_graph_->get_tensor_dim(in_tensors[0], bottom_dim);

//   bottom_dim[0] = in_tensor->n_slice;
//   bottom_dim[2] = in_tensor->h_slice;

//   TLEltwiseParameter* out_param = layer->mutable_tl_eltwise_param();

//   if (CHIP_IS_BM188X) {
//     mem_buffer_key_t key = {timestep_idx, ir->imm_tensors[0].get()->id(), true};
//     const mem_buffer_value_t* imm = time_step->get_mem_buffer_value(&key);
//     out_param->set_working(imm->local_mem_offset);
//   }
//   assert(layer->input_shape_size() == 0);
//   for (int i = 0; i < in_tensors.size(); i++) {
//     out_param->add_input(net_graph_->get_tensor_local_offset(in_tensors[i]));
//     if (param.coeff_size() > 0) {
//       out_param->add_coeff(param.coeff(i));
//     }

//     BlobShape* input_shape = layer->add_input_shape();
//     input_shape->add_dim(bottom_dim[0]);
//     input_shape->add_dim(bottom_dim[1]);
//     input_shape->add_dim(bottom_dim[2]);
//     input_shape->add_dim(bottom_dim[3]);
//   }

//   out_param->set_output(net_graph_->get_tensor_local_offset(out_tensors[0]));
//   assert(layer->output_shape_size() == 0);
//   BlobShape* output_shape = layer->add_output_shape();
//   output_shape->add_dim(bottom_dim[0]);
//   output_shape->add_dim(bottom_dim[1]);
//   output_shape->add_dim(bottom_dim[2]);
//   output_shape->add_dim(bottom_dim[3]);

//   out_param->set_op_code(param.operation());
//   out_param->set_if_relu(param.do_activation());
//   if (param.activation_arg_size() > 0) {
//     out_param->set_relu_slope(param.activation_arg(0));
//   }
// }

// void MixNet::_add_tl_activation_param(LayerParameter* layer, const ImLayer* im_layer,
//                                       const vector<int>& in_tensors, const vector<int>& out_tensors,
//                                       net_timestep* time_step, int timestep_idx, bool is_h_split) {
//   const TGActivationParameter& param = im_layer->op()->tg_activation_param();
//   const Tensor* in_tensor = net_graph_->get_tensor_by_id(in_tensors[0]);

//   int bottom_dim[4];
//   net_graph_->get_tensor_dim(in_tensors[0], bottom_dim);

//   bottom_dim[0] = in_tensor->n_slice;
//   bottom_dim[2] = in_tensor->h_slice;

//   TLActivationParameter* out_param = layer->mutable_tl_activation_param();
//   out_param->set_input(net_graph_->get_tensor_local_offset(in_tensors[0]));

//   assert(layer->input_shape_size() == 0);
//   BlobShape* input_shape = layer->add_input_shape();
//   input_shape->add_dim(bottom_dim[0]);
//   input_shape->add_dim(bottom_dim[1]);
//   input_shape->add_dim(bottom_dim[2]);
//   input_shape->add_dim(bottom_dim[3]);

//   out_param->set_output(net_graph_->get_tensor_local_offset(out_tensors[0]));
//   assert(layer->output_shape_size() == 0);
//   BlobShape* output_shape = layer->add_output_shape();
//   output_shape->add_dim(bottom_dim[0]);
//   output_shape->add_dim(bottom_dim[1]);
//   output_shape->add_dim(bottom_dim[2]);
//   output_shape->add_dim(bottom_dim[3]);

//   out_param->set_activation(param.activation());
//   out_param->set_channel_shared(param.channel_shared());
//   out_param->set_global_slope(param.global_slope());
//   if (in_tensors.size() > 1) {
//     out_param->set_weight(net_graph_->get_tensor_local_offset(in_tensors[1]));
//   }

//   for (u32 i = 0; i < param.activation_arg_size(); i++) {
//     out_param->add_activation_arg(param.activation_arg(i));
//   }
// }

// void MixNet::add_transport_param_to_next_layer(const TENSOR_STEP& tensor, net_timestep* time_step,
//                                                int timestep_idx, bool current_stage) {
//   vector<TLTransportParameter*> params =
//       _add_transport_param(tensor, time_step, timestep_idx, current_stage);
//   suspend_transports_.insert(suspend_transports_.end(), params.begin(), params.end());
// }

// void MixNet::add_transport_param_to_last_layer(const TENSOR_STEP& tensor, net_timestep* time_step,
//                                                int timestep_idx, bool current_stage) {
//   LayerParameter* last_layer = out_net_.mutable_layer(out_net_.layer_size() - 1);
//   vector<TLTransportParameter*> params =
//       _add_transport_param(tensor, time_step, timestep_idx, current_stage);

//   for (int k = 0; k < params.size(); k++) {
//     TLTransportParameter* transport = last_layer->add_tl_transport_param();
//     transport->CopyFrom(*params[k]);
//     delete params[k];
//   }
// }

// vector<TLTransportParameter*> MixNet::_add_transport_param(const TENSOR_STEP& tensor,
//                                                            net_timestep* time_step,
//                                                            int timestep_idx, bool current_stage) {
//   vector<TLTransportParameter*> params;
//   int tensor_id = tensor.first;

//   if (tensor.second == TIMESTEP_LOAD) {
//     _add_load_param(params, tensor_id, time_step, timestep_idx);
//     if (!current_stage) {
//       for (int i = 0; i < params.size(); i++) {
//         params[i]->set_stage(TLTransportParameter::PRE);
//       }
//     }
//   } else if (tensor.second == TIMESTEP_DDR_TO_TSM) {
//     _add_ddr_to_tsm_i8(params, tensor_id, time_step, timestep_idx);
//     if (!current_stage) {
//       for (int i = 0; i < params.size(); i++) {
//         params[i]->set_stage(TLTransportParameter::PRE);
//       }
//     }
//   } else if (tensor.second == TIMESTEP_TSM_TO_LMEM) {
//     _add_tsm_to_lmem_i8(params, tensor_id, time_step, timestep_idx);
//     if (!current_stage) {
//       for (int i = 0; i < params.size(); i++) {
//         params[i]->set_stage(TLTransportParameter::PRE);
//       }
//     }
//   } else if (tensor.second == TIMESTEP_LMEM_TO_TSM) {
//     assert(false && "not support now");
//     exit(-1);

//   } else if (tensor.second == TIMESTEP_TSM_TO_DDR) {
//     assert(false && "not support now");
//     exit(-1);

//   } else if (tensor.second == TIMESTEP_STORE) {
//     _add_store_param(params, tensor_id, time_step, timestep_idx);
//     if (!current_stage) {
//       for (int i = 0; i < params.size(); i++) {
//         params[i]->set_stage(TLTransportParameter::POST);
//       }
//     }
//   }
//   return params;
// }

// void MixNet::_add_load_param(vector<TLTransportParameter*>& out_params, int tensor_id,
//                              net_timestep* time_step, int timestep_idx) {
//   if (CHIP_IS_BM1880) {
//     return _add_load_param_bm1880(out_params, tensor_id, time_step, timestep_idx);
//   }
//   else if (CHIP_IS_BM1880V2) {
//     return _add_load_param_bm1880v2(out_params, tensor_id, time_step, timestep_idx);
//   }

//   int tensor_dim[4];
//   const tensor_type_t tensor_type = net_graph_->get_tensor_type(tensor_id);
//   const Tensor* tensor = net_graph_->get_tensor_by_id(tensor_id);
//   net_graph_->get_tensor_dim(tensor_id, tensor_dim);
//   TLTransportParameter* out_param = new TLTransportParameter();

//   BlobShape* local_shape = out_param->mutable_local_shape();
//   BlobShape* global_shape = out_param->mutable_global_shape();

//   out_param->set_direction(TLTransportParameter::S2L);
//   out_param->set_name(tensor->name());

//   u64 global = net_graph_->get_tensor_global_mem(tensor_id);

//   if (tensor_type == TENSOR_COEFF) {
//     u32 local = net_graph_->get_tensor_local_offset(tensor_id);

//     local_shape->add_dim(1);
//     local_shape->add_dim(tensor_dim[1]);
//     local_shape->add_dim(tensor_dim[0]);
//     local_shape->add_dim(tensor_dim[2] * tensor_dim[3]);

//     global_shape->add_dim(1);
//     global_shape->add_dim(tensor_dim[1]);
//     global_shape->add_dim(tensor_dim[0]);
//     global_shape->add_dim(tensor_dim[2] * tensor_dim[3]);

//     CHECK(local != 0xFFFFFFFF) << "tensor:" << tensor_id;
//     out_param->set_local_address(local);
//     out_param->set_global_offset(global);
//     out_param->set_aligned(false);
//     out_param->set_transpose(false);
//     out_param->set_data_type(TLTransportParameter::COEFF);
//   } else {
//     int n_idx = tensor->n_idx;
//     int n_slice = tensor->n_slice;
//     int h_idx = tensor->h_idx;
//     int h_slice = tensor->h_slice;
//     int h_end = h_idx + h_slice;
//     h_idx = h_idx > 0 ? h_idx : 0;
//     h_slice = h_end > tensor_dim[2] ? (tensor_dim[2] - h_idx) : (h_end - h_idx);

//     mem_buffer_key_t key = {timestep_idx, tensor_id, false};
//     const mem_buffer_value_t* value = time_step->get_mem_buffer_value(&key);
//     u32 local = value->local_mem_offset;

//     global += (n_idx * tensor_dim[1] * tensor_dim[2] * tensor_dim[3] + h_idx * tensor_dim[3]) *
//               tensor->unit_size();

//     local_shape->add_dim(n_slice);
//     local_shape->add_dim(tensor_dim[1]);
//     local_shape->add_dim(h_slice);
//     local_shape->add_dim(tensor_dim[3]);

//     global_shape->add_dim(tensor_dim[0]);
//     global_shape->add_dim(tensor_dim[1]);
//     global_shape->add_dim(tensor_dim[2]);
//     global_shape->add_dim(tensor_dim[3]);

//     CHECK(local != 0xFFFFFFFF) << "tensor:" << tensor_id;
//     out_param->set_local_address(local);
//     out_param->set_global_offset(global);
//     if (tensor_type == TENSOR_NEURON || tensor_type == TENSOR_NEURON_WINOGRAD) {
//       out_param->set_aligned(true);
//     } else {  // TENSOR_COEFF_NEURON
//       if (tensor_type != TENSOR_NEURON_AS_COEFF) {
//         out_param->set_data_type(TLTransportParameter::COEFF);
//       }

//       out_param->set_aligned(false);
//     }

//     out_param->set_transpose(false);

//     net_graph_->set_tensor_local_offest(tensor_id, local);
//   }

//   out_params.push_back(out_param);
// }

// void MixNet::_add_load_param_bm1880(vector<TLTransportParameter*>& out_params, int tensor_id,
//                                     net_timestep* time_step, int timestep_idx) {
//   int tensor_dim[4];
//   const tensor_type_t tensor_type = net_graph_->get_tensor_type(tensor_id);
//   const Tensor* tensor = net_graph_->get_tensor_by_id(tensor_id);
//   net_graph_->get_tensor_dim(tensor_id, tensor_dim);
//   TLTransportParameter* out_param = new TLTransportParameter();

//   BlobShape* local_shape = out_param->mutable_local_shape();
//   BlobShape* global_shape = out_param->mutable_global_shape();

//   out_param->set_direction(TLTransportParameter::S2L);
//   out_param->set_name(tensor->name());

//   u64 global = net_graph_->get_tensor_global_mem(tensor_id);

//   if (tensor_type == TENSOR_COEFF) {
//     u32 local = net_graph_->get_tensor_local_offset(tensor_id);

//     local_shape->add_dim(1);
//     local_shape->add_dim(tensor_dim[1]);
//     local_shape->add_dim(tensor_dim[2] * tensor_dim[3]);
//     local_shape->add_dim(tensor_dim[0]);

//     global_shape->add_dim(1);
//     global_shape->add_dim(tensor_dim[1]);
//     global_shape->add_dim(tensor_dim[2] * tensor_dim[3]);
//     global_shape->add_dim(tensor_dim[0]);

//     out_param->set_local_address(local);
//     out_param->set_global_offset(global);
//     out_param->set_aligned(false);
//     out_param->set_transpose(false);
//     out_param->set_data_type(TLTransportParameter::COEFF);
//   } else if (tensor_type == TENSOR_COEFF_WINOGRAD) {
//     u32 local = net_graph_->get_tensor_local_offset(tensor_id);

//     local_shape->add_dim(tensor_dim[0]);
//     local_shape->add_dim(tensor_dim[1]);
//     local_shape->add_dim(1);
//     local_shape->add_dim(16);

//     global_shape->add_dim(tensor_dim[0]);
//     global_shape->add_dim(tensor_dim[1] * tensor->group);
//     global_shape->add_dim(1);
//     global_shape->add_dim(16);

//     out_param->set_local_address(local);
//     out_param->set_global_offset(global);
//     out_param->set_aligned(false);
//     out_param->set_transpose(false);
//     out_param->set_data_type(TLTransportParameter::COEFF);
//   } else if (tensor_type == TENSOR_BIAS) {
//     u32 local = net_graph_->get_tensor_local_offset(tensor_id);

//     local_shape->add_dim(tensor_dim[0]);
//     local_shape->add_dim(tensor_dim[1]);
//     local_shape->add_dim(tensor_dim[2]);
//     local_shape->add_dim(tensor_dim[3]);

//     global_shape->add_dim(tensor_dim[0]);
//     global_shape->add_dim(tensor_dim[1] * tensor->group);
//     global_shape->add_dim(tensor_dim[2]);
//     global_shape->add_dim(tensor_dim[3]);

//     out_param->set_local_address(local);
//     out_param->set_global_offset(global);
//     out_param->set_aligned(false);
//     out_param->set_transpose(false);
//     out_param->set_data_type(TLTransportParameter::COEFF);
//   } else if (tensor_type == TENSOR_DEPTHCONV_OPD1) {
//     u32 local = net_graph_->get_tensor_local_offset(tensor_id);

//     local_shape->add_dim(tensor_dim[0]);
//     local_shape->add_dim(tensor_dim[1]);
//     local_shape->add_dim(tensor_dim[2]);
//     local_shape->add_dim(tensor_dim[3]);

//     global_shape->add_dim(tensor_dim[0]);
//     global_shape->add_dim(tensor_dim[1]);
//     global_shape->add_dim(tensor_dim[2]);
//     global_shape->add_dim(tensor_dim[3]);

//     out_param->set_local_address(local);
//     out_param->set_global_offset(global);
//     out_param->set_aligned(true);
//     out_param->set_transpose(false);
//     out_param->set_data_type(TLTransportParameter::COEFF);
//   } else {
//     int n_idx = tensor->n_idx;
//     int n_slice = tensor->n_slice;
//     int h_idx = tensor->h_idx;
//     int h_slice = tensor->h_slice;
//     int h_end = h_idx + h_slice;
//     h_idx = h_idx > 0 ? h_idx : 0;
//     h_slice = h_end > tensor_dim[2] ? (tensor_dim[2] - h_idx) : (h_end - h_idx);

//     mem_buffer_key_t key = {timestep_idx, tensor_id, false};
//     const mem_buffer_value_t* value = time_step->get_mem_buffer_value(&key);
//     u32 local = value->local_mem_offset;

//     global += (n_idx * tensor_dim[1] * tensor_dim[2] * tensor_dim[3] + h_idx * tensor_dim[3]) *
//               tensor->unit_size();

//     local_shape->add_dim(n_slice);
//     local_shape->add_dim(tensor_dim[1]);
//     local_shape->add_dim(h_slice);
//     local_shape->add_dim(tensor_dim[3]);

//     global_shape->add_dim(tensor_dim[0]);
//     global_shape->add_dim(tensor_dim[1]);
//     global_shape->add_dim(tensor_dim[2]);
//     global_shape->add_dim(tensor_dim[3]);

//     out_param->set_local_address(local);
//     out_param->set_global_offset(global);
//     if (tensor_type == TENSOR_NEURON || tensor_type == TENSOR_NEURON_WINOGRAD) {
//       out_param->set_aligned(true);
//     } else {  // TENSOR_COEFF_NEURON
//       if (tensor_type != TENSOR_NEURON_AS_COEFF) {
//         out_param->set_data_type(TLTransportParameter::COEFF);
//       }

//       out_param->set_aligned(false);
//     }

//     out_param->set_transpose(false);

//     net_graph_->set_tensor_local_offest(tensor_id, local);
//   }

//   out_params.push_back(out_param);
// }


// void MixNet::_add_load_param_bm1880v2(vector<TLTransportParameter*>& out_params, int tensor_id,
//                                     net_timestep* time_step, int timestep_idx) {
//   int tensor_dim[4];
//   const tensor_type_t tensor_type = net_graph_->get_tensor_type(tensor_id);
//   const Tensor* tensor = net_graph_->get_tensor_by_id(tensor_id);
//   net_graph_->get_tensor_dim(tensor_id, tensor_dim);
//   TLTransportParameter* out_param = new TLTransportParameter();

//   BlobShape* local_shape = out_param->mutable_local_shape();
//   BlobShape* global_shape = out_param->mutable_global_shape();

//   out_param->set_direction(TLTransportParameter::S2L);
//   out_param->set_name(tensor->name());

//   u64 global = net_graph_->get_tensor_global_mem(tensor_id);

//   if (tensor_type == TENSOR_COEFF) {
//     u32 local = net_graph_->get_tensor_local_offset(tensor_id);

//     local_shape->add_dim(1);
//     local_shape->add_dim(tensor_dim[1]);
//     local_shape->add_dim(tensor_dim[2] * tensor_dim[3]);
//     local_shape->add_dim(tensor_dim[0]);

//     global_shape->add_dim(1);
//     global_shape->add_dim(tensor_dim[1]);
//     global_shape->add_dim(tensor_dim[2] * tensor_dim[3]);
//     global_shape->add_dim(tensor_dim[0]);

//     out_param->set_local_address(local);
//     out_param->set_global_offset(global);
//     out_param->set_aligned(false);
//     out_param->set_transpose(false);
//     out_param->set_data_type(TLTransportParameter::COEFF);
//   } else if (tensor_type == TENSOR_BIAS) {
//     u32 local = net_graph_->get_tensor_local_offset(tensor_id);

//     local_shape->add_dim(tensor_dim[0]);
//     local_shape->add_dim(tensor_dim[1]);
//     local_shape->add_dim(tensor_dim[2]);
//     local_shape->add_dim(tensor_dim[3]);

//     global_shape->add_dim(tensor_dim[0]);
//     global_shape->add_dim(tensor_dim[1] * tensor->group);
//     global_shape->add_dim(tensor_dim[2]);
//     global_shape->add_dim(tensor_dim[3]);

//     out_param->set_local_address(local);
//     out_param->set_global_offset(global);
//     out_param->set_aligned(false);
//     out_param->set_transpose(false);
//     out_param->set_data_type(TLTransportParameter::COEFF);
//   } else if (tensor_type == TENSOR_DEPTHCONV_OPD1) {
//     u32 local = net_graph_->get_tensor_local_offset(tensor_id);

//     local_shape->add_dim(tensor_dim[0]);
//     local_shape->add_dim(tensor_dim[1]);
//     local_shape->add_dim(tensor_dim[2]);
//     local_shape->add_dim(tensor_dim[3]);

//     global_shape->add_dim(tensor_dim[0]);
//     global_shape->add_dim(tensor_dim[1]);
//     global_shape->add_dim(tensor_dim[2]);
//     global_shape->add_dim(tensor_dim[3]);

//     out_param->set_local_address(local);
//     out_param->set_global_offset(global);
//     out_param->set_aligned(true);
//     out_param->set_transpose(false);
//     out_param->set_data_type(TLTransportParameter::COEFF);
//   } else {
//     int n_idx = tensor->n_idx;
//     int n_slice = tensor->n_slice;
//     int h_idx = tensor->h_idx;
//     int h_slice = tensor->h_slice;
//     int h_end = h_idx + h_slice;
//     h_idx = h_idx > 0 ? h_idx : 0;
//     h_slice = h_end > tensor_dim[2] ? (tensor_dim[2] - h_idx) : (h_end - h_idx);

//     mem_buffer_key_t key = {timestep_idx, tensor_id, false};
//     const mem_buffer_value_t* value = time_step->get_mem_buffer_value(&key);
//     u32 local = value->local_mem_offset;

//     global += (n_idx * tensor_dim[1] * tensor_dim[2] * tensor_dim[3] + h_idx * tensor_dim[3]) *
//               tensor->unit_size();

//     local_shape->add_dim(n_slice);
//     local_shape->add_dim(tensor_dim[1]);
//     local_shape->add_dim(h_slice);
//     local_shape->add_dim(tensor_dim[3]);

//     global_shape->add_dim(tensor_dim[0]);
//     global_shape->add_dim(tensor_dim[1]);
//     global_shape->add_dim(tensor_dim[2]);
//     global_shape->add_dim(tensor_dim[3]);

//     out_param->set_local_address(local);
//     out_param->set_global_offset(global);
//     if (tensor_type == TENSOR_NEURON || tensor_type == TENSOR_NEURON_WINOGRAD) {
//       out_param->set_aligned(true);
//     } else {  // TENSOR_COEFF_NEURON
//       if (tensor_type != TENSOR_NEURON_AS_COEFF) {
//         out_param->set_data_type(TLTransportParameter::COEFF);
//       }

//       out_param->set_aligned(false);
//     }

//     out_param->set_transpose(false);

//     net_graph_->set_tensor_local_offest(tensor_id, local);
//   }

//   out_params.push_back(out_param);
// }

// void MixNet::_add_store_param(vector<TLTransportParameter*>& out_params, int tensor_id,
//                               net_timestep* time_step, int timestep_idx) {
//   int tensor_dim[4];
//   const Tensor* tensor = net_graph_->get_tensor_by_id(tensor_id);
//   const vector<int>& dst_layers = net_graph_->get_tensor_to_layer(tensor_id);
//   u32 local = net_graph_->get_tensor_local_offset(tensor_id);
//   u64 global = net_graph_->get_tensor_global_mem(tensor_id);
//   int gmem_to_be_allocated = 0;

//   net_graph_->get_tensor_dim(tensor_id, tensor_dim);

//   int n_idx = tensor->n_idx;
//   int n_slice = tensor->n_slice;
//   int h_idx = tensor->h_idx;
//   int h_slice = tensor->h_slice;
//   int h_end = h_idx + h_slice;
//   h_idx = h_idx > 0 ? h_idx : 0;
//   h_slice = h_end > tensor_dim[2] ? (tensor_dim[2] - h_idx) : (h_end - h_idx);

//   std::vector<TLTransportParameter*> cat_out_vec;
//   if (!dst_layers.empty()) {
//     for (int k = 0; k < dst_layers.size(); k++) {
//       int target_layer = dst_layers[k];
//       const ImLayer* im_layer = net_graph_->get_layer_by_id(target_layer);

//       if (net_graph_->is_concat_optimized_case(im_layer->id(), -1)) {
//         int concat_out_tensor = net_graph_->get_out_tensors_of_layer(target_layer)[0];
//         const vector<int>& concat_in_tensors = net_graph_->get_in_tensors_of_layer(target_layer);
//         int c_idx = 0;
//         int concat_dim[4];
//         TLTransportParameter* out_param = new TLTransportParameter();
//         cat_out_vec.push_back(out_param);

//         out_param->set_direction(TLTransportParameter::L2S);
//         out_param->set_name(tensor->name());

//         BlobShape* local_shape = out_param->mutable_local_shape();
//         BlobShape* global_shape = out_param->mutable_global_shape();

//         local_shape->add_dim(n_slice);
//         local_shape->add_dim(tensor_dim[1]);
//         local_shape->add_dim(h_slice);
//         local_shape->add_dim(tensor_dim[3]);

//         for (u32 i = 0; i < concat_in_tensors.size(); ++i) {
//           if (concat_in_tensors[i] == tensor_id) {
//             break;
//           }
//           c_idx += net_graph_->get_tensor_channels(concat_in_tensors[i]);
//         }

//         net_graph_->get_tensor_dim(concat_out_tensor, concat_dim);
//         global_shape->add_dim(concat_dim[0]);
//         global_shape->add_dim(concat_dim[1]);
//         global_shape->add_dim(concat_dim[2]);
//         global_shape->add_dim(concat_dim[3]);

//         u64 concat_global = net_graph_->get_tensor_global_mem(concat_out_tensor);

//         concat_global += (n_idx * concat_dim[1] * concat_dim[2] * concat_dim[3] +
//                           c_idx * concat_dim[2] * concat_dim[3] + h_idx * concat_dim[3]) *
//                          tensor->unit_size();

//         out_param->set_local_address(local);
//         out_param->set_global_offset(concat_global);
//         out_param->set_aligned(true);
//         out_param->set_transpose(false);
//       } else if (!gmem_to_be_allocated) {
//         /*
//           we have another simple way to assert target_layer
//           is beyond current cluster by (gaddr==TBD_GADDR);
//           but it seems gaddr has only int32 in some platform,
//           that cause this comparison failed. Need further
//           investigation
//           */
//         int step_num = time_step->get_timestep_num();
//         int step;
//         for (step = 0; step < step_num; step++) {
//           if (target_layer <= time_step->get_layer(step)) {
//             break;
//           }
//         }
//         if (step == step_num) {
//           gmem_to_be_allocated = 1;
//         }
//       }
//     }
//   }

//   if (dst_layers.empty() || gmem_to_be_allocated) {
//     TLTransportParameter* out_param = new TLTransportParameter();
//     out_params.push_back(out_param);

//     out_param->set_direction(TLTransportParameter::L2S);
//     out_param->set_name(tensor->name());

//     BlobShape* local_shape = out_param->mutable_local_shape();
//     BlobShape* global_shape = out_param->mutable_global_shape();

//     local_shape->add_dim(n_slice);
//     local_shape->add_dim(tensor_dim[1]);
//     local_shape->add_dim(h_slice);
//     local_shape->add_dim(tensor_dim[3]);

//     global_shape->add_dim(tensor_dim[0]);
//     global_shape->add_dim(tensor_dim[1]);
//     global_shape->add_dim(tensor_dim[2]);
//     global_shape->add_dim(tensor_dim[3]);

//     global += (n_idx * tensor_dim[1] * tensor_dim[2] * tensor_dim[3] + h_idx * tensor_dim[3]) *
//               tensor->unit_size();

//     out_param->set_local_address(local);
//     out_param->set_global_offset(global);
//     out_param->set_aligned(true);
//     out_param->set_transpose(false);

//     for (auto it = cat_out_vec.begin(); it != cat_out_vec.end(); ) {
//       if ((*it)->global_offset() == out_param->global_offset()) {
//         it = cat_out_vec.erase(it);
//       } else {
//         ++it;
//       }
//     }
//   }

//   for (auto cat_unit : cat_out_vec) {
//     out_params.push_back(cat_unit);
//   }

//   // TODO, need to handle special case for concat and reorg layer.
// }

// void MixNet::_add_tl_arithmetic_param(LayerParameter* layer, const ImLayer* im_layer,
//                                       const vector<int>& in_tensors, const vector<int>& out_tensors,
//                                       net_timestep* time_step, int timestep_idx, bool is_h_split) {
//   const TGArithmeticParameter& param = im_layer->op()->tg_arithmetic_param();
//   const Tensor* in_tensor = net_graph_->get_tensor_by_id(in_tensors[0]);

//   int bottom_dim[4];
//   net_graph_->get_tensor_dim(in_tensors[0], bottom_dim);

//   bottom_dim[0] = in_tensor->n_slice;
//   bottom_dim[2] = in_tensor->h_slice;

//   TLArithmeticParameter* out_param = layer->mutable_tl_arithmetic_param();
//   out_param->set_input(net_graph_->get_tensor_local_offset(in_tensors[0]));
//   out_param->set_output(net_graph_->get_tensor_local_offset(out_tensors[0]));

//   assert(layer->input_shape_size() == 0);
//   BlobShape* input_shape = layer->add_input_shape();
//   input_shape->add_dim(bottom_dim[0]);
//   input_shape->add_dim(bottom_dim[1]);
//   input_shape->add_dim(bottom_dim[2]);
//   input_shape->add_dim(bottom_dim[3]);

//   assert(layer->output_shape_size() == 0);
//   BlobShape* output_shape = layer->add_output_shape();
//   output_shape->add_dim(bottom_dim[0]);
//   output_shape->add_dim(bottom_dim[1]);
//   output_shape->add_dim(bottom_dim[2]);
//   output_shape->add_dim(bottom_dim[3]);

//   out_param->set_global_b(param.global_b());
//   out_param->set_operation(static_cast<TLArithmeticParameter_ArithmeticOp>(param.operation()));
//   out_param->set_is_b_const(param.is_b_const());

//   if (!param.is_b_const()) {
//     int b_shape_dim[4];
//     net_graph_->get_tensor_dim(in_tensors[1], b_shape_dim);
//     BlobShape* b_shape = out_param->mutable_b_shape();
//     b_shape->add_dim(b_shape_dim[0]);
//     b_shape->add_dim(b_shape_dim[1]);
//     b_shape->add_dim(b_shape_dim[2]);
//     b_shape->add_dim(b_shape_dim[3]);
//     out_param->set_global_b(net_graph_->get_tensor_local_offset(in_tensors[1]));
//   }
// }

// void MixNet::_add_tl_quantization_param(LayerParameter* layer, const ImLayer* im_layer,
//                                       const vector<int>& in_tensors, const vector<int>& out_tensors,
//                                       net_timestep* time_step, int timestep_idx, bool is_h_split) {
//   const TGQuantizationParameter& param = im_layer->op()->tg_quantization_param();
//   const Tensor* in_tensor = net_graph_->get_tensor_by_id(in_tensors[0]);

//   int bottom_dim[4];
//   net_graph_->get_tensor_dim(in_tensors[0], bottom_dim);

//   bottom_dim[0] = in_tensor->n_slice;
//   bottom_dim[2] = in_tensor->h_slice;

//   TLQuantizationParameter* out_param = layer->mutable_tl_quantization_param();
//   out_param->set_operation(param.operation());
//   out_param->set_input(net_graph_->get_tensor_local_offset(in_tensors[0]));
//   out_param->set_output(net_graph_->get_tensor_local_offset(out_tensors[0]));

//   assert(layer->input_shape_size() == 0);
//   BlobShape* input_shape = layer->add_input_shape();
//   input_shape->add_dim(bottom_dim[0]);
//   input_shape->add_dim(bottom_dim[1]);
//   input_shape->add_dim(bottom_dim[2]);
//   input_shape->add_dim(bottom_dim[3]);

//   assert(layer->output_shape_size() == 0);
//   BlobShape* output_shape = layer->add_output_shape();
//   output_shape->add_dim(bottom_dim[0]);
//   output_shape->add_dim(bottom_dim[1]);
//   output_shape->add_dim(bottom_dim[2]);
//   output_shape->add_dim(bottom_dim[3]);
// }

}
