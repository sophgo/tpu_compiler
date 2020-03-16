/*
 * Copyright (C) Bitmain Technologies Inc.
 * All Rights Reserved.
 */
#ifndef CLUSTER_OUT_GRAPH_H
#define CLUSTER_OUT_GRAPH_H

#include "utils.hpp"
#include "NetGraph.hpp"

namespace mlir {

class MixNet {
 public:
  explicit MixNet(NetGraph* net_graph) : net_graph_(net_graph), out_net_() {}

  void set_fn(FuncOp * fn, MLIRContext * context) { fn_ = fn; context_ = context;}
  void set_net_in_tensor(int tensor_id);
  void set_net_out_tensor(int tensor_id);
  // void add_start_layer(const std::string input_name, const int data_type_size);
  // void add_end_layer(u64 neuron_size);
  // void add_group_start_layer(int group_idx, Group* cluster, int n_secs, int h_secs);
  // void add_group_end_layer(int group_idx, Group* cluster, int n_secs, int h_secs);
  void add_tg_layer(int layer_id);
  // void add_tl_layer(int group_idx, int layer_id, net_timestep* time_step, int timestep_idx,
  //                   bool is_h_split, int n_loop, int h_loop);
  // void add_transport_param_to_next_layer(const TENSOR_STEP& tensor, net_timestep* time_step,
  //                                        int timestep_idx, bool current_stage);
  // void add_transport_param_to_last_layer(const TENSOR_STEP& tensor, net_timestep* time_step,
  //                                        int timestep_idx, bool current_stage);

  Operation* get_net() { return out_net_; }

 private:
  // void _add_tl_convolution_param(int layer_id, LayerParameter* layer, const ImLayer* im_layer,
  //                                const vector<int>& in_tensors, const vector<int>& out_tensors,
  //                                net_timestep* time_step, int timestep_idx, bool is_h_split);

  // void _add_tl_deconvolution_param(int layer_id, LayerParameter* layer, const ImLayer* im_layer,
  //                                  const vector<int>& in_tensors, const vector<int>& out_tensors,
  //                                  net_timestep* time_step, int timestep_idx, bool is_h_split);

  // void _add_tl_pooling_param(LayerParameter* layer, const ImLayer* im_layer,
  //                            const vector<int>& in_tensors, const vector<int>& out_tensors,
  //                            net_timestep* time_step, int timestep_idx, bool is_h_split);

  // void _add_tl_upsample_param(LayerParameter* layer, const ImLayer* im_layer,
  //                             const vector<int>& in_tensors, const vector<int>& out_tensors,
  //                             net_timestep* time_step, int timestep_idx, bool is_h_split);

  // void _add_tl_lrn_param(int layer_id, LayerParameter* layer, const ImLayer* im_layer,
  //                        const vector<int>& in_tensors, const vector<int>& out_tensors,
  //                        net_timestep* time_step, int timestep_idx, bool is_h_split);

  // void _add_tl_batchnorm_param(LayerParameter* layer, const ImLayer* im_layer,
  //                              const vector<int>& in_tensors, const vector<int>& out_tensors,
  //                              net_timestep* time_step, int timestep_idx, bool is_h_split);

  // void _add_tl_scale_param(LayerParameter* layer, const ImLayer* im_layer,
  //                          const vector<int>& in_tensors, const vector<int>& out_tensors,
  //                          net_timestep* time_step, int timestep_idx, bool is_h_split);

  // void _add_tl_mac_param(LayerParameter* layer, const ImLayer* im_layer,
  //                        const vector<int>& in_tensors, const vector<int>& out_tensors,
  //                        net_timestep* time_step, int timestep_idx, bool is_h_split);

  // void _add_tl_innerproduct_param(LayerParameter* layer, const ImLayer* im_layer,
  //                                 const vector<int>& in_tensors, const vector<int>& out_tensors,
  //                                 net_timestep* time_step, int timestep_idx, bool is_h_split);

  // void _add_tl_eltwise_param(int layer_id, LayerParameter* layer, const ImLayer* im_layer,
  //                            const vector<int>& in_tensors, const vector<int>& out_tensors,
  //                            net_timestep* time_step, int timestep_idx, bool is_h_split);

  // void _add_tl_activation_param(LayerParameter* layer, const ImLayer* im_layer,
  //                               const vector<int>& in_tensors, const vector<int>& out_tensors,
  //                               net_timestep* time_step, int timestep_idx, bool is_h_split);

  // void _add_tl_shuffle_channel_param(LayerParameter* layer, const ImLayer* im_layer,
  //                                    const vector<int>& in_tensors, const vector<int>& out_tensors,
  //                                    net_timestep* time_step, int timestep_idx, bool is_h_split);

  // void _add_tl_arithmetic_param(int layer_id, LayerParameter* layer, const ImLayer* im_layer,
  //                               const vector<int>& in_tensors, const vector<int>& out_tensors,
  //                               net_timestep* time_step, int timestep_idx, bool is_h_split);

  // vector<TLTransportParameter*> _add_transport_param(const TENSOR_STEP& tensor,
  //                                                    net_timestep* time_step, int timestep_idx,
  //                                                    bool current_stage);

  // void _add_load_param(vector<TLTransportParameter*>& out_param, int tensor_id,
  //                      net_timestep* time_step, int timestep_idx);

  // void _add_load_param_bm1880(vector<TLTransportParameter*>& out_param, int tensor_id,
  //                             net_timestep* time_step, int timestep_idx);

  // void _add_load_param_bm1880v2(vector<TLTransportParameter*>& out_param, int tensor_id,
  //                               net_timestep* time_step, int timestep_idx);

  // void _add_store_param(vector<TLTransportParameter*>& out_param, int tensor_id,
  //                       net_timestep* time_step, int timestep_idx);

  // void _add_tl_arithmetic_param(LayerParameter* layer, const ImLayer* im_layer,
  //                               const vector<int>& in_tensors, const vector<int>& out_tensors,
  //                               net_timestep* time_step, int timestep_idx, bool is_h_split);

  // void _add_tl_quantization_param(LayerParameter* layer, const ImLayer* im_layer,
  //                                 const vector<int>& in_tensors, const vector<int>& out_tensors,
  //                                 net_timestep* time_step, int timestep_idx, bool is_h_split);

  // void _add_tsm_to_ddr_i8(vector<TLTransportParameter*>& out_params, int tensor_id,
  //                         net_timestep* time_step, int timestep_idx);

  // void _add_lmem_to_tsm_i8(vector<TLTransportParameter*>& out_params, int tensor_id,
  //                          net_timestep* time_step, int timestep_idx);

  // void _add_ddr_to_tsm_i8(vector<TLTransportParameter*>& out_params, int tensor_id,
  //                         net_timestep* time_step, int timestep_idx);

  // void _add_tsm_to_lmem_i8(vector<TLTransportParameter*>& out_params, int tensor_id,
  //                          net_timestep* time_step, int timestep_idx);

  NetGraph* net_graph_;
  Operation *out_net_;
  // vector<TLTransportParameter*> suspend_transports_;
  vector<int> net_in_tensors_;
  vector<int> net_out_tensors_;
  FuncOp * fn_;
  MLIRContext * context_;
};

}
#endif
