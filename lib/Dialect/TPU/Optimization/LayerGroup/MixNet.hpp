/*
 * Copyright (C) Bitmain Technologies Inc.
 * All Rights Reserved.
 */
#ifndef CLUSTER_OUT_GRAPH_H
#define CLUSTER_OUT_GRAPH_H

#include "utils.hpp"
#include "NetGraph.hpp"
#include "Group.hpp"

namespace mlir {

class MixNet;
class MixOp {
public:
  MixOp(MixNet * net, int layer_id) {
    mix_net_ = net;
    layer_id_ = layer_id;
  }
  void set_name(std::string name) { op_name_ = name; };
  std::string name() { return op_name_; }
  void set_param();
  void set_type(std::string type) { type_ = type; }
  std::string bottom_name(int idx) {
    assert(idx < (int)operands_.size());
    return operands_[idx];
  }
  std::string top_name(int idx) {
    assert(idx < (int)results_.size());
    return results_[idx];
  }

  void add_bottom_name(std::string bottom_name);
  void add_top_name(std::string top_name);
  int get_layer_id() { return layer_id_; }

private:
  std::vector<std::string> operands_;
  std::vector<std::string> results_;
  std::string op_name_;
  std::string type_;
  int layer_id_;
  MixNet * mix_net_;
};

class MixNet {
 public:
  explicit MixNet(NetGraph* net_graph, FuncOp * fn, MLIRContext * context);

  Value* get_op_from_name(std::string name);
  void add_opd_to_list(std::string op_name, Value * opd, bool);
  void set_net_in_tensor(int tensor_id);
  void set_net_out_tensor(int tensor_id);
  void add_group_start_ops(int group_idx, Group* group,
                           Operation * op, int n_secs, int h_secs);
  void add_group_end_ops(int group_idx, Group* cluster,
                         int n_secs, int h_secs);
  void add_tl_layer(int group_idx, int layer_id,
                    net_timestep* time_step, int timestep_idx,
                    bool is_h_split, int n_loop, int h_loop);
  void add_transport_op(int group_idx, const TENSOR_STEP& tensor,
                        net_timestep* time_step, int timestep_idx);


  // parallel start and end
  void parallel_start();
  void parallel_end();

  FuncOp* get_net() { return fn_; }
  void set_start_op(Operation * op) { start_op_ = op; }
  Operation* get_start_op() { return start_op_; }

 private:
  void _add_tl_convolution_op(MixOp* mix_op,
                              const std::vector<int>& in_tensors,
                              const std::vector<int>& out_tensors,
                              net_timestep* time_step,
                              int timestep_idx, bool is_h_split);

  void _add_tl_deconvolution_op(MixOp* mix_op,
                                const std::vector<int>& in_tensors,
                                const std::vector<int>& out_tensors,
                                net_timestep* time_step,
                                int timestep_idx, bool is_h_split);

  void _add_tl_pooling_op(MixOp * mix_op,
                          const std::vector<int>& in_tensors,
                          const std::vector<int>& out_tensors,
                          net_timestep* time_step,
                          int timestep_idx, bool is_h_split);

  void _add_tl_eltwise_op(MixOp* mix_op,
                          const std::vector<int>& in_tensors,
                          const std::vector<int>& out_tensors,
                          net_timestep* time_step,
                          int timestep_idx, bool is_h_split);

  void _add_tl_eltwise_add_op(MixOp* mix_op,
                              const std::vector<int>& in_tensors,
                              const std::vector<int>& out_tensors,
                              net_timestep* time_step,
                              int timestep_idx, bool is_h_split);

  void _add_tl_eltwise_mul_op(MixOp* mix_op,
                              const std::vector<int>& in_tensors,
                              const std::vector<int>& out_tensors,
                              net_timestep* time_step,
                              int timestep_idx, bool is_h_split);

  void _add_tl_lrn_op(MixOp * mix_op,
                      const std::vector<int>& in_tensors,
                      const std::vector<int>& out_tensors,
                      net_timestep* time_step,
                      int timestep_idx, bool is_h_split);

  void _add_tl_activation_op(MixOp * mix_op,
                             const std::vector<int>& in_tensors,
                             const std::vector<int>& out_tensors,
                             net_timestep* time_step,
                             int timestep_idx, bool is_h_split);

  void _add_tl_quant_op(MixOp * mix_op,
                             const std::vector<int>& in_tensors,
                             const std::vector<int>& out_tensors,
                             net_timestep* time_step,
                             int timestep_idx, bool is_h_split);
  void _add_tl_abs_op(MixOp * mix_op,
                       const std::vector<int>& in_tensors,
                       const std::vector<int>& out_tensors,
                       net_timestep* time_step,
                       int timestep_idx,
                       bool is_h_split);

  void _add_tl_broadcast_mul_op(MixOp * mix_op,
                                const std::vector<int>& in_tensors,
                                const std::vector<int>& out_tensors,
                                net_timestep* time_step,
                                int timestep_idx, bool is_h_split);

  void _add_tl_upsample_op(MixOp * mix_op,
                           const std::vector<int>& in_tensors,
                           const std::vector<int>& out_tensors,
                           net_timestep* time_step,
                           int timestep_idx,
                           bool is_h_split);

  void _add_tl_leaky_relu_op(MixOp * mix_op,
                            const std::vector<int>& in_tensors,
                            const std::vector<int>& out_tensors,
                            net_timestep* time_step,
                            int timestep_idx,
                            bool is_h_split);

  void _add_tl_sigmoid_op(MixOp * mix_op,
                          const std::vector<int>& in_tensors,
                          const std::vector<int>& out_tensors,
                          net_timestep* time_step,
                          int timestep_idx,
                          bool is_h_split);

  void _add_tl_prelu_op(MixOp * mix_op,
                        const std::vector<int>& in_tensors,
                        const std::vector<int>& out_tensors,
                        net_timestep* time_step,
                        int timestep_idx,
                        bool is_h_split);

  void _add_tl_concat_op(MixOp * mix_op,
                         const std::vector<int>& in_tensors,
                         const std::vector<int>& out_tensors,
                         net_timestep* time_step,
                         int timestep_idx,
                         bool is_h_split);

  void _add_tl_pad_op(MixOp * mix_op,
                      const std::vector<int>& in_tensors,
                      const std::vector<int>& out_tensors,
                      net_timestep* time_step,
                      int timestep_idx,
                      bool is_h_split);

  void _add_tl_crop_op(MixOp * mix_op,
                      const std::vector<int>& in_tensors,
                      const std::vector<int>& out_tensors,
                      net_timestep* time_step,
                      int timestep_idx,
                      bool is_h_split);

  void _add_tl_relu_op(MixOp * mix_op,
                       const std::vector<int>& in_tensors,
                       const std::vector<int>& out_tensors,
                       net_timestep* time_step,
                       int timestep_idx,
                       bool is_h_split);

  void _add_tl_zero_mask_op(MixOp * mix_op,
                            const std::vector<int>& in_tensors,
                            const std::vector<int>& out_tensors,
                            net_timestep* time_step,
                            int timestep_idx,
                            bool is_h_split);

  void _add_tl_slice_op(MixOp * mix_op,
                       const std::vector<int>& in_tensors,
                       const std::vector<int>& out_tensors,
                       net_timestep* time_step,
                       int timestep_idx,
                       bool is_h_split);


  void _add_load_op(int group_idx, int tensor_id,
                    net_timestep* time_step, int timestep_idx);

  void _add_store_op(int group_idx, int tensor_id,
                     net_timestep* time_step, int timestep_idx);

  NetGraph* net_graph_;
  std::vector<int> net_in_tensors_;
  std::vector<int> net_out_tensors_;
  FuncOp * fn_;
  MLIRContext * context_;
  std::map<std::string, Value *> name_op_map_;
  std::vector<Operation *> parallel_list_;
  Operation * start_op_;
  Operation * weightFileOp_;
};

}
#endif
