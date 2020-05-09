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
  void set_name(string name) { op_name_ = name; };
  string name() { return op_name_; }
  void set_param();
  void set_type(string type) { type_ = type; }
  string bottom_name(int idx) {
    assert(idx < operands_.size());
    return operands_[idx];
  }
  string top_name(int idx) {
    assert(idx < results_.size());
    return results_[idx];
  }

  void add_bottom_name(string bottom_name);
  void add_top_name(string top_name);
  int get_layer_id() { return layer_id_; }

private:
  vector<string> operands_;
  vector<string> results_;
  string op_name_;
  string type_;
  int layer_id_;
  MixNet * mix_net_;
};

class MixNet {
 public:
  explicit MixNet(NetGraph* net_graph, FuncOp * fn, MLIRContext * context);

  Value* get_op_from_name(string name);
  void add_opd_to_list(string op_name, Value * opd, bool);
  void set_net_in_tensor(int tensor_id);
  void set_net_out_tensor(int tensor_id);
  void add_group_start_ops(int group_idx, Group* group,
                           Operation * op, int n_secs, int h_secs);
  void add_group_end_ops(int group_idx, Group* cluster,
                         int n_secs, int h_secs);
  void add_tl_layer(int group_idx, int layer_id,
                    net_timestep* time_step, int timestep_idx,
                    bool is_h_split, int n_loop, int h_loop);
  void add_transport_op(const TENSOR_STEP& tensor,
                              net_timestep* time_step, int timestep_idx);


  // parallel start and end
  void parallel_start();
  void parallel_end();

  FuncOp* get_net() { return fn_; }
  void set_start_op(Operation * op) { start_op_ = op; }
  Operation* get_start_op() { return start_op_; }

 private:
  void _add_tl_convolution_op(MixOp* mix_op,
                              const vector<int>& in_tensors,
                              const vector<int>& out_tensors,
                              net_timestep* time_step,
                              int timestep_idx, bool is_h_split);

  void _add_tl_pooling_op(MixOp * mix_op,
                          const vector<int>& in_tensors,
                          const vector<int>& out_tensors,
                          net_timestep* time_step,
                          int timestep_idx, bool is_h_split);

  void _add_tl_eltwise_op(MixOp* mix_op,
                          const vector<int>& in_tensors,
                          const vector<int>& out_tensors,
                          net_timestep* time_step,
                          int timestep_idx, bool is_h_split);

  void _add_tl_lrn_op(MixOp * mix_op,
                      const vector<int>& in_tensors,
                      const vector<int>& out_tensors,
                      net_timestep* time_step,
                      int timestep_idx, bool is_h_split);

  void _add_load_op(int tensor_id,
                    net_timestep* time_step, int timestep_idx);

  void _add_store_op(int tensor_id,
                     net_timestep* time_step, int timestep_idx);

  NetGraph* net_graph_;
  vector<int> net_in_tensors_;
  vector<int> net_out_tensors_;
  FuncOp * fn_;
  MLIRContext * context_;
  map<string, Value *> name_op_map_;
  vector<Operation *> parallel_list_;
  Operation * start_op_;
  Operation * weightFileOp_;
};

}
#endif
