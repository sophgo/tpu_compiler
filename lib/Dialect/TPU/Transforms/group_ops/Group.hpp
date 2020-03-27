/*
 * Copyright (C) Bitmain Technologies Inc.
 * All Rights Reserved.
 */
#ifndef GROUPOPS_GROUP_H
#define GROUPOPS_GROUP_H

// #include "cluster_common.hpp"
#include "ImLayer.hpp"
// #include "layer_stage.hpp"
#include "NetGraph.hpp"
#include "Tensor.hpp"
#include "utils.hpp"
#include "Steps.hpp"

namespace mlir {

class net_timestep;

class Group {
 public:
  explicit Group(NetGraph *net_graph) : time_step(nullptr),
              net_graph_(net_graph), layers_(), lowered_(false) {}

  Group(NetGraph *net_graph, vector<int> layers)
      : time_step(nullptr), net_graph_(net_graph), layers_(layers), lowered_(false) {}

  Group(NetGraph *net_graph, vector<int>::iterator begin, vector<int>::iterator end)
      : time_step(nullptr), net_graph_(net_graph), layers_(), lowered_(false) {
    layers_.assign(begin, end);
  }

  ~Group();

  void set_lowered(bool f) { lowered_ = f; }

  bool lowered() { return lowered_; }

  void append(int layer_id) { layers_.push_back(layer_id); }

  bool empty() { return layers_.empty(); }

  vector<int>::iterator begin() { return layers_.begin(); }

  const vector<int> &layers() const { return layers_; }

  int size() const { return static_cast<int>(layers_.size()); }

  int get_batch_num() const;

  vector<int> get_group_out_tensors();

  set<int> get_group_in_neuron_tensors();

  bool check_valid();

  bmerr_t update_tensor_slices(int nsecs, int hsecs, int nslice_idx = -1, int hslice_idx = -1);

  bmerr_t assign_steps();
  // bmerr_t assign_steps_with_tsm();
  bmerr_t assign_steps_without_tsm();

  bool is_group_out_tensor(int tensor_id);

  void show_group();
  bool is_group_in_neuron_tensor(int tensor_id);
  void set_group_id(int group_id) { group_id_ = group_id; }
  int get_group_id() { return group_id_; }
  void clear_temp_data();
  void print(std::ostream &pOs) const;

  net_timestep *time_step;
  pair<int, int> nsecs_and_hsecs;

 private:
  NetGraph *net_graph_;
  vector<int> layers_;
  bool lowered_;
  int group_id_;

  bool validate_tensor_slice();

  void reset_tensor_slice();
  void reset_tensor_hslice_max();

  bool backward_slice(int out_tensor_id, list<int> &branches,
                      bool max_h_slice, bool no_split_h, int n_loop, int h_loop);

  bool group_has_winograd_tensors();
  bmerr_t group_winograd_out_tensors_check();
};

}
#endif
