/*
 * Copyright (C) Bitmain Technologies Inc.
 * All Rights Reserved.
 */
#ifndef GROUPOPS_GROUP_H
#define GROUPOPS_GROUP_H

#include "ImLayer.hpp"
#include "NetGraph.hpp"
#include "Tensor.hpp"
#include "utils.hpp"
#include "Steps.hpp"

namespace mlir {

class net_timestep;

class Group {
 public:
  explicit Group(NetGraph *net_graph) : time_step(nullptr),
              net_graph_(net_graph), layers_(), lowered_(false),
              slice_limit_(USE_FIT_H_SLICE),
              slice_dim_(LG_Slice_Dim_H) {}

  Group(NetGraph *net_graph, std::vector<int> layers)
      : time_step(nullptr), net_graph_(net_graph), layers_(layers),
        lowered_(false), slice_limit_(USE_FIT_H_SLICE),
        slice_dim_(LG_Slice_Dim_H) {}

  Group(NetGraph *net_graph, std::vector<int>::iterator begin,
        std::vector<int>::iterator end)
      : time_step(nullptr), net_graph_(net_graph), layers_(),
        lowered_(false), slice_limit_(USE_FIT_H_SLICE),
        slice_dim_(LG_Slice_Dim_H) {
    layers_.assign(begin, end);
  }

  ~Group();

  void set_lowered(bool f) { lowered_ = f; }

  bool lowered() { return lowered_; }

  void append(int layer_id) { layers_.push_back(layer_id); }

  bool empty() { return layers_.empty(); }

  std::vector<int>::iterator begin() { return layers_.begin(); }

  const std::vector<int> &layers() const { return layers_; }

  int size() const { return static_cast<int>(layers_.size()); }

  int get_batch_num() const;

  int get_max_secs();

  std::vector<int> get_group_out_tensors();

  std::set<int> get_group_in_neuron_tensors();

  bool check_valid();
  bool check_valid_wrap();

  bool check_if_pattern_support();

  bool check_if_can_slice_group();

  bmerr_t update_slices(int nsecs, int hsecs, int nslice_idx = -1, int hslice_idx = -1);
  bmerr_t update_nw_slices(int nsecs, int hsecs, int nslice_idx = -1, int hslice_idx = -1);
  bmerr_t update_nh_slices(int nsecs, int hsecs, int nslice_idx = -1, int hslice_idx = -1);

  bmerr_t assign_steps();
  // bmerr_t assign_steps_with_tsm();
  bmerr_t assign_steps_without_tsm();

  bool is_group_inside_tensor(int tid);

  bool is_group_out_tensor(int tensor_id);

  void show_group();

  void show_group_layers();

  bool is_group_in_neuron_tensor(int tensor_id);

  void set_group_id(int group_id) { group_id_ = group_id; }

  int get_group_id() { return group_id_; }

  void set_slice_limit(int s);

  void set_slice_dim(LG_Slice_Dim slice_dim);

  void clear_temp_data();

  void print(std::ostream &pOs) const;

  net_timestep *time_step;
  // layer slice info, now only support
  // n slice + h slice or n slice + w slice
  std::pair<int, int> group_slice_;

 private:
  NetGraph *net_graph_;
  std::vector<int> layers_;
  bool lowered_;
  int group_id_;
  LG_Slice_Limit slice_limit_;
  LG_Slice_Dim slice_dim_;

  bool validate_nh_slice();
  bool validate_nw_slice();

  void reset_tensor_slice();

  void reset_tensor_hwslice_max();

  bool backward_nh_slice(int out_tensor_id, std::list<int> &branches,
                      bool max_h_slice, bool no_split_h, int n_loop, int h_loop);

  bool backward_nw_slice(int out_tensor_id, std::list<int> &branches,
                      bool max_h_slice, bool no_split_h, int n_loop, int h_loop);
};

}
#endif
