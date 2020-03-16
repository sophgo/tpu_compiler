/*
 * Copyright (C) Bitmain Technologies Inc.
 * All Rights Reserved.
 */
#ifndef GROUPOPS_STEPS_H
#define GROUPOPS_STEPS_H

#include "Group.hpp"
#include "ImLayer.hpp"
#include "LayerStage.hpp"
#include "NetGraph.hpp"
#include "Tensor.hpp"

namespace mlir {

using TensorStep = vector<int>;

class ClusterSteps {
 public:
  explicit ClusterSteps(NetGraph* net_graph)
      : net_graph_(net_graph), layers_(), loads_(), stores_(), max_step_num_(0) {}

  ~ClusterSteps() {
    layers_.clear();
    loads_.clear();
    tsm_to_lmem_.clear();
    ddr_to_tsm_.clear();
    lmem_to_tsm_.clear();
    tsm_to_ddr_.clear();
    stores_.clear();
  }

  void append(int layer, TensorStep& load_tensors, TensorStep& store_tensors);
  void insert(int layer, TensorStep& load_tensors, TensorStep& store_tensors, int pos = 0);
  void rearrange_steps();
  void assign(Group* cluster);
  void to_timestep(net_timestep* time_step);

  // TSM Support
  void assign_with_tsm(Group* cluster);
  void to_timestep_with_tsm(net_timestep* time_step);

  static void timestep_assgin(NetGraph* net_graph, Group* cluster, net_timestep* time_step);

  static void timestep_assign_with_tsm(NetGraph* net_graph, Group* cluster,
                                       net_timestep* time_step);

  static bmerr_t balance_gdma_bdc_steps(NetGraph* net_graph, Group* cluster,
                                        net_timestep* time_step,
                                        const pair<int, int>& nsecs_and_hsec);

 private:
  NetGraph* net_graph_;
  vector<int> layers_;
  vector<TensorStep> loads_;
  vector<TensorStep> stores_;
  vector<TensorStep> tsm_to_lmem_;
  vector<TensorStep> ddr_to_tsm_;
  vector<TensorStep> lmem_to_tsm_;
  vector<TensorStep> tsm_to_ddr_;
  int max_step_num_;
};

}
#endif
