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

using TensorStep = std::vector<int>;

class GroupSteps {
 public:
  explicit GroupSteps(NetGraph* net_graph)
      : net_graph_(net_graph), layers_(), loads_(), stores_(), max_step_num_(0) {}

  ~GroupSteps() {
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
  void assign(Group* group);
  void to_timestep(net_timestep* time_step);

  // TSM Support
  void assign_with_tsm(Group* group);
  void to_timestep_with_tsm(net_timestep* time_step);

  static void timestep_assgin(NetGraph* net_graph, Group* group, net_timestep* time_step);

  static void timestep_assign_with_tsm(NetGraph* net_graph, Group* group,
                                       net_timestep* time_step);

  static bmerr_t balance_tdma_tiu_steps(NetGraph* net_graph, Group* group,
                                        net_timestep* time_step,
                                        const std::pair<int, int>& nsecs_and_hsec);

  static bmerr_t balance_tdma_tiu(NetGraph* net_graph, Group* group,
                                  net_timestep** time_step,
                                  const std::pair<int, int>& nsecs_and_hsec);

 private:
  NetGraph* net_graph_;
  std::vector<int> layers_;
  std::vector<TensorStep> loads_;
  std::vector<TensorStep> stores_;
  std::vector<TensorStep> tsm_to_lmem_;
  std::vector<TensorStep> ddr_to_tsm_;
  std::vector<TensorStep> lmem_to_tsm_;
  std::vector<TensorStep> tsm_to_ddr_;
  int max_step_num_;
};

}
#endif
