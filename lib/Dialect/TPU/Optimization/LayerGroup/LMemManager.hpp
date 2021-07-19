/*
 * Copyright (C) Bitmain Technologies Inc.
 * All Rights Reserved.
 */
#ifndef GROUPOPS_LMEM_H
#define GROUPOPS_LMEM_H

#include "Group.hpp"
#include "LayerStage.hpp"
#include "NetGraph.hpp"
#include "Tensor.hpp"

namespace mlir {

class LmemManager {
 public:
  explicit LmemManager(NetGraph* net_graph);

  bmerr_t assign_local_memory(Group* group, net_timestep* time_step, bool one_shoot);

 private:
  NetGraph* net_graph_;
};

}
#endif
