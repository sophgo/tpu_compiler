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

typedef struct {
  int start;
  int size;
  int tid;
  int step;
  bool busy;
} LMEM_BLOCK;

class LmemManager {
 public:
  explicit LmemManager(NetGraph* net_graph);

  bmerr_t assign_local_memory(Group* cluster, net_timestep* time_step, bool one_shoot);

 private:
  NetGraph* net_graph_;
  vector<list<LMEM_BLOCK>> block_record_;
  list<LMEM_BLOCK> block_list;

  bool is_tensor_resident_in_lmem(int tid);

  void recycle_lmem(list<LMEM_BLOCK>& block_list, net_timestep* time_step, int cur_step,
                    bool one_shoot);

  bool alloc_block(list<LMEM_BLOCK>& block_list, int tid, int step_idx);

  void merge_free_blocks(list<LMEM_BLOCK>& block_list);

  bool figure_out_tensors_real_addr(net_timestep* time_step);

  void show_blocks(list<LMEM_BLOCK>& block_list);
};

}
#endif
