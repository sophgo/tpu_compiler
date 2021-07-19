#pragma once
#include "Group.hpp"
#include "LayerStage.hpp"
#include "NetGraph.hpp"
#include "Tensor.hpp"

#define TBD_LADDR 0xFFFFFFFF
#define DEBUG_TYPE "group_ops"

namespace mlir {
class LMemAllocMethod {
public:
  LMemAllocMethod();
  virtual ~LMemAllocMethod();
  virtual bmerr_t assign_local_memory(Group *group, NetGraph *net_graph, net_timestep *time_step, bool one_shoot) = 0;
protected:
  virtual bool is_tensor_resident_in_lmem(tensor_type type);
};

typedef struct {
  int start;
  int size;
  int tid;
  int step;
  bool busy;
} LMEM_BLOCK;

class LMemAllocFitFirst : public LMemAllocMethod {
public:
  LMemAllocFitFirst();
  ~LMemAllocFitFirst() override;
  bmerr_t assign_local_memory(Group *group, NetGraph *net_graph, net_timestep *time_step, bool one_shoot) override;

private:
  void init(std::list<LMEM_BLOCK>& block_list);

  void recycle_lmem(std::list<LMEM_BLOCK>& block_list, net_timestep* time_step, int cur_step,
                    bool one_shoot);

  bool alloc_block(std::list<LMEM_BLOCK>& block_list, int tid, int step_idx);

  void merge_free_blocks(std::list<LMEM_BLOCK>& block_list);

  bool figure_out_tensors_real_addr(net_timestep* time_step);

  void show_blocks(std::list<LMEM_BLOCK>& block_list);

private:
  NetGraph* net_graph_;
  std::vector<std::list<LMEM_BLOCK>> block_record_;
};

class LMemAllocSizeOrder : public LMemAllocMethod {
public:
struct TensorRect {
  int first; // x1
  int last;  // x2
  int start = 0; // y1
  int end = 0;   // y2
  int size;      // height
  int step_idx;
  int tid;
  TensorRect(int _tid, int _first, int _last, int _step_idx, int _size) {
    tid = _tid;
    first = _first;
    last = _last;
    step_idx = _step_idx;
    size = _size;
  }
};

  LMemAllocSizeOrder();
  ~LMemAllocSizeOrder();
  bmerr_t assign_local_memory(Group *group, NetGraph *net_graph, net_timestep *time_step, bool one_shoot) override;

private:
  int figure_out_tensor_real_addr(
      std::list<std::shared_ptr<TensorRect>> &tensor_list, NetGraph *net_graph,
      net_timestep *time_step);
};

// reference Profile-guided_memory_optimization_for_deep_neural.pdf
class LMemAllocProfileGuided : public LMemAllocMethod {
public:
struct TensorRect {
  int first; // x1
  int last;  // x2
  int start = 0; // y1
  int end = 0;   // y2
  int size;      // height
  int step_idx;
  int tid;
  int width;
  TensorRect(int _tid, int _first, int _last, int _step_idx, int _size) {
    tid = _tid;
    first = _first;
    last = _last;
    step_idx = _step_idx;
    size = _size;
    width = _last - _first;
  }
};

struct Line {
  int start = 0;
  int first;
  int last;
  Line(int _start, int _first, int _last) {
    start = _start;
    first = _first;
    last = _last;
  }
};

  LMemAllocProfileGuided();
  ~LMemAllocProfileGuided();
  bmerr_t assign_local_memory(Group *group, NetGraph *net_graph, net_timestep *time_step, bool one_shoot) override;

private:
  int figure_out_tensor_real_addr(
      std::list<std::shared_ptr<TensorRect>> &tensor_list, NetGraph *net_graph,
      net_timestep *time_step);
};

} // namespace mlir