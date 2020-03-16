#ifndef GROUPOPS_GMEM_H
#define GROUPOPS_GMEM_H

#include "Group.hpp"
#include "NetGraph.hpp"
#include "Tensor.hpp"

namespace mlir {

typedef struct {
  u64 start;
  u64 size;
  int tid;
  bool busy;
} GMEM_BLOCK;

class GmemManager {
 public:
  explicit GmemManager(NetGraph* net_graph);

  u64 assign_global_memory(const vector<Group*>& clusters, bool gmem_recycle = true);

 private:
  NetGraph* net_graph_;
  std::vector<list<GMEM_BLOCK>> block_record_;
  int tg_join_out_tensor_;
  std::map<int, u64> tg_join_input_tensors_;
  std::set<int> net_in_tensors_;
  std::set<int> net_out_tensors_;

  int set_in_place_layer_tensor_gaddr_unit(const int tid, const uint64_t value);
  bool set_in_place_layer_tensor_gaddr(int tid, uint64_t value);

  void find_tg_join_tensors();

  bool is_tensor_of_join_layer(int tid);

  void prealloc_io_tensors(list<GMEM_BLOCK>& block_list);

  void recycle_cluster_gmem(list<GMEM_BLOCK>& block_list, Group* cluster, Group* prev_cluster);

  void prealloc_cluster_gmem(list<GMEM_BLOCK>& block_list, Group* cluster);

  void alloc_block(list<GMEM_BLOCK>& block_list, int tid);

  void merge_free_blocks(list<GMEM_BLOCK>& block_list);

  u64 figure_out_tensor_offset();

  void show_blocks(list<GMEM_BLOCK>& block_list);
};

}
#endif
