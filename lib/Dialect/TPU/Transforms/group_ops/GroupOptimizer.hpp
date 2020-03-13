#ifndef GROUPOPS_OPTIMIZER_H
#define GROUPOPS_OPTIMIZER_H

#include "Group.hpp"
// #include "group_common.hpp"
// #include "gmem.hpp"
#include "ImLayer.hpp"
// #include "layer_stage.hpp"
#include "NetGraph.hpp"
#include "Tensor.hpp"
#include "MixNet.hpp"
#include "GMemManager.hpp"

namespace mlir {
class GroupOptimizer {
 public:
  explicit GroupOptimizer(NetGraph* net_graph);
  ~GroupOptimizer();
  bmerr_t optimize();
  Operation* build_fn(FuncOp * fn, MLIRContext * context, GroupOptimizer * optimizer);
  bool is_tg_op(Operation *op);
  uint64_t setOpGAddr(Operation * op);

 private:
  NetGraph* net_graph_;
  vector<Group*> groups_;
  MixNet mix_net_;
  GmemManager gmem_mgr_;

  void do_group(vector<Group*>& groups);
  void do_group_seso(vector<Group*>& groups);
  void add_valid_custers(vector<Group*>& groups, Group* target);
  vector<int> optimize_cut_points(Group* target, const vector<int>& cut_points);
  int calc_group_out_tensors_size(Group* target, const vector<int>& cut_points);
  void set_input_output_tensor();
  // void group2Graph(int gid, Group* group);
};

}
#endif
