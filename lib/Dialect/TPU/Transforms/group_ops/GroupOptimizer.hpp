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
  explicit GroupOptimizer(NetGraph* net_graph, FuncOp * fn, MLIRContext * context);
  ~GroupOptimizer();
  bmerr_t optimize();
  void build_fn(MLIRContext * context);
  bool is_tg_op(Operation *op);
  uint64_t setOpGAddr(Operation * op);
  void lower_to_tl_group(MLIRContext * context);
  void lower_to_tg_group(MLIRContext * context);
  void assign_weight_address(MLIRContext * context);
  MixNet * get_net() { return &mix_net_; }
  bool is_group_start(Operation *op, int * id);
  void lower_to_tl(Operation *op, int group_id);

 private:
  NetGraph* net_graph_;
  vector<Group*> groups_;
  MixNet mix_net_;
  GmemManager gmem_mgr_;
  FuncOp * fn_;
  FuncOp out_fn_;
  MLIRContext * context_;

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
