#ifndef GROUPOPS_OPTIMIZER_H
#define GROUPOPS_OPTIMIZER_H

#include "Group.hpp"
#include "ImLayer.hpp"
#include "NetGraph.hpp"
#include "Tensor.hpp"
#include "MixNet.hpp"

namespace mlir {
class GroupOptimizer {
 public:
  explicit GroupOptimizer(NetGraph* net_graph, FuncOp * fn, MLIRContext * context);
  ~GroupOptimizer();
  bmerr_t optimize();
  void build_fn(MLIRContext * context);
  MixNet * get_net() { return &mix_net_; }
  bool is_group_start(Operation *op, int * id);
  void lower_to_tl(Operation *op, int group_id);

 private:
  NetGraph* net_graph_;
  std::vector<Group*> groups_;
  std::vector<std::vector<Group*>> groups_v_;
  MixNet mix_net_;
  FuncOp * fn_;
  FuncOp out_fn_;
  MLIRContext * context_;
  LG_Slice_Limit slice_limit_;
  std::vector<uint64_t> cost_;

  void do_group();
  void do_group_with_h_slice();
  void do_group_with_w_slice();
  bool isGroupFusible(Group * group);
  void add_valid_group(Group* target);
  std::vector<int> optimize_cut_points(Group* target, const std::vector<int>& cut_points);
  int calc_group_out_tensors_size(Group* target, const std::vector<int>& cut_points);
  void set_input_output_tensor();
  void layer_group();
  uint64_t cal_group_cost();
  void choose_best_group();
  void set_slice_limit(int s);
};

}
#endif
