#include "GroupOptimizer.hpp"
#include "llvm/Support/ToolOutputFile.h"

#define DEBUG_TYPE "group_ops"
namespace mlir {


GroupOptimizer::GroupOptimizer(NetGraph* net_graph, FuncOp * fn, MLIRContext * context)
    : net_graph_(net_graph),
      mix_net_(net_graph, fn, context),
      fn_(fn), context_(context) {}

GroupOptimizer::~GroupOptimizer() {
  for (auto group : groups_) {
    delete group;
  }
}

// This function is used to group all layers. In a group, the top layer can directly
// use the results of the bottom layer without going through the store/load process,
// which reduces GDMA operations.
void GroupOptimizer::do_group(std::vector<Group*>& Groups) {
  Group* rough = new Group(net_graph_);

  for (auto layer = ImLayer::layers.begin(); layer != ImLayer::layers.end(); ++layer) {
    int id = (*layer)->id();

    if (!(*layer)->fusible) {
      // let these layers to be a single layer group.
      if (!rough->empty()) {
        add_valid_custers(Groups, rough);
        delete rough;
        rough = new Group(net_graph_);
      }

      rough->append(id);
      add_valid_custers(Groups, rough);
      delete rough;
      rough = new Group(net_graph_);
    } else {
      rough->append(id);
    }
  }

  if (!rough->empty()) {
    add_valid_custers(Groups, rough);
  }

  for (auto Group : groups_) {
    Group->clear_temp_data();
  }

  delete rough;
}

// This fucntion does two things:
//   1. Use the binary search to group all layers;
//   2. Adjust the position before and after grouping to minimize global memory.
void GroupOptimizer::add_valid_custers(std::vector<Group*>& groups, Group* target) {
  int num = target->size();
  int start = 0;
  const int end = num - 1;
  int cut = end, right = end;
  int left = start, valid = start;
  std::vector<int> cut_points;

  while (start < num) {
    if (cut == left) {
      cut_points.push_back(valid);
      start = valid + 1;
      cut = end;
      valid = start;
      left = start;
      right = end;
      continue;
    }

    auto group =
        std::make_shared<Group>(net_graph_, target->begin() + start, target->begin() + cut + 1);

    if (group->check_valid()) {
      valid = cut;
      left = cut;
    } else {
      right = cut;
    }

    cut = (left + right) / 2;
  }

  cut_points = optimize_cut_points(target, cut_points);

  start = 0;
  for (int i = 0; i < static_cast<int>(cut_points.size()); i++) {
    int end = cut_points[i];
    auto* group = new Group(net_graph_, target->begin() + start, target->begin() + end + 1);
    groups.push_back(group);
    start = end + 1;
  }
}

// After getting the cut node of the whole network, we can adjust the position of the node
// before and after, so that the size of the global memory used is the smallest.
std::vector<int> GroupOptimizer::optimize_cut_points(Group* target, const std::vector<int>& cut_points) {
  std::vector<int> best_solution(cut_points);
  int minimum_ddr_occupied = calc_group_out_tensors_size(target, cut_points);

  if (cut_points.size() == 1) {
    return best_solution;
  }

  int moving_idx = cut_points.size() - 2;
  while (moving_idx >= 0) {
    std::vector<std::vector<int>> solutions;
    std::vector<int> solution(best_solution);

    int left_point = (moving_idx == 0) ? 0 : solution[moving_idx - 1] + 1;

    // move cut_point to left step by step and
    // check if sub group of two side are all valid.
    for (solution[moving_idx]--; solution[moving_idx] >= left_point; solution[moving_idx]--) {
      // check if sub group of right side is valid.
      Group* group = new Group(net_graph_, target->begin() + solution[moving_idx] + 1,
                                 target->begin() + solution[moving_idx + 1] + 1);

      if (!group->check_valid()) {
        delete group;
        break;
      }

      // check left side
      delete group;
      group = new Group(net_graph_, target->begin() + left_point,
                          target->begin() + solution[moving_idx] + 1);

      // push all options to set.
      if (group->check_valid()) {
        solutions.push_back(solution);
      }
      delete group;
    }

    // update min ddr size and cut points
    for (auto iter = solutions.begin(); iter != solutions.end(); ++iter) {
      int group_out_data_size = calc_group_out_tensors_size(target, *iter);

      if (group_out_data_size < minimum_ddr_occupied) {
        minimum_ddr_occupied = group_out_data_size;
        best_solution = *iter;
      }
    }

    solutions.clear();
    moving_idx--;
  }

  return best_solution;
}

// return the output data size of layer group, unit: word
int GroupOptimizer::calc_group_out_tensors_size(Group* target, const std::vector<int>& cut_points) {
  std::vector<Group> Groups;

  int start = 0;
  for (int i = 0; i < static_cast<int>(cut_points.size()); i++) {
    int end = cut_points[i];
    Group Group(net_graph_, target->begin() + start, target->begin() + end + 1);
    Groups.push_back(Group);
    start = end + 1;
  }

  // get total out data size
  int total_data_size = 0;

  for (auto Group : Groups) {
    std::vector<int> out_tensors = Group.get_group_out_tensors();

    for (int j = 0; j < (int32_t)out_tensors.size(); ++j) {
      Tensor* tensor = net_graph_->get_tensor_by_id(out_tensors[j]);
      total_data_size += tensor->gmem_size();
    }
  }
  return total_data_size;
}

bmerr_t GroupOptimizer::optimize() {

  do_group(groups_);

  int group_id = 0;
  for (auto group : groups_) {
    bmerr_t status = group->assign_steps();
    if (status == BM_ERR_FAILURE) {
      llvm::errs() << "local memory allocate failed\n";
      assert(0);
    }

    if (group->size() == 1) {
      int id = group->layers()[0];
      ImLayer* layer = const_cast<ImLayer*>(net_graph_->get_layer_by_id(id));
      layer->is_tg_layer = true;
    }
    group->set_group_id(group_id);
    group_id++;
  }

  std::string DumpLayerGroupInfo = "layer_optimizer.out";
  if (DumpLayerGroupInfo != "") {
    std::fstream f;
    f.open(DumpLayerGroupInfo, std::ios::out);
    for (auto* group : groups_) {
      f << "group[" << group->get_group_id() << "]\n";
      group->print(f);
    }
    f.close();
  }

  return BM_SUCCESS;
}


void GroupOptimizer::set_input_output_tensor() {
  for (auto& layer : ImLayer::layers) {
    IR_TYPE type = layer->type();

    for (auto& tensor : layer->in_tensors) {
      int tid = tensor->id();
      if (net_graph_->get_tensor_type(tid) == TENSOR_NEURON ||
          net_graph_->get_tensor_type(tid) == TENSOR_NEURON_WINOGRAD) {
        int from_layer = net_graph_->get_tensor_from_layer(tid);
        if (from_layer == -1) {
          mix_net_.set_net_in_tensor(tid);
          LLVM_DEBUG(llvm::errs() << "Input tensor: " << tid << "\n";);
        }
      }
    }

    for (auto& tensor : layer->out_tensors) {
      int tid = tensor->id();
      const std::vector<int>& to_layers = net_graph_->get_tensor_to_layer(tid);
      if (to_layers.empty() && type != IR_MULTIINPUT) {
        mix_net_.set_net_out_tensor(tid);
        LLVM_DEBUG(llvm::errs() << "Output tensor: " << tid << "\n";);
      }
    }
  }
}

// use name to judge if ref to the same op
static bool is_same_layer(Operation * op, const ImLayer * layer) {
  if (isValidLayerGroupOp(op)) {
    std::string op_name = getOpName(op).str();
    // op_name.erase(std::remove(op_name.begin(), op_name.end(), '\b'), op_name.end());
    if (op_name == layer->name())
      return true;
    else return false;
  }
  return false;
}

bool GroupOptimizer::is_group_start(Operation * op, int * gid) {
  if (!isValidLayerGroupOp(op)) {
    return false;
  }
  int group_id = 0;
  for(auto group: groups_) {
    // check if is the first layer in the group
    int id = group->layers()[0];
    const ImLayer * start_layer = net_graph_->get_layer_by_id(id);
    if (is_same_layer(op, start_layer)) {
      if (group->layers().size() > 1) {
        *gid = group_id;
        return true;
      }
    }
    group_id++;
  }

  return false;
}

void GroupOptimizer::lower_to_tl(Operation *op, int gid) {
  Group * group = groups_[gid];
  if (group->lowered()) {
    return;
  }
  int n_secs = group->nsecs_and_hsecs.first;
  int h_secs = group->nsecs_and_hsecs.second;

  mix_net_.set_start_op(op);
  mix_net_.add_group_start_ops(gid, group, op, n_secs, h_secs);

  bool first_loop = true;
  for (int n_loop = 0; n_loop < n_secs; n_loop++) {
    for (int h_loop = 0; h_loop < h_secs; h_loop++) {

      group->update_tensor_slices(n_secs, h_secs, n_loop, h_loop);

      for (int step_id = 0; step_id < group->time_step->get_timestep_num(); ++step_id) {
        mix_net_.parallel_start();
        int cur_layer = group->time_step->get_layer(step_id);

        if (cur_layer != -1) {
          mix_net_.add_tl_layer(
              gid, cur_layer, group->time_step, step_id,
              h_secs != 1, n_loop, h_loop);
        }

        const std::vector<TENSOR_STEP>& cur_tensors = group->time_step->get_tensors(step_id);
        for (uint32_t i = 0; i < cur_tensors.size(); ++i) {
          // check if tensor is kept in lmem.
          if ((!first_loop) && cur_tensors[i].second == TIMESTEP_LOAD &&
              group->time_step->is_tensor_hold_in_memory(cur_tensors[i].first)) {
            continue;
          }

          mix_net_.add_transport_op(
              gid, cur_tensors[i], group->time_step, step_id);
        }

        mix_net_.parallel_end();
      }

      first_loop = false;
    }
  }

  mix_net_.add_group_end_ops(gid, group, n_secs, h_secs);

  group->set_lowered(true);
}

template<typename TyOp>
struct LGLoweringPattern : public RewritePattern {
  LGLoweringPattern(FuncOp * fn , MLIRContext *context, GroupOptimizer * optimizer)
      : RewritePattern(TyOp::getOperationName(), 1, context),
      optimizer_(optimizer), fn_(fn), context_(context){
        //mix_net_ = optimizer->get_net();
      }

  PatternMatchResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    // if already lowered to tl, return false
    int group_id = 0;
    if (optimizer_->is_group_start(op, &group_id)) {
      LLVM_DEBUG(llvm::errs()
        << "Find group start: " << getOpName(op) << "\n";);
      optimizer_->lower_to_tl(op, group_id);
    }
    return matchSuccess();
  }

  GroupOptimizer *optimizer_;
  FuncOp * fn_;
  MLIRContext * context_;
  MixNet * mix_net_;
};


// lower to tl inst according to layer group result
void GroupOptimizer::build_fn(MLIRContext * context) {
  set_input_output_tensor();

  OwningRewritePatternList patterns_pack;
  patterns_pack.insert<
      LGLoweringPattern<tpu::TG_INT8_PC_Conv2DOp>,
      LGLoweringPattern<tpu::TG_INT8_PC_DeConv2DOp>,
      LGLoweringPattern<tpu::TG_INT8_EltwiseAddOp>,
      LGLoweringPattern<tpu::TG_INT8_EltwiseMulOp>,
      LGLoweringPattern<tpu::TG_INT8_EltwiseMaxOp>,
      LGLoweringPattern<tpu::TG_INT8_EltwiseMinOp>,
      LGLoweringPattern<tpu::TG_INT8_PoolAvg2DOp>,
      LGLoweringPattern<tpu::TG_INT8_PoolMax2DOp>,
      LGLoweringPattern<tpu::TG_INT8_LutOp>,
      LGLoweringPattern<tpu::TG_BF16_LutOp>,
      LGLoweringPattern<tpu::TG_INT8_LrnOp>,
      LGLoweringPattern<tpu::TG_INT8_BroadcastMulOp>,
      LGLoweringPattern<tpu::TG_INT8_UpsampleOp>,
      LGLoweringPattern<tpu::TG_INT8_LeakyReluOp>,
      LGLoweringPattern<tpu::TG_INT8_ConcatOp>,
      LGLoweringPattern<tpu::TG_INT8_PadOp>,
      LGLoweringPattern<tpu::TG_INT8_CropOp>,
      LGLoweringPattern<tpu::TG_INT8_ReluOp>,
      LGLoweringPattern<tpu::TG_BF16_INT8_CastOp>,
      LGLoweringPattern<tpu::TG_INT8_ZeroMaskOp>,
      LGLoweringPattern<tpu::TG_INT8_SliceOp>,
      // BF16
      LGLoweringPattern<tpu::TG_BF16_Conv2DOp>,
      LGLoweringPattern<tpu::TG_BF16_DeConv2DOp>,
      LGLoweringPattern<tpu::TG_BF16_EltwiseAddOp>,
      LGLoweringPattern<tpu::TG_BF16_EltwiseMulOp>,
      LGLoweringPattern<tpu::TG_BF16_EltwiseMaxOp>,
      LGLoweringPattern<tpu::TG_BF16_EltwiseMinOp>,
      LGLoweringPattern<tpu::TG_BF16_PoolAvg2DOp>,
      LGLoweringPattern<tpu::TG_BF16_PoolMax2DOp>,
      LGLoweringPattern<tpu::TG_BF16_LutOp>,
      LGLoweringPattern<tpu::TG_BF16_LrnOp>,
      LGLoweringPattern<tpu::TG_BF16_BroadcastMulOp>,
      LGLoweringPattern<tpu::TG_BF16_UpsampleOp>,
      LGLoweringPattern<tpu::TG_BF16_LeakyReluOp>,
      LGLoweringPattern<tpu::TG_BF16_ConcatOp>,
      LGLoweringPattern<tpu::TG_BF16_PadOp>,
      LGLoweringPattern<tpu::TG_BF16_CropOp>,
      LGLoweringPattern<tpu::TG_BF16_ReluOp>,
      LGLoweringPattern<tpu::TG_INT8_BF16_CastOp>,
      LGLoweringPattern<tpu::TG_BF16_ZeroMaskOp>,
      LGLoweringPattern<tpu::TG_BF16_SliceOp>
      >(fn_, context, this);
  applyPatternsGreedily(*fn_, patterns_pack);

}

}
