/*
 * Copyright (C) Bitmain Technologies Inc.
 * All Rights Reserved.
 */
#include "Steps.hpp"
#include "Group.hpp"
#include "LayerStage.hpp"
#include "TiuCycle.hpp"
#include "TdmaCycle.hpp"
#include "LMemManager.hpp"

#define DEBUG_TYPE "group_ops"

namespace mlir {

static llvm::cl::OptionCategory clOptionsCategory("Layer Group Option");

static llvm::cl::opt<bool> clEnableLayerBalance(
    "enable-layer-balance",
    llvm::cl::desc("Enable layer balance in layer group"),
    llvm::cl::init(false),
    llvm::cl::cat(clOptionsCategory));

void GroupSteps::append(int layer, TensorStep& load_tensors, TensorStep& store_tensors) {
  if (layer == -1 && load_tensors.empty() && store_tensors.empty()) {
    return;
  }

  layers_.push_back(layer);
  loads_.push_back(load_tensors);
  stores_.push_back(store_tensors);

  max_step_num_++;
}

void GroupSteps::insert(int layer, TensorStep& load_tensors, TensorStep& store_tensors, int pos) {
  if (layer == -1 && load_tensors.empty() && store_tensors.empty()) {
    return;
  }

  layers_.insert(layers_.begin() + pos, layer);
  loads_.insert(loads_.begin() + pos, load_tensors);
  stores_.insert(stores_.begin() + pos, store_tensors);

  max_step_num_++;
}

static void dump(const std::vector<int>& v) {
  for (int i = 0; i < static_cast<int>(v.size()); i++) {
    std::cout << v[i];
  }
  std::cout << "\n";
}

static std::vector<int> intersection(std::vector<int> v1, std::vector<int> v2) {
  std::vector<int> v;
  sort(v1.begin(), v1.end());
  sort(v2.begin(), v2.end());
  set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), back_inserter(v));
  if (!v.empty()) {
    std::cout << "intersection: ";
    dump(v);
  }

  return v;
}

static std::vector<int> difference(std::vector<int> v1, std::vector<int> v2) {
  std::vector<int> v;
  sort(v1.begin(), v1.end());
  sort(v2.begin(), v2.end());
  set_difference(v1.begin(), v1.end(), v2.begin(), v2.end(), back_inserter(v));
  if (!v.empty()) {
    std::cout << "difference: ";
    dump(v);
  }

  return v;
}

// This functions is used to adjust the data position of layers_, loads_ and stores.
// The final result is:
//   1. Insert -1 at the beginning and the end of layers.
//   2. Insert 2 null at the end of loads_.
//   3. Insert 2 null at the beginning of stores_
void GroupSteps::rearrange_steps() {
  TensorStep null;

  // move load tensor to prev step
  for (int i = 0; i < static_cast<int>(loads_.size()); i++) {
    if (loads_[i].empty()) {
      continue;
    }
    if (i == 0) {
      insert(-1, loads_[i], null);
    } else {
      loads_[i - 1].insert(loads_[i - 1].end(), loads_[i].begin(), loads_[i].end());
    }
    loads_[i].clear();
  }

  // move store tensor to next step.
  for (int i = stores_.size() - 1; i >= 0; i--) {
    if (stores_[i].empty()) {
      continue;
    }
    if (i == (int)stores_.size() - 1) {
      append(-1, null, stores_[i]);
    } else {
      stores_[i + 1].insert(stores_[i + 1].end(), stores_[i].begin(), stores_[i].end());
    }
    stores_[i].clear();
  }

again:  // clean & merge steps
  for (int i = 0; i < static_cast<int>(layers_.size()); i++) {
    if (layers_[i] != -1) {
      continue;
    }

    if (loads_[i].empty() && stores_[i].empty()) {
      layers_.erase(layers_.begin() + i);
      loads_.erase(loads_.begin() + i);
      stores_.erase(stores_.begin() + i);
      max_step_num_--;
      goto again;
    }

    if (i < static_cast<int>(layers_.size()) - 1 && layers_[i + 1] == -1) {
      loads_[i].insert(loads_[i].end(), loads_[i + 1].begin(), loads_[i + 1].end());
      stores_[i].insert(stores_[i].end(), stores_[i + 1].begin(), stores_[i + 1].end());
      loads_[i + 1].clear();
      stores_[i + 1].clear();
    }
  }

loop:
  for (int i = 0; i < static_cast<int>(layers_.size()); i++) {
    bool restart = false;
    int id = layers_[i];
    if (id == -1) {
      continue;
    }

    const std::vector<int>& in_tensors = net_graph_->get_in_tensors_of_layer(id);
    const std::vector<int>& out_tensors = net_graph_->get_out_tensors_of_layer(id);

    if (!loads_[i].empty()) {
      std::vector<int> conflict = intersection(in_tensors, loads_[i]);
      if (!conflict.empty()) {
        restart = true;
        loads_[i] = difference(loads_[i], conflict);
        loads_[i - 1].insert(loads_[i - 1].end(), conflict.begin(), conflict.end());
      }
    }

    if (!stores_[i].empty()) {
      std::vector<int> conflict = intersection(out_tensors, stores_[i]);
      if (!conflict.empty()) {
        restart = true;
        stores_[i] = difference(stores_[i], conflict);
        stores_[i + 1].insert(stores_[i + 1].end(), conflict.begin(), conflict.end());
      }
    }

    if (restart) {
      goto loop;
    }
  }
}

void GroupSteps::assign_with_tsm(Group* group) {
  std::set<int> tensors_in_lmem;
  std::set<int> tensors_in_tsm;

  // Ignore tg layer
  if (group->size() == 1) {
    return;
  }

  for (auto id : group->layers()) {
    int layer_step = -1;

    layer_step = id;
    TensorStep ddr_to_tsm;
    TensorStep tsm_to_lmem;
    TensorStep lmem_to_tsm;
    TensorStep tsm_to_ddr;

    TensorStep ddr_to_lmem;
    TensorStep lmem_to_ddr;

    // FIXME(arcbbb): concat layer support

    // Check Input tensors: generate load inst. if necessary
    const std::vector<int>& in_tensors = net_graph_->get_in_tensors_of_layer(id);
    for (int j = 0; j < static_cast<int>(in_tensors.size()); ++j) {
      int tid = in_tensors[j];

      auto ret = tensors_in_lmem.insert(tid);
      // load tensor from TSM if LMEM doesn't have it
      if (ret.second == true) {
        if (net_graph_->get_tensor_type(tid) == TENSOR_NEURON) {
          // current policy: always load neuron from ddr
          ddr_to_lmem.push_back(tid);
        } else {
          // current policy: always load weight from tsm
          tsm_to_lmem.push_back(tid);
          auto ret = tensors_in_tsm.insert(tid);
          // load tensor from DDR if TSM doesn't have it
          if (ret.second == true) {
            ddr_to_tsm.push_back(tid);
          }
        }
      }
    }

    // Check Output tensors: generate store inst. if necessary
    const std::vector<int>& out_tensors = net_graph_->get_out_tensors_of_layer(id);
    for (int j = 0; j < static_cast<int>(out_tensors.size()); ++j) {
      int tid = out_tensors[j];
      tensors_in_lmem.insert(tid);
      if (group->is_group_out_tensor(tid)) {
        // TODO(arcbbb): keep tensor in TSM for next user.
        // Now we flush it to DDR directly.
        lmem_to_ddr.push_back(tid);
      }
    }

    // Add one step in GroupSteps
    // TODO(arcbbb): overload ClusterStep::append(...)
    {
      // add layer
      layers_.push_back(layer_step);

      // add tensor movement
      loads_.push_back(ddr_to_lmem);
      stores_.push_back(lmem_to_ddr);
      tsm_to_lmem_.push_back(tsm_to_lmem);
      ddr_to_tsm_.push_back(ddr_to_tsm);
      lmem_to_tsm_.push_back(lmem_to_tsm);
      tsm_to_ddr_.push_back(tsm_to_ddr);
      max_step_num_++;
    }
  }
}

// This funtion is used to collect information in the group.
// This information including:
//   1. layers_ save the contained layers.
//   2. loads_ save the tensor need to load.
//   3. stores_ save the tensor need to store.
void GroupSteps::assign(Group* group) {
  std::set<int> tensors_in_lmem;

  bool single_layer_cluster = (group->size() == 1);

  for (auto id : group->layers()) {
    int layer_step = -1;

    TensorStep load_tensors;
    TensorStep store_tensors;

    // push layer to layer_step if not special case.
    bool ignore_concat_layer = net_graph_->is_concat_special_case(id, -1);
    if (!ignore_concat_layer) {
      layer_step = id;

      // push all layers' in tensors to tensors_in_lmem and tensor step
      const std::vector<int>& in_tensors = net_graph_->get_in_tensors_of_layer(id);
      for (int j = 0; j < static_cast<int>(in_tensors.size()); ++j) {
        auto ret = tensors_in_lmem.insert(in_tensors[j]);
        if (ret.second == true) {
          // std::cout << "load tensor: " << in_tensors[j] << "\n";
          load_tensors.push_back(in_tensors[j]);
        }
      }
    }

    // push out tensors of layer to local mem.
    const std::vector<int>& out_tensors = net_graph_->get_out_tensors_of_layer(id);
    for (int j = 0; j < static_cast<int>(out_tensors.size()); ++j) {
      int tid = out_tensors[j];
      tensors_in_lmem.insert(tid);

      const std::vector<int>& to_layers = net_graph_->get_tensor_to_layer(tid);
      // Why need ignore_concat_layer?
      if (group->is_group_out_tensor(tid)) {
        // you are not concat layer, and your output tensor is shared
        store_tensors.push_back(tid);
        // if your consumer is concat layer in next group,
        // add this tensor to the ignore std::list of concat layer
        std::vector<std::pair<int, int>> ignore_pair;
        bool is_concat_in_place = true;
        for (int k = 0; k < static_cast<int>(to_layers.size()); ++k) {
          if (net_graph_->is_concat_optimized_case(to_layers[k], tid, group->size())) {
            std::vector<int> in_tensors = net_graph_->get_in_tensors_of_layer(to_layers[k]);
            for (int x = 0; x < static_cast<int>(in_tensors.size()); x++) {
              if (in_tensors[x] == tid) {
                ignore_pair.push_back(std::pair<int, int>(k,x));
              }
            }
          } else if (net_graph_->get_layer_by_id(to_layers[k])->is_inplace_layer && single_layer_cluster) {
            is_concat_in_place = false;
          }
        }
        if (is_concat_in_place) {
          for (auto ignore_unit : ignore_pair) {
            auto im_layer = const_cast<ImLayer *>(net_graph_->get_layer_by_id(to_layers[ignore_unit.first]));
            ImConcat* im_concat_layer = dynamic_cast<ImConcat *>(im_layer);
            im_concat_layer->ignored_bottoms.insert(ignore_unit.second);
          }
        }
      } else {
        for (int k = 0; k < static_cast<int>(to_layers.size()); ++k) {
          int to_layer = to_layers[k];
          if (net_graph_->is_concat_special_case(to_layer, tid, group->size())) {
            int concat_out_tensor = net_graph_->get_out_tensors_of_layer(to_layer)[0];
            if (group->is_group_out_tensor(concat_out_tensor)) {
              store_tensors.push_back(tid);
            }
          }
        }
      }
    }

    append(layer_step, load_tensors, store_tensors);
  }

  rearrange_steps();
}

// Check if the current tensor conflicts with the tensors used by npu execution
static bool tensor_conflict_with_npu_exe(NetGraph* net_graph, net_timestep* time_step,
                                         int tensor_id, int timestep_idx) {
  int layer_id = time_step->get_layer(timestep_idx);
  if (layer_id != -1) {
    const std::vector<int>& in_tensors = net_graph->get_in_tensors_of_layer(layer_id);

    if (std::find(in_tensors.begin(), in_tensors.end(), tensor_id) != in_tensors.end()) {
      return true;
    }

    const std::vector<int>& out_tensors = net_graph->get_out_tensors_of_layer(layer_id);
    if (std::find(out_tensors.begin(), out_tensors.end(), tensor_id) != out_tensors.end()) {
      return true;
    }
  }
  return false;
}

bmerr_t GroupSteps::balance_tdma_tiu(NetGraph* net_graph, Group* group,
                                     net_timestep** time_step,
                                     const std::pair<int, int>& group_slice_) {

  if (group->size() == 1) {
    return BM_SUCCESS;
  }

  int nsecs = group_slice_.first;
  int hsecs = group_slice_.second;

  bmerr_t status = group->update_slices(nsecs, hsecs);
  if (status == BM_ERR_FAILURE) {
    return BM_ERR_FAILURE;
  }

  bool one_shot = group_slice_.first == 1 && group_slice_.second == 1;
  LmemManager lmem(net_graph);
  if (clEnableLayerBalance) {
    net_timestep* new_timestep = new net_timestep(**time_step);
    balance_tdma_tiu_steps(net_graph, group, new_timestep, group_slice_);

    // check if satisfy the local memory
    (*time_step)->update_mem_buffer_size();
    bmerr_t is_ok_without_balance = lmem.assign_local_memory(group, *time_step, one_shot);

    new_timestep->update_mem_buffer_size();
    bmerr_t is_ok_with_balance =  lmem.assign_local_memory(group, new_timestep, one_shot);

    if (is_ok_with_balance == BM_ERR_FAILURE) {
      // if without balance is ok, then revert
      delete new_timestep;
      if (is_ok_without_balance == BM_SUCCESS) {
        (*time_step)->update_mem_buffer_size();
        lmem.assign_local_memory(group, *time_step, one_shot);
        return BM_SUCCESS;
      } else {
        return BM_ERR_FAILURE;
      }
    } else {
      delete *time_step;
      *time_step = new_timestep;
      return BM_SUCCESS;
    }
  } else {
    // disable balance layer
    (*time_step)->update_mem_buffer_size();
    return lmem.assign_local_memory(group, *time_step, one_shot);
  }

}

// This fucntion moved up/down gdma load/store operations for balancing bdc and
// gdma cycle.
// The array timestep_slack holds the number of execution cycles per step.
// When the value is greater than 0, it means that the execution cycle of bdc
// instruction covers gdma instruction, otherwise we can move up/down load/store
// operations to keep balance.
bmerr_t GroupSteps::balance_tdma_tiu_steps(NetGraph* net_graph, Group* group,
                                           net_timestep* time_step,
                                           const std::pair<int, int>& group_slice_) {
  int timestep_num = time_step->get_timestep_num();
  int* timestep_slack = new int[timestep_num];
  for (int i = 0; i < timestep_num; ++i) {
    timestep_slack[i] = 0;
  }

  std::map<int, int> tensor_gdma_cycle;
  std::map<int, int> tensor_size;
  std::vector<std::list<TENSOR_STEP>> ts_tensors;

  // get cycle slack of each time step
  int tensor_id;
  int tensor_local_size;

  std::list<TENSOR_STEP> list_tensors;

  for (int i = 0; i < timestep_num; ++i) {
    int cur_layer = time_step->get_layer(i);
    const std::vector<TENSOR_STEP>& cur_tensors = time_step->get_tensors(i);
    // add layer cycle for each time step
    if (cur_layer != -1) {
      TiuCycle tc(net_graph);
      int layer_cycle = tc.get_cycle(cur_layer);
      timestep_slack[i] += layer_cycle;
    }
    // sub tensor gdma cycle for each time step
    list_tensors.clear();
    for (uint32_t j = 0; j < cur_tensors.size(); ++j) {
      tensor_id = cur_tensors[j].first;
      TdmaCycle tc(net_graph);
      int gdma_cycle = tc.get_cycle(cur_tensors[j]);
      tensor_gdma_cycle[tensor_id] = gdma_cycle;

      // get local memory require size
      Tensor* tensor = net_graph->get_tensor_by_id(tensor_id);
      tensor_local_size = tensor->lmem_size();
      tensor_size[tensor_id] = tensor_local_size;
      // generate tensor gdma std::list
      list_tensors.push_back(cur_tensors[j]);
      // slack minus gdma cycle
      timestep_slack[i] -= tensor_gdma_cycle[tensor_id];
    }

    ts_tensors.push_back(list_tensors);
  }

  // move gdma time step for reducing total cycle of time steps
  int cycle_profit, max_cycle_profit;
  int sel_to_ts_idx, sel_tensor_id;
  int cur_profit, dst_cost;
  std::list<TENSOR_STEP>::iterator ts_tensor;
  std::list<TENSOR_STEP>::iterator sel_ts_tensor;

  int to_ts_idx, pre_ts;
  bool valid_flag;
  TIMESTEP_LD_ST tdma_type;

  LLVM_DEBUG(llvm::errs() << LOG_TAB_L4
                          <<"[Balance Layer] Begin\n";);

  for (int cur_ts_idx = 0; cur_ts_idx < timestep_num;) {
    // if need to move gdma time step
    int cur_slack = timestep_slack[cur_ts_idx];
    if (cur_slack >= 0) {
      ++cur_ts_idx;
      continue;
    }

    // get the src gdma tensor and dst time step
    max_cycle_profit = 0;
    sel_to_ts_idx = -1;
    // check each tensor in current timestep, move it up or down
    // and check if have profit
    for (ts_tensor = ts_tensors[cur_ts_idx].begin(); ts_tensor != ts_tensors[cur_ts_idx].end();
         ++ts_tensor) {
      tensor_id = ts_tensor->first;
      int cur_new_slack = cur_slack + tensor_gdma_cycle[tensor_id];
      cur_profit = (cur_new_slack >= 0 ? 0 : cur_new_slack) - cur_slack;

      int range_end = time_step->tensor_range_end_timestep(*ts_tensor);

      if (ts_tensor->second == TIMESTEP_LOAD) {
        to_ts_idx = cur_ts_idx - 1;
        valid_flag = to_ts_idx >= range_end;
      } else {
        to_ts_idx = cur_ts_idx + 1;
        valid_flag = to_ts_idx <= range_end;
      }

      while (valid_flag) {
        if (!(ts_tensor->second == TIMESTEP_STORE &&
              tensor_conflict_with_npu_exe(net_graph, time_step, tensor_id, to_ts_idx))) {
          if (timestep_slack[to_ts_idx] > 0) {
            dst_cost = timestep_slack[to_ts_idx] - tensor_gdma_cycle[tensor_id];
            dst_cost = dst_cost >= 0 ? 0 : dst_cost;

            cycle_profit = cur_profit + dst_cost;

            if (cycle_profit > max_cycle_profit) {
              max_cycle_profit = cycle_profit;
              sel_to_ts_idx = to_ts_idx;
              sel_ts_tensor = ts_tensor;
              sel_tensor_id = tensor_id;
            } else if (cycle_profit == max_cycle_profit && sel_to_ts_idx != -1) {
              if ((tensor_size[tensor_id] * (abs(to_ts_idx - cur_ts_idx) + 1)) <
                  (tensor_size[sel_tensor_id] * (abs(sel_to_ts_idx - cur_ts_idx) + 1))) {
                max_cycle_profit = cycle_profit;
                sel_to_ts_idx = to_ts_idx;
                sel_ts_tensor = ts_tensor;
                sel_tensor_id = tensor_id;
              }
            }
          }
        }
        if (ts_tensor->second == TIMESTEP_LOAD) {
          --to_ts_idx;
          valid_flag = to_ts_idx >= range_end;
        } else {
          ++to_ts_idx;
          valid_flag = to_ts_idx <= range_end;
        }
      }
    }
    if (sel_to_ts_idx == -1) {
      ++cur_ts_idx;
      continue;
    }

    LLVM_DEBUG(llvm::errs() << LOG_TAB_L5
                            << "[Balance Layer][timestep "
                            << cur_ts_idx << "]"
                            << " Max Profit Action: Move tensor " << sel_tensor_id
                            << " from " << cur_ts_idx << " to " << sel_to_ts_idx
                            << "\n";);

    // update to_ts_idx for following loop
    if (sel_ts_tensor->second == TIMESTEP_LOAD) {
      to_ts_idx = cur_ts_idx - 1;
      valid_flag = to_ts_idx >= sel_to_ts_idx;
      tdma_type = TIMESTEP_LOAD;
    } else {
      to_ts_idx = cur_ts_idx + 1;
      valid_flag = to_ts_idx <= sel_to_ts_idx;
      tdma_type = TIMESTEP_STORE;
    }
    pre_ts = cur_ts_idx;

    // bubble cur_tensor from cur_ts_idx to sel_ts_idx
    while (valid_flag) {
      ts_tensors[to_ts_idx].push_back(*sel_ts_tensor);
      timestep_slack[to_ts_idx] -= tensor_gdma_cycle[sel_tensor_id];
      ts_tensors[pre_ts].erase(sel_ts_tensor);
      timestep_slack[pre_ts] += tensor_gdma_cycle[sel_tensor_id];

      LLVM_DEBUG(llvm::errs() << LOG_TAB_L5
                              << "[Balance Layer] Action Valid: Move tensor "
                              << sel_ts_tensor->first
                              << " from ts " << pre_ts
                              << " to ts " << to_ts_idx << " with profit: "
                              << max_cycle_profit << "\n";);

      if (to_ts_idx == sel_to_ts_idx) {
        break;
      }

      // find next tensor in the current bubble timestep
      if (tdma_type == TIMESTEP_STORE &&
          tensor_conflict_with_npu_exe(net_graph, time_step, sel_tensor_id, to_ts_idx)) {
        sel_ts_tensor = ts_tensors[to_ts_idx].end();
        --sel_ts_tensor;
      } else {
        max_cycle_profit = 0;
        sel_tensor_id = -1;
        for (ts_tensor = ts_tensors[to_ts_idx].begin();
             ts_tensor != ts_tensors[to_ts_idx].end(); ++ts_tensor) {
          if (tdma_type != ts_tensor->second) {
            continue;
          }

          // add for software pipeline
          int new_range_end = time_step->tensor_range_end_timestep(*ts_tensor);
          if ((tdma_type == TIMESTEP_LOAD && new_range_end > sel_to_ts_idx) ||
              (tdma_type == TIMESTEP_STORE && new_range_end < sel_to_ts_idx)) {
            continue;
          }

          tensor_id = ts_tensor->first;
          int cur_new_slack = timestep_slack[to_ts_idx] + tensor_gdma_cycle[tensor_id];
          cur_profit = (cur_new_slack >= 0 ? 0 : cur_new_slack) -
                       (timestep_slack[to_ts_idx] >= 0 ? 0 : timestep_slack[to_ts_idx]);

          dst_cost = timestep_slack[sel_to_ts_idx] - tensor_gdma_cycle[tensor_id];
          dst_cost = dst_cost >= 0 ? 0 : dst_cost;

          cycle_profit = cur_profit + dst_cost;

          if (cycle_profit > max_cycle_profit ||
              (cycle_profit == max_cycle_profit &&
               tensor_size[tensor_id] < tensor_size[sel_tensor_id])) {
            sel_ts_tensor = ts_tensor;
            max_cycle_profit = cycle_profit;
            sel_tensor_id = tensor_id;
          }
        }

        if (sel_tensor_id == -1) {
          LLVM_DEBUG(llvm::errs()
            << LOG_TAB_L5
            << "[Balance Layer][WARNING]: tensor gdma has not been moved to dest time step"
            << "\n";);
          break;
        }
      }
      pre_ts = to_ts_idx;
      if (tdma_type == TIMESTEP_LOAD) {
        --to_ts_idx;
        valid_flag = to_ts_idx >= sel_to_ts_idx;
      } else {
        ++to_ts_idx;
        valid_flag = to_ts_idx <= sel_to_ts_idx;
      }
    }
  }

  // update time step
  std::vector<TENSOR_STEP> new_tensor_timestep;
  for (uint32_t i = 0; i < ts_tensors.size(); ++i) {
    new_tensor_timestep.clear();
    for (ts_tensor = ts_tensors[i].begin(); ts_tensor != ts_tensors[i].end();
         ++ts_tensor) {
      new_tensor_timestep.push_back(*ts_tensor);
    }
    time_step->update_tensor_timestep(i, new_tensor_timestep);
  }

  delete[] timestep_slack;
  return BM_SUCCESS;
}

void GroupSteps::to_timestep_with_tsm(net_timestep* time_step) {
  struct tmp_step {
    int layer_id;
    std::vector<TENSOR_STEP> tensor_step;
  };
  std::vector<struct tmp_step> step;

  step.resize(max_step_num_ + 4);
  for (int i = 0; i < static_cast<int>(step.size()); i++) {
    step[i].layer_id = -1;
  }
  for (int i = 0; i < static_cast<int>(layers_.size()); i++) {
    // TimeStep Base Schedule
    // i + 0 DDR to TSM
    // i + 1 TSM to LMEM, DDR to LMEM
    // i + 2 BDC
    // i + 3 LMEM to TSM, LMEM to DDR
    // i + 4 TSM to DDR

    // DMA
    for (int j = 0; j < (int)ddr_to_tsm_[i].size(); j++) {
      int tid = ddr_to_tsm_[i][j];
      step[i].tensor_step.push_back(std::make_pair(tid, TIMESTEP_DDR_TO_TSM));
    }
    for (int j = 0; j < (int)tsm_to_lmem_[i].size(); j++) {
      int tid = tsm_to_lmem_[i][j];
      step[i + 1].tensor_step.push_back(std::make_pair(tid, TIMESTEP_TSM_TO_LMEM));
    }
    for (int j = 0; j < (int)loads_[i].size(); j++) {
      int tid = loads_[i][j];
      step[i + 1].tensor_step.push_back(std::make_pair(tid, TIMESTEP_LOAD));
    }

    // BDC
    step[i + 2].layer_id = layers_[i];

    // DMA
    for (int j = 0; j < (int)stores_[i].size(); j++) {
      int tid = stores_[i][j];
      step[i + 3].tensor_step.push_back(std::make_pair(tid, TIMESTEP_STORE));
    }
    for (int j = 0; j < (int)lmem_to_tsm_[i].size(); j++) {
      int tid = lmem_to_tsm_[i][j];
      step[i + 3].tensor_step.push_back(std::make_pair(tid, TIMESTEP_LMEM_TO_TSM));
    }
    for (int j = 0; j < (int)tsm_to_ddr_[i].size(); j++) {
      int tid = tsm_to_ddr_[i][j];
      step[i + 4].tensor_step.push_back(std::make_pair(tid, TIMESTEP_TSM_TO_DDR));
    }
  }

  for (int i = 0; i < (int)step.size(); i++) {
    int layer_id = step[i].layer_id;

    if ((layer_id == -1) && (step[i].tensor_step.empty())) {
      continue;
    }

    time_step->add_layer_tensor_step(layer_id, step[i].tensor_step);
  }
}

// Collect information to initialize layer_to_execute, tensor_load_store
// and timestep_num in time_step.
void GroupSteps::to_timestep(net_timestep* time_step) {
  for (int i = 0; i < static_cast<int>(layers_.size()); i++) {
    std::vector<TENSOR_STEP> tensor_step;

    if (!loads_[i].empty()) {
      for (int j = 0; j < static_cast<int>(loads_[i].size()); j++) {
        tensor_step.push_back(std::make_pair(loads_[i][j], TIMESTEP_LOAD));
      }
    }
    if (!stores_[i].empty()) {
      for (int j = 0; j < static_cast<int>(stores_[i].size()); j++) {
        tensor_step.push_back(std::make_pair(stores_[i][j], TIMESTEP_STORE));
      }
    }

    time_step->add_layer_tensor_step(layers_[i], tensor_step);
  }
}

void GroupSteps::timestep_assgin(NetGraph* net_graph, Group* group, net_timestep* time_step) {
  GroupSteps steps(net_graph);
  steps.assign(group);
  steps.to_timestep(time_step);
}

void GroupSteps::timestep_assign_with_tsm(NetGraph* net_graph, Group* group,
                                            net_timestep* time_step) {
  GroupSteps steps(net_graph);
  steps.assign_with_tsm(group);
  steps.to_timestep_with_tsm(time_step);
}

}
