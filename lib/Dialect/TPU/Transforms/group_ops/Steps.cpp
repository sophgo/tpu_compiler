/*
 * Copyright (C) Bitmain Technologies Inc.
 * All Rights Reserved.
 */
#include "Steps.hpp"
#include "Group.hpp"
#include "LayerStage.hpp"

namespace mlir {

void ClusterSteps::append(int layer, TensorStep& load_tensors, TensorStep& store_tensors) {
  if (layer == -1 && load_tensors.empty() && store_tensors.empty()) {
    return;
  }

  layers_.push_back(layer);
  loads_.push_back(load_tensors);
  stores_.push_back(store_tensors);

  max_step_num_++;
}

void ClusterSteps::insert(int layer, TensorStep& load_tensors, TensorStep& store_tensors, int pos) {
  if (layer == -1 && load_tensors.empty() && store_tensors.empty()) {
    return;
  }

  layers_.insert(layers_.begin() + pos, layer);
  loads_.insert(loads_.begin() + pos, load_tensors);
  stores_.insert(stores_.begin() + pos, store_tensors);

  max_step_num_++;
}

static void dump(const vector<int>& v) {
  for (int i = 0; i < static_cast<int>(v.size()); i++) {
    cout << v[i];
  }
  cout << endl;
}

static vector<int> intersection(vector<int> v1, vector<int> v2) {
  vector<int> v;
  sort(v1.begin(), v1.end());
  sort(v2.begin(), v2.end());
  set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), back_inserter(v));
  if (!v.empty()) {
    cout << "intersection: ";
    dump(v);
  }

  return v;
}

static vector<int> difference(vector<int> v1, vector<int> v2) {
  vector<int> v;
  sort(v1.begin(), v1.end());
  sort(v2.begin(), v2.end());
  set_difference(v1.begin(), v1.end(), v2.begin(), v2.end(), back_inserter(v));
  if (!v.empty()) {
    cout << "difference: ";
    dump(v);
  }

  return v;
}

// This functions is used to adjust the data position of layers_, loads_ and stores.
// The final result is:
//   1. Insert -1 at the beginning and the end of layers.
//   2. Insert 2 null at the end of loads_.
//   3. Insert 2 null at the beginning of stores_
void ClusterSteps::rearrange_steps() {
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
    TensorStep& cur = stores_[i];
    if (stores_[i].empty()) {
      continue;
    }
    if (i == stores_.size() - 1) {
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

    const vector<int>& in_tensors = net_graph_->get_in_tensors_of_layer(id);
    const vector<int>& out_tensors = net_graph_->get_out_tensors_of_layer(id);

    if (!loads_[i].empty()) {
      vector<int> conflict = intersection(in_tensors, loads_[i]);
      if (!conflict.empty()) {
        restart = true;
        loads_[i] = difference(loads_[i], conflict);
        loads_[i - 1].insert(loads_[i - 1].end(), conflict.begin(), conflict.end());
      }
    }

    if (!stores_[i].empty()) {
      vector<int> conflict = intersection(out_tensors, stores_[i]);
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

void ClusterSteps::assign_with_tsm(Group* cluster) {
  set<int> tensors_in_lmem;
  set<int> tensors_in_tsm;

  // Ignore tg layer
  if (cluster->size() == 1) {
    return;
  }

  for (auto id : cluster->layers()) {
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
    const vector<int>& in_tensors = net_graph_->get_in_tensors_of_layer(id);
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
    const vector<int>& out_tensors = net_graph_->get_out_tensors_of_layer(id);
    for (int j = 0; j < static_cast<int>(out_tensors.size()); ++j) {
      int tid = out_tensors[j];
      tensors_in_lmem.insert(tid);
      if (cluster->is_group_out_tensor(tid)) {
        // TODO(arcbbb): keep tensor in TSM for next user.
        // Now we flush it to DDR directly.
        lmem_to_ddr.push_back(tid);
      }
    }

    // Add one step in ClusterSteps
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

// This funtion is used to collect information in the cluster.
// This information including:
//   1. layers_ save the contained layers.
//   2. loads_ save the tensor need to load.
//   3. stores_ save the tensor need to store.
void ClusterSteps::assign(Group* cluster) {
  set<int> tensors_in_lmem;

  bool single_layer_cluster = (cluster->size() == 1);

  for (auto id : cluster->layers()) {
    int layer_step = -1;

    TensorStep load_tensors;
    TensorStep store_tensors;

    // push layer to layer_step if not special case.
    bool ignore_concat_layer = net_graph_->is_concat_special_case(id, -1);
    if (!ignore_concat_layer) {
      layer_step = id;

      // push all layers' in tensors to tensors_in_lmem and tensor step
      const vector<int>& in_tensors = net_graph_->get_in_tensors_of_layer(id);
      for (int j = 0; j < static_cast<int>(in_tensors.size()); ++j) {
        auto ret = tensors_in_lmem.insert(in_tensors[j]);
        if (ret.second == true) {
          // cout << "load tensor: " << in_tensors[j] << endl;
          load_tensors.push_back(in_tensors[j]);
        }
      }
    }

    // push out tensors of layer to local mem.
    const vector<int>& out_tensors = net_graph_->get_out_tensors_of_layer(id);
    for (int j = 0; j < static_cast<int>(out_tensors.size()); ++j) {
      int tid = out_tensors[j];
      tensors_in_lmem.insert(tid);

      const vector<int>& to_layers = net_graph_->get_tensor_to_layer(tid);
      // Why need ignore_concat_layer?
      if (cluster->is_group_out_tensor(tid)) {
        // you are not concat layer, and your output tensor is shared
        store_tensors.push_back(tid);
        // if your consumer is concat layer in next group,
        // add this tensor to the ignore list of concat layer
        std::vector<std::pair<int, int>> ignore_pair;
        bool is_concat_in_place = true;
        for (int k = 0; k < static_cast<int>(to_layers.size()); ++k) {
          if (net_graph_->is_concat_optimized_case(to_layers[k], tid, cluster->size())) {
            vector<int> in_tensors = net_graph_->get_in_tensors_of_layer(to_layers[k]);
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
            ImConcat* im_layer = (ImConcat*)net_graph_->get_layer_by_id(to_layers[ignore_unit.first]);
            im_layer->ignored_bottoms.insert(ignore_unit.second);
          }
        }
      } else {
        for (int k = 0; k < static_cast<int>(to_layers.size()); ++k) {
          int to_layer = to_layers[k];
          if (net_graph_->is_concat_special_case(to_layer, tid, cluster->size())) {
            int concat_out_tensor = net_graph_->get_out_tensors_of_layer(to_layer)[0];
            if (cluster->is_group_out_tensor(concat_out_tensor)) {
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
    const vector<int>& in_tensors = net_graph->get_in_tensors_of_layer(layer_id);

    if (std::find(in_tensors.begin(), in_tensors.end(), tensor_id) != in_tensors.end()) {
      return true;
    }

    const vector<int>& out_tensors = net_graph->get_out_tensors_of_layer(layer_id);
    if (std::find(out_tensors.begin(), out_tensors.end(), tensor_id) != out_tensors.end()) {
      return true;
    }
  }
  return false;
}

// This fucntion moved up/down gdma load/store operations for balancing bdc and
// gdma cycle.
// The array timestep_cycle_slack holds the number of execution cycles per step.
// When the value is greater than 0, it means that the execution cycle of bdc
// instruction covers gdma instruction, otherwise we can move up/down load/store
// operations to keep balance.
bmerr_t ClusterSteps::balance_gdma_bdc_steps(NetGraph* net_graph, Group* cluster,
                                             net_timestep* time_step,
                                             const pair<int, int>& nsecs_and_hsecs) {
  if (cluster->size() == 1) {
    return BM_SUCCESS;
  }

  int nsecs = nsecs_and_hsecs.first;
  int hsecs = nsecs_and_hsecs.second;

  bmerr_t status = cluster->update_tensor_slices(nsecs, hsecs);
  if (status == BM_ERR_FAILURE) {
    return BM_ERR_FAILURE;
  }

  int timestep_num = time_step->get_timestep_num();
  int* timestep_cycle_slack = new int[timestep_num];
  for (int i = 0; i < timestep_num; ++i) {
    timestep_cycle_slack[i] = 0;
  }

  map<int, int> tensor_to_gdma_cycle;
  map<int, int> tensor_to_buffer_size;
  vector<list<TENSOR_STEP>> tensor_timesteps;

  // get cycle slack of each time step
  int tensor_id;
  int tensor_gdma_cycle;
  int tensor_local_size;

  list<TENSOR_STEP> list_tensors;

  for (int i = 0; i < timestep_num; ++i) {
    int cur_layer = time_step->get_layer(i);
    const vector<TENSOR_STEP>& cur_tensors = time_step->get_tensors(i);
    // add layer cycle for each time step
    if (cur_layer != -1) {
      //timestep_cycle_slack[i] += get_layer_cycle_count(net_graph, cur_layer);
      timestep_cycle_slack[i] += 0;//get_layer_cycle_count(net_graph, cur_layer);
    }
    // sub tensor gdma cycle for each time step
    list_tensors.clear();
    for (u32 j = 0; j < cur_tensors.size(); ++j) {
      tensor_id = cur_tensors[j].first;
      // get cycle
      //tensor_gdma_cycle = get_gdma_cycle_count(net_graph, cur_tensors[j]);
      //(MK-TODO)
      tensor_gdma_cycle = 0;
      tensor_to_gdma_cycle[tensor_id] = tensor_gdma_cycle;

      // get local memory require size
      Tensor* tensor = net_graph->get_tensor_by_id(tensor_id);
      tensor_local_size = tensor->lmem_size();

      tensor_to_buffer_size[tensor_id] = tensor_local_size;

      // generate tensor gdma list
      list_tensors.push_back(cur_tensors[j]);

      // slack minus gdma cycle
      timestep_cycle_slack[i] -= tensor_gdma_cycle;
    }
    tensor_timesteps.push_back(list_tensors);
  }

  // move gdma time step for reducing total cycle of time steps
  int cycle_profit, max_cycle_profit;
  int best_ts_to, best_sel_tensor;
  int cur_profit, dst_cost;
  list<TENSOR_STEP>::iterator list_iter;
  list<TENSOR_STEP>::iterator sel_list_iter;

  int to_ts_i, pre_ts;
  bool valid_flag;
  TIMESTEP_LD_ST move_type;

  for (int ts_i = 0; ts_i < timestep_num;) {
    // if need to move gdma time step
    int cur_slack = timestep_cycle_slack[ts_i];
    if (cur_slack >= 0) {
      ++ts_i;
      continue;
    }

    // get the src gdma tensor and dst time step
    max_cycle_profit = 0;
    best_ts_to = -1;

    for (list_iter = tensor_timesteps[ts_i].begin(); list_iter != tensor_timesteps[ts_i].end();
         ++list_iter) {
      tensor_id = list_iter->first;
      int cur_new_slack = cur_slack + tensor_to_gdma_cycle[tensor_id];
      cur_profit = (cur_new_slack >= 0 ? 0 : cur_new_slack) - cur_slack;

      // add for software pipeline
      int range_end = time_step->tensor_range_end_timestep(*list_iter);

      if (list_iter->second == TIMESTEP_LOAD) {
        to_ts_i = ts_i - 1;
        // valid_flag = to_ts_i >=0;
        valid_flag = to_ts_i >= range_end;
      } else {
        to_ts_i = ts_i + 1;
        // valid_flag = to_ts_i < timestep_num;
        valid_flag = to_ts_i <= range_end;
      }

      while (valid_flag) {
        if (!(list_iter->second == TIMESTEP_STORE &&
              tensor_conflict_with_npu_exe(net_graph, time_step, tensor_id, to_ts_i))) {
          if (timestep_cycle_slack[to_ts_i] > 0) {
            dst_cost = timestep_cycle_slack[to_ts_i] - tensor_to_gdma_cycle[tensor_id];
            dst_cost = dst_cost >= 0 ? 0 : dst_cost;

            cycle_profit = cur_profit + dst_cost;

            if (cycle_profit > max_cycle_profit) {
              max_cycle_profit = cycle_profit;
              best_ts_to = to_ts_i;
              sel_list_iter = list_iter;
              best_sel_tensor = tensor_id;
            } else if (cycle_profit == max_cycle_profit && best_ts_to != -1) {
              if ((tensor_to_buffer_size[tensor_id] * (abs(to_ts_i - ts_i) + 1)) <
                  (tensor_to_buffer_size[best_sel_tensor] * (abs(best_ts_to - ts_i) + 1))) {
                max_cycle_profit = cycle_profit;
                best_ts_to = to_ts_i;
                sel_list_iter = list_iter;
                best_sel_tensor = tensor_id;
              }
            }
          }
        }
        if (list_iter->second == TIMESTEP_LOAD) {
          --to_ts_i;
          // valid_flag = to_ts_i >=0;
          valid_flag = to_ts_i >= range_end;
        } else {
          ++to_ts_i;
          // valid_flag = to_ts_i < timestep_num;
          valid_flag = to_ts_i <= range_end;
        }
      }
    }
    if (best_ts_to == -1) {
      ++ts_i;
      continue;
    }

    // bubble src tensor gmda
    if (sel_list_iter->second == TIMESTEP_LOAD) {
      to_ts_i = ts_i - 1;
      valid_flag = to_ts_i >= best_ts_to;
      move_type = TIMESTEP_LOAD;
    } else {
      to_ts_i = ts_i + 1;
      valid_flag = to_ts_i <= best_ts_to;
      move_type = TIMESTEP_STORE;
    }
    pre_ts = ts_i;

    while (valid_flag) {
      // bubble gdma tensor
      tensor_timesteps[to_ts_i].push_back(*sel_list_iter);
      timestep_cycle_slack[to_ts_i] -= tensor_to_gdma_cycle[best_sel_tensor];
      tensor_timesteps[pre_ts].erase(sel_list_iter);
      timestep_cycle_slack[pre_ts] += tensor_to_gdma_cycle[best_sel_tensor];

      if (to_ts_i == best_ts_to) {
        break;
      }

      // find next tensor in the current bubble timestep
      if (move_type == TIMESTEP_STORE &&
          tensor_conflict_with_npu_exe(net_graph, time_step, best_sel_tensor, to_ts_i)) {
        sel_list_iter = tensor_timesteps[to_ts_i].end();
        --sel_list_iter;
      } else {
        max_cycle_profit = 0;
        best_sel_tensor = -1;
        for (list_iter = tensor_timesteps[to_ts_i].begin();
             list_iter != tensor_timesteps[to_ts_i].end(); ++list_iter) {
          if (move_type != list_iter->second) {
            continue;
          }

          // add for software pipeline
          int new_range_end = time_step->tensor_range_end_timestep(*list_iter);
          if ((move_type == TIMESTEP_LOAD && new_range_end > best_ts_to) ||
              (move_type == TIMESTEP_STORE && new_range_end < best_ts_to)) {
            continue;
          }

          tensor_id = list_iter->first;
          int cur_new_slack = timestep_cycle_slack[to_ts_i] + tensor_to_gdma_cycle[tensor_id];
          cur_profit = (cur_new_slack >= 0 ? 0 : cur_new_slack) -
                       (timestep_cycle_slack[to_ts_i] >= 0 ? 0 : timestep_cycle_slack[to_ts_i]);

          dst_cost = timestep_cycle_slack[best_ts_to] - tensor_to_gdma_cycle[tensor_id];
          dst_cost = dst_cost >= 0 ? 0 : dst_cost;

          cycle_profit = cur_profit + dst_cost;

          if (cycle_profit > max_cycle_profit ||
              (cycle_profit == max_cycle_profit &&
               tensor_to_buffer_size[tensor_id] < tensor_to_buffer_size[best_sel_tensor])) {
            sel_list_iter = list_iter;
            max_cycle_profit = cycle_profit;
            best_sel_tensor = tensor_id;
          }
        }

        if (best_sel_tensor == -1) {
          llvm::errs() << "WARNING: tensor gdma has not been moved to dest time step" << "\n";
          break;
        }
      }
      pre_ts = to_ts_i;
      if (move_type == TIMESTEP_LOAD) {
        --to_ts_i;
        valid_flag = to_ts_i >= best_ts_to;
      } else {
        ++to_ts_i;
        valid_flag = to_ts_i <= best_ts_to;
      }
    }
  }

  // update time step
  vector<TENSOR_STEP> new_tensor_timestep;
  for (u32 i = 0; i < tensor_timesteps.size(); ++i) {
    new_tensor_timestep.clear();
    for (list_iter = tensor_timesteps[i].begin(); list_iter != tensor_timesteps[i].end();
         ++list_iter) {
      new_tensor_timestep.push_back(*list_iter);
    }
    time_step->update_tensor_timestep(i, new_tensor_timestep);
  }

  delete[] timestep_cycle_slack;
  return status;
}

void ClusterSteps::to_timestep_with_tsm(net_timestep* time_step) {
  struct tmp_step {
    int layer_id;
    vector<TENSOR_STEP> tensor_step;
  };
  vector<struct tmp_step> step;

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
    for (int j = 0; j < ddr_to_tsm_[i].size(); j++) {
      int tid = ddr_to_tsm_[i][j];
      step[i].tensor_step.push_back(make_pair(tid, TIMESTEP_DDR_TO_TSM));
    }
    for (int j = 0; j < tsm_to_lmem_[i].size(); j++) {
      int tid = tsm_to_lmem_[i][j];
      step[i + 1].tensor_step.push_back(make_pair(tid, TIMESTEP_TSM_TO_LMEM));
    }
    for (int j = 0; j < loads_[i].size(); j++) {
      int tid = loads_[i][j];
      step[i + 1].tensor_step.push_back(make_pair(tid, TIMESTEP_LOAD));
    }

    // BDC
    step[i + 2].layer_id = layers_[i];

    // DMA
    for (int j = 0; j < stores_[i].size(); j++) {
      int tid = stores_[i][j];
      step[i + 3].tensor_step.push_back(make_pair(tid, TIMESTEP_STORE));
    }
    for (int j = 0; j < lmem_to_tsm_[i].size(); j++) {
      int tid = lmem_to_tsm_[i][j];
      step[i + 3].tensor_step.push_back(make_pair(tid, TIMESTEP_LMEM_TO_TSM));
    }
    for (int j = 0; j < tsm_to_ddr_[i].size(); j++) {
      int tid = tsm_to_ddr_[i][j];
      step[i + 4].tensor_step.push_back(make_pair(tid, TIMESTEP_TSM_TO_DDR));
    }
  }

  for (int i = 0; i < step.size(); i++) {
    int layer_id = step[i].layer_id;

    if ((layer_id == -1) && (step[i].tensor_step.empty())) {
      continue;
    }

    time_step->add_layer_tensor_step(layer_id, step[i].tensor_step);
  }
}

// Collect information to initialize layer_to_execute, tensor_load_store
// and timestep_num in time_step.
void ClusterSteps::to_timestep(net_timestep* time_step) {
  for (int i = 0; i < static_cast<int>(layers_.size()); i++) {
    vector<TENSOR_STEP> tensor_step;

    if (!loads_[i].empty()) {
      for (int j = 0; j < static_cast<int>(loads_[i].size()); j++) {
        tensor_step.push_back(make_pair(loads_[i][j], TIMESTEP_LOAD));
      }
    }
    if (!stores_[i].empty()) {
      for (int j = 0; j < static_cast<int>(stores_[i].size()); j++) {
        tensor_step.push_back(make_pair(stores_[i][j], TIMESTEP_STORE));
      }
    }

    time_step->add_layer_tensor_step(layers_[i], tensor_step);
  }
}

void ClusterSteps::timestep_assgin(NetGraph* net_graph, Group* cluster, net_timestep* time_step) {
  ClusterSteps steps(net_graph);
  steps.assign(cluster);
  steps.to_timestep(time_step);
}

void ClusterSteps::timestep_assign_with_tsm(NetGraph* net_graph, Group* cluster,
                                            net_timestep* time_step) {
  ClusterSteps steps(net_graph);
  steps.assign_with_tsm(cluster);
  steps.to_timestep_with_tsm(time_step);
}

}
