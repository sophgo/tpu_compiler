#include "LayerStage.hpp"
#include <algorithm>
#include <cmath>
#include "NetGraph.hpp"
#include "Group.hpp"

#define DEBUG_TYPE "optimizer_cluster"

namespace mlir {

#define SPLIT_RATIO 0.9

net_timestep::net_timestep(NetGraph* net_graph) : net_graph_(net_graph) {
  timestep_num = 0;
  layer_to_execute.clear();
  tensor_load_store.clear();
}

net_timestep::net_timestep(const net_timestep& src) {
  net_graph_ = src.net_graph_;
  timestep_num = src.timestep_num;
  layer_to_execute = src.layer_to_execute;
  tensor_load_store = src.tensor_load_store;
  mem_buffer = src.mem_buffer;
  hold_coeff_tensor = src.hold_coeff_tensor;
}

net_timestep::~net_timestep() = default;

void net_timestep::add_layer_tensor_step(int layer_step, const vector<TENSOR_STEP>& tensor_step) {
  layer_to_execute.push_back(layer_step);
  tensor_load_store.push_back(tensor_step);
  timestep_num++;
}

int net_timestep::get_timestep_num() { return timestep_num; }

int net_timestep::get_layer(int time_step) { return layer_to_execute[time_step]; }

const vector<TENSOR_STEP>& net_timestep::get_tensors(int time_step) {
  return tensor_load_store[time_step];
}

void net_timestep::add_mem_buffer(int start_step, int end_step, int id, bool imm_tensor) {
  mem_buffer_key_t key;
  mem_buffer_value_t value;

  key.start_timestep = start_step;
  key.id_num = id;
  key.is_layer_imm = imm_tensor;

  value.end_timestep = end_step;
  value.local_mem_offset = 0;
  value.local_mem_size = 0;

  mem_buffer[key] = value;
}

void net_timestep::generate_tsm_buffer(bool one_loop) {
  tsm_buffer.clear();
  for (int i = 0; i < timestep_num; ++i) {
    for (u32 j = 0; j < tensor_load_store[i].size(); ++j) {
      int tid = tensor_load_store[i][j].first;
      auto ldst_type = tensor_load_store[i][j].second;

      if ((ldst_type == TIMESTEP_DDR_TO_TSM) || (ldst_type == TIMESTEP_LMEM_TO_TSM)) {
        // add mem buffer
        tensor_mem_t mem_param;
        Tensor* tensor = net_graph_->get_tensor_by_id(tid);
        mem_param.start_timestep = i;
        mem_param.end_timestep = -1;
        mem_param.size = tensor->gmem_size();
        mem_param.offset = 0;
        tsm_buffer[tid] = mem_param;

      } else if ((ldst_type == TIMESTEP_TSM_TO_DDR) || (ldst_type == TIMESTEP_TSM_TO_LMEM)) {
        tensor_type_t tensor_type = net_graph_->get_tensor_type(tid);
        // look up mem buffer
        assert(tsm_buffer.find(tid) != tsm_buffer.end());
        if (tsm_buffer.find(tid) == tsm_buffer.end()) {
          cout << "ERROR: tid not found in tsm_buffer\n";
          exit(-1);
        }
        tensor_mem_t& mem_param = tsm_buffer[tid];

        if (tensor_type == TENSOR_NEURON) {
          // Neuron
          mem_param.end_timestep = i;
        } else {
          // Weight
          if (ldst_type == TIMESTEP_TSM_TO_DDR) {
            cout << "ERROR: move weight back to DDR\n";
            exit(-1);
          }
          // TIMESTEP_TSM_TO_LMEM
          if (one_loop) {
            mem_param.end_timestep = i;
          } else {
            mem_param.end_timestep = timestep_num;
          }
        }
      }
    }
  }
}

// Since the load store information for each step has been collected,
// this function is used to determine the def-use life cycle of each load/store tensor,
// and save this information in mem_buf.
void net_timestep::generate_mem_buffer() {
  int layer_id, tensor_id;
  map<mem_buffer_key_t, mem_buffer_value_t>::iterator iter;
  int begin_timestep;

  mem_buffer.clear();

  for (int i = 0; i < timestep_num; ++i) {
    if (layer_to_execute[i] != -1) {
      layer_id = layer_to_execute[i];
      const ImLayer* layer = net_graph_->get_layer_by_id(layer_id);

      // consumers
      for (auto& tensor : layer->in_tensors) {
        begin_timestep = -1;
        for (iter = mem_buffer.begin(); iter != mem_buffer.end(); ++iter) {
          if ((iter->first).id_num == tensor->id() && !(iter->first).is_layer_imm) {
            begin_timestep = (iter->first).start_timestep;
          }
        }
        if (begin_timestep == -1) {
          cout << "ERROR: cannot find the producer that is consumed by layer, tensor_id "
               << tensor->id() << "\n";
          assert(0);
        }
        add_mem_buffer(begin_timestep, i, tensor->id(), false);
      }

      // producers
      //<! place after consumers for prevent input/output with the same tensor
      for (auto& tensor : layer->out_tensors) {
        add_mem_buffer(i, -1, tensor->id(), false);
      }

      for (auto& tensor : layer->imm_tensors) {
        add_mem_buffer(i, i, tensor->id(), true);
      }
    }

    // process tensor timestep
    if (!tensor_load_store[i].empty()) {
      for (u32 j = 0; j < tensor_load_store[i].size(); ++j) {
        tensor_id = tensor_load_store[i][j].first;

        auto ldst_type = tensor_load_store[i][j].second;
        if ((ldst_type == TIMESTEP_LOAD) || (ldst_type == TIMESTEP_TSM_TO_LMEM)) {
          add_mem_buffer(i, -1, tensor_id, false);

        } else if ((ldst_type == TIMESTEP_STORE) || (ldst_type == TIMESTEP_LMEM_TO_TSM)) {
          begin_timestep = -1;
          for (iter = mem_buffer.begin(); iter != mem_buffer.end(); ++iter) {
            if ((iter->first).id_num == tensor_id && !(iter->first).is_layer_imm) {
              begin_timestep = (iter->first).start_timestep;
            }
          }

          if (begin_timestep == -1) {
            cout << "ERROR: cannot find the producer that is consumed by storing, tensor_id "
                 << tensor_id << "\n";
            assert(0);
          }

          add_mem_buffer(begin_timestep, i, tensor_id, false);
        }
      }
    }
  }

  for (iter = mem_buffer.begin(); iter != mem_buffer.end(); ++iter) {
    if ((iter->second).end_timestep == -1) {
      if ((iter->first).is_layer_imm) {
        cout << "ERROR: layer imm " << (iter->first).id_num << " has no end timestep." << "\n";
      } else {
        cout << "ERROR: tensor " << (iter->first).id_num << " has no end timestep." << "\n";
      }
      assert(0);
    }
  }
}

// This fucntion does the following things:
//   1. Save reused coefficients.
//   2. For each layer of in/out tensor, build time step information, which is stored in membuf.
//   3. Initialize the local memory size in the membuf
void net_timestep::update_mem_buffer_size() {
  generate_hold_coeff_tensor();
  generate_mem_buffer();

  int tensor_local_mem_size;

  map<mem_buffer_key_t, mem_buffer_value_t>::iterator iter;

  for (iter = mem_buffer.begin(); iter != mem_buffer.end(); ++iter) {
    int tensor_id = (iter->first).id_num;
    Tensor* tensor = net_graph_->get_tensor_by_id(tensor_id);
    tensor_local_mem_size = tensor->lmem_size();

    // Disable temporarily. Reopen if needed.
    // // handled fc;
    // u32 updated_fc_tensor_local_mem_size = 0;
    // vector<int> layer_ids = net_graph_->get_tensor_to_layer(tensor_id);
    // for (int i = 0; i < layer_ids.size(); i++) {
    //   const ImLayer* layer = net_graph_->get_layer_by_id(layer_ids[i]);
    //   if (layer->type() == IR_INNERPRODUCT) {
    //     if ((net_graph_->get_tensor_type(tensor_id) == TENSOR_NEURON) ||
    //         (net_graph_->get_tensor_type(tensor_id) == TENSOR_COEFF) ||
    //         (net_graph_->get_tensor_type(tensor_id) == TENSOR_MATRIX)) {
    //       updated_fc_tensor_local_mem_size = tensor->lmem_size(true);
    //     }
    //   }
    // }

    // DEBUG_BMNET(std::cout << tensor->name() << " local mem size: "
    //                       << tensor_local_mem_size << ". fc size: "
    //                       << updated_fc_tensor_local_mem_size << "\n";);
    // if (tensor_local_mem_size < updated_fc_tensor_local_mem_size) {
    //   tensor_local_mem_size = updated_fc_tensor_local_mem_size;
    // }

    if (tensor_local_mem_size < 0) {
      cout << "wrong local mem size " << tensor_local_mem_size << "\n";
      assert(0);
    }

    (iter->second).local_mem_size = tensor_local_mem_size;
  }
}

const mem_buffer_value_t* net_timestep::get_mem_buffer_value(const mem_buffer_key_t* key) {
  map<mem_buffer_key_t, mem_buffer_value_t>::iterator iter;
  iter = mem_buffer.find(*key);
  if (iter != mem_buffer.end()) {
    return &(iter->second);
  } else {
    cout << "No key " << key->id_num << " in mem buffer,"
         << "start step id " << key->start_timestep << ", imm: " << key->is_layer_imm << "\n";
    assert(0);
  }
}

int net_timestep::get_mem_buffer_size(const mem_buffer_key_t* key) {
  map<mem_buffer_key_t, mem_buffer_value_t>::iterator iter;
  iter = mem_buffer.find(*key);
  if (iter != mem_buffer.end()) {
    return (iter->second).local_mem_size;
  } else {
    cout << "No key in mem buffer" << "\n";
    assert(0);
  }
}

int net_timestep::get_mem_buffer_offset(const mem_buffer_key_t* key) {
  map<mem_buffer_key_t, mem_buffer_value_t>::iterator iter;
  iter = mem_buffer.find(*key);
  if (iter != mem_buffer.end()) {
    return (iter->second).local_mem_offset;
  } else {
    cout << "No key in mem buffer" << "\n";
    assert(0);
  }
}

int net_timestep::get_mem_buffer_end_timestep(const mem_buffer_key_t* key) {
  map<mem_buffer_key_t, mem_buffer_value_t>::iterator iter;
  iter = mem_buffer.find(*key);
  if (iter != mem_buffer.end()) {
    return (iter->second).end_timestep;
  } else {
    cout << "No key in mem buffer" << "\n";
    assert(0);
  }
}

void net_timestep::set_tsm_offset(int tensor_id, uint64_t offset) {
  tsm_buffer[tensor_id].offset = offset;
}

void net_timestep::set_local_mem_offset(const mem_buffer_key_t* key, int local_mem_offset) {
  map<mem_buffer_key_t, mem_buffer_value_t>::iterator iter;
  iter = mem_buffer.find(*key);
  if (iter != mem_buffer.end()) {
    (iter->second).local_mem_offset = local_mem_offset;
  } else {
    cout << "No key in mem buffer" << "\n";
    assert(0);
  }
}

void net_timestep::show_timestep() {
    show_timestep(cout);
}

void net_timestep::show_timestep(std::ostream& pOs) {
  pOs << "******show time step******" << "\n";
  pOs << "Bank size: " << LOCAL_BANK_SIZE << "\n";
  for (int time_idx = 0; time_idx < timestep_num; ++time_idx) {
    pOs << "Step <" << time_idx << "> ";
    if (layer_to_execute[time_idx] != -1) {
      int id = layer_to_execute[time_idx];
      const ImLayer* layer = net_graph_->get_layer_by_id(id);
      pOs << "layer_id " << id << ", ";
      pOs << layer->name() << ", " << layer->type_name();
      pOs << ", in tensors: ";
      for (auto& tensor : layer->in_tensors) {
        pOs << tensor->id() << ", ";
      }
      pOs << "out tensors: ";
      for (auto& tensor : layer->out_tensors) {
        pOs << tensor->id() << ", ";
      }
    }

    for (u32 i = 0; i < tensor_load_store[time_idx].size(); ++i) {
      if (tensor_load_store[time_idx][i].second == TIMESTEP_LOAD) {
        pOs << "ddr-lmem tensor_id ";
      } else if (tensor_load_store[time_idx][i].second == TIMESTEP_DDR_TO_TSM) {
        pOs << "ddr-tsm tensor_id ";
      } else if (tensor_load_store[time_idx][i].second == TIMESTEP_TSM_TO_LMEM) {
        pOs << "tsm-lmem tensor_id ";
      } else if (tensor_load_store[time_idx][i].second == TIMESTEP_TSM_TO_DDR) {
        pOs << "tsm-ddr tensor_id ";
      } else if (tensor_load_store[time_idx][i].second == TIMESTEP_LMEM_TO_TSM) {
        pOs << "lmem-tsm tensor_id ";
      } else if (tensor_load_store[time_idx][i].second == TIMESTEP_STORE) {
        pOs << "lmem-ddr tensor_id ";
      }
      pOs << tensor_load_store[time_idx][i].first << ", ";
    }
    pOs << "\n";

    map<mem_buffer_key_t, mem_buffer_value_t>::iterator iter;
    mem_buffer_key_t key;
    mem_buffer_value_t value;

    for (iter = mem_buffer.begin(); iter != mem_buffer.end(); ++iter) {
      key = iter->first;
      value = iter->second;
      if (key.start_timestep > time_idx || time_idx > value.end_timestep) {
        continue;
      }

      pOs << value.local_mem_offset << " \t~ " << value.local_mem_size + value.local_mem_offset
          << " \tbank:" << (value.local_mem_offset / 65536) << " "
          << ((value.local_mem_size + value.local_mem_offset) / 65536) << "\t[" << key.id_num << "]"
          << net_graph_->get_tensor_by_id(key.id_num)->name() << "\t"
          << net_graph_->get_tensor_by_id(key.id_num)->lmem_size() << " "
          << (is_tensor_hold_in_memory(key.id_num) ? "*" : "") << "\n";
    }
  }
}

void net_timestep::show_mem_buffer() { show_mem_buffer(cout); }

void net_timestep::show_tsm_buffer() { show_tsm_buffer(cout); }

void net_timestep::show_tsm_buffer(std::ostream& pOs) {
  pOs << "******show tsm buffer******" << "\n";
  for (auto iter = tsm_buffer.begin(); iter != tsm_buffer.end(); ++iter) {
    int tid = iter->first;
    tensor_mem_t& param = iter->second;
    pOs << "tsm buffer: tensor_id " << tid << " start " << param.start_timestep << " end "
        << param.end_timestep << " offset " << param.offset << " size " << param.size << "\n";
  }
}

void net_timestep::show_mem_buffer(std::ostream& pOs) {
  pOs << "******show mem buffer******" << "\n";
  map<mem_buffer_key_t, mem_buffer_value_t>::iterator iter;
  mem_buffer_key_t key;
  mem_buffer_value_t value;
  int index = 0;
  for (iter = mem_buffer.begin(); iter != mem_buffer.end(); ++iter) {
    key = iter->first;
    value = iter->second;
    if (key.is_layer_imm) {
      pOs << "mem buffer " << index << ": layer_id " << key.id_num << " start "
          << key.start_timestep << " end " << value.end_timestep << " local_mem_offset "
          << value.local_mem_offset << " local_mem_size " << value.local_mem_size << "\n";
    } else {
      pOs << "mem buffer " << index << ": tensor_id " << key.id_num << " start "
          << key.start_timestep << " end " << value.end_timestep << " local_mem_offset "
          << value.local_mem_offset << " local_mem_size " << value.local_mem_size << "\n";
    }
    index++;
  }
}

void net_timestep::generate_hold_coeff_tensor() {
  // hope that all coefficients can be hold for reusing coefficients
  hold_coeff_tensor.clear();

  for (int i = 0; i < timestep_num; ++i) {
    for (u32 j = 0; j < tensor_load_store[i].size(); ++j) {
      int tensor_id = tensor_load_store[i][j].first;
      tensor_type_t tensor_type = net_graph_->get_tensor_type(tensor_id);

      if (tensor_type == TENSOR_COEFF || tensor_type == TENSOR_BIAS ||
          tensor_type == TENSOR_COEFF_WINOGRAD || tensor_type == TENSOR_DEPTHCONV_OPD1 ) {
        hold_coeff_tensor[tensor_id] = i;

      } else if (tensor_type == TENSOR_COEFF_NEURON || tensor_type == TENSOR_DEPTHCONV_OPD1 ||
                 tensor_type == TENSOR_NEURON_AS_COEFF) {
        int n = net_graph_->get_tensor_nums(tensor_id);
        int h = net_graph_->get_tensor_height(tensor_id);
        if (n == 1 && h == 1) {
          hold_coeff_tensor[tensor_id] = i;
        }
      }
    }
  }
}

bool net_timestep::is_tensor_hold_in_memory(int tensor_id) {
  map<int, int>::iterator iter = hold_coeff_tensor.find(tensor_id);
  if (iter != hold_coeff_tensor.end()) {
    return true;
  } else {
    return false;
  }
}

bool net_timestep::is_tensor_weight(tensor_type_t tensor_type) {
  if (tensor_type == TENSOR_COEFF || tensor_type == TENSOR_BIAS ||
      tensor_type == TENSOR_COEFF_NEURON || tensor_type == TENSOR_COEFF_WINOGRAD ||
      tensor_type == TENSOR_NEURON_AS_COEFF) {
    return true;
  } else {
    return false;
  }
}

int net_timestep::get_timestep_coeff_memory_req(int time_step) {
  int total_memory_req = 0;
  int start_timestep, end_timestep;
  map<mem_buffer_key_t, mem_buffer_value_t>::iterator iter;

  for (iter = mem_buffer.begin(); iter != mem_buffer.end(); ++iter) {
    if (!(iter->first).is_layer_imm &&
        is_tensor_weight(net_graph_->get_tensor_type((iter->first).id_num))) {
      start_timestep = (iter->first).start_timestep;
      end_timestep = (iter->second).end_timestep;
      if (is_tensor_hold_in_memory((iter->first).id_num)) {
        total_memory_req += (iter->second).local_mem_size;
      } else if ((time_step >= start_timestep && time_step <= end_timestep) ||
                 ((start_timestep > end_timestep) &&
                  (time_step >= start_timestep || time_step <= end_timestep))) {
        total_memory_req += (iter->second).local_mem_size;
      }
    }
  }

  return total_memory_req;
}

int net_timestep::get_timestep_memory_req(int time_step) {
  int total_memory_req = 0;
  map<mem_buffer_key_t, mem_buffer_value_t>::iterator iter;

  for (iter = mem_buffer.begin(); iter != mem_buffer.end(); ++iter) {
    int start_timestep = (iter->first).start_timestep;
    int end_timestep = (iter->second).end_timestep;

    if (!(iter->first).is_layer_imm && is_tensor_hold_in_memory((iter->first).id_num)) {
      total_memory_req += (iter->second).local_mem_size;
    } else if ((time_step >= start_timestep && time_step <= end_timestep) ||
               ((start_timestep > end_timestep) &&
                (time_step >= start_timestep || time_step <= end_timestep))) {
      total_memory_req += (iter->second).local_mem_size;
    }
  }

  return total_memory_req;
}

// Returns true if a suitable split result is found. The result will
// be saved in nsecs_and_hsecs, that holds the slcing number of cuts
// by n and h. The slicing strategy is based on the size of local memory
// used by each step.
bmerr_t net_timestep::find_best_split(Group* cluster, int batch_num,
                                      pair<int, int>& nsecs_and_hsecs) {
  nsecs_and_hsecs.first = batch_num;
  nsecs_and_hsecs.second = 1;

  // slice group into n pieces,
  // then we could find the best splits
  if (BM_ERR_FAILURE == cluster->update_tensor_slices(batch_num, 1)) {
    return BM_ERR_FAILURE;
  }

  update_mem_buffer_size();

  float max_secs = static_cast<float>(batch_num);

  for (int i = 0; i < timestep_num; ++i) {
    int cur_mem_require = get_timestep_memory_req(i);
    int cur_coeff_mem_require = get_timestep_coeff_memory_req(i);
    float max_secs_tmp = static_cast<float>(LOCAL_MEM_SIZE - cur_coeff_mem_require) /
                         (cur_mem_require - cur_coeff_mem_require);
    llvm::errs() << "cur_mem_require: " << cur_mem_require << " cur_coeff_mem_require: "
                 << cur_coeff_mem_require << " max_secs_tmp: " << max_secs_tmp << "\n";
    // [xun] to reduce loop count for speed
    if (max_secs_tmp < 0.0) {
      llvm::errs() << "split data in h dimension failed " << "\n";
      return BM_ERR_FAILURE;
    }

    if (max_secs > max_secs_tmp) {
      max_secs = max_secs_tmp;
    }
  }

  if (max_secs < 1) {
    nsecs_and_hsecs.first = batch_num;
    nsecs_and_hsecs.second = static_cast<int>(ceil(1.0f / (max_secs * SPLIT_RATIO)));
    if (nsecs_and_hsecs.second < 1) {
      llvm::errs() << "split data in h dimension failed, max_nsecs " << max_secs << "\n";
      return BM_ERR_FAILURE;
    }
  } else {
    int max_num = static_cast<int>(max_secs);
    nsecs_and_hsecs.first = (batch_num + max_num - 1) / max_num;
  }

  llvm::errs() << "[SPLIT] n: " << nsecs_and_hsecs.first << ", h: " << nsecs_and_hsecs.second << "\n";

  return BM_SUCCESS;
}

void net_timestep::update_tensor_timestep(int time_idx,
                                          const vector<TENSOR_STEP>& new_tensor_timestep) {
  tensor_load_store[time_idx].clear();
  tensor_load_store[time_idx] = new_tensor_timestep;
}

int net_timestep::tensor_range_end_timestep(const TENSOR_STEP& tensor_timestep) {
  if (tensor_timestep.second == TIMESTEP_LOAD) {
    return 0;
  }

  return timestep_num - 1;
}
}
