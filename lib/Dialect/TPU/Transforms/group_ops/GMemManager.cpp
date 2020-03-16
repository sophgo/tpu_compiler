#include <algorithm>
#include "GMemManager.hpp"

#define DEBUG_TYPE "optimizer_cluster"

namespace mlir {

#define TBD_GADDR 0xFFFFFFFFFFFFFFFF

GmemManager::GmemManager(NetGraph* net_graph)
    : net_graph_(net_graph), tg_join_out_tensor_(-1), tg_join_input_tensors_() {
  GMEM_BLOCK block = {.start = 0, .size = GLOBAL_MEM_SIZE, .tid = -1, .busy = false};

  list<GMEM_BLOCK> block_list;
  block_list.push_back(block);
  block_record_.push_back(block_list);
}

u64 GmemManager::assign_global_memory(const vector<Group*>& clusters, bool gmem_recycle) {
  llvm::errs() << "====== assign global memory ======" << "\n";

  find_tg_join_tensors();

  for (int i = 0; i < static_cast<int>(clusters.size()); i++) {
    if (i == 0) {
      prealloc_io_tensors(block_record_[0]);
    } else {
      block_record_.push_back(block_record_[i - 1]);
    }

    if (gmem_recycle) {
      recycle_cluster_gmem(block_record_[i], clusters[i], i == 0 ? nullptr : clusters[i - 1]);
    }

    prealloc_cluster_gmem(block_record_[i], clusters[i]);

    show_blocks(block_record_[i]);

    llvm::errs() << "cluster " << i << " end" << "\n";
  }

  return figure_out_tensor_offset();
}

int GmemManager::set_in_place_layer_tensor_gaddr_unit(const int tid, const uint64_t value) {
  int from_layer = net_graph_->get_tensor_from_layer(tid);
  Tensor *tensor = net_graph_->get_tensor_by_id(tid);
  const ImLayer* layer = net_graph_->get_layer_by_id(from_layer);
  if (layer->is_tg_layer && layer->is_inplace_layer) {
    int idx = 0;
    for (int idx = 0; idx < layer->out_tensors.size(); idx++) {
      if (layer->out_tensors[idx]->name() == tensor->name()) {
        break;
      }
    }
    if (idx < layer->in_tensors.size()) {
      layer->in_tensors[idx]->gaddr = value;
      return layer->in_tensors[idx]->id();
    }
    llvm::errs() << "Error correcting in place layer " << layer->name() << " address.\n";
  }
  return -1;
}

bool GmemManager::set_in_place_layer_tensor_gaddr(int tid, uint64_t value) {
  while(tid != -1) {
    tid = set_in_place_layer_tensor_gaddr_unit(tid, value);
  }
  return true;
}

void GmemManager::find_tg_join_tensors() {
  int join_layer = -1;
  IR_TYPE type;

  for (auto iter = ImLayer::layers.rbegin(); iter != ImLayer::layers.rend(); ++iter) {
    type = (*iter)->type();
    if (type == IR_JOIN) {
      join_layer = (*iter)->id();
      break;
    }
  }

  if (join_layer == -1) {
    return;
  }

  tg_join_out_tensor_ = net_graph_->get_out_tensors_of_layer(join_layer)[0];
  const vector<int>& bottoms = net_graph_->get_in_tensors_of_layer(join_layer);

  u64 offset = 0;
  for (int i = 0; i < static_cast<int>(bottoms.size()); i++) {
    int tid = bottoms[i];
    int from = net_graph_->get_tensor_from_layer(tid);
    const ImLayer* layer = net_graph_->get_layer_by_id(from);
    while (layer->is_tg_layer && layer->is_inplace_layer) {
      tid = net_graph_->get_in_tensors_of_layer(from)[0];
      from = net_graph_->get_tensor_from_layer(tid);
      layer = net_graph_->get_layer_by_id(from);
    }

    tg_join_input_tensors_[tid] = offset;
    offset += net_graph_->get_tensor_gmem_size(tid);
  }
}

bool GmemManager::is_tensor_of_join_layer(int tid) {
  if (tid == tg_join_out_tensor_) {
    return true;
  }

  if (tg_join_input_tensors_.find(tid) != tg_join_input_tensors_.end()) {
    return true;
  }

  return false;
}

void GmemManager::prealloc_io_tensors(list<GMEM_BLOCK>& block_list) {
  for (auto& layer : ImLayer::layers) {
    if (isa<tpu::InputOp>(layer->op()) || isa<tpu::TG_INT8_InputOp>(layer->op())) {
      for (auto& tensor : layer->out_tensors) {
        int tid = tensor->id();
        if (tensor->type() == TENSOR_NEURON || tensor->type() == TENSOR_NEURON_WINOGRAD) {
          alloc_block(block_list, tid);
          net_in_tensors_.insert(tid);
        }
      }
    }

    for (auto& tensor : layer->out_tensors) {
      int tid = tensor->id();
      const vector<int>& to_layers = net_graph_->get_tensor_to_layer(tid);
      if (to_layers.empty()) {
        alloc_block(block_list, tid);
        net_out_tensors_.insert(tid);
      }
    }
  }
}

static void find_final_to_layer(NetGraph* net_graph_, int tid, vector<int>& to_layers) {
  const vector<int>& layers = vector<int>(net_graph_->get_tensor_to_layer(tid));
  if (layers.empty()) {
    return;
  }

  // depth first search for final to layers of tensor.
  for (auto id : layers) {
    to_layers.push_back(id);
    const ImLayer* layer = net_graph_->get_layer_by_id(id);
    if (layer->is_tg_layer && layer->is_inplace_layer) {
      for (auto tid : net_graph_->get_out_tensors_of_layer(id)) {
        find_final_to_layer(net_graph_, tid, to_layers);
      }
    }
  }
}

void GmemManager::recycle_cluster_gmem(list<GMEM_BLOCK>& block_list, Group* cluster,
                                       Group* prev_cluster) {
  int first_layer = cluster->layers()[0];
  const ImLayer* layer = net_graph_->get_layer_by_id(first_layer);

  vector<int> to_layers;

  for (auto iter = block_list.begin(); iter != block_list.end(); ++iter) {
    if (!iter->busy) {
      continue;
    }

    int tid = iter->tid;
    if (net_in_tensors_.find(tid) != net_in_tensors_.end()) {
      continue;
    }

    if (net_out_tensors_.find(tid) != net_out_tensors_.end()) {
      continue;
    }

    to_layers.clear();
    find_final_to_layer(net_graph_, tid, to_layers);
    if (to_layers.empty()) {
      continue;
    }

    bool keep = false;

    for (auto id : to_layers) {
      if (id >= first_layer) {
        keep = true;
      }
    }

    if (!keep && layer->type() == IR_CONCAT && prev_cluster) {
      LLVM_DEBUG(cout << "layer is bmnet_concat, tid:" << tid << "\n");
      if (prev_cluster->is_group_in_neuron_tensor(tid)) {
        LLVM_DEBUG(cout << "keep tensor id, " << tid << "\n");
        keep = true;
      }
    }

    if (!keep) {
      llvm::errs() << "recycle tensor:" << iter->tid << ", threshold layer:" << first_layer << "\n";
      iter->tid = -1;
      iter->busy = false;
    }
  }

  merge_free_blocks(block_list);
}

void GmemManager::prealloc_cluster_gmem(list<GMEM_BLOCK>& block_list, Group* cluster) {
  net_timestep* timestep = cluster->time_step;

  for (int i = 0; i < timestep->get_timestep_num(); ++i) {
    const vector<TENSOR_STEP>& tensor_steps = timestep->get_tensors(i);

    for (u32 j = 0; j < tensor_steps.size(); ++j) {
      // Allocate ddr space for lmem-to-ddr & tsm-to-ddr operation
      if ((tensor_steps[j].second == TIMESTEP_STORE) ||
          (tensor_steps[j].second == TIMESTEP_TSM_TO_DDR)) {
        int tid = tensor_steps[j].first;

        int from_layer = net_graph_->get_tensor_from_layer(tid);
        const ImLayer* layer = net_graph_->get_layer_by_id(from_layer);

        const vector<int>& to_layers = net_graph_->get_tensor_to_layer(tid);
        /*
         * if current layer is in a group and connect to a concat layer,
         * then the output tensor should share the gmem with
         * the output tensor of concat layer. So no need to allocate gmem
         * in such case.
         */
        // [xun]: for multiple concat to-layer case
        bool is_concat_optimized = false;
        bool gmem_allocated = false;
        std::vector<int> concat_out_tensors;
        const vector<int>& layers = cluster->layers();

        for (int k = 0; k < static_cast<int>(to_layers.size()); k++) {
          if (net_graph_->is_concat_optimized_case(to_layers[k], tid)) {
            concat_out_tensors.push_back(net_graph_->get_out_tensors_of_layer(to_layers[k])[0]);
          } else {
            if (net_graph_->get_layer_by_id(to_layers[k])->is_inplace_layer) {
              gmem_allocated = true;  // [xun] only one time for global memory
            }
          }
        }

        // Currently BM168X uses npu to copy data. If axis = 1 will make input stride
        // not equal to output stride.
        if (cluster->size() == 1 && concat_out_tensors.size() > 0) {
          LLVM_DEBUG(llvm::errs() << "Tg tensor " << net_graph_->get_tensor_by_id(tid)->name()
                    << " connects to a concat layer.\n");
        }

        for (auto concat_out_tensor : concat_out_tensors) {
          alloc_block(block_list, concat_out_tensor);
        }
        if ((concat_out_tensors.size() == 0 || gmem_allocated) &&
            !(layer->is_tg_layer && layer->is_inplace_layer)) {
          // Only allocate block if this layer is not (tg && in place)
          alloc_block(block_list, tid);
        }

        // Check if the ignore_bottom param is properly set.
        Tensor *tensor = net_graph_->get_tensor_by_id(tid);
        if (tensor->gaddr != 0xFFFFFFFF && tensor->gaddr != TBD_GADDR) {
          for (auto concat_out_tensor : concat_out_tensors) {
            ImConcat *c_layer = (ImConcat*)net_graph_->get_layer_by_id(to_layers[concat_out_tensor]);
            size_t tensor_idx = 0;
            for (;c_layer->in_tensors.size(); tensor_idx++) {
              if (c_layer->in_tensors[tensor_idx]->name() == tensor->name()) {
                break;
              }
            }
            for (auto ig_idx : c_layer->ignored_bottoms) {
              if (ig_idx == tensor_idx) {
                llvm::errs() << "This tg tensor cannot be ignored cause it links to multiple layers.\n";
              }
            }
          }
        }
      }
    }
  }
}

void GmemManager::alloc_block(list<GMEM_BLOCK>& block_list, int tid) {
  Tensor* tensor = net_graph_->get_tensor_by_id(tid);
  if (tensor->gaddr == TBD_GADDR) {
    return;
  }

  // check if has reserved block of tg_join layer.
  if (is_tensor_of_join_layer(tid)) {
    tensor->gaddr = TBD_GADDR;
    for (auto& iter : block_list) {
      if (iter.tid == tg_join_out_tensor_) {
        return;
      }
    }
    // if not, allocate it.
    tensor = net_graph_->get_tensor_by_id(tg_join_out_tensor_);
  }

  llvm::errs() << "prealloc cluster gmem: " << tid << "\n";

  auto last = --block_list.end();
  auto avail_it = last;
  u64 max_free_size = 0;

  for (auto iter = block_list.begin(); iter != last; ++iter) {
    if (!iter->busy && iter->size > max_free_size) {
      avail_it = iter;
      max_free_size = iter->size;
    }
  }

  tensor->gaddr = TBD_GADDR;
  auto iter = avail_it;

  if (iter->size > tensor->gmem_size()) {
    GMEM_BLOCK block;
    block.start = iter->start + tensor->gmem_size();
    block.size = iter->size - tensor->gmem_size();
    block.tid = -1;
    block.busy = false;

    iter->tid = tensor->id();
    iter->size = tensor->gmem_size();
    iter->busy = true;

    block_list.insert(++iter, block);
  } else {
    iter->tid = tensor->id();
    iter->size = tensor->gmem_size();
    iter->busy = true;

    // correct the offset of subsequent tensors.
    u64 offset = iter->start + iter->size;

    while (++iter != block_list.end()) {
      iter->start = offset;
      offset += iter->size;
    }
  }

  merge_free_blocks(block_list);
}

void GmemManager::merge_free_blocks(list<GMEM_BLOCK>& block_list) {
  auto iter = block_list.begin();
  while (iter != block_list.end()) {
    auto cur = iter++;
    while (iter != block_list.end() && !cur->busy && !iter->busy) {
      cur->size += iter->size;
      block_list.erase(iter++);
    }
  }
}

u64 GmemManager::figure_out_tensor_offset() {
  u64 total_neuron_size = 0;

  for (int i = block_record_.size() - 1; i >= 0; --i) {
    list<GMEM_BLOCK>& block_list = block_record_[i];

    u64 offset = 0;
    for (auto iter = block_list.begin(); iter != block_list.end(); ++iter) {
      if (iter->busy) {
        Tensor* tensor = net_graph_->get_tensor_by_id(iter->tid);
        iter->start = (tensor->gaddr == TBD_GADDR) ? iter->start : tensor->gaddr;

        if (iter->start <= offset) {
          iter->start = offset;
        }

        if (tensor->gaddr == TBD_GADDR) {
          tensor->gaddr = iter->start;
          u64 end = iter->start + iter->size;
          total_neuron_size = total_neuron_size < end ? end : total_neuron_size;

          llvm::errs() << "[CONFIRM] tid:" << iter->tid << ", " << iter->start << " ~ " << end
                  << ", size:" << iter->size << "\n";

          if (iter->tid == tg_join_out_tensor_) {
            for (auto m : tg_join_input_tensors_) {
              int tid = m.first;
              tensor = net_graph_->get_tensor_by_id(tid);
              tensor->gaddr = iter->start + m.second;

              llvm::errs() << "[CONFIRM] tid:" << tid << ", " << tensor->gaddr << " ~ "
                      << tensor->gaddr + tensor->gmem_size() << ", size:" << tensor->gmem_size()
                      << "\n";
            }
          }
        }

        offset = iter->start + iter->size;

      } else {
        iter->start = offset;
        offset += iter->size;
      }
    }
  }

  for (auto& layer : ImLayer::layers) {
    if (layer->type() == IR_CONCAT) {
      uint64_t out_gaddr = layer->out_tensors[0].get()->gaddr;
      if (out_gaddr == 0xFFFFFFFF) {
        // Should do something else or just assert?
        llvm::errs() << "tg_concat " << layer->out_tensors[0]->name() << " not allocated.\n";
        continue;
      }
      uint64_t g_offset = 0;
      for (auto in : layer->in_tensors) {
        /**
         * Here we need to fix in place layer address to make concat in place.
         * in->gaddr = 0xFFFFFFFF -> Default value
         */
        if (in->gaddr == 0xFFFFFFFF && net_graph_->is_concat_optimized_case(layer->id(), in->id())) {
          LLVM_DEBUG(llvm::errs() << "In place tensor " << in->name() << " found. Fixing global address.\n");
          in->gaddr = out_gaddr + g_offset;
          /**
           * Correct in place layer output addr by assigning output addr to intput addr.
           * Make sure the in_place layer and concat layer are both in place.
           */
          set_in_place_layer_tensor_gaddr(in->id(), in->gaddr);
        }
        g_offset += in->gmem_size();
      }
    }
  }
  for (auto& layer : ImLayer::layers) {
    // Correct in place layer output addr by assigning input addr to output addr.
    if (layer->is_tg_layer && layer->is_inplace_layer) {
      Tensor* in_tensor = layer->in_tensors[0].get();
      uint64_t offset = 0;
      for (auto& out : layer->out_tensors) {
        if (out->gaddr != 0xFFFFFFFF)
          continue;
        out->gaddr = in_tensor->gaddr + offset;
        offset += out->gmem_size();
      }
    }
  }

  llvm::errs() << "total neuron size " << total_neuron_size << "\n";
  return total_neuron_size;
}

void GmemManager::show_blocks(list<GMEM_BLOCK>& block_list) {
  for (auto iter = block_list.begin(); iter != block_list.end(); ++iter) {
    llvm::errs() << "[BLOCK] start:" << iter->start << " ~ " << iter->start + iter->size
            << ", size:" << iter->size << ", tid:" << iter->tid << ", busy:" << iter->busy << "\n";
  }
}
}
