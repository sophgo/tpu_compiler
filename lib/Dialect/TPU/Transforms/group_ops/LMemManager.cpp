/*
 * Copyright (C) Bitmain Technologies Inc.
 * All Rights Reserved.
 */
#include "LMemManager.hpp"

#define DEBUG_TYPE "optimizer_cluster"

namespace mlir {

#define TBD_LADDR 0xFFFFFFFF

LmemManager::LmemManager(NetGraph* net_graph) : net_graph_(net_graph) {
  LMEM_BLOCK block = {.start = 0, .size = LOCAL_MEM_SIZE, .tid = -1, .step = 0, .busy = false};

  block_list.push_back(block);
  block_record_.push_back(block_list);
}

bmerr_t LmemManager::assign_local_memory(Group* cluster, net_timestep* time_step, bool one_shoot) {
  llvm::errs() << "====== assign local memory ======" << "\n";

  for (auto layer_id : cluster->layers()) {
    auto* ir = net_graph_->get_layer_by_id(layer_id);
    for (auto& tensor : ir->in_tensors)
      tensor->laddr = 0;
    for (auto& tensor : ir->out_tensors)
      tensor->laddr = 0;
  }

  if (!one_shoot) {
    for (int i = 0; i < time_step->get_timestep_num(); ++i) {
      const vector<TENSOR_STEP>& timestep_tensors = time_step->get_tensors(i);
      for (auto& step : timestep_tensors) {
        if (step.second == TIMESTEP_LOAD && is_tensor_resident_in_lmem(step.first)) {
          if (!alloc_block(block_list, step.first, i))
            return BM_ERR_FAILURE;
        }
      }
    }
  }

  for (int i = 0; i < time_step->get_timestep_num(); ++i) {
    int layer_id = time_step->get_layer(i);
    const vector<TENSOR_STEP>& timestep_tensors = time_step->get_tensors(i);

    recycle_lmem(block_list, time_step, i, one_shoot);

    for (auto& step : timestep_tensors) {
      if (!alloc_block(block_list, step.first, i))
        return BM_ERR_FAILURE;
    }

    if (layer_id != -1) {
      const ImLayer* layer = net_graph_->get_layer_by_id(layer_id);
      for (auto& tensor : layer->in_tensors) {
        if (!alloc_block(block_list, tensor->id(), i))
          return BM_ERR_FAILURE;
      }

      for (auto& tensor : layer->out_tensors) {
        if (!alloc_block(block_list, tensor->id(), i))
          return BM_ERR_FAILURE;
      }

      for (auto& tensor : layer->imm_tensors) {
        if (!alloc_block(block_list, tensor->id(), i))
          return BM_ERR_FAILURE;
      }
    }

    block_record_.push_back(block_list);
  }

  if (!figure_out_tensors_real_addr(time_step)) {
    return BM_ERR_FAILURE;
  }
  return BM_SUCCESS;
}

bool LmemManager::is_tensor_resident_in_lmem(int tid) {
  tensor_type_t type = net_graph_->get_tensor_type(tid);
  if (type == TENSOR_COEFF || type == TENSOR_COEFF_NEURON || type == TENSOR_BIAS ||
      type == TENSOR_DEPTHCONV_OPD1 || type == TENSOR_COEFF_WINOGRAD ||
      type == TENSOR_NEURON_AS_COEFF) {
    return true;
  }
  return false;
}

void LmemManager::recycle_lmem(list<LMEM_BLOCK>& block_list, net_timestep* time_step, int cur_step,
                               bool one_shoot) {
  for (auto iter = block_list.begin(); iter != block_list.end(); ++iter) {
    // skip free blocks.
    if (!iter->busy) {
      continue;
    }

    // resident in lmem if tensor is COEFF or BIAS.
    if (!one_shoot && is_tensor_resident_in_lmem(iter->tid)) {
      continue;
    }

    // resident in lmem if tensor is useb by next IR.
    bool resident = false;
    for (int i = cur_step; i < time_step->get_timestep_num(); ++i) {
      int layer_id = time_step->get_layer(i);
      const ImLayer* layer = net_graph_->get_layer_by_id(layer_id);
      const vector<TENSOR_STEP>& timestep_tensors = time_step->get_tensors(i);
      for (auto& step : timestep_tensors) {
        if (step.first == iter->tid) {
          resident = true;
          break;
        }
      }

      if (resident) {
        break;
      }

      if (layer_id == -1) {
        continue;
      }

      for (auto& tensor : layer->in_tensors) {
        if (tensor->id() == iter->tid) {
          resident = true;
          break;
        }
      }

      if (resident) {
        break;
      }

      for (auto& tensor : layer->out_tensors) {
        if (tensor->id() == iter->tid) {
          resident = true;
          break;
        }
      }

      if (resident) {
        break;
      }
    }

    if (!resident) {
      iter->tid = -1;
      iter->busy = false;
    }
  }

  merge_free_blocks(block_list);
}

bool LmemManager::alloc_block(list<LMEM_BLOCK>& block_list, int tid, int step_idx) {
  //llvm::errs() << "alloc_block: " << tid << "\n";

  Tensor* tensor = net_graph_->get_tensor_by_id(tid);
  if (tensor->laddr == TBD_LADDR) {
    return true;
  }

  //llvm::errs() << "prealloc cluster lmem: " << tid << "\n";

  auto last = --block_list.end();
  auto avail_it = last;
  u64 max_free_size = 0;

  // find max free block size.
  for (auto iter = block_list.begin(); iter != last; ++iter) {
    if (!iter->busy && iter->size > max_free_size) {
      avail_it = iter;
      max_free_size = iter->size;
    }
  }

  tensor->laddr = TBD_LADDR;
  auto iter = avail_it;

  int tensor_size = ALIGN(tensor->lmem_size(), EU_NUM * tensor->unit_size());

  if (iter->size > tensor_size) {
    LMEM_BLOCK block;
    block.start = iter->start + tensor_size;
    block.size = iter->size - tensor_size;
    block.tid = -1;
    block.busy = false;

    iter->tid = tensor->id();
    iter->size = tensor_size;
    iter->step = step_idx;
    iter->busy = true;

    block_list.insert(++iter, block);
  } else {
    iter->tid = tensor->id();
    iter->size = tensor_size;
    iter->step = step_idx;
    iter->busy = true;

    // correct the offset of subsequent tensors.
    u64 offset = iter->start + iter->size;
    if (offset >= LOCAL_MEM_SIZE) {
      llvm::errs() << "offset " << offset << " of tensor: " << iter->tid << " larger than " << LOCAL_MEM_SIZE << "\n";
      return false;
    }

    while (++iter != block_list.end()) {
      iter->start = offset;
      offset += iter->size;
    }
  }

  last = --block_list.end();
  if (last->start >= LOCAL_MEM_SIZE) {
    llvm::errs() << "lmem overflow: " << last->start - LOCAL_MEM_SIZE << "\n";
    return false;
  }

  merge_free_blocks(block_list);
  return true;
}

void LmemManager::merge_free_blocks(list<LMEM_BLOCK>& block_list) {
  auto iter = block_list.begin();
  while (iter != block_list.end()) {
    auto cur = iter++;
    while (iter != block_list.end() && !cur->busy && !iter->busy) {
      cur->size += iter->size;
      block_list.erase(iter++);
    }
  }
}

bool LmemManager::figure_out_tensors_real_addr(net_timestep* time_step) {
  u32 total_lmem_occupied = 0;
  mem_buffer_key_t key;

  for (int i = block_record_.size() - 1; i >= 0; --i) {
    list<LMEM_BLOCK>& block_list = block_record_[i];

    u32 offset = 0;
    for (auto iter = block_list.begin(); iter != block_list.end(); ++iter) {
      if (iter->busy) {
        Tensor* tensor = net_graph_->get_tensor_by_id(iter->tid);
        // if tensor was allocated, relocate the start offset of block.
        iter->start = (tensor->laddr == TBD_LADDR) ? iter->start : tensor->laddr;

        // if start offset of block is already allocated,
        // relocate it to current offset point.
        if (iter->start <= offset) {
          iter->start = offset;
        }

        if (tensor->laddr == TBD_LADDR) {
          tensor->laddr = iter->start;

          key.start_timestep = iter->step;
          key.id_num = iter->tid;
          key.is_layer_imm = (tensor->type() == TENSOR_IMM);

          time_step->set_local_mem_offset(&key, tensor->laddr);

          u32 end = iter->start + iter->size;

          total_lmem_occupied = total_lmem_occupied < end ? end : total_lmem_occupied;

          llvm::errs() << "[CONFIRM]";
          llvm::errs() << "[stage: " << iter->step << "]" << "tensor_id:" << iter->tid << ", " << iter->start << " ~ "
                  << end << ", size:" << iter->size << "is_imm:" << key.is_layer_imm << "\n";
        }

        offset = iter->start + iter->size;

      } else {
        iter->start = offset;
        offset += iter->size;
      }
    }
  }

  if (total_lmem_occupied > LOCAL_MEM_SIZE) {
    llvm::errs() << "total needed lmem size: " << total_lmem_occupied << "\n";
    return false;
  }
  return true;
}

void LmemManager::show_blocks(list<LMEM_BLOCK>& block_list) {
  for (auto iter = block_list.begin(); iter != block_list.end(); ++iter) {
    llvm::errs() << "[BLOCK] start:" << iter->start << " ~ " << iter->start + iter->size
            << ", size:" << iter->size << ", tid:" << iter->tid << ", busy:" << iter->busy << "\n";
  }
}
}
