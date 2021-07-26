#include "LMemAllocMethod.h"

namespace mlir {

LMemAllocMethod::LMemAllocMethod() {}
LMemAllocMethod::~LMemAllocMethod() {}

bool LMemAllocMethod::is_tensor_resident_in_lmem(tensor_type type) {
  //tensor_type_t type = net_graph_->get_tensor_type(tid);
  if (type == TENSOR_COEFF_CONV || type == TENSOR_COEFF_NEURON ||
      type == TENSOR_COEFF || type == TENSOR_DEPTHCONV_OPD1 ||
      type == TENSOR_NEURON_AS_COEFF) {
    return true;
  }
  return false;
}

LMemAllocFitFirst::LMemAllocFitFirst() {}
LMemAllocFitFirst::~LMemAllocFitFirst() {}

bmerr_t LMemAllocFitFirst::assign_local_memory(Group *group,
                                               NetGraph *net_graph,
                                               net_timestep *time_step,
                                               bool one_shoot) {
  net_graph_ = net_graph;
  LLVM_DEBUG(llvm::errs() << LOG_TAB_L4
                          << "[Assign Lmem] Begin" << "\n";);

  std::list<LMEM_BLOCK> block_list;
  init(block_list);
  for (auto layer_id : group->layers()) {
    auto* ir = net_graph_->get_layer_by_id(layer_id);
    for (auto& tensor : ir->in_tensors)
      tensor->laddr = 0;
    for (auto& tensor : ir->out_tensors)
      tensor->laddr = 0;
    for (auto& tensor : ir->imm_tensors)
      tensor->laddr = 0;
  }

  if (!one_shoot) {
    for (int i = 0; i < time_step->get_timestep_num(); ++i) {
      const std::vector<TENSOR_STEP>& timestep_tensors = time_step->get_tensors(i);
      for (auto& step : timestep_tensors) {
        if (step.second == TIMESTEP_LOAD &&
            is_tensor_resident_in_lmem(
                net_graph->get_tensor_type(step.first))) {
          if (!alloc_block(block_list, step.first, i))
            return BM_ERR_FAILURE;
        }
      }
    }
  }

  for (int i = 0; i < time_step->get_timestep_num(); ++i) {
    int layer_id = time_step->get_layer(i);
    const std::vector<TENSOR_STEP>& timestep_tensors = time_step->get_tensors(i);

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

void LMemAllocFitFirst::init(std::list<LMEM_BLOCK> &block_list) {
  LMEM_BLOCK block;
  block.start = 0;
  block.size = LOCAL_MEM_SIZE;
  block.tid = -1;
  block.step = 0;
  block.busy = false;

  block_list.push_back(block);
  block_record_.clear();
  block_record_.push_back(block_list);
}

void LMemAllocFitFirst::recycle_lmem(std::list<LMEM_BLOCK> &block_list,
                                     net_timestep *time_step, int cur_step,
                                     bool one_shoot) {
  for (auto iter = block_list.begin(); iter != block_list.end(); ++iter) {
    // skip free blocks.
    if (!iter->busy) {
      continue;
    }

    // resident in lmem if tensor is COEFF or BIAS.
    if (!one_shoot &&
        is_tensor_resident_in_lmem(net_graph_->get_tensor_type(iter->tid))) {
      continue;
    }

    // resident in lmem if tensor is useb by next IR.
    bool resident = false;
    for (int i = cur_step; i < time_step->get_timestep_num(); ++i) {
      int layer_id = time_step->get_layer(i);
      const ImLayer *layer = net_graph_->get_layer_by_id(layer_id);
      const std::vector<TENSOR_STEP> &timestep_tensors =
          time_step->get_tensors(i);
      for (auto &step : timestep_tensors) {
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

      for (auto &tensor : layer->in_tensors) {
        if (tensor->id() == iter->tid) {
          resident = true;
          break;
        }
      }

      if (resident) {
        break;
      }

      for (auto &tensor : layer->out_tensors) {
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

bool LMemAllocFitFirst::alloc_block(std::list<LMEM_BLOCK> &block_list, int tid,
                                    int step_idx) {

  Tensor *tensor = net_graph_->get_tensor_by_id(tid);
  if (tensor->laddr == TBD_LADDR) {
    return true;
  }

  auto last = --block_list.end();
  auto avail_it = last;
  int max_free_size = 0;

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
    int offset = iter->start + iter->size;
    if ((uint32_t)offset >= LOCAL_MEM_SIZE) {
      LLVM_DEBUG(llvm::errs() << LOG_TAB_L4 << "[Assign Lmem][Warning] offset "
                              << offset << " of tensor: " << iter->tid
                              << " larger than " << LOCAL_MEM_SIZE << "\n";);
      return false;
    }

    while (++iter != block_list.end()) {
      iter->start = offset;
      offset += iter->size;
    }
  }

  last = --block_list.end();
  if ((uint32_t)last->start >= LOCAL_MEM_SIZE) {
    LLVM_DEBUG(llvm::errs()
                   << LOG_TAB_L4 << "[Assign Lmem][Warning] lmem overflow: "
                   << last->start - LOCAL_MEM_SIZE << "\n";);
    return false;
  }

  merge_free_blocks(block_list);
  return true;
}

void LMemAllocFitFirst::merge_free_blocks(std::list<LMEM_BLOCK> &block_list) {
  auto iter = block_list.begin();
  while (iter != block_list.end()) {
    auto cur = iter++;
    while (iter != block_list.end() && !cur->busy && !iter->busy) {
      cur->size += iter->size;
      block_list.erase(iter++);
    }
  }
}

bool LMemAllocFitFirst::figure_out_tensors_real_addr(net_timestep *time_step) {
  uint32_t total_lmem_occupied = 0;
  mem_buffer_key_t key;

  for (int i = block_record_.size() - 1; i >= 0; --i) {
    std::list<LMEM_BLOCK>& block_list = block_record_[i];

    int32_t offset = 0;
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

          uint32_t end = iter->start + iter->size;

          total_lmem_occupied = total_lmem_occupied < end ? end : total_lmem_occupied;

          LLVM_DEBUG(
            llvm::errs()
                         << LOG_TAB_L4 << "[Assign Lmem][stage: " << iter->step
                         << "]"
                         << "tensor_id:" << iter->tid << ", " << iter->start
                         << " ~ " << end << ", size:" << iter->size
                         << "is_imm:" << key.is_layer_imm << "\n";
                         );
        }

        offset = iter->start + iter->size;

      } else {
        iter->start = offset;
        offset += iter->size;
      }
    }
  }

  if (total_lmem_occupied > (uint32_t)LOCAL_MEM_SIZE) {
    LLVM_DEBUG(llvm::errs() << LOG_TAB_L4
                            << "[Assign Lmem] Failed with total needed lmem size: "
                            << total_lmem_occupied << " < Local Memory("
                            << LOCAL_MEM_SIZE << ")\n";);
    return false;
  }
  return true;
}

void LMemAllocFitFirst::show_blocks(std::list<LMEM_BLOCK> &block_list) {
  for (auto iter = block_list.begin(); iter != block_list.end(); ++iter) {
    LLVM_DEBUG(
      llvm::errs()
      << LOG_TAB_L4
      << "[Assign Lmem][BLOCK] start:" << iter->start << " ~ " << iter->start + iter->size
      << ", size:" << iter->size << ", tid:" << iter->tid
      << ", busy:" << iter->busy << "\n";);
  }
}



LMemAllocSizeOrder::LMemAllocSizeOrder() {}
LMemAllocSizeOrder::~LMemAllocSizeOrder() {}

#if 0
bmerr_t LMemAllocSizeOrder::assign_local_memory(Group *group,
                                               NetGraph *net_graph,
                                               net_timestep *time_step,
                                               bool one_shoot) {
  // calc tensor live range [first, last)
  std::map<int, std::shared_ptr<TensorRect>> tensor_map;
  std::list<std::shared_ptr<TensorRect>> resident_tensor_list;

  auto insert_tensor = [&tensor_map, &net_graph](int tensor_id, int time) {
    if (tensor_map.end() != tensor_map.find(tensor_id)) {
      tensor_map[tensor_id]->last = std::max(time + 1, tensor_map[tensor_id]->last);
    } else {
      Tensor *tensor = net_graph->get_tensor_by_id(tensor_id);
      int tensor_size = ALIGN(tensor->lmem_size(), EU_NUM * tensor->unit_size());
      std::shared_ptr<TensorRect> tensor_rect = std::make_shared<TensorRect>(
          tensor_id, time, time + 1, time, tensor_size);
      tensor_map[tensor_id] = tensor_rect;
    }
  };

  int step_end = time_step->get_timestep_num();
  for (int i = 0; i < time_step->get_timestep_num(); ++i) {
    int layer_id = time_step->get_layer(i);
    const std::vector<TENSOR_STEP>& timestep_tensors = time_step->get_tensors(i);
    for (auto& step : timestep_tensors) {
      // if not one_shoot
      if (!one_shoot &&
          is_tensor_resident_in_lmem(net_graph->get_tensor_type(step.first))) {
        int step_begin = 0;
        if (step.second != TIMESTEP_LOAD) {
          step_begin = i;
        }
        Tensor *tensor = net_graph->get_tensor_by_id(step.first);
        int tensor_size =
            ALIGN(tensor->lmem_size(), EU_NUM * tensor->unit_size());
        std::shared_ptr<TensorRect> tensor_rect = std::make_shared<TensorRect>(
            step.first, step_begin, step_end, i, tensor_size);
        resident_tensor_list.emplace_back(tensor_rect);
      } else {
        insert_tensor(step.first, i);
      }
    }
    if (layer_id != -1) {
      const ImLayer *layer = net_graph->get_layer_by_id(layer_id);
      for (auto &tensor : layer->in_tensors) {
        insert_tensor(tensor->id(), i);
      }
      for (auto &tensor : layer->out_tensors) {
        insert_tensor(tensor->id(), i);
      }
      for (auto &tensor : layer->imm_tensors) {
        insert_tensor(tensor->id(), i);
      }
    }
  }

  std::list<std::shared_ptr<TensorRect>> tensor_list;
  std::list<std::shared_ptr<TensorRect>> allocated_tensor_list;

  // sort resident tensor by start time
  resident_tensor_list.sort(
      [](std::shared_ptr<TensorRect> &a, std::shared_ptr<TensorRect> &b) {
        return a->first < b->first;
      });

  int start_addr = 0;

  // alloc addr for resident tensor
  for (auto &tensor_rect : resident_tensor_list) {
    // erase dup tensor from tensor_map
    if (tensor_map.find(tensor_rect->tid) != tensor_map.end()) {
      tensor_map.erase(tensor_rect->tid);
    }
    tensor_rect->start = start_addr;
    tensor_rect->end = tensor_rect->start + tensor_rect->size;
    start_addr = tensor_rect->end;
    if (LOCAL_MEM_SIZE < static_cast<uint32_t>(start_addr)) {
      LLVM_DEBUG(llvm::errs()
                     << LOG_TAB_L4 << "[Assign Lmem][Warning] lmem overflow: "
                     << start_addr - LOCAL_MEM_SIZE
                     << "\n";);
      return BM_ERR_FAILURE;
    }
    allocated_tensor_list.emplace_back(tensor_rect);
  }

  // sort tensor by size
  for (auto &tensor : tensor_map) {
    auto iter = std::find_if(tensor_list.begin(), tensor_list.end(),
                             [&tensor](std::shared_ptr<TensorRect> &a) {
                               if (a->size < tensor.second->size) {
                                 return true;
                               } else if (a->size == tensor.second->size) {
                                 return (a->last - a->first) <= (tensor.second->last - tensor.second->first);
                               } else {
                                 return false;
                               }
                             });
    tensor_list.emplace(iter, tensor.second);
  }

  // alloc addr
  int total_consumption = 0;
  for (auto &tensor_rect : tensor_list) {
    int prev_offset = 0;
    int best_offset = -1;
    int smallest_gap = std::numeric_limits<int>::max();
    for (auto &allocated_tensor_rect : allocated_tensor_list) {
      int max_first = std::max(tensor_rect->first, allocated_tensor_rect->first);
      int min_last = std::min(tensor_rect->last, allocated_tensor_rect->last);
      if (max_first < min_last) {
        int gap = allocated_tensor_rect->start - prev_offset;
        if (gap >= tensor_rect->size && gap < smallest_gap) {
          smallest_gap = gap;
          best_offset = prev_offset;
        }
        prev_offset = std::max(prev_offset, allocated_tensor_rect->end);
      }
    }
    if (best_offset == -1) {
      best_offset = prev_offset;
    }
    tensor_rect->start = best_offset;
    tensor_rect->end = tensor_rect->start + tensor_rect->size;
    if (LOCAL_MEM_SIZE < static_cast<uint32_t>(tensor_rect->end)) {
      LLVM_DEBUG(llvm::errs()
                     << LOG_TAB_L4 << "[Assign Lmem][Warning] lmem overflow: "
                     << tensor_rect->end - LOCAL_MEM_SIZE
                     << "\n";);
      return BM_ERR_FAILURE;
    }
    total_consumption = std::max(total_consumption, tensor_rect->end);
    auto iter = std::find_if(allocated_tensor_list.begin(), allocated_tensor_list.end(),
                     [&tensor_rect](std::shared_ptr<TensorRect> &a) {
                       return a->start >= tensor_rect->start;
                     });
    allocated_tensor_list.emplace(iter, tensor_rect);
  }

  // alloc real addr
  figure_out_tensor_real_addr(allocated_tensor_list, net_graph, time_step);
  llvm::errs() << "[Assgin Lmem SizeOrder1] finish\n";
  return BM_SUCCESS;
}
#else
bmerr_t LMemAllocSizeOrder::assign_local_memory(Group *group,
                                               NetGraph *net_graph,
                                               net_timestep *time_step,
                                               bool one_shoot) {
  // calc tensor live range [first, last)
  std::map<int, std::shared_ptr<TensorRect>> tensor_map;

  auto insert_tensor = [&tensor_map, &net_graph](int tensor_id, int start, int end, int step_idx) {
    if (tensor_map.end() != tensor_map.find(tensor_id)) {
      tensor_map[tensor_id]->last = std::max(end, tensor_map[tensor_id]->last);
    } else {
      Tensor *tensor = net_graph->get_tensor_by_id(tensor_id);
      int tensor_size = ALIGN(tensor->lmem_size(), EU_NUM * tensor->unit_size());
      std::shared_ptr<TensorRect> tensor_rect = std::make_shared<TensorRect>(
          tensor_id, start, end, step_idx, tensor_size);
      tensor_map[tensor_id] = tensor_rect;
    }
  };

  int step_end = time_step->get_timestep_num();
  for (int i = 0; i < time_step->get_timestep_num(); ++i) {
    int layer_id = time_step->get_layer(i);
    const std::vector<TENSOR_STEP>& timestep_tensors = time_step->get_tensors(i);
    for (auto& step : timestep_tensors) {
      // if not one_shoot
      if (!one_shoot &&
          is_tensor_resident_in_lmem(net_graph->get_tensor_type(step.first))) {
        if (step.second == TIMESTEP_LOAD) {
          insert_tensor(step.first, 0, step_end, i);
        } else {
          insert_tensor(step.first, i, step_end, i);
        }
      } else {
        insert_tensor(step.first, i, i + 1, i);
      }
    }
    if (layer_id != -1) {
      const ImLayer* layer = net_graph->get_layer_by_id(layer_id);
      for (auto &tensor : layer->in_tensors) {
        insert_tensor(tensor->id(), i, i + 1, i);
      }
      for (auto &tensor : layer->out_tensors) {
        insert_tensor(tensor->id(), i, i + 1, i);
      }
      for (auto &tensor : layer->imm_tensors) {
        insert_tensor(tensor->id(), i, i + 1, i);
      }
    }
  }

  std::list<std::shared_ptr<TensorRect>> tensor_list;
  std::list<std::shared_ptr<TensorRect>> allocated_tensor_list;

  // sort tensor by size
  for (auto &tensor : tensor_map) {
    auto iter = std::find_if(tensor_list.begin(), tensor_list.end(),
                             [&tensor](std::shared_ptr<TensorRect> &a) {
                               if (a->size < tensor.second->size) {
                                 return true;
                               } else if (a->size == tensor.second->size) {
                                 return (a->last - a->first) <= (tensor.second->last - tensor.second->first);
                               } else {
                                 return false;
                               }
                             });
    tensor_list.emplace(iter, tensor.second);
  }

  // alloc addr
  int total_consumption = 0;
  for (auto &tensor_rect : tensor_list) {
    int prev_offset = 0;
    int best_offset = -1;
    int smallest_gap = std::numeric_limits<int>::max();
    for (auto &allocated_tensor_rect : allocated_tensor_list) {
      int max_first = std::max(tensor_rect->first, allocated_tensor_rect->first);
      int min_last = std::min(tensor_rect->last, allocated_tensor_rect->last);
      if (max_first < min_last) {
        int gap = allocated_tensor_rect->start - prev_offset;
        if (gap >= tensor_rect->size && gap < smallest_gap) {
          smallest_gap = gap;
          best_offset = prev_offset;
        }
        prev_offset = std::max(prev_offset, allocated_tensor_rect->end);
      }
    }
    if (best_offset == -1) {
      best_offset = prev_offset;
    }
    tensor_rect->start = best_offset;
    tensor_rect->end = tensor_rect->start + tensor_rect->size;
    LLVM_DEBUG(llvm::errs() << LOG_TAB_L4 << "[Assign Lmem][stage: " << tensor_rect->first
                 << " ~ " << tensor_rect->last << "]"
                 << " tensor_id:" << tensor_rect->tid << ", addr:"
                 << tensor_rect->start << " ~ " << tensor_rect->end
                 << ", size:" << tensor_rect->size << "\n");
    if (LOCAL_MEM_SIZE < static_cast<uint32_t>(tensor_rect->end)) {
      LLVM_DEBUG(llvm::errs()
                     << LOG_TAB_L4 << "[Assign Lmem][Warning] lmem overflow: "
                     << tensor_rect->end - LOCAL_MEM_SIZE
                     << "\n";);
      return BM_ERR_FAILURE;
    }
    total_consumption = std::max(total_consumption, tensor_rect->end);
    auto iter = std::find_if(allocated_tensor_list.begin(), allocated_tensor_list.end(),
                     [&tensor_rect](std::shared_ptr<TensorRect> &a) {
                       return a->start >= tensor_rect->start;
                     });
    allocated_tensor_list.emplace(iter, tensor_rect);
  }

  // alloc real addr
  figure_out_tensor_real_addr(allocated_tensor_list, net_graph, time_step);
  LLVM_DEBUG(llvm::errs() << "[Assgin Lmem SizeOrder2] finish\n");
  return BM_SUCCESS;
}
#endif

int LMemAllocSizeOrder::figure_out_tensor_real_addr(
    std::list<std::shared_ptr<TensorRect>> &tensor_list, NetGraph *net_graph,
    net_timestep *time_step) {
  for (auto &tensor_rect : tensor_list) {
    Tensor *tensor = net_graph->get_tensor_by_id(tensor_rect->tid);
    tensor->laddr = tensor_rect->start;
    mem_buffer_key_t key;
    key.start_timestep = tensor_rect->step_idx;
    key.id_num = tensor_rect->tid;
    key.is_layer_imm = (tensor->type() == TENSOR_IMM);
    time_step->set_local_mem_offset(&key, tensor_rect->start);
    /*
    llvm::errs() << LOG_TAB_L4 << "[Assign Lmem][stage: " << tensor_rect->first
                 << " ~ " << tensor_rect->last << "]"
                 << " tensor_id:" << tensor_rect->tid << ", addr:"
                 << tensor_rect->start << " ~ " << tensor_rect->end
                 << ", size:" << tensor_rect->size
                 << "  is_imm:" << key.is_layer_imm << "\n";
                 */
  }
  return 0;
}

LMemAllocProfileGuided::LMemAllocProfileGuided() {}
LMemAllocProfileGuided::~LMemAllocProfileGuided() {}

bmerr_t LMemAllocProfileGuided::assign_local_memory(Group *group,
                                               NetGraph *net_graph,
                                               net_timestep *time_step,
                                               bool one_shoot) {
  // calc tensor live range [first, last)
  std::map<int, std::shared_ptr<TensorRect>> tensor_map;

  auto insert_tensor = [&tensor_map, &net_graph](int tensor_id, int start, int end, int step_idx) {
    if (tensor_map.end() != tensor_map.find(tensor_id)) {
      tensor_map[tensor_id]->last = std::max(end, tensor_map[tensor_id]->last);
    } else {
      Tensor *tensor = net_graph->get_tensor_by_id(tensor_id);
      int tensor_size = ALIGN(tensor->lmem_size(), EU_NUM * tensor->unit_size());
      std::shared_ptr<TensorRect> tensor_rect = std::make_shared<TensorRect>(
          tensor_id, start, end, step_idx, tensor_size);
      tensor_map[tensor_id] = tensor_rect;
    }
  };

  int step_end = time_step->get_timestep_num();
  for (int i = 0; i < time_step->get_timestep_num(); ++i) {
    int layer_id = time_step->get_layer(i);
    const std::vector<TENSOR_STEP>& timestep_tensors = time_step->get_tensors(i);
    for (auto& step : timestep_tensors) {
      // if not one_shoot
      if (!one_shoot &&
          is_tensor_resident_in_lmem(net_graph->get_tensor_type(step.first))) {
        if (step.second == TIMESTEP_LOAD) {
          insert_tensor(step.first, 0, step_end, i);
        } else {
          insert_tensor(step.first, i, step_end, i);
        }
      } else {
        insert_tensor(step.first, i, i + 1, i);
      }
    }
    if (layer_id != -1) {
      const ImLayer* layer = net_graph->get_layer_by_id(layer_id);
      for (auto &tensor : layer->in_tensors) {
        insert_tensor(tensor->id(), i, i + 1, i);
      }
      for (auto &tensor : layer->out_tensors) {
        insert_tensor(tensor->id(), i, i + 1, i);
      }
      for (auto &tensor : layer->imm_tensors) {
        insert_tensor(tensor->id(), i, i + 1, i);
      }
    }
  }

  std::list<std::shared_ptr<TensorRect>> tensor_list;
  std::list<std::shared_ptr<TensorRect>> allocated_tensor_list;
  std::list<Line> offset_line_list;

  // sort tensor by width
  for (auto &tensor : tensor_map) {
    auto iter = std::find_if(tensor_list.begin(), tensor_list.end(),
                             [&tensor](std::shared_ptr<TensorRect> &a) {
                               return a->width < tensor.second->width;
                             });
    tensor_list.emplace(iter, tensor.second);
  }


  auto insert_offset_line = [&offset_line_list](int start, int first, int end) {
    auto iter = std::find_if(offset_line_list.begin(), offset_line_list.end(), [&](Line& line) {
      if (line.start > start) {
        return true;
      } else if (line.start == start) {
        return line.first > first;
      } else {
        return false;
      }
    });
    offset_line_list.emplace(iter, start, first, end);
  };

  offset_line_list.emplace_back(0, 0, step_end);
  // alloc addr
  while(!tensor_list.empty()) {

    // choosing the lowest and leftmost offset line
    auto line_iter = offset_line_list.begin();
    bool success_alloc = false;
    Line cur_line = *line_iter;

    // find longest lifetime tensor to place
    for (auto tensor_iter = tensor_list.begin(); tensor_iter != tensor_list.end();
         ++tensor_iter) {
      if ((*tensor_iter)->first >= line_iter->first &&
          (*tensor_iter)->last <= line_iter->last) {
        (*tensor_iter)->start = line_iter->start;
        (*tensor_iter)->end = (*tensor_iter)->start + (*tensor_iter)->size;
        if (LOCAL_MEM_SIZE < static_cast<uint32_t>((*tensor_iter)->end)) {
          LLVM_DEBUG(llvm::errs()
                         << LOG_TAB_L4
                         << "[Assign Lmem][Warning] lmem overflow: "
                         << (*tensor_iter)->end - LOCAL_MEM_SIZE << "\n";);
          return BM_ERR_FAILURE;
        }
        allocated_tensor_list.emplace_back(*tensor_iter);

        // split offset line
        offset_line_list.erase(line_iter);
        if ((*tensor_iter)->first - cur_line.first > 0) {
          insert_offset_line(cur_line.start, cur_line.first, (*tensor_iter)->first);
        }
        insert_offset_line((*tensor_iter)->end, (*tensor_iter)->first,
                           (*tensor_iter)->last);
        if (cur_line.last - (*tensor_iter)->last > 0) {
          insert_offset_line(cur_line.start, (*tensor_iter)->last, cur_line.last);
        }

        {
          LLVM_DEBUG(
          auto &tensor_rect = *tensor_iter;
          llvm::errs() << LOG_TAB_L4
                       << "[Assign Lmem][stage: " << tensor_rect->first << " ~ "
                       << tensor_rect->last << "]"
                       << " tensor_id:" << tensor_rect->tid
                       << ", addr:" << tensor_rect->start << " ~ "
                       << tensor_rect->end << ", size:" << tensor_rect->size
                       << "\n");
        }

        tensor_list.erase(tensor_iter);
        success_alloc = true;
        break;
      }
    }

    if (success_alloc) {
      continue;
    }

    // no appropriate tensor, lift up the line
    offset_line_list.erase(line_iter);
    for (auto iter = offset_line_list.begin(); iter != offset_line_list.end(); ++iter) {
      // if left neighbour can lift up
      if (iter->last == cur_line.first) {
        iter->last = cur_line.last;
        auto next_iter = std::next(iter, 1);
        if (next_iter != offset_line_list.end() &&
            next_iter->start == iter->start && next_iter->first == iter->last) {
          iter->last = next_iter->last;
          offset_line_list.erase(next_iter);
        }
        break;
      }

      // if right neighbour can lift up
      if (iter->first == cur_line.last) {
        iter->first = cur_line.first;
        break;
      }
    }
  }

  // alloc real addr
  figure_out_tensor_real_addr(allocated_tensor_list, net_graph, time_step);
  LLVM_DEBUG(llvm::errs() << "[Assgin Lmem Profile-guided] finish\n");
  return BM_SUCCESS;
}

int LMemAllocProfileGuided::figure_out_tensor_real_addr(
    std::list<std::shared_ptr<TensorRect>> &tensor_list, NetGraph *net_graph,
    net_timestep *time_step) {
  for (auto &tensor_rect : tensor_list) {
    Tensor *tensor = net_graph->get_tensor_by_id(tensor_rect->tid);
    tensor->laddr = tensor_rect->start;
    mem_buffer_key_t key;
    key.start_timestep = tensor_rect->step_idx;
    key.id_num = tensor_rect->tid;
    key.is_layer_imm = (tensor->type() == TENSOR_IMM);
    time_step->set_local_mem_offset(&key, tensor_rect->start);
    /*
    llvm::errs() << LOG_TAB_L4 << "[Assign Lmem][stage: " << tensor_rect->first
                 << " ~ " << tensor_rect->last << "]"
                 << " tensor_id:" << tensor_rect->tid << ", addr:"
                 << tensor_rect->start << " ~ " << tensor_rect->end
                 << ", size:" << tensor_rect->size
                 << "  is_imm:" << key.is_layer_imm << "\n";
                 */
  }
  return 0;
}
} // namespace mlir