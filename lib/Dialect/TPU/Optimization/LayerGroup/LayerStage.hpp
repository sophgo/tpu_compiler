/*
 * Copyright (C) Bitmain Technologies Inc.
 * All Rights Reserved.
 */
#ifndef GROUPOPS_LAYERSTAGE_H
#define GROUPOPS_LAYERSTAGE_H

#include "NetGraph.hpp"
#include "Tensor.hpp"
#include "utils.hpp"

namespace mlir {

class Group;

typedef struct mem_buffer_key {
  int start_timestep;
  int id_num;
  bool is_layer_imm;

  bool operator<(const mem_buffer_key& other) const {
    if (start_timestep < other.start_timestep) {
      return true;
    } else if (start_timestep == other.start_timestep) {
      return id_num < other.id_num;
    }
    return false;
  }
} mem_buffer_key_t;

typedef struct mem_buffer_value {
  int end_timestep;
  int local_mem_offset;
  int local_mem_size;
} mem_buffer_value_t;

typedef struct {
  int start_timestep;
  int end_timestep;
  int offset;
  int size;
} tensor_mem_t;

class net_timestep {
 public:
  explicit net_timestep(NetGraph* net_graph);
  net_timestep(const net_timestep& src);
  ~net_timestep();

  void add_layer_tensor_step(int layer_step, const std::vector<TENSOR_STEP>& tensor_step);

  int get_timestep_num();
  int get_layer(int time_step);
  const std::vector<TENSOR_STEP>& get_tensors(int time_step);

  void generate_mem_buffer();
  void update_mem_buffer_size();
  int get_minimal_slice_for_hw_limitation(Group * group);

  void generate_tsm_buffer(bool one_loop);

  const mem_buffer_value_t* get_mem_buffer_value(const mem_buffer_key_t* key);
  int get_mem_buffer_size(const mem_buffer_key_t* key);
  int get_mem_buffer_offset(const mem_buffer_key_t* key);
  int get_mem_buffer_end_timestep(const mem_buffer_key_t* key);
  void set_local_mem_offset(const mem_buffer_key_t* key, int local_mem_offset);
  void set_tsm_offset(int tensor_id, uint64_t offset);

  void show_timestep(std::ostream& pOs);
  void show_mem_buffer(std::ostream& pOs);
  void show_tsm_buffer(std::ostream& pOs);

  void show_timestep();
  void show_mem_buffer();
  void show_tsm_buffer();

  bool is_tensor_hold_in_memory(int tensor_id);

  void update_tensor_timestep(int time_idx, const std::vector<TENSOR_STEP>& new_tensor_timestep);

  const std::map<mem_buffer_key_t, mem_buffer_value_t>& get_memory_buffer() { return mem_buffer; }
  const std::map<int, tensor_mem_t>& get_tsm_buffer() { return tsm_buffer; }

  int tensor_range_end_timestep(const TENSOR_STEP& tensor_timestep);

  bmerr_t get_min_secs(bool b_hold_coeff, float &min_secs);
  bmerr_t find_minimal_slice(Group* group, int max_n_slice, int max_h_slice,
                          std::pair<int, int>& group_slice_);

 protected:
  NetGraph* net_graph_;
  int timestep_num;
  std::vector<int> layer_to_execute;
  std::vector<std::vector<TENSOR_STEP>> tensor_load_store;

  std::map<mem_buffer_key_t, mem_buffer_value_t> mem_buffer;  // for LMEM alloc
  std::map<int, tensor_mem_t> tsm_buffer;                     // for TSM alloc
  std::map<int, int> hold_coeff_tensor;

  void add_mem_buffer(int start_step, int end_step, int id, bool imm_tensor);
  bool is_tensor_weight(tensor_type_t tensor_type);
  int get_timestep_coeff_memory_req(int time_step);
  int get_timestep_memory_req(int time_step);
  void generate_hold_coeff_tensor();
};
}
#endif
