
#ifndef CLUSTER_TENSOR_H
#define CLUSTER_TENSOR_H

#include "utils.hpp"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
typedef enum tensor_type {
  TENSOR_NEURON = 0,
  TENSOR_COEFF,
  TENSOR_COEFF_CONV,
  TENSOR_IMM,  // intermediate buf for IR compute
} tensor_type_t;

typedef enum ld_st_type {
  TIMESTEP_LOAD = 0,
  TIMESTEP_STORE = 1,
  TIMESTEP_TSM_TO_LMEM = 2,
  TIMESTEP_LMEM_TO_TSM = 3,
  TIMESTEP_DDR_TO_TSM = 4,
  TIMESTEP_TSM_TO_DDR = 5,
} TIMESTEP_LD_ST;

typedef std::pair<int, TIMESTEP_LD_ST> TENSOR_STEP;

class ImmTensor;

class Tensor {
 public:
  Tensor(int id, int n, int c, int h, int w, int unit_size, const std::string& name,
         tensor_type_t type, int layer_id, bool eu_align = true);
  Tensor(int id, int n, int c, int h, int w, int unit_size, std::string &storage,
         const std::string& name, tensor_type_t type, int layer_id, bool eu_align = true);

  int n() const { return dims_[0]; }
  int c() const { return dims_[1]; }
  int h() const { return dims_[2]; }
  int w() const { return dims_[3]; }

  int unit_size() const { return unit_size_; }

  const int (&dims())[4] { return dims_; }

  tensor_type_t type() const { return type_; }

  const std::string& name() const { return name_; }

  const std::string& storage() { return storage_; }

  int id() const { return id_; }

  bool eu_align() const { return eu_align_; }

  int layer_id() const { return layer_id_; }

  virtual uint32_t lmem_size();

  virtual uint32_t lmem_size(bool bMatrixTpye);

  uint64_t gmem_size();

  void set_nh_slice(int n_idx, int n_slice, int h_idx, int h_slice);
  void set_nw_slice(int n_idx, int n_slice, int w_idx, int w_slice);

  void set_postfix(int group_id, int n_loop, int h_loop) {
    group = group_id;
    n_loop_ = n_loop;
    h_loop_ = h_loop;
  }

  int get_group_id() const { return group; }
  int get_n_loop() { return n_loop_; }
  int get_h_loop() { return h_loop_; }
  LG_Slice_Dim get_slice_dim() { return slice_dim_; }

  void set_h_slice_skip_first() { this->h_slice_skip_first = true; }

  void set_h_slice_skip_last() { this->h_slice_skip_last = true; }

  static std::shared_ptr<Tensor>
  register_tensor(int n, int c, int h, int w, int unit_size,
                  std::string &storage, const std::string &name,
                  tensor_type_t type, int layer_id, bool eu_align = true);

  static std::shared_ptr<Tensor>
  register_tensor(ShapedType *s_type, const std::string &name,
                  tensor_type_t type, int layer_id, bool eu_align = true,
                  std::string storage = "INT8");

  static std::shared_ptr<Tensor>
  register_imm_tensor(const std::shared_ptr<Tensor> associate, int count,
                      const std::string &name);

  static void unregister_tensors() {
    map_id_to_tensor.clear();
    map_name_to_id_.clear();
    max_tensor_id = 0;
  }

  uint64_t tsm_addr;
  uint32_t laddr;

  int n_idx;
  int n_slice;
  int h_idx;
  int h_slice;
  int h_slice_max;
  int w_idx;
  int w_slice;
  int w_slice_max;
  LG_Slice_Dim slice_dim_;
  int group;
  int h_loop_;
  int n_loop_;
  static int max_tensor_id;
  static std::map<int, std::shared_ptr<Tensor>> map_id_to_tensor;
  bool h_slice_skip_first;
  bool h_slice_skip_last;

 protected:
  int id_;
  int dims_[4];
  int unit_size_;
  tensor_type_t type_;
  std::string name_;
  std::string storage_;
  static std::map<std::string, int> map_name_to_id_;
  int layer_id_; // keep in mlir SSA
  bool eu_align_;
};

class ImmTensor : public Tensor {
 public:
  ImmTensor(int id, std::shared_ptr<Tensor> associate, int count, const std::string& name)
      : Tensor(id, 0, 0, 0, 0, 0, name, TENSOR_IMM, associate->layer_id()),
        associate_(associate),
        count_(count) {
    dims_[0] = associate->n();
    dims_[1] = associate->c();
    dims_[2] = associate->h();
    dims_[3] = associate->w();
    unit_size_ = associate->unit_size();
    eu_align_ = associate->eu_align();
  }

  uint32_t lmem_size() override { return count_ * associate_.get()->lmem_size(); }

 private:
  std::shared_ptr<Tensor> associate_;
  int count_;
};

}
#endif
