
#ifndef CLUSTER_TENSOR_H
#define CLUSTER_TENSOR_H

#include "utils.hpp"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
typedef enum tensor_type {
  TENSOR_NEURON = 0,
  TENSOR_MATRIX,
  TENSOR_COEFF,
  TENSOR_BIAS,
  TENSOR_COEFF_NEURON,
  TENSOR_NEURON_AS_COEFF,
  TENSOR_DEPTHCONV_OPD1,
  TENSOR_COEFF_WINOGRAD,
  TENSOR_NEURON_WINOGRAD,
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

typedef pair<int, TIMESTEP_LD_ST> TENSOR_STEP;

class ImmTensor;

class Tensor {
 public:
  Tensor(int id, int n, int c, int h, int w, int unit_size, const string& name,
         tensor_type_t type, uint64_t gaddr);
  Tensor(int id, int n, int c, int h, int w, int unit_size, string &storage,
         const string& name, tensor_type_t type,uint64_t gaddr);

  int n() const { return dims_[0]; }
  int c() const { return dims_[1]; }
  int h() const { return dims_[2]; }
  int w() const { return dims_[3]; }

  int unit_size() const { return unit_size_; }

  const int (&dims())[4] { return dims_; }

  tensor_type_t type() const { return type_; }

  const string& name() const { return name_; }

  const string& storage() { return storage_; }

  int id() const { return id_; }

  virtual uint32_t lmem_size();

  virtual uint32_t lmem_size(bool bMatrixTpye);

  uint64_t gmem_size();

  void set_nh_slice(int n_idx, int n_slice, int h_idx, int h_slice);

  void set_postfix(int group_id, int n_loop, int h_loop) {
    group = group_id;
    n_loop_ = n_loop;
    h_loop_ = h_loop;
  }

  int get_group_id() { return group; }
  int get_n_loop() { return n_loop_; }
  int get_h_loop() { return h_loop_; }

  void set_h_slice_skip_first() { this->h_slice_skip_first = true; }

  void set_h_slice_skip_last() { this->h_slice_skip_last = true; }

  static shared_ptr<Tensor> register_tensor(int n, int c, int h, int w, int unit_size, string& storage,
                                            const string& name, tensor_type_t type, uint64_t gaddr);

  static shared_ptr<Tensor> register_tensor(ShapedType *s_type, const string& name,
                                            tensor_type_t type, uint64_t gaddr);

  static shared_ptr<Tensor> register_imm_tensor(const shared_ptr<Tensor> associate, int count,
                                                const string& name);

  static void unregister_tensors() {
    map_id_to_tensor.clear();
    map_name_to_id_.clear();
    max_tensor_id = 0;
  }

  uint64_t gaddr;
  uint64_t tsm_addr;
  uint32_t laddr;

  int n_idx;
  int n_slice;
  int h_idx;
  int h_slice;
  int h_slice_max;
  int group;
  int h_loop_;
  int n_loop_;
  static int max_tensor_id;
  static map<int, shared_ptr<Tensor>> map_id_to_tensor;
  bool h_slice_skip_first;
  bool h_slice_skip_last;

 protected:
  int id_;
  int dims_[4];
  int unit_size_;
  tensor_type_t type_;
  string name_;
  string storage_;
  static map<string, int> map_name_to_id_;
};

class ImmTensor : public Tensor {
 public:
  ImmTensor(int id, shared_ptr<Tensor> associate, int count, const string& name)
      : Tensor(id, 0, 0, 0, 0, 0, name, TENSOR_IMM, 0xFFFFFFFF),
        associate_(associate),
        count_(count) {
    dims_[0] = associate->n();
    dims_[1] = associate->c();
    dims_[2] = associate->h();
    dims_[3] = associate->w();
    unit_size_ = associate->unit_size();
  }

  uint32_t lmem_size() override { return count_ * associate_.get()->lmem_size(); }

 private:
  shared_ptr<Tensor> associate_;
  int count_;
};

}
#endif
