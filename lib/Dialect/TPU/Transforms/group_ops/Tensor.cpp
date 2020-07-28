#include "Tensor.hpp"
#include <memory>

namespace mlir {

Tensor::Tensor(int id, int n, int c, int h, int w, int unit_size, const std::string& name,
               tensor_type_t type, int layer_id)
    : laddr(0),
      n_idx(-1),
      n_slice(-1),
      h_idx(-1),
      h_slice(-1),
      h_slice_max(-1),
      h_slice_skip_first(false),
      h_slice_skip_last(false),
      id_(id),
      layer_id_(layer_id),
      unit_size_(unit_size),
      type_(type),
      name_(name) {
  dims_[0] = n;
  dims_[1] = c;
  dims_[2] = h;
  dims_[3] = w;

  group = 1;
}

Tensor::Tensor(int id, int n, int c, int h, int w, int unit_size, std::string& storage, const std::string& name,
               tensor_type_t type, int layer_id)
    : laddr(0),
      n_idx(-1),
      n_slice(-1),
      h_idx(-1),
      h_slice(-1),
      h_slice_max(-1),
      h_slice_skip_first(false),
      h_slice_skip_last(false),
      id_(id),
      layer_id_(layer_id),
      unit_size_(unit_size),
      type_(type),
      name_(name),
      storage_(storage) {
  dims_[0] = n;
  dims_[1] = c;
  dims_[2] = h;
  dims_[3] = w;

  group = 1;
}

std::shared_ptr<Tensor> Tensor::register_tensor(int n, int c, int h, int w, int unit_size, std::string & storage,
                                           const std::string& name, tensor_type_t type, int layer_id) {
  int id;
  std::shared_ptr<Tensor> tensor;
  auto iter = map_name_to_id_.find(name);
  if (iter != map_name_to_id_.end()) {
    id = iter->second;
    tensor = map_id_to_tensor[id];
  } else {
    id = max_tensor_id++;
    map_name_to_id_[name] = id;
    tensor = std::make_shared<Tensor>(id, n, c, h, w, unit_size, storage, name, type, layer_id);
    map_id_to_tensor[id] = tensor;
  }

  return tensor;
}

std::shared_ptr<Tensor> Tensor::register_tensor(ShapedType *s_type, const std::string& name,
                                           tensor_type_t type, int layer_id) {
  int n = 0, c = 0, h = 0, w = 0;
  std::vector<int64_t> shape = s_type->getShape();
  switch (s_type->getRank()) {
    case 5:
      n = shape[0];
      c = shape[1] * shape[2];
      h = shape[3];
      w = shape[4];
      break;
    case 4:
      w = shape[3];
    case 3:
      h = shape[2];
    case 2:
      c = shape[1];
    case 1:
      n = shape[0];
      break;
    default:
      llvm::errs() << "Shape's dim size " << s_type->getRank() << " is unsupported.\n";
      assert(0);
  }

  int unit_size = s_type->getElementTypeBitWidth()/8;
  //  TODO: update storage
  std::string storage = "INT8";

  return register_tensor(n, c, h, w, unit_size, storage, name, type, layer_id);
}

std::shared_ptr<Tensor> Tensor::register_imm_tensor(const std::shared_ptr<Tensor> associate, int count,
                                               const std::string& name) {
  int id;
  std::shared_ptr<Tensor> tensor;
  auto iter = map_name_to_id_.find(name);
  if (iter != map_name_to_id_.end()) {
    id = iter->second;
    tensor = map_id_to_tensor[id];
  } else {
    id = max_tensor_id++;
    map_name_to_id_[name] = id;
    tensor = std::make_shared<ImmTensor>(id, associate, count, name);
    map_id_to_tensor[id] = tensor;
  }

  return tensor;
}

uint32_t Tensor::lmem_size() {
  int n = n_slice < 1 ? dims_[0] : n_slice;
  int c = dims_[1];
  int h = h_slice < 1 ? dims_[2] : h_slice;
  if (h_slice_max != -1) {
    h = h_slice_max;
  }
  int w = dims_[3];

  if (n == 0) {
    n = 1;
  }
  if (c == 0) {
    c = 1;
  }
  if (h == 0) {
    h = 1;
  }
  if (w == 0) {
    w = 1;
  }

  if (type_ == TENSOR_NEURON_WINOGRAD) {
    h = ALIGN(h, 2);
    w = ALIGN(w, 2);
  }

  if (type_ == TENSOR_IMM || type_ == TENSOR_NEURON || type_ == TENSOR_MATRIX ||
      type_ == TENSOR_DEPTHCONV_OPD1 || type_ == TENSOR_NEURON_WINOGRAD) {
    return n * ceiling_func(c, NPU_NUM) * ALIGN(h * w, EU_NUM) * unit_size_;
  } else if (type_ == TENSOR_COEFF_WINOGRAD) {
    return n * ceiling_func(c, NPU_NUM) * 4 * 4 * unit_size_;
  } else {
    return n * ceiling_func(c, NPU_NUM) * h * w * unit_size_;
  }
}

uint32_t Tensor::lmem_size(bool bMatrixTpye) {
  if (!bMatrixTpye) {
    return lmem_size();
  }

  int n, c, h, w;
  n = dims_[0];
  c = dims_[1];
  h = dims_[2];
  w = dims_[3];
  if (n == 0) {
    n = 1;
  }
  if (c == 0) {
    c = 1;
  }
  if (h == 0) {
    h = 1;
  }
  if (w == 0) {
    w = 1;
  }
  int row = n;
  int col = c * h * w;
  int channel_size_local = EU_NUM * unit_size_;
  uint32_t tensor_local_mem_size;
  tensor_local_mem_size =
      row * ceiling_func(ceiling_func(col, EU_NUM), NPU_NUM) * channel_size_local;
  return tensor_local_mem_size;
}

void Tensor::set_nh_slice(int n_idx, int n_slice, int h_idx, int h_slice) {
  this->n_idx = n_idx;
  this->n_slice = n_slice;
  this->h_idx = h_idx;
  this->h_slice = h_slice;
  if (h_slice != dims_[2] && h_slice > this->h_slice_max) {
    this->h_slice_max = h_slice;
  }
}

uint64_t Tensor::gmem_size() {
  int n = dims_[0];
  int c = dims_[1];
  int h = dims_[2];
  int w = dims_[3];

  if (n == 0 && c == 0 && h == 0 && w == 0) {
    return 0;
  }

  if (n == 0) {
    n = 1;
  }
  if (c == 0) {
    c = 1;
  }
  if (h == 0) {
    h = 1;
  }
  if (w == 0) {
    w = 1;
  }

  return static_cast<uint64_t>(n) * c * h * w * unit_size_;
}

int Tensor::max_tensor_id = 0;
std::map<std::string, int> Tensor::map_name_to_id_;
std::map<int, std::shared_ptr<Tensor>> Tensor::map_id_to_tensor;

}
