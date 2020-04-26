#include "NetGraph.hpp"
#include "ImLayer.hpp"

namespace mlir {

void NetGraph::parse_graph(FuncOp * fn){
  fn->walk([&](Operation * op) {
    if (isa<tpu::LoadWeightOp>(op) || isa<tpu::WeightFileOp>(op) ||
        isa<tpu::NoneOp>(op) ||
        isa<ReturnOp>(op)|| isa<FuncOp>(op)) {;}
    else{
      shared_ptr<ImLayer> layer = ImLayer::create(op);
      ImLayer::register_it(layer);

      vector<int> input_tensor_id;
      vector<int> output_tensor_id;

      int id = layer->id();

      llvm::errs() << "[" << id << "] " << layer->name() << " " << layer->type_name() << "\n";

      const auto& in_tensors = layer->in_tensors;
      const auto& out_tensors = layer->out_tensors;
      const auto& imm_tensors = layer->imm_tensors;

      for (auto& tensor : in_tensors) {
        int tid = tensor->id();
        input_tensor_id.push_back(tid);
        tensor_to_layer_id[tid].push_back(id);

        llvm::errs() << "IN:" << tid << ", name:" << tensor->name() << ", gsize:" << tensor->gmem_size()
                << ", lsize:" << tensor->lmem_size() << "\n";
      }

      for (auto& tensor : out_tensors) {
        int tid = tensor->id();
        output_tensor_id.push_back(tid);
        tensor_from_layer_id[tid] = id;

        llvm::errs() << "OUT:" << tid << ", name:" << tensor->name() << ", gsize:" << tensor->gmem_size()
                << ", lsize:" << tensor->lmem_size() << "\n";
      }

      if (imm_tensors.size() > 0) {
        auto& tensor = imm_tensors[0];
        int tid = tensor->id();
        tensor_from_layer_id[tid] = id;

        llvm::errs() << "IMM:" << tid << ", name:" << tensor->name() << ", gsize: 0"
                << ", lsize:" << tensor->lmem_size() << "\n";
      }

      layer_id_to_inout_tensor[id] = make_pair(input_tensor_id, output_tensor_id);

    }
  });
}

const ImLayer * NetGraph::get_layer_by_op(Operation * op) {
  for ( int i = 0; i < ImLayer::layers.size(); i++) {
    auto& layer = ImLayer::layers[i];
    if (layer->name() == getOpName(op).str()) {
      return layer.get();
    }
  }
  llvm::errs() << "can not get imlayer for op: " << getOpName(op) << "\n";
  assert(false);
  return NULL;
}

const ImLayer* NetGraph::get_layer_by_id(int layer_id) {
  auto& layer = ImLayer::layers[layer_id];
  return layer.get();
}

int NetGraph::getImLayerSize() {
  return ImLayer::layers.size();
}
const vector<int>& NetGraph::get_in_tensors_of_layer(int layer_id) {
  auto iter = layer_id_to_inout_tensor.find(layer_id);
  return iter->second.first;
}

const vector<int>& NetGraph::get_out_tensors_of_layer(int layer_id) {
  auto iter = layer_id_to_inout_tensor.find(layer_id);
  return iter->second.second;
}

bool NetGraph::layer_inplace_compute(int layer_id) {
  auto& layer = ImLayer::layers[layer_id];
  return layer->is_inplace_layer;
}

tensor_type_t NetGraph::get_tensor_type(int tensor_id) {
  auto iter = Tensor::map_id_to_tensor.find(tensor_id);
  return iter->second.get()->type();
}

int NetGraph::get_tensor_gmem_size(int tensor_id) {
  auto iter = Tensor::map_id_to_tensor.find(tensor_id);
  return iter->second.get()->gmem_size();
}

int NetGraph::get_tensor_nums(int tensor_id) {
  auto iter = Tensor::map_id_to_tensor.find(tensor_id);
  return iter->second.get()->n();
}

int NetGraph::get_tensor_channels(int tensor_id) {
  auto iter = Tensor::map_id_to_tensor.find(tensor_id);
  return iter->second.get()->c();
}

int NetGraph::get_tensor_height(int tensor_id) {
  auto iter = Tensor::map_id_to_tensor.find(tensor_id);
  return iter->second.get()->h();
}

int NetGraph::get_tensor_width(int tensor_id) {
  auto iter = Tensor::map_id_to_tensor.find(tensor_id);
  return iter->second.get()->w();
}

int NetGraph::get_tensor_unit_size(int tensor_id) {
  auto iter = Tensor::map_id_to_tensor.find(tensor_id);
  return iter->second.get()->unit_size();
}

void NetGraph::get_tensor_dim(int tensor_id, int* tensor_dim) {
  auto iter = Tensor::map_id_to_tensor.find(tensor_id);
  tensor_dim[0] = iter->second.get()->n();
  tensor_dim[1] = iter->second.get()->c();
  tensor_dim[2] = iter->second.get()->h();
  tensor_dim[3] = iter->second.get()->w();
}

gaddr_t NetGraph::get_tensor_global_mem(int tensor_id) {
  auto iter = Tensor::map_id_to_tensor.find(tensor_id);
  return iter->second.get()->gaddr;
}

gaddr_t NetGraph::get_tensor_tsm_mem(int tensor_id) {
  auto iter = Tensor::map_id_to_tensor.find(tensor_id);
  return iter->second.get()->tsm_addr;
}

void NetGraph::set_tensor_global_mem(int tensor_id, gaddr_t gaddr) {
  auto iter = Tensor::map_id_to_tensor.find(tensor_id);
  iter->second.get()->gaddr = gaddr;
}

void NetGraph::set_tensor_local_offest(int tensor_id, int local_mem_offset) {
  auto iter = Tensor::map_id_to_tensor.find(tensor_id);
  iter->second.get()->laddr = local_mem_offset;
}

void NetGraph::set_tensor_tsm_offest(int tensor_id, gaddr_t gaddr) {
  auto iter = Tensor::map_id_to_tensor.find(tensor_id);
  iter->second.get()->tsm_addr = gaddr;
}

void NetGraph::set_tensor_num_height_slice(int tensor_id, int n_idx, int n_slice, int h_idx,
                                            int h_slice, bool h_slice_skip_first,
                                            bool h_slice_skip_last) {
  auto iter = Tensor::map_id_to_tensor.find(tensor_id);
  iter->second.get()->n_idx = n_idx;
  iter->second.get()->n_slice = n_slice;
  iter->second.get()->h_idx = h_idx;
  iter->second.get()->h_slice = h_slice;
  iter->second.get()->h_slice_skip_first = h_slice_skip_first;
  iter->second.get()->h_slice_skip_last = h_slice_skip_last;
}

void NetGraph::set_tensor_height_slice_max(int tensor_id, int h_slice_max) {
  auto iter = Tensor::map_id_to_tensor.find(tensor_id);
  iter->second.get()->h_slice_max = h_slice_max;
}

Tensor* NetGraph::get_tensor_by_id(int id) {
  auto iter = Tensor::map_id_to_tensor.find(id);
  if (iter != Tensor::map_id_to_tensor.end()) {
    return iter->second.get();
  } else {
    cout << "wrong tensor id " << id << " when get tensor" << endl;
    assert(0);
  }
}

int NetGraph::get_tensor_local_offset(int tensor_id) {
  Tensor* tensor = get_tensor_by_id(tensor_id);
  return tensor->laddr;
}

int NetGraph::get_tensor_from_layer(int tensor_id) {
  auto iter = tensor_from_layer_id.find(tensor_id);
  if (iter != tensor_from_layer_id.end()) {
    return iter->second;
  } else {
    return (-1);
  }
}

const vector<int>& NetGraph::get_tensor_to_layer(int tensor_id) {
  auto iter = tensor_to_layer_id.find(tensor_id);
  if (iter != tensor_to_layer_id.end()) {
    return iter->second;
  } else {
    return dummy;
  }
}


/**
 * cluster_size = 0 UNKNOWN
 * cluster_size = 1 TG
 * cluster_size = 2 TL
 */
bool NetGraph::is_concat_special_case(int layer_id, int tid, int cluster_size) {
  const ImLayer* im_layer = get_layer_by_id(layer_id);
  if (im_layer->type() != IR_CONCAT) {
    return false;
  }

  auto op = cast<tpu::ConcatOp>(im_layer->op());
  const int axis = op.axis().getLimitedValue();
  assert(axis < 4);
  // if you don't consider the in tensor connected to the concat layer, let tid = -1.
  bool is_tg_layer = false;
  if (tid >= 0) {
    switch(cluster_size) {
    case 0: {
      const ImLayer* from_im_layer = get_layer_by_id(get_tensor_from_layer(tid));
      is_tg_layer = from_im_layer->is_tg_layer;
    } break;
    case 1: {
      is_tg_layer = true;
    } break;
    default:
      is_tg_layer = false;
      break;
    }
  }

  bool axis_special = false;
  if (is_tg_layer) {
    Tensor *tensor = get_tensor_by_id(tid);
    Operation *op = im_layer->op();
    int idx = 0;
    for (idx = 0; idx < bottom_size(op); idx++) {
      if (bottom_name(op, idx) == tensor->name()) {
        break;
      }
    }
    assert(bottom_name(op, idx) == tensor->name());

    /**
     * We need to make sure if we can make tg_concat in place
     * For axis = 1, n = 1
     * For axis = 2, n = 1 && c = 1
     * For axis = 3, n = 1 && c = 1 && h = 1
     *
     * and the dimensions of the input and output axis larger than
     * tg_concat_param().axis() must be the same.
     *
     * mul will always be one if in place is valid.
     */
    const vector<int64_t> &in_shape = input_shape(op, idx);
    const vector<int64_t> &out_shape = output_shape(op, 0);
    int mul = 1;
    for (int i = 0; i < axis; i++) {
      mul *= in_shape[i] * out_shape[i];
    }
    if (in_shape.size() == out_shape.size()) {
      for (int i = axis + 1; i < out_shape.size(); i++) {
        if (in_shape[i] != out_shape[i]) {
          mul *= 2;
        }
      }
    } else {
      mul *= 2;
    }

    if (0/*!FlagInst::get()->flagOpt(hwflags::OPT::TG_CONCAT_IN_PLACE)*/) {
      llvm::errs() << "Currently this target chip does not support tg tensor concat in place." << "\n";
      axis_special = false;
    } else if (mul == 1) {
      axis_special = true;
    }
  } else {
    if (axis == 1) {
      axis_special = true;
    }
  }
  // TODO
  //if (axis_special && (im_layer->op()->tg_concat_param().need_quantize_num() == 0)) {
  if (axis_special && 0) {
    return true;
  }
  return false;
}

// BM188X check whether concat layer can be optimized
bool NetGraph::is_concat_optimized_case(int layer_id, int tid, int cluster_size) {
  // TODO: open it later
  return false;
  if (is_concat_special_case(layer_id, tid, cluster_size)) {
    return true;
  }
  return false;
}




}