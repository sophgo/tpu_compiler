#include "Group.hpp"
#include "LayerStage.hpp"
#include "LMemManager.hpp"
#include "Steps.hpp"

#define DEBUG_TYPE "group_ops"

namespace mlir {

Group::~Group() {
  if (time_step) {
    delete time_step;
  }
}

// check if a tensor point to inside layer of current group
bool Group::is_group_inside_tensor(int tid) {
  const std::vector<int>& to_layers = net_graph_->get_tensor_to_layer(tid);
  if (to_layers.empty()) {
    return false;
  }

  for (int i = 0; i < (int)to_layers.size(); i++) {
    int id = to_layers[i];

    // if find to layers is in the group
    if (find(layers_.begin(), layers_.end(), id) != layers_.end()) {
      return true;
    }
  }

  return false;
}

// to check if a tensor points to out of group.
bool Group::is_group_out_tensor(int tid) {
  const std::vector<int>& to_layers = net_graph_->get_tensor_to_layer(tid);
  if (to_layers.empty()) {
    return true;
  }

  for (int i = 0; i < (int)to_layers.size(); i++) {
    int id = to_layers[i];

    if (find(layers_.begin(), layers_.end(), id) == layers_.end()) {
      return true;
    }
  }
  return false;
}

// Check if a tensor belongs to neuron tensor.
bool Group::is_group_in_neuron_tensor(int tid) {
  std::set<int> in_neurons = get_group_in_neuron_tensors();
  return in_neurons.find(tid) != in_neurons.end();
}

// Get out tensor not in the group
std::vector<int> Group::get_group_out_tensors() {
  std::vector<int> group_out_tensors;

  for (int i = 0; i < static_cast<int>(layers_.size()); i++) {
    int id = layers_[i];
    const std::vector<int>& out_tensors = net_graph_->get_out_tensors_of_layer(id);

    for (int j = 0; j < static_cast<int>(out_tensors.size()); ++j) {
      int tid = out_tensors[j];
      if (is_group_out_tensor(tid)) {
        group_out_tensors.push_back(tid);
      }
    }
  }

  return group_out_tensors;
}

// Get neuron tensor in group.
std::set<int> Group::get_group_in_neuron_tensors() {
  std::set<int> group_in_neuron_tensors;

  for (int i = 0; i < static_cast<int>(layers_.size()); i++) {
    const ImLayer* layer = net_graph_->get_layer_by_id(layers_[i]);

    for (auto& tensor : layer->in_tensors) {
      int tid = tensor->id();
      tensor_type_t type = tensor->type();
      if (TENSOR_NEURON != type && TENSOR_MATRIX != type) {
        continue;
      }

      int from_layer = net_graph_->get_tensor_from_layer(tid);
      if (from_layer == -1) {
        continue;
      }

      if (find(layers_.begin(), layers_.end(), from_layer) != layers_.end()) {
        continue;
      }

      group_in_neuron_tensors.insert(tid);
    }
  }

  return group_in_neuron_tensors;
}

void Group::show_group_layers() {
  for (int i = 0; i < (int)layers_.size(); i++)
    LLVM_DEBUG(llvm::errs() << " " << layers_[i];);
  LLVM_DEBUG(llvm::errs() << "\n";);
}

// This function is used to construct the time step and find the appropriate partitioning
// strategy according to the time step.
bmerr_t Group::assign_steps() {
  bmerr_t status = BM_SUCCESS;
  LLVM_DEBUG(llvm::errs() << LOG_TAB_L1 << "[Check Group] Begin: ";);
  show_group_layers();

  // check layer pattern we do not support
  status = check_if_pattern_support();
  if (BM_ERR_FAILURE == status) {
    LLVM_DEBUG(llvm::errs() << LOG_TAB_L2
                          << "[Find_Fit_NH_Slice] Failed with pattern not support\n";);
    return status;
  }

  // check if we can slice layers
  status = check_if_can_slice_group();
  if (BM_ERR_FAILURE == status) {
    LLVM_DEBUG(llvm::errs() << LOG_TAB_L1 << "[Check Group] End with Failed: ";);
    show_group_layers();
    return status;
  }

  LLVM_DEBUG(llvm::errs() << LOG_TAB_L1 << "[Check Group] End with Valid: ";);
  show_group_layers();

  return status;
}

void Group::set_slice_dim(LG_Slice_Dim slice_dim) {
  slice_dim_ = slice_dim;
}

LG_Slice_Dim Group::get_slice_dim() {
  return slice_dim_;
}

bool Group::check_valid() {
  set_slice_dim(LG_Slice_Dim_H);
  if(assign_steps() == BM_SUCCESS) {
    return true;
  }

  // set_slice_dim(LG_Slice_Dim_W);
  // if(assign_steps() == BM_SUCCESS) {
  //   return true;
  // }

  return false;
}

bool Group::check_if_can_slice_group(){
  bmerr_t status = BM_SUCCESS;
  // assgin timestep
  time_step = new net_timestep(net_graph_);
  GroupSteps::timestep_assgin(net_graph_, this, time_step);

  if (layers_.size() == 1) {
    return BM_SUCCESS;
  }

  // try to slice group
  group_slice_ = {1, 1};
  int max_n_slice = get_batch_num();
  int max_hw_slice = get_max_secs();
  reset_tensor_hwslice_max();
  status = time_step->find_minimal_slice(this, max_n_slice,
                                      max_hw_slice, group_slice_);
  if (status == BM_ERR_FAILURE) {
    return BM_ERR_FAILURE;
  }

  LLVM_DEBUG(llvm::errs() << LOG_TAB_L2
                          << "[Find_Fit_NH_Slice] Begin\n";);
  while (group_slice_.first <= max_n_slice && group_slice_.second <= max_hw_slice) {
    LLVM_DEBUG(llvm::errs() << LOG_TAB_L3
                            << "[Find_Fit_NH_Slice] check n_slice and h_slice: ("
                            << group_slice_.first << "/" << max_n_slice
                            << ", " << group_slice_.second << "/" << max_hw_slice << ")\n";);
    reset_tensor_hwslice_max();
    // check validation of layer group if group_slice_.second > 1,
    // and update h_slice_max
    if (group_slice_.second > 1) {
      for (int slice_idx = 0; slice_idx < group_slice_.second; slice_idx++) {
        status = update_slices(group_slice_.first,
                                      group_slice_.second, 0, slice_idx);
        if (status == BM_ERR_FAILURE) {
          LLVM_DEBUG(llvm::errs() << LOG_TAB_L3
                                  << "[Find_Fit_NH_Slice] End with failed: Update tensor slice failed\n";);
          return BM_ERR_FAILURE;
        }
      }
    }

    status = GroupSteps::balance_tdma_tiu(net_graph_, this, &time_step, group_slice_);

    if (status == BM_ERR_FAILURE) {
      if (group_slice_.first < max_n_slice) {
        group_slice_.first++;
      } else {
        group_slice_.second++;
      }
    } else {
      break;
    }
  }
  LLVM_DEBUG(llvm::errs() << LOG_TAB_L2
                          << "[Find_Fit_NH_Slice] Success with n slice: "
                          << group_slice_.first << " h slice: "
                          << group_slice_.second << "\n";);
  return status;

}


int Group::get_batch_num() const {
  int any_layer = layers_[0];
  int any_tensor = net_graph_->get_in_tensors_of_layer(any_layer)[0];
  return net_graph_->get_tensor_nums(any_tensor);
}

void Group::set_slice_limit(int s) {
  slice_limit_ = (LG_Slice_Limit)s;
}

int Group::get_max_secs() {
  int max_slice = 32;
  if (slice_limit_ == USE_FIT_H_SLICE) {
    int any_layer = layers_[0];
    int any_tensor = net_graph_->get_in_tensors_of_layer(any_layer)[0];
    // the temporary code for the large input tensor, such as yolo-608
    // TODO: to get better h split, balance gdma cost and abundant computation

    // when FC group with FC, don't split for now.
    if (net_graph_->get_tensor_height(any_tensor) == 0) {
      max_slice = 1;
      return max_slice;
    }
    if (net_graph_->get_tensor_height(any_tensor) >= 600 ||
        net_graph_->get_tensor_width(any_tensor) >= 600) {
      max_slice = 16;
    } else if (net_graph_->get_tensor_height(any_tensor) >= 400 ||
              net_graph_->get_tensor_width(any_tensor) >= 400) {
      max_slice = 8;
    } else if (net_graph_->get_tensor_height(any_tensor) >= 250 ||
              net_graph_->get_tensor_width(any_tensor) >= 250) {
      max_slice = 4;
    } else {
      max_slice = 8;
    }
  } else if (slice_limit_ == USE_MAX_H_SLICE) {
    max_slice = 1024;
  }

  return max_slice;
}



static bool is_output_op(Operation * op) {
  for (auto &use : op->getResult(0).getUses()) {
    auto useOp = use.getOwner();
    if (isa<ReturnOp>(useOp))
      return true;
  }
  return false;
}

// pattern that not support for group
bool Group::check_if_pattern_support() {
  // pattern 1: tl concat's input cannot be group output tensor
  for (auto id : layers_) {
    const ImLayer* im_layer = net_graph_->get_layer_by_id(id);
    if (im_layer->type() == IR_CONCAT){
      std::vector<int> out_tensors = get_group_out_tensors();
      for (auto& tensor : im_layer->in_tensors) {
        int in_id = tensor->id();
        if (find(out_tensors.begin(), out_tensors.end(), in_id) != out_tensors.end()) {
          return BM_ERR_FAILURE;
        }
      }
    }
  }

  // output should not be the input of other layer
  // in the same group
  for (auto id : layers_) {
    const ImLayer* im_layer = net_graph_->get_layer_by_id(id);
    Operation * op = im_layer->op();
    if (is_output_op(op)) {
      // is out tensor point to inside layer, return failure
      for (auto& tensor : im_layer->out_tensors) {
        if (is_group_inside_tensor(tensor->id())) {
          return BM_ERR_FAILURE;
        }
      }
    }
  }

  // deconv not support slice w
  if (slice_dim_ == LG_Slice_Dim_W) {
    for (auto id : layers_) {
      const ImLayer* im_layer = net_graph_->get_layer_by_id(id);
      Operation * op = im_layer->op();
      if (isa<tpu::TG_INT8_PC_DeConv2DOp>(op) ||
          isa<tpu::TG_BF16_DeConv2DOp>(op)) {
        return BM_ERR_FAILURE;
      }
    }
  }
  return BM_SUCCESS;
}

bool Group::validate_nh_slice() {
  for (auto id : layers_) {
    const ImLayer* im_layer = net_graph_->get_layer_by_id(id);

    for (auto& tensor : im_layer->in_tensors) {
      if (tensor->type() == TENSOR_COEFF || tensor->type() == TENSOR_BIAS ||
          tensor->type() == TENSOR_COEFF_LUT ||
          tensor->type() == TENSOR_DEPTHCONV_OPD1) {
        continue;
      }

      if (tensor->h_slice < 1) {
        LLVM_DEBUG(llvm::errs() << "FAIL: h_slice of tensor[" << tensor->id() << "] = "
                                << tensor->h_slice << " is smaller than kh = 1" << "\n";);
        return false;
      } else if (tensor->h_slice > tensor->h()) {
        LLVM_DEBUG(llvm::errs() << "FAIL: h_slice " << tensor->h_slice
                     << " of tensor[" << tensor->id() << "] is larger than tensor height: "
                     << tensor->h() <<  "\n";);
        return false;
      }
    }
  }

  // Validate out tensor slice
  std::vector<int> out_tensors = get_group_out_tensors();
  for (auto tid : out_tensors) {
    Tensor* tensor = net_graph_->get_tensor_by_id(tid);

    if (tensor->h_idx >= tensor->h()) {
      LLVM_DEBUG(llvm::errs() << "FAIL: h_idx of out tensor[" << tensor->id()
                   << "] = " << tensor->h_idx << " is larger than tensor height = "
                   << tensor->h() << "\n";);
      return false;
    }
  }

  return true;
}

bool Group::validate_nw_slice() {
  for (auto id : layers_) {
    const ImLayer* im_layer = net_graph_->get_layer_by_id(id);

    for (auto& tensor : im_layer->in_tensors) {
      if (tensor->type() == TENSOR_COEFF || tensor->type() == TENSOR_BIAS ||
          tensor->type() == TENSOR_COEFF_LUT ||
          tensor->type() == TENSOR_DEPTHCONV_OPD1) {
        continue;
      }

      if (tensor->w_slice < 1) {
        LLVM_DEBUG(llvm::errs() << "FAIL: w_slice of tensor[" << tensor->id() << "] = "
                                << tensor->w_slice << " is smaller than kw = 1" << "\n";);
        return false;
      } else if (tensor->w_slice > tensor->w()) {
        LLVM_DEBUG(llvm::errs() << "FAIL: w_slice " << tensor->w_slice
                     << " of tensor[" << tensor->id() << "] is larger than tensor height: "
                     << tensor->w() <<  "\n";);
        return false;
      }
    }
  }

  // Validate out tensor slice
  std::vector<int> out_tensors = get_group_out_tensors();
  for (auto tid : out_tensors) {
    Tensor* tensor = net_graph_->get_tensor_by_id(tid);

    if (tensor->w_idx >= tensor->w()) {
      LLVM_DEBUG(llvm::errs() << "FAIL: w_idx of out tensor[" << tensor->id()
                   << "] = " << tensor->w_idx << " is larger than tensor height = "
                   << tensor->w() << "\n";);
      return false;
    }
  }

  return true;
}

void Group::reset_tensor_slice() {
  for (auto id : layers_) {
    const ImLayer* im_layer = net_graph_->get_layer_by_id(id);

    for (auto& tensor : im_layer->in_tensors) {
      net_graph_->reset_tensor_slice(tensor->id());
    }

    for (auto& tensor : im_layer->out_tensors) {
      net_graph_->reset_tensor_slice(tensor->id());
    }
  }
}

void Group::reset_tensor_hwslice_max() {
  for (auto id : layers_) {
    const ImLayer* im_layer = net_graph_->get_layer_by_id(id);

    for (auto& tensor : im_layer->in_tensors) {
      net_graph_->set_tensor_h_slice_max(tensor->id(), -1);
      net_graph_->set_tensor_w_slice_max(tensor->id(), -1);
    }

    for (auto& tensor : im_layer->out_tensors) {
      net_graph_->set_tensor_h_slice_max(tensor->id(), -1);
      net_graph_->set_tensor_w_slice_max(tensor->id(), -1);
    }
  }
}

// Breadth-first traversal of all the tensors and set n_slice and h_slice.
bool Group::backward_nh_slice(int out_tensor_id, std::list<int>& branches, bool max_h_slice,
                           bool no_split_h, int n_loop, int h_loop) {
  int id = net_graph_->get_tensor_from_layer(out_tensor_id);

  // the out tensor is the input tensor of the group
  if (id < 0 || find(layers_.begin(), layers_.end(), id) == layers_.end()) {
    return true;
  }

  const ImLayer* im_layer = net_graph_->get_layer_by_id(id);
  IR_TYPE layer_type = im_layer->type();

  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w;
  int sh, sw, pt, pb, pl, pr, dh = 1, dw, pad_value;

  if (layer_type == IR_CONVOLUTION ||
      layer_type == IR_DECONVOLUTION) {
    bool do_ic_align = false;
    bool do_leaky_relu = false;
    getConvParam(im_layer->op(), n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw,
                 pt, pb, pl, pr, dh, dw, is_dw, with_bias, do_relu, do_ic_align,
                 do_leaky_relu, pad_value);

    if (dh > 1) {
      kh = dh * (kh - 1) + 1;
    }
  } else if (layer_type == IR_POOLING) {
    bool is_global = false;
    bool count_include_pad;
    getPoolingParam(im_layer->op(),
                    n, ic, ih, iw, oh, ow,
                    kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
                    is_global, do_relu, count_include_pad);
  }

  Tensor* out_tensor = net_graph_->get_tensor_by_id(out_tensor_id);
  int n_slice = out_tensor->n_slice;
  int out_h_slice = out_tensor->h_slice;

  int n_idx = out_tensor->n_idx > 0 ? out_tensor->n_idx : 0;
  int out_h_idx = out_tensor->h_idx > 0 ? out_tensor->h_idx : 0;
  int height = net_graph_->get_tensor_height(out_tensor_id);

  if (!max_h_slice) {
    int h_end = out_tensor->h_idx + out_h_slice;
    out_h_slice = h_end > height ? height - out_h_idx : h_end - out_h_idx;
  }

  int h_slice, h_idx;
  const std::vector<int>& back_tensors =
        net_graph_->get_in_tensors_of_layer(id);

  for (uint32_t i = 0; i < back_tensors.size(); ++i) {
    Tensor* tensor = net_graph_->get_tensor_by_id(back_tensors[i]);

    if (tensor->type() == TENSOR_COEFF || tensor->type() == TENSOR_BIAS ||
        tensor->type() == TENSOR_COEFF_LUT ||
        tensor->type() == TENSOR_DEPTHCONV_OPD1) {
      continue;
    }

    int cur_h_idx = tensor->h_idx;
    int cur_h_slice = tensor->h_slice;

    if (no_split_h) {
      h_idx = 0;
      h_slice = tensor->h();
    } else if (layer_type == IR_CONVOLUTION || layer_type == IR_POOLING) {
      h_idx = out_h_idx * sh - pt;
      h_slice = (out_h_slice - 1) * sh + kh;
    } else if (layer_type == IR_DECONVOLUTION) {
      int bottom_h = tensor->h();
      int height_insert0 = (bottom_h - 1) * sh + 1;
      int real_o_h_t = out_h_idx;
      int real_o_h_b = out_h_idx + out_h_slice;
      int kh_ext = (kh - 1) * dh + 1;
      pt = kh_ext - pt - 1;
      int if_pad_h_t = real_o_h_t;
      int if_pad_h_b = real_o_h_b + kh_ext - 1;
      int if_insert_h_t = 0;
      if (if_pad_h_t >= pt) {
        if_insert_h_t = if_pad_h_t - pt;
      }
      int if_insert_h_b = height_insert0;
      if ((if_pad_h_b - pt) < height_insert0) {
        if_insert_h_b = if_pad_h_b - pt;
      }
      h_idx = (if_insert_h_t + sh - 1) / sh;
      h_slice = (if_insert_h_b + sh - 1) / sh - h_idx;
    } else if (layer_type == IR_UPSAMPLE) {
      int size_h = 1;
      int size_w = 1;
      getUpsampleParam(im_layer->op(), size_h, size_w);

      if (out_h_slice % size_h) {
        LLVM_DEBUG(llvm::errs() << LOG_TAB_L3
                                << "FAIL: fractional upsample input h slice" << "\n";);
        return false;
      }

      h_idx = out_h_idx / size_h;
      h_slice = out_h_slice / size_h;
    } else if ( layer_type == IR_ELTWISE ) {
      h_idx = out_h_idx;
      h_slice = out_h_slice;
      if (auto op = dyn_cast<tpu::TG_INT8_EltwiseAddOp>(im_layer->op())) {
        bool do_early_stride = false;
        int h_stride = 0, w_stride = 0;
        getEltwiseAddParam(im_layer->op(), do_early_stride, h_stride, w_stride);
        if (do_early_stride) {
          h_idx = out_h_idx * h_stride;
          h_slice = out_h_slice * h_stride;
        }
      } else if (auto op = dyn_cast<tpu::TG_BF16_EltwiseAddOp>(im_layer->op())) {
        bool do_early_stride = false;
        int h_stride = 0, w_stride = 0;
        getEltwiseAddParam(im_layer->op(), do_early_stride, h_stride, w_stride);
        if (do_early_stride) {
          h_idx = out_h_idx * h_stride;
          h_slice = out_h_slice * h_stride;
        }
      }
    } else if (layer_type == IR_PAD) {
      h_slice = out_h_slice;
      std::vector<int32_t> pads;
      if (isa<tpu::TG_INT8_PadOp>(im_layer->op())) {
        auto pad_op = cast<tpu::TG_INT8_PadOp>(im_layer->op());
        arrayAttrToVector(pad_op.pads().getValue(), pads);
      } else if (isa<tpu::TG_BF16_PadOp>(im_layer->op())) {
        auto pad_op = cast<tpu::TG_BF16_PadOp>(im_layer->op());
        arrayAttrToVector(pad_op.pads().getValue(), pads);
      }

      h_idx = out_h_idx ? out_h_idx - pads[2] : 0;
      if (out_h_idx == 0) {
        if (out_h_slice == out_tensor->h())
          h_slice = out_h_slice - pads[2] - pads[6];
        else
          h_slice = out_h_slice - pads[2];
      } else
        h_slice = out_h_slice;
    } else if (layer_type == IR_CROP) {
      std::vector<int32_t> crop_offsets;
      if (isa<tpu::TG_INT8_CropOp>(im_layer->op())) {
        auto crop_op = cast<tpu::TG_INT8_CropOp>(im_layer->op());
        arrayAttrToVector(crop_op.crop_offset().getValue(), crop_offsets);
      } else if(isa<tpu::TG_BF16_CropOp>(im_layer->op())) {
        auto crop_op = cast<tpu::TG_BF16_CropOp>(im_layer->op());
        arrayAttrToVector(crop_op.crop_offset().getValue(), crop_offsets);
      }

      h_idx = out_h_idx ? out_h_idx + crop_offsets[2] : 0;
      if (out_h_idx == 0) {
        h_slice = out_h_slice + crop_offsets[2];
      } else
        h_slice = out_h_slice;
    } else {
      h_idx = out_h_idx;
      h_slice = out_h_slice;
    }

    if (tensor->type() == TENSOR_COEFF_NEURON || tensor->type() == TENSOR_DEPTHCONV_OPD1 ||
        tensor->type() == TENSOR_NEURON_AS_COEFF) {
      if (tensor->n() == 1) {
        n_idx = 0;
        n_slice = 1;
      }
      if (tensor->h() == 1) {
        h_idx = 0;
        h_slice = 1;
      }
    }

    if (cur_h_slice != -1 && (cur_h_slice != h_slice || cur_h_idx != h_idx)) {
      LLVM_DEBUG(llvm::errs()
        << LOG_TAB_L3
        << "[Update Tensor Slice][Warning]: "
        << "data slice in h dimension is conflicted for tensor "
        << back_tensors[i] << " cur_h_idx:" << cur_h_idx << " h_idx:" << h_idx
        << " cur_h_slice:" << cur_h_slice << " h_slice:" << h_slice << "\n";);
      return false;
    }

    if (n_slice < 1 || h_slice < 1) {
      LLVM_DEBUG(llvm::errs()
        << LOG_TAB_L3
        << "[Update Tensor Slice][Warning]: "
        << "slice length should >= 1, but "
        << "n_slice is: " << n_slice << ", h_slice is: " << h_slice << "\n";);
      return false;
    }

    tensor->set_nh_slice(n_idx, n_slice, h_idx, h_slice);
    tensor->set_postfix(group_id_, n_loop, h_loop);

    if (cur_h_slice == -1) {
      branches.push_back(back_tensors[i]);
    }
  }

  return true;
}

// Breadth-first traversal of all the tensors and set n_slice and h_slice.
bool Group::backward_nw_slice(int out_tensor_id, std::list<int>& branches, bool max_w_slice,
                           bool no_split_w, int n_loop, int w_loop) {
  int id = net_graph_->get_tensor_from_layer(out_tensor_id);

  // the out tensor is the input tensor of the group
  if (id < 0 || find(layers_.begin(), layers_.end(), id) == layers_.end()) {
    return true;
  }

  const ImLayer* im_layer = net_graph_->get_layer_by_id(id);
  IR_TYPE layer_type = im_layer->type();

  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w;
  int sh, sw, pt, pb, pl, pr, dh = 1, dw, pad_value;

  if (layer_type == IR_CONVOLUTION ||
      layer_type == IR_DECONVOLUTION) {
    bool do_ic_align = false;
    bool do_leaky_relu = false;
    getConvParam(im_layer->op(), n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw,
                 pt, pb, pl, pr, dh, dw, is_dw, with_bias, do_relu, do_ic_align,
                 do_leaky_relu, pad_value);
    if (dw > 1) {
      kw = dw * (kw - 1) + 1;
    }
  } else if (layer_type == IR_POOLING) {
    bool is_global = false;
    bool count_include_pad;
    getPoolingParam(im_layer->op(),
                    n, ic, ih, iw, oh, ow,
                    kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
                    is_global, do_relu, count_include_pad);
  }

  Tensor* out_tensor = net_graph_->get_tensor_by_id(out_tensor_id);
  int n_slice = out_tensor->n_slice;
  int out_w_slice = out_tensor->w_slice;

  int n_idx = out_tensor->n_idx > 0 ? out_tensor->n_idx : 0;
  int out_w_idx = out_tensor->w_idx > 0 ? out_tensor->w_idx : 0;
  int width = net_graph_->get_tensor_width(out_tensor_id);

  if (!max_w_slice) {
    int w_end = out_tensor->w_idx + out_w_slice;
    out_w_slice = w_end > width ? width - out_w_idx : w_end - out_w_idx;
  }

  int w_slice, w_idx;
  const std::vector<int>& back_tensors =
        net_graph_->get_in_tensors_of_layer(id);

  for (uint32_t i = 0; i < back_tensors.size(); ++i) {
    Tensor* tensor = net_graph_->get_tensor_by_id(back_tensors[i]);

    if (tensor->type() == TENSOR_COEFF || tensor->type() == TENSOR_BIAS ||
        tensor->type() == TENSOR_COEFF_LUT ||
        tensor->type() == TENSOR_DEPTHCONV_OPD1) {
      continue;
    }

    int cur_w_idx = tensor->w_idx;
    int cur_w_slice = tensor->w_slice;

    if (no_split_w) {
      w_idx = 0;
      w_slice = tensor->w();
    } else if (layer_type == IR_CONVOLUTION || layer_type == IR_POOLING) {
      w_idx = out_w_idx * sw - pl;
      w_slice = (out_w_slice - 1) * sw + kw;
    } else if (layer_type == IR_DECONVOLUTION) {
      int right_w = tensor->w();
      int width_insert0 = (right_w - 1) * sw + 1;
      int real_o_w_t = out_w_idx;
      int real_o_w_b = out_w_idx + out_w_slice;
      int kw_ext = (kw - 1) * dw + 1;
      pl = kw_ext - pl - 1;
      int if_pad_w_l = real_o_w_t;
      int if_pad_w_r = real_o_w_b + kw_ext - 1;
      int if_insert_w_l = 0;
      if (if_pad_w_l >= pl) {
        if_insert_w_l = if_pad_w_l - pl;
      }
      int if_insert_w_r = width_insert0;
      if ((if_pad_w_r - pl) < width_insert0) {
        if_insert_w_r = if_pad_w_r - pl;
      }
      w_idx = (if_insert_w_l + sw - 1) / sw;
      w_slice = (if_insert_w_r + sw - 1) / sw - w_idx;
    } else if (layer_type == IR_UPSAMPLE) {
      int size_h = 1;
      int size_w = 1;
      getUpsampleParam(im_layer->op(), size_h, size_w);

      if (out_w_slice % size_w) {
        LLVM_DEBUG(llvm::errs() << LOG_TAB_L3
                                << "FAIL: fractional upsample input h slice" << "\n";);
        return false;
      }

      w_idx = out_w_idx / size_w;
      w_slice = out_w_slice / size_w;
    } else if ( layer_type == IR_ELTWISE ) {
      w_idx = out_w_idx;
      w_slice = out_w_slice;
      if (auto op = dyn_cast<tpu::TG_INT8_EltwiseAddOp>(im_layer->op())) {
        bool do_early_stride = false;
        int h_stride = 0, w_stride = 0;
        getEltwiseAddParam(im_layer->op(), do_early_stride, h_stride, w_stride);
        if (do_early_stride) {
          w_idx = out_w_idx * w_stride;
          w_slice = out_w_slice * w_stride;
        }
      } else if (auto op = dyn_cast<tpu::TG_BF16_EltwiseAddOp>(im_layer->op())) {
        bool do_early_stride = false;
        int h_stride = 0, w_stride = 0;
        getEltwiseAddParam(im_layer->op(), do_early_stride, h_stride, w_stride);
        if (do_early_stride) {
          w_idx = out_w_idx * w_stride;
          w_slice = out_w_slice * w_stride;
        }
      }
    } else if (layer_type == IR_PAD) {
      w_slice = out_w_slice;
      std::vector<int32_t> pads;
      if (isa<tpu::TG_INT8_PadOp>(im_layer->op())) {
        auto pad_op = cast<tpu::TG_INT8_PadOp>(im_layer->op());
        arrayAttrToVector(pad_op.pads().getValue(), pads);
      } else if (isa<tpu::TG_BF16_PadOp>(im_layer->op())) {
        auto pad_op = cast<tpu::TG_BF16_PadOp>(im_layer->op());
        arrayAttrToVector(pad_op.pads().getValue(), pads);
      }

      w_idx = out_w_idx ? out_w_idx - pads[3] : 0;
      if (out_w_idx == 0) {
        if (out_w_slice == out_tensor->w())
          w_slice = out_w_slice - pads[3] - pads[7];
        else
          w_slice = out_w_slice - pads[3];
      } else
        w_slice = out_w_slice;
    } else if (layer_type == IR_CROP) {
      std::vector<int32_t> crop_offsets;
      if (isa<tpu::TG_INT8_CropOp>(im_layer->op())) {
        auto crop_op = cast<tpu::TG_INT8_CropOp>(im_layer->op());
        arrayAttrToVector(crop_op.crop_offset().getValue(), crop_offsets);
      } else if(isa<tpu::TG_BF16_CropOp>(im_layer->op())) {
        auto crop_op = cast<tpu::TG_BF16_CropOp>(im_layer->op());
        arrayAttrToVector(crop_op.crop_offset().getValue(), crop_offsets);
      }

      w_idx = out_w_idx ? out_w_idx + crop_offsets[3] : 0;
      if (out_w_idx == 0) {
        w_slice = out_w_slice + crop_offsets[3];
      } else
        w_slice = out_w_slice;
    } else {
      w_idx = out_w_idx;
      w_slice = out_w_slice;
    }

    if (tensor->type() == TENSOR_COEFF_NEURON || tensor->type() == TENSOR_DEPTHCONV_OPD1 ||
        tensor->type() == TENSOR_NEURON_AS_COEFF) {
      if (tensor->n() == 1) {
        n_idx = 0;
        n_slice = 1;
      }
      if (tensor->w() == 1) {
        w_idx = 0;
        w_slice = 1;
      }
    }

    if (cur_w_slice != -1 && (cur_w_slice != w_slice || cur_w_idx != w_idx)) {
      LLVM_DEBUG(llvm::errs()
        << LOG_TAB_L3
        << "[Update Tensor Slice][Warning]: "
        << "data slice in w dimension is conflicted for tensor "
        << back_tensors[i] << " cur_w_idx:" << cur_w_idx << " w_idx:" << w_idx
        << " cur_w_slice:" << cur_w_slice << " w_slice:" << w_slice << "\n";);
      return false;
    }

    if (n_slice < 1 || w_slice < 1) {
      LLVM_DEBUG(llvm::errs()
        << LOG_TAB_L3
        << "[Update Tensor Slice][Warning]: "
        << "slice length should >= 1, but "
        << "n_slice is: " << n_slice << ", w_slice is: " << w_slice << "\n";);
      return false;
    }

    tensor->set_nw_slice(n_idx, n_slice, w_idx, w_slice);
    tensor->set_postfix(group_id_, n_loop, w_loop);

    if (cur_w_slice == -1) {
      branches.push_back(back_tensors[i]);
    }
  }

  return true;
}

// According to the slicing number and index, update each tensor's
// information in current group by breadth-first search algorithm.
bmerr_t Group::update_nh_slices(int nsecs, int hsecs, int nslice_idx, int hslice_idx) {
  std::vector<int> out_tensors = get_group_out_tensors();

  reset_tensor_slice();

  for (auto tid : out_tensors) {
    Tensor* tensor = net_graph_->get_tensor_by_id(tid);
    int batch_num = tensor->n();
    int n_step = ceiling_func(batch_num, nsecs);
    int height = tensor->h();
    int h_step = ceiling_func(height, hsecs);
    tensor->set_postfix(group_id_, nslice_idx, hslice_idx);

    if (nslice_idx == -1) {
      tensor->set_nh_slice(0, n_step, 0, h_step);
    } else {
      int n_idx = n_step * nslice_idx;
      int n_slice = (batch_num - n_idx) > n_step ? n_step : (batch_num - n_idx);
      int h_idx = h_step * hslice_idx;
      int h_slice = (height - h_idx) > h_step ? h_step : (height - h_idx);
      tensor->set_nh_slice(n_idx, n_slice, h_idx, h_slice);
    }

    std::list<int> branches;
    branches.push_back(tid);
    // breadth-first search algorithm
    while (!branches.empty()) {
      int tensor_id = branches.front();
      branches.pop_front();
      bool success = backward_nh_slice(tensor_id, branches, nslice_idx == -1,
                                    hsecs == 1, nslice_idx, hslice_idx);
      if (!success) {
        return BM_ERR_FAILURE;
      }
    }
  }

  if (validate_nh_slice() == false) {
    return BM_ERR_FAILURE;
  }

  return BM_SUCCESS;
}

// According to the slicing number and index, update each tensor's
// information in current group by breadth-first search algorithm.
bmerr_t Group::update_nw_slices(int nsecs, int wsecs, int nslice_idx, int wslice_idx) {
  std::vector<int> out_tensors = get_group_out_tensors();

  reset_tensor_slice();

  for (auto tid : out_tensors) {
    Tensor* tensor = net_graph_->get_tensor_by_id(tid);
    int batch_num = tensor->n();
    int n_step = ceiling_func(batch_num, nsecs);
    int width = tensor->w();
    int w_step = ceiling_func(width, wsecs);
    tensor->set_postfix(group_id_, nslice_idx, wslice_idx);

    if (nslice_idx == -1) {
      tensor->set_nw_slice(0, n_step, 0, w_step);
    } else {
      int n_idx = n_step * nslice_idx;
      int n_slice = (batch_num - n_idx) > n_step ? n_step : (batch_num - n_idx);
      int w_idx = w_step * wslice_idx;
      int w_slice = (width - w_idx) > w_step ? w_step : (width - w_idx);
      tensor->set_nw_slice(n_idx, n_slice, w_idx, w_slice);
    }

    std::list<int> branches;
    branches.push_back(tid);
    // breadth-first search algorithm
    while (!branches.empty()) {
      int tensor_id = branches.front();
      branches.pop_front();
      bool success = backward_nw_slice(tensor_id, branches, nslice_idx == -1,
                                    wsecs == 1, nslice_idx, wslice_idx);
      if (!success) {
        return BM_ERR_FAILURE;
      }
    }
  }

  if (validate_nw_slice() == false) {
    return BM_ERR_FAILURE;
  }

  return BM_SUCCESS;
}

bmerr_t Group::update_slices(int nsecs, int hwsecs, int nslice_idx, int hwslice_idx) {
  if (slice_dim_ == LG_Slice_Dim_H) {
    return update_nh_slices(nsecs, hwsecs, nslice_idx, hwslice_idx);
  } else if (slice_dim_ == LG_Slice_Dim_W) {
    return update_nw_slices(nsecs, hwsecs, nslice_idx, hwslice_idx);
  } else {
    LLVM_DEBUG(llvm::errs() << "Not support slice dim for layer group.\n";);
    return BM_ERR_FAILURE;
  }
  return BM_ERR_FAILURE;
}

void Group::show_group() {
  LLVM_DEBUG(llvm::errs()
    <<  "<n_slice: " << group_slice_.first
    << " , h_slice: " << group_slice_.second << " >" << "\n";);
  for (uint i = 0; i < layers_.size(); i++) {
    LLVM_DEBUG(llvm::errs() << " " << layers_[i];);
  }
  LLVM_DEBUG(llvm::errs() <<  "\n";);
}

void Group::print(std::ostream& pOs) const {
  pOs << "==============================================\n";
  int n_sec = group_slice_.first;
  int h_sec = group_slice_.second;
  if (slice_dim_ == LG_Slice_Dim_H)
    pOs << "(NSec, HSec) = (" << n_sec << ", " << h_sec << ")\n";
  else
    pOs << "(NSec, WSec) = (" << n_sec << ", " << h_sec << ")\n";
  pOs << "layer number: " << layers().size() << "\n";
  pOs << "layers: ";
  for (auto layer_id : layers()) {
    const ImLayer* im_layer = net_graph_->get_layer_by_id(layer_id);
    pOs << im_layer->type_name() << "(" << im_layer->name() << "), ";
  }
  pOs << "\n";
  time_step->show_timestep(pOs);
  time_step->show_mem_buffer(pOs);
  time_step->show_tsm_buffer(pOs);
}

void Group::clear_temp_data() {
  for (int id : layers_) {
    ImLayer* layer = const_cast<ImLayer*>(net_graph_->get_layer_by_id(id));
    layer->clear_temp_data();
  }
}
}
