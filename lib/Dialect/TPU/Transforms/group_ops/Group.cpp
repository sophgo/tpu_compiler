#include "Group.hpp"
#include "LayerStage.hpp"
#include "LMemManager.hpp"
#include "Steps.hpp"

#define DEBUG_TYPE "group"

namespace mlir {

namespace cl = llvm::cl;
// cl::opt<int> LayerGroupWithTSM("layer-group-with-tsm", cl::desc("Specify TSM Size in KB"),
//                                cl::value_desc("size"), cl::init(0), cl::cat(CatBM188X), cl::ReallyHidden);

// cl::opt<bool> OptIgnoreBankConflict("ignore-bank-conflict", cl::init(false), cl::cat(CatOptimizer));

Group::~Group() {
  if (time_step) {
    delete time_step;
  }
}

// to check if a tensor points to out of group.
bool Group::is_group_out_tensor(int tid) {
  const vector<int>& to_layers = net_graph_->get_tensor_to_layer(tid);
  if (to_layers.empty()) {
    return true;
  }

  for (int i = 0; i < to_layers.size(); i++) {
    int id = to_layers[i];

    if (find(layers_.begin(), layers_.end(), id) == layers_.end()) {
      return true;
    }
  }
  return false;
}

// Check if a tensor belongs to neuron tensor.
bool Group::is_group_in_neuron_tensor(int tid) {
  set<int> in_neurons = get_group_in_neuron_tensors();
  return in_neurons.find(tid) != in_neurons.end();
}

// Get out tensor not in the group
vector<int> Group::get_group_out_tensors() {
  vector<int> group_out_tensors;

  for (int i = 0; i < static_cast<int>(layers_.size()); i++) {
    int id = layers_[i];
    const vector<int>& out_tensors = net_graph_->get_out_tensors_of_layer(id);

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
set<int> Group::get_group_in_neuron_tensors() {
  set<int> group_in_neuron_tensors;

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

// Check if there is winograd tensor in group.
bool Group::group_has_winograd_tensors() {
  for (int i = 0; i < static_cast<int>(layers_.size()); i++) {
    int id = layers_[i];
    const vector<int>& out_tensors = net_graph_->get_out_tensors_of_layer(id);
    for (int j = 0; j < static_cast<int>(out_tensors.size()); ++j) {
      int tid = out_tensors[j];
      if (net_graph_->get_tensor_type(tid) == TENSOR_NEURON_WINOGRAD) {
        return true;
      }
    }
  }
  return false;
}

// Check out winograd tensor.
bmerr_t Group::group_winograd_out_tensors_check() {
  int nsecs = nsecs_and_hsecs.first;
  int hsecs = nsecs_and_hsecs.second;
  for (int hslice_idx = 0; hslice_idx < hsecs; hslice_idx++) {
    bmerr_t status = update_tensor_slices(nsecs, hsecs, 0, hslice_idx);
    if (status == BM_ERR_FAILURE) {
      return BM_ERR_FAILURE;
    }
    for (int i = 0; i < static_cast<int>(layers_.size()); i++) {
      int id = layers_[i];
      const vector<int>& out_tensors = net_graph_->get_out_tensors_of_layer(id);
      for (int j = 0; j < static_cast<int>(out_tensors.size()); ++j) {
        int tid = out_tensors[j];
        if (net_graph_->get_tensor_type(tid) == TENSOR_NEURON_WINOGRAD) {
          const Tensor* out_tensor = net_graph_->get_tensor_by_id(tid);
          int res_h = out_tensor->h_slice;
          int res_w = net_graph_->get_tensor_width(tid);
          if (std::ceil((1.0 * res_h / 2)) * std::ceil((1.0 * res_w / 2)) >= 4096) {
            llvm::errs() << "invalid winograd output res_h=" << res_h << ",res_w=" << res_w
                             << "\n";
            return BM_ERR_NOT_SUPPORTED;
          }
        }
      }
    }
  }
  return BM_SUCCESS;
}

bool Group::check_valid() {
  // return false;
  bmerr_t status = assign_steps();
  if (status != BM_SUCCESS) {
    LLVM_DEBUG(llvm::errs() << "layer group invalid: ";);
    for (int i = 0; i < layers_.size(); i++)
      LLVM_DEBUG(llvm::errs() << " " << layers_[i];);
    LLVM_DEBUG(llvm::errs() << "\n";);
    return false;
  }

  LLVM_DEBUG(llvm::errs() << "valid layer group: ";);
  for (int i = 0; i < layers_.size(); i++)
    LLVM_DEBUG(llvm::errs() << " " << layers_[i];);
  LLVM_DEBUG(llvm::errs() << "\n";);

  return true;
}

int Group::get_batch_num() const {
  int any_layer = layers_[0];
  int any_tensor = net_graph_->get_in_tensors_of_layer(any_layer)[0];
  return net_graph_->get_tensor_nums(any_tensor);
}

static int get_max_hsecs(NetGraph* net_graph_, const vector<int>& layer_group) {
  int max_hsecs;
  int any_layer = layer_group[0];
  int any_tensor = net_graph_->get_in_tensors_of_layer(any_layer)[0];
  // the temporary code for the large input tensor, such as yolo-608
  // TODO: to get better h split, balance gdma cost and abundant computation

  // when FC group with FC, don't split for now.
  if (net_graph_->get_tensor_height(any_tensor) == 0) {
    max_hsecs = 1;
    return max_hsecs;
  }
  if (net_graph_->get_tensor_height(any_tensor) >= 600 ||
      net_graph_->get_tensor_width(any_tensor) >= 600) {
    max_hsecs = 16;
  } else if (net_graph_->get_tensor_height(any_tensor) >= 400 ||
             net_graph_->get_tensor_width(any_tensor) >= 400) {
    max_hsecs = 8;
  } else if (net_graph_->get_tensor_height(any_tensor) >= 250 ||
             net_graph_->get_tensor_width(any_tensor) >= 250) {
    max_hsecs = 4;
  } else {
    max_hsecs = 8;
  }

  return max_hsecs;
}

// bmerr_t Group::assign_steps_with_tsm() {
//   // clear time_step and nescs_and_hsecs.
//   if (time_step) {
//     delete time_step;
//   }

//   nsecs_and_hsecs = {1, 1};
//   time_step = new net_timestep(net_graph_);

//   ClusterSteps::timestep_assign_with_tsm(net_graph_, this, time_step);

//   if (layers_.size() == 1) {
//     return BM_SUCCESS;
//   }

//   int batch_num = get_batch_num();
//   int max_hsecs = get_max_hsecs(net_graph_, layers_);

//   bmerr_t status;
//   {
//     if (BM_ERR_FAILURE == update_tensor_slices(batch_num, 1)) {
//       return BM_ERR_FAILURE;
//     }

//     time_step->update_mem_buffer_size();
//   }

//   status = BM_ERR_FAILURE;
//   while (nsecs_and_hsecs.first <= batch_num && nsecs_and_hsecs.second <= max_hsecs) {
//     // check validation of layer group if nsecs_and_hsecs.second > 1
//     if (nsecs_and_hsecs.second > 1) {
//       // to check first loop and last loop of h slices.
//       int nsecs = nsecs_and_hsecs.first;
//       int hsecs = nsecs_and_hsecs.second;
//       status = update_tensor_slices(nsecs, hsecs, 0, 0);
//       if (status == BM_ERR_FAILURE) {
//         return BM_ERR_FAILURE;
//       }

//       status = update_tensor_slices(nsecs, hsecs, 0, hsecs - 1);
//       if (status == BM_ERR_FAILURE) {
//         return BM_ERR_FAILURE;
//       }
//     }

//     net_timestep* tmp_timestep = new net_timestep(*time_step);

//     tmp_timestep->update_mem_buffer_size();

//     // Always assume one loop to reuse LMEM
//     // because we can reload tensor from TSM quickly
//     pair<int, int> one_loop_sec = {1, 1};
//     status = local_mem_allocate(net_graph_, layers_, tmp_timestep, one_loop_sec);
//     if (status != BM_ERR_FAILURE) {
//       bool one_loop = (nsecs_and_hsecs.first == 1) && (nsecs_and_hsecs.second == 1);
//       tmp_timestep->generate_tsm_buffer(one_loop);
//       status = tsm_allocate(net_graph_, layers_, tmp_timestep);
//     }

//     if (status == BM_ERR_FAILURE) {
//       delete tmp_timestep;
//       if (nsecs_and_hsecs.first < batch_num) {
//         nsecs_and_hsecs.first++;
//       } else {
//         nsecs_and_hsecs.second++;
//       }
//     } else {
//       delete time_step;
//       time_step = tmp_timestep;
//       break;
//     }
//   }

//   return status;
// }

bmerr_t Group::assign_steps() {
  return assign_steps_without_tsm();
}

// This function is used to construct the time step and find the appropriate partitioning
// strategy according to the time step.
bmerr_t Group::assign_steps_without_tsm() {
  // clear time_step and nescs_and_hsecs.
  if (time_step) {
    delete time_step;
  }

  nsecs_and_hsecs = {1, 1};
  time_step = new net_timestep(net_graph_);

  ClusterSteps::timestep_assgin(net_graph_, this, time_step);

  if (layers_.size() == 1) {
    return BM_SUCCESS;
  }

  int batch_num = get_batch_num();
  int max_hsecs = get_max_hsecs(net_graph_, layers_);

  reset_tensor_hslice_max();
  bmerr_t status = time_step->find_best_split(this, batch_num, nsecs_and_hsecs);
  if (status == BM_ERR_FAILURE) {
    return BM_ERR_FAILURE;
  }

  status = BM_ERR_FAILURE;

  if (!(nsecs_and_hsecs.first <= batch_num && nsecs_and_hsecs.second <= max_hsecs)) {
    LLVM_DEBUG(llvm::errs() << "FAIL: n_slice and h_slice exceed max value: ("
                            << nsecs_and_hsecs.first << "/" << batch_num
                            << ", " << nsecs_and_hsecs.second << "/" << max_hsecs << ")\n";);
  }
  while (nsecs_and_hsecs.first <= batch_num && nsecs_and_hsecs.second <= max_hsecs) {
    LLVM_DEBUG(llvm::errs() << "check n_slice and h_slice after split layer: ("
                            << nsecs_and_hsecs.first << "/" << batch_num
                            << ", " << nsecs_and_hsecs.second << "/" << max_hsecs << ")\n";);
    reset_tensor_hslice_max();
    if (group_has_winograd_tensors()) {
      status = group_winograd_out_tensors_check();
      if (status == BM_ERR_FAILURE) {
        return BM_ERR_FAILURE;
      } else if (status == BM_ERR_NOT_SUPPORTED) {
        nsecs_and_hsecs.second++;
        continue;
      }
    } else {
      // check validation of layer group if nsecs_and_hsecs.second > 1
      if (nsecs_and_hsecs.second > 1) {
        // to check first loop and last loop of h slices.
        int nsecs = nsecs_and_hsecs.first;
        int hsecs = nsecs_and_hsecs.second;
        LLVM_DEBUG(llvm::errs() << "check first loop of h slice.\n";);
        status = update_tensor_slices(nsecs, hsecs, 0, 0);
        if (status == BM_ERR_FAILURE) {
          llvm::errs() << "update tensor_slice fail.\n";
          return BM_ERR_FAILURE;
        }

        LLVM_DEBUG(llvm::errs() << "check last loop of h slice.\n";);
        status = update_tensor_slices(nsecs, hsecs, 0, hsecs - 1);
        if (status == BM_ERR_FAILURE) {
          llvm::errs() << "update tensor_slice fail....\n";
          return BM_ERR_FAILURE;
        }
      }
    }

    net_timestep* tmp_timestep = new net_timestep(*time_step);
    ClusterSteps::balance_gdma_bdc_steps(net_graph_, this, tmp_timestep, nsecs_and_hsecs);

    tmp_timestep->update_mem_buffer_size();

    if (1) {
      LmemManager lmem(net_graph_);
      bool one_shot = nsecs_and_hsecs.first == 1 && nsecs_and_hsecs.second == 1;
      status = lmem.assign_local_memory(this, tmp_timestep, one_shot);
    } else {
      assert(0);
      // status = local_mem_allocate(net_graph_, layers_, tmp_timestep, nsecs_and_hsecs);
    }

    if (status == BM_ERR_FAILURE) {
      delete tmp_timestep;
      if (nsecs_and_hsecs.first < batch_num) {
        nsecs_and_hsecs.first++;
      } else {
        nsecs_and_hsecs.second++;
      }
    } else {
      delete time_step;
      time_step = tmp_timestep;
      break;
    }
  }

  return status;
}

bool Group::validate_tensor_slice() {
  for (auto id : layers_) {
    const ImLayer* im_layer = net_graph_->get_layer_by_id(id);

    // if (im_layer->type() == IR_CONVOLUTION || im_layer->type() == IR_POOLING ||
    //     im_layer->type() == IR_DECONVOLUTION || im_layer->type() == IR_INNERPRODUCT) {
    //   continue;
    // }

    for (auto& tensor : im_layer->in_tensors) {
      if (tensor->type() == TENSOR_COEFF || tensor->type() == TENSOR_BIAS ||
          tensor->type() == TENSOR_COEFF_WINOGRAD || tensor->type() == TENSOR_DEPTHCONV_OPD1) {
        continue;
      }

      if (tensor->h_slice < 1) {
        llvm::errs() << "FAIL: h_slice of tensor[" << tensor->id() << "] = " << tensor->h_slice
                         << " is smaller than kh = 1" << "\n";
        return false;
      } else if (tensor->h_slice > tensor->h()) {
        llvm::errs() << "FAIL: h_slice " << tensor->h_slice << " of tensor[" << tensor->id()
                         << "] is larger than tensor height: " << tensor->h() <<  "\n";
        return false;
      }
    }
  }

  // Validate out tensor slice
  vector<int> out_tensors = get_group_out_tensors();
  for (auto tid : out_tensors) {
    Tensor* tensor = net_graph_->get_tensor_by_id(tid);

    if (tensor->h_idx >= tensor->h()) {
      llvm::errs() << "FAIL: h_idx of out tensor[" << tensor->id() << "] = " << tensor->h_idx
                       << " is larger than tensor height = " << tensor->h() << "\n";
      return false;
    }
  }

  return true;
}

void Group::reset_tensor_slice() {
  for (auto id : layers_) {
    const ImLayer* im_layer = net_graph_->get_layer_by_id(id);

    for (auto& tensor : im_layer->in_tensors) {
      net_graph_->set_tensor_num_height_slice(tensor->id(), -1, -1, -1, -1, 0, 0);
    }

    for (auto& tensor : im_layer->out_tensors) {
      net_graph_->set_tensor_num_height_slice(tensor->id(), -1, -1, -1, -1, 0, 0);
    }
  }
}

void Group::reset_tensor_hslice_max() {
  for (auto id : layers_) {
    const ImLayer* im_layer = net_graph_->get_layer_by_id(id);

    for (auto& tensor : im_layer->in_tensors) {
      net_graph_->set_tensor_height_slice_max(tensor->id(), -1);
    }

    for (auto& tensor : im_layer->out_tensors) {
      net_graph_->set_tensor_height_slice_max(tensor->id(), -1);
    }
  }
}

// Breadth-first traversal of all the tensors and set n_slice and h_slice.
bool Group::backward_slice(int out_tensor_id, list<int>& branches, bool max_h_slice,
                           bool no_split_h, int n_loop, int h_loop) {
  int id = net_graph_->get_tensor_from_layer(out_tensor_id);

  // the out tensor is the input tensor of the group
  if (id < 0 || find(layers_.begin(), layers_.end(), id) == layers_.end()) {
    return true;
  }

  const ImLayer* im_layer = net_graph_->get_layer_by_id(id);
  IR_TYPE layer_type = im_layer->type();
  int kh, sh, ph;
  int dh = 1;

  if (layer_type == IR_CONVOLUTION || layer_type == IR_DECONVOLUTION) {
    auto op = cast<tpu::Conv2DOp>(im_layer->op());
    bool is_dw, with_bias, do_relu;
    int n, ic, ih, iw, oc, oh, ow, g, kw, sw, pw, dw;
    bool is_deconv = isa<tpu::DeConv2DOp>(im_layer->op());
    parseConvParam(op.param(), is_deconv, op.input(), op.output(), op.filter(),
                   n, ic, ih, iw, oc, oh, ow, g,
                   kh, kw, sh, sw, ph, pw, dh, dw, is_dw, with_bias, do_relu);
    if (dh > 1) {
      kh = dh * (kh - 1) + 1;
    }
  } else if (layer_type == IR_POOLING) {
    if (isa<tpu::PoolAvg2DOp>(im_layer->op())) {
      auto op = cast<tpu::PoolAvg2DOp>(im_layer->op());
      bool is_global, do_relu;
      int n, c, ih, iw, oh, ow, kw, sw, pb, pl, pr;
      parsePoolParam(op.param(), op.input(), op.output(),
                    n, c, ih, iw, oh, ow,
                    kh, kw, sh, sw, ph, pb, pl, pr,
                    is_global, do_relu);
    } else if (isa<tpu::PoolMax2DOp>(im_layer->op())) {
      auto op = cast<tpu::PoolMax2DOp>(im_layer->op());
      bool is_global, do_relu;
      int n, c, ih, iw, oh, ow, kw, sw, pb, pl, pr;
      parsePoolParam(op.param(), op.input(), op.output(),
                    n, c, ih, iw, oh, ow,
                    kh, kw, sh, sw, ph, pb, pl, pr,
                    is_global, do_relu);
    }
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
  const vector<int>& back_tensors = net_graph_->get_in_tensors_of_layer(id);

  for (u32 i = 0; i < back_tensors.size(); ++i) {
    Tensor* tensor = net_graph_->get_tensor_by_id(back_tensors[i]);

    if (tensor->type() == TENSOR_COEFF || tensor->type() == TENSOR_BIAS ||
        tensor->type() == TENSOR_COEFF_WINOGRAD || tensor->type() == TENSOR_DEPTHCONV_OPD1) {
      continue;
    }

    int cur_h_idx = tensor->h_idx;
    int cur_h_slice = tensor->h_slice;

    if (no_split_h) {
      h_idx = 0;
      h_slice = tensor->h();
    } else if (layer_type == IR_CONVOLUTION || layer_type == IR_POOLING) {
      h_idx = out_h_idx * sh - ph;
      h_slice = (out_h_slice - 1) * sh + kh;
      if (out_tensor->type() == TENSOR_NEURON_WINOGRAD) {
        if (out_h_slice % 2 == 1) {
          if ((h_idx + ph) % 2 == 1) {
            h_idx -= 1;
            h_slice += 1;
            out_tensor->set_h_slice_skip_first();
          } else {
            h_slice += 1;
            out_tensor->set_h_slice_skip_last();
          }
        }
      }
    } else if (layer_type == IR_DECONVOLUTION) {
      int bottom_h = tensor->h();
      int height_insert0 = (bottom_h - 1) * sh + 1;
      int real_o_h_t = out_h_idx;
      int real_o_h_b = out_h_idx + out_h_slice;
      int kh_ext = (kh - 1) * dh + 1;
      ph = kh_ext - ph - 1;
      int if_pad_h_t = real_o_h_t;
      int if_pad_h_b = real_o_h_b + kh_ext - 1;
      int if_insert_h_t = 0;
      if (if_pad_h_t >= ph) {
        if_insert_h_t = if_pad_h_t - ph;
      }
      int if_insert_h_b = height_insert0;
      if ((if_pad_h_b - ph) < height_insert0) {
        if_insert_h_b = if_pad_h_b - ph;
      }
      h_idx = (if_insert_h_t + sh - 1) / sh;
      h_slice = (if_insert_h_b + sh - 1) / sh - h_idx;
    } else if (layer_type == IR_UPSAMPLE) {
      auto op = cast<tpu::UpsampleOp>(im_layer->op());
      int size = op.scale().getLimitedValue();

      if (out_h_slice % size) {
        llvm::errs() << "FAIL: fractional upsample input h slice" << "\n";
        return false;
      }

      h_idx = out_h_idx / size;
      h_slice = out_h_slice / size;
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


    LLVM_DEBUG(llvm::errs() << "tensor_id: " << tensor->id() << " n_idx: "
                            << n_idx << " h_idx: " << h_idx
                            << ", n_slice: " << n_slice << ", h_slice: " << h_slice
                            << "out_h_idx: " << out_h_idx << " out_h_slice: " << out_h_slice
                            << " ph: " << ph << " sh: " << sh << " kh: " << kh << "\n";);


    if (cur_h_slice != -1 && (cur_h_slice != h_slice || cur_h_idx != h_idx)) {
      llvm::errs() << "FAIL: data slice in h dimension is conflicted for tensor "
                       << back_tensors[i] << " cur_h_idx:" << cur_h_idx << " h_idx:" << h_idx
                       << " cur_h_slice:" << cur_h_slice << " h_slice:" << h_slice << "\n";
      return false;
    }

    if (n_slice < 1 || h_slice < 1) {
      llvm::errs() << "slice is smaller than than the minimum"
                       << ", n_slice: " << n_slice << ", h_slice: " << h_slice << "\n";
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

// According to the slicing number and index, update each tensor's
// information in current group by breadth-first search algorithm.
bmerr_t Group::update_tensor_slices(int nsecs, int hsecs, int nslice_idx, int hslice_idx) {
  vector<int> out_tensors = get_group_out_tensors();

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

    list<int> branches;
    branches.push_back(tid);

    // breadth-first search algorithm
    while (!branches.empty()) {
      int tensor_id = branches.front();
      branches.pop_front();

      bool success = backward_slice(tensor_id, branches, nslice_idx == -1,
                                    hsecs == 1, nslice_idx, hslice_idx);
      if (!success) {
        return BM_ERR_FAILURE;
      }
    }
  }

  if (validate_tensor_slice() == false) {
    return BM_ERR_FAILURE;
  }

  return BM_SUCCESS;
}

void Group::show_group() {
  llvm::errs() <<  "<n_slice: " << nsecs_and_hsecs.first
                        << " , h_slice: " << nsecs_and_hsecs.second << " >" << "\n";
  for (uint i = 0; i < layers_.size(); i++) {
    llvm::errs() << " " << layers_[i];
  }
  llvm::errs() <<  "\n";
}

void Group::print(std::ostream& pOs) const {
  pOs << "==============================================\n";
  int n_sec = nsecs_and_hsecs.first;
  int h_sec = nsecs_and_hsecs.second;
  pOs << "(NSec, HSec) = (" << n_sec << ", " << h_sec << ")\n";
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
  llvm::errs() << "clear temp data, layers_ size: " << layers_.size() <<
                  " Imlayer size:" << net_graph_->getImLayerSize() << "\n";
  for (int id : layers_) {
    ImLayer* layer = const_cast<ImLayer*>(net_graph_->get_layer_by_id(id));
    layer->clear_temp_data();
  }
}
}
