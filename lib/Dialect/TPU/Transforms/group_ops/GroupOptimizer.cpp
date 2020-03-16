#include "GroupOptimizer.hpp"
#include "llvm/Support/ToolOutputFile.h"
namespace mlir {

static llvm::cl::opt<std::string> clNeuronMapFilename(
    "layer-group-neuron-map-filename",
    llvm::cl::desc("record neuron offset with its name into a csv map file"),
    llvm::cl::init("-"));

static llvm::cl::opt<bool> clDisableGMemOptimize(
    "layer-group-gm-opt",
    llvm::cl::desc("Enable global memory optimzation for layer group, default enable"),
    llvm::cl::init(true));


GroupOptimizer::GroupOptimizer(NetGraph* net_graph)
    : net_graph_(net_graph)
     , mix_net_(net_graph), gmem_mgr_(net_graph)
    {}

GroupOptimizer::~GroupOptimizer() {
  for (auto group : groups_) {
    delete group;
  }
}

// This function is used to group all layers. In a group, the top layer can directly
// use the results of the bottom layer without going through the store/load process,
// which reduces GDMA operations.
void GroupOptimizer::do_group(vector<Group*>& Groups) {
  Group* rough = new Group(net_graph_);

  for (auto layer = ImLayer::layers.begin(); layer != ImLayer::layers.end(); ++layer) {
    int id = (*layer)->id();

    if (!(*layer)->fusible) {
      // let these layers to be a single layer group.
      if (!rough->empty()) {
        add_valid_custers(Groups, rough);
        delete rough;
        rough = new Group(net_graph_);
      }

      rough->append(id);
      add_valid_custers(Groups, rough);
      delete rough;
      rough = new Group(net_graph_);
    } else {
      rough->append(id);
    }
  }

  if (!rough->empty()) {
    add_valid_custers(Groups, rough);
  }

  for (auto Group : groups_) {
    Group->clear_temp_data();
  }

  delete rough;
}

// This fucntion does two things:
//   1. Use the binary search to group all layers;
//   2. Adjust the position before and after grouping to minimize global memory.
void GroupOptimizer::add_valid_custers(vector<Group*>& groups, Group* target) {
  int num = target->size();
  int start = 0;
  const int end = num - 1;
  int cut = end, right = end;
  int left = start, valid = start;
  vector<int> cut_points;

  while (start < num) {
    if (cut == left) {
      cut_points.push_back(valid);
      start = valid + 1;
      cut = end;
      valid = start;
      left = start;
      right = end;
      continue;
    }

    auto group =
        std::make_shared<Group>(net_graph_, target->begin() + start, target->begin() + cut + 1);

    if (group->check_valid()) {
      valid = cut;
      left = cut;
    } else {
      right = cut;
    }

    cut = (left + right) / 2;
  }

  cut_points = optimize_cut_points(target, cut_points);

  start = 0;
  for (int i = 0; i < static_cast<int>(cut_points.size()); i++) {
    int end = cut_points[i];
    auto* group = new Group(net_graph_, target->begin() + start, target->begin() + end + 1);
    groups.push_back(group);
    start = end + 1;
  }
}

// After getting the cut node of the whole network, we can adjust the position of the node
// before and after, so that the size of the global memory used is the smallest.
vector<int> GroupOptimizer::optimize_cut_points(Group* target, const vector<int>& cut_points) {
  vector<int> best_solution(cut_points);
  int minimum_ddr_occupied = calc_group_out_tensors_size(target, cut_points);

  if (cut_points.size() == 1) {
    return best_solution;
  }

  int moving_idx = cut_points.size() - 2;
  while (moving_idx >= 0) {
    vector<vector<int>> solutions;
    vector<int> solution(best_solution);

    int left_point = (moving_idx == 0) ? 0 : solution[moving_idx - 1] + 1;

    // move cut_point to left step by step and
    // check if sub group of two side are all valid.
    for (solution[moving_idx]--; solution[moving_idx] >= left_point; solution[moving_idx]--) {
      // check if sub group of right side is valid.
      Group* group = new Group(net_graph_, target->begin() + solution[moving_idx] + 1,
                                 target->begin() + solution[moving_idx + 1] + 1);

      if (!group->check_valid()) {
        delete group;
        break;
      }

      // check left side
      delete group;
      group = new Group(net_graph_, target->begin() + left_point,
                          target->begin() + solution[moving_idx] + 1);

      // push all options to set.
      if (group->check_valid()) {
        solutions.push_back(solution);
      }
      delete group;
    }

    // update min ddr size and cut points
    for (auto iter = solutions.begin(); iter != solutions.end(); ++iter) {
      int group_out_data_size = calc_group_out_tensors_size(target, *iter);

      if (group_out_data_size < minimum_ddr_occupied) {
        minimum_ddr_occupied = group_out_data_size;
        best_solution = *iter;
      }
    }

    solutions.clear();
    moving_idx--;
  }

  return best_solution;
}

// return the output data size of layer group, unit: word
int GroupOptimizer::calc_group_out_tensors_size(Group* target, const vector<int>& cut_points) {
  vector<Group> Groups;

  int start = 0;
  for (int i = 0; i < static_cast<int>(cut_points.size()); i++) {
    int end = cut_points[i];
    Group Group(net_graph_, target->begin() + start, target->begin() + end + 1);
    Groups.push_back(Group);
    start = end + 1;
  }

  // get total out data size
  int total_data_size = 0;

  for (auto Group : Groups) {
    vector<int> out_tensors = Group.get_group_out_tensors();

    for (int j = 0; j < static_cast<u32>(out_tensors.size()); ++j) {
      Tensor* tensor = net_graph_->get_tensor_by_id(out_tensors[j]);
      total_data_size += tensor->gmem_size();
    }
  }
  return total_data_size;
}

bmerr_t GroupOptimizer::optimize() {

  do_group(groups_);

  for (auto Group : groups_) {
    bmerr_t status = Group->assign_steps();
    if (status == BM_ERR_FAILURE) {
      cout << "local memory allocate failed" << endl;
      assert(0);
    }

    if (Group->size() == 1) {
      int id = Group->layers()[0];
      ImLayer* layer = const_cast<ImLayer*>(net_graph_->get_layer_by_id(id));
      layer->is_tg_layer = true;
    }
  }

  string DumpLayerGroupInfo = "layer_optimizer.out";
  if (DumpLayerGroupInfo != "") {
    std::fstream f;
    f.open(DumpLayerGroupInfo, std::ios::out);
    int group_id = 0;
    for (auto* Group : groups_) {
      f << "Group[" << group_id << "]\n";
      Group->print(f);
      group_id++;
    }
    f.close();
  }

  return BM_SUCCESS;
}


void GroupOptimizer::set_input_output_tensor() {
  for (auto& layer : ImLayer::layers) {
    IR_TYPE type = layer->type();

    for (auto& tensor : layer->in_tensors) {
      int tid = tensor->id();
      if (net_graph_->get_tensor_type(tid) == TENSOR_NEURON ||
          net_graph_->get_tensor_type(tid) == TENSOR_NEURON_WINOGRAD) {
        int from_layer = net_graph_->get_tensor_from_layer(tid);
        if (from_layer == -1) {
          mix_net_.set_net_in_tensor(tid);
          llvm::errs() << "Input tensor: " << tid << "\n";
        }
      }
    }

    for (auto& tensor : layer->out_tensors) {
      int tid = tensor->id();
      const vector<int>& to_layers = net_graph_->get_tensor_to_layer(tid);
      if (to_layers.empty() && type != IR_MULTIINPUT) {
        mix_net_.set_net_out_tensor(tid);
        llvm::errs() << "Output tensor: " << tid << "\n";
      }
    }
  }
}

// static std::ofstream layer_id_name_mapping_file_fp;
// void _add_layer_id_name_mapping_str(const LayerParameter* last_layer, std::string name = "") {
//   layer_id_name_mapping_file_fp <<
//     last_layer->id() << "," <<
//     (name == "" ? last_layer->name() : name) << "," <<
//     last_layer->type() <<
//     "\n";
// }


// void GroupOptimizer::Group2Graph(int gid, Group* Group) {
//   int n_secs = Group->nsecs_and_hsecs.first;
//   int h_secs = Group->nsecs_and_hsecs.second;

//   static const NetParameter* out_net_ = mix_net_.get_net();
//   mix_net_.add_group_start_layer(gid, Group, n_secs, h_secs);
//   if (!layer_id_name_mapping_file.empty()) {
//     const LayerParameter last_layer = out_net_->layer(out_net_->layer_size() - 1);
//     std::string layer_name = std::string("group ") + std::to_string(gid) + std::string(" start");
//     _add_layer_id_name_mapping_str(&last_layer, layer_name);
//   }

//   bool first_loop = true;
//   for (int n_loop = 0; n_loop < n_secs; n_loop++) {
//     for (int h_loop = 0; h_loop < h_secs; h_loop++) {

//       Group->update_tensor_slices(n_secs, h_secs, n_loop, h_loop);

//       for (int step_id = 0; step_id < Group->time_step->get_timestep_num(); ++step_id) {
//         int cur_layer = Group->time_step->get_layer(step_id);
//         if (cur_layer != -1) {
//           mix_net_.add_tl_layer(
//               gid, cur_layer, Group->time_step, step_id,
//               h_secs != 1, n_loop, h_loop);
//           if (!layer_id_name_mapping_file.empty()) {
//             const LayerParameter last_layer = out_net_->layer(out_net_->layer_size() - 1);
//             _add_layer_id_name_mapping_str(&last_layer);
//           }
//         }

//         const vector<TENSOR_STEP>& cur_tensors = Group->time_step->get_tensors(step_id);
//         for (u32 i = 0; i < cur_tensors.size(); ++i) {
//           // check if tensor is kept in lmem.
//           if ((!first_loop) && cur_tensors[i].second == TIMESTEP_LOAD &&
//               Group->time_step->is_tensor_hold_in_memory(cur_tensors[i].first)) {
//             continue;
//           }

//           // if tensor is loaded in tsm and holded, don't load it again.
//           if ((!first_loop) && cur_tensors[i].second == TIMESTEP_DDR_TO_TSM &&
//               Group->time_step->is_tensor_hold_in_memory(cur_tensors[i].first)) {
//             continue;
//           }

//           if (cur_layer == -1 && step_id == 0) {
//             mix_net_.add_transport_param_to_next_layer(
//                 cur_tensors[i], Group->time_step, step_id, false);
//           } else {
//             mix_net_.add_transport_param_to_last_layer(
//                 cur_tensors[i], Group->time_step,
//                 step_id, cur_layer != -1);
//           }
//         }
//       }

//       first_loop = false;
//     }
//   }

//   mix_net_.add_group_end_layer(gid, Group, n_secs, h_secs);

//   if (!layer_id_name_mapping_file.empty()) {
//     const LayerParameter last_layer = out_net_->layer(out_net_->layer_size() - 1);
//     std::string layer_name = std::string("group ") + std::to_string(gid) + std::string(" end");
//     _add_layer_id_name_mapping_str(&last_layer, layer_name);
//   }
// }

bool GroupOptimizer::is_tg_op(Operation * op) {
  llvm::errs() << getOpName(op);
  for (auto group : groups_) {
    if (group->size() == 1) {
      int layer_id = group->layers()[0];
      const ImLayer * im_layer = net_graph_->get_layer_by_id(layer_id);
      if (getOpName(im_layer->op()) == getOpName(op)) {
        llvm::errs() << "   is TG layer.\n";
        return true;
      }
    }
  }
  llvm::errs() << "   is not TG layer.\n";
  return false;
}

// set global address for current inst input and output
uint64_t GroupOptimizer::setOpGAddr(Operation * op) {
  const ImLayer * layer = net_graph_->get_layer_by_op(op);
  llvm::errs() << "OP: " << getOpName(op) << "\n";
  for (int i = 0; i < op->getNumResults(); i++) {
    for (auto& tensor : layer->out_tensors) {
      if (tensor->type() != TENSOR_NEURON &&
        tensor->type() != TENSOR_NEURON_AS_COEFF &&
        tensor->type() != TENSOR_NEURON_WINOGRAD &&
        tensor->type() != TENSOR_MATRIX)
          continue;
      if (tensor->name() == top_name(op, i).str()) {
        Operation * top = op->getResult(i)->getDefiningOp();
        if (isa<tpu::ReshapeOp>(top))
          continue;
        setOpAddress(top, tensor->gaddr);
        llvm::errs() << "  set top:" << getOpName(top) << " to address: " << tensor->gaddr << "\n";
        assert(op->getNumResults() == 1);
        return tensor->gaddr;
      }
    }
  }
}

template<typename OpTy>
struct addGroupTGLayerPattern : public RewritePattern {
  addGroupTGLayerPattern(MLIRContext *context, GroupOptimizer * optimizer)
      : RewritePattern(OpTy::getOperationName(), 1, context) {opt_ = optimizer;}
  GroupOptimizer * opt_;

  PatternMatchResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {

    if (!isa<OpTy>(op))
      return matchFailure();

    if (!opt_->is_tg_op(op))
      return matchFailure();

    auto tpuOp = llvm::dyn_cast<tpu::TpuOpLowerInterface>(op);
    if (!tpuOp) {
      return matchFailure();
    }

    auto tg_op = tpuOp.convertToTG();
    if (!tg_op) {
       return matchFailure();
    }

    rewriter.replaceOp(op, {tg_op});

    return matchSuccess();
  }
};


template<typename OpTy>
struct addTGLayerGAddrPattern : public RewritePattern {
  addTGLayerGAddrPattern(MLIRContext *context, GroupOptimizer * optimizer,
                         llvm::raw_ostream &map_os)
      : RewritePattern(OpTy::getOperationName(), 1, context),
      opt_(optimizer),
      map_os_(map_os){}

  PatternMatchResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    auto cast_op = cast<OpTy>(op);
    if (cast_op.gaddr().hasValue()) {
      return matchFailure();
    }

    uint64_t offset = opt_->setOpGAddr(op);
    llvm::errs() << "handle layer: " << getOpName(op) << "\n";
    // generate neuron map
    auto type = op->getResult(0)->getType().cast<TensorType>();
    std::vector<int64_t> shape = type.getShape();
    string dtype = "int8";
    while (shape.size() < 4)
      shape.insert(shape.begin(), 1);
    map_os_ << getOpName(op) << "," << llvm::format_hex(offset, 10) << ","
            << dtype << ","
            << shape[0] << "," << shape[1] << ","
            << shape[2] << "," << shape[3] << "\n";
    return matchSuccess();
  }

  GroupOptimizer * opt_;
  llvm::raw_ostream &map_os_;
};

template<typename T>
static void transposeFullyConnectedFilter(std::vector<T> &w,
    std::vector<int64_t> &s) {
  assert(s.size() == 2);
  int row = s[0];
  int col = s[1];
  std::vector<T> w_t(w.size());
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      w_t[j * row + i] = w[i * col  + j];
    }
  }
  w.assign(w_t.begin(), w_t.end());
}

static void transposeBiasInt16(std::vector<int16_t> &w_int16) {
  int8_t *ptr = reinterpret_cast<int8_t *>(w_int16.data());
  std::vector<int8_t> w(ptr, ptr + w_int16.size() * sizeof(int16_t));
  std::vector<int8_t> w_t(w.size());
  for (size_t i = 0; i < w_int16.size(); i++) {
    for (size_t j = 0; j < 2; j++) {
      w_t[j * w_int16.size() + i] = w[i * 2 + j];
    }
  }
  memcpy(ptr, w_t.data(), w_t.size());
}

template<typename T>
static void transposeConvolutionFilter(std::vector<T> &w,
    std::vector<int64_t> &s) {
  int64_t oc, ic, ks;
  if (s.size() == 4) {
    oc = s[0];
    ic = s[1];
    ks = s[2] * s[3];
  } else if (s.size() == 5) {
    // g, oc/g, ic/g, kh, kw
    oc = s[0] * s[1];
    ic = s[2];
    ks = s[3] * s[4];
  } else {
    assert(false);
  }

  std::vector<T> w_t(w.size());
  if (ks == 1 || ic == 1) {
    return;
  } else {
    // for other conv, transpose ic <-> kh*kw
    for (int64_t i = 0; i < oc; i++) {
      for (int64_t j = 0; j < ic; j++) {
        for (int64_t k = 0; k < ks; k++) {
          w_t[i * ic * ks + k * ic + j] = w[i * ic * ks + j * ks + k];
        }
      }
    }
  }
  w.assign(w_t.begin(), w_t.end());
}

template <typename OpTy>
struct LowerConv2DOpWeightPattern : public RewritePattern {
  LowerConv2DOpWeightPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    auto convOp = cast<OpTy>(op);
    auto filterOp = cast<tpu::LoadWeightOp>(convOp.filter()->getDefiningOp());
    if (filterOp.lowered()) {
      // lowered already
      return matchFailure();
    }
    llvm::errs() << "Lower Weight for Conv2D: " << getOpName(op) << "\n";
    TensorFile *wTF = getWeightTensorFile(op);

    if (getOpQuant(op) == "INT8") {
      // lower filter
      {
        assert(filterOp.storage() == "INT8");
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(convOp.filter(), shape, size);
        auto filter = readAndDeleteWeightTensor<float>(convOp.filter(), wTF);
        std::vector<int8_t> filter_int8(filter->begin(), filter->end());
        // transpose ic <-> kh*kw
        // if kh*kw == 1 or ic/g == 1, transposeConvolutionFilter() will do nothing
        assert(shape.size() == 4 || shape.size() == 5);
        transposeConvolutionFilter<int8_t>(filter_int8, shape);

        // save it
        addWeightTensorAndUpdateWeightOp<int8_t>(convOp.filter(),
            "lowered", filter_int8, shape, "INT8", wTF);
        filterOp.setAttr("lowered", rewriter.getBoolAttr(true));
      }

      // lower bias
      if ( !isTensorNone(convOp.bias()) ) {
        auto biasOp = cast<tpu::LoadWeightOp>(convOp.bias()->getDefiningOp());
        if (isOpQuantPerchannel(op)
            && getOpQuantParamType(op) == "RSHIFT_AND_M_I32") {
          // lowered already, in pack
          assert(biasOp.lowered());
          assert(biasOp.storage() == "UINT8");
        } else if (isOpQuantPerchannel(op)) {
          // per-channel mode, bias is INT32
          assert(biasOp.storage() == "INT32");
          assert(false && "REMINDER: NOT sure if per-channel bias needs transpose");

          // TODO:

          // save it
          //StringRef storageType = "INT32";
          //addWeightTensorAndUpdateWeightOp<int32_t>(convOp.bias(),
          //    "lowered", bias_int16, shape, storageType, wTF);
          biasOp.setAttr("lowered", rewriter.getBoolAttr(true));
        } else {
          // per-tensor mode, bias is INT16
          assert(biasOp.storage() == "INT16");
          std::vector<int64_t> shape;
          int64_t size;
          getTensorShapeAndSize(convOp.bias(), shape, size);
          auto bias = readAndDeleteWeightTensor<float>(convOp.bias(), wTF);
          std::vector<int16_t> bias_int16(bias->begin(), bias->end());
          transposeBiasInt16(bias_int16);
          std::vector<uint16_t> bias_uint16(size);
          memcpy(bias_uint16.data(), bias_int16.data(), size * sizeof(int16_t));

          // save it
          // after transpose, this is not INT16 anymore, it is 2 stripes of UINT8
          // we save it as UINT16, to carry the eltment bitwidth, so we don`t need
          // to change the shape.
          addWeightTensorAndUpdateWeightOp<uint16_t>(convOp.bias(),
              "lowered", bias_uint16, shape, "UINT16", wTF);
          biasOp.setAttr("lowered", rewriter.getBoolAttr(true));
        }
      }
    } else if (getOpQuant(op) == "BF16") {
      // lower filter
      {
        assert(filterOp.storage() == "BF16");
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(convOp.filter(), shape, size);
        auto filter = readAndDeleteWeightTensor<bfloat16>(convOp.filter(), wTF);
        std::vector<uint16_t> filter_bf16(filter->begin(), filter->end());

        // transpose ic <-> kh*kw
        // if kh*kw == 1 or ic/g == 1, transposeConvolutionFilter() will do nothing
        assert(shape.size() == 4 || shape.size() == 5);
        transposeConvolutionFilter<uint16_t>(filter_bf16, shape);

        // save it
        StringRef storageType = "BF16";
        addWeightTensorAndUpdateWeightOp<uint16_t>(convOp.filter(),
            "lowered", filter_bf16, shape, storageType, wTF);
        filterOp.setAttr("lowered", rewriter.getBoolAttr(true));
      }

      // lower bias
      if ( !isTensorNone(convOp.bias()) ) {
        auto biasOp = cast<tpu::LoadWeightOp>(convOp.bias()->getDefiningOp());
        assert(biasOp.storage() == "BF16");
        // NOTE: for 1880v2, bias is fp32, rather than bf16
        // however, for simplicity, in quantizeBf16, we quantize all tensor into bf16
        // before lowering to hardware, we need to expand the bf16 to fp32 first
        // then transpose into 2 stripes of uint16_t
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(convOp.bias(), shape, size);
        auto bias = readAndDeleteWeightTensor<bfloat16>(convOp.bias(), wTF);
        std::vector<uint16_t> bias_bf16(bias->begin(), bias->end());
        // rather than expand to fp32, then transpose, we simply add a new stripe
        // of uint16_t with all 0x0000
        size_t sz = bias_bf16.size();
        for (size_t i = 0; i < sz; ++i) {
          bias_bf16.push_back(0x0000);
        }
        // then copy into uint32_t
        std::vector<uint32_t> bias_uint32(sz);
        memcpy(bias_uint32.data(), bias_bf16.data(), sz * sizeof(uint32_t));

        // save it
        // after expand to FB32 and transpose, this is not FB32 anymore
        // it is 2 stripes of UINT16(BF16)
        // we save it as UINT32, to carry the eltment bitwidth, so we don`t need
        // to change the shape
        StringRef storageType = "UINT32";
        addWeightTensorAndUpdateWeightOp<uint32_t>(convOp.bias(),
            "lowered", bias_uint32, shape, storageType, wTF);
        biasOp.setAttr("lowered", rewriter.getBoolAttr(true));
      }
    }

    return matchSuccess();
  }
};

struct LowerWeightFullyConnectedOpPattern : public RewritePattern {
  LowerWeightFullyConnectedOpPattern(MLIRContext *context)
      : RewritePattern("tpu.fully_connected", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    auto fcOp = cast<tpu::FullyConnectedOp>(op);
    auto filterOp = cast<tpu::LoadWeightOp>(fcOp.filter()->getDefiningOp());
    if (filterOp.lowered()) {
      // lowered already
      return matchFailure();
    }
    llvm::errs() << "Lower Weight for FullyConnectedOp: " << getOpName(op) << "\n";
    TensorFile *wTF = getWeightTensorFile(op);

    if (getOpQuant(op) == "INT8") {
      // lower filter
      {
        assert(filterOp.storage() == "INT8");
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(fcOp.filter(), shape, size);
        auto filter = readAndDeleteWeightTensor<float>(fcOp.filter(), wTF);
        std::vector<int8_t> filter_int8(filter->begin(), filter->end());
        // transpose k,n
        assert(shape.size() == 2);
        transposeFullyConnectedFilter<int8_t>(filter_int8, shape);

        // save it
        addWeightTensorAndUpdateWeightOp<int8_t>(fcOp.filter(),
            "lowered", filter_int8, shape, "INT8", wTF);
        filterOp.setAttr("lowered", rewriter.getBoolAttr(true));
      }

      // lower bias
      if ( !isTensorNone(fcOp.bias()) ) {
        auto biasOp = cast<tpu::LoadWeightOp>(fcOp.bias()->getDefiningOp());
        // per-tensor mode, bias is INT16
        assert(biasOp.storage() == "INT16");
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(fcOp.bias(), shape, size);
        auto bias = readAndDeleteWeightTensor<float>(fcOp.bias(), wTF);
        std::vector<int16_t> bias_int16(bias->begin(), bias->end());
        transposeBiasInt16(bias_int16);
        std::vector<uint16_t> bias_uint16(size);
        memcpy(bias_uint16.data(), bias_int16.data(), size * sizeof(int16_t));

        // save it
        // after transpose, this is not INT16 anymore, it is 2 stripes of UINT8
        // we save it as UINT16, to carry the eltment bitwidth, so we don`t need
        // to change the shape.
        addWeightTensorAndUpdateWeightOp<uint16_t>(fcOp.bias(),
            "lowered", bias_uint16, shape, "UINT16", wTF);
        biasOp.setAttr("lowered", rewriter.getBoolAttr(true));
      }
    } else if (getOpQuant(op) == "BF16") {
      // lower filter
      {
        assert(filterOp.storage() == "BF16");
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(fcOp.filter(), shape, size);
        auto filter = readAndDeleteWeightTensor<bfloat16>(fcOp.filter(), wTF);
        std::vector<uint16_t> filter_bf16(filter->begin(), filter->end());
        // transpose h,n
        assert(shape.size() == 2);
        transposeFullyConnectedFilter<uint16_t>(filter_bf16, shape);

        // save it
        StringRef storageType = "BF16";
        addWeightTensorAndUpdateWeightOp<uint16_t>(fcOp.filter(),
            "lowered", filter_bf16, shape, storageType, wTF);
        filterOp.setAttr("lowered", rewriter.getBoolAttr(true));
      }

      // lower bias
      if ( !isTensorNone(fcOp.bias()) ) {
        auto biasOp = cast<tpu::LoadWeightOp>(fcOp.bias()->getDefiningOp());
        assert(biasOp.storage() == "BF16");
        // NOTE: for 1880v2, bias is fp32, rather than bf16
        // however, for simplicity, in quantizeBf16, we quantize all tensor into bf16
        // before lowering to hardware, we need to expand the bf16 to fp32 first
        // then transpose into 2 stripes of uint16_t
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(fcOp.bias(), shape, size);
        auto bias = readAndDeleteWeightTensor<bfloat16>(fcOp.bias(), wTF);
        std::vector<uint16_t> bias_bf16(bias->begin(), bias->end());
        // rather than expand to fp32, then transpose, we simply add a new stripe
        // of uint16_t with all 0x0000
        size_t sz = bias_bf16.size();
        for (size_t i = 0; i < sz; ++i) {
          bias_bf16.push_back(0x0000);
        }
        // then copy into uint32_t
        std::vector<uint32_t> bias_uint32(sz);
        memcpy(bias_uint32.data(), bias_bf16.data(), sz * sizeof(uint32_t));

        // save it
        // after expand to FB32 and transpose, this is not FB32 anymore
        // it is 2 stripes of UINT16(BF16)
        // we save it as UINT32, to carry the eltment bitwidth, so we don`t need
        // to change the shape
        StringRef storageType = "UINT32";
        addWeightTensorAndUpdateWeightOp<uint32_t>(fcOp.bias(),
            "lowered", bias_uint32, shape, storageType, wTF);
        biasOp.setAttr("lowered", rewriter.getBoolAttr(true));
      }
    }

    return matchSuccess();
  }
};

static std::unique_ptr<std::vector<uint8_t> > packWeight(
    std::vector<float> *bias, std::vector<float> *rshift,
    std::vector<float> *multiplier, int64_t oc,
    std::vector<int64_t> &shape) {
  if (bias)
    assert(bias->size() == (size_t)oc);
  assert(rshift->size() == (size_t)oc);
  assert(multiplier->size() == (size_t)oc);

  int64_t isz = bias ? 9 : 5;
  shape = std::vector<int64_t>{oc, 1, isz};

  auto packed = std::make_unique<std::vector<uint8_t> >(oc * isz);

  uint8_t *ptr = packed->data();
  for (int i = 0; i < oc; i++) {
    if (bias) {
      uint32_t val = (uint32_t)(*bias)[i];
      *ptr = (uint8_t)(val & 0xff);
      ptr++;
      *ptr = (uint8_t)((val >> 8) & 0xff);
      ptr++;
      *ptr = (uint8_t)((val >> 16) & 0xff);
      ptr++;
      *ptr = (uint8_t)((val >> 24) & 0xff);
      ptr++;
    }

    {
      uint32_t val = (uint32_t)(*multiplier)[i];
      *ptr = (uint8_t)(val & 0xff);
      ptr++;
      *ptr = (uint8_t)((val >> 8) & 0xff);
      ptr++;
      *ptr = (uint8_t)((val >> 16) & 0xff);
      ptr++;
      *ptr = (uint8_t)((val >> 24) & 0xff);
      ptr++;
    }

    {
      uint8_t val = (uint8_t)(*rshift)[i];
      *ptr = (uint8_t)val;
      ptr++;
    }
  }

  return std::move(packed);
}


template <typename OpTy>
struct PackWeightConv2DOpPattern : public RewritePattern {
  PackWeightConv2DOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    auto convOp = cast<OpTy>(op);
    if (getOpQuant(op) != "INT8" || !isOpQuantPerchannel(op)
        || getOpQuantParamType(op) != "RSHIFT_AND_M_I32") {
      // for perchannel multiplier mode only
      return matchFailure();
    }
    if ( !isTensorNone(convOp.bias()) ) {
      auto biasOp = cast<tpu::LoadWeightOp>(convOp.bias()->getDefiningOp());
      if (biasOp.lowered()) {
        // packed already
        return matchFailure();
      }
    }
    assert( !isTensorNone(convOp.quant_rshift()) );
    assert( !isTensorNone(convOp.quant_multiplier()) );
    llvm::errs() << "Pack Weight for Conv2D: " << getOpName(op) << "\n";
    TensorFile *wTF = getWeightTensorFile(op);
    Value *wfV = getWeightFileValue(op);

    // get param
    auto filter_type = convOp.filter()->getType().template cast<TensorType>();
    std::vector<int64_t> filter_shape(filter_type.getShape());
    int64_t oc;
    auto g = convOp.param().group().getValue().getLimitedValue();
    if (g != 1) {
      assert(filter_shape.size() == 5);
      oc = filter_shape[0] * filter_shape[1];
    } else {
      assert(filter_shape.size() == 4);
      oc = filter_shape[0];
    }

    // get tensor
    std::unique_ptr<std::vector<float> > bias = nullptr;
    if ( !isTensorNone(convOp.bias()) ) {
      bias = readAndDeleteWeightTensor<float>(convOp.bias(), wTF);
    }
    auto rshift = readAndDeleteWeightTensor<float>(convOp.quant_rshift(), wTF);
    auto multiplier = readAndDeleteWeightTensor<float>(convOp.quant_multiplier(), wTF);

    // pack the weights
    std::vector<int64_t> packedShape;
    auto packed = packWeight(bias.get(), rshift.get(), multiplier.get(), oc,
                             packedShape);

    // store to the packed per_channel operand in "UINT8"
    if (bias) {
      addWeightTensorAndUpdateWeightOp<uint8_t>(convOp.bias(),
          "pack", *packed, packedShape, "UINT8", wTF);
    } else {
      auto packed_op = addWeightTensorAndCreateWeightOp<uint8_t>(
          op, "pack", *packed, packedShape, "UINT8",
          wTF, wfV);
      convOp.setOperand(2, packed_op);
    }
    auto biasOp = cast<tpu::LoadWeightOp>(convOp.bias()->getDefiningOp());
    biasOp.setAttr("lowered", rewriter.getBoolAttr(true));

    // erase quant_rshift and quant_multiplier tensor
    auto NoneOp = OpBuilder(op).create<tpu::NoneOp>(
        rewriter.getUnknownLoc(), rewriter.getNoneType());
    convOp.setOperand(5, NoneOp);
    convOp.setOperand(6, NoneOp);

    return matchSuccess();
  }
};

template<typename OpTy>
struct DefaultErasePattern : public RewritePattern {
  DefaultErasePattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, {op->getOperand(0)});
    return matchSuccess();
  }
};

Operation* GroupOptimizer::build_fn(FuncOp *fn, MLIRContext * context, GroupOptimizer * optimizer) {

  set_input_output_tensor();
  u64 neuron_size = gmem_mgr_.assign_global_memory(groups_, clDisableGMemOptimize);

  // create a map file
  std::unique_ptr<llvm::ToolOutputFile> neuronMapFile = nullptr;
  if (clNeuronMapFilename != "-") {
    std::string errorMessage;
    neuronMapFile = openOutputFile(clNeuronMapFilename, &errorMessage);
    if (!neuronMapFile) {
      llvm::errs() << errorMessage << "\n";
      exit(1);
    }
  }

  // first, merge conv rshift/multiplier/bias into one packed tensor
  OwningRewritePatternList patterns_pack;
  patterns_pack.insert<
      PackWeightConv2DOpPattern<tpu::Conv2DOp>,
      PackWeightConv2DOpPattern<tpu::DeConv2DOp>
      >(context);
  applyPatternsGreedily(*fn, patterns_pack);

  // second, do weight lower on weight tensors
  // lower means transpose and save as storageType (int8/bf16,etc)
  OwningRewritePatternList patterns_lower;
  patterns_lower.insert<
      LowerConv2DOpWeightPattern<tpu::Conv2DOp>,
      LowerWeightFullyConnectedOpPattern
      >(context);
  applyPatternsGreedily(*fn, patterns_lower);

  OwningRewritePatternList tg_patterns;
  tg_patterns.insert<
      addGroupTGLayerPattern<tpu::InputOp>,
      addGroupTGLayerPattern<tpu::Conv2DOp>,
      addGroupTGLayerPattern<tpu::EltwiseAddOp>,
      addGroupTGLayerPattern<tpu::EltwiseMaxOp>,
      addGroupTGLayerPattern<tpu::EltwiseMulOp>,
      addGroupTGLayerPattern<tpu::PoolAvg2DOp>,
      addGroupTGLayerPattern<tpu::PoolMax2DOp>,
      addGroupTGLayerPattern<tpu::FullyConnectedOp>
  >(context, optimizer);
  applyPatternsGreedily(*fn, tg_patterns);

  // write input data in first row of neuron map file
  OwningRewritePatternList tg_addr_patterns;
  tg_addr_patterns.insert<
      addTGLayerGAddrPattern<tpu::TG_INT8_InputOp>
  >(context, optimizer, neuronMapFile->os());
  applyPatternsGreedily(*fn, tg_addr_patterns);

  tg_addr_patterns.clear();
  tg_addr_patterns.insert<
      addTGLayerGAddrPattern<tpu::TG_INT8_PC_Conv2DOp>,
      addTGLayerGAddrPattern<tpu::TG_INT8_EltwiseAddOp>,
      addTGLayerGAddrPattern<tpu::TG_INT8_EltwiseMaxOp>,
      addTGLayerGAddrPattern<tpu::TG_INT8_EltwiseMulOp>,
      addTGLayerGAddrPattern<tpu::TG_INT8_PoolAvg2DOp>,
      addTGLayerGAddrPattern<tpu::TG_INT8_PoolMax2DOp>,
      addTGLayerGAddrPattern<tpu::TG_INT8_FullyConnectedOp>
  >(context, optimizer, neuronMapFile->os());
  applyPatternsGreedily(*fn, tg_addr_patterns);

  tg_addr_patterns.clear();
  tg_addr_patterns.insert<
      DefaultErasePattern<tpu::SoftmaxOp>
  >(context);
  applyPatternsGreedily(*fn, tg_addr_patterns);

  if (neuronMapFile) {
    neuronMapFile->keep();
  }

}

}