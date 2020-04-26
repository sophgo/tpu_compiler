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


GroupOptimizer::GroupOptimizer(NetGraph* net_graph, FuncOp * fn, MLIRContext * context)
    : net_graph_(net_graph), fn_(fn), context_(context),
      mix_net_(net_graph, fn, context), gmem_mgr_(net_graph){}

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

  int group_id = 0;
  for (auto group : groups_) {
    bmerr_t status = group->assign_steps();
    if (status == BM_ERR_FAILURE) {
      cout << "local memory allocate failed" << endl;
      assert(0);
    }

    if (group->size() == 1) {
      int id = group->layers()[0];
      ImLayer* layer = const_cast<ImLayer*>(net_graph_->get_layer_by_id(id));
      layer->is_tg_layer = true;
    }
    group->set_group_id(group_id);
    group_id++;
  }

  string DumpLayerGroupInfo = "layer_optimizer.out";
  if (DumpLayerGroupInfo != "") {
    std::fstream f;
    f.open(DumpLayerGroupInfo, std::ios::out);
    for (auto* group : groups_) {
      f << "group[" << group->get_group_id() << "]\n";
      group->print(f);
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

// use name to judge if ref to the same op
static bool is_same_layer(Operation * op, const ImLayer * layer) {
  if (isValidLayerGroupOp(op)) {
    string op_name = getOpName(op).str();
    // op_name.erase(std::remove(op_name.begin(), op_name.end(), '\b'), op_name.end());
    if (op_name == layer->name())
      return true;
    else return false;
  }
  return false;
}

bool GroupOptimizer::is_group_start(Operation * op, int * gid) {
  if (!isValidLayerGroupOp(op)) {
    return false;
  }
  int group_id = 0;
  for(auto group: groups_) {
    // check if is the first layer in the group
    int id = group->layers()[0];
    const ImLayer * start_layer = net_graph_->get_layer_by_id(id);
    if (is_same_layer(op, start_layer)) {
      // if only one layer, return false
      if (group->layers().size() > 1) {
        *gid = group_id;
        // llvm::errs() << " success.\n";
        return true;
      }
    }
    group_id++;
  }

  return false;
}

void GroupOptimizer::lower_to_tl(PatternRewriter & rewriter, Operation *op, int gid) {
  Group * group = groups_[gid];
  if (group->lowered()) {
    return;
  }
  int n_secs = group->nsecs_and_hsecs.first;
  int h_secs = group->nsecs_and_hsecs.second;

  mix_net_.set_rewriter(&rewriter);
  mix_net_.set_start_op(op);
  mix_net_.add_group_start_ops(gid, group, op, n_secs, h_secs);

  bool first_loop = true;
  for (int n_loop = 0; n_loop < n_secs; n_loop++) {
    for (int h_loop = 0; h_loop < h_secs; h_loop++) {

      group->update_tensor_slices(n_secs, h_secs, n_loop, h_loop);

      for (int step_id = 0; step_id < group->time_step->get_timestep_num(); ++step_id) {
        mix_net_.parallel_start();
        int cur_layer = group->time_step->get_layer(step_id);

        if (cur_layer != -1) {
          mix_net_.add_tl_layer(
              gid, cur_layer, group->time_step, step_id,
              h_secs != 1, n_loop, h_loop);
        }

        const vector<TENSOR_STEP>& cur_tensors = group->time_step->get_tensors(step_id);
        for (u32 i = 0; i < cur_tensors.size(); ++i) {
          // check if tensor is kept in lmem.
          if ((!first_loop) && cur_tensors[i].second == TIMESTEP_LOAD &&
              group->time_step->is_tensor_hold_in_memory(cur_tensors[i].first)) {
            continue;
          }

          // if tensor is loaded in tsm and holded, don't load it again.
          if ((!first_loop) && cur_tensors[i].second == TIMESTEP_DDR_TO_TSM &&
              group->time_step->is_tensor_hold_in_memory(cur_tensors[i].first)) {
            continue;
          }

          mix_net_.add_transport_param(
              cur_tensors[i], group->time_step, step_id);
        }

        mix_net_.parallel_end();
      }

      first_loop = false;
    }
  }

  mix_net_.add_group_end_ops(gid, group, n_secs, h_secs);

  group->set_lowered(true);
}

static llvm::cl::opt<std::string> clWeightMapFilename(
    "weight-map",
    llvm::cl::desc("record weight offset with its name into a csv map file"),
    llvm::cl::init("-"));

static llvm::cl::opt<std::string> clWeightBinFilename(
    "weight-bin",
    llvm::cl::desc("weight bin filename"),
    llvm::cl::init("-"));

template <typename opTy>
struct TpuLoadWeightOpPattern : public RewritePattern {
  TpuLoadWeightOpPattern(MLIRContext *context,
      llvm::raw_fd_ostream *weightBinaryFile, llvm::raw_ostream &map_os,
      size_t alignment)
      : RewritePattern(opTy::getOperationName(), 1, context),
        weightBinaryFile_(weightBinaryFile),
        map_os_(map_os),
        alignment_(alignment) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    TensorFile *wTF = getWeightTensorFile(op);
    auto weightOp = cast<opTy>(op);
    llvm::StringRef tensor_name;
    TensorType type;
    if (isa<tpu::LoadWeightOp>(op)) {
      auto w_Op = cast<tpu::LoadWeightOp>(op);
      if (w_Op.offset().hasValue()) {
        // assigned already
        return matchFailure();
      }
      tensor_name = w_Op.name().getValue();
      type = w_Op.getResult()->getType().cast<TensorType>();
    } else if (isa<tpu::TL_LG_LoadCoeffOp>(op)) {
      auto w_Op = cast<tpu::TL_LG_LoadCoeffOp>(op);
      if (w_Op.gaddr().hasValue()) {
        // assigned already
        return matchFailure();
      }
      tensor_name = w_Op.name();
      type = w_Op.getResult()->getType().cast<TensorType>();
    }

    // read the tensor
    llvm::errs() << "lower weight for tensor: " << tensor_name << "\n";
    //auto type = weightOp.getResult(0)->getType().cast<TensorType>();
    assert(weightOp.lowered());
    auto curPos = weightBinaryFile_->tell();
    size_t size = 0;
    if (weightOp.storage() == "INT8") {
      std::vector<int8_t> weight_int8;
      auto weight = wTF->readTensor<int8_t>(tensor_name, type);
      weight_int8.assign(weight->begin(), weight->end());
      size = weight_int8.size();

      // pad to alignment
      if ( weight_int8.size() % alignment_ ) {
        size_t pad = alignment_ - (weight_int8.size() % alignment_);
        for (size_t i = 0; i < pad; ++i) {
          weight_int8.push_back(-128); // assign a special value for debugging
        }
      }
      weightBinaryFile_->write(reinterpret_cast<const char*>(weight_int8.data()),
          weight_int8.size() * sizeof(int8_t));
    } else if (weightOp.storage() == "UINT8") {
      // UINT8 is used for packed per-channel info or LUT table
      std::vector<uint8_t> weight_uint8;
      auto weight = wTF->readTensor<uint8_t>(tensor_name, type);
      weight_uint8.assign(weight->begin(), weight->end());
      size = weight_uint8.size();

      // pad to alignment
      if ( weight_uint8.size() % alignment_ ) {
        size_t pad = alignment_ - (weight_uint8.size() % alignment_);
        for (size_t i = 0; i < pad; ++i) {
          weight_uint8.push_back(0xff); // assign a special value for debugging
        }
      }
      weightBinaryFile_->write(reinterpret_cast<const char*>(weight_uint8.data()),
          weight_uint8.size() * sizeof(uint8_t));
    } else if (weightOp.storage() == "INT16") {
      // INT16 is used for bias in INT8 per-tensor mode
      // after lowering, this should be UINT16 already
      assert (false);
    } else if (weightOp.storage() == "UINT16") {
      // this is NOT BF16 (BF16 uses `BF16` directly)
      // this is for lowered and transposed INT16 bias
      auto weight = wTF->readTensor<uint16_t>(tensor_name, type);
      size = weight->size();
      std::vector<uint16_t> weight_uint16(weight->begin(), weight->end());
      size = weight_uint16.size() * sizeof(uint16_t);

      // pad to alignment
      if ((weight_uint16.size() * sizeof(uint16_t)) % alignment_) {
        size_t pad = (alignment_ - (weight_uint16.capacity() % alignment_)) /
                     sizeof(uint16_t);
        for (size_t i = 0; i < pad; ++i) {
          weight_uint16.push_back(0xffff); // assign a special value for debugging
        }
      }
      weightBinaryFile_->write(
          reinterpret_cast<const char *>(weight_uint16.data()),
          weight_uint16.size() * sizeof(uint16_t));
    } else if (weightOp.storage() == "BF16") {
      std::vector<uint16_t> weight_bf16;
      auto weight = wTF->readTensor<uint16_t>(tensor_name, type);
      weight_bf16.assign(weight->begin(), weight->end());
      size = weight_bf16.size() * sizeof(uint16_t);

      // pad to alignment
      if ( (weight_bf16.size()* sizeof(uint16_t)) % alignment_ ) {
        size_t pad = ( alignment_ - ( weight_bf16.capacity() % alignment_ ) )
                     / sizeof(uint16_t);
        for (size_t i = 0; i < pad; ++i) {
          weight_bf16.push_back(0xffff); // assign a special value for debugging
        }
      }
      weightBinaryFile_->write(reinterpret_cast<const char*>(weight_bf16.data()),
          weight_bf16.size() * sizeof(uint16_t));
    } else if (weightOp.storage() == "UINT32") {
      // UINT32 is for lowered Conv Bias
      // 1. Per-Channel (no mulitplier) Conv Bias is supposed to be INT32
      // after transpose, it is stored in striped way (NOT sure yet)
      // 2. BF16 Conv Bias is supposed to be FP32
      // 1880v2 requires storing fp32 into a 2 stripes 16-bit way
      // one stripe for high 16-bit, and one for low 16-bit
      // after the lowering, we store the data as `UINT32`
      std::vector<uint32_t> weight_uint32;
      auto weight = wTF->readTensor<uint32_t>(tensor_name, type);
      weight_uint32.assign(weight->begin(), weight->end());
      size = weight_uint32.size() * sizeof(uint32_t);

      // pad to alignment
      if ( (weight_uint32.size()* sizeof(uint32_t)) % alignment_ ) {
        size_t pad = ( alignment_ - ( weight_uint32.capacity() % alignment_ ) )
                     / sizeof(uint32_t);
        for (size_t i = 0; i < pad; ++i) {
          weight_uint32.push_back(0xffffffff); // assign a special value for debugging
        }
      }
      weightBinaryFile_->write(reinterpret_cast<const char*>(weight_uint32.data()),
          weight_uint32.size() * sizeof(uint32_t));
    } else if (weightOp.storage() == "FP32") {
      assert(false);
    } else if (weightOp.storage() == "NONE") {
      return matchSuccess();
    } else {
      llvm::errs() << tensor_name << " weight storage type "
                   << weightOp.storage() << "\n";
      assert(0 && "not supported weight storage type");
    }

    auto newPos = weightBinaryFile_->tell();
    map_os_ << tensor_name << "," << llvm::format_hex(curPos, 10) << "\n";

    llvm::errs() << llvm::format("[%-36s][%8d] : [ ",
                                 tensor_name.str().c_str(), size)
                 << llvm::format_hex(curPos, 10) << " --> "
                 << llvm::format_hex(newPos, 10) << " ]\n";

    // assign the address to weightOp
    if (isa<tpu::LoadWeightOp>(op))
      weightOp.setAttr("offset", rewriter.getI64IntegerAttr(curPos));
    else if(isa<tpu::TL_LG_LoadCoeffOp>(op)) {
      weightOp.setAttr("gaddr", rewriter.getI64IntegerAttr(curPos));
      llvm::errs() << "set gaddr : " << curPos << "\n";
    }


    return matchSuccess();
  }

  llvm::raw_fd_ostream *weightBinaryFile_;
  llvm::raw_ostream &map_os_;
  size_t alignment_;
};

template<typename TyOp>
struct LGLoweringPattern : public RewritePattern {
  LGLoweringPattern(FuncOp * fn , MLIRContext *context, GroupOptimizer * optimizer)
      : RewritePattern(TyOp::getOperationName(), 1, context),
      optimizer_(optimizer), fn_(fn), context_(context){
        //mix_net_ = optimizer->get_net();
      }

  PatternMatchResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    // if already lowered to tl, return false
    int group_id = 0;
    if (optimizer_->is_group_start(op, &group_id)) {
      llvm::errs() << "Find group start: " << getOpName(op) << "\n";
      optimizer_->lower_to_tl(rewriter, op, group_id);
    }
    return matchSuccess();
  }

  GroupOptimizer *optimizer_;
  FuncOp * fn_;
  MLIRContext * context_;
  MixNet * mix_net_;
};


// lower to tl inst according to layer group result
void GroupOptimizer::lower_to_tl_group(MLIRContext * context) {
  set_input_output_tensor();
  u64 neuron_size = gmem_mgr_.assign_global_memory(groups_, clDisableGMemOptimize);

  OwningRewritePatternList patterns_pack;
  patterns_pack.insert<
      LGLoweringPattern<tpu::Conv2DOp>,
      LGLoweringPattern<tpu::EltwiseAddOp>,
      LGLoweringPattern<tpu::EltwiseMulOp>,
      LGLoweringPattern<tpu::EltwiseMaxOp>,
      LGLoweringPattern<tpu::PoolAvg2DOp>,
      LGLoweringPattern<tpu::PoolMax2DOp>
      >(fn_, context, this);
  applyPatternsGreedily(*fn_, patterns_pack);

}

void GroupOptimizer::lower_to_tg_group(MLIRContext * context) {
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

  OwningRewritePatternList tg_patterns;
  tg_patterns.insert<
      addGroupTGLayerPattern<tpu::InputOp>,
      addGroupTGLayerPattern<tpu::Conv2DOp>,
      addGroupTGLayerPattern<tpu::EltwiseAddOp>,
      addGroupTGLayerPattern<tpu::EltwiseMaxOp>,
      addGroupTGLayerPattern<tpu::EltwiseMulOp>,
      addGroupTGLayerPattern<tpu::PoolAvg2DOp>,
      addGroupTGLayerPattern<tpu::PoolMax2DOp>,
      addGroupTGLayerPattern<tpu::ConcatOp>,
      addGroupTGLayerPattern<tpu::FullyConnectedOp>
  >(context, this);
  applyPatternsGreedily(*fn_, tg_patterns);

  // write input data in first row of neuron map file
  OwningRewritePatternList tg_addr_patterns;
  // tg_addr_patterns.insert<
  //     addTGLayerGAddrPattern<tpu::InputOp>
  // >(context, this, neuronMapFile->os());
  // applyPatternsGreedily(*fn_, tg_addr_patterns);

  tg_addr_patterns.clear();
  tg_addr_patterns.insert<
      addTGLayerGAddrPattern<tpu::TG_INT8_PC_Conv2DOp>,
      addTGLayerGAddrPattern<tpu::TG_INT8_EltwiseAddOp>,
      addTGLayerGAddrPattern<tpu::TG_INT8_EltwiseMaxOp>,
      addTGLayerGAddrPattern<tpu::TG_INT8_EltwiseMulOp>,
      addTGLayerGAddrPattern<tpu::TG_INT8_PoolAvg2DOp>,
      addTGLayerGAddrPattern<tpu::TG_INT8_PoolMax2DOp>,
      addTGLayerGAddrPattern<tpu::TG_INT8_FullyConnectedOp>,
      addTGLayerGAddrPattern<tpu::TG_INT8_ConcatOp>,
      addTGLayerGAddrPattern<tpu::GenericCpuOp>
  >(context, this, neuronMapFile->os());
  applyPatternsGreedily(*fn_, tg_addr_patterns);

  if (neuronMapFile) {
    neuronMapFile->keep();
  }
}

void GroupOptimizer::assign_weight_address(MLIRContext * context) {
// update coeff weight address
  // create a bin file
    std::error_code ec;
    assert(clWeightBinFilename != "-");
    llvm::raw_fd_ostream weightBinaryFile(clWeightBinFilename, ec);

    // create a map file
    std::unique_ptr<llvm::ToolOutputFile> weightMapFile = nullptr;
    if (clWeightMapFilename != "-") {
      std::string errorMessage;
      weightMapFile = openOutputFile(clWeightMapFilename, &errorMessage);
      if (!weightMapFile) {
        llvm::errs() << errorMessage << "\n";
        exit(1);
      }
    }

    OwningRewritePatternList patterns;
    // assign address and generate bin file
    patterns.insert<
      TpuLoadWeightOpPattern<tpu::LoadWeightOp>,
      TpuLoadWeightOpPattern<tpu::TL_LG_LoadCoeffOp>
    >(context,
        &weightBinaryFile, weightMapFile->os(), 16);
    applyPatternsGreedily(*fn_, patterns);

    weightBinaryFile.close();

    if (weightMapFile) {
      weightMapFile->keep();
    }

}

void GroupOptimizer::build_fn(MLIRContext * context) {
  lower_to_tl_group(context);
  lower_to_tg_group(context);
  assign_weight_address(context);
}

}