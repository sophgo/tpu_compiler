
#include "ImLayer.hpp"

#define DEBUG_TYPE "group_ops"
namespace mlir {

ImLayer::ImLayer(IR_TYPE type, Operation* op, bool fusible)
    : do_relu(false),
      in_tensors(),
      out_tensors(),
      imm_tensors(),
      is_tg_layer(false),
      fusible(fusible),
      id_(-1),
      type_(type),
      op_(op) {
  name_ = mlir::getOpName(op);
  //is_inplace_layer = op->in_place();
  is_inplace_layer = false;
}

ImLayer::~ImLayer() = default;

void ImLayer::add_in_tensor(int n, int c, int h, int w, int unit_size, string& storage , const string& name,
                            tensor_type_t type, gaddr_t gaddr) {
  shared_ptr<Tensor> tensor = Tensor::register_tensor(n, c, h, w, unit_size, storage, name, type, gaddr);
  in_tensors.push_back(tensor);
}

void ImLayer::add_in_tensor(ShapedType* shape, const string& name, tensor_type_t type,
                            gaddr_t gaddr) {
  shared_ptr<Tensor> tensor = Tensor::register_tensor(shape, name, type, gaddr);
  in_tensors.push_back(tensor);
}

void ImLayer::add_in_tensor(Value * v, tensor_type_t type, gaddr_t gaddr) {
  auto def_op = v->getDefiningOp();
  auto shape = v->getType().dyn_cast<TensorType>();
  if (def_op && !isa<tpu::NoneOp>(def_op) && !isa<tpu::WeightFileOp>(def_op)
      && !isa<ReturnOp>(def_op)) {
    if (auto load_op = dyn_cast<tpu::LoadWeightOp>(def_op)) {
      string name = load_op.name();
      shared_ptr<Tensor> tensor = Tensor::register_tensor(&shape, name, TENSOR_COEFF, gaddr);
      in_tensors.push_back(tensor);
    } else {
      string name = mlir::getOpName(def_op);
      shared_ptr<Tensor> tensor = Tensor::register_tensor(&shape, name, type, gaddr);
      in_tensors.push_back(tensor);
    }
  }
}

void ImLayer::add_out_tensor(Value * v, tensor_type_t type, gaddr_t gaddr) {
  auto def_op = v->getDefiningOp();
  auto shape = v->getType().dyn_cast<TensorType>();
  if (!isa<tpu::NoneOp>(def_op) && !isa<tpu::WeightFileOp>(def_op)) {
    if (auto load_op = dyn_cast<tpu::LoadWeightOp>(def_op)) {
      string name = load_op.name();
      shared_ptr<Tensor> tensor = Tensor::register_tensor(&shape, name, TENSOR_COEFF, gaddr);
      out_tensors.push_back(tensor);
    } else if (auto ret_op = dyn_cast<ReturnOp>(def_op)) {
      shared_ptr<Tensor> tensor = Tensor::register_tensor(&shape, "return", TENSOR_NEURON, gaddr);
      out_tensors.push_back(tensor);
    } else {
      string name = mlir::getOpName(def_op);
      shared_ptr<Tensor> tensor = Tensor::register_tensor(&shape, name, type, gaddr);
      out_tensors.push_back(tensor);
    }
  }
}

void ImLayer::add_imm_tensor(const shared_ptr<Tensor> associcate, int count, const string& name) {
  shared_ptr<Tensor> tensor = Tensor::register_imm_tensor(associcate, count, name_ + "_imm");
  imm_tensors.push_back(tensor);
}

shared_ptr<ImLayer> ImLayer::create(Operation* op) {
  shared_ptr<ImLayer> layer;
  if (isa<tpu::TG_INT8_PC_Conv2DOp>(op)) {
    layer = make_shared<ImConv>(op);
  } else if (isa<tpu::TG_INT8_PC_DeConv2DOp>(op)) {
    layer = make_shared<ImDeconv>(op);
  } else if (isa<tpu::TG_INT8_EltwiseAddOp>(op)
             ||isa<tpu::TG_INT8_EltwiseMulOp>(op)) {
    layer = make_shared<ImEltwise>(op);
  } else if (isa<tpu::TG_INT8_FullyConnectedOp>(op)) {
    layer = make_shared<ImInnerproduct>(op);
  } else if (isa<tpu::ReshapeOp>(op)){
    layer = make_shared<ImCommon>(op, true, IR_OTHER);
  } else if (isa<tpu::TG_INT8_PoolAvg2DOp>(op) ||
             isa<tpu::TG_INT8_PoolMax2DOp>(op)) {
    layer = make_shared<ImPooling>(op);
  } else if (isa<tpu::TG_INT8_ConcatOp>(op)) {
    layer = make_shared<ImConcat>(op);
  }else if ( isa<tpu::TG_INT8_LutOp>(op)) {
    layer = make_shared<ImActivation>(op);
  } else if ( isa<tpu::TG_INT8_PReluOp>(op)) {
    layer = make_shared<ImPRelu>(op);
  } else if (isa<tpu::TG_INT8_ShuffleChannelOp>(op)) {
    layer = make_shared<ImShuffleChannel>(op);
  } else if (isa<tpu::TG_INT8_SliceOp>(op)) {
    layer = make_shared<ImSlice>(op);
  } else if (isa<tpu::TG_INT8_LrnOp>(op)) {
    layer = make_shared<ImLrn>(op);
  } else if (isa<tpu::TG_INT8_BroadcastMulOp>(op)) {
    layer = make_shared<ImBroadcastMul>(op);
  } else if (isa<tpu::TG_INT8_UpsampleOp>(op)) {
    layer = make_shared<ImUpsample>(op);
  } else if (isa<tpu::TG_INT8_LeakyReluOp>(op)) {
    layer = make_shared<ImLeakyRelu>(op);
  } else if (isa<tpu::GenericCpuOp>(op)) {
    layer = make_shared<ImCommon>(op, false, IR_OTHER);
  } else if (isa<tpu::QuantOp>(op) ||
             isa<tpu::InputOp>(op) ) {
    layer = make_shared<ImCommon>(op, true, IR_OTHER);
  } else {
    LLVM_DEBUG(llvm::errs()
      << "Not support ImLayer: " << getOpName(op) << "\n";);
    //assert(0);
    layer = make_shared<ImCommon>(op, false, IR_OTHER);
  }
  return layer;
}

void ImLayer::register_it(shared_ptr<ImLayer>& layer) {
  layer->set_id(layers.size());
  layers.push_back(layer);
}

void ImLayer::unregister_all() { layers.clear(); }

vector<shared_ptr<ImLayer>> ImLayer::layers;

static string getOperandStorage(Operation *p) {
  auto op = cast<tpu::LoadWeightOp>(p);
  return op.storage();
}

ImConv::ImConv(Operation* p) : ImLayer(IR_CONVOLUTION, p, true), conv1x1_to_fc(false) {
  auto op = dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(p);
  assert(op);
  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, pt, pb, pl, pr, dh, dw;
  bool is_deconv = isa<tpu::TG_INT8_PC_DeConv2DOp>(op.getOperation());
  parseConvParam(op.param(), is_deconv, op.input(), op.output(), op.filter(),
                  n, ic, ih, iw, oc, oh, ow, g,
                  kh, kw, sh, sw, pt, pb, pl, pr, dh, dw, is_dw, with_bias, do_relu);

  // add input tensor
  add_in_tensor(op.input(), TENSOR_NEURON);

  // add weight tensor
  auto weightOp = cast<tpu::LoadWeightOp>(op.filter()->getDefiningOp());
  // if has ic align
  bool do_ic_alignment = op.do_ic_alignment().hasValue() ?
                            op.do_ic_alignment().getValue() : false;
  int w_ic = ic;
  if (do_ic_alignment && (ic % 2 != 0)) {
    w_ic += 1;
  }
  string weightOpName = weightOp.name().str();
  int32_t unit_size = getOperandStorageSize(weightOp);
  string storage = getOperandStorage(weightOp);
  if (is_dw) {
    add_in_tensor(1, oc, kh, kw, unit_size, storage, weightOpName, TENSOR_DEPTHCONV_OPD1);
  }
  else {
    // tensor shape in local memory should be (1, oc, kh*kw, ic/g)
    add_in_tensor(w_ic / g, oc, kh, kw, unit_size, storage, weightOpName, TENSOR_COEFF);
  }

  // add bias tensor
  int perchannel_size = with_bias ? 9 : 5;
  auto load_bias = cast<tpu::LoadWeightOp>(op.getOperand(2)->getDefiningOp());
  string bias_name = load_bias.name().str();
  string bias_storage = getOperandStorage(load_bias);
  int bias_usize = getOperandStorageSize(load_bias);

  if (is_dw)
    add_in_tensor(1, oc, 1, perchannel_size, bias_usize, storage, bias_name, TENSOR_BIAS);
  else {
    // bias tensor start address must from tpu0, but input and result
    // can start from tpux, so we use the shape (g, oc/g, 1, 9), not
    // (1, oc, 1, 9)
    add_in_tensor(g, oc/g, 1, perchannel_size, bias_usize, storage, bias_name, TENSOR_BIAS);
  }

  // add out tensor
  add_out_tensor(op.output(), TENSOR_NEURON);
  if (op.fused_leaky()) {
    add_imm_tensor(out_tensors[0], 1, name_ + "_imm");
  }
}

ImDeconv::ImDeconv(Operation* p) : ImLayer(IR_DECONVOLUTION, p, true) {
  auto op = dyn_cast<tpu::TG_INT8_PC_DeConv2DOp>(p);
  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, pt, pb, pl, pr, dh, dw;
  bool is_deconv = isa<tpu::TG_INT8_PC_DeConv2DOp>(op.getOperation());
  parseConvParam(op.param(), is_deconv, op.input(), op.output(), op.filter(),
                  n, ic, ih, iw, oc, oh, ow, g,
                  kh, kw, sh, sw, pt, pb, pl, pr, dh, dw, is_dw, with_bias, do_relu);

  // add input tensor
  add_in_tensor(op.input(), TENSOR_NEURON);

  // add weight tensor
  auto weightOp = cast<tpu::LoadWeightOp>(op.filter()->getDefiningOp());
  // if has ic align
  bool do_ic_alignment = op.do_ic_alignment().hasValue() ?
                            op.do_ic_alignment().getValue() : false;
  int w_ic = ic;
  if (do_ic_alignment && (ic % 2 != 0)) {
    w_ic += 1;
  }
  string weightOpName = weightOp.name().str();
  int32_t unit_size = getOperandStorageSize(weightOp);
  string storage = getOperandStorage(weightOp);
  auto weight_type = op.filter()->getType().template cast<TensorType>();

  if (is_dw) {
    add_in_tensor(1, oc, kh, kw, unit_size, storage, weightOpName, TENSOR_DEPTHCONV_OPD1);
  } else {
    // tensor shape in local memory should be (1, oc, kh*kw, ic/g)
    add_in_tensor(w_ic / g, oc, kh, kw, unit_size, storage, weightOpName, TENSOR_COEFF);
  }

  // add bias tensor
  int perchannel_size = with_bias ? 9 : 5;
  auto load_bias = cast<tpu::LoadWeightOp>(op.getOperand(2)->getDefiningOp());
  string bias_name = load_bias.name().str();
  string bias_storage = getOperandStorage(load_bias);
  int bias_usize = getOperandStorageSize(load_bias);

  if (is_dw) {
    add_in_tensor(1, oc, 1, perchannel_size, bias_usize, storage, bias_name, TENSOR_BIAS);
  } else {
    // bias tensor start address must from tpu0, but the same as input and result that
    // start address can start from tpux, so here we use the shape (g, oc/g, 1, 9), not
    // (1, oc, 1, 9)
    add_in_tensor(g, oc/g, 1, perchannel_size, bias_usize, storage, bias_name, TENSOR_BIAS);
  }

  // add out tensor
  add_out_tensor(op.output(), TENSOR_NEURON);
}

ImPooling::ImPooling(Operation* op) : ImLayer(IR_POOLING, op, true) {
  add_in_tensor(op->getOperand(0), TENSOR_NEURON);
  add_out_tensor(op->getResult(0), TENSOR_NEURON);
}

ImInnerproduct::ImInnerproduct(Operation* op) : ImLayer(IR_INNERPRODUCT, op) {

  add_in_tensor(op->getOperand(0), TENSOR_NEURON);

  // weight
  auto weightOp = cast<tpu::LoadWeightOp>(op->getOperand(1)->getDefiningOp());
  string weightOpName = weightOp.name().str();
  auto s_type = op->getOperand(1)->getType().dyn_cast<TensorType>();
  add_in_tensor(&s_type, weightOpName, TENSOR_COEFF);

  // if bias is not noneop
  if (!isa<tpu::NoneOp>(op->getOperand(2)->getDefiningOp())) {
    auto load_bias = cast<tpu::LoadWeightOp>(op->getOperand(2)->getDefiningOp());
    auto opd_type = op->getOperand(2)->getType().dyn_cast<TensorType>();
    std::vector<int64_t> shape = opd_type.getShape();
    int bias_usize = getOperandStorageSize(load_bias);
    string storage = getOperandStorage(load_bias);
    add_in_tensor(2, 0, 0, shape[0], bias_usize, storage, name_ + "_bias", TENSOR_BIAS);
  }
  add_out_tensor(op->getResult(0), TENSOR_MATRIX);
}

ImEltwise::ImEltwise(Operation* op) : ImLayer(IR_ELTWISE, op, true) {
  // skip rshift and multiplier
  int nInputs = op->getNumOperands();
  for (u32 i = 0; i < nInputs; ++i) {
    add_in_tensor(op->getOperand(i), TENSOR_NEURON);
  }

  add_out_tensor(op->getResult(0), TENSOR_NEURON);

  if (isa<tpu::TG_INT8_EltwiseAddOp>(op))
    add_imm_tensor(out_tensors[0], 1, name_ + "_imm");
}


ImCommon::ImCommon(Operation* op, bool inplace_compute, IR_TYPE type) : ImLayer(type, op) {
  is_inplace_layer = (is_inplace_layer || inplace_compute);
  if (isa<tpu::TG_INT8_EltwiseMaxOp>(op) ||
      isa<tpu::TG_INT8_EltwiseMinOp>(op))
      fusible = false;
  // skip rshift and multiplier
  int nInputs = op->getNumOperands();
  for (u32 i = 0; i < nInputs; ++i) {
    if (BlockArgument::classof(op->getOperand(i))) {
      auto shape = op->getResult(0)->getType().dyn_cast<TensorType>();
      add_in_tensor(&shape, "arg0", TENSOR_NEURON);
    } else {
      add_in_tensor(op->getOperand(i),TENSOR_NEURON);
    }
  }

  for (u32 i = 0; i < op->getNumResults(); ++i) {
    add_out_tensor(op->getResult(i), TENSOR_NEURON);
  }
}

ImConcat::ImConcat(Operation* op) : ImLayer(IR_CONCAT, op, true) {
  // only support axis = 1 for fuse
  auto concat_op = dyn_cast<tpu::TG_INT8_ConcatOp>(op);
  int axis = concat_op.axis().getLimitedValue();
  if (axis != 1)
    fusible = false;
  for (u32 i = 0; i < op->getNumOperands(); ++i) {
      add_in_tensor(op->getOperand(i),TENSOR_NEURON);
  }

  for (u32 i = 0; i < op->getNumResults(); ++i) {
    add_out_tensor(op->getResult(i), TENSOR_NEURON);
  }
}

ImActivation::ImActivation(Operation* op) : ImLayer(IR_ACTIVATION, op, true) {
  if (isa<tpu::TanHOp>(op)) {
    this->fusible = false;
  }

  add_in_tensor(op->getOperand(0), TENSOR_NEURON);

  // add y table
  auto load_y_table = cast<tpu::LoadWeightOp>(op->getOperand(1)->getDefiningOp());
  int usize = getOperandStorageSize(load_y_table);
  string storage = getOperandStorage(load_y_table);
  string y_table_name = load_y_table.name().str();
  add_in_tensor(1, 32, 16, 16, usize, storage, y_table_name, TENSOR_COEFF);

  // add m_table
  if (0) {
    auto load_m_table = cast<tpu::LoadWeightOp>(op->getOperand(2)->getDefiningOp());
    int usize = getOperandStorageSize(load_m_table);
    string storage = getOperandStorage(load_m_table);
    string m_table_name = load_m_table.name().str();
    add_in_tensor(1, 32, 16, 16, usize, storage, m_table_name, TENSOR_COEFF);
  }
  add_out_tensor(op->getResult(0), TENSOR_NEURON);
}

ImPRelu::ImPRelu(Operation* op) : ImLayer(IR_PRELU, op, true) {
  add_in_tensor(op->getOperand(0), TENSOR_NEURON);

  auto load_slope = cast<tpu::LoadWeightOp>(op->getOperand(1)->getDefiningOp());
  string weightOpName = load_slope.name().str();
  auto s_type = op->getOperand(1)->getType().dyn_cast<TensorType>();
  add_in_tensor(&s_type, weightOpName, TENSOR_DEPTHCONV_OPD1);

  add_out_tensor(op->getResult(0), TENSOR_NEURON);
}

// shufflechannel as tg layer
ImShuffleChannel::ImShuffleChannel(Operation *op): ImLayer(IR_SHUFFLECHANNEL, op, false) {
  add_in_tensor(op->getOperand(0), TENSOR_NEURON);
  add_out_tensor(op->getResult(0), TENSOR_NEURON);
}

ImSlice::ImSlice(Operation *op): ImLayer(IR_SLICE, op, false) {
  std::vector<int64_t> dst_shape = getTensorShape(op->getResult(0));
  auto slice_op = dyn_cast<tpu::TG_INT8_SliceOp>(op);
  int axis = slice_op.axis().getLimitedValue();
  // optimization for batch 1, set as inplace layer
  if ((dst_shape[0] == 1) && (axis == 1))
    is_inplace_layer = true;
  add_in_tensor(op->getOperand(0), TENSOR_NEURON);
  add_out_tensor(op->getResult(0), TENSOR_NEURON);
}

ImLrn::ImLrn(Operation *op): ImLayer(IR_LRN, op, true) {
  add_in_tensor(op->getOperand(0), TENSOR_NEURON);
  add_out_tensor(op->getResult(0), TENSOR_NEURON);

  // add sqr weight
  auto load_sqr = cast<tpu::LoadWeightOp>(op->getOperand(1)->getDefiningOp());
  int usize = getOperandStorageSize(load_sqr);
  string storage = getOperandStorage(load_sqr);
  string sqr_name = load_sqr.name().str();
  add_in_tensor(1, 32, 16, 16, usize, storage, sqr_name, TENSOR_COEFF);

  // add power weight
  auto load_pow = cast<tpu::LoadWeightOp>(op->getOperand(2)->getDefiningOp());
  usize = getOperandStorageSize(load_pow);
  storage = getOperandStorage(load_pow);
  string pow_name = load_pow.name().str();
  add_in_tensor(1, 32, 16, 16, usize, storage, pow_name, TENSOR_COEFF);

  add_imm_tensor(in_tensors[0], 5, name_ + "_imm");
}

ImBroadcastMul::ImBroadcastMul(Operation *op): ImLayer(IR_BROADCAST_MUL, op, true) {
  auto bd_op = dyn_cast<tpu::TG_INT8_BroadcastMulOp>(op);
  auto input_type = op->getOperand(0)->getType().dyn_cast<TensorType>();
  auto input_shape = input_type.getShape();
  add_in_tensor(op->getOperand(0), TENSOR_NEURON);
  add_in_tensor(op->getOperand(1), TENSOR_NEURON);
  add_out_tensor(op->getResult(0), TENSOR_NEURON);
  // add bias tensor
  bool with_bias = false;
  int perchannel_size = with_bias ? 9 : 5;
  auto load_bias = cast<tpu::LoadWeightOp>(op->getOperand(2)->getDefiningOp());
  string bias_name = load_bias.name().str();
  string bias_storage = getOperandStorage(load_bias);
  int bias_usize = getOperandStorageSize(load_bias);
  add_in_tensor(input_shape[0], input_shape[1], 1, perchannel_size, bias_usize, bias_storage,
                bias_name, TENSOR_BIAS);
}

ImUpsample::ImUpsample(Operation *op): ImLayer(IR_UPSAMPLE, op, true) {
  add_in_tensor(op->getOperand(0), TENSOR_NEURON);
  add_out_tensor(op->getResult(0), TENSOR_NEURON);
}

ImLeakyRelu::ImLeakyRelu(Operation *op): ImLayer(IR_LEAKY_RELU, op, true) {
  add_in_tensor(op->getOperand(0), TENSOR_NEURON);
  add_out_tensor(op->getResult(0), TENSOR_NEURON);
}

}
