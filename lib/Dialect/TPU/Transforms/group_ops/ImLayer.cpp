
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

void ImLayer::add_in_tensor(int n, int c, int h, int w, int unit_size,
                            std::string& storage , const std::string& name,
                            tensor_type_t type) {
  std::shared_ptr<Tensor> tensor =
      Tensor::register_tensor(n, c, h, w, unit_size, storage, name, type);
  in_tensors.push_back(tensor);
}

void ImLayer::add_in_tensor(ShapedType* shape, const std::string& name,
                            tensor_type_t type) {
  std::shared_ptr<Tensor> tensor =
      Tensor::register_tensor(shape, name, type);
  in_tensors.push_back(tensor);
}

void ImLayer::add_in_tensor(Value * v, tensor_type_t type) {
  auto def_op = v->getDefiningOp();
  auto shape = v->getType().dyn_cast<TensorType>();
  if (def_op && !isa<tpu::NoneOp>(def_op) && !isa<tpu::WeightFileOp>(def_op)
      && !isa<ReturnOp>(def_op)) {
    if (auto load_op = dyn_cast<tpu::LoadWeightOp>(def_op)) {
      std::string name = load_op.name();
      std::shared_ptr<Tensor> tensor =
          Tensor::register_tensor(&shape, name, TENSOR_COEFF);
      in_tensors.push_back(tensor);
    } else {
      std::string name = mlir::getOpName(def_op);
      std::shared_ptr<Tensor> tensor =
          Tensor::register_tensor(&shape, name, type);
      in_tensors.push_back(tensor);
    }
  }
}

void ImLayer::add_out_tensor(Value * v, tensor_type_t type) {
  auto def_op = v->getDefiningOp();
  auto shape = v->getType().dyn_cast<TensorType>();
  if (!isa<tpu::NoneOp>(def_op) && !isa<tpu::WeightFileOp>(def_op)) {
    if (auto load_op = dyn_cast<tpu::LoadWeightOp>(def_op)) {
      std::string name = load_op.name();
      std::shared_ptr<Tensor> tensor =
          Tensor::register_tensor(&shape, name, TENSOR_COEFF);
      out_tensors.push_back(tensor);
    } else if (auto ret_op = dyn_cast<ReturnOp>(def_op)) {
      std::shared_ptr<Tensor> tensor =
          Tensor::register_tensor(&shape, "return", TENSOR_NEURON);
      out_tensors.push_back(tensor);
    } else {
      std::string name = mlir::getOpName(def_op);
      std::shared_ptr<Tensor> tensor =
          Tensor::register_tensor(&shape, name, type);
      out_tensors.push_back(tensor);
    }
  }
}

void ImLayer::add_imm_tensor(const std::shared_ptr<Tensor> associcate,
                             int count, const std::string& name) {
  std::shared_ptr<Tensor> tensor =
      Tensor::register_imm_tensor(associcate, count, name_ + "_imm");
  imm_tensors.push_back(tensor);
}

std::shared_ptr<ImLayer> ImLayer::create(Operation* op) {
  std::shared_ptr<ImLayer> layer;
  if (isa<tpu::TG_INT8_PC_Conv2DOp>(op) ||
      isa<tpu::TG_BF16_Conv2DOp>(op)) {
    layer = std::make_shared<ImConv>(op);
  } else if (isa<tpu::TG_INT8_PC_DeConv2DOp>(op) ||
             isa<tpu::TG_BF16_DeConv2DOp>(op)) {
    layer = std::make_shared<ImDeconv>(op);
  } else if (isa<tpu::TG_INT8_EltwiseAddOp>(op) ||
             isa<tpu::TG_INT8_EltwiseMulOp>(op) ||
             isa<tpu::TG_BF16_EltwiseMulOp>(op) ||
             isa<tpu::TG_BF16_EltwiseMulOp>(op)) {
    layer = std::make_shared<ImEltwise>(op);
  } else if (isa<tpu::TG_INT8_FullyConnectedOp>(op) ||
             isa<tpu::TG_BF16_FullyConnectedOp>(op)) {
    layer = std::make_shared<ImInnerproduct>(op);
  } else if (isa<tpu::ReshapeOp>(op)){
    layer = std::make_shared<ImCommon>(op, true, IR_OTHER);
  } else if (isa<tpu::TG_INT8_PoolAvg2DOp>(op) ||
             isa<tpu::TG_INT8_PoolMax2DOp>(op) ||
             isa<tpu::TG_BF16_PoolAvg2DOp>(op) ||
             isa<tpu::TG_BF16_PoolMax2DOp>(op)) {
    layer = std::make_shared<ImPooling>(op);
  } else if (isa<tpu::TG_INT8_ConcatOp>(op) ||
             isa<tpu::TG_BF16_ConcatOp>(op)) {
    layer = std::make_shared<ImConcat>(op);
  }else if (isa<tpu::TG_INT8_LutOp>(op) ||
            isa<tpu::TG_BF16_LutOp>(op)) {
    layer = std::make_shared<ImActivation>(op);
  } else if (isa<tpu::TG_INT8_PReluOp>(op) ||
             isa<tpu::TG_BF16_PReluOp>(op)) {
    layer = std::make_shared<ImPRelu>(op);
  } else if (isa<tpu::TG_INT8_ShuffleChannelOp>(op) ||
             isa<tpu::TG_BF16_ShuffleChannelOp>(op)) {
    layer = std::make_shared<ImShuffleChannel>(op);
  } else if (isa<tpu::TG_INT8_SliceOp>(op) ||
             isa<tpu::TG_BF16_SliceOp>(op)) {
    layer = std::make_shared<ImSlice>(op);
  } else if (isa<tpu::TG_INT8_LrnOp>(op) ||
             isa<tpu::TG_BF16_LrnOp>(op)) {
    layer = std::make_shared<ImLrn>(op);
  } else if (isa<tpu::TG_INT8_BroadcastMulOp>(op) ||
             isa<tpu::TG_BF16_BroadcastMulOp>(op)) {
    layer = std::make_shared<ImBroadcastMul>(op);
  } else if (isa<tpu::TG_INT8_UpsampleOp>(op) ||
             isa<tpu::TG_BF16_UpsampleOp>(op)) {
    layer = std::make_shared<ImUpsample>(op);
  } else if (isa<tpu::TG_INT8_LeakyReluOp>(op) ||
             isa<tpu::TG_BF16_LeakyReluOp>(op)) {
    layer = std::make_shared<ImLeakyRelu>(op);
  } else if (isa<tpu::TG_INT8_PadOp>(op) ||
             isa<tpu::TG_BF16_PadOp>(op)) {
    layer = std::make_shared<ImPad>(op);
  } else if (isa<tpu::TG_INT8_CropOp>(op) ||
             isa<tpu::TG_BF16_CropOp>(op)) {
    layer = std::make_shared<ImCrop>(op);
  } else if (isa<tpu::TG_INT8_ReluOp>(op) ||
             isa<tpu::TG_BF16_ReluOp>(op)) {
    layer = std::make_shared<ImRelu>(op);
  } else if (isa<tpu::TG_CastOp>(op)) {
    layer = std::make_shared<ImCommon>(op, false, IR_OTHER);
  } else if (isa<tpu::GenericCpuOp>(op)) {
    layer = std::make_shared<ImCommon>(op, false, IR_OTHER);
  } else if (isa<tpu::TG_INT8_QuantOp>(op) || isa<tpu::TG_BF16_QuantOp>(op)) {
    layer = std::make_shared<ImQuant>(op);
  } else if (isa<tpu::QuantOp>(op) ||
             isa<tpu::InputOp>(op) ) {
    layer = std::make_shared<ImCommon>(op, true, IR_OTHER);
  } else {
    LLVM_DEBUG(llvm::errs()
      << "Not support ImLayer: " << getOpName(op) << "\n";);
    layer = std::make_shared<ImCommon>(op, false, IR_OTHER);
  }
  return layer;
}

void ImLayer::register_it(std::shared_ptr<ImLayer>& layer) {
  layer->set_id(layers.size());
  layers.push_back(layer);
}

void ImLayer::unregister_all() { layers.clear(); }

std::vector<std::shared_ptr<ImLayer>> ImLayer::layers;

static int getOpResultUnitSize(Operation *op) {
  RankedTensorType result_type =
    op->getResult(0)->getType().cast<RankedTensorType>();

  int usize = result_type.getElementTypeBitWidth()/8;
  return usize;
}

static std::string getWeightStorage(Operation *p) {
  auto op = cast<tpu::LoadWeightOp>(p);
  return op.storage();
}

static void getConvParam( Operation *p,
                          int &n, int &ic, int &ih, int &iw,
                          int &oc, int &oh, int &ow, int &g,
                          int &kh, int &kw,
                          bool &is_dw, bool &with_bias,
                          bool &do_relu,
                          bool &do_ic_align,
                          bool &fuse_leaky) {
  if (isa<tpu::TG_INT8_PC_Conv2DOp>(p)) {
    auto op = dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(p);
    int sh, sw, pt, pb, pl, pr, dh, dw;
    bool is_deconv = isa<tpu::TG_INT8_PC_DeConv2DOp>(op.getOperation());
    parseConvParam(op.param(), is_deconv, op.input(), op.output(), op.filter(),
                    n, ic, ih, iw, oc, oh, ow, g,
                    kh, kw, sh, sw, pt, pb, pl, pr, dh, dw, is_dw, with_bias, do_relu);
    do_ic_align = op.do_ic_alignment().hasValue() ?
                  op.do_ic_alignment().getValue() : false;
    fuse_leaky = op.fused_leaky();
  } else if (isa<tpu::TG_BF16_Conv2DOp>(p)) {
    auto op = dyn_cast<tpu::TG_BF16_Conv2DOp>(p);
    int sh, sw, pt, pb, pl, pr, dh, dw;
    bool is_deconv = isa<tpu::TG_BF16_DeConv2DOp>(op.getOperation());
    parseConvParam(op.param(), is_deconv, op.input(), op.output(), op.filter(),
                    n, ic, ih, iw, oc, oh, ow, g,
                    kh, kw, sh, sw, pt, pb, pl, pr, dh, dw, is_dw, with_bias, do_relu);
    do_ic_align = op.do_ic_alignment().hasValue() ?
                  op.do_ic_alignment().getValue() : false;
    fuse_leaky = op.fused_leaky();
  } else {
    assert("Only support INT8/BF16 Conv in LayerGroup");
  }
}

ImConv::ImConv(Operation* p) : ImLayer(IR_CONVOLUTION, p, true) {
  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw;
  bool do_ic_align = false;
  bool fuse_leaky = false;
  getConvParam(p, n, ic, ih, iw, oc, oh, ow,
               g, kh, kw, is_dw, with_bias,
               do_relu, do_ic_align,
               fuse_leaky);

  int w_ic = ic;
  if (do_ic_align && (ic % 2 != 0)) {
    w_ic += 1;
  }

  // add input tensor
  add_in_tensor(p->getOperand(0), TENSOR_NEURON);

  // add weight tensor
  auto weightOp = cast<tpu::LoadWeightOp>(p->getOperand(1)->getDefiningOp());
  std::string weightOpName = weightOp.name().str();
  int32_t unit_size = getOpResultUnitSize(weightOp);
  std::string storage = getWeightStorage(weightOp);
  if (is_dw) {
    add_in_tensor(1, oc, kh, kw, unit_size, storage, weightOpName, TENSOR_DEPTHCONV_OPD1);
  }
  else {
    // tensor shape in local memory should be (1, oc, kh*kw, ic/g)
    add_in_tensor(w_ic / g, oc, kh, kw, unit_size, storage, weightOpName, TENSOR_COEFF);
  }

  // add bias tensor
  int perchannel_size = with_bias ? 9 : 5;
  auto load_bias = cast<tpu::LoadWeightOp>(p->getOperand(2)->getDefiningOp());
  std::string bias_name = load_bias.name().str();
  std::string bias_storage = getWeightStorage(load_bias);
  int bias_usize = getOpResultUnitSize(load_bias);

  if (is_dw)
    add_in_tensor(1, oc, 1, perchannel_size, bias_usize, storage, bias_name, TENSOR_BIAS);
  else {
    // bias tensor start address must from tpu0, but input and result
    // can start from tpux, so we use the shape (g, oc/g, 1, 9), not
    // (1, oc, 1, 9)
    add_in_tensor(g, oc/g, 1, perchannel_size, bias_usize, storage, bias_name, TENSOR_BIAS);
  }

  // add out tensor
  add_out_tensor(p->getResult(0), TENSOR_NEURON);
  if (fuse_leaky) {
    add_imm_tensor(out_tensors[0], 1, name_ + "_imm");
  }
}

static void getDeconvParam( Operation *p,
                            int &n, int &ic, int &ih, int &iw,
                            int &oc, int &oh, int &ow, int &g,
                            int &kh, int &kw,
                            bool &is_dw, bool &with_bias,
                            bool &do_relu,
                            bool &do_ic_align) {
  if (isa<tpu::TG_INT8_PC_DeConv2DOp>(p)) {
    auto op = dyn_cast<tpu::TG_INT8_PC_DeConv2DOp>(p);
    int sh, sw, pt, pb, pl, pr, dh, dw;
    bool is_deconv = isa<tpu::TG_INT8_PC_DeConv2DOp>(op.getOperation());
    parseConvParam(op.param(), is_deconv, op.input(), op.output(), op.filter(),
                    n, ic, ih, iw, oc, oh, ow, g,
                    kh, kw, sh, sw, pt, pb, pl, pr, dh, dw, is_dw, with_bias, do_relu);
    do_ic_align = op.do_ic_alignment().hasValue() ?
                  op.do_ic_alignment().getValue() : false;
  } else if (isa<tpu::TG_BF16_DeConv2DOp>(p)) {
    auto op = dyn_cast<tpu::TG_BF16_DeConv2DOp>(p);
    int sh, sw, pt, pb, pl, pr, dh, dw;
    bool is_deconv = isa<tpu::TG_BF16_DeConv2DOp>(op.getOperation());
    parseConvParam(op.param(), is_deconv, op.input(), op.output(), op.filter(),
                    n, ic, ih, iw, oc, oh, ow, g,
                    kh, kw, sh, sw, pt, pb, pl, pr, dh, dw, is_dw, with_bias, do_relu);
    do_ic_align = op.do_ic_alignment().hasValue() ?
                  op.do_ic_alignment().getValue() : false;
  } else {
    assert("Only support INT8/BF16 DeConv in LayerGroup");
  }
}

ImDeconv::ImDeconv(Operation* p) : ImLayer(IR_DECONVOLUTION, p, true) {
  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw;
  bool do_ic_align;
  getDeconvParam(p, n, ic, ih, iw, oc, oh, ow,
                 g, kh, kw, is_dw, with_bias, do_relu, do_ic_align);

  // handle ic align for double conv
  int w_ic = ic;
  if (do_ic_align && (ic % 2 != 0)) {
    w_ic += 1;
  }
  // add input tensor
  add_in_tensor(p->getOperand(0), TENSOR_NEURON);

  // add weight tensor
  auto weightOp = cast<tpu::LoadWeightOp>(p->getOperand(1)->getDefiningOp());
  std::string weightOpName = weightOp.name().str();
  int32_t unit_size = getOpResultUnitSize(weightOp);
  std::string storage = getWeightStorage(weightOp);

  if (is_dw) {
    add_in_tensor(1, oc, kh, kw, unit_size, storage,
                  weightOpName, TENSOR_DEPTHCONV_OPD1);
  } else {
    // tensor shape in local memory should be (1, oc, kh*kw, ic/g)
    add_in_tensor(w_ic / g, oc, kh, kw, unit_size, storage,
                  weightOpName, TENSOR_COEFF);
  }

  // add bias tensor
  int perchannel_size = with_bias ? 9 : 5;
  auto load_bias = cast<tpu::LoadWeightOp>(p->getOperand(2)->getDefiningOp());
  std::string bias_name = load_bias.name().str();
  std::string bias_storage = getWeightStorage(load_bias);
  int bias_usize = getOpResultUnitSize(load_bias);

  if (is_dw) {
    add_in_tensor(1, oc, 1, perchannel_size, bias_usize,
                  storage, bias_name, TENSOR_BIAS);
  } else {
    // bias tensor start address must from tpu0,
    // but the same as input and result that
    // start address can start from tpux,
    // so here we use the shape (g, oc/g, 1, 9), not (1, oc, 1, 9)
    add_in_tensor(g, oc/g, 1, perchannel_size, bias_usize,
                  storage, bias_name, TENSOR_BIAS);
  }

  // add out tensor
  add_out_tensor(p->getResult(0), TENSOR_NEURON);
}

ImPooling::ImPooling(Operation* op) : ImLayer(IR_POOLING, op, true) {
  add_in_tensor(op->getOperand(0), TENSOR_NEURON);
  add_out_tensor(op->getResult(0), TENSOR_NEURON);
}

ImInnerproduct::ImInnerproduct(Operation* op) : ImLayer(IR_INNERPRODUCT, op) {

  add_in_tensor(op->getOperand(0), TENSOR_NEURON);

  // weight
  auto weightOp = cast<tpu::LoadWeightOp>(op->getOperand(1)->getDefiningOp());
  std::string weightOpName = weightOp.name().str();
  auto s_type = op->getOperand(1)->getType().dyn_cast<TensorType>();
  add_in_tensor(&s_type, weightOpName, TENSOR_COEFF);

  // if bias is not noneop
  if (!isa<tpu::NoneOp>(op->getOperand(2)->getDefiningOp())) {
    auto load_bias = cast<tpu::LoadWeightOp>(op->getOperand(2)->getDefiningOp());
    auto opd_type = op->getOperand(2)->getType().dyn_cast<TensorType>();
    std::vector<int64_t> shape = opd_type.getShape();
    int bias_usize = getOpResultUnitSize(load_bias);
    std::string storage = getWeightStorage(load_bias);
    add_in_tensor(2, 0, 0, shape[0], bias_usize, storage, name_ + "_bias", TENSOR_BIAS);
  }
  add_out_tensor(op->getResult(0), TENSOR_MATRIX);
}

ImEltwise::ImEltwise(Operation* op) : ImLayer(IR_ELTWISE, op, true) {
  // skip rshift and multiplier
  int nInputs = op->getNumOperands();
  for (uint32_t i = 0; i < nInputs; ++i) {
    add_in_tensor(op->getOperand(i), TENSOR_NEURON);
  }

  add_out_tensor(op->getResult(0), TENSOR_NEURON);

  if (isa<tpu::TG_INT8_EltwiseAddOp>(op) ||
      isa<tpu::TG_BF16_EltwiseAddOp>(op))
    add_imm_tensor(out_tensors[0], 1, name_ + "_imm");
}


ImCommon::ImCommon(Operation* op, bool inplace_compute, IR_TYPE type) : ImLayer(type, op) {
  is_inplace_layer = (is_inplace_layer || inplace_compute);
  if (isa<tpu::TG_INT8_EltwiseMaxOp>(op) ||
      isa<tpu::TG_INT8_EltwiseMinOp>(op) ||
      isa<tpu::TG_BF16_EltwiseMaxOp>(op) ||
      isa<tpu::TG_BF16_EltwiseMinOp>(op))
      fusible = false;
  // skip rshift and multiplier
  int nInputs = op->getNumOperands();
  for (uint32_t i = 0; i < nInputs; ++i) {
    if (BlockArgument::classof(op->getOperand(i))) {
      auto shape = op->getResult(0)->getType().dyn_cast<TensorType>();
      add_in_tensor(&shape, "arg0", TENSOR_NEURON);
    } else {
      add_in_tensor(op->getOperand(i),TENSOR_NEURON);
    }
  }

  for (uint32_t i = 0; i < op->getNumResults(); ++i) {
    add_out_tensor(op->getResult(i), TENSOR_NEURON);
  }
}

ImConcat::ImConcat(Operation* op) : ImLayer(IR_CONCAT, op, true) {
  // only support axis = 1 for fuse
  auto concat_op = dyn_cast<tpu::TG_INT8_ConcatOp>(op);
  int axis = 0;
  if (isa<tpu::TG_INT8_ConcatOp>(op)) {
    auto concat_op = dyn_cast<tpu::TG_INT8_ConcatOp>(op);
    axis = concat_op.axis().getLimitedValue();
  } else if (isa<tpu::TG_BF16_ConcatOp>(op)){
    auto concat_op = dyn_cast<tpu::TG_BF16_ConcatOp>(op);
    axis = concat_op.axis().getLimitedValue();
  }

  if (axis != 1)
    fusible = false;
  for (uint32_t i = 0; i < op->getNumOperands(); ++i) {
      add_in_tensor(op->getOperand(i),TENSOR_NEURON);
  }

  for (uint32_t i = 0; i < op->getNumResults(); ++i) {
    add_out_tensor(op->getResult(i), TENSOR_NEURON);
  }
}

ImActivation::ImActivation(Operation* op) : ImLayer(IR_ACTIVATION, op, true) {

  // FIXME: only support sigmoid now

  add_in_tensor(op->getOperand(0), TENSOR_NEURON);
  bool isBF16 = isa<tpu::TG_BF16_LutOp>(op);
  int table_h = 16;
  int table_w = 16; // 1880 setting
  if (isBF16) {
    // TODO: get chip from `chipname` field
    table_h = 32;
    table_w = 8; // 1880v2 setting
  }

  // add y table
  auto load_y_table = cast<tpu::LoadWeightOp>(op->getOperand(1)->getDefiningOp());
  int usize = getOpResultUnitSize(load_y_table);
  std::string storage = getWeightStorage(load_y_table);
  std::string y_table_name = load_y_table.name().str();
  add_in_tensor(1, NPU_NUM, table_h, table_w, usize, storage, y_table_name, TENSOR_COEFF_LUT);
  add_out_tensor(op->getResult(0), TENSOR_NEURON);

  // add m_table
  if (isBF16) {
    // FIXME: support other bf16 activation ops
    auto load_m_table = cast<tpu::LoadWeightOp>(op->getOperand(2)->getDefiningOp());
    int usize = getOpResultUnitSize(load_m_table);
    std::string storage = getWeightStorage(load_m_table);
    std::string m_table_name = load_m_table.name().str();
    add_in_tensor(1, NPU_NUM, table_h, table_w, usize, storage, m_table_name, TENSOR_COEFF_LUT);

    // add working table
    // NOTICE: 4 dims
    add_imm_tensor(out_tensors[0], 1, name_ + "_imm");
    //std::vector<int64_t> i_s(op->getResult(0)->getType().cast<TensorType>().getShape());
    //add_in_tensor(i_s[0], i_s[1], i_s[2], i_s[3], usize, storage,
    //    getOpName(op).str() + "_working", TENSOR_NEURON);
  }
}

ImQuant::ImQuant(Operation* op) : ImLayer(IR_QUANT, op, true) {

  add_in_tensor(op->getOperand(0), TENSOR_NEURON);
  add_out_tensor(op->getResult(0), TENSOR_NEURON);
}

ImPRelu::ImPRelu(Operation* op) : ImLayer(IR_PRELU, op, true) {
  add_in_tensor(op->getOperand(0), TENSOR_NEURON);

  auto load_slope = cast<tpu::LoadWeightOp>(op->getOperand(1)->getDefiningOp());
  std::string weightOpName = load_slope.name().str();
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
  int axis = 0;

  if (isa<tpu::TG_INT8_SliceOp>(op)) {
    auto slice_op = dyn_cast<tpu::TG_INT8_SliceOp>(op);
    axis = slice_op.axis().getLimitedValue();
  } else if (isa<tpu::TG_BF16_SliceOp>(op)) {
    auto slice_op = dyn_cast<tpu::TG_BF16_SliceOp>(op);
    axis = slice_op.axis().getLimitedValue();
  }

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
  int usize = getOpResultUnitSize(load_sqr);
  std::string storage = getWeightStorage(load_sqr);
  std::string sqr_name = load_sqr.name().str();
  add_in_tensor(1, 32, 16, 16, usize, storage, sqr_name, TENSOR_COEFF);

  // add power weight
  auto load_pow = cast<tpu::LoadWeightOp>(op->getOperand(2)->getDefiningOp());
  usize = getOpResultUnitSize(load_pow);
  storage = getWeightStorage(load_pow);
  std::string pow_name = load_pow.name().str();
  add_in_tensor(1, 32, 16, 16, usize, storage, pow_name, TENSOR_COEFF);

  add_imm_tensor(in_tensors[0], 5, name_ + "_imm");
}

ImBroadcastMul::ImBroadcastMul(Operation *op): ImLayer(IR_BROADCAST_MUL, op, true) {
  auto input_type = op->getOperand(0)->getType().dyn_cast<TensorType>();
  auto input_shape = input_type.getShape();
  add_in_tensor(op->getOperand(0), TENSOR_NEURON);
  add_in_tensor(op->getOperand(1), TENSOR_NEURON);
  add_out_tensor(op->getResult(0), TENSOR_NEURON);
  // add bias tensor
  bool with_bias = false;
  int perchannel_size = with_bias ? 9 : 5;
  auto load_bias = cast<tpu::LoadWeightOp>(op->getOperand(2)->getDefiningOp());
  std::string bias_name = load_bias.name().str();
  std::string bias_storage = getWeightStorage(load_bias);
  int bias_usize = getOpResultUnitSize(load_bias);
  add_in_tensor(input_shape[0], input_shape[1], 1, perchannel_size, bias_usize, bias_storage,
                bias_name, TENSOR_BIAS);
}

ImUpsample::ImUpsample(Operation *op): ImLayer(IR_UPSAMPLE, op, true) {
  int scale = 0;
  if (isa<tpu::TG_INT8_UpsampleOp>(op)) {
    auto upsample_op = dyn_cast<tpu::TG_INT8_UpsampleOp>(op);
    scale = upsample_op.scale().getLimitedValue();
  } else if (isa<tpu::TG_BF16_UpsampleOp>(op)) {
    auto upsample_op = dyn_cast<tpu::TG_BF16_UpsampleOp>(op);
    scale = upsample_op.scale().getLimitedValue();
  }
  // ins_h/ins_w can not exceed 16 for average pooling in tl_upsample
  // which has only 4 bits in hw
  if (scale >= 16)
    fusible = false;
  add_in_tensor(op->getOperand(0), TENSOR_NEURON);
  add_out_tensor(op->getResult(0), TENSOR_NEURON);
}

ImLeakyRelu::ImLeakyRelu(Operation *op): ImLayer(IR_LEAKY_RELU, op, true) {
  add_in_tensor(op->getOperand(0), TENSOR_NEURON);
  add_out_tensor(op->getResult(0), TENSOR_NEURON);
}

ImPad::ImPad(Operation *op): ImLayer(IR_PAD, op, true) {
  add_in_tensor(op->getOperand(0), TENSOR_NEURON);
  add_out_tensor(op->getResult(0), TENSOR_NEURON);
}

ImCrop::ImCrop(Operation *op): ImLayer(IR_CROP, op, true) {
  add_in_tensor(op->getOperand(0), TENSOR_NEURON);
  add_out_tensor(op->getResult(0), TENSOR_NEURON);
}

ImRelu::ImRelu(Operation *op): ImLayer(IR_RELU, op, true) {
  add_in_tensor(op->getOperand(0), TENSOR_NEURON);
  add_out_tensor(op->getResult(0), TENSOR_NEURON);
}
}
