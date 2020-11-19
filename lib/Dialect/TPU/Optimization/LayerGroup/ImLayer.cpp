
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
  name_ = mlir::getOpName(op).str();
  layer_id_ = getOpLayerId(op);
  //is_inplace_layer = op->in_place();
  is_inplace_layer = false;
}

ImLayer::~ImLayer() = default;

void ImLayer::add_in_tensor(int n, int c, int h, int w, int unit_size,
                            std::string& storage , const std::string& name,
                            tensor_type_t type) {
  std::shared_ptr<Tensor> tensor =
      Tensor::register_tensor(n, c, h, w, unit_size, storage, name, type, layer_id_);
  in_tensors.push_back(tensor);
}

void ImLayer::add_in_tensor(ShapedType* shape, const std::string& name,
                            tensor_type_t type) {
  std::shared_ptr<Tensor> tensor =
      Tensor::register_tensor(shape, name, type, layer_id_);
  in_tensors.push_back(tensor);
}

void ImLayer::add_in_tensor(Value v, tensor_type_t type) {
  auto def_op = v.getDefiningOp();
  auto shape = v.getType().dyn_cast<TensorType>();
  if (def_op && !isa<tpu::NoneOp>(def_op) && !isa<tpu::WeightFileOp>(def_op)
      && !isa<ReturnOp>(def_op)) {
    if (auto load_op = dyn_cast<tpu::LoadWeightOp>(def_op)) {
      std::string name = load_op.name().str();
      std::string storage = load_op.storage().str();
      std::shared_ptr<Tensor> tensor =
          Tensor::register_tensor(&shape, name, type, layer_id_, storage);
      in_tensors.push_back(tensor);
    } else {
      std::string name = mlir::getOpName(def_op).str();
      std::shared_ptr<Tensor> tensor =
          Tensor::register_tensor(&shape, name, type, layer_id_);
      in_tensors.push_back(tensor);
    }
  }
}

void ImLayer::add_out_tensor(Value v, tensor_type_t type, std::string storage) {
  auto def_op = v.getDefiningOp();
  auto shape = v.getType().dyn_cast<TensorType>();
  if (!isa<tpu::NoneOp>(def_op) && !isa<tpu::WeightFileOp>(def_op)
      && !isa<ReturnOp>(def_op)) {
    if (auto load_op = dyn_cast<tpu::LoadWeightOp>(def_op)) {
      assert(0);
    } else {
      std::string name = mlir::getOpName(def_op).str();
      std::shared_ptr<Tensor> tensor =
          Tensor::register_tensor(&shape, name, type, layer_id_, storage);
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

static bool is_channel_align(Operation* op) {
  // ONLY shift channel offset align NPU_NUM
  std::vector<int32_t> crop_offsets;
  if (auto crop_op = dyn_cast<tpu::TG_INT8_CropOp>(op)) {
    arrayAttrToVector(crop_op.crop_offset().getValue(), crop_offsets);
  }
  else if(auto crop_op = dyn_cast<tpu::TG_BF16_CropOp>(op)) {
    arrayAttrToVector(crop_op.crop_offset().getValue(), crop_offsets);
  }
  else  {
    llvm_unreachable("unsupported op");
  }

  // offset should be n/c/h/w
  bool _is_channel_align = crop_offsets[1] % NPU_NUM == 0;
  return _is_channel_align;
}

std::shared_ptr<ImLayer> ImLayer::create(Operation* op) {
  std::shared_ptr<ImLayer> layer;
  if (isa<tpu::TG_INT8_AbsOp>(op) ||
             isa<tpu::TG_BF16_AbsOp>(op)) {
    layer = std::make_shared<ImAbs>(op);
  } else if (isa<tpu::TG_INT8_PC_Conv2DOp>(op) ||
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
  } else if (isa<tpu::TG_INT8_MatMulOp>(op) ||
             isa<tpu::TG_BF16_MatMulOp>(op)) {
    layer = std::make_shared<ImMatMul>(op);
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
    if (is_channel_align(op)) {
      layer = std::make_shared<ImCrop>(op);
    }
    else {
      LLVM_DEBUG(llvm::errs()
          << "Not support crop with channel setting, : " << getOpName(op) << "\n";);
      layer = std::make_shared<ImCommon>(op, false, IR_OTHER);
    }
  } else if (isa<tpu::TG_INT8_ReluOp>(op) || isa<tpu::TG_BF16_ReluOp>(op)) {
    layer = std::make_shared<ImRelu>(op);
  } else if (isa<tpu::TG_QuantOp>(op)) {
    layer = std::make_shared<ImCommon>(op, false, IR_OTHER);
  } else if (isa<tpu::GenericCpuOp>(op)) {
    layer = std::make_shared<ImCommon>(op, false, IR_OTHER);
  } else if (isa<tpu::TG_QuantOp>(op)) {
    layer = std::make_shared<ImQuant>(op);
  } else if (isa<tpu::QuantOp>(op) ||
             isa<tpu::InputOp>(op) ) {
    layer = std::make_shared<ImCommon>(op, true, IR_OTHER);
  } else if (isa<tpu::TG_INT8_ZeroMaskOp>(op) ||
             isa<tpu::TG_BF16_ZeroMaskOp>(op)) {
    layer = std::make_shared<ImZeroMask>(op);
  } else {
    LLVM_DEBUG(llvm::errs()
                   << "Not support ImLayer: " << getOpName(op) << "\n";);
    layer = std::make_shared<ImCommon>(op, false, IR_OTHER);
  }
  return layer;
}

std::string ImLayer::getStorage(Value v) {
  RankedTensorType _type = v.getType().cast<RankedTensorType>();
  if (_type.getElementType().isBF16()) {
    return "BF16";
  } else if (_type.getElementType().isF32()) {
    return "FP32";
  } else if (_type.getElementType().isInteger(8)) {
    return "INT8";
  } else if (_type.getElementType().isInteger(16)) {
    return "INT16";
  } else {
    assert(!"Not supported storage type.\n");
  }
  return "";
}

void ImLayer::register_it(std::shared_ptr<ImLayer>& layer) {
  layer->set_id(layers.size());
  layers.push_back(layer);
}

void ImLayer::unregister_all() { layers.clear(); }

std::vector<std::shared_ptr<ImLayer>> ImLayer::layers;

static int getOpResultUnitSize(Operation *op) {
  RankedTensorType result_type =
    op->getResult(0).getType().cast<RankedTensorType>();

  int usize = result_type.getElementTypeBitWidth()/8;
  return usize;
}

static std::string getWeightStorage(Operation *p) {
  auto op = cast<tpu::LoadWeightOp>(p);
  return op.storage().str();
}

ImConv::ImConv(Operation* p) : ImLayer(IR_CONVOLUTION, p, true) {
  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw;
  int sh, sw, pt, pb, pl, pr, dh, dw, pad_value;
  bool do_ic_align = false;
  bool fuse_leaky = false;
  bool bInt8ConvOp = isa<tpu::TG_INT8_PC_Conv2DOp>(p);
  getConvParam(p, n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, pt, pb, pl, pr,
               dh, dw, is_dw, with_bias, do_relu, do_ic_align, fuse_leaky,
               pad_value);

  if (g == 1 && ic > 4095) {
    // hw limitation: ic should be smaller than 4096, otherwise
    // we need to split ic and output fp32 patial sum tensor,
    // which occupies too much memory. it has no benefit to
    // do fusion in such case.
    fusible = false;
  } else if (g > 1 && false == is_dw) {
    // for group conv
    // if oc / g > 32, then we will have two bias at one lane without
    // EU_NUM align,
    // so we can only specify the align type to bias memory layout
    // but skip the oc/g>32 cases.
    if (oc/g > (int)NPU_NUM)
      fusible = false;
  }

  int w_ic = ic;
  if (do_ic_align && (ic % 2 != 0)) {
    w_ic += 1;
  }

  // add input tensor
  add_in_tensor(p->getOperand(0), TENSOR_NEURON);

  // add weight tensor
  auto weightOp = cast<tpu::LoadWeightOp>(p->getOperand(1).getDefiningOp());
  std::string weightOpName = weightOp.name().str();
  int32_t unit_size = getOpResultUnitSize(weightOp);

  // get is dilate activation
  bool is_ins = false;
  if (isa<tpu::TG_INT8_PC_Conv2DOp>(p)) {
    auto op = dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(p);
    std::vector<int32_t> ins;
    arrayAttrToVector(op.param().ins(), ins);
    is_ins = !ins.empty();
  }

  if (is_ins) {
    // ins mode cant slice h/w
    fusible = false;
  }

  std::string weight_storage = getWeightStorage(weightOp);
  if (is_dw) {
    add_in_tensor(1, oc, kh, kw, unit_size,
                  weight_storage, weightOpName, TENSOR_DEPTHCONV_OPD1);
  }
  else {
    // tensor shape in local memory should be (1, oc, kh*kw, ic/g)
    add_in_tensor(w_ic / g, oc, kh, kw, unit_size,
                 weight_storage, weightOpName, TENSOR_COEFF);
  }

  // add bias tensor
  if (bInt8ConvOp) {
    int perchannel_size = with_bias ? 9 : 5;
    auto load_bias = cast<tpu::LoadWeightOp>(p->getOperand(2).getDefiningOp());
    std::string bias_name = load_bias.name().str();
    std::string bias_storage = getWeightStorage(load_bias);
    int bias_usize = getOpResultUnitSize(load_bias);

    if (is_dw)
      add_in_tensor(1, oc, 1, perchannel_size, bias_usize,
                    bias_storage, bias_name, TENSOR_BIAS);
    else {
      // if is group conv, bias need to align.
      tensor_type_t bias_type = (g > 1) ? TENSOR_DEPTHCONV_OPD1 : TENSOR_BIAS;
      add_in_tensor(1, oc, 1, perchannel_size, bias_usize,
                    bias_storage, bias_name, bias_type);
    }
  } else if (!bInt8ConvOp && with_bias) {
    // bf16 with bias
    auto load_bias = cast<tpu::LoadWeightOp>(p->getOperand(2).getDefiningOp());
    std::string bias_name = load_bias.name().str();
    std::string bias_storage = "UINT16";
    int bias_usize = 2;

    add_in_tensor(2, oc, 1, 1, bias_usize,
                  bias_storage, bias_name, TENSOR_BIAS);

  }

  // add out tensor
  add_out_tensor(p->getResult(0), TENSOR_NEURON);
  if (fuse_leaky) {
    add_imm_tensor(out_tensors[0], 1, name_ + "_imm");
  }
}

ImDeconv::ImDeconv(Operation* p) : ImLayer(IR_DECONVOLUTION, p, true) {
  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw;
  int sh, sw, pt, pb, pl, pr, dh, dw;
  int pad_value;
  bool do_ic_align, do_leaky_relu;
  bool bInt8ConvOp = isa<tpu::TG_INT8_PC_DeConv2DOp>(p);
  getConvParam(p, n, ic, ih, iw, oc, oh, ow,
                 g, kh, kw, sh, sw,
                 pt, pb, pl, pr, dh, dw,
                 is_dw, with_bias,
                 do_relu, do_ic_align, do_leaky_relu, pad_value);

  // handle ic align for double conv
  int w_ic = ic;
  if (do_ic_align && (ic % 2 != 0)) {
    w_ic += 1;
  }
  // add input tensor
  add_in_tensor(p->getOperand(0), TENSOR_NEURON);

  // add weight tensor
  auto weightOp = cast<tpu::LoadWeightOp>(p->getOperand(1).getDefiningOp());
  std::string weightOpName = weightOp.name().str();
  int32_t unit_size = getOpResultUnitSize(weightOp);
  std::string weight_storage = getWeightStorage(weightOp);

  if (is_dw) {
    add_in_tensor(1, oc, kh, kw, unit_size, weight_storage,
                  weightOpName, TENSOR_DEPTHCONV_OPD1);
  } else {
    // tensor shape in local memory should be (1, oc, kh*kw, ic/g)
    add_in_tensor(w_ic / g, oc, kh, kw, unit_size, weight_storage,
                  weightOpName, TENSOR_COEFF);
  }

  // add bias tensor
  if (bInt8ConvOp) {
    int perchannel_size = with_bias ? 9 : 5;
    auto load_bias = cast<tpu::LoadWeightOp>(p->getOperand(2).getDefiningOp());
    std::string bias_name = load_bias.name().str();
    std::string bias_storage = getWeightStorage(load_bias);
    int bias_usize = getOpResultUnitSize(load_bias);

    if (is_dw) {
      add_in_tensor(1, oc, 1, perchannel_size, bias_usize,
                    bias_storage, bias_name, TENSOR_BIAS);
    } else {
      // if is group conv, bias need to align.
      tensor_type_t bias_type = (g > 1) ? TENSOR_DEPTHCONV_OPD1 : TENSOR_BIAS;
      add_in_tensor(1, oc, 1, perchannel_size, bias_usize,
                    bias_storage, bias_name, bias_type);
    }
  } else if(!bInt8ConvOp && with_bias) {
    // bf16 with bias
    auto load_bias = cast<tpu::LoadWeightOp>(p->getOperand(2).getDefiningOp());
    std::string bias_name = load_bias.name().str();
    std::string bias_storage = "UINT16";
    int bias_usize = 2;

    if (is_dw)
      add_in_tensor(2, oc, 1, 1, bias_usize,
                    bias_storage, bias_name, TENSOR_BIAS);
    else
      add_in_tensor(g*2, oc/g, 1, 1, bias_usize,
                    bias_storage, bias_name, TENSOR_BIAS);
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
  add_in_tensor(op->getOperand(1), TENSOR_COEFF);

  // if bias is not noneop
  if (!isa<tpu::NoneOp>(op->getOperand(2).getDefiningOp())) {
    auto load_bias = cast<tpu::LoadWeightOp>(op->getOperand(2).getDefiningOp());
    auto opd_type = op->getOperand(2).getType().dyn_cast<TensorType>();
    std::vector<int64_t> shape = opd_type.getShape();
    int bias_usize = getOpResultUnitSize(load_bias);
    std::string storage = getWeightStorage(load_bias);
    add_in_tensor(2, 0, 0, shape[0], bias_usize, storage, name_ + "_bias", TENSOR_BIAS);
  }
  add_out_tensor(op->getResult(0), TENSOR_MATRIX);
}

ImMatMul::ImMatMul(Operation* op) : ImLayer(IR_MATMUL, op) {
  add_in_tensor(op->getOperand(0), TENSOR_NEURON);
  add_in_tensor(op->getOperand(1), TENSOR_NEURON);
  add_out_tensor(op->getResult(0), TENSOR_MATRIX);
}

ImEltwise::ImEltwise(Operation* op) : ImLayer(IR_ELTWISE, op, true) {
  uint32_t nInputs = op->getNumOperands();
  for (uint32_t i = 0; i < nInputs; ++i) {
    //
    bool isCoeffLoad = isa<tpu::LoadWeightOp>(op->getOperand(i).getDefiningOp());
    if (isCoeffLoad) {
      // not support weight to split now
      fusible = false;
    }
    tensor_type_t t_type = isCoeffLoad ? TENSOR_COEFF : TENSOR_NEURON;
    add_in_tensor(op->getOperand(i), t_type);
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
  uint32_t nInputs = op->getNumOperands();
  for (uint32_t i = 0; i < nInputs; ++i) {
    if (BlockArgument::classof(op->getOperand(i))) {
      auto shape = op->getResult(0).getType().dyn_cast<TensorType>();
      add_in_tensor(&shape, "arg0", TENSOR_NEURON);
    } else {
      add_in_tensor(op->getOperand(i),TENSOR_NEURON);
    }
  }

  for (uint32_t i = 0; i < op->getNumResults(); ++i) {
    add_out_tensor(op->getResult(i), TENSOR_NEURON, getStorage(op->getResult(i)));
  }
}

ImConcat::ImConcat(Operation* op) : ImLayer(IR_CONCAT, op, true) {
  // only support axis = 1 for fuse
  int axis = 0;
  bool do_relu = false;

  getConcatParam(op, axis, do_relu);

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
  auto load_y_table = cast<tpu::LoadWeightOp>(op->getOperand(1).getDefiningOp());
  int usize = getOpResultUnitSize(load_y_table);
  std::string storage = getWeightStorage(load_y_table);
  std::string y_table_name = load_y_table.name().str();
  add_in_tensor(1, NPU_NUM, table_h, table_w, usize, storage, y_table_name, TENSOR_COEFF_LUT);
  add_out_tensor(op->getResult(0), TENSOR_NEURON);

  // add m_table
  if (isBF16) {
    // FIXME: support other bf16 activation ops
    auto load_m_table = cast<tpu::LoadWeightOp>(op->getOperand(2).getDefiningOp());
    int usize = getOpResultUnitSize(load_m_table);
    std::string storage = getWeightStorage(load_m_table);
    std::string m_table_name = load_m_table.name().str();
    add_in_tensor(1, NPU_NUM, table_h, table_w, usize, storage, m_table_name, TENSOR_COEFF_LUT);

    // add working table
    add_imm_tensor(out_tensors[0], 2, name_ + "_imm");
  }
}

ImQuant::ImQuant(Operation *op) : ImLayer(IR_QUANT, op, true) {
  auto quantOp = cast<tpu::TG_QuantOp>(op);
  std::string from = quantOp.from().str();
  std::string to = quantOp.to().str();
  if (quantOp.threshold().hasValue() == false) {
    fusible = false;
  } else if ((from == "INT8" || from == "UINT8") && to == "BF16") {
  } else if (from == "BF16" && to == "INT8") {
  } else {
    fusible = false;
  }
  add_in_tensor(op->getOperand(0), TENSOR_NEURON);
  add_out_tensor(op->getResult(0), TENSOR_NEURON);
}

ImPRelu::ImPRelu(Operation* op) : ImLayer(IR_PRELU, op, true) {
  add_in_tensor(op->getOperand(0), TENSOR_NEURON);
  add_in_tensor(op->getOperand(1), TENSOR_DEPTHCONV_OPD1);
  add_out_tensor(op->getResult(0), TENSOR_NEURON);
}

// shufflechannel as tg layer
ImShuffleChannel::ImShuffleChannel(Operation *op): ImLayer(IR_SHUFFLECHANNEL, op, false) {
  add_in_tensor(op->getOperand(0), TENSOR_NEURON);
  add_out_tensor(op->getResult(0), TENSOR_NEURON);
}

ImSlice::ImSlice(Operation *op): ImLayer(IR_SLICE, op, true) {
  std::vector<int64_t> dst_shape = getTensorShape(op->getResult(0));
  int axis = 0;
  getSliceParam(op, axis);

  // optimization for batch 1, set as inplace layer
  is_inplace_layer = true;
  for (int i = 0; i < axis; i++) {
    if (dst_shape[i] != 1) {
      is_inplace_layer = false;
      break;
    }
  }
  add_in_tensor(op->getOperand(0), TENSOR_NEURON);
  add_out_tensor(op->getResult(0), TENSOR_NEURON);
}

ImLrn::ImLrn(Operation *op): ImLayer(IR_LRN, op, true) {
  add_in_tensor(op->getOperand(0), TENSOR_NEURON);
  add_out_tensor(op->getResult(0), TENSOR_NEURON);

  int working_size = 5;
  int table_h = 16;
  int table_w = 16;
  if (isa<tpu::TG_BF16_LrnOp>(op)) {
    working_size = 2;
    table_h = 32;
    table_w = 8;
  }

  // add sqr weight
  auto load_sqr = cast<tpu::LoadWeightOp>(op->getOperand(1).getDefiningOp());
  int usize = getOpResultUnitSize(load_sqr);
  std::string storage = getWeightStorage(load_sqr);
  std::string sqr_name = load_sqr.name().str();
  add_in_tensor(1, NPU_NUM, table_h, table_w, usize, storage, sqr_name, TENSOR_COEFF_LUT);

  // add power weight
  auto load_pow = cast<tpu::LoadWeightOp>(op->getOperand(2).getDefiningOp());
  usize = getOpResultUnitSize(load_pow);
  storage = getWeightStorage(load_pow);
  std::string pow_name = load_pow.name().str();
  add_in_tensor(1, NPU_NUM, table_h, table_w, usize, storage, pow_name, TENSOR_COEFF_LUT);

  add_imm_tensor(in_tensors[0], working_size, name_ + "_imm");
}

ImAbs::ImAbs(Operation *op): ImLayer(IR_ABS, op, true) {
  add_in_tensor(op->getOperand(0), TENSOR_NEURON);
  add_out_tensor(op->getResult(0), TENSOR_NEURON);
}

ImBroadcastMul::ImBroadcastMul(Operation *op): ImLayer(IR_BROADCAST_MUL, op, true) {
  auto input_type = op->getOperand(0).getType().dyn_cast<TensorType>();
  bool isInt8Op = isa<tpu::TG_INT8_BroadcastMulOp>(op);
  auto input_shape = input_type.getShape();
  add_in_tensor(op->getOperand(0), TENSOR_NEURON);
  add_in_tensor(op->getOperand(1), TENSOR_NEURON);
  add_out_tensor(op->getResult(0), TENSOR_NEURON);

  // add bias tensor
  if (isInt8Op) {
    bool with_bias = false;
    int perchannel_size = with_bias ? 9 : 5;
    auto load_bias = cast<tpu::LoadWeightOp>(op->getOperand(2).getDefiningOp());
    std::string bias_name = load_bias.name().str();
    std::string bias_storage = getWeightStorage(load_bias);
    int bias_usize = getOpResultUnitSize(load_bias);
    add_in_tensor(input_shape[0], input_shape[1], 1, perchannel_size, bias_usize, bias_storage,
                  bias_name, TENSOR_BIAS);
  }
}

ImUpsample::ImUpsample(Operation *op): ImLayer(IR_UPSAMPLE, op, true) {
  int scale_h = 0;
  int scale_w = 0;
  getUpsampleParam(op, scale_h, scale_w);
  // ins_h/ins_w can not exceed 16 for average pooling in tl_upsample
  // which has only 4 bits in hw
  if (scale_h >= 16 || scale_w >= 16)
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

ImZeroMask::ImZeroMask(Operation *op): ImLayer(IR_ZERO_MASK, op, true) {
  add_in_tensor(op->getOperand(0), TENSOR_NEURON);
  add_out_tensor(op->getResult(0), TENSOR_NEURON);
  if(isa<tpu::TG_INT8_ZeroMaskOp>(op)) {
    add_imm_tensor(out_tensors[0], 1, name_ + "_imm");
  }
}
}
