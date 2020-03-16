
#include "ImLayer.hpp"


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

void ImLayer::add_in_tensor(int n, int c, int h, int w, int unit_size, const string& name,
                            tensor_type_t type, gaddr_t gaddr) {
  shared_ptr<Tensor> tensor = Tensor::register_tensor(n, c, h, w, unit_size, name, type, gaddr);
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
      string name = load_op.name().getValue();
      shared_ptr<Tensor> tensor = Tensor::register_tensor(&shape, name, TENSOR_COEFF, gaddr);
      in_tensors.push_back(tensor);
    } else {
      string name = mlir::getOpName(def_op);
      shared_ptr<Tensor> tensor = Tensor::register_tensor(&shape, name, type, gaddr);
      in_tensors.push_back(tensor);
    }
  }
}

void ImLayer::add_out_tensor(int n, int c, int h, int w, int unit_size, const string& name,
                             tensor_type_t type, gaddr_t gaddr) {
  shared_ptr<Tensor> tensor = Tensor::register_tensor(n, c, h, w, unit_size, name, type, gaddr);
  out_tensors.push_back(tensor);
}

void ImLayer::add_out_tensor(ShapedType* shape, const string& name, tensor_type_t type,
                             gaddr_t gaddr) {
  shared_ptr<Tensor> tensor = Tensor::register_tensor(shape, name, type, gaddr);
  out_tensors.push_back(tensor);
}

void ImLayer::add_out_tensor(Value * v, tensor_type_t type, gaddr_t gaddr) {
  auto def_op = v->getDefiningOp();
  auto shape = v->getType().dyn_cast<TensorType>();
  if (!isa<tpu::NoneOp>(def_op) && !isa<tpu::WeightFileOp>(def_op)) {
    if (auto load_op = dyn_cast<tpu::LoadWeightOp>(def_op)) {
      string name = load_op.name().getValue();
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



// tpu::BroadcastMulOp,
// tpu::CropOp,
// tpu::DeConv2DOp,
// tpu::DequantizationOp,
// tpu::DetectionOutputOp,
// tpu::DivOp,
// tpu::LeakyReluOp,
// tpu::NormalizeOp,
// tpu::PReluOp,
// tpu::PermuteOp,
// tpu::PowerOp,
// tpu::PriorBoxOp,
// tpu::QuantizationOp,
// tpu::ReluOp,
// tpu::ShuffleChannelOp,
// tpu::SliceOp,
// tpu::SoftmaxOp,

shared_ptr<ImLayer> ImLayer::create(Operation* op) {
  shared_ptr<ImLayer> layer;
  if (isa<tpu::Conv2DOp>(op)) {
    layer = make_shared<ImConv>(op);
  } else if (isa<tpu::EltwiseAddOp>(op) ||
             isa<tpu::EltwiseMaxOp>(op)  ||
             isa<tpu::EltwiseMulOp>(op)) {
    layer = make_shared<ImEltwise>(op);
  } else if (isa<tpu::FullyConnectedOp>(op)) {
    layer = make_shared<ImInnerproduct>(op);
  } else if (isa<tpu::ReshapeOp>(op)){
    layer = make_shared<ImCommon>(op, true, IR_OTHER);
  } else if (isa<tpu::BatchNormOp>(op)) {
    layer = make_shared<ImBatchnorm>(op);
  } else if (isa<tpu::PoolAvg2DOp>(op) ||
             isa<tpu::PoolMax2DOp>(op)) {
    layer = make_shared<ImPooling>(op);
  } else if (isa<tpu::ScaleOp>(op)) {
    layer = make_shared<ImScale>(op);
  } else if (isa<tpu::ConcatOp>(op)) {
    layer = make_shared<ImConcat>(op);
  } else if (isa<tpu::TanHOp>(op) ||
             isa<tpu::SigmoidOp>(op) ||
             isa<tpu::SqrtOp>(op)) {
    layer = make_shared<ImActivation>(op);
  } else if (isa<tpu::SoftmaxOp>(op)) {
    layer = make_shared<ImCommon>(op, false, IR_OTHER);
  } else if (isa<tpu::QuantOp>(op) ||
             isa<tpu::InputOp>(op)) {
    layer = make_shared<ImCommon>(op, true, IR_OTHER);
  } else {
    llvm::errs() << "Not support ImLayer: " << getOpName(op) << "\n";
    assert(0);
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

static int getOperandStoargeSize(Operation *p) {
  auto op = cast<tpu::LoadWeightOp>(p);

  if (op.storage() == "INT8" || op.storage() == "UINT8" ) {
    return 1;
  } else if (op.storage() == "BF16" || op.storage() == "INT16" ||
             op.storage() == "UINT16" ) {
    return 2;
  } else if(op.storage() == "FP32" || op.storage() == "INT32" ||
            op.storage() == "UINT32") {
    return 4;
  }
}

ImConv::ImConv(Operation* p) : ImLayer(IR_CONVOLUTION, p, true), conv1x1_to_fc(false) {
  auto op = dyn_cast<tpu::Conv2DOp>(p);
  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, ph, pw, dh, dw;
  bool is_deconv = isa<tpu::DeConv2DOp>(op.getOperation());
  parseConvParam(op.param(), is_deconv, op.input(), op.output(), op.filter(),
                  n, ic, ih, iw, oc, oh, ow, g,
                  kh, kw, sh, sw, ph, pw, dh, dw, is_dw, with_bias, do_relu);

  // add input tensor
  add_in_tensor(op.input(), TENSOR_NEURON);

  // add weight tensor
  auto weightOp = cast<tpu::LoadWeightOp>(op.filter()->getDefiningOp());
  uint32_t unit_size = getOperandStoargeSize(weightOp);
  if (is_dw) {
    add_in_tensor(1, oc, kh, kw, unit_size, name_ + "_weight", TENSOR_DEPTHCONV_OPD1);
  }
  else {
    add_in_tensor(ic, oc, kh, kw, unit_size, name_ + "_weight", TENSOR_COEFF);
  }

  // add bias tensor
  if (with_bias) {
    auto load_bias = cast<tpu::LoadWeightOp>(op.getOperand(2)->getDefiningOp());
    int bias_usize = getOperandStoargeSize(load_bias);
    if (g == oc && g == ic && g != 1) {
      add_in_tensor(2, oc, 1, 1, bias_usize, name_ + "_bias", TENSOR_BIAS);
    } else {
      int group_oc = oc / g;
      assert(oc % g == 0);
      for (int i = 0; i < g; i++) {
        shared_ptr<Tensor> tensor = Tensor::register_tensor(
            2, oc / g, 1, 1, bias_usize, name_ + "_bias_" + std::to_string(i),
            TENSOR_BIAS, i * group_oc * bias_usize);
        tensor.get()->group = g;
        in_tensors.push_back(tensor);
      }
    }
  }

  // add out tensor
  add_out_tensor(op.output(), TENSOR_NEURON);

  if (do_relu) {
    add_imm_tensor(out_tensors[0], 1, name_ + "_imm");
  }
}

ImPooling::ImPooling(Operation* op) : ImLayer(IR_POOLING, op, true) {
  add_in_tensor(op->getOperand(0), TENSOR_NEURON);
  add_out_tensor(op->getResult(0), TENSOR_NEURON);
}

ImInnerproduct::ImInnerproduct(Operation* op) : ImLayer(IR_INNERPRODUCT, op) {

  add_in_tensor(op->getOperand(0), TENSOR_NEURON);

  // weight
  auto s_type = op->getOperand(1)->getType().dyn_cast<TensorType>();
  add_in_tensor(&s_type, name_ + "_weight", TENSOR_COEFF);

  // if bias is not noneop
  if (!isa<tpu::NoneOp>(op->getOperand(2)->getDefiningOp())) {
    auto opd_type = op->getOperand(2)->getType().dyn_cast<TensorType>();
    std::vector<int64_t> shape = opd_type.getShape();
    add_in_tensor(2, 0, 0, shape[0], 2, name_ + "_bias", TENSOR_BIAS);
  }
  add_out_tensor(op->getResult(0), TENSOR_MATRIX);
}

ImEltwise::ImEltwise(Operation* op) : ImLayer(IR_ELTWISE, op, true) {
  for (u32 i = 0; i < op->getNumOperands(); ++i) {
    add_in_tensor(op->getOperand(i), TENSOR_NEURON);
  }

  add_out_tensor(op->getResult(0), TENSOR_NEURON);

  add_imm_tensor(in_tensors[0], 1, name_ + "_imm");
}

ImBatchnorm::ImBatchnorm(Operation* op) : ImLayer(IR_BATCHNORM, op, true) {
  auto input_type = op->getOperand(0)->getType().dyn_cast<TensorType>();
  std::vector<int64_t> input_shape = input_type.getShape();
  add_in_tensor(op->getOperand(0), TENSOR_NEURON);

  add_in_tensor(2, input_shape[1], 1, 1, DATA_TYPE_SIZE,
                name_ + "_mean_param", TENSOR_BIAS);

  add_in_tensor(1, input_shape[1], 1, 1, DATA_TYPE_SIZE,
                name_ + "_variance_param", TENSOR_COEFF_NEURON);

  add_out_tensor(op->getResult(0), TENSOR_NEURON);
}

ImCommon::ImCommon(Operation* op, bool inplace_compute, IR_TYPE type) : ImLayer(type, op) {
  is_inplace_layer = (is_inplace_layer || inplace_compute);

  for (u32 i = 0; i < op->getNumOperands(); ++i) {
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

ImScale::ImScale(Operation* op) : ImLayer(IR_SCALE, op, true) {
  auto opd_type = op->getOperand(1)->getType().dyn_cast<TensorType>();
  std::vector<int64_t> opd_shape = opd_type.getShape();
  tensor_type_t tensor_type = TENSOR_COEFF_NEURON;
  if (isa<tpu::LoadWeightOp>(op->getOperand(1)->getDefiningOp()))
    tensor_type = TENSOR_NEURON_AS_COEFF;

  // add input 0
  add_in_tensor(op->getOperand(0), TENSOR_NEURON);

  // add input 1 or weight
  add_in_tensor(1, opd_shape[0], 1, 1, DATA_TYPE_SIZE,
                name_ + "_input1", tensor_type);

  if (op->getNumOperands() == 3) {
    add_in_tensor(2, opd_shape[0], 1, 1, DATA_TYPE_SIZE,
                  name_ + "_bias_factor", TENSOR_BIAS);
  }

  add_out_tensor(op->getResult(0), TENSOR_NEURON);
}


ImConcat::ImConcat(Operation* op) : ImLayer(IR_CONCAT, op) {
  for (u32 i = 0; i < op->getNumOperands(); ++i) {
      add_in_tensor(op->getOperand(i),TENSOR_NEURON);
  }

  for (u32 i = 0; i < op->getNumResults(); ++i) {
    add_out_tensor(op->getResult(i), TENSOR_NEURON);
  }
}

ImActivation::ImActivation(Operation* op) : ImLayer(IR_ACTIVATION, op, true) {
  if (isa<tpu::SigmoidOp>(op) || isa<tpu::TanHOp>(op)) {
    this->fusible = false;
  }

  add_in_tensor(op->getOperand(0), TENSOR_NEURON);

  if (isa<tpu::PReluOp>(op) && (op->getNumOperands() == 2 )) {
    auto opd_type = op->getOperand(0)->getType().dyn_cast<TensorType>();
    std::vector<int64_t> opd_shape = opd_type.getShape();
    add_in_tensor(1, opd_shape[1], 1, 1, DATA_TYPE_SIZE,
                  name_ + "_slope_param", TENSOR_COEFF_NEURON);
  }

  add_out_tensor(op->getResult(0), TENSOR_NEURON);
}


// ImMac::ImMac(Operation* op) : ImLayer(IR_MAC, op, true) {
//   const TGMacParameter& param = op->tg_mac_param();
//   const TensorShape& input_shape = op->input_shape(0);
//   const TensorShape& output_shape = op->output_shape(0);
//   int unit_size = input_shape.data_type_size();
//   int c = input_shape.dim(1);

//   if (param.do_activation() && param.activation() == RELU && param.activation_arg_size() > 0 &&
//       param.activation_arg(0) != 0.0f) {
//     is_inplace_layer = false;
//   } else {
//     is_inplace_layer = true;
//   }

//   add_in_tensor(input_shape, op->bottom(0), TENSOR_NEURON);
//   add_in_tensor(
//       1, c, 1, 1, unit_size, name_ + "_scale",
//       TENSOR_COEFF_NEURON, param.global_scale());
//   add_in_tensor(
//       1, c, 1, 1, unit_size, name_ + "_bias",
//       TENSOR_COEFF_NEURON, param.global_bias());
//   add_out_tensor(output_shape, op->top(0), TENSOR_NEURON);
// }


// ImLrn::ImLrn(Operation* op) : ImLayer(IR_LRN, op, true) {
//   const TGLRNParameter& param = op->tg_lrn_param();

//   add_in_tensor(op->input_shape(0), op->bottom(0), TENSOR_NEURON);
//   add_out_tensor(op->output_shape(0), op->top(0), TENSOR_NEURON);

//   if (CHIP_IS_BM1880) {
//     // 1880 has 32 lanes(npu num)
//     add_in_tensor(1, 32, 16, 16, 1, name_ + "sqr_lut_weight", TENSOR_DEPTHCONV_OPD1,
//                   param.sqr_lut_weight());
//     add_in_tensor(1, 32, 16, 16, 1, name_ + "power_lut_weight", TENSOR_DEPTHCONV_OPD1,
//                   param.power_lut_weight());
//     add_imm_tensor(in_tensors[0], 5, name_ + "_imm");
//   } else if (CHIP_IS_BM1880V2) {
//     // 1880v2 has 32 lanes(npu num)
//     // Use coeff because lookup table can not be sliced
//     add_in_tensor(1, 32, 16, 16, 1, name_ + "sqr_lut_weight", TENSOR_COEFF,
//                   param.sqr_lut_weight());
//     add_in_tensor(1, 32, 16, 16, 1, name_ + "power_lut_weight", TENSOR_COEFF,
//                   param.power_lut_weight());
//     add_imm_tensor(in_tensors[0], 5, name_ + "_imm");
//   } else {
//     add_imm_tensor(in_tensors[0], 1, name_ + "_imm");
//   }
// }



// ImUpsample::ImUpsample(Operation* op) : ImLayer(IR_UPSAMPLE, op, true) {
//   add_in_tensor(op->input_shape(0), op->bottom(0), TENSOR_NEURON);
//   add_out_tensor(op->output_shape(0), op->top(0), TENSOR_NEURON);
// }

// ImDeconv::ImDeconv(Operation* op) : ImLayer(IR_DECONVOLUTION, op, true) {
//   const TGConvolutionParameter& param = op->tg_convolution_param();
//   const TensorShape& input_shape = op->input_shape(0);
//   const TensorShape& output_shape = op->output_shape(0);
//   int unit_size = input_shape.data_type_size();
//   int input_c = input_shape.dim(1);
//   int output_c = output_shape.dim(1);
//   int kh = param.kernel_size(0);
//   int kw = param.kernel_size(1);
//   int groups = param.group();
//   bool has_bias = param.bias_term();

//   add_in_tensor(input_shape, op->bottom(0), TENSOR_NEURON);

//   add_in_tensor(input_c / groups, output_c, kh, kw, unit_size, name_ + "_weight", TENSOR_COEFF,
//                 param.global_weight());

//   if (has_bias) {
//     if (CHIP_IS_BM188X) {
//       int group_oc = output_c / groups;
//       assert(output_c % groups == 0);
//       for (int i = 0; i < groups; i++) {
//         shared_ptr<Tensor> tensor = Tensor::register_tensor(
//             2, output_c / groups, 1, 1, unit_size, name_ + "_bias_" + std::to_string(i),
//             TENSOR_BIAS, param.global_bias() + i * group_oc * unit_size);
//         tensor.get()->group = groups;
//         in_tensors.push_back(tensor);
//       }
//     } else {
//       add_in_tensor(1, output_c, 1, 1, unit_size, name_ + "_bias", TENSOR_COEFF,
//                     param.global_bias());
//     }
//   }

//   if (CHIP_IS_BM188X && FlagInst::get()->flagOpt(hwflags::OPT::SUPPORT_BIAS_INT32)) {
//     // <! FIXME: tiling output_c and not hard code 9, 5
//     add_in_tensor(1, output_c, 1, (has_bias ? 9 : 5), 1, name_ + "multi_channel", TENSOR_BIAS,
//         param.global_per_channel());
//   }

//   add_out_tensor(output_shape, op->top(0), TENSOR_NEURON);
// }

// ImShuffleChannel::ImShuffleChannel(Operation* op) : ImLayer(IR_SHUFFLECHANNEL, op, true) {
//   const TensorShape& input_shape = op->input_shape(0);
//   const TensorShape& output_shape = op->output_shape(0);

//   add_in_tensor(input_shape, op->bottom(0), TENSOR_NEURON);
//   add_out_tensor(output_shape, op->top(0), TENSOR_NEURON);
// }

// ImArithmetic::ImArithmetic(Operation* op) : ImLayer(IR_ARITHMETIC, op, true) {
//   const TGArithmeticParameter& param = op->tg_arithmetic_param();
//   const TensorShape& input_shape = op->input_shape(0);
//   const TensorShape& output_shape = op->output_shape(0);
//   const TensorShape& b_shape = param.b_shape();
//   int unit_size = input_shape.data_type_size();

//   is_inplace_layer = false;

//   add_in_tensor(input_shape, op->bottom(0), TENSOR_NEURON);

//   if (CHIP_IS_BM188X) {
//     // if mul, fallback to tg
//     if (param.operation() == TGArithmeticParameter_ArithmeticOp_MUL) {
//       this->fusible = false;
//     } else if (param.operation() == TGArithmeticParameter_ArithmeticOp_ADD) {
//       this->fusible = false;
//     } else {
//       std::cout << "Arithmetric op for 1880 only supports mul and add currently!\n";
//       assert(0);
//     }
//   }

//   // FIXME: Is unit_size needed?
//   if (op->global_input_size() == 2) {
//     add_in_tensor(b_shape, op->bottom(1), TENSOR_NEURON, param.global_b());

//   } else if (op->global_input_size() == 1 && param.is_b_const() == false) {
//     add_in_tensor(b_shape, name_ + "_global_b_param", TENSOR_COEFF, param.global_b());
//   }

//   add_out_tensor(output_shape, op->top(0), TENSOR_NEURON);
// }

// ImQuantization::ImQuantization(Operation* op) : ImLayer(IR_QUANTIZATION, op, true) {
//   const TGQuantizationParameter& param = op->tg_quantization_param();
//   const TensorShape& input_shape = op->input_shape(0);
//   const TensorShape& output_shape = op->output_shape(0);

//   // inplace if not dump-all-neuron
//   is_inplace_layer = false;

//   add_in_tensor(input_shape, op->bottom(0), TENSOR_NEURON);

//   // FIXME:
//   if (!CHIP_IS_BM188X) {
//     this->fusible = false;
//   }

//   add_out_tensor(output_shape, op->top(0), TENSOR_NEURON);
// }


}