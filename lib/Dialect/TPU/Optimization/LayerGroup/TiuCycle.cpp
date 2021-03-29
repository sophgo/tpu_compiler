#include "TiuCycle.hpp"
#include <algorithm>
#include <cmath>
#include "NetGraph.hpp"
#include "Group.hpp"
#include "MixNet.hpp" // get_op_from_name
#include "mlir/IR/BuiltinTypes.h" // BuiltinTypes.h

#define DEBUG_TYPE "tiu_cycle"

#define BM_ASSERT(x) assert(0 && x)
#define W_IN(w) (((float)(w - 1) * inst.conv_op_x_str) / (inst.conv_opd0_x_ins0 + 1) + 1)

namespace mlir {

void TiuCycle::setup_hw_config() {
    FuncOp * fn = net_graph_->getFn();
    std::string chipname = "cx1835";
    if ((*fn)->getAttr("chipname")) {
      chipname = (*fn)->getAttr("chipname").cast<StringAttr>().getValue().str();
    }
    if (chipname == "cv183x") {
      // 1835 config
      tpu_frequency_ = 650;
    } else if (chipname == "cv182x") {
      // 1822 config
      tpu_frequency_ = 650;
    } else {
      assert(!"chip setting not configed.\n");
    }
}

int TiuCycle::get_cycle(int cur_layer) {
  layer_id_ = cur_layer;
  const ImLayer* im_layer = net_graph_->get_layer_by_id(cur_layer);
  op = im_layer->op();

  const std::vector<int>& in_tensors = net_graph_->get_in_tensors_of_layer(layer_id_);
  const std::vector<int>& out_tensors = net_graph_->get_out_tensors_of_layer(layer_id_);

  int bottom_dim[4];
  int top_dim[4];

  net_graph_->get_tensor_dim(in_tensors[0], bottom_dim);
  net_graph_->get_tensor_dim(out_tensors[0], top_dim);
  Tensor* in_tensor = net_graph_->get_tensor_by_id(in_tensors[0]);
  Tensor* out_tensor = net_graph_->get_tensor_by_id(out_tensors[0]);

  // set default value
  isPerChannelQuan = false;
  auto input_type = op->getOperand(0).getType().cast<RankedTensorType>();
  isBfloat16 = input_type.getElementType().isBF16();
  inst.tsk_opd_num = op->getNumOperands();
  inst.res0_addr = 0;

  // set tiled
  bottom_dim[0] = in_tensor->n_slice;
  bottom_dim[2] = in_tensor->h_slice;

  top_dim[0] = out_tensor->n_slice;
  top_dim[2] = out_tensor->h_slice;

  // default inputs
  inst.opd0_n = bottom_dim[0];
  inst.opd0_c = bottom_dim[1];
  inst.opd0_h = bottom_dim[2];
  inst.opd0_w = bottom_dim[3];

  inst.opd1_n = bottom_dim[0];
  inst.opd1_c = bottom_dim[1];
  inst.opd1_h = bottom_dim[2];
  inst.opd1_w = bottom_dim[3];

  // default output
  inst.res0_n = top_dim[0];
  inst.res0_c = top_dim[1];
  inst.res0_h = top_dim[2];
  inst.res0_w = top_dim[3];

  inst.tsk_eu_typ = 0;
  inst.conv_opd1_x_ins0 = 0;
  inst.conv_opd1_y_ins0 = 0;

  inst.opt_opd0_int8 = isBfloat16;
  // TODO: check int16 case
  inst.opt_res0_int8 =
      op->getResult(0).getType().cast<RankedTensorType>().getElementType().isBF16();
  // TODO: check opt_opd1_const
  inst.opt_opd1_const = 0;

  // TODO: support mul matrix
  inst.tsk_typ = TensorArithmetic;
  // TODO: breakdown non-atomic
  inst.tsk_eu_typ = 0;

  // NOTICE: update under conv/depthwise
  inst.res0_c_str = 16;
  inst.tens_mdsum = 0;

  // copy from MixNet.cpp
  IR_TYPE layer_type = im_layer->type();
  bool is_support = true;

  switch (layer_type) {
    case IR_CONVOLUTION:
      set_tl_conv_param();
      break;
    case IR_DECONVOLUTION:
      set_tl_deconv_param();
      break;
    case IR_ELTWISE:
      set_tl_eltwise_param();
      break;
    case IR_POOLING:
      inst.tsk_typ = Pooling;
      set_tl_pooling_param();
      break;
    case IR_RELU:
    case IR_PRELU:
    case IR_LEAKY_RELU:
      break;
    case IR_SCALE:
      set_tl_scale_param();
      break;
    case IR_ACTIVATION:
      set_tl_activation_param();
      break;
    // tdma operation involved
    case IR_PAD:
      set_tl_pad_param();
      break;
    case IR_UPSAMPLE:
      set_tl_upsample_param();
      break;
    case IR_CROP:
      set_tl_crop_param();
      break;
    case IR_CONCAT:
      set_tl_concat_param();
      break;
    case IR_LRN:
    case IR_QUANT:
      is_support = false;
      break;
    default:
      // TODO: support non-atomic
      is_support = false;
      LLVM_DEBUG(llvm::errs() << "un support layer type:" << layer_type << "\n";);
  }

  // init
  inst.opt_chl_quan = isPerChannelQuan;
  inst.opd_typ = isBfloat16;
  // TODO: support ps32
  inst.ps32_md = 0;
  inst.double_conv = (inst.opd1_n % 2 == 0);

  uint64_t cycleCount = 0;
  if (is_support) {
    cycleCount = calCycle(inst);
  }

  // LLVM_DEBUG(llvm::errs() << llvm::format( "  [Balance Layer] tpu %s cycle is %d\n",
  //       im_layer->name().c_str(), (uint32_t)cycleCount));

  // convert cycle to time(ns)
  uint64_t total_time = cycleCount * 1000 / tpu_frequency_;
  return (uint32_t)total_time;
}

void TiuCycle::set_tl_conv_param() {
  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw;
  int sh, sw, pt, pb, pl, pr, dh = 1, dw, pad_value;
  bool do_ic_align = false;
  bool do_leaky_relu = false;
  bool bInt8ConvOp = isa<tpu::TG_INT8_PC_Conv2DOp>(op);

  getConvParam(op, n, ic, ih, iw, oc, oh, ow, g,
               kh, kw, sh, sw, pt, pb, pl, pr,
               dh, dw, is_dw, with_bias, do_relu, do_ic_align,
               do_leaky_relu, pad_value);

  bool has_bias_op = (bInt8ConvOp || (!bInt8ConvOp && with_bias));

  isPerChannelQuan = bInt8ConvOp;
  inst.tsk_typ = Conv2D;
  inst.tsk_opd_num = has_bias_op ? 3 : 2;
  inst.double_conv = ic % 2 == 0;

  // set weight shape
  const std::vector<int> & in_tensors =
        net_graph_->get_in_tensors_of_layer(layer_id_);
  int dims[4];
  net_graph_->get_tensor_dim(in_tensors[1], dims);
  inst.opd1_n = dims[0];
  inst.opd1_c = dims[1];
  inst.opd1_h = dims[2];
  inst.opd1_w = dims[3];

  inst.conv_opd1_x_ins0 = dw - 1;
  inst.conv_opd1_y_ins0 = dh - 1;
  inst.conv_op_x_str = sw;
  inst.conv_op_y_str = sh;

  if (is_dw) {
    inst.tsk_typ = Pooling;
    inst.tsk_eu_typ = 2;
  }
}

void TiuCycle::set_tl_deconv_param(){
  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw;
  int sh, sw, pt, pb, pl, pr, dh = 1, dw, pad_value;
  bool do_ic_align = false;
  bool do_leaky_relu = false;
  bool bInt8ConvOp = isa<tpu::TG_INT8_PC_DeConv2DOp>(op);

  getConvParam(op, n, ic, ih, iw, oc, oh, ow, g,
               kh, kw, sh, sw, pt, pb, pl, pr,
               dh, dw, is_dw, with_bias, do_relu, do_ic_align,
               do_leaky_relu, pad_value);

  bool has_bias_op = (bInt8ConvOp || (!bInt8ConvOp && with_bias));

  isPerChannelQuan = bInt8ConvOp;
  inst.tsk_opd_num = has_bias_op ? 3 : 2;
  inst.double_conv = ic % 2 == 0;

  // set weight shape
  const std::vector<int> & in_tensors =
        net_graph_->get_in_tensors_of_layer(layer_id_);
  int dims[4];
  net_graph_->get_tensor_dim(in_tensors[1], dims);
  inst.opd1_n = dims[0];
  inst.opd1_c = dims[1];
  inst.opd1_h = dims[2];
  inst.opd1_w = dims[3];

  inst.tsk_typ = Pooling;
  inst.tsk_eu_typ = 2;
  inst.conv_opd1_x_ins0 = dw - 1;
  inst.conv_opd1_y_ins0 = dh - 1;
  inst.conv_op_x_str = 1; //sw
  inst.conv_op_y_str = 1; //sh;
}


void TiuCycle::set_tl_pooling_param() {
  bool is_global, do_relu, count_include_pad;
  int n, c, ih, iw, oh, ow, kh, kw;
  int sh, sw, pt, pb, pl, pr, pad_value;
  getPoolingParam(op,
                  n, c, ih, iw, oh, ow,
                  kh, kw, sh, sw,
                  pt, pb, pl, pr, pad_value,
                  is_global, do_relu, count_include_pad);

  inst.conv_opd1_x_ins0 = 1;
  inst.conv_opd1_y_ins0 = 1;
  inst.conv_op_x_str = sw;
  inst.conv_op_y_str = sh;
}

void TiuCycle::set_tl_eltwise_param() {
  if (isa<tpu::TG_INT8_EltwiseMulOp>(op) ||
      isa<tpu::TG_BF16_EltwiseMulOp>(op)) {
    inst.tsk_eu_typ = 0;
  }
  else {
    bool do_early_stride = false;
    int h_stride, w_stride;
    getEltwiseAddParam(op, do_early_stride, h_stride, w_stride);

    inst.tsk_eu_typ = 2;
    if (do_early_stride) {
      inst.opd0_h = inst.opd0_h / h_stride;
      inst.opd0_w = inst.opd0_w / w_stride;
      inst.res0_h = inst.opd0_h;
      inst.res0_w = inst.opd0_w;
    }
  }
}

void TiuCycle::set_tl_scale_param() {
  // scale use depthwise conv
  inst.tsk_typ = Pooling;
  inst.tsk_eu_typ = 2;
  inst.opt_chl_quan = true;
  inst.conv_opd1_x_ins0 = 0;
  inst.conv_opd1_y_ins0 = 0;
  inst.conv_op_x_str = 1; //sw
  inst.conv_op_y_str = 1; //sh;

  inst.opd1_n = 1;
  inst.opd1_c = inst.opd0_c;
  inst.opd1_h = 1;
  inst.opd1_w = 1;
}

void TiuCycle::set_tl_activation_param() {
  inst.tsk_eu_typ = 12;
  inst.tens_lookup = 1;
}

void TiuCycle::set_tl_pad_param() {
  // take pad as ALU operation
  inst.res0_n = inst.opd0_n;
  inst.res0_c = inst.opd0_c;
  inst.res0_h = inst.opd0_h;
  inst.res0_w = inst.opd0_w;
}

void TiuCycle::set_tl_upsample_param() {
  int scale_h = 1;
  int scale_w = 1;
  getUpsampleParam(op, scale_h, scale_w);

  inst.tsk_typ = Pooling;
  inst.tsk_eu_typ = 1;
  inst.opt_chl_quan = true;
  inst.conv_opd1_x_ins0 = 0;
  inst.conv_opd1_y_ins0 = 0;
  inst.conv_op_x_str = 1;
  inst.conv_op_x_str = 1;

  inst.opd1_h = scale_h;
  inst.opd1_w = scale_w;
}

void TiuCycle::set_tl_concat_param() {
  // take concat as ALU operation
  inst.opd0_n = inst.res0_n;
  inst.opd0_c = inst.res0_c;
  inst.opd0_h = inst.res0_h;
  inst.opd0_w = inst.res0_w;
}

void TiuCycle::set_tl_crop_param() {
  // take concat as ALU operation
  inst.opd0_n = inst.res0_n;
  inst.opd0_c = inst.res0_c;
  inst.opd0_h = inst.res0_h;
  inst.opd0_w = inst.res0_w;
}

uint64_t TiuCycle::calCycle(tiu_inst_t inst) {
  Des_tsk_typ desTskType = static_cast<Des_tsk_typ> (inst.tsk_typ);
  uint64_t tempCycle = 0;
  uint64_t resCStart = (inst.res0_addr - LOCAL_MEM_START_ADDR) >> LOCAL_MEM_ADDRWIDTH;
  Tuple4D kernelShape = Tuple4D(inst.opd1_n, inst.opd1_h, inst.opd1_w, inst.opd1_c);
  Tuple4D inputShape = Tuple4D(inst.opd0_n, inst.opd0_h, inst.opd0_w, inst.opd0_c);
  Tuple4D outputShape = Tuple4D(inst.res0_n, inst.res0_h, inst.res0_w, inst.res0_c);
  //partial R/W needs 4 cycle
  int pSumModeLat = (inst.ps32_md == 0) ? 0 :
                    (inst.ps32_md == 1) ? 4 : (inst.ps32_md == 2) ? 4 : 8;
  //37T : 2Array, 37 + 2 + 2 = 4array BDC + cmd latency
  const int syncR0R1PathCycle = 37 + 2 + 2;
  // Without broadcast initLat + broadcast lane 24T : 2Array,
  // 24 + 2 = 4array (cmd latency
  const int cmdLatency = 15 + 8 + 1 + 2;
  // magic ratio every mac
  float misc;
  bool isPerChannelQuan = (inst.opt_chl_quan == 1);
  bool isBfloat16 = (inst.opd_typ == 1);
  // Todo : confirm this
  int perChannelQuanLeastCycle = EU_NUMBER;
  // 18T EU 16 + 1 + 1
  const int postProcessCycle = (isPerChannelQuan) ? 31 : 12;
  // Mac(kh * kw * ic) should be larger than this cycle,
  // or bubble will appear
  int activatedEuNumber = (isBfloat16) ? EU_NUMBER / 2 : EU_NUMBER;
  // load 16 bit data need 2 cycle
  int biasLat = (inst.tsk_opd_num == 3) ? (isPerChannelQuan) ? 0 : 2 : 0;
  int shiftRoundShiftCycle = isBfloat16 ? 2 : (isPerChannelQuan) ? 1 : 4;
  int kernel_hw = kernelShape.w * kernelShape.h;

  switch (desTskType) {
    case(Conv2D):
      {
        //ToDo : Analyze inputStride effect
        //ToDo : partial sum
        int channelNumPerCyc = (inst.double_conv == 1) ?
                                (!isBfloat16) ? 2 : 1 : 1;
        uint64_t channelCycle = ceiling_func(inputShape.c, channelNumPerCyc);

        int fetchCyclePerTime = ceil((((float)(EU_NUMBER - 1) * inst.conv_op_x_str) /
                                        (inst.conv_opd0_x_ins0 + 1) + 1) / EU_NUMBER) + 1;

        // Another bubble ~= 25 cycle
        int bubble = (fetchCyclePerTime > (int)inputShape.c) ?
                      ceil(kernel_hw) * (fetchCyclePerTime - inputShape.c) : 0;

        int loopCycle = int(ceil((kernel_hw * channelCycle +  bubble))
                                + biasLat + shiftRoundShiftCycle + pSumModeLat);
        loopCycle = (isPerChannelQuan && !isBfloat16) ?
                      ((loopCycle > perChannelQuanLeastCycle) ?
                          loopCycle : perChannelQuanLeastCycle) : loopCycle;

        tempCycle = syncR0R1PathCycle;
        tempCycle +=
          outputShape.n *
          ceiling_func(resCStart + outputShape.c, LANE_NUMBER) *
          ceiling_func(outputShape.w * outputShape.h, activatedEuNumber) * loopCycle;
        tempCycle += postProcessCycle;
        break;
      }
    case(Pooling):
      {
        // 0 -> max pooling
        // 1 -> avg pooling
        // 2 -> depthwise
        //ToDo : Analyze inputStride effect
        switch (inst.tsk_eu_typ) {
          case 0: // max pooling
            {
              int fakeOw = activatedEuNumber;
              //Consider small OW
              misc = W_IN(fakeOw) / activatedEuNumber;
              // Consider inst > stride
              misc += W_IN(fakeOw) >= activatedEuNumber ?
                ((float) activatedEuNumber - 1) / activatedEuNumber :
                (W_IN(fakeOw) - 1) / activatedEuNumber;
              misc = misc < 1 ? 1 : misc;
              misc = ((outputShape.w * outputShape.h == 1)) ? 1 : misc;
              const int magicHazard = 2;
              // This magicHazard appears when data gathers need < 2T
              // cout << "misc = " << misc << endl;
              tempCycle =
                outputShape.n *
                ceiling_func((resCStart + outputShape.c), LANE_NUMBER) *
                ceiling_func(outputShape.w * outputShape.h, activatedEuNumber) *
                (kernelShape.h * kernelShape.w * misc + magicHazard);
              break;
            }
          case 1: // avg pooling
            {

              int fakeOw = activatedEuNumber;
              //Consider small OW
              misc = W_IN(fakeOw) / activatedEuNumber;
              // Consider inst > stride
              misc += W_IN(fakeOw) >= activatedEuNumber ?
                ((float) activatedEuNumber - 1) / activatedEuNumber :
                (W_IN(fakeOw) - 1) / activatedEuNumber;
              // cout << misc  << endl;
              misc = misc < 1 ? 1 : misc;
              misc = ((outputShape.w * outputShape.h == 1)) ? 1 : misc;
              int loopCycle = kernelShape.h * kernelShape.w * misc;
              loopCycle = (isPerChannelQuan)
                          ? (loopCycle > perChannelQuanLeastCycle) ? loopCycle : perChannelQuanLeastCycle
                          : loopCycle;
              tempCycle =
                outputShape.n *
                ceiling_func((resCStart + outputShape.c), LANE_NUMBER) *
                ceiling_func(outputShape.w * outputShape.h, activatedEuNumber) *
                (loopCycle + shiftRoundShiftCycle);
              tempCycle += postProcessCycle;

              break;
            }
          case 2: // depthwise
            {
              shiftRoundShiftCycle = ((kernelShape.w * kernelShape.h == 1) &&
                                     (!isPerChannelQuan || !isBfloat16)) ? 3 : shiftRoundShiftCycle;

              int fakeOw = outputShape.w >= EU_NUMBER ? EU_NUMBER : outputShape.w;
              float ohNumberOneTime = outputShape.w >= EU_NUMBER ? 1 : (float)EU_NUMBER / outputShape.w;
              //Consider small OW
              misc = W_IN(fakeOw) / EU_NUMBER * ohNumberOneTime;
              misc += ((float) EU_NUMBER - 1) / EU_NUMBER;
              misc = ((kernelShape.w * kernelShape.h == 1)) ? 1 : misc;
              int loopCycle = kernelShape.h * kernelShape.w * misc;
              loopCycle = (isPerChannelQuan) ?
                            (loopCycle > perChannelQuanLeastCycle) ? loopCycle : perChannelQuanLeastCycle :
                            loopCycle;

              tempCycle =
                outputShape.n *
                ceiling_func((resCStart + outputShape.c), LANE_NUMBER) *
                ceiling_func(outputShape.w * outputShape.h, activatedEuNumber) *
                (loopCycle + biasLat + shiftRoundShiftCycle);
              tempCycle += postProcessCycle;
              break;
            }
          default:
            BM_ASSERT(false);
        }
        break;
      }
    case(MatrixMul):
      {
        //ToDo : Analyze act+wt bank conflict
        uint32_t addResLat = (inst.opt_res_add) ? 2 : 0; //16bit

        tempCycle = syncR0R1PathCycle;
        tempCycle += outputShape.n *
          ceiling_func(outputShape.c, LANE_NUMBER) * ceiling_func(outputShape.w, activatedEuNumber) *
          (kernelShape.n +  biasLat + shiftRoundShiftCycle + pSumModeLat + addResLat);
        tempCycle += postProcessCycle;
        break;
      }
    case(TensorArithmetic):
      {
        bool isInput8BitMode = (inst.opt_opd0_int8 == 1);
        bool isRes8BitMode = (inst.opt_res0_int8 == 1);
        bool isOpd1Const = (inst.opt_opd1_const == 1);
        //mode 0 : mul res8bit/add/sub
        //mode 1 : mac
        //mode 2 : max/min/shift/logic/mul res16bit
        //mode 3 : mdsum
        //mode 4 : lut
        int mode = getTensorArithmeticMode(inst.tsk_eu_typ, isRes8BitMode);

        float euLat = ((inst.tens_lookup == 1)) ? 0 :
                      getEltwiseLatency(inst.tsk_eu_typ, isInput8BitMode, isOpd1Const, mode);
        euLat += isRes8BitMode ? 0 : 1;
        //consider stride more
        float writeLat = (inst.res0_c_str % 16 == 0) ? 1 : (float)30 / 16;
        tempCycle =
          outputShape.n *
          ceiling_func(resCStart + outputShape.c, LANE_NUMBER) *
          ceiling_func(outputShape.w * outputShape.h, activatedEuNumber) *
          (euLat * writeLat);

        if(inst.tens_lookup == 1) {
          tempCycle =
            outputShape.n *
            ceiling_func(resCStart + outputShape.c, LANE_NUMBER) *
            ceiling_func(outputShape.w * outputShape.h, activatedEuNumber) *
            (1 + inst.opd1_h + 2); //opa, opb, bubble
        }

        if(inst.tens_mdsum == 1) {
          tempCycle =
            outputShape.n *
            ceiling_func(resCStart + outputShape.c, LANE_NUMBER) *
            ceiling_func(outputShape.w * outputShape.h, activatedEuNumber) *
            (1 + 2); //opa, bubble + latency
        }

        tempCycle += cmdLatency;
        tempCycle += postProcessCycle;
        break;
      }
    case(MatrixMul2):
      {
        //cout << "Not supported now" << endl;
        BM_ASSERT(0);
        break;
      }
    default:
      {
        //cout << "Not supported now" << endl;
        BM_ASSERT(0);
        break;
      }
  }
  return tempCycle;
}

int TiuCycle::getTensorArithmeticMode(int taskType, bool is8BitMode) {
  //mode 0 : mul res8bit/add/sub
  //mode 1 : mac
  //mode 2 : max/min/shift/logic/mul res16bit
  //mode 3 : mdsum
  //mode 4 : lut
  int ret;
  if (taskType == 0) {
    // mul
    ret = (is8BitMode) ? 0 : 2;
  } else if (taskType == 1) {
    // mac
    ret = 1;
  } else if (taskType == 2) {
    // add
    ret = 0;
  } else if (taskType == 3) {
    // sub
    ret = 0;
  } else if (taskType == 4) {
    // max
    ret = 2;
  } else if (taskType == 5) {
    // min
    ret = 2;
  } else if (taskType == 6) {
    // shift
    ret = 2;
  } else if (taskType == 7) {
    // and
    ret = 2;
  } else if (taskType == 8) {
    // or
    ret = 2;
  } else if (taskType == 9) {
    // xor
    ret = 2;
  } else if (taskType == 10) {
    // copy
    ret = 2;
  } else if (taskType == 11) {
    // md_sum
    ret = 3;
  } else if (taskType == 12) {
    // lut
    ret = 4;
  }else {
    //cout << "Not supported now" << endl;
    BM_ASSERT(0);
  }
  return ret;
}

float TiuCycle::getEltwiseLatency(int taskType, bool is8BitMode, bool isOpd1Const, int mode) {
  float ret;
  if (taskType == 0) { //Todo : random bubble, HW will erase it 2.5->2
    // mul
    ret = (is8BitMode) ? 2.5 : 4.5; //8bit : rounding shift
    ret += (mode == 0) ? 3 : 0;
    // ret -= (isOpd1Const) ? 1 : 0;
  } else if (taskType == 1) {
    // mac
    ret = (is8BitMode) ? 9 : 11;
  } else if (taskType == 2) {
    // add
    ret = (is8BitMode) ? 5 : 7;
    ret -= (!isOpd1Const) ? 0 : (is8BitMode) ? 0 : 0.5; //remove bank conflic
  } else if (taskType == 3) {
    // sub
    ret = (is8BitMode) ? 5 : 7;
    ret -= (!isOpd1Const) ? 0 : (is8BitMode) ? 0 : 0.5; //remove bank conflic
  } else if (taskType == 4) {
    // max
    ret = (is8BitMode) ? 3 : 5;
  } else if (taskType == 5) {
    // min
    ret = (is8BitMode) ? 3 : 5;
  } else if (taskType == 6) {
    // shift
    ret = (is8BitMode) ? 4 : 4;
  } else if (taskType == 7) {
    // and
    ret = (is8BitMode) ? 3 : 5;
  } else if (taskType == 8) {
    // or
    ret = (is8BitMode) ? 3 : 5;
  } else if (taskType == 9) {
    // xor
    ret = (is8BitMode) ? 3 : 5;
  } else if (taskType == 10) {
    // copy
    ret = (is8BitMode) ? 2 : 4;
  } else {
    //cout << "Not supported now" << endl;
    BM_ASSERT(0);
  }
  return ret;
}

}