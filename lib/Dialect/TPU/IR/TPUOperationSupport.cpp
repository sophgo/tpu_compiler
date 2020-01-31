#include <numeric>
#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

llvm::StringRef getOpName(Operation *op) {
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::BatchNormOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::ConcatOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::Conv2DOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::CropOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::DeConv2DOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::DetectionOutputOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::DivOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::EltwiseOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::FullyConnectedOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::InputOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::NormalizeOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::PermuteOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::Pool2DOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::PowerOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::PriorBoxOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::ReluOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::ReshapeOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::PReluOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::ScaleOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::SigmoidOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::SliceOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::SoftmaxOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::SqrtOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::TanHOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::UpsampleOp>(op)) {
    return cast_op.name().getValue();
  }

  //---------------------------------------------------------------------------
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::QuantizationOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::DequantizationOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::DummyDataOp>(op)) {
    return cast_op.name().getValue();
  }

  llvm::errs() << "getOpName: " << op->getName() << " Not Found\n ";
  assert(false);
  return "not_found";
}

llvm::StringRef getPreviousOpName(Operation *op, uint index = 0) {
  if ( op->getNumOperands() < (index + 1) ) {
    assert(false);
    return llvm::StringRef();
  }
  return getOpName(op->getOperand(index)->getDefiningOp());
}

std::string getOpQuant(Operation *op) {
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::BatchNormOp>(op)) {
    return cast_op.quant();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::ConcatOp>(op)) {
    return cast_op.quant();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::Conv2DOp>(op)) {
    return cast_op.quant();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::CropOp>(op)) {
    return cast_op.quant();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::DeConv2DOp>(op)) {
    return cast_op.quant();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::DetectionOutputOp>(op)) {
    return cast_op.quant();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::DivOp>(op)) {
    return cast_op.quant();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::EltwiseOp>(op)) {
    return cast_op.quant();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::FullyConnectedOp>(op)) {
    return cast_op.quant();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::InputOp>(op)) {
    return cast_op.quant();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::NormalizeOp>(op)) {
    return cast_op.quant();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::PermuteOp>(op)) {
    return cast_op.quant();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::Pool2DOp>(op)) {
    return cast_op.quant();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::PowerOp>(op)) {
    return cast_op.quant();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::PriorBoxOp>(op)) {
    return cast_op.quant();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::ReluOp>(op)) {
    return cast_op.quant();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::ReshapeOp>(op)) {
    return cast_op.quant();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::PReluOp>(op)) {
    return cast_op.quant();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::ScaleOp>(op)) {
    return cast_op.quant();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::SigmoidOp>(op)) {
    return cast_op.quant();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::SliceOp>(op)) {
    return cast_op.quant();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::SoftmaxOp>(op)) {
    return cast_op.quant();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::SqrtOp>(op)) {
    return cast_op.quant();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::TanHOp>(op)) {
    return cast_op.quant();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::UpsampleOp>(op)) {
    return cast_op.quant();
  }

  //---------------------------------------------------------------------------
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::QuantizationOp>(op)) {
    return cast_op.quant();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::DequantizationOp>(op)) {
    return cast_op.quant();
  }

  llvm::errs() << "getOpQuant: " << op->getName() << " Not Found\n ";
  assert(false);
  return "NONE";
}

float getOpThreshold(Operation *op) {
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::BatchNormOp>(op)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::ConcatOp>(op)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::Conv2DOp>(op)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::CropOp>(op)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::DeConv2DOp>(op)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::DetectionOutputOp>(op)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::DivOp>(op)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::EltwiseOp>(op)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::FullyConnectedOp>(op)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::InputOp>(op)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::NormalizeOp>(op)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::PermuteOp>(op)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::Pool2DOp>(op)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::PowerOp>(op)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::PriorBoxOp>(op)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::ReluOp>(op)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::ReshapeOp>(op)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::PReluOp>(op)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::ScaleOp>(op)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::SigmoidOp>(op)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::SliceOp>(op)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::SoftmaxOp>(op)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::SqrtOp>(op)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::TanHOp>(op)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::UpsampleOp>(op)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }

  //---------------------------------------------------------------------------
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::TL_LA_Conv2DOp>(op)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::TL_LW_Conv2DOp>(op)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }

  llvm::errs() << "getOpThreshold: " << op->getName() << " Not Found\n ";
  assert(false);
  return NAN;
}

float getPreviousOpThreshold(Operation *op, uint index = 0) {
  if ( op->getNumOperands() < (index + 1) ) {
    assert(false);
    return NAN;
  }
  auto formerOp = op->getOperand(index)->getDefiningOp();
  return getOpThreshold(formerOp);
}

uint64_t getPreviousOpAddress(Operation *op, uint index = 0) {
  if ( op->getNumOperands() < (index + 1) ) {
    assert(false);
    return 0xFFFFFFFFFFFFFFFF;
  }

  auto formerOp = op->getOperand(index)->getDefiningOp();
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::QuantizationOp>(formerOp)) {
    return cast_op.offset().getValue().getLimitedValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::TL_LA_Conv2DOp>(formerOp)) {
    return cast_op.offset().getValue().getLimitedValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::TL_LW_Conv2DOp>(formerOp)) {
    return cast_op.offset().getValue().getLimitedValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::ConcatOp>(formerOp)) {
    return cast_op.offset().getValue().getLimitedValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::Conv2DOp>(formerOp)) {
    return cast_op.offset().getValue().getLimitedValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::DivOp>(formerOp)) {
    return cast_op.offset().getValue().getLimitedValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::FullyConnectedOp>(formerOp)) {
    return cast_op.offset().getValue().getLimitedValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::Pool2DOp>(formerOp)) {
    return cast_op.offset().getValue().getLimitedValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::PermuteOp>(formerOp)) {
    return cast_op.offset().getValue().getLimitedValue();
  }  
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::PowerOp>(formerOp)) {
    return cast_op.offset().getValue().getLimitedValue();
  }  
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::PReluOp>(formerOp)) {
    return cast_op.offset().getValue().getLimitedValue();
  }  
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::ReluOp>(formerOp)) {
    return cast_op.offset().getValue().getLimitedValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::EltwiseOp>(formerOp)) {
    return cast_op.offset().getValue().getLimitedValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::SigmoidOp>(formerOp)) {
    return cast_op.offset().getValue().getLimitedValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::CropOp>(formerOp)) {
    return cast_op.offset().getValue().getLimitedValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::ScaleOp>(formerOp)) {
    return cast_op.offset().getValue().getLimitedValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::SliceOp>(formerOp)) {
    return cast_op.offset().getValue().getLimitedValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::SqrtOp>(formerOp)) {
    return cast_op.offset().getValue().getLimitedValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::TanHOp>(formerOp)) {
    return cast_op.offset().getValue().getLimitedValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::ReshapeOp>(formerOp)) {
    // for reshape, we need to go to this one's previous
    // this is recursive ...
    return getPreviousOpAddress(cast_op);
  }
  llvm::errs() << "Former Op: " << formerOp->getName() << "\n";
  llvm::errs() << op->getName() << " Not find\n";
  assert(0);
  return 0xFFFFFFFFFFFFFFFF;
}

uint64_t getWeightOpAddress(Operation *op) {
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(op)) {
    return cast_op.offset().getValue().getLimitedValue();
  }
  assert(false);
  return 0xFFFFFFFFFFFFFFFF;
}

/***********************************************************
 * TPU Ops parameter helpers
 ***********************************************************/
 #define calcConv2DSpatialOutput(_i_, _k_, _s_, _p_, _d_) \
    (((_i_) + 2 * (_p_) - ((_k_+ (_d_-1)*(_k_-1)) - 1) - 1) / (_s_) + 1)

static int64_t findPadForSamePadding(int64_t i, int64_t o, int64_t k, int64_t s, int64_t d) {
  //llvm::errs() << "i: " << i << ", o: " << o << ", k: " << k << ", s: " << s << ", d: " << d << "\n";
  if (k == 1) {
    return 0;
  }
  for (int64_t p = 1; p <= (k - 1 +((d-1)*(k-1))); ++p) {
    if (calcConv2DSpatialOutput(i, k, s, p, d) == o) {
      return p;
    }
  }
  assert(false);
  return 0;
}

template<typename T>
void getConv2DOpParam(T &op,
    int &n, int &ic, int &ih, int &iw, int &oc, int &oh, int &ow, int &g,
    int &kh, int &kw, int &sh, int &sw, int &ph, int &pw, int &dh, int &dw,
    bool &with_bias, bool &do_relu) {
  dh = op.dilation_h_factor().getLimitedValue();
  dw = op.dilation_w_factor().getLimitedValue();
  sh = op.stride_h().getLimitedValue();
  sw = op.stride_w().getLimitedValue();
  auto input_type = op.input()->getType().template cast<TensorType>();
  std::vector<int64_t> i_s(input_type.getShape());
  auto output_type = op.output()->getType().template cast<TensorType>();
  std::vector<int64_t> o_s(output_type.getShape());
  auto filter_type = op.filter()->getType().template cast<TensorType>();
  std::vector<int64_t> f_s(filter_type.getShape());

  assert((i_s[0] == o_s[0]) && "input N not equal to output N");
  n = i_s[0];
  ic = i_s[1];
  ih = i_s[2];
  iw = i_s[3];
  oc = o_s[1];
  oh = o_s[2];
  ow = o_s[3];
  auto f_dim = f_s.size();
  kh = f_s[f_dim - 2];
  kw = f_s[f_dim - 1];
  if (op.padding() == "SAME") {
    ph = findPadForSamePadding(ih, oh, kh, sh, dh);
    pw = findPadForSamePadding(iw, ow, kw, sw, dw);
  } else if (op.padding() == "VALID") {
    ph = 0;
    pw = 0;
  } else {
    assert(false);
  }
  g = op.group().getLimitedValue();
  if (g != 1) {
    // only support depthwise group for now (not support normal group)
    assert(f_s.size() == 5 && g == f_s[0]);
    // assert(f_s[1] == 1 && f_s[2] == 1);
    // assert(g == ic && g == oc);
  } else {
    assert(f_s.size() == 4);
  }
  if (op.fused_activation_function() == "NONE") {
    do_relu = false;
  } else if (op.fused_activation_function() == "RELU") {
    do_relu = true;
  } else {
    assert(0);
  }
  with_bias = op.with_bias();
}

template void getConv2DOpParam<tpu::Conv2DOp>(tpu::Conv2DOp &op,
    int &n, int &ic, int &ih, int &iw, int &oc, int &oh, int &ow, int &g,
    int &kh, int &kw, int &sh, int &sw, int &ph, int &pw, int &dh, int &dw,
    bool &with_bias, bool &do_relu);
template void getConv2DOpParam<tpu::TL_LA_Conv2DOp>(tpu::TL_LA_Conv2DOp &op,
    int &n, int &ic, int &ih, int &iw, int &oc, int &oh, int &ow, int &g,
    int &kh, int &kw, int &sh, int &sw, int &ph, int &pw, int &dh, int &dw,
    bool &with_bias, bool &do_relu);
template void getConv2DOpParam<tpu::TL_LW_Conv2DOp>(tpu::TL_LW_Conv2DOp &op,
    int &n, int &ic, int &ih, int &iw, int &oc, int &oh, int &ow, int &g,
    int &kh, int &kw, int &sh, int &sw, int &ph, int &pw, int &dh, int &dw,
    bool &with_bias, bool &do_relu);

void getDeConv2DOpParam(tpu::DeConv2DOp &op,
    int &n, int &ic, int &ih, int &iw, int &oc, int &oh, int &ow, int &g,
    int &kh, int &kw, int &sh, int &sw, int &ph, int &pw, int &dh, int &dw,
    bool &with_bias) {
  dh = op.dilation_h_factor().getLimitedValue();
  dw = op.dilation_w_factor().getLimitedValue();
  sh = op.stride_h().getLimitedValue();
  sw = op.stride_w().getLimitedValue();
  auto input_type = op.input()->getType().template cast<TensorType>();
  std::vector<int64_t> i_s(input_type.getShape());
  auto output_type = op.output()->getType().template cast<TensorType>();
  std::vector<int64_t> o_s(output_type.getShape());
  auto filter_type = op.filter()->getType().template cast<TensorType>();
  std::vector<int64_t> f_s(filter_type.getShape());
  assert((i_s[0] == o_s[0]) && "input N not equal to output N");
  n = i_s[0];
  ic = i_s[1];
  ih = i_s[2];
  iw = i_s[3];
  oc = o_s[1];
  oh = o_s[2];
  ow = o_s[3];
  auto f_dim = f_s.size();
  kh = f_s[f_dim - 2];
  kw = f_s[f_dim - 1];
  // TODO - padding
  assert(op.padding() == "VALID");
  ph = 0;
  pw = 0;

  g = op.group().getLimitedValue();
  with_bias = op.with_bias();
}

void getPool2DOpParam(tpu::Pool2DOp &op,
    bool &is_average_pool, int &n, int &c, int &ih, int &iw, int &oh, int &ow,
    int &kh, int &kw, int &sh, int &sw, int &pt, int &pb, int &pl, int &pr, bool &do_relu) {
  auto pool_method = op.getAttrOfType<StringAttr>("pool");
  if (pool_method.getValue() == "AVE") {
    is_average_pool = true;
  } else if (pool_method.getValue() == "MAX") {
    is_average_pool = false;
  } else {
    assert(false);
  }
  kh = op.filter_height().getLimitedValue();
  kw = op.filter_width().getLimitedValue();
  sh = op.stride_h().getLimitedValue();
  sw = op.stride_w().getLimitedValue();
  auto input_type = op.input()->getType().cast<TensorType>();
  std::vector<int64_t> i_s(input_type.getShape());
  auto output_type = op.output()->getType().cast<TensorType>();
  std::vector<int64_t> o_s(output_type.getShape());
  assert((i_s[0] == o_s[0]) && "input N not equal to output N");
  assert((i_s[1] == o_s[1]) && "input C not equal to output C");
  n = i_s[0];
  c = i_s[1];
  ih = i_s[2];
  iw = i_s[3];
  oh = o_s[2];
  ow = o_s[3];

  if (op.padding().hasValue()) {
    auto padding_attr = op.getAttrOfType<StringAttr>("padding");
    if (padding_attr.getValue() == "SAME") {
      llvm::errs() << "same\n";
      pt = findPadForSamePadding(ih, oh, kh, sh, 1);
      pb = pt;
      pl = findPadForSamePadding(iw, ow, kw, sw, 1);
      pr = pl;
    } else if (padding_attr.getValue() == "VALID") {
      llvm::errs() << "valid\n";
      pt = 0;
      pb = 0;
      pl = 0;
      pr = 0;
    } else {
      assert(false);
    }
  } else if (op.pad_top().hasValue() && op.pad_bottom().hasValue() &&
            op.pad_left().hasValue() && op.pad_right().hasValue()) {
    pt = op.pad_top().getValue().getLimitedValue();
    pb = op.pad_bottom().getValue().getLimitedValue();
    pl = op.pad_left().getValue().getLimitedValue();
    pr = op.pad_right().getValue().getLimitedValue();
  } else {
    assert(false);
  }
  if (op.fused_activation_function() == "NONE") {
    do_relu = false;
  } else if (op.fused_activation_function() == "RELU") {
    do_relu = true;
  } else {
    assert(0);
  }
}

void getFullyConnectedOpParam(tpu::FullyConnectedOp &op,
    bool &with_transpose, int &m, int &k, int &n,
    bool &with_bias, bool &do_relu) {
  auto input_type = op.input()->getType().cast<TensorType>();
  std::vector<int64_t> i_s(input_type.getShape());
  auto output_type = op.output()->getType().cast<TensorType>();
  std::vector<int64_t> o_s(output_type.getShape());
  auto filter_type = op.filter()->getType().cast<TensorType>();
  std::vector<int64_t> f_s(filter_type.getShape());
  assert((i_s[0] == o_s[0]) && "input M not equal to output M");
  m = i_s[0];
  // assuming transpose is false
  assert((i_s[1] == f_s[1]) && "input K not equal to filter K");
  k = i_s[1];
  assert((f_s[0] == o_s[1]) && "filter N not equal to output N");
  n = o_s[1];
  if (op.fused_activation_function() == "NONE") {
    do_relu = false;
  } else if (op.fused_activation_function() == "RELU") {
    do_relu = true;
  } else {
    assert(0);
  }
  with_transpose = op.with_transpose();
  with_bias = op.with_bias();
}

void getConv2DOpVariadicTensors(tpu::Conv2DOp &op,
    std::vector<std::shared_ptr<std::vector<float> > > &opdT,
    std::shared_ptr<std::vector<float> > &bias,
    std::shared_ptr<std::vector<float> > &rshift,
    std::shared_ptr<std::vector<float> > &multiplier,
    std::shared_ptr<std::vector<float> > &per_channel_info,
    std::shared_ptr<std::vector<float> > &eltwise_input) {
  unsigned idx = 2;  // first 2 opdT are always input and filter
  if (op.per_channel_info_is_aggregated()) {
    // only INT8 related quantization use aggregated per_channel_info
    assert(op.quant() == "INT8" || op.quant() == "INT8_PER_CHANNEL"
           || op.quant() == "INT8_MULTIPLIER");
    per_channel_info = opdT[idx];
    idx += 1;
  }
  else {
    if (op.with_bias()) {
      bias = opdT[idx];
      idx += 1;
    }

    if (op.quant() == "INT8" || op.quant() == "INT8_PER_CHANNEL"
           || op.quant() == "INT8_MULTIPLIER") {
      rshift = opdT[idx];
      idx += 1;
    }

    if (op.quant() == "INT8_MULTIPLIER") {
      multiplier = opdT[idx];
      idx += 1;
    }
  }
  if (op.fused_eltwise_method() != "NONE") {
    eltwise_input = opdT[idx];
    idx += 1;
  }
  if (idx != opdT.size()) {
    llvm::errs() << op.name() << ": opdT.size=" << opdT.size()
                 << ", idx=" << idx << "\n";
    assert(0);
  }
}

// TODO - same as getConv2DOpVariadicTensors
// Use template implementation to reuse the code
void getDeConv2DOpVariadicTensors(tpu::DeConv2DOp &op,
    std::vector<std::shared_ptr<std::vector<float> > > &opdT,
    std::shared_ptr<std::vector<float> > &bias,
    std::shared_ptr<std::vector<float> > &rshift,
    std::shared_ptr<std::vector<float> > &multiplier,
    std::shared_ptr<std::vector<float> > &per_channel_info,
    std::shared_ptr<std::vector<float> > &eltwise_input) {
  unsigned idx = 2;  // first 2 opdT are always input and filter
  if (op.per_channel_info_is_aggregated()) {
    // only INT8 related quantization use aggregated per_channel_info
    assert(op.quant() == "INT8" || op.quant() == "INT8_PER_CHANNEL"
           || op.quant() == "INT8_MULTIPLIER");
    per_channel_info = opdT[idx];
    idx += 1;
  }
  else {
    if (op.with_bias()) {
      bias = opdT[idx];
      idx += 1;
    }

    if (op.quant() == "INT8" || op.quant() == "INT8_PER_CHANNEL"
           || op.quant() == "INT8_MULTIPLIER") {
      rshift = opdT[idx];
      idx += 1;
    }

    if (op.quant() == "INT8_MULTIPLIER") {
      multiplier = opdT[idx];
      idx += 1;
    }
  }
  if (op.fused_eltwise_method() != "NONE") {
    eltwise_input = opdT[idx];
    idx += 1;
  }
  if (idx != opdT.size()) {
    llvm::errs() << op.name() << ": opdT.size=" << opdT.size()
                 << ", idx=" << idx << "\n";
    assert(0);
  }
}
void getScaleOpVariadicTensors(
    tpu::ScaleOp &op, std::vector<std::shared_ptr<std::vector<float>>> &opdT,
    std::shared_ptr<std::vector<float>> &bias,
    std::shared_ptr<std::vector<float>> &rshift,
    std::shared_ptr<std::vector<float>> &multiplier) {
  unsigned idx = 2; // first 2 opdT are always input and scale
  if (op.with_bias()) {
    bias = opdT[idx];
    idx += 1;
  }
  if (op.quant() == "INT8" || op.quant() == "INT8_PER_CHANNEL") {
    rshift = opdT[idx];
    multiplier = opdT[idx+1];
    idx += 2;
  }

  if (idx != opdT.size()) {
    llvm::errs() << op.name() << ": opdT.size=" << opdT.size()
                 << ", idx=" << idx << "\n";
    assert(0 && "opdT size wrong");
  }
}

void getFullyConnectedOpVariadicTensors(tpu::FullyConnectedOp &op,
    std::vector<std::shared_ptr<std::vector<float> > > &opdT,
    std::shared_ptr<std::vector<float> > &bias,
    std::shared_ptr<std::vector<float> > &rshift) {
  unsigned idx = 2;  // first 2 opdT are always input and filter
  if (op.with_bias()) {
    bias = opdT[idx];
    idx += 1;
  }
  if (op.quant() == "INT8") {
    rshift = opdT[idx];
    idx += 1;
  }
  if (idx != opdT.size()) {
    llvm::errs() << op.name() << ": opdT.size=" << opdT.size()
                 << ", idx=" << idx << "\n";
    assert(0);
  }
}

void getPReluOpVariadicTensors(tpu::PReluOp &op,
    std::vector<std::shared_ptr<std::vector<float> > > &opdT,
    std::shared_ptr<std::vector<float> > &rshift_pos,
    std::shared_ptr<std::vector<float> > &rshift_neg,
    std::shared_ptr<std::vector<float> > &multiplier_pos,
    std::shared_ptr<std::vector<float> > &multiplier_neg) {
  unsigned idx = 2;  // first 2 opdT are always input and filter
  if (op.quant() == "INT8" || op.quant() == "INT8_PER_CHANNEL"
          || op.quant() == "INT8_MULTIPLIER") {
    rshift_pos = opdT[idx];
    idx += 1;
    rshift_neg = opdT[idx];
    idx += 1;
  }

  if (op.quant() == "INT8_MULTIPLIER") {
    multiplier_pos = opdT[idx];
    idx += 1;
    multiplier_neg = opdT[idx];
    idx += 1;
  }

  if (idx != opdT.size()) {
    llvm::errs() << op.name() << ": opdT.size=" << opdT.size()
                 << ", idx=" << idx << "\n";
    assert(0);
  }
}

void getPReluOpVariadicTensors(tpu::PReluOp &op,
    std::vector<std::shared_ptr<std::vector<float> > > &opdT,
    std::shared_ptr<std::vector<float> > &rshift_pos,
    std::shared_ptr<std::vector<float> > &multiplier_pos,
    std::shared_ptr<std::vector<float> > &rshift_neg) {
  unsigned idx = 2;  // first 2 opdT are always input and filter
  if (op.quant() == "INT8" || op.quant() == "INT8_PER_CHANNEL"
          || op.quant() == "INT8_MULTIPLIER") {
    rshift_pos = opdT[idx];
    idx += 1;
    multiplier_pos = opdT[idx];
    idx += 1;
    rshift_neg = opdT[idx];
    idx += 1;
  }

  if (idx != opdT.size()) {
    llvm::errs() << op.name() << ": opdT.size=" << opdT.size()
                 << ", idx=" << idx << "\n";
    assert(0);
  }
}

} // namespace
