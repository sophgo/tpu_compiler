#include <numeric>
#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

void arrayAttrToVector(const ArrayAttr &arrayAttr,
                              std::vector<int32_t> &vector) {
  vector.clear();
  for (auto en : llvm::enumerate(arrayAttr)) {
    auto attr = en.value().dyn_cast<IntegerAttr>();
    vector.push_back(attr.getInt());
  }
}

llvm::StringRef getOpName(Operation *op) {
  if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpCommonInterface>(op)) {
    return tpuOp.getOpName();
  }

  // to be deprecated
  if (auto tpuOp = llvm::dyn_cast<tpu::TpuInterface>(op)) {
    return tpuOp.getOpName();
  } else {
    //assert(false);
    return llvm::StringRef();
  }
}

llvm::StringRef getPreviousOpName(Operation *op, uint index = 0) {
  if ( op->getNumOperands() < (index + 1) ) {
    assert(false);
    return llvm::StringRef();
  }
  return getOpName(op->getOperand(index)->getDefiningOp());
}

int getOpLayerId(Operation *op) {
  if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpCommonInterface>(op)) {
    return tpuOp.getOpLayerId();
  }

  // to be deprecated
  if (auto tpuOp = llvm::dyn_cast<tpu::TpuInterface>(op)) {
    return tpuOp.getOpLayerId();
  } else {
    //assert(false);
    return -1;
  }
}

LogicalResult setOpLayerId(Operation *op, int id) {
  if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpCommonInterface>(op)) {
    return tpuOp.setOpLayerId(id);
  } else {
    return failure();
  }
}

llvm::StringRef getOpQuant(Operation *op) {
  if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(op)) {
    return tpuOp.getOpQuant();
  }

  // to be deprecated
  if (auto tpuOp = llvm::dyn_cast<tpu::TpuInterface>(op)) {
    return tpuOp.getOpQuant();
  } else {
    assert(false);
    return llvm::StringRef();
  }
}

LogicalResult setOpQuant(Operation *op, llvm::StringRef mode) {
  if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(op)) {
    return tpuOp.setOpQuantMode(mode);
  } else {
    llvm::errs() << __func__ << " failed " << getOpName(op) << "\n";
    assert(false);
    return failure();
  }
}

llvm::StringRef getOpQuantParamType(Operation *op) {
  if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(op)) {
    return tpuOp.getOpQuantParamType();
  } else {
    llvm::errs() << __func__ << " failed " << getOpName(op) << "\n";
    assert(false);
    return llvm::StringRef();
  }
}

LogicalResult setOpQuantParamType(Operation *op, llvm::StringRef type) {
  if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(op)) {
    return tpuOp.setOpQuantParamType(type);
  } else {
    llvm::errs() << __func__ << " failed " << getOpName(op) << "\n";
    assert(false);
    return failure();
  }
}

bool isOpQuantPerchannel(Operation *op) {
  if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(op)) {
    return tpuOp.isOpQuantPerchannel();
  } else {
    llvm::errs() << __func__ << " failed " << getOpName(op) << "\n";
    assert(false);
    return false;
  }
}

LogicalResult setOpQuantPerchannel(Operation *op, bool flag) {
  if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(op)) {
    return tpuOp.setOpQuantPerchannel(flag);
  } else {
    llvm::errs() << __func__ << " failed " << getOpName(op) << "\n";
    assert(false);
    return failure();
  }
}

bool isOpQuantAsymmetric(Operation *op) {
  if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(op)) {
    return tpuOp.isOpQuantAsymmetric();
  } else {
    llvm::errs() << __func__ << " failed " << getOpName(op) << "\n";
    assert(false);
    return false;
  }
}

LogicalResult setOpQuantAsymmetric(Operation *op, bool flag) {
  if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(op)) {
    return tpuOp.setOpQuantAsymmetric(flag);
  } else {
    llvm::errs() << __func__ << " failed " << getOpName(op) << "\n";
    assert(false);
    return failure();
  }
}

float getOpThreshold(Operation *op) {
  if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(op)) {
    return tpuOp.getOpQuantThreshold();
  }

  // to be removed
  if (auto tpuOp = llvm::dyn_cast<tpu::TpuInterface>(op)) {
    return tpuOp.getResultQuantThreshold();
  } else {
    llvm::errs() << __func__ << " failed " << getOpName(op) << "\n";
    assert(false);
    return NAN;
  }
}

LogicalResult setOpThreshold(Operation *op, float threshold) {
  if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(op)) {
    return tpuOp.setOpQuantThreshold(threshold);
  }

  // to be removed
  if (auto tpuOp = llvm::dyn_cast<tpu::TpuInterface>(op)) {
    return tpuOp.setResultQuantThreshold(threshold);
  } else {
    llvm::errs() << __func__ << " failed " << getOpName(op) << "\n";
    assert(false);
    return failure();
  }
}

float getPreviousOpThreshold(Operation *op, uint index = 0) {
  if ( op->getNumOperands() < (index + 1) ) {
    llvm::errs() << __func__ << " failed " << getOpName(op)
                 << ", opd " << index << "\n";
    assert(false);
    return NAN;
  }
  auto formerOp = op->getOperand(index)->getDefiningOp();
  return getOpThreshold(formerOp);
}

uint64_t getOpAddress(Operation *op) {
  if (auto tpuTGOp = llvm::dyn_cast<tpu::TpuTGOpCodegenInterface>(op)) {
    return tpuTGOp.getGAddr();
  }

  if (auto tpuOp = llvm::dyn_cast<tpu::TpuInterface>(op)) {
    return tpuOp.getResultAddress().getLimitedValue();
  } else {
    llvm::errs() << __func__ << " failed " << getOpName(op) << "\n";
    assert(false);
    return 0xFFFFFFFFFFFFFFFF;
  }
}

LogicalResult setOpAddress(Operation *op, uint64_t gaddr) {
  if (auto tpuTGOp = llvm::dyn_cast<tpu::TpuTGOpCodegenInterface>(op)) {
    return tpuTGOp.setGAddr(gaddr);
  }

  if (auto tpuOp = llvm::dyn_cast<tpu::TpuInterface>(op)) {
    return tpuOp.setResultAddress(gaddr);
  } else {
    llvm::errs() << __func__ << " failed " << getOpName(op) << "\n";
    assert(false);
    return failure();
  }
}

uint64_t getPreviousOpAddress(Operation *op, uint index = 0) {
  if ( op->getNumOperands() < (index + 1) ) {
    assert(false);
    return 0xFFFFFFFFFFFFFFFF;
  }
  auto formerOp = op->getOperand(index)->getDefiningOp();
  return getOpAddress(formerOp);
}

uint64_t getWeightOpAddress(Operation *op) {
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(op)) {
    return cast_op.offset().getValue().getLimitedValue();
  } else {
    llvm::errs() << __func__ << " failed " << getOpName(op) << "\n";
    assert(false);
    return 0xFFFFFFFFFFFFFFFF;
  }
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

tpu::QuantParam getDefaultQuantParam(Builder &builder) {
  return tpu::QuantParam::get(
      builder.getStringAttr("NONE"),
      builder.getStringAttr("NONE"),
      builder.getBoolAttr(false),
      builder.getBoolAttr(false),
      builder.getF32FloatAttr(0.0),
      builder.getF32FloatAttr(0.0),
      builder.getContext());
}

void parseConvParam(const tpu::ConvParam &p, bool is_deconv,
    Value *input, Value *output, Value *filter,
    int &n, int &ic, int &ih, int &iw, int &oc, int &oh, int &ow, int &g,
    int &kh, int &kw, int &sh, int &sw, int &ph, int &pw, int &dh, int &dw,
    bool &is_dw, bool &with_bias, bool &do_relu) {
  dh = p.dilation_h().getValue().getLimitedValue();
  dw = p.dilation_w().getValue().getLimitedValue();
  sh = p.stride_h().getValue().getLimitedValue();
  sw = p.stride_w().getValue().getLimitedValue();
  auto input_type = input->getType().template cast<TensorType>();
  std::vector<int64_t> i_s(input_type.getShape());
  auto output_type = output->getType().template cast<TensorType>();
  std::vector<int64_t> o_s(output_type.getShape());
  auto filter_type = filter->getType().template cast<TensorType>();
  std::vector<int64_t> f_s(filter_type.getShape());

  assert((i_s[0] == o_s[0]) && "input N not equal to output N");
  if(i_s.size() == 4){
    n = i_s[0];
    ic = i_s[1];
    ih = i_s[2];
    iw = i_s[3];
    oc = o_s[1];
    oh = o_s[2];
    ow = o_s[3];
  } else if(i_s.size() == 2){
    n = i_s[0];
    ic = i_s[1];
    ih = 1;
    iw = 1;
    oc = o_s[1];
    oh = 1;
    ow = 1;
  } else{
    assert(false);
  }
  kh = f_s[f_s.size() - 2];
  kw = f_s[f_s.size() - 1];

  auto padding = p.padding().getValue();
  if (padding == "SAME") {
    if (!is_deconv) {
      ph = findPadForSamePadding(ih, oh, kh, sh, dh);
      pw = findPadForSamePadding(iw, ow, kw, sw, dw);
    } else {
      assert(false && "not implemented yet for deconv with padding");
    }
  } else if (padding == "VALID") {
    ph = 0;
    pw = 0;
  } else {
    assert(false);
  }
  g = p.group().getValue().getLimitedValue();
  if (g != 1) {
    // f_s is in (g, oc/g, ic/g, kh, kw)
    assert(f_s.size() == 5);
    assert(g == f_s[0]);
    assert(oc/g == f_s[1]);
    assert(ic/g == f_s[2]);
    if (g == oc) {
      is_dw = true;
    } else {
      is_dw = false;
    }
  } else {
    assert(f_s.size() == 4);
    assert(oc == f_s[0]);
    assert(ic == f_s[1]);
    is_dw = false;
  }
  do_relu = p.do_relu().getValue();
  with_bias = p.with_bias().getValue();
}

void parsePoolParam(const tpu::PoolParam &p,
    Value *input, Value *output,
    int &n, int &c, int &ih, int &iw, int &oh, int &ow,
    int &kh, int &kw, int &sh, int &sw, int &pt, int &pb, int &pl, int &pr,
    bool &is_global, bool &do_relu) {
  kh = p.kernel_h().getValue().getLimitedValue();
  kw = p.kernel_w().getValue().getLimitedValue();
  sh = p.stride_h().getValue().getLimitedValue();
  sw = p.stride_w().getValue().getLimitedValue();
  auto input_type = input->getType().template cast<TensorType>();
  std::vector<int64_t> i_s(input_type.getShape());
  auto output_type = output->getType().template cast<TensorType>();
  std::vector<int64_t> o_s(output_type.getShape());
  assert((i_s[0] == o_s[0]) && "input N not equal to output N");
  assert((i_s[1] == o_s[1]) && "input C not equal to output C");
  n = i_s[0];
  c = i_s[1];
  ih = i_s[2];
  iw = i_s[3];
  oh = o_s[2];
  ow = o_s[3];
  pt = p.padding_t().getValue().getLimitedValue();
  pb = p.padding_b().getValue().getLimitedValue();
  pl = p.padding_l().getValue().getLimitedValue();
  pr = p.padding_r().getValue().getLimitedValue();
  is_global = false;
  if (kh == ih && kw == iw) {
    assert(oh == 1 && ow == 1);
    is_global = true;
  }
  do_relu = p.do_relu().getValue();
}

void parseFullyConnectedParam(
    Value *input, Value *output, Value *filter,
    int &m, int &k, int &n) {
  auto input_type = input->getType().template cast<TensorType>();
  std::vector<int64_t> i_s(input_type.getShape());
  auto output_type = output->getType().template cast<TensorType>();
  std::vector<int64_t> o_s(output_type.getShape());
  auto filter_type = filter->getType().cast<TensorType>();
  std::vector<int64_t> f_s(filter_type.getShape());
  assert((i_s[0] == o_s[0]) && "input M not equal to output M");
  m = i_s[0];
  // assuming transpose is false
  assert((i_s[1] == f_s[1]) && "input K not equal to filter K");
  k = i_s[1];
  assert((f_s[0] == o_s[1]) && "filter N not equal to output N");
  n = o_s[1];
}


} // namespace
