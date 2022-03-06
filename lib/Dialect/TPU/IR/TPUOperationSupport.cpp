#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "tpuc/CustomOpParam.h"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUCompressUtil.h"
#include "tpuc/TPUTensorSupport.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>
#include <cctype>

namespace mlir {

std::string toupper(std::string str) {
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char ch) { return std::toupper(ch); });
  return std::move(str);
}

void convertAttributesToOpParam(const DictionaryAttr &attrs, cvi::OpParam &param) {
  auto getBoolValue = [](const Attribute& attr) {
    return (bool)attr.cast<BoolAttr>().getValue();
  };
  auto getIntValue = [](const Attribute& attr) {
    return (int32_t)attr.cast<IntegerAttr>().getInt();
  };
  auto getFloatValue = [](const Attribute& attr) {
    return (float)attr.cast<FloatAttr>().getValue().convertToFloat();
  };
  auto getStringValue = [](const Attribute& attr) {
    return attr.cast<StringAttr>().getValue().str();
  };

  for (auto& a : attrs) {
    auto name = a.first.str();
    auto &attr = a.second;

    if (auto boolAttr = attr.dyn_cast<BoolAttr>()) {
      param.put<bool>(name, getBoolValue(attr));
    } else if (auto intAttr = attr.dyn_cast<IntegerAttr>()) {
      if (intAttr.getType().isInteger(16)) {
        param.put<int16_t>(name, (int16_t)getIntValue(attr));
      } else if (intAttr.getType().isInteger(8)) {
        param.put<int8_t>(name, (int8_t)getIntValue(attr));
      } else {
        param.put<int32_t>(name, (int32_t)getIntValue(attr));
      }
    } else if (auto floatAttr = attr.dyn_cast<FloatAttr>()) {
      param.put<float>(name, getFloatValue(attr));
    } else if (auto stringAttr = attr.dyn_cast<StringAttr>()) {
      param.put<std::string>(name, getStringValue(attr));
    } else if (auto array = attr.dyn_cast<ArrayAttr>()) {
      auto elementAttr = array[0];
      if (auto boolAttr = elementAttr.dyn_cast<BoolAttr>()) {
        std::vector<bool> vec;
        for (auto& item : array) {
          vec.push_back(getBoolValue(item));
        }
        param.put<std::vector<bool>>(name, vec);
      } else if (auto intAttr = elementAttr.dyn_cast<IntegerAttr>()) {
        std::vector<int32_t> vec;
        for (auto& item : array) {
          vec.push_back(getIntValue(item));
        }
        param.put<std::vector<int32_t>>(name, vec);
      } else if (auto floatAttr = elementAttr.dyn_cast<FloatAttr>()) {
        std::vector<float> vec;
        for (auto& item : array) {
          vec.push_back(getFloatValue(item));
        }
        param.put<std::vector<float>>(name, vec);
      } else if (auto stringAttr = elementAttr.dyn_cast<StringAttr>()) {
        std::vector<std::string> vec;
        for (auto& item : array) {
          vec.push_back(getStringValue(item));
        }
        param.put<std::vector<std::string>>(name, vec);
      } else {
        llvm_unreachable("unsupported attribute");
      }
    }
  }
}

void convertOpParamToAttributes(
    mlir::Builder &builder, cvi::OpParam &param,
    std::vector<NamedAttribute> &out) {
  auto isInt32 = [](std::shared_ptr<cvi::FieldBase> &f) {
    return strcmp(typeid(int32_t).name(), f->signature) == 0;
  };
  auto isFloat = [](std::shared_ptr<cvi::FieldBase> &f) {
    return strcmp(typeid(float).name(), f->signature) == 0;
  };
  auto isInt8 = [](std::shared_ptr<cvi::FieldBase> &f) {
    return strcmp(typeid(int8_t).name(), f->signature) == 0;
  };
  auto isBool = [](std::shared_ptr<cvi::FieldBase> &f) {
    return strcmp(typeid(bool).name(), f->signature) == 0;
  };
  auto isString = [](std::shared_ptr<cvi::FieldBase> &f) {
    return strcmp(typeid(std::string).name(), f->signature) == 0;
  };

  for (auto &it : param.fields) {
    if (isInt32(it.second)) {
      int32_t val = dynamic_cast<cvi::Field<int32_t> *>(it.second.get())->data;
      out.push_back(builder.getNamedAttr(it.first, builder.getI32IntegerAttr(val)));
    } else if (isInt8(it.second)) {
      int8_t val = dynamic_cast<cvi::Field<int8_t> *>(it.second.get())->data;
      out.push_back(builder.getNamedAttr(it.first, builder.getI8IntegerAttr(val)));
    } else if (isFloat(it.second)) {
      float val = dynamic_cast<cvi::Field<float> *>(it.second.get())->data;
      out.push_back(builder.getNamedAttr(it.first, builder.getF32FloatAttr(val)));
    } else if (isBool(it.second)) {
      bool val = dynamic_cast<cvi::Field<bool> *>(it.second.get())->data;
      out.push_back(builder.getNamedAttr(it.first, builder.getBoolAttr(val)));
    } else if (isString(it.second)) {
      std::string val = dynamic_cast<cvi::Field<std::string> *>(it.second.get())->data;
      out.push_back(builder.getNamedAttr(it.first, builder.getStringAttr(val)));
    } else {
      llvm::errs() << "field type:" << it.second->signature << "\n";
      assert(0);
    }
  }
}

void arrayAttrToVector(const ArrayAttr &arrayAttr,
                       std::vector<int32_t> &vector) {
  vector.clear();
  for (auto en : llvm::enumerate(arrayAttr)) {
    auto attr = en.value().dyn_cast<IntegerAttr>();
    vector.push_back(attr.getInt());
  }
}

void arrayAttrToVector(const ArrayAttr &arrayAttr, std::vector<float> &vector) {
  vector.clear();
  for (auto en : llvm::enumerate(arrayAttr)) {
    auto attr = en.value().dyn_cast<FloatAttr>();
    vector.push_back((float)attr.getValueAsDouble());
  }
}

llvm::StringRef getOpName(Operation *op) {
  if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpCommonInterface>(op)) {
    return tpuOp.getOpName();
  } else if (auto castOp = llvm::dyn_cast<tpu::LoadWeightOp>(op)) {
    return castOp.name();
  } else if (isa<ReturnOp>(op)) {
    return llvm::StringRef("std.return");
  } else {
    std::string errorMsg = std::string(__func__) + " failed, Op " +
                           op->getName().getStringRef().str() + "\n";
    llvm_unreachable(errorMsg.c_str());
  }
}

llvm::StringRef getPreviousOpName(Operation *op, uint index = 0) {
  if (op->getNumOperands() < (index + 1)) {
    llvm_unreachable("wrong index");
  }
  return getOpName(op->getOperand(index).getDefiningOp());
}

int getOpLayerId(Operation *op) {
  if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpCommonInterface>(op)) {
    // get op id according the line number of op's position.
    auto loc = op->getLoc().cast<FileLineColLoc>();
    return loc.getLine() - 3;
  } else {
    std::string errorMsg = std::string(__func__) + " failed, Op " +
                           op->getName().getStringRef().str() + "\n";
    llvm_unreachable(errorMsg.c_str());
  }
}

llvm::StringRef getOpQuant(Operation *op) {
  if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(op)) {
    return tpuOp.getOpQuant();
  } else {
    // if no quantization, return NONE
    return llvm::StringRef("NONE");
  }
}

LogicalResult setOpQuant(Operation *op, llvm::StringRef mode) {
  if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(op)) {
    return tpuOp.setOpQuantMode(mode);
  } else {
    std::string errorMsg = std::string(__func__) + " failed, Op " +
                           op->getName().getStringRef().str() + "\n";
    llvm_unreachable(errorMsg.c_str());
  }
}

void setOpResultType(Value value, Type eltType) {
  auto shape = value.getType().cast<TensorType>().getShape();
  auto type = RankedTensorType::get(shape, eltType);
  value.setType(type);
}

llvm::StringRef getOpQuantParamType(Operation *op) {
  if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(op)) {
    return tpuOp.getOpQuantParamType();
  } else {
    std::string errorMsg = std::string(__func__) + " failed, Op " +
                           op->getName().getStringRef().str() + "\n";
    llvm_unreachable(errorMsg.c_str());
  }
}

LogicalResult setOpQuantParamType(Operation *op, llvm::StringRef type) {
  if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(op)) {
    return tpuOp.setOpQuantParamType(type);
  } else {
    std::string errorMsg = std::string(__func__) + " failed, Op " +
                           op->getName().getStringRef().str() + "\n";
    llvm_unreachable(errorMsg.c_str());
  }
}

float getOpThreshold(Operation *op) {
  if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(op)) {
    return tpuOp.getOpQuantThreshold();
  } else {
    std::string errorMsg = std::string(__func__) + " failed, Op " +
                           op->getName().getStringRef().str() + "\n";
    llvm_unreachable(errorMsg.c_str());
  }
}

LogicalResult setOpThreshold(Operation *op, float threshold) {
  if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(op)) {
    return tpuOp.setOpQuantThreshold(threshold);
  } else {
    std::string errorMsg = std::string(__func__) + " failed, Op " +
                           op->getName().getStringRef().str() + "\n";
    llvm_unreachable(errorMsg.c_str());
  }
}

float getPreviousOpThreshold(Operation *op, uint index = 0) {
  if ( op->getNumOperands() < (index + 1) ) {
    std::string errorMsg = std::string(__func__) + " failed, Op " +
                           op->getName().getStringRef().str() + "\n";
    llvm_unreachable(errorMsg.c_str());
  }
  auto formerOp = op->getOperand(index).getDefiningOp();
  return getOpThreshold(formerOp);
}

uint64_t getOpAddress(Operation *op) {
  if (isa<tpu::TpuTGOpCodegenInterface>(op)) {
    auto tpuTGOp = llvm::dyn_cast<tpu::TpuTGOpCodegenInterface>(op);
    return tpuTGOp.getGAddr();
  } else if (auto castOp = llvm::dyn_cast<tpu::GenericCpuOp>(op)) {
    if (castOp.gaddr().hasValue()) {
      return castOp.gaddr().getValue();
    }
    llvm_unreachable("unsupported op");
  } else if (isa<tpu::TpuTLOpCodegenInterface>(op)) {
    auto tpuTLOp = llvm::dyn_cast<tpu::TpuTLOpCodegenInterface>(op);
    return tpuTLOp.getGAddr();
  } else if (auto inputOp = llvm::dyn_cast<tpu::InputOp>(op)) {
    if (inputOp.gaddr().hasValue()) {
      return inputOp.gaddr().getValue();
    }
  } else if (isa<tpu::NoneOp>(op)) {
    return 0;
  } else {
    std::string errorMsg = std::string(__func__) + " failed, Op " +
                           op->getName().getStringRef().str() + "\n";
    llvm_unreachable(errorMsg.c_str());
  }
  return 0;
}

LogicalResult setOpAddress(Operation *op, uint64_t gaddr) {
  if (auto tpuTGOp = llvm::dyn_cast<tpu::TpuTGOpCodegenInterface>(op)) {
    return tpuTGOp.setGAddr(gaddr);
  } else if (auto tpuTLOp = llvm::dyn_cast<tpu::TpuTLOpCodegenInterface>(op)) {
    return tpuTLOp.setGAddr(gaddr);
  }  else if (auto castOp = llvm::dyn_cast<tpu::GenericCpuOp>(op)) {
    castOp->setAttr("gaddr", Builder(castOp.getOperation()->getContext()).getI64IntegerAttr(gaddr));
    return success();
  } else if (auto inputOp = llvm::dyn_cast<tpu::InputOp>(op)) {
    inputOp->setAttr("gaddr", Builder(inputOp.getOperation()->getContext()).getI64IntegerAttr(gaddr));
  } else {
    std::string errorMsg = std::string(__func__) + " failed, Op " +
                           op->getName().getStringRef().str() + "\n";
    llvm_unreachable(errorMsg.c_str());
  }
  return failure();
}

uint64_t getWeightOpAddress(Operation *op) {
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(op)) {
    return cast_op.offset().getValue();
  } else {
    std::string errorMsg = std::string(__func__) + " failed, Op " +
                           op->getName().getStringRef().str() + "\n";
    llvm_unreachable(errorMsg.c_str());
  }
  return 0;
}

uint64_t getPreviousOpAddress(Operation *op, uint index = 0) {
  if (op->getNumOperands() < (index + 1)) {
    llvm_unreachable("index exceeds the number of operations");
  }
  auto formerOp = op->getOperand(index).getDefiningOp();
  if (isa<tpu::LoadWeightOp>(formerOp)) {
    return getWeightOpAddress(formerOp);
  } else {
    return getOpAddress(formerOp);
  }
}

Operation* getNextOp(Operation *op) {
  Operation *nextOp = nullptr;
  if (op->getResult(0).hasOneUse()) {
    for (auto &use : op->getResult(0).getUses()) {
      nextOp = use.getOwner();
      break;
    }
    assert(nextOp && "nextOp is nullptr");
  }
  // if not found, will return NULL
  return nextOp;
}

LogicalResult setOpBufferReused(Operation *op, bool flag) {
  if (auto tpuTGOp = llvm::dyn_cast<tpu::TpuTGOpCodegenInterface>(op)) {
    return tpuTGOp.setBufferReused(flag);
  } else {
    std::string errorMsg = std::string(__func__) + " failed, Op " +
                           op->getName().getStringRef().str() + "\n";
    llvm_unreachable(errorMsg.c_str());
  }
  return failure();
}

tpu::QuantParam getDefaultQuantParam(Builder &builder) {
  return tpu::QuantParam::get(
      builder.getStringAttr("NONE"),
      builder.getStringAttr("NONE"),
      builder.getF32FloatAttr(0.0),
      builder.getContext());
}

bool isOpSupportRelu(Operation *op) {
  if (isa<tpu::Conv2DOp>(op) || isa<tpu::Conv3DOp>(op) ||
      isa<tpu::DeConv2DOp>(op) || isa<tpu::EltwiseAddOp>(op) ||
      isa<tpu::EltwiseMaxOp>(op) || isa<tpu::EltwiseMinOp>(op) ||
      isa<tpu::EltwiseMulOp>(op) || isa<tpu::MatMulOp>(op) ||
      isa<tpu::FullyConnectedOp>(op) || isa<tpu::BroadcastMulOp>(op) ||
      isa<tpu::BroadcastAddOp>(op) || isa<tpu::MulConstOp>(op) ||
      isa<tpu::ScaleOp>(op) || isa<tpu::ConcatOp>(op)) {
    return true;
  }
  return false;
}

void parseConvParam(const tpu::ConvParam &p, bool is_deconv, Value input,
                    Value output, int &n, int &ic, int &ih, int &iw, int &oc,
                    int &oh, int &ow, int &g, int &kh, int &kw, int &ins_h,
                    int &ins_w, int &sh, int &sw, int &pt, int &pb, int &pl,
                    int &pr, int &dh, int &dw, bool &is_dw, bool &with_bias,
                    int &pad_value) {
  kh = p.kernel_h().getInt();
  kw = p.kernel_w().getInt();
  dh = p.dilation_h().getInt();
  dw = p.dilation_w().getInt();
  sh = p.stride_h().getInt();
  sw = p.stride_w().getInt();
  pt = p.padding_t().getInt();
  pb = p.padding_b().getInt();
  pl = p.padding_l().getInt();
  pr = p.padding_r().getInt();
  is_dw = p.is_dw().getValue();
  auto i_s = getTensorShape(input);
  auto o_s = getTensorShape(output);
  if (kw == 0) {
    kw = 1;
    i_s.push_back(1);
    o_s.push_back(1);
  }

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
  } else if (i_s.size() == 3) {
    n = i_s[0];
    ic = i_s[1];
    ih = i_s[2];
    iw = 1;
    oc = o_s[1];
    oh = o_s[2];
    ow = 1;
  } else{
    output.dump();
    llvm_unreachable("unsupported shape size");
  }
  std::vector<int32_t> ins;
  arrayAttrToVector(p.ins(), ins);
  ins.resize(2, 0);
  ins_h = ins[1];
  ins_w = ins[0];
  g = p.group().getInt();
  with_bias = p.with_bias().getValue();
  pad_value= p.pad_value().getInt();
}

void parseConv3dParam(const tpu::Conv3dParam &p, bool is_deconv,
    Value input, Value output,
    int &n, int &ic, int &id, int &ih, int &iw,
    int &oc, int &od, int &oh, int &ow, int &g,
    int &kd, int &kh, int &kw,
    int &sd, int &sh, int &sw,
    int &pd0, int &pd1, int &pt, int &pb, int &pl, int &pr,
    int &dd, int &dh, int &dw,
    bool &is_dw, bool &with_bias) {
  kd = p.kernel_d().getInt();
  kh = p.kernel_h().getInt();
  kw = p.kernel_w().getInt();
  dd = p.dilation_d().getInt();
  dh = p.dilation_h().getInt();
  dw = p.dilation_w().getInt();
  sd = p.stride_d().getInt();
  sh = p.stride_h().getInt();
  sw = p.stride_w().getInt();
  pd0 = p.padding_d0().getInt();
  pd1 = p.padding_d1().getInt();
  pt = p.padding_t().getInt();
  pb = p.padding_b().getInt();
  pl = p.padding_l().getInt();
  pr = p.padding_r().getInt();
  is_dw = p.is_dw().getValue();
  auto input_type = input.getType().template cast<TensorType>();
  std::vector<int64_t> i_s(input_type.getShape());
  auto output_type = output.getType().template cast<TensorType>();
  std::vector<int64_t> o_s(output_type.getShape());

  assert((i_s[0] == o_s[0]) && "input N not equal to output N");
  if (i_s.size() == 5) {
    n = i_s[0];
    ic = i_s[1];
    id = i_s[2];
    ih = i_s[3];
    iw = i_s[4];
    oc = o_s[1];
    od = o_s[2];
    oh = o_s[3];
    ow = o_s[4];
  } else{
    llvm_unreachable("unsupported shape size");
  }

  g = p.group().getInt();
  with_bias = p.with_bias().getValue();
}

void parsePoolParam(const tpu::PoolParam &p, Value input, Value output, int &n,
                    int &c, int &ih, int &iw, int &oh, int &ow, int &kh,
                    int &kw, int &sh, int &sw, int &pt, int &pb, int &pl,
                    int &pr, int &pad_value, bool &is_global,
                    bool &count_include_pad) {
  kh = p.kernel_h().getInt();
  kw = p.kernel_w().getInt();
  sh = p.stride_h().getInt();
  sw = p.stride_w().getInt();
  auto i_s = getTensorShape(input);
  auto o_s = getTensorShape(output);
  if (kw == 0) {
    kw = 1;
    i_s.push_back(1);
    o_s.push_back(1);
  }
  assert(i_s.size() == o_s.size());
  int num_dims = i_s.size();
  if (num_dims >= 3) {
    n = std::accumulate(i_s.begin(), i_s.begin() + num_dims - 3, 1,
                        std::multiplies<int64_t>());
    c = i_s[num_dims - 3];
  } else if (num_dims == 2) {
    n = 1;
    c = 1;
  } else {
    llvm_unreachable("input num dims error\n");
  }
  ih = i_s[num_dims - 2];
  iw = i_s[num_dims - 1];
  oh = o_s[num_dims - 2];
  ow = o_s[num_dims - 1];
  pt = p.padding_t().getInt();
  pb = p.padding_b().getInt();
  pl = p.padding_l().getInt();
  pr = p.padding_r().getInt();
  is_global = false;
  if (kh == ih && kw == iw && oh == 1 && ow == 1) {
    // assert(oh == 1 && ow == 1);
    is_global = true;
  }
  pad_value = p.pad_value().getInt();
  count_include_pad = p.count_include_pad().getValue();
}

void parsePool3dParam(const tpu::Pool3dParam &p,
    Value input, Value output,
    int &n, int &c, int &id, int &ih, int &iw,
    int &od, int &oh, int &ow,
    int &kd, int &kh, int &kw,
    int &sd, int &sh, int &sw,
    int &pd0, int &pd1, int &pt, int &pb, int &pl, int &pr,
    bool &is_global, bool &count_include_pad) {
  kd = p.kernel_d().getInt();
  kh = p.kernel_h().getInt();
  kw = p.kernel_w().getInt();
  sd = p.stride_d().getInt();
  sh = p.stride_h().getInt();
  sw = p.stride_w().getInt();
  auto input_type = input.getType().template cast<TensorType>();
  std::vector<int64_t> i_s(input_type.getShape());
  auto output_type = output.getType().template cast<TensorType>();
  std::vector<int64_t> o_s(output_type.getShape());
  assert((i_s[0] == o_s[0]) && "input N not equal to output N");
  assert((i_s[1] == o_s[1]) && "input C not equal to output C");
  n = i_s[0];
  c = i_s[1];
  id = i_s[2];
  ih = i_s[3];
  iw = i_s[4];
  od = o_s[2];
  oh = o_s[3];
  ow = o_s[4];
  pd0 = p.padding_d0().getInt();
  pd1 = p.padding_d1().getInt();
  pt = p.padding_t().getInt();
  pb = p.padding_b().getInt();
  pl = p.padding_l().getInt();
  pr = p.padding_r().getInt();
  is_global = false;
  if (kh == ih && kw == iw && oh == 1 && ow == 1) {
    //assert(oh == 1 && ow == 1);
    is_global = true;
  }
  count_include_pad = p.count_include_pad().getValue();
}

// [4, 3, 28] dot [4,5,28] => [4,3,5] : batch = 4, m = 3, k = 28, n = 5
// [4, 3, 28] dot [5,28]   => [4,3,5] : batch = 1, m = 12, k = 28, n = 5
template <typename OpTy>
void parseFullyConnectedParam(Operation *op, int &batch_high, int &batch_low,
                              int &m, int &k, int &n)
{
  auto castOp = cast<OpTy>(op);
  auto a_s = getTensorShape(castOp.input());
  auto b_s = getTensorShape(castOp.filter());
  auto o_s = getTensorShape(castOp.output());
  bool input_trans = castOp.input_transpose();
  bool output_trans = castOp.output_transpose();
  size_t o_dim = o_s.size();
  size_t b_dim = b_s.size();
  assert(b_dim >= 2);
  k = b_s[b_dim - 1];
  n = b_s[b_dim - 2];
  batch_low = 1;
  batch_high = 1;
  if (input_trans) {
    batch_low = a_s[o_dim - 2];
    m = a_s[o_dim - 3];
    batch_high = std::accumulate(a_s.data(), a_s.data() + o_dim - 3, 1,
                                 std::multiplies<int64_t>());
  } else if (output_trans) {
    batch_low = o_s[o_dim - 2];
    m = o_s[o_dim - 3];
    batch_high = std::accumulate(o_s.data(), o_s.data() + o_dim - 3, 1,
                                 std::multiplies<int64_t>());
  } else {
    batch_low = std::accumulate(b_s.data(), b_s.data() + b_dim - 2, 1,
                                std::multiplies<int64_t>());
    if (batch_low > 1) {
      m = a_s[o_dim - 2];
    } else {
      m = std::accumulate(a_s.data(), a_s.data() + o_dim - 1, 1,
                          std::multiplies<int64_t>());
    }
  }
}

template void parseFullyConnectedParam<tpu::FullyConnectedOp>(
    Operation *op, int &batch_high, int &batch_low, int &m, int &k, int &n);
template void parseFullyConnectedParam<tpu::TG_INT8_FullyConnectedOp>(
    Operation *op, int &batch_high, int &batch_low, int &m, int &k, int &n);
template void parseFullyConnectedParam<tpu::TG_BF16_FullyConnectedOp>(
    Operation *op, int &batch_high, int &batch_low, int &m, int &k, int &n);

template <typename OpTy>
void parseMatMulParam(Operation *op, int &batch_high, int &batch_low, int &m,
                      int &k, int &n) {
  auto castOp = cast<OpTy>(op);
  auto a_s = getTensorShape(op->getOperand(0));
  auto b_s = getTensorShape(op->getOperand(1));
  auto o_s = getTensorShape(castOp.output());
  bool left_trans = castOp.left_transpose();
  bool right_trans = castOp.right_transpose();
  bool output_trans = castOp.output_transpose();
  int64_t axis = o_s.size() - 1;
  batch_low = 1;
  batch_high = 1;
  if (left_trans || right_trans || output_trans) {
    // if has tranpose, num_dims == 4 or 3
    k = a_s[axis];
    n = o_s[axis];
    if (left_trans) {
      batch_low = a_s[axis - 1];
      m = a_s[axis - 2];
    } else {
      batch_low = a_s[axis - 2];
      m = a_s[axis - 1];
    }
    if (axis == 3) {
      batch_high = a_s[0];
    }
    return;
  }

  for (int i = 0; i < axis - 1; i++) {
    assert((a_s[i] == o_s[i]) && "lhs B not equal to output B");
    assert((a_s[i] == b_s[i]) && "lhs B not equal to rhs B");
    batch_high *= a_s[i];
  }
  m = a_s[axis - 1];
  k = a_s[axis];
  assert((k == b_s[axis - 1]) && "lhs K not equal to rhs K");
  assert((m == o_s[axis - 1]) && "lhs M not equal to output M");
  n = b_s[axis];
  assert((n == o_s[axis]) && "rhs N not equal to output N");
}

template void parseMatMulParam<tpu::MatMulOp>(Operation *op, int &batch_high,
                                              int &batch_low, int &m, int &k,
                                              int &n);
template void parseMatMulParam<tpu::TG_INT8_MatMulOp>(Operation *op,
                                                      int &batch_high,
                                                      int &batch_low, int &m,
                                                      int &k, int &n);
template void parseMatMulParam<tpu::TG_BF16_MatMulOp>(Operation *op,
                                                      int &batch_high,
                                                      int &batch_low, int &m,
                                                      int &k, int &n);

template<typename T>
static int remove_value(std::vector<T> & v, T value) {
  int idx = 0;
  for(auto iter = v.begin(); iter != v.end(); iter++, idx++) {
    if (*iter == value) {
      v.erase(iter);
      return idx;
    }
  }
  return -1;
}

static void refresh(std::vector<int> &order, int idx) {
  for(auto &v:order) {
    if (v > idx) {
      v--;
    }
  }
}

template <typename OpTy>
void parsePermuteParam(Operation *op, std::vector<int64_t> &shape_4,
                       std::vector<int> &order_4) {
  auto pmOp = llvm::dyn_cast<OpTy>(op);
  auto shape = getTensorShape(pmOp.input());
  std::vector<int> order;
  arrayAttrToVector(pmOp.order(), order);
  int num_dims = order.size();
  if (num_dims > 4) {
    // remove dims = 1
    while(num_dims > 4) {
      int idx = remove_value<int64_t>(shape, 1);
      if (idx < 0) {
        break;
      }
      remove_value(order, idx);
      refresh(order, idx);
      num_dims--;
    }
    // remove continous order
    while(num_dims > 4) {
      bool done = false;
      for (int i = 0; i < num_dims-1; i++) {
        if (order[i] +1 == order[i+1]) {
          int idx = order[i];
          shape[idx] *= shape[idx+1];
          shape.erase(shape.begin() + idx + 1);
          order.erase(order.begin() + i + 1);
          refresh(order, idx + 1);
          num_dims--;
          done = true;
          break;
        }
      }
      if (done == false) {
        break;
      }
    }
    if (num_dims > 4) {
      llvm_unreachable("permute shape not support");
    }
  }
  order_4 = {0, 1, 2, 3};
  shape_4 = {1, 1, 1, 1};
  for (int end = num_dims - 1, idx = 3; end >= 0 && idx >= 0; end--, idx--) {
    shape_4[idx] = shape[end];
    order_4[idx] = order[end] + idx - end;
  }
}

template void parsePermuteParam<tpu::PermuteOp>(Operation *op,
                                                std::vector<int64_t> &shape_4,
                                                std::vector<int> &order_4);
template void parsePermuteParam<tpu::TG_INT8_PermuteOp>(
    Operation *op, std::vector<int64_t> &shape_4, std::vector<int> &order_4);
template void parsePermuteParam<tpu::TG_BF16_PermuteOp>(
    Operation *op, std::vector<int64_t> &shape_4, std::vector<int> &order_4);

template <typename OpTy>
void parseCopyParam(Operation *op, std::vector<int> &shape_4,
                    std::vector<int> &i_stride_4,
                    std::vector<int> &o_stride_4) {
  auto castOp = cast<OpTy>(op);
  std::vector<int> shape;
  std::vector<int> i_stride;
  std::vector<int> o_stride;
  arrayAttrToVector(castOp.shape(), shape);
  arrayAttrToVector(castOp.input_stride(), i_stride);
  arrayAttrToVector(castOp.output_stride(), o_stride);
  shape_4 = {1, 1, 1, 1};
  i_stride_4 = {0, 0, 0, 0};
  o_stride_4 = {0, 0, 0, 0};
  int num_dims = shape.size();
  assert(num_dims <= 4);
  assert(i_stride.size() == shape.size());
  assert(o_stride.size() == shape.size());
  for (int end = num_dims - 1, idx = 3; end >= 0 && idx >= 0; end--, idx--) {
    shape_4[idx] = shape[end];
    i_stride_4[idx] = i_stride[end];
    o_stride_4[idx] = o_stride[end];
  }
}
template void parseCopyParam<tpu::CopyOp>(Operation *op,
                                          std::vector<int> &shape_4,
                                          std::vector<int> &i_stride_4,
                                          std::vector<int> &o_stride_4);
template void parseCopyParam<tpu::TG_INT8_CopyOp>(Operation *op,
                                                  std::vector<int> &shape_4,
                                                  std::vector<int> &i_stride_4,
                                                  std::vector<int> &o_stride_4);
template void parseCopyParam<tpu::TG_BF16_CopyOp>(Operation *op,
                                                  std::vector<int> &shape_4,
                                                  std::vector<int> &i_stride_4,
                                                  std::vector<int> &o_stride_4);

template <typename OpTy>
void parseCropParam(Operation *op, std::vector<int64_t> &is_4,
                    std::vector<int64_t> &os_4, std::vector<int> &offset_4,
                    std::vector<int> &step_4, bool &fusible) {
  auto castOp = llvm::dyn_cast<OpTy>(op);
  auto is = getTensorShape(castOp.input());
  auto os = getTensorShape(castOp.output());
  int num_dims = is.size();
  std::vector<int> crop_offset;
  std::vector<int> steps;
  arrayAttrToVector(castOp.crop_offset(), crop_offset);
  if (castOp.steps().hasValue()) {
    arrayAttrToVector(castOp.steps().getValue(), steps);
  } else {
    steps.assign(num_dims, 1);
  }

  assert(crop_offset.size() == steps.size());
  assert(is.size() == steps.size());
  assert(is.size() == os.size());

  if (num_dims > 4) {
    // remove dims = 1
    while (num_dims > 4) {
      int idx = remove_value<int64_t>(is, 1);
      if (idx < 0) {
        break;
      }
      crop_offset.erase(crop_offset.begin() + idx);
      steps.erase(steps.begin() + idx);
      os.erase(os.begin() + idx);
      num_dims--;
    }
    // remove continous
    while (num_dims > 4) {
      bool done = false;
      for (int i = 0; i < num_dims - 1; i++) {
        if (is[i] == os[i] && is[i + 1] == os[i + 1]) {
          is[i] *= is[i + 1];
          os[i] *= os[i + 1];
          is.erase(is.begin() + i + 1);
          os.erase(os.begin() + i + 1);
          steps.erase(steps.begin() + i + 1);
          crop_offset.erase(crop_offset.begin() + i + 1);
          num_dims--;
          done = true;
          break;
        }
      }
      if (done == false) {
        break;
      }
    }
    if (num_dims > 4) {
      llvm_unreachable("permute shape not support");
    }
  }
  is_4 = {1, 1, 1, 1};
  os_4 = {1, 1, 1, 1};
  step_4 = {1, 1, 1, 1};
  offset_4 = {0, 0, 0, 0};
  std::vector<int>real_axes;
  bool no_step = true;
  for (int idx = 0; idx < num_dims; idx++) {
    is_4[idx] = is[idx];
    os_4[idx] = os[idx];
    step_4[idx] = steps[idx];
    offset_4[idx] = crop_offset[idx];
    if (no_step && steps[idx] != 1) {
      no_step = false;
    }
    if (is_4[idx] != os_4[idx]) {
      real_axes.push_back(idx);
    }
  }
  fusible = false;
  if (no_step && real_axes.size() == 1) {
    int axis = real_axes[0];
    int outer_dim = std::accumulate(is_4.begin(), is_4.begin() + axis, 1, std::multiplies<int64_t>());
    if (outer_dim == 1) {
      fusible = true;
    }
  }
}

template void parseCropParam<tpu::CropOp>(Operation *op,
                                          std::vector<int64_t> &is_4,
                                          std::vector<int64_t> &os_4,
                                          std::vector<int> &offset_4,
                                          std::vector<int> &step_4,
                                          bool &fusible);
template void parseCropParam<tpu::TG_INT8_CropOp>(Operation *op,
                                                  std::vector<int64_t> &is_4,
                                                  std::vector<int64_t> &os_4,
                                                  std::vector<int> &offset_4,
                                                  std::vector<int> &step_4,
                                                  bool &fusible);
template void parseCropParam<tpu::TG_BF16_CropOp>(Operation *op,
                                                  std::vector<int64_t> &is_4,
                                                  std::vector<int64_t> &os_4,
                                                  std::vector<int> &offset_4,
                                                  std::vector<int> &step_4,
                                                  bool &fusible);

template <typename OpTy>
void parsePadParam(Operation *op, std::vector<int64_t> &is_4,
                   std::vector<int64_t> &os_4, std::vector<int> &pad_4) {
  auto castOp = llvm::dyn_cast<OpTy>(op);
  auto is = getTensorShape(castOp.input());
  auto os = getTensorShape(castOp.output());
  int num_dims = is.size();
  std::vector<int> pads;
  arrayAttrToVector(castOp.pads(), pads);
  assert(is.size() * 2 == pads.size());
  assert(is.size() == os.size());

  if (num_dims > 4) {
    // remove continous
    while (num_dims > 4) {
      bool done = false;
      for (int i = 0; i < num_dims - 1; i++) {
        if (is[i] == os[i] && is[i + 1] == os[i + 1]) {
          is[i] *= is[i + 1];
          os[i] *= os[i + 1];
          is.erase(is.begin() + i + 1);
          os.erase(os.begin() + i + 1);
          pads.erase(pads.begin() + i + 1);
          num_dims--;
          done = true;
          break;
        }
      }
      if (done == false) {
        break;
      }
    }
    if (num_dims > 4) {
      llvm_unreachable("Pad shape not support");
    }
  }
  is_4 = {1, 1, 1, 1};
  os_4 = {1, 1, 1, 1};
  pad_4 = {0, 0, 0, 0, 0, 0, 0, 0};
  switch (num_dims) {
  case 1:
    is_4[3] = is[0];
    os_4[3] = os[0];
    pad_4[3] = pads[0];
    pad_4[7] = pads[1];
    break;
  case 2:
    is_4[1] = is[0];
    is_4[3] = is[1];
    os_4[1] = os[0];
    os_4[3] = os[1];
    pad_4[1] = pads[0];
    pad_4[3] = pads[1];
    pad_4[5] = pads[2];
    pad_4[7] = pads[3];
    break;
  default:
    for (int idx = 0; idx < num_dims; idx++) {
      is_4[idx] = is[idx];
      os_4[idx] = os[idx];
      pad_4[idx] = pads[idx];
      pad_4[idx + 4] = pads[idx + num_dims];
    }
    break;
  }
}

template void parsePadParam<tpu::PadOp>(Operation *op,
                                        std::vector<int64_t> &is_4,
                                        std::vector<int64_t> &os_4,
                                        std::vector<int> &pad_4);
template void parsePadParam<tpu::TG_INT8_PadOp>(Operation *op,
                                                std::vector<int64_t> &is_4,
                                                std::vector<int64_t> &os_4,
                                                std::vector<int> &pad_4);
template void parsePadParam<tpu::TG_BF16_PadOp>(Operation *op,
                                                std::vector<int64_t> &is_4,
                                                std::vector<int64_t> &os_4,
                                                std::vector<int> &pad_4);

template <typename OpTy>
void parseLeakyReluParam(Operation *op, int8_t &pos_rshift, int8_t &pos_m_i8,
                         int8_t &neg_rshift, int8_t &neg_m_i8,
                         float &negative_slope) {
  auto lreluOp = llvm::dyn_cast<OpTy>(op);
  assert(lreluOp);

  if (lreluOp.m_i8_pos().hasValue()) {
    pos_m_i8 = lreluOp.m_i8_pos().getValue();
    pos_rshift = lreluOp.rshift_pos().getValue();
    assert(pos_m_i8);
  } else {
    pos_m_i8 = 0;
    pos_rshift = 0;
  }

  if (lreluOp.m_i8_neg().hasValue()) {
    neg_m_i8 = lreluOp.m_i8_neg().getValue();
    neg_rshift = lreluOp.rshift_neg().getValue();
    assert(neg_m_i8);
  } else {
    neg_m_i8 = 0;
    neg_rshift = 0;
  }

  negative_slope = lreluOp.negative_slope().getValue().convertToFloat();
}

template void parseLeakyReluParam<tpu::TG_INT8_Conv2DOp>(
    Operation *op, int8_t &pos_rshift, int8_t &pos_m_i8,
    int8_t &neg_rshift, int8_t &neg_m_i8, float &negative_slope);
template void parseLeakyReluParam<tpu::TG_INT8_LeakyReluOp>(
    Operation *op, int8_t &pos_rshift, int8_t &pos_m_i8,
    int8_t &neg_rshift, int8_t &neg_m_i8, float &negative_slope);
template void parseLeakyReluParam<tpu::TG_BF16_LeakyReluOp>(
    Operation *op, int8_t &pos_rshift, int8_t &pos_m_i8,
    int8_t &neg_rshift, int8_t &neg_m_i8, float &negative_slope);

bool isBf16Tensor(Value val) {
  auto valType = val.getType().dyn_cast<TensorType>();
  auto elementType = valType.getElementType();
  return elementType.isBF16();
}

// Tiled compressed activation split as (tiled_n, tiled_c, tiled_h, W)
// Global memory shape: (N/tiled_n, C/tiled_c, tiled_h, tiled_c, W)
//
// output shape       (1, 64, 112, 112)
// tiled output shape (1, 32,  26, 112)
//
// tiled TDMA store
//   (1, 32, 26, 112)      c=[31:0]
//   (1, 32, 26, 112)
//   (1, 32, 26, 112)
//   (1, 32, 26, 112)
//   (1, 32,  8, 112)
//   ----------------
//   (1, 32, 26, 112)      c=[63:32]
//   (1, 32, 26, 112)
//   (1, 32, 26, 112)
//   (1, 32, 26, 112)
//   (1, 32,  8, 112)
//
//  n    h  c  w
//  0    0  0  0       | header | compressed shape (1, h=26, c=32, w) |
//  0   26  0  0
//
//  0  104  0  0
//  -------------------------------------------------------------------
//  0   0  32  0
//  0  26  32  0
//
//  0  104 32  0
//
void getTiledCompressedSize(int n, int c, int h, int w, int n_step, int c_step,
    int h_step, int isBf16, int64_t &stepSize, int64_t &totalSize) {
  int data_type_size = isBf16 ? 2 : 1;
  int tiledSize = n_step * c_step * h_step * w * data_type_size;
  stepSize = getCompressedDataSize(tiledSize, isBf16);

  // Compressed tiled activation size
  //   ceil(batch, tiled_n) * ceil(channel/tiled_channel) *
  //   ceil(height/tiled_height) * compressed(step_size)
  totalSize = llvm::divideCeil(n, n_step) *
              llvm::divideCeil(c, c_step) *
              llvm::divideCeil(h, h_step) *
              stepSize;
}

void getTiledCompressedActSize(Operation *op, int n_step, int oc_step,
    int oh_step, int ow, int64_t &stepSize, int64_t &totalSize) {
  int64_t resultSize;
  std::vector<int64_t> shapes;
  getTensorShapeAndSize(op->getResult(0), shapes, resultSize);

  int isBf16 = isBf16Tensor(op->getResult(0));

  getTiledCompressedSize(shapes[0], shapes[1], shapes[2], shapes[3], n_step,
                         oc_step, oh_step, isBf16, stepSize, totalSize);
}

int getDataTypeSize(Value val) {
  unsigned sizeInBits = 0;

  auto eltType = val.getType();
  if (auto tp = eltType.dyn_cast<RankedTensorType>()) {
    auto eltType = tp.getElementType();
    sizeInBits = eltType.getIntOrFloatBitWidth();
  } else if (auto tp = eltType.dyn_cast<MemRefType>()) {
    auto eltType = tp.getElementType();
    sizeInBits = eltType.getIntOrFloatBitWidth();
  }

  return (int)llvm::divideCeil(sizeInBits, 8);
}

} // namespace
