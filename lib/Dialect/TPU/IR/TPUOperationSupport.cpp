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

namespace mlir {

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

void parseConvParam(const tpu::ConvParam &p, bool is_deconv,
    Value input, Value output, Value filter,
    int &n, int &ic, int &ih, int &iw, int &oc, int &oh, int &ow, int &g,
    int &kh, int &kw, int &ins_h, int &ins_w, int &sh, int &sw, int &pt, int &pb, int &pl, int &pr, int &dh, int &dw,
    bool &is_dw, bool &with_bias, bool &do_relu, int &pad_value) {
  dh = p.dilation_h().getInt();
  dw = p.dilation_w().getInt();
  sh = p.stride_h().getInt();
  sw = p.stride_w().getInt();
  pt = p.padding_t().getInt();
  pb = p.padding_b().getInt();
  pl = p.padding_l().getInt();
  pr = p.padding_r().getInt();
  auto input_type = input.getType().template cast<TensorType>();
  std::vector<int64_t> i_s(input_type.getShape());
  auto output_type = output.getType().template cast<TensorType>();
  std::vector<int64_t> o_s(output_type.getShape());
  auto filter_type = filter.getType().template cast<TensorType>();
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
  } else if (i_s.size() == 3) {
    n = i_s[0];
    ic = i_s[1];
    ih = i_s[2];
    iw = 1;
    oc = o_s[1];
    oh = o_s[2];
    ow = 1;
  } else{
    llvm_unreachable("unsupported shape size");
  }
  kh = f_s[f_s.size() - 2];
  kw = f_s[f_s.size() - 1];

  std::vector<int32_t> ins;
  arrayAttrToVector(p.ins(), ins);
  ins.resize(2, 0);
  ins_h = ins[1];
  ins_w = ins[0];

  g = p.group().getInt();
  if (g != 1 || f_s.size() == 5) {
    if (g == oc && g == ic) {
      is_dw = true;
    } else {
      is_dw = false;
    }
    // f_s is in (g, oc/g, ic/g, kh, kw)
    if(f_s.size() == 5) {
      assert(g == f_s[0]);
      assert(oc/g == f_s[1]);
      assert(ic/g == f_s[2]);
    } else if (f_s.size() == 4) {
      // tl_layer has filter size of 4
      if (is_dw) {
        // (1, oc, kh, kw)
        assert(ic/g == 1);
        assert(oc == f_s[1]);
      } else {
        // (oc, ic/g, kh, kw)
        assert(oc == f_s[0]);
        assert(ic/g == f_s[1]);
      }
    }
  } else {
    assert(f_s.size() == 4);
    assert(oc == f_s[0]);
    assert(ic == f_s[1] || (ic % 2 != 0));
    is_dw = false;
  }
  do_relu = p.do_relu().getValue();
  with_bias = p.with_bias().getValue();
  pad_value= p.pad_value().getInt();
}

void parseConv3dParam(const tpu::Conv3dParam &p, bool is_deconv,
    Value input, Value output, Value filter,
    int &n, int &ic, int &id, int &ih, int &iw,
    int &oc, int &od, int &oh, int &ow, int &g,
    int &kd, int &kh, int &kw,
    int &sd, int &sh, int &sw,
    int &pd0, int &pd1, int &pt, int &pb, int &pl, int &pr,
    int &dd, int &dh, int &dw,
    bool &is_dw, bool &with_bias, bool &do_relu) {
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
  auto input_type = input.getType().template cast<TensorType>();
  std::vector<int64_t> i_s(input_type.getShape());
  auto output_type = output.getType().template cast<TensorType>();
  std::vector<int64_t> o_s(output_type.getShape());
  auto filter_type = filter.getType().template cast<TensorType>();
  std::vector<int64_t> f_s(filter_type.getShape());

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
  kd = f_s[f_s.size() - 3];
  kh = f_s[f_s.size() - 2];
  kw = f_s[f_s.size() - 1];

  g = p.group().getInt();
  if (g != 1 || f_s.size() == 5) {
    if (g == oc && g == ic) {
      is_dw = true;
    } else {
      is_dw = false;
    }

    // f_s is in (g, oc/g, ic/g, kd, kh, kw)
    if(f_s.size() == 6) {
      assert(g == f_s[0]);
      assert(oc/g == f_s[1]);
      assert(ic/g == f_s[2]);
    } else if (f_s.size() == 5) {
      // tl_layer has filter size of 5
      if (is_dw) {
        // (1, oc, kd, kh, kw)
        assert(ic/g == 1);
        assert(oc == f_s[1]);
      } else {
        // (oc, ic/g, kd, kh, kw)
        assert(oc == f_s[0]);
        assert(ic/g == f_s[1]);
      }
    }
  } else {
    assert(f_s.size() == 5);
    assert(oc == f_s[0]);
    assert(ic == f_s[1] || (ic % 2 != 0));
    is_dw = false;
  }
  do_relu = p.do_relu().getValue();
  with_bias = p.with_bias().getValue();
}

void parsePoolParam(const tpu::PoolParam &p,
    Value input, Value output,
    int &n, int &c, int &ih, int &iw, int &oh, int &ow,
    int &kh, int &kw, int &sh, int &sw, int &pt, int &pb, int &pl, int &pr,
    int &pad_value, bool &is_global, bool &do_relu, bool &count_include_pad) {
  kh = p.kernel_h().getInt();
  kw = p.kernel_w().getInt();
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
  ih = i_s[2];
  iw = i_s[3];
  oh = o_s[2];
  ow = o_s[3];
  pt = p.padding_t().getInt();
  pb = p.padding_b().getInt();
  pl = p.padding_l().getInt();
  pr = p.padding_r().getInt();
  is_global = false;
  if (kh == ih && kw == iw && oh == 1 && ow == 1) {
    //assert(oh == 1 && ow == 1);
    is_global = true;
  }
  pad_value = p.pad_value().getInt();
  do_relu = p.do_relu().getValue();
  count_include_pad = p.count_include_pad().getValue();
}

void parsePool3dParam(const tpu::Pool3dParam &p,
    Value input, Value output,
    int &n, int &c, int &id, int &ih, int &iw,
    int &od, int &oh, int &ow,
    int &kd, int &kh, int &kw,
    int &sd, int &sh, int &sw,
    int &pd0, int &pd1, int &pt, int &pb, int &pl, int &pr,
    bool &is_global, bool &do_relu, bool &count_include_pad) {
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
  do_relu = p.do_relu().getValue();
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

template <typename OpTy>
void parsePermuteParam(Operation *op, std::vector<int64_t> &shape_4,
                       std::vector<int> &order_4) {
  auto pmOp = llvm::dyn_cast<OpTy>(op);
  auto shape = getTensorShape(pmOp.input());
  std::vector<int> order;
  arrayAttrToVector(pmOp.order(), order);
  int num_dims = order.size();
  if (num_dims > 4) {
    for (int i = 0; i < num_dims - 4; i++) {
      assert(shape[i] == 1 && "> 4 dim, should be 1");
      assert(order[i] == i && "> 4 dim, order can't change");
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

template<typename OpTy>
void parseLeakyReluParam(Operation *op,
    int8_t &pos_rshift, int8_t &pos_m_i8,
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

template void parseLeakyReluParam<tpu::TG_INT8_PT_Conv2DOp>(
    Operation *op, int8_t &pos_rshift, int8_t &pos_m_i8,
    int8_t &neg_rshift, int8_t &neg_m_i8, float &negative_slope);
template void parseLeakyReluParam<tpu::TG_INT8_PC_Conv2DOp>(
    Operation *op, int8_t &pos_rshift, int8_t &pos_m_i8,
    int8_t &neg_rshift, int8_t &neg_m_i8, float &negative_slope);
template void parseLeakyReluParam<tpu::TG_INT8_LeakyReluOp>(
    Operation *op, int8_t &pos_rshift, int8_t &pos_m_i8,
    int8_t &neg_rshift, int8_t &neg_m_i8, float &negative_slope);
template void parseLeakyReluParam<tpu::TG_BF16_LeakyReluOp>(
    Operation *op, int8_t &pos_rshift, int8_t &pos_m_i8,
    int8_t &neg_rshift, int8_t &neg_m_i8, float &negative_slope);

void parseActCompressParam(const tpu::ActCmprParam &param, int &cmpr_n,
    int &cmpr_c, int &cmpr_h, int64_t &step_size, int64_t &total_size) {
  std::vector<int> shapes;
  cmpr_n = param.n_step().getInt();
  cmpr_c = param.c_step().getInt();
  cmpr_h = param.h_step().getInt();
  step_size = param.step_size().getInt();
  total_size = param.total_size().getInt();
}

bool isBf16Tensor(Value val) {
  auto valType = val.getType().dyn_cast<TensorType>();
  auto elementType = valType.getElementType();
  return elementType.isBF16();
}

int64_t getTotalCompressedActivationSize(Operation *op) {
  int64_t cmrSize =
      llvm::TypeSwitch<Operation *, int64_t>(op)
          .Case<tpu::TG_INT8_PT_Conv2DOp, tpu::TG_INT8_PC_Conv2DOp,
                tpu::TG_BF16_Conv2DOp, tpu::TG_INT8_EltwiseAddOp,
                tpu::TG_BF16_EltwiseAddOp, tpu::TG_INT8_PoolMax2DOp>(
              [&](auto tpuOp) {
                if (tpuOp.store_compr_act_param().hasValue())
                  return tpuOp.store_compr_act_param()
                      .getValue()
                      .total_size()
                      .getInt();
                else
                  return (int64_t)0;
              })
          .Case([&](tpu::TL_LG_JoinOp) {
            auto tpuOp = llvm::dyn_cast<tpu::TL_LG_StoreOp>(
                op->getOperand(0).getDefiningOp());

            // tl_lg_store -> tl_lg_join
            if (tpuOp.compr_act_param().hasValue())
              return tpuOp.compr_act_param().getValue().total_size().getInt();
            else
              return (int64_t)0;
          })
          .Default([](Operation *) { return (int64_t)0; });

  return cmrSize;
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
