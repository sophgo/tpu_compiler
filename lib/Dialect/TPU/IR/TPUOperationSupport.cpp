#include <numeric>
#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/TPU/CustomOpParam.h"

namespace mlir {

void convertAttributesToOpParam(const DictionaryAttr &attrs, cvi::OpParam &param) {
  auto getBoolValue = [](const Attribute& attr) {
    return (bool)attr.cast<BoolAttr>().getValue();
  };
  auto getIntValue = [](const Attribute& attr) {
    return (int32_t)attr.cast<IntegerAttr>().getValue().getSExtValue();
  };
  auto getFloatValue = [](const Attribute& attr) {
    return (float)attr.cast<FloatAttr>().getValue().convertToFloat();
  };
  auto getStringValue = [](const Attribute& attr) {
    return attr.cast<StringAttr>().getValue().str();
  };
  auto getArrayKind = [](const ArrayAttr& array) {
    auto& attr = *(array.begin());
    return attr.getKind();
  };

  for (auto& a : attrs) {
    auto name = a.first.str();
    auto &attr = a.second;
    switch (attr.getKind()) {
      case StandardAttributes::Bool:
        param.put<bool>(name, getBoolValue(attr));
        break;
      case StandardAttributes::Integer: {
        auto intAttr = attr.cast<IntegerAttr>();
        if (intAttr.getType().isInteger(16)) {
          param.put<int16_t>(name, (int16_t)getIntValue(attr));
        } else if (intAttr.getType().isInteger(8)) {
          param.put<int8_t>(name, (int8_t)getIntValue(attr));
        } else {
          param.put<int32_t>(name, (int32_t)getIntValue(attr));
        }
        break;
      }
      case StandardAttributes::Float:
        param.put<float>(name, getFloatValue(attr));
        break;
      case StandardAttributes::String:
        param.put<std::string>(name, getStringValue(attr));
        break;
      case StandardAttributes::Array: {
        auto array = attr.cast<ArrayAttr>();
        switch (getArrayKind(array)) {
          case StandardAttributes::Bool: {
            std::vector<bool> vec;
            for (auto& item : array) {
              vec.push_back(getBoolValue(item));
            }
            param.put<std::vector<bool>>(name, vec);
            break;
          }
          case StandardAttributes::Integer: {
            std::vector<int32_t> vec;
            for (auto& item : array) {
              vec.push_back(getIntValue(item));
            }
            param.put<std::vector<int32_t>>(name, vec);
            break;
          }
          case StandardAttributes::Float: {
            std::vector<float> vec;
            for (auto& item : array) {
              vec.push_back(getFloatValue(item));
            }
            param.put<std::vector<float>>(name, vec);
            break;
          }
          case StandardAttributes::String: {
            std::vector<std::string> vec;
            for (auto& item : array) {
              vec.push_back(getStringValue(item));
            }
            param.put<std::vector<std::string>>(name, vec);
            break;
          }
          default:
            llvm_unreachable("unsupported attribute");
        }
        break;
      }
      default:
        llvm_unreachable("unsupported attribute");
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
  return getOpName(op->getOperand(index)->getDefiningOp());
}

int getOpLayerId(Operation *op) {
  if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpCommonInterface>(op)) {
    // get op id according the line number of op's position.
    auto loc = op->getLoc().cast<FileLineColLoc>();
    return loc.getLine() - 5;
  } else {
    std::string errorMsg = std::string(__func__) + " failed, Op " +
                           op->getName().getStringRef().str() + "\n";
    llvm_unreachable(errorMsg.c_str());
  }
}

LogicalResult setChipName(Operation *op, llvm::StringRef chipname) {
  if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpCommonInterface>(op)) {
    return tpuOp.setChipName(chipname);
  } else {
    std::string errorMsg = std::string(__func__) + " failed, Op " +
                           op->getName().getStringRef().str() + "\n";
    llvm_unreachable(errorMsg.c_str());
  }
}

llvm::StringRef getChipName(Operation *op) {
  if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpCommonInterface>(op)) {
    return tpuOp.getChipName();
  } else {
    std::string errorMsg = std::string(__func__) + " failed, Op " +
                           op->getName().getStringRef().str() + "\n";
    llvm_unreachable(errorMsg.c_str());
  }
}
llvm::StringRef getOpQuant(Operation *op) {
  if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(op)) {
    return tpuOp.getOpQuant();
  } else if (isa<tpu::DetectionOutputOp>(op)
             || isa<tpu::FrcnDetectionOp>(op)
             || isa<tpu::RetinaFaceDetectionOp>(op)
             || isa<tpu::PreprocessOp>(op)
             || isa<tpu::PriorBoxOp>(op)
             || isa<tpu::ProposalOp>(op)
             || isa<tpu::ROIPoolingOp>(op)
             || isa<tpu::SoftmaxCpuOp>(op)
             || isa<tpu::TransposeOp>(op)
             || isa<tpu::YoloDetectionOp>(op)
             || isa<tpu::CastOp>(op)
             || isa<tpu::QuantOp>(op)
             ) {
    // cpu Ops return NONE
    return llvm::StringRef("NONE");
  } else {
    std::string errorMsg = std::string(__func__) + " failed, Op " +
                           op->getName().getStringRef().str() + "\n";
    llvm_unreachable(errorMsg.c_str());
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

void setOpResultType(Value *value, StandardTypes::Kind kind, int width) {
  auto builder = Builder(value->getContext());
  Type eltType;
  if (kind == StandardTypes::F32) {
    eltType = FloatType::getF32(builder.getContext());
  } else if (kind == StandardTypes::BF16) {
    eltType = FloatType::getBF16(builder.getContext());
  } else if (kind == StandardTypes::Integer) {
    assert(width != 0);
    eltType = IntegerType::get(width, builder.getContext());
  } else {
    llvm_unreachable("unsupported type");
  }
  auto shape = value->getType().cast<TensorType>().getShape();
  auto type = RankedTensorType::get(shape, eltType);
  value->setType(type);
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

bool isOpQuantPerchannel(Operation *op) {
  if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(op)) {
    return tpuOp.isOpQuantPerchannel();
  } else {
    std::string errorMsg = std::string(__func__) + " failed, Op " +
                           op->getName().getStringRef().str() + "\n";
    llvm_unreachable(errorMsg.c_str());
  }
}

LogicalResult setOpQuantPerchannel(Operation *op, bool flag) {
  if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(op)) {
    return tpuOp.setOpQuantPerchannel(flag);
  } else {
    std::string errorMsg = std::string(__func__) + " failed, Op " +
                           op->getName().getStringRef().str() + "\n";
    llvm_unreachable(errorMsg.c_str());
  }
}

bool isOpQuantAsymmetric(Operation *op) {
  if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(op)) {
    return tpuOp.isOpQuantAsymmetric();
  } else {
    std::string errorMsg = std::string(__func__) + " failed, Op " +
                           op->getName().getStringRef().str() + "\n";
    llvm_unreachable(errorMsg.c_str());
  }
}

LogicalResult setOpQuantAsymmetric(Operation *op, bool flag) {
  if (auto tpuOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(op)) {
    return tpuOp.setOpQuantAsymmetric(flag);
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
  auto formerOp = op->getOperand(index)->getDefiningOp();
  return getOpThreshold(formerOp);
}

uint64_t getOpAddress(Operation *op) {
  if (isa<tpu::TpuTGOpCodegenInterface>(op)) {
    auto tpuTGOp = llvm::dyn_cast<tpu::TpuTGOpCodegenInterface>(op);
    return tpuTGOp.getGAddr();
  } else if (auto castOp = llvm::dyn_cast<tpu::GenericCpuOp>(op)) {
    if (castOp.gaddr().hasValue()) {
      return castOp.gaddr().getValue().getZExtValue();
    }
    llvm_unreachable("unsupported op");
  } else if (isa<tpu::TpuTLOpCodegenInterface>(op)) {
    auto tpuTLOp = llvm::dyn_cast<tpu::TpuTLOpCodegenInterface>(op);
    return tpuTLOp.getGAddr();
  } else if (auto inputOp = llvm::dyn_cast<tpu::InputOp>(op)) {
    if (inputOp.gaddr().hasValue()) {
      return inputOp.gaddr().getValue().getZExtValue();
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
    castOp.setAttr("gaddr", Builder(castOp.getOperation()->getContext()).getI64IntegerAttr(gaddr));
    return success();
  } else if (auto inputOp = llvm::dyn_cast<tpu::InputOp>(op)) {
    inputOp.setAttr("gaddr", Builder(inputOp.getOperation()->getContext()).getI64IntegerAttr(gaddr));
  } else {
    std::string errorMsg = std::string(__func__) + " failed, Op " +
                           op->getName().getStringRef().str() + "\n";
    llvm_unreachable(errorMsg.c_str());
  }
  return failure();
}

uint64_t getWeightOpAddress(Operation *op) {
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(op)) {
    return cast_op.offset().getValue().getLimitedValue();
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
  auto formerOp = op->getOperand(index)->getDefiningOp();
  if (isa<tpu::LoadWeightOp>(formerOp)) {
    return getWeightOpAddress(formerOp);
  } else {
    return getOpAddress(formerOp);
  }
}

Operation* getNextOp(Operation *op) {
  Operation *nextOp = nullptr;
  if (op->getResult(0)->hasOneUse()) {
    for (auto &use : op->getResult(0)->getUses()) {
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
      builder.getBoolAttr(false),
      builder.getBoolAttr(false),
      builder.getF32FloatAttr(0.0),
      builder.getF32FloatAttr(0.0),
      builder.getContext());
}

void parseConvParam(const tpu::ConvParam &p, bool is_deconv,
    Value *input, Value *output, Value *filter,
    int &n, int &ic, int &ih, int &iw, int &oc, int &oh, int &ow, int &g,
    int &kh, int &kw, int &sh, int &sw, int &pt, int &pb, int &pl, int &pr, int &dh, int &dw,
    bool &is_dw, bool &with_bias, bool &do_relu) {
  dh = p.dilation_h().getValue().getLimitedValue();
  dw = p.dilation_w().getValue().getLimitedValue();
  sh = p.stride_h().getValue().getLimitedValue();
  sw = p.stride_w().getValue().getLimitedValue();
  pt = p.padding_t().getValue().getLimitedValue();
  pb = p.padding_b().getValue().getLimitedValue();
  pl = p.padding_l().getValue().getLimitedValue();
  pr = p.padding_r().getValue().getLimitedValue();
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


  g = p.group().getValue().getLimitedValue();
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
}

void parsePoolParam(const tpu::PoolParam &p,
    Value *input, Value *output,
    int &n, int &c, int &ih, int &iw, int &oh, int &ow,
    int &kh, int &kw, int &sh, int &sw, int &pt, int &pb, int &pl, int &pr,
    bool &is_global, bool &do_relu, bool &count_include_pad) {
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
  if (kh == ih && kw == iw && oh == 1 && ow == 1) {
    //assert(oh == 1 && ow == 1);
    is_global = true;
  }
  do_relu = p.do_relu().getValue();
  count_include_pad = p.count_include_pad().getValue();
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

template<typename OpTy>
void parseLeakyReluParam(Operation *op,
    int8_t &pos_rshift, int8_t &pos_m_i8,
    int8_t &neg_rshift, int8_t &neg_m_i8,
    float &negative_slope) {
  auto lreluOp = llvm::dyn_cast<OpTy>(op);
  assert(lreluOp);

  if (lreluOp.m_i8_pos().hasValue()) {
    pos_m_i8 = lreluOp.m_i8_pos().getValue().getLimitedValue();
    pos_rshift = lreluOp.rshift_pos().getValue().getLimitedValue();
    assert(pos_m_i8);
  } else {
    pos_m_i8 = 0;
    pos_rshift = 0;
  }

  if (lreluOp.m_i8_neg().hasValue()) {
    neg_m_i8 = lreluOp.m_i8_neg().getValue().getLimitedValue();
    neg_rshift = lreluOp.rshift_neg().getValue().getLimitedValue();
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


void parseGruParam(
    Value *input, Value *weight,
    int &seq_len, int &batch_size, int &input_size, int& hidden_size) {
  auto input_type = input->getType().cast<TensorType>();
  std::vector<int64_t> i_s(input_type.getShape());

  auto weight_type = weight->getType().cast<TensorType>();
  std::vector<int64_t> w_s(weight_type.getShape());

  seq_len = i_s[0];
  batch_size = 1;
  input_size = w_s[2];
  hidden_size = w_s[1] / 3;
}

void parseLstmParam(
    Value *input, Value *weight,
    int &seq_len, int &batch_size, int &input_size, int& hidden_size) {
  auto input_type = input->getType().cast<TensorType>();
  std::vector<int64_t> i_s(input_type.getShape());

  auto weight_type = weight->getType().cast<TensorType>();
  std::vector<int64_t> w_s(weight_type.getShape());

  seq_len = i_s[0];
  batch_size = 1;
  input_size = w_s[2];
  hidden_size = w_s[1] / 4;
}

} // namespace
