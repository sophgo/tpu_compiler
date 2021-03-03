#include <numeric>
#include <ctime>
#include <iomanip>
#include <stdlib.h>
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

/***********************************************************
 * Tensor helpers
 ***********************************************************/

bool isTensorNone(Value value) {
  return (!value.getType().dyn_cast_or_null<RankedTensorType>());
}

int64_t getTensorSize(Value value) {
  std::vector<int64_t> shape = value.getType().cast<TensorType>().getShape();
  return std::accumulate(std::begin(shape), std::end(shape), 1,
                         std::multiplies<>());
}

std::vector<int64_t> getTensorShape(Value value) {
  return value.getType().cast<TensorType>().getShape();
}

void getTensorShapeAndSize(Value value, std::vector<int64_t> &shape,
    int64_t &size) {
  shape = getTensorShape(value);
  size = getTensorSize(value);
}

void getNCHW(std::vector<int64_t> &shape,
    int64_t &n, int64_t &c, int64_t &h, int64_t &w) {
  if(shape.size() == 5 && shape[0] == 1) {
    n = shape[1];
    c = shape[2];
    h = shape[3];
    w = shape[4];
  } else if(shape.size() == 4) {
    n = shape[0];
    c = shape[1];
    h = shape[2];
    w = shape[3];
  } else if (shape.size() == 3) {
    n = shape[0];
    c = shape[1];
    h = shape[2];
    w = 1;
  } else if (shape.size() == 2) {
    n = shape[0];
    c = shape[1];
    h = 1;
    w = 1;
  } else {
    llvm_unreachable("unsupported shape size");
  }
}

void getNCDHW(std::vector<int64_t> &shape,
    int64_t &n, int64_t &c, int64_t &d, int64_t &h, int64_t &w) {
  if(shape.size() == 5) {
    n = shape[0];
    c = shape[1];
    d = shape[2];
    h = shape[3];
    w = shape[4];
  } else {
    llvm_unreachable("unsupported shape size");
  }
}

std::vector<std::vector<int64_t>> getOperandShapes(Operation *op) {
  std::vector<std::vector<int64_t>> shapes;
  for (auto operand : op->getOperands()) {
    if (isTensorNone(operand) ) {
      shapes.push_back({0});
      continue;
    }
    shapes.push_back(getTensorShape(operand));
  }
  return shapes;
}

/***********************************************************
 * Weight helpers
 ***********************************************************/

Value getWeightFileValue(Operation *op) {
  if (auto fn = cast<FuncOp>(op->getParentOp())) {
    Value wfV = nullptr;
    fn.walk([&](tpu::WeightFileOp op) {
       wfV = op.getResult();
    });
    if (!wfV) {
      auto builder = OpBuilder(op);
      auto elementType = mlir::FloatType::getF32(builder.getContext());
      auto weightFileName =
          TensorFile::generateName("unknown", (int)random());
      auto weight_type = MemRefType::get({0x80000000}, elementType);
      auto weight_attr = builder.getStringAttr(weightFileName);
      auto weightOp = builder.create<tpu::WeightFileOp>(builder.getUnknownLoc(),
                                                  weight_type, weight_attr);
      weightOp.getOperation()->moveBefore(op);
      wfV = weightOp.getResult();
    }
    // assert(wfV && "wfV is nullptr");
    return wfV;
  } else {
    llvm_unreachable("no FuncOp found");
  }
}

TensorFile* getWeightTensorFile(Operation *op) {
  auto wfV = getWeightFileValue(op);
  auto wfOp = cast<tpu::WeightFileOp>(wfV.getDefiningOp());
  assert(wfOp && "wfOp is nullptr");
  TensorFile *wTF = wfOp.get();
  assert(wTF && "no tensor file found");
  return wTF;
}

template<typename T>
std::unique_ptr<std::vector<T> > readWeightTensor(
    Value opd, TensorFile *wTF) {
  auto weightOp = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
      opd.getDefiningOp());
  auto name = weightOp.name();
  auto type = weightOp.getResult().getType().cast<TensorType>();
  auto tensor = wTF->readTensor<T>(name, type);
  return std::move(tensor);
}
template std::unique_ptr<std::vector<float> > readWeightTensor(
    Value opd, TensorFile *wTF);

template<typename T>
std::unique_ptr<std::vector<T> > readAndDeleteWeightTensor(
    Value opd, TensorFile *wTF) {
  if (auto weightOp = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
                  opd.getDefiningOp())) {
    auto name = weightOp.name();
    auto type = weightOp.getResult().getType().cast<TensorType>();
    auto tensor = wTF->readTensor<T>(name, type);
    wTF->deleteTensor<T>(name);
    return std::move(tensor);
  } else if (auto weightOp = llvm::dyn_cast_or_null<tpu::TL_LG_LoadCoeffOp>(
                  opd.getDefiningOp())) {
    auto name = weightOp.name();
    auto type = weightOp.getResult().getType().cast<TensorType>();
    auto tensor = wTF->readTensor<T>(name, type);
    wTF->deleteTensor<T>(name);
    return std::move(tensor);
  } else {
    assert(0);
  }
}
template std::unique_ptr<std::vector<float> > readAndDeleteWeightTensor(
    Value opd, TensorFile *wTF);
template std::unique_ptr<std::vector<uint16_t> > readAndDeleteWeightTensor(
    Value opd, TensorFile *wTF);
template std::unique_ptr<std::vector<int8_t> > readAndDeleteWeightTensor(
    Value opd, TensorFile *wTF);

template<typename T>
void addWeightTensorAndUpdateWeightOp(Value opd,
    StringRef suffix, std::vector<T> &weight, std::vector<int64_t> &shape,
    StringRef storageType, TensorFile *wTF) {
  auto builder = Builder(opd.getContext());
  Type eltType;
  if ( typeid(T) == typeid(float) ) {
    eltType = FloatType::getF32(builder.getContext());
  } else if ( typeid(T) == typeid(uint32_t) ) {
    eltType = IntegerType::get(builder.getContext(), 32);
  } else if ( typeid(T) == typeid(uint16_t) ) {
    if (storageType == "BF16") {
      eltType = FloatType::getBF16(builder.getContext());
    } else if (storageType == "UINT16") {
      eltType = IntegerType::get(builder.getContext(), 16);
    } else {
      llvm_unreachable("unsupported storage type");
    }
  } else if ( typeid(T) == typeid(int16_t) ) {
    eltType = IntegerType::get(builder.getContext(), 16);
  } else if ( typeid(T) == typeid(uint8_t) ) {
    eltType = IntegerType::get(builder.getContext(), 8);
  } else if ( typeid(T) == typeid(int8_t) ) {
    eltType = IntegerType::get(builder.getContext(), 8);
  } else {
    llvm_unreachable("unsupported type");
  }

  if (auto weightOp = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
      opd.getDefiningOp())) {
    auto type = RankedTensorType::get(shape, eltType);
    auto name = weightOp.name().str();
    if (!suffix.empty()) {
      name = name + "_" + suffix.str();
    }
    wTF->addTensor<T>(name, &weight, type);
    weightOp->setAttr("name", builder.getStringAttr(name));
    weightOp->setAttr("storage", builder.getStringAttr(storageType));
    weightOp.getResult().setType(type);
  } else if (auto weightOp = llvm::dyn_cast_or_null<tpu::TL_LG_LoadCoeffOp>(
      opd.getDefiningOp())) {
    auto type = RankedTensorType::get(shape, eltType);
    auto name = weightOp.name().str();
    if (!suffix.empty()) {
      name = name + "_" + suffix.str();
    }
    wTF->addTensor<T>(name, &weight, type);
    weightOp->setAttr("name", builder.getStringAttr(name));
    weightOp->setAttr("storage", builder.getStringAttr(storageType));
    weightOp.getResult().setType(type);
  }
}

template void addWeightTensorAndUpdateWeightOp(Value opd,
    StringRef suffix, std::vector<float> &weight,
    std::vector<int64_t> &shape, StringRef storageType, TensorFile *wTF);
template void addWeightTensorAndUpdateWeightOp(Value opd,
    StringRef suffix, std::vector<int8_t> &weight,
    std::vector<int64_t> &shape, StringRef storageType, TensorFile *wTF);
template void addWeightTensorAndUpdateWeightOp(Value opd,
    StringRef suffix, std::vector<uint8_t> &weight,
    std::vector<int64_t> &shape, StringRef storageType, TensorFile *wTF);
template void addWeightTensorAndUpdateWeightOp(Value opd,
    StringRef suffix, std::vector<int16_t> &weight,
    std::vector<int64_t> &shape, StringRef storageType, TensorFile *wTF);
template void addWeightTensorAndUpdateWeightOp(Value opd,
    StringRef suffix, std::vector<uint16_t> &weight,
    std::vector<int64_t> &shape, StringRef storageType, TensorFile *wTF);
template void addWeightTensorAndUpdateWeightOp(Value opd,
    StringRef suffix, std::vector<uint32_t> &weight,
    std::vector<int64_t> &shape, StringRef storageType, TensorFile *wTF);

template<typename T>
Value addWeightTensorAndCreateWeightOp(Operation *op,
    StringRef suffix, std::vector<T> &weight,
    std::vector<int64_t> &shape, StringRef storageType,
    TensorFile *wTF, Value wFV) {
  auto name = getOpName(op).str() + "_" + suffix.str();
  auto builder = Builder(op->getContext());
  Type eltType;
  if ( typeid(T) == typeid(float) ) {
    eltType = FloatType::getF32(builder.getContext());
  } else if ( typeid(T) == typeid(uint8_t) ) {
    eltType = IntegerType::get(builder.getContext(), 8);
  } else {
    std::string errorMsg = "add weight tensor failed, name = " + name +
                           ", type =" + typeid(T).name();
    llvm_unreachable(errorMsg.c_str());
  }
  auto type = RankedTensorType::get(shape, eltType);
  wTF->addTensor<T>(name, &weight, type);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", builder.getStringAttr(name)));
  attrs.push_back(builder.getNamedAttr("storage",
      builder.getStringAttr(storageType)));
  return OpBuilder(op).create<tpu::LoadWeightOp>(op->getLoc(), type,
      ArrayRef<Value>{wFV}, ArrayRef<NamedAttribute>{attrs});
}
template Value addWeightTensorAndCreateWeightOp(Operation *op,
    StringRef suffix, std::vector<float> &weight,
    std::vector<int64_t> &shape, StringRef storageType,
    TensorFile *wTF, Value wFV);
template Value addWeightTensorAndCreateWeightOp(Operation *op,
    StringRef suffix, std::vector<uint8_t> &weight,
    std::vector<int64_t> &shape, StringRef storageType,
    TensorFile *wTF, Value wFV);

} // namespace
