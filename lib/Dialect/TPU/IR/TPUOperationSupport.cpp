#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

llvm::StringRef getOpName(Operation *op) {
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::InputOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::Conv2DOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::FullyConnectedOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::Pool2DOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::BatchNormOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::ScaleOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::ReluOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::EltwiseOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::SoftmaxOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::ReshapeOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::QuantizationOp>(op)) {
    return cast_op.name().getValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::DequantizationOp>(op)) {
    return cast_op.name().getValue();
  }
  llvm::errs() << op->getName() << "\n";
  assert(false);
  return "not_found";
}

float getPreviousOpThreshold(Operation *op, uint index = 0) {
  if ( op->getNumOperands() < (index + 1) ) {
    assert(false);
    return NAN;
  }
  auto formerOp = op->getOperand(index)->getDefiningOp();
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::InputOp>(formerOp)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::Conv2DOp>(formerOp)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::FullyConnectedOp>(formerOp)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::Pool2DOp>(formerOp)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::BatchNormOp>(formerOp)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::ScaleOp>(formerOp)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::ReluOp>(formerOp)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::EltwiseOp>(formerOp)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::ReshapeOp>(formerOp)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::SoftmaxOp>(formerOp)) {
    return cast_op.threshold_y().getValue().convertToFloat();
  }
  assert(false);
  return NAN;
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
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::Conv2DOp>(formerOp)) {
    return cast_op.offset().getValue().getLimitedValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::FullyConnectedOp>(formerOp)) {
    return cast_op.offset().getValue().getLimitedValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::Pool2DOp>(formerOp)) {
    return cast_op.offset().getValue().getLimitedValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::ReluOp>(formerOp)) {
    return cast_op.offset().getValue().getLimitedValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::EltwiseOp>(formerOp)) {
    return cast_op.offset().getValue().getLimitedValue();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::ReshapeOp>(formerOp)) {
    // for reshape, we need to go to this one's previous
    // this is recursive ...
    return getPreviousOpAddress(cast_op);
  }
  assert(0);
  return 0xFFFFFFFFFFFFFFFF;
}

uint64_t getWeightOpAddress(Operation *op) {
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(op)) {
    return cast_op.offset().getValue().getLimitedValue();
  }
  assert(0);
  return 0xFFFFFFFFFFFFFFFF;
}

} // namespace