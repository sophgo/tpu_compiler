#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

LogicalResult getPreviousOpThreshold(Operation *op, float *threshold, uint index = 0) {
  if ( op->getNumOperands() < (index + 1) ) {
    assert(false);
    return failure();
  }
  auto formerOp = op->getOperand(index)->getDefiningOp();
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::InputOp>(formerOp)) {
    *threshold = cast_op.threshold_y().getValue().convertToFloat();
    return success();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::Conv2DOp>(formerOp)) {
    *threshold = cast_op.threshold_y().getValue().convertToFloat();
    return success();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::FullyConnectedOp>(formerOp)) {
    *threshold = cast_op.threshold_y().getValue().convertToFloat();
    return success();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::Pool2DOp>(formerOp)) {
    *threshold = cast_op.threshold_y().getValue().convertToFloat();
    return success();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::BatchNormOp>(formerOp)) {
    *threshold = cast_op.threshold_y().getValue().convertToFloat();
    return success();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::ScaleOp>(formerOp)) {
    *threshold = cast_op.threshold_y().getValue().convertToFloat();
    return success();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::ReluOp>(formerOp)) {
    *threshold = cast_op.threshold_y().getValue().convertToFloat();
    return success();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::EltwiseOp>(formerOp)) {
    *threshold = cast_op.threshold_y().getValue().convertToFloat();
    return success();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::ReshapeOp>(formerOp)) {
    *threshold = cast_op.threshold_y().getValue().convertToFloat();
    return success();
  }
  if (auto cast_op = llvm::dyn_cast_or_null<tpu::SoftmaxOp>(formerOp)) {
    *threshold = cast_op.threshold_y().getValue().convertToFloat();
    return success();
  }
  assert(0);
  return failure();
}

// TODO: move to some other place
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