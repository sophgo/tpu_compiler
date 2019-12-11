#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

LogicalResult getPreviousOpThreshold(Operation *op, float *threshold, int index = 0) {
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
  return failure();
}

} // namespace