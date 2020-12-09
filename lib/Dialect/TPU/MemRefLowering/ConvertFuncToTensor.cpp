#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tpuc/Support/TensorFile.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/ADT/StringRef.h"

#define DEBUG_TYPE "ConvertTgMemRefOpToTensor"

using namespace mlir;
using namespace tpu;

namespace {
struct MemRefToTensorTypeConverter : public TypeConverter {
  using TypeConverter::TypeConverter;

  LogicalResult convertType(Type t, SmallVectorImpl<Type> &results) override {
    if (auto memRefType = t.dyn_cast<MemRefType>()) {
      results.push_back(RankedTensorType::get(memRefType.getShape(),
                                              memRefType.getElementType()));
      return success();
    }

    results.push_back(t);
    return success();
  }
};

struct AllocOpConverter : public ConversionPattern {
  AllocOpConverter(MLIRContext *ctx)
      : ConversionPattern(AllocOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    // Cannot erase AllocOp in ReturnOpConverter.
    // AllocOp references is still kept.
    rewriter.eraseOp(op);
    return success();
  }
};

struct ReturnOpConverter : public ConversionPattern {
  ReturnOpConverter(MLIRContext *ctx)
      : ConversionPattern(ReturnOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto returnOp = dyn_cast<ReturnOp>(op);

    // AllocOp and TensorStoreOp were created during lowering.
    // Remove them when back to TensorType.
    SmallVector<Value, 4> newOperands;
    SmallVector<Operation *, 4> erasedOps;
    for (auto operand : op->getOperands()) {
      bool replaced = false;
      for (auto &user : operand.getUses()) {
        Operation *userOp = user.getOwner();
        if (auto tensorStoreOp = dyn_cast<TensorStoreOp>(userOp)) {
          auto lastTpuOp = userOp->getOperand(0).getDefiningOp();
          newOperands.push_back(lastTpuOp->getResult(0));
          erasedOps.push_back(userOp);
          replaced = true;
          break;
        }
      }
      if (!replaced)
        newOperands.push_back(operand);
    }

    rewriter.replaceOpWithNewOp<ReturnOp>(returnOp, newOperands);

    for (auto op : erasedOps)
      rewriter.eraseOp(op);

    return success();
  }
};

struct TensorLoadOpConverter : public ConversionPattern {
  TensorLoadOpConverter(MLIRContext *ctx)
      : ConversionPattern(TensorLoadOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    // Remove MemRefType to TensorType conversion
    op->getResult(0).replaceAllUsesWith(op->getOperand(0));

    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertFuncToTensorPass
    : public mlir::PassWrapper<ConvertFuncToTensorPass, FunctionPass> {
  void runOnModule() override {
    MemRefToTensorTypeConverter converter;
    OwningRewritePatternList patterns;
    ConversionTarget target(getContext());

    target.addLegalOp<tpu::LoadWeightOp>();
    target.addLegalOp<tpu::ReshapeOp>();
    target.addLegalOp<tpu::GenericCpuOp>();

    target.addLegalOp<tpu::TG_INT8_BroadcastMulOp>();
    target.addLegalOp<tpu::TG_BF16_BroadcastMulOp>();
    target.addLegalOp<tpu::TG_INT8_ConcatOp>();
    target.addLegalOp<tpu::TG_BF16_ConcatOp>();
    target.addLegalOp<tpu::TG_INT8_PT_Conv2DOp>();
    target.addLegalOp<tpu::TG_INT8_PC_Conv2DOp>();
    target.addLegalOp<tpu::TG_BF16_Conv2DOp>();
    target.addLegalOp<tpu::TG_INT8_CropOp>();
    target.addLegalOp<tpu::TG_BF16_CropOp>();
    target.addLegalOp<tpu::TG_INT8_PT_DeConv2DOp>();
    target.addLegalOp<tpu::TG_INT8_PC_DeConv2DOp>();
    target.addLegalOp<tpu::TG_BF16_DeConv2DOp>();
    target.addLegalOp<tpu::TG_INT8_EltwiseAddOp>();
    target.addLegalOp<tpu::TG_INT8_EltwiseMaxOp>();
    target.addLegalOp<tpu::TG_INT8_EltwiseMinOp>();
    target.addLegalOp<tpu::TG_INT8_EltwiseMulOp>();
    target.addLegalOp<tpu::TG_BF16_EltwiseAddOp>();
    target.addLegalOp<tpu::TG_BF16_EltwiseMaxOp>();
    target.addLegalOp<tpu::TG_BF16_EltwiseMinOp>();
    target.addLegalOp<tpu::TG_BF16_EltwiseMulOp>();
    target.addLegalOp<tpu::TG_INT8_FullyConnectedOp>();
    target.addLegalOp<tpu::TG_BF16_FullyConnectedOp>();
    target.addLegalOp<tpu::TG_INT8_GenericTpuOp>();
    target.addLegalOp<tpu::TG_BF16_GenericTpuOp>();
    target.addLegalOp<tpu::TG_INT8_LeakyReluOp>();
    target.addLegalOp<tpu::TG_BF16_LeakyReluOp>();
    target.addLegalOp<tpu::TG_INT8_LrnOp>();
    target.addLegalOp<tpu::TG_BF16_LrnOp>();
    target.addLegalOp<tpu::TG_INT8_LutOp>();
    target.addLegalOp<tpu::TG_BF16_LutOp>();
    target.addLegalOp<tpu::TG_INT8_PermuteOp>();
    target.addLegalOp<tpu::TG_BF16_PermuteOp>();
    target.addLegalOp<tpu::TG_INT8_PoolAvg2DOp>();
    target.addLegalOp<tpu::TG_INT8_PoolMax2DOp>();
    target.addLegalOp<tpu::TG_BF16_PoolAvg2DOp>();
    target.addLegalOp<tpu::TG_BF16_PoolMax2DOp>();
    target.addLegalOp<tpu::TG_INT8_ReverseOp>();
    target.addLegalOp<tpu::TG_BF16_ReverseOp>();
    target.addLegalOp<tpu::TG_INT8_ShuffleChannelOp>();
    target.addLegalOp<tpu::TG_BF16_ShuffleChannelOp>();
    target.addLegalOp<tpu::TG_INT8_PixelShuffleOp>();
    target.addLegalOp<tpu::TG_BF16_PixelShuffleOp>();
    target.addLegalOp<tpu::TG_INT8_PReluOp>();
    target.addLegalOp<tpu::TG_BF16_PReluOp>();
    target.addLegalOp<tpu::TG_INT8_ReluOp>();
    target.addLegalOp<tpu::TG_BF16_ReluOp>();
    target.addLegalOp<tpu::TG_INT8_ReorgOp>();
    target.addLegalOp<tpu::TG_BF16_ReorgOp>();
    target.addLegalOp<tpu::TG_INT8_SliceOp>();
    target.addLegalOp<tpu::TG_BF16_SliceOp>();
    target.addLegalOp<tpu::TG_INT8_SwapChannelOp>();
    target.addLegalOp<tpu::TG_BF16_SwapChannelOp>();
    target.addLegalOp<tpu::TG_INT8_UpsampleOp>();
    target.addLegalOp<tpu::TG_BF16_UpsampleOp>();
    target.addLegalOp<tpu::TG_INT8_GruOp>();
    target.addLegalOp<tpu::TG_BF16_GruOp>();
    target.addLegalOp<tpu::TG_INT8_LstmOp>();
    target.addLegalOp<tpu::TG_BF16_LstmOp>();
    target.addLegalOp<tpu::TG_INT8_SoftmaxOp>();
    target.addLegalOp<tpu::TG_BF16_SoftmaxOp>();
    target.addLegalOp<tpu::TG_INT8_ZeroMaskOp>();
    target.addLegalOp<tpu::TG_BF16_ZeroMaskOp>();

    target.addDynamicallyLegalOp<FuncOp>(
        [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });

    target.addDynamicallyLegalOp<ReturnOp>(
        [&] (Operation *op) -> bool {
          return op->getOperand(0).getType().isa<TensorType>();
        });

    auto module = getModule();
    populateFuncOpTypeConversionPattern(patterns, &getContext(), converter);

    patterns.insert<
        AllocOpConverter,
        ReturnOpConverter,
        TensorLoadOpConverter
    >(&getContext());

    if (failed(applyPartialConversion(module, target, patterns, &converter))) {
      signalPassFailure();
    }
  }
};

}  // anonymous space

std::unique_ptr<OpPassBase<ModuleOp>> createConvertFuncToTensorPass() {
  return std::make_unique<ConvertFuncToTensorPass>();
}
