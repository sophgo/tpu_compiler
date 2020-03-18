#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TensorFile.h"
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

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const final {

    // Cannot erase AllocOp in ReturnOpConverter.
    // AllocOp references is still kept.
    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

struct TensorLoadOpConverter : public ConversionPattern {
  TensorLoadOpConverter(MLIRContext *ctx)
      : ConversionPattern(TensorLoadOp::getOperationName(), 1, ctx) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const final {

    // Remove MemRefType to TensorType conversion
    op->getResult(0)->replaceAllUsesWith(op->getOperand(0));

    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

struct ReturnOpConverter : public ConversionPattern {
  ReturnOpConverter(MLIRContext *ctx)
      : ConversionPattern(ReturnOp::getOperationName(), 1, ctx) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto returnOp = dyn_cast<ReturnOp>(op);

    // AllocOp and TensorStoreOp were created during lowering.
    // Remove them when back to TensorType.
    for (auto &user : op->getOperand(0)->getUses()) {
      Operation *userOp = user.getOwner();
      if (auto tensorStoreOp = dyn_cast<TensorStoreOp>(userOp)) {
        auto lastTpuOp = userOp->getOperand(0)->getDefiningOp();
        rewriter.replaceOpWithNewOp<ReturnOp>(returnOp,
                                              lastTpuOp->getResult(0));

        rewriter.eraseOp(tensorStoreOp);
      }
    }

    return matchSuccess();
  }
};

struct ConvertFuncToTensorPass
    : public ModulePass<ConvertFuncToTensorPass> {
  void runOnModule() override {
    MemRefToTensorTypeConverter converter;
    OwningRewritePatternList patterns;
    ConversionTarget target(getContext());
    target.addLegalOp<tpu::LoadWeightOp>();
    target.addLegalOp<tpu::QuantOp>();
    target.addLegalOp<tpu::ReshapeOp>();
    target.addLegalOp<tpu::SoftmaxOp>();
    target.addLegalOp<tpu::TG_INT8_InputOp>();
    target.addLegalOp<tpu::TG_INT8_EltwiseAddOp>();
    target.addLegalOp<tpu::TG_INT8_FullyConnectedOp>();
    target.addLegalOp<tpu::TG_INT8_PC_Conv2DOp>();
    target.addLegalOp<tpu::TG_INT8_PoolAvg2DOp>();
    target.addLegalOp<tpu::TG_INT8_PoolMax2DOp>();

    target.addDynamicallyLegalOp<FuncOp>(
        [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });

    target.addDynamicallyLegalOp<ReturnOp>(
        [&] (Operation *op) -> bool {
          return op->getOperand(0)->getType().isa<TensorType>();
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

static PassRegistration<ConvertFuncToTensorPass>
  pass("convert-func-to-tensor",
       "Convert func from MemRefType to TensorType");
