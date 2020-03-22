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

template <typename MemRefTyOp, typename TensorTyOp>
class convertTgOpToTensorPattern : public ConversionPattern {
public:
  explicit convertTgOpToTensorPattern(MLIRContext *context)
     : ConversionPattern(MemRefTyOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const final {
    SmallVector<Value *, 4> buffer_args;

    // Insert tensor_load if op input is function argument
    for (unsigned i = 0; i < operands.size() - 1; ++i) {
      for (auto funArg : op->getBlock()->getArguments()) {
        if (operands[i] == funArg) {
          auto tensorLoadOp =
              rewriter.create<TensorLoadOp>(op->getLoc(), funArg);

          // Convert function argument at the beginning of the block.
          auto *parentBlock = op->getBlock();
          tensorLoadOp.getOperation()->moveBefore(&parentBlock->front());

          buffer_args.push_back(tensorLoadOp.getResult());
        } else {
          buffer_args.push_back(operands[i]);
        }
      }
    }

    // Last operand is result
    auto lastOperand = operands[op->getNumOperands() - 1];
    auto memrefResult = lastOperand->getType().cast<MemRefType>();
    auto tensorTypeResult =
        RankedTensorType::get(memrefResult.getShape(),
                              memrefResult.getElementType());

    // Remove the lowered AllocOp/DeallocOp
    if (auto allocOp =
            dyn_cast_or_null<AllocOp>(lastOperand->getDefiningOp())) {
      for (auto &user : allocOp.getResult()->getUses()) {
        Operation *owner = user.getOwner();
        if (auto deallocOp = dyn_cast_or_null<DeallocOp>(owner)) {
          rewriter.eraseOp(deallocOp);
        }
      }
      rewriter.eraseOp(allocOp);
    }

    auto newTensorOp =
        rewriter.create<TensorTyOp>(op->getLoc(), tensorTypeResult, buffer_args,
                                    op->getAttrs());
    lastOperand->replaceAllUsesWith(newTensorOp.getResult());
    rewriter.eraseOp(op);

    // The result of all replaced tpu op are TensorType, but the returned of
    // function is still MemRefType, so create new AllocOp to store the last tpu
    // op.
    // And replace the operand of std.return with the result of AllocOp.
    for (auto &user : newTensorOp.getResult()->getUses()) {
      if (auto returnOp = dyn_cast_or_null<ReturnOp>(user.getOwner())) {
        auto resultType =
            newTensorOp.getResult()->getType().template cast<TensorType>();
        auto allocOp =
            rewriter.create<AllocOp>(op->getLoc(),
                MemRefType::get(resultType.getShape(),
                                resultType.getElementType(), {},
                                2/*TPU_MEM_REGION_ACTIVATION*/));

        rewriter.create<TensorStoreOp>(op->getLoc(), newTensorOp.getResult(),
                                       allocOp);

        rewriter.replaceOpWithNewOp<ReturnOp>(returnOp, allocOp.getResult());
      }
    }

    return matchSuccess();
  }
};

struct ConvertTgOpToTensorPass : public FunctionPass<ConvertTgOpToTensorPass> {
  void runOnFunction() override {
    auto fn = getFunction();
    auto *context = &getContext();
    ConversionTarget target(getContext());
    OwningRewritePatternList patterns;

    target.addLegalOp<tpu::LoadWeightOp>();
    target.addLegalOp<tpu::QuantOp>();
    target.addLegalOp<tpu::ReshapeOp>();
    target.addLegalOp<tpu::TG_INT8_InputOp>();
    target.addLegalOp<tpu::TG_INT8_EltwiseAddOp>();
    target.addLegalOp<tpu::TG_INT8_FullyConnectedOp>();
    target.addLegalOp<tpu::TG_INT8_PC_Conv2DOp>();
    target.addLegalOp<tpu::TG_INT8_PoolAvg2DOp>();
    target.addLegalOp<tpu::TG_INT8_PoolMax2DOp>();
    target.addLegalOp<AllocOp>();
    target.addLegalOp<ReturnOp>();
    target.addLegalOp<TensorLoadOp>();
    target.addLegalOp<TensorStoreOp>();

    patterns.insert<
        convertTgOpToTensorPattern<tpu::TG_MemRef_INT8_InputOp, tpu::TG_INT8_InputOp>,
        convertTgOpToTensorPattern<tpu::TG_MemRef_INT8_EltwiseAddOp, tpu::TG_INT8_EltwiseAddOp>,
        convertTgOpToTensorPattern<tpu::TG_MemRef_INT8_FullyConnectedOp, tpu::TG_INT8_FullyConnectedOp>,
        convertTgOpToTensorPattern<tpu::TG_MemRef_INT8_PC_Conv2DOp, tpu::TG_INT8_PC_Conv2DOp>,
        convertTgOpToTensorPattern<tpu::TG_MemRef_INT8_PoolAvg2DOp, tpu::TG_INT8_PoolAvg2DOp>,
        convertTgOpToTensorPattern<tpu::TG_MemRef_INT8_PoolMax2DOp, tpu::TG_INT8_PoolMax2DOp>,
        convertTgOpToTensorPattern<tpu::TG_MemRef_LoadWeightOp, tpu::LoadWeightOp>,
        convertTgOpToTensorPattern<tpu::TG_MemRef_ReshapeOp, tpu::ReshapeOp>
        >(context);
    if (failed(applyPartialConversion(fn, target, patterns)))
      signalPassFailure();
  }
};

}  // anonymous space

std::unique_ptr<OpPassBase<FuncOp>> mlir::createConvertTgOpToTensorPass() {
  return std::make_unique<ConvertTgOpToTensorPass>();
}

static PassRegistration<ConvertTgOpToTensorPass>
  pass("convert-tg-op-to-tensor",
       "Convert tg op from MemRefType to TensorType");
