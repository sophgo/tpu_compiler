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

#define DEBUG_TYPE "ConvertTgOpToMemRef"

using namespace mlir;
using namespace tpu;

namespace {

static bool isConvertedOpNeeded(Operation *op) {
  if (dyn_cast<TpuTGOpCodegenInterface>(op))
    llvm::dbgs() << "  isConvertedOpNeeded op " << op->getName() << ", is TgCodeGen\n";
  else
    llvm::dbgs() << "  isConvertedOpNeeded op " << op->getName() << ", is not TgCodeGen\n";

  for (auto &user : op->getResult(0)->getUses()) {
    Operation *userOp = user.getOwner();

    if (dyn_cast<TpuTGOpCodegenInterface>(userOp))
      llvm::dbgs() << "    userOp " << userOp->getName() << ", is TgCodeGend\n";
    else
      llvm::dbgs() << "    userOp " << userOp->getName() << ", is not TgCodeGend\n";

    if (dyn_cast<TpuTGOpCodegenInterface>(op)) {
      // op is Tg CodeGenOp

      // Handle Quant, treat quant as not-lowed op
      if (dyn_cast<QuantOp>(op)) {
        if (dyn_cast<TpuTGOpCodegenInterface>(userOp)) {
          // Quant connected to lowed Op.
          // E.g. Quant -> conv
          llvm::dbgs() << "    converted op is needed\n";
          return true;
        } else {
          // Quant connected to not-lowed op.
          llvm::dbgs() << "    converted op is not needed\n";
          return false;
        }
      }
      if (dyn_cast<QuantOp>(userOp)) {
        // Lower op connected to Quant
        llvm::dbgs() << "    converted op is needed\n";
        return true;
      }

      if (!dyn_cast<TpuTGOpCodegenInterface>(userOp)) {
        // At least one userOp is not Tg CodeGen Op
        llvm::dbgs() << "    converted op is needed\n";
        return true;
      }

    } else {
      // op is not Tg CodeGenOp

      // Handle LoadWeight
      if (dyn_cast<LoadWeightOp>(op)) {
          llvm::dbgs() << "    converted op is not needed\n";
          return false;
      }

      if (dyn_cast<TpuTGOpCodegenInterface>(userOp)) {
        // At least one userOp is Tg CodeGen Op

        llvm::dbgs() << "    converted op is needed\n";
        return true;

      } else {
        // Not Tg CodeGenOp, but still need converted op
      }
    }
  }

  llvm::dbgs() << "    converted op is not needed\n";

  return false;
}

template <typename TensorTyOp>
class AddTypeConvertedForNotLowedOpPattern : public OpRewritePattern<TensorTyOp> {
public:
  using OpRewritePattern<TensorTyOp>::OpRewritePattern;

  AddTypeConvertedForNotLowedOpPattern(MLIRContext *ctx)
      : OpRewritePattern<TensorTyOp>(ctx) {}

  PatternMatchResult matchAndRewrite(TensorTyOp tensorTyOp,
                                     PatternRewriter &rewriter) const override {
    auto op = tensorTyOp.getOperation();

    llvm::dbgs() << "AddTypeConvertedForNotLowedOpPattern op "
                 << op->getName() << "\n";

    for (auto &user : op->getResult(0)->getUses()) {
      Operation *userOp = user.getOwner();
      llvm::dbgs() << "  userOp " << userOp->getName() << "\n";
      if (dyn_cast<TG_TensorToMemRefOp>(userOp)) {
        llvm::dbgs() << "    TensorToMemRefOp already added, matchFailure\n";
        return Pattern::matchFailure();
      }
    }

    if (isConvertedOpNeeded(op)) {
      auto newTpuOp = rewriter.template create<TensorTyOp>(op->getLoc(),
          op->getResult(0)->getType(),
          op->getOperands(),
          op->getAttrs());

      llvm::dbgs() << "    add TensorToMemRef, MemRefToTensor\n";
      auto tensorToMemRefOp =
          rewriter.create<TG_TensorToMemRefOp>(newTpuOp.getLoc(),
                                               newTpuOp.getResult());
      auto memRefToTensorOp =
          rewriter.create<TG_MemRefToTensorOp>(tensorToMemRefOp.getLoc(),
                                               tensorToMemRefOp.getResult());

      auto oldResult = op->getResult(0);
      oldResult->replaceAllUsesWith(memRefToTensorOp.getResult());
      rewriter.eraseOp(op);

      return Pattern::matchSuccess();
    }

    return Pattern::matchFailure();
  }
};

constexpr StringRef kTempBufferAttr = "temp";

Operation *GetLastUse(Value *value) {
  Operation *last = value->getDefiningOp();
  for (auto &user : value->getUses()) {
    Operation *user_op = user.getOwner();
    if (!user_op->isBeforeInBlock(last)) {
      last = user_op;
    }
  }
  return last;
}

// TPU_MemRegionAttr
//   TPU_MEM_REGION_INPUT, TPU_MEM_REGION_OUTPUT,
//   TPU_MEM_REGION_ACTIVATION, TPU_MEM_REGION_WEIGHT
unsigned getMemorySpace(Operation *op) {
  if (dyn_cast<InputOp>(op))
    return 0; // TPU_MEM_REGION_INPUT
  else if (dyn_cast<LoadWeightOp>(op))
    return 3; // TPU_MEM_REGION_WEIGHT

  return 2; // TPU_MEM_REGION_ACTIVATION
}

Value *InsertAllocAndDealloc(Location loc, Value *result,
                             PatternRewriter *rewriter) {
  auto *op = result->getDefiningOp();
  auto result_type = result->getType().dyn_cast<ShapedType>();
  auto memref_type =
      MemRefType::get(result_type.getShape(), result_type.getElementType(),
                      {}, getMemorySpace(op));

  OpBuilder allocBuilder(op);
  auto alloc = allocBuilder.create<AllocOp>(loc, memref_type);
  alloc.setAttr(kTempBufferAttr, rewriter->getBoolAttr(true));

  auto *lastUsedOp = GetLastUse(result);
  allocBuilder.setInsertionPoint(op->getBlock(),
                                 std::next(Block::iterator(lastUsedOp)));
  auto deallocOp = allocBuilder.create<DeallocOp>(loc, alloc);

  // Place DeallOp after last use
  lastUsedOp->moveBefore(deallocOp);

  return alloc;
}

Value *GetBufferForResultValue(Location loc, Value *result,
                               PatternRewriter *rewriter) {
  return InsertAllocAndDealloc(loc, result, rewriter);
}

template <typename TensorTyOp, typename MemRefTyOp>
class convertTgOpToMemRefPattern : public ConversionPattern {
public:
  explicit convertTgOpToMemRefPattern(MLIRContext *context)
      : ConversionPattern(TensorTyOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const final {
    llvm::dbgs() << "convertTgOpToMemRefPattern op " << op->getName() << "\n";

    const auto &original_results = op->getResults();

    SmallVector<Value *, 4> buffer_args;

    // Operand from not-lowed Op
    for (auto *operand : operands) {
      auto operandOp = operand->getDefiningOp();
      bool useOriginalOperand = true;

      // It is possible that operandOp is null during conversion.
      // E.g QuantOp is conneted to TensorLoadOp, but TensorLoadOp is erased.
      if (operandOp) {
        llvm::dbgs() << "  operand " << operandOp->getName() << "\n";
        if (dyn_cast<TG_MemRefToTensorOp>(operandOp)) {
          llvm::dbgs() << "    erase MemRefToTensorOp\n";
          auto newOperand = operandOp->getOperand(0);
          operandOp->getResult(0)->replaceAllUsesWith(newOperand);
          rewriter.eraseOp(operandOp);
          buffer_args.push_back(operandOp->getOperand(0));

          useOriginalOperand = false;
        }
      }
      if (useOriginalOperand) {
        buffer_args.push_back(operand);
      }
    }

    // Result used by not-lowed Op
    for (auto result : original_results) {
      for (auto &user : result->getUses()) {
        auto *userOp = user.getOwner();
        if (userOp) {
          llvm::dbgs() << "  result userOp " << userOp->getName() << "\n";
          if (dyn_cast<TG_TensorToMemRefOp>(userOp)) {
            llvm::dbgs() << "    erase TensorToMemRefOp\n";
            userOp->getResult(0)->replaceAllUsesWith(result);
            rewriter.eraseOp(userOp);
          }
        }
      }

      buffer_args.push_back(
          GetBufferForResultValue(op->getLoc(), result, &rewriter));
    }

    rewriter.create<MemRefTyOp>(op->getLoc(), llvm::None, buffer_args,
                                op->getAttrs());
    rewriter.replaceOp(
        op, ArrayRef<Value *>(buffer_args).slice(op->getOperands().size()),
        original_results);

    return Pattern::matchSuccess();
  }
};

template <typename TensorTyOp, typename MemRefTyOp>
class convertTgInputOpToMemRefPattern : public ConversionPattern {
public:
  explicit convertTgInputOpToMemRefPattern(MLIRContext *context)
      : ConversionPattern(TensorTyOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const final {

    llvm::dbgs() << "convertTgInputOpToMemRefPattern op " << op->getName() << "\n";
    const auto &original_results = op->getResults();

    SmallVector<Value *, 4> buffer_args(operands.begin(), operands.end());
    for (auto result : original_results) {
      buffer_args.push_back(
          GetBufferForResultValue(op->getLoc(), result, &rewriter));
    }

    rewriter.create<MemRefTyOp>(op->getLoc(), llvm::None, buffer_args,
                                op->getAttrs());
    rewriter.replaceOp(op,
                       ArrayRef<Value *>(buffer_args).slice(operands.size()),
                       original_results);

    return matchSuccess();
  }
};

class convertMemRefToTensorOpPattern : public ConversionPattern {
public:
  explicit convertMemRefToTensorOpPattern(MLIRContext *context)
      : ConversionPattern(TG_MemRefToTensorOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const final {

    // Not-lowed op -> TensorToMemRef -> MemRefToTensor -> lowed op
    // Not-lowed op -> TensorToMemRef -> lowed op
    for (auto &user : op->getResult(0)->getUses()) {
      Operation *userOp = user.getOwner();
      if (dyn_cast<TpuTGOpCodegenInterface>(userOp)) {
        op->getResult(0)->replaceAllUsesWith(op->getOperand(0));
        rewriter.eraseOp(op);
        return matchSuccess();
      }
    }
    return matchFailure();
  }
};

#if 0
class convertTensorLoadOpPattern : public ConversionPattern {
public:
  explicit convertTensorLoadOpPattern(MLIRContext *context)
      : ConversionPattern(TensorLoadOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const final {


    // TensorLoadOp does MemRefType -> TensorType conversion.
    // Replace all uses then erase itself.
    auto *result = op->getResult(0);
    result->replaceAllUsesWith(operands[0]);

    rewriter.eraseOp(op);

    return matchSuccess();
  }
};
#endif

class convertTensorStoreOpPattern : public ConversionPattern {
public:
  explicit convertTensorStoreOpPattern(MLIRContext *context)
      : ConversionPattern(TensorStoreOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const final {

    // Not lowed op -> TensorStore
    if (!dyn_cast<TpuTGOpCodegenInterface>(operands[0]->getDefiningOp())) {
      llvm::dbgs() << "convertTensorStoreOpPattern operandOp "
                   << operands[0]->getDefiningOp()->getName() << "\n";
      return matchFailure();
    }

    // TensorStoreOp does TensorType -> MemRefType conversion.
    // Replace all uses then erase itself.
    auto source = op->getOperand(0);
    auto dest = op->getOperand(1);
    dest->replaceAllUsesWith(source);

    auto allocOp = dest->getDefiningOp();
    for (auto &user : allocOp->getResult(0)->getUses()) {
        Operation *user_op = user.getOwner();
        if (auto deallocOp = dyn_cast_or_null<DeallocOp>(user_op))
          rewriter.eraseOp(deallocOp);
        else
          assert("Expect DeallocOp");
    }
    rewriter.eraseOp(allocOp);

    rewriter.eraseOp(op);

    return matchSuccess();
  }
};

struct ConvertTgOpToMemRefPass : public FunctionPass<ConvertTgOpToMemRefPass> {
  void runOnFunction() override;
};
} // anonymous namespace

void ConvertTgOpToMemRefPass::runOnFunction() {
  auto fn = getFunction();
  auto *context = &getContext();

  // Add converted operation between between lowed and not-lowed ones.
  // OpRewritePattern handle op from tail to head.
  // I need to insert dummy operations using OpRewritePattern and remove
  // some during lowering for non-lowed op.
  OwningRewritePatternList patterns;
  patterns.insert<
      AddTypeConvertedForNotLowedOpPattern<tpu::LoadWeightOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::ReshapeOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_INT8_EltwiseAddOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_INT8_FullyConnectedOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_INT8_PC_Conv2DOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_INT8_PoolAvg2DOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_INT8_PoolMax2DOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::QuantOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::SoftmaxOp>
      >(context);
  applyPatternsGreedily(fn, patterns);

#if 1
  // Lower op.
  // ConversionPattern handle op from head to tail.
  patterns.clear();
  ConversionTarget target(getContext());

  target.addLegalOp<tpu::TG_MemRef_INT8_InputOp>();
  target.addLegalOp<tpu::TG_MemRef_INT8_EltwiseAddOp>();
  target.addLegalOp<tpu::TG_MemRef_INT8_FullyConnectedOp>();
  target.addLegalOp<tpu::TG_MemRef_INT8_PC_Conv2DOp>();
  target.addLegalOp<tpu::TG_MemRef_INT8_PoolAvg2DOp>();
  target.addLegalOp<tpu::TG_MemRef_INT8_PoolMax2DOp>();
  target.addLegalOp<tpu::TG_MemRef_LoadWeightOp>();
  target.addLegalOp<tpu::TG_MemRef_ReshapeOp>();
  target.addLegalOp<tpu::TG_MemRefToTensorOp>();
  target.addLegalOp<tpu::TG_TensorToMemRefOp>();

  patterns.insert<
      convertTgInputOpToMemRefPattern<tpu::TG_INT8_InputOp, tpu::TG_MemRef_INT8_InputOp>,
      convertTgOpToMemRefPattern<tpu::LoadWeightOp, tpu::TG_MemRef_LoadWeightOp>,
      convertTgOpToMemRefPattern<tpu::ReshapeOp, tpu::TG_MemRef_ReshapeOp>,
      convertTgOpToMemRefPattern<tpu::TG_INT8_EltwiseAddOp, tpu::TG_MemRef_INT8_EltwiseAddOp>,
      convertTgOpToMemRefPattern<tpu::TG_INT8_FullyConnectedOp, tpu::TG_MemRef_INT8_FullyConnectedOp>,
      convertTgOpToMemRefPattern<tpu::TG_INT8_PC_Conv2DOp, tpu::TG_MemRef_INT8_PC_Conv2DOp>,
      convertTgOpToMemRefPattern<tpu::TG_INT8_PoolAvg2DOp, tpu::TG_MemRef_INT8_PoolAvg2DOp>,
      convertTgOpToMemRefPattern<tpu::TG_INT8_PoolMax2DOp, tpu::TG_MemRef_INT8_PoolMax2DOp>,
      convertMemRefToTensorOpPattern,
      convertTensorStoreOpPattern
      >(context);
  if (failed(applyPartialConversion(fn, target, patterns)))
    signalPassFailure();
#endif

}

std::unique_ptr<OpPassBase<FuncOp>> mlir::createConvertTgOpToMemRefPass() {
  return std::make_unique<ConvertTgOpToMemRefPass>();
}

static PassRegistration<ConvertTgOpToMemRefPass>
    pass("convert-tg-op-to-memref", "Convert tg op from TensorType to MemRefType");
