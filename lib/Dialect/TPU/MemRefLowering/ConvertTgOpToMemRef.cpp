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

static bool isDetectionOutputOp(Operation* op) {
  if (auto castOp = dyn_cast<GenericCpuOp>(op)) {
    if (castOp.operation_name() == "tpu.detectionoutput") {
      return true;
    }
  }
  return false;
}

static bool isQuantOp(Operation *op) {
  if (dyn_cast<QuantOp>(op)) {
    return true;
  }
  if (auto castOp = dyn_cast<GenericCpuOp>(op)) {
    if (castOp.operation_name() == QuantOp::getOperationName()) {
      return true;
    }
  }
  return false;
}

static bool isConvertedOpNeeded(Operation *op) {
  LLVM_DEBUG(
    if (dyn_cast<TpuTGOpCodegenInterface>(op))
      llvm::dbgs() << "  isConvertedOpNeeded op " << op->getName() << ", is TgCodeGen\n";
    else
     llvm::dbgs() << "  isConvertedOpNeeded op " << op->getName() << ", is not TgCodeGen\n";
  );

  for (auto &user : op->getResult(0)->getUses()) {
    Operation *userOp = user.getOwner();

    LLVM_DEBUG(
      if (dyn_cast<TpuTGOpCodegenInterface>(userOp))
        llvm::dbgs() << "    userOp " << userOp->getName() << ", is TgCodeGend\n";
      else
        llvm::dbgs() << "    userOp " << userOp->getName() << ", is not TgCodeGend\n";
    );

    // Handle Quant, treat quant as not-lowed op
    if (isQuantOp(op)) {
      if (dyn_cast<TpuTGOpCodegenInterface>(userOp)) {
        // Quant connected to lowed Op.
        // E.g. Quant -> conv
        LLVM_DEBUG(llvm::dbgs() << "    converted op is needed\n";);
        return true;
      } else {
        // Quant connected to not-lowed op.
        LLVM_DEBUG(llvm::dbgs() << "    converted op is not needed\n";);
        return false;
      }
    }

    if (dyn_cast<TpuTGOpCodegenInterface>(op)) {
      // op is Tg CodeGenOp

      if (isQuantOp(userOp)) {
        // Lower op connected to Quant
        LLVM_DEBUG(llvm::dbgs() << "    converted op is needed\n";);
        return true;
      }

      if (!dyn_cast<TpuTGOpCodegenInterface>(userOp)) {
        // At least one userOp is not Tg CodeGen Op
        LLVM_DEBUG(llvm::dbgs() << "    converted op is needed\n";);
        return true;
      }

    } else {
      // op is not Tg CodeGenOp

      // Handle LoadWeight
      if (dyn_cast<LoadWeightOp>(op)) {
          if (isDetectionOutputOp(userOp)) {
            LLVM_DEBUG(llvm::dbgs() << "    converted op is needed\n";);
            return true;
          }
          LLVM_DEBUG(llvm::dbgs() << "    converted op is not needed\n";);
          return false;
      }

      if (dyn_cast<TpuTGOpCodegenInterface>(userOp)) {
        // At least one userOp is Tg CodeGen Op

        LLVM_DEBUG(llvm::dbgs() << "    converted op is needed\n";);
        return true;

      } else {
        // Not Tg CodeGenOp, but still need converted op
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "    converted op is not needed\n";);

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

    LLVM_DEBUG(llvm::dbgs() << "AddTypeConvertedForNotLowedOpPattern op "
                 << op->getName() << "\n";);

    for (auto &user : op->getResult(0)->getUses()) {
      Operation *userOp = user.getOwner();
      LLVM_DEBUG(llvm::dbgs() << "  userOp " << userOp->getName() << "\n";);
      if (dyn_cast<TG_TensorToMemRefOp>(userOp)) {
        LLVM_DEBUG(llvm::dbgs() << "    TensorToMemRefOp already added, matchFailure\n";);
        return Pattern::matchFailure();
      }
    }

    if (isConvertedOpNeeded(op)) {
      auto newTpuOp = rewriter.template create<TensorTyOp>(op->getLoc(),
          op->getResult(0)->getType(),
          op->getOperands(),
          op->getAttrs());

      LLVM_DEBUG(llvm::dbgs() << "    add TensorToMemRef, MemRefToTensor\n";);
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

bool isMemoryAliasOp(Operation *op) {
  if (dyn_cast<tpu::ReshapeOp>(op)) {
    return true;
  } else if (dyn_cast<tpu::TG_INT8_SliceOp>(op) ||
             dyn_cast<tpu::TG_BF16_SliceOp>(op)) {
    auto resultType = op->getResult(0)->getType();
    auto tensorType = resultType.dyn_cast<RankedTensorType>();
    auto batch = tensorType.getShape()[0];

    // Avoid copy when batch = 1
    if (batch == 1)
      return true;
  }
  return false;
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

  if (isMemoryAliasOp(lastUsedOp)) {
    // DeallocOp should be after last use of memory-aliased op.
    auto lastUsedByAliasOp = GetLastUse(lastUsedOp->getResult(0));

    // Cannot use moveBefore to place deallocOp just after last use of
    // memory-aliased op.
    // Put far away from last use
    LLVM_DEBUG(llvm::dbgs() << "InsertAllocAndDealloc: result op " << op->getName()
                 << ", lastUsedOp " << lastUsedOp->getName()
                 << ", lasedUsedByAliasOp " << lastUsedByAliasOp->getName()
                 << ", last use of lastUsedByAliasOp "
                 << GetLastUse(lastUsedByAliasOp->getResult(0))->getName()
                 << "\n";);
    auto deallocOp =
        allocBuilder.create<DeallocOp>(lastUsedByAliasOp->getLoc(), alloc);
    deallocOp.getOperation()->moveBefore(
        GetLastUse(lastUsedByAliasOp->getResult(0)));
  } else {
    // Place DeallOp after last use.
    auto deallocOp = allocBuilder.create<DeallocOp>(loc, alloc);
    lastUsedOp->moveBefore(deallocOp);
  }

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
    LLVM_DEBUG(llvm::dbgs() << "convertTgOpToMemRefPattern op " << op->getName() << "\n";);

    const auto &original_results = op->getResults();

    SmallVector<Value *, 4> buffer_args;

    // Operand from not-lowed Op
    for (auto *operand : operands) {
      auto operandOp = operand->getDefiningOp();
      bool useOriginalOperand = true;

      // It is possible that operandOp is null during conversion.
      // E.g QuantOp is conneted to TensorLoadOp, but TensorLoadOp is erased.
      if (operandOp) {
        LLVM_DEBUG(llvm::dbgs() << "  operand " << operandOp->getName() << "\n";);
        if (dyn_cast<TG_MemRefToTensorOp>(operandOp)) {
          LLVM_DEBUG(llvm::dbgs() << "    erase MemRefToTensorOp\n";);
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
          LLVM_DEBUG(llvm::dbgs() << "  result userOp " << userOp->getName() << "\n";);
          if (dyn_cast<TG_TensorToMemRefOp>(userOp)) {
            LLVM_DEBUG(llvm::dbgs() << "    erase TensorToMemRefOp\n";);
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

    LLVM_DEBUG(llvm::dbgs() << "convertTgInputOpToMemRefPattern op " << op->getName() << "\n";);
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

class convertTensorStoreOpPattern : public ConversionPattern {
public:
  explicit convertTensorStoreOpPattern(MLIRContext *context)
      : ConversionPattern(TensorStoreOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const final {

    // Not lowed op -> TensorStore
    if (!dyn_cast<TpuTGOpCodegenInterface>(operands[0]->getDefiningOp())) {
      LLVM_DEBUG(llvm::dbgs() << "convertTensorStoreOpPattern operandOp "
                   << operands[0]->getDefiningOp()->getName() << ", skipped\n";);
      return matchFailure();
    }

    // Treat QuantOp as not-lowed op
    if (isQuantOp(operands[0]->getDefiningOp())) {
      LLVM_DEBUG(llvm::dbgs() << "convertTensorStoreOpPattern operandOp "
                   << operands[0]->getDefiningOp()->getName() << ", skipped\n";);
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
          llvm_unreachable("Expect DeallocOp");
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
      AddTypeConvertedForNotLowedOpPattern<tpu::QuantOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::ReshapeOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::SoftmaxOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::GenericCpuOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_INT8_BroadcastMulOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_BF16_BroadcastMulOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_INT8_ConcatOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_BF16_ConcatOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_INT8_PT_Conv2DOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_INT8_PC_Conv2DOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_BF16_Conv2DOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_INT8_CropOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_BF16_CropOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_INT8_ClipOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_BF16_ClipOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_INT8_PT_DeConv2DOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_INT8_PC_DeConv2DOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_BF16_DeConv2DOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_INT8_EltwiseAddOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_INT8_EltwiseMaxOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_INT8_EltwiseMulOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_BF16_EltwiseAddOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_BF16_EltwiseMaxOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_BF16_EltwiseMulOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_INT8_FullyConnectedOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_BF16_FullyConnectedOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_INT8_LeakyReluOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_BF16_LeakyReluOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_INT8_LutOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_BF16_LutOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_INT8_LrnOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_BF16_LrnOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_INT8_PermuteOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_BF16_PermuteOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_INT8_PoolAvg2DOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_INT8_PoolMax2DOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_BF16_PoolAvg2DOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_BF16_PoolMax2DOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_INT8_ShuffleChannelOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_BF16_ShuffleChannelOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_INT8_PixelShuffleOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_BF16_PixelShuffleOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_INT8_PReluOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_BF16_PReluOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_INT8_ReluOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_BF16_ReluOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_INT8_SliceOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_BF16_SliceOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_INT8_SwapChannelOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_BF16_SwapChannelOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_INT8_UpsampleOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TG_BF16_UpsampleOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TL_LA_Conv2DOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TL_LW_Conv2DOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TL_EltwiseAddOp>,
      AddTypeConvertedForNotLowedOpPattern<tpu::TL_LutOp>
      >(context);
  applyPatternsGreedily(fn, patterns);

#if 1
  // Lower op.
  // ConversionPattern handle op from head to tail.
  patterns.clear();
  ConversionTarget target(getContext());

  target.addLegalOp<tpu::TG_MemRef_INT8_BroadcastMulOp>();
  target.addLegalOp<tpu::TG_MemRef_BF16_BroadcastMulOp>();
  target.addLegalOp<tpu::TG_MemRef_INT8_ConcatOp>();
  target.addLegalOp<tpu::TG_MemRef_BF16_ConcatOp>();
  target.addLegalOp<tpu::TG_MemRef_INT8_PT_Conv2DOp>();
  target.addLegalOp<tpu::TG_MemRef_INT8_PC_Conv2DOp>();
  target.addLegalOp<tpu::TG_MemRef_BF16_Conv2DOp>();
  target.addLegalOp<tpu::TG_MemRef_INT8_CropOp>();
  target.addLegalOp<tpu::TG_MemRef_BF16_CropOp>();
  target.addLegalOp<tpu::TG_MemRef_INT8_ClipOp>();
  target.addLegalOp<tpu::TG_MemRef_BF16_ClipOp>();
  target.addLegalOp<tpu::TG_MemRef_INT8_PT_DeConv2DOp>();
  target.addLegalOp<tpu::TG_MemRef_INT8_PC_DeConv2DOp>();
  target.addLegalOp<tpu::TG_MemRef_BF16_DeConv2DOp>();
  target.addLegalOp<tpu::TG_MemRef_INT8_EltwiseAddOp>();
  target.addLegalOp<tpu::TG_MemRef_INT8_EltwiseMaxOp>();
  target.addLegalOp<tpu::TG_MemRef_INT8_EltwiseMulOp>();
  target.addLegalOp<tpu::TG_MemRef_BF16_EltwiseAddOp>();
  target.addLegalOp<tpu::TG_MemRef_BF16_EltwiseMaxOp>();
  target.addLegalOp<tpu::TG_MemRef_BF16_EltwiseMulOp>();
  target.addLegalOp<tpu::TG_MemRef_INT8_FullyConnectedOp>();
  target.addLegalOp<tpu::TG_MemRef_BF16_FullyConnectedOp>();
  target.addLegalOp<tpu::TG_MemRef_INT8_LeakyReluOp>();
  target.addLegalOp<tpu::TG_MemRef_BF16_LeakyReluOp>();
  target.addLegalOp<tpu::TG_MemRef_INT8_LutOp>();
  target.addLegalOp<tpu::TG_MemRef_BF16_LutOp>();
  target.addLegalOp<tpu::TG_MemRef_INT8_LrnOp>();
  target.addLegalOp<tpu::TG_MemRef_BF16_LrnOp>();
  target.addLegalOp<tpu::TG_MemRef_INT8_PermuteOp>();
  target.addLegalOp<tpu::TG_MemRef_BF16_PermuteOp>();
  target.addLegalOp<tpu::TG_MemRef_INT8_PoolAvg2DOp>();
  target.addLegalOp<tpu::TG_MemRef_INT8_PoolMax2DOp>();
  target.addLegalOp<tpu::TG_MemRef_BF16_PoolAvg2DOp>();
  target.addLegalOp<tpu::TG_MemRef_BF16_PoolMax2DOp>();
  target.addLegalOp<tpu::TG_MemRef_INT8_ShuffleChannelOp>();
  target.addLegalOp<tpu::TG_MemRef_BF16_ShuffleChannelOp>();
  target.addLegalOp<tpu::TG_MemRef_INT8_PixelShuffleOp>();
  target.addLegalOp<tpu::TG_MemRef_BF16_PixelShuffleOp>();
  target.addLegalOp<tpu::TG_MemRef_INT8_PReluOp>();
  target.addLegalOp<tpu::TG_MemRef_BF16_PReluOp>();
  target.addLegalOp<tpu::TG_MemRef_INT8_ReluOp>();
  target.addLegalOp<tpu::TG_MemRef_BF16_ReluOp>();
  target.addLegalOp<tpu::TG_MemRef_INT8_SliceOp>();
  target.addLegalOp<tpu::TG_MemRef_BF16_SliceOp>();
  target.addLegalOp<tpu::TG_MemRef_INT8_SwapChannelOp>();
  target.addLegalOp<tpu::TG_MemRef_BF16_SwapChannelOp>();
  target.addLegalOp<tpu::TG_MemRef_INT8_UpsampleOp>();
  target.addLegalOp<tpu::TG_MemRef_BF16_UpsampleOp>();
  target.addLegalOp<tpu::TL_MemRef_LA_Conv2DOp>();
  target.addLegalOp<tpu::TL_MemRef_LW_Conv2DOp>();
  target.addLegalOp<tpu::TL_MemRef_EltwiseAddOp>();
  target.addLegalOp<tpu::TL_MemRef_LutOp>();

  target.addLegalOp<tpu::TG_MemRef_LoadWeightOp>();
  target.addLegalOp<tpu::TG_MemRef_ReshapeOp>();

  target.addLegalOp<tpu::TG_MemRefToTensorOp>();
  target.addLegalOp<tpu::TG_TensorToMemRefOp>();

  patterns.insert<
      convertTgOpToMemRefPattern<tpu::LoadWeightOp, tpu::TG_MemRef_LoadWeightOp>,
      convertTgOpToMemRefPattern<tpu::ReshapeOp, tpu::TG_MemRef_ReshapeOp>,
      convertTgOpToMemRefPattern<tpu::TG_INT8_BroadcastMulOp, tpu::TG_MemRef_INT8_BroadcastMulOp>,
      convertTgOpToMemRefPattern<tpu::TG_BF16_BroadcastMulOp, tpu::TG_MemRef_BF16_BroadcastMulOp>,
      convertTgOpToMemRefPattern<tpu::TG_INT8_ConcatOp, tpu::TG_MemRef_INT8_ConcatOp>,
      convertTgOpToMemRefPattern<tpu::TG_BF16_ConcatOp, tpu::TG_MemRef_BF16_ConcatOp>,
      convertTgOpToMemRefPattern<tpu::TG_INT8_PT_Conv2DOp, tpu::TG_MemRef_INT8_PT_Conv2DOp>,
      convertTgOpToMemRefPattern<tpu::TG_INT8_PC_Conv2DOp, tpu::TG_MemRef_INT8_PC_Conv2DOp>,
      convertTgOpToMemRefPattern<tpu::TG_BF16_Conv2DOp, tpu::TG_MemRef_BF16_Conv2DOp>,
      convertTgOpToMemRefPattern<tpu::TG_INT8_CropOp, tpu::TG_MemRef_INT8_CropOp>,
      convertTgOpToMemRefPattern<tpu::TG_BF16_CropOp, tpu::TG_MemRef_BF16_CropOp>,
      convertTgOpToMemRefPattern<tpu::TG_INT8_ClipOp, tpu::TG_MemRef_INT8_ClipOp>,
      convertTgOpToMemRefPattern<tpu::TG_BF16_ClipOp, tpu::TG_MemRef_BF16_ClipOp>,
      convertTgOpToMemRefPattern<tpu::TG_INT8_PT_DeConv2DOp, tpu::TG_MemRef_INT8_PT_DeConv2DOp>,
      convertTgOpToMemRefPattern<tpu::TG_INT8_PC_DeConv2DOp, tpu::TG_MemRef_INT8_PC_DeConv2DOp>,
      convertTgOpToMemRefPattern<tpu::TG_BF16_DeConv2DOp, tpu::TG_MemRef_BF16_DeConv2DOp>,
      convertTgOpToMemRefPattern<tpu::TG_INT8_EltwiseAddOp, tpu::TG_MemRef_INT8_EltwiseAddOp>,
      convertTgOpToMemRefPattern<tpu::TG_INT8_EltwiseMaxOp, tpu::TG_MemRef_INT8_EltwiseMaxOp>,
      convertTgOpToMemRefPattern<tpu::TG_INT8_EltwiseMulOp, tpu::TG_MemRef_INT8_EltwiseMulOp>,
      convertTgOpToMemRefPattern<tpu::TG_BF16_EltwiseAddOp, tpu::TG_MemRef_BF16_EltwiseAddOp>,
      convertTgOpToMemRefPattern<tpu::TG_BF16_EltwiseMaxOp, tpu::TG_MemRef_BF16_EltwiseMaxOp>,
      convertTgOpToMemRefPattern<tpu::TG_BF16_EltwiseMulOp, tpu::TG_MemRef_BF16_EltwiseMulOp>,
      convertTgOpToMemRefPattern<tpu::TG_INT8_FullyConnectedOp, tpu::TG_MemRef_INT8_FullyConnectedOp>,
      convertTgOpToMemRefPattern<tpu::TG_BF16_FullyConnectedOp, tpu::TG_MemRef_BF16_FullyConnectedOp>,
      convertTgOpToMemRefPattern<tpu::TG_INT8_LeakyReluOp, tpu::TG_MemRef_INT8_LeakyReluOp>,
      convertTgOpToMemRefPattern<tpu::TG_BF16_LeakyReluOp, tpu::TG_MemRef_BF16_LeakyReluOp>,
      convertTgOpToMemRefPattern<tpu::TG_INT8_LrnOp, tpu::TG_MemRef_INT8_LrnOp>,
      convertTgOpToMemRefPattern<tpu::TG_BF16_LrnOp, tpu::TG_MemRef_BF16_LrnOp>,
      convertTgOpToMemRefPattern<tpu::TG_INT8_LutOp, tpu::TG_MemRef_INT8_LutOp>,
      convertTgOpToMemRefPattern<tpu::TG_BF16_LutOp, tpu::TG_MemRef_BF16_LutOp>,
      convertTgOpToMemRefPattern<tpu::TG_INT8_PermuteOp, tpu::TG_MemRef_INT8_PermuteOp>,
      convertTgOpToMemRefPattern<tpu::TG_BF16_PermuteOp, tpu::TG_MemRef_BF16_PermuteOp>,
      convertTgOpToMemRefPattern<tpu::TG_INT8_PoolAvg2DOp, tpu::TG_MemRef_INT8_PoolAvg2DOp>,
      convertTgOpToMemRefPattern<tpu::TG_INT8_PoolMax2DOp, tpu::TG_MemRef_INT8_PoolMax2DOp>,
      convertTgOpToMemRefPattern<tpu::TG_BF16_PoolAvg2DOp, tpu::TG_MemRef_BF16_PoolAvg2DOp>,
      convertTgOpToMemRefPattern<tpu::TG_BF16_PoolMax2DOp, tpu::TG_MemRef_BF16_PoolMax2DOp>,
      convertTgOpToMemRefPattern<tpu::TG_INT8_ShuffleChannelOp, tpu::TG_MemRef_INT8_ShuffleChannelOp>,
      convertTgOpToMemRefPattern<tpu::TG_BF16_ShuffleChannelOp, tpu::TG_MemRef_BF16_ShuffleChannelOp>,
      convertTgOpToMemRefPattern<tpu::TG_INT8_PixelShuffleOp, tpu::TG_MemRef_INT8_PixelShuffleOp>,
      convertTgOpToMemRefPattern<tpu::TG_BF16_PixelShuffleOp, tpu::TG_MemRef_BF16_PixelShuffleOp>,
      convertTgOpToMemRefPattern<tpu::TG_INT8_PReluOp, tpu::TG_MemRef_INT8_PReluOp>,
      convertTgOpToMemRefPattern<tpu::TG_BF16_PReluOp, tpu::TG_MemRef_BF16_PReluOp>,
      convertTgOpToMemRefPattern<tpu::TG_INT8_ReluOp, tpu::TG_MemRef_INT8_ReluOp>,
      convertTgOpToMemRefPattern<tpu::TG_BF16_ReluOp, tpu::TG_MemRef_BF16_ReluOp>,
      convertTgOpToMemRefPattern<tpu::TG_INT8_SliceOp, tpu::TG_MemRef_INT8_SliceOp>,
      convertTgOpToMemRefPattern<tpu::TG_BF16_SliceOp, tpu::TG_MemRef_BF16_SliceOp>,
      convertTgOpToMemRefPattern<tpu::TG_INT8_SwapChannelOp, tpu::TG_MemRef_INT8_SwapChannelOp>,
      convertTgOpToMemRefPattern<tpu::TG_BF16_SwapChannelOp, tpu::TG_MemRef_BF16_SwapChannelOp>,
      convertTgOpToMemRefPattern<tpu::TG_INT8_UpsampleOp, tpu::TG_MemRef_INT8_UpsampleOp>,
      convertTgOpToMemRefPattern<tpu::TG_BF16_UpsampleOp, tpu::TG_MemRef_BF16_UpsampleOp>,
      convertTgOpToMemRefPattern<tpu::TL_LA_Conv2DOp, tpu::TL_MemRef_LA_Conv2DOp>,
      convertTgOpToMemRefPattern<tpu::TL_LW_Conv2DOp, tpu::TL_MemRef_LW_Conv2DOp>,
      convertTgOpToMemRefPattern<tpu::TL_EltwiseAddOp, tpu::TL_MemRef_EltwiseAddOp>,
      convertTgOpToMemRefPattern<tpu::TL_LutOp, tpu::TL_MemRef_LutOp>,
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
