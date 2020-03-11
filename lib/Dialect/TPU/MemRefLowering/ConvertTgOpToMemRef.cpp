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
  if (llvm::dyn_cast<tpu::InputOp>(op))
    return 0; // TPU_MEM_REGION_INPUT
  else if (llvm::dyn_cast<tpu::LoadWeightOp>(op))
    return 3; // TPU_MEM_REGION_WEIGHT

  return 2; // TPU_MEM_REGION_ACTIVATION
}

Value *InsertAllocAndDealloc(Location loc, Value *result,
                             ConversionPatternRewriter *rewriter) {
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
                               ConversionPatternRewriter *rewriter) {
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

class convertTensorStoreOpPattern : public ConversionPattern {
public:
  explicit convertTensorStoreOpPattern(MLIRContext *context)
      : ConversionPattern(TensorStoreOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const final {
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
  void runOnFunction() override {
    auto fn = getFunction();
    auto *context = &getContext();
    ConversionTarget target(getContext());
    OwningRewritePatternList patterns;

    target.addLegalOp<tpu::TG_MemRef_INT8_InputOp>();
    target.addLegalOp<tpu::TG_MemRef_INT8_EltwiseAddOp>();
    target.addLegalOp<tpu::TG_MemRef_INT8_FullyConnectedOp>();
    target.addLegalOp<tpu::TG_MemRef_INT8_PC_Conv2DOp>();
    target.addLegalOp<tpu::TG_MemRef_INT8_PoolAvg2DOp>();
    target.addLegalOp<tpu::TG_MemRef_INT8_PoolMax2DOp>();
    target.addLegalOp<tpu::TG_MemRef_LoadWeightOp>();
    target.addLegalOp<tpu::TG_MemRef_QuantOp>();
    target.addLegalOp<tpu::TG_MemRef_ReshapeOp>();
    target.addLegalOp<ReturnOp>();

    patterns.insert<
        convertTgOpToMemRefPattern<tpu::LoadWeightOp, tpu::TG_MemRef_LoadWeightOp>,
        convertTgOpToMemRefPattern<tpu::QuantOp, tpu::TG_MemRef_QuantOp>,
        convertTgOpToMemRefPattern<tpu::ReshapeOp, tpu::TG_MemRef_ReshapeOp>,
        convertTgOpToMemRefPattern<tpu::TG_INT8_InputOp, tpu::TG_MemRef_INT8_InputOp>,
        convertTgOpToMemRefPattern<tpu::TG_INT8_EltwiseAddOp, tpu::TG_MemRef_INT8_EltwiseAddOp>,
        convertTgOpToMemRefPattern<tpu::TG_INT8_FullyConnectedOp, tpu::TG_MemRef_INT8_FullyConnectedOp>,
        convertTgOpToMemRefPattern<tpu::TG_INT8_PC_Conv2DOp, tpu::TG_MemRef_INT8_PC_Conv2DOp>,
        convertTgOpToMemRefPattern<tpu::TG_INT8_PoolAvg2DOp, tpu::TG_MemRef_INT8_PoolAvg2DOp>,
        convertTgOpToMemRefPattern<tpu::TG_INT8_PoolMax2DOp, tpu::TG_MemRef_INT8_PoolMax2DOp>,
        convertTensorLoadOpPattern,
        convertTensorStoreOpPattern
        >(context);
    if (failed(applyPartialConversion(fn, target, patterns)))
      signalPassFailure();
  }
};

} // anonymous namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createConvertTgOpToMemRefPass() {
  return std::make_unique<ConvertTgOpToMemRefPass>();
}

static PassRegistration<ConvertTgOpToMemRefPass>
    pass("convert-tg-op-to-memref", "Convert tg op from TensorType to MemRefType");
