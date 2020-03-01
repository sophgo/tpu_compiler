
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
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/ADT/StringRef.h"


#define DEBUG_TYPE "ConvertFuncToMemRef"

using namespace mlir;

namespace {
class ConvertFuncToMemRefPass : public ModulePass<ConvertFuncToMemRefPass> {
public:
  void runOnModule() final;
};

// Borrow from IREE
// Attemps to resolve the use of a value back to the MemRef it was loaded from.
// Returns either a MemRef view containing the value or nullptr if the value was
// not loaded from a MemRef (or is possibly unknown).
Value *resolveValueToSourceMemRef(Value *value, Operation *useOp) {
  auto *defInstr = value->getDefiningOp();
  if (auto loadOp = dyn_cast_or_null<LoadOp>(defInstr)) {
    return loadOp.getMemRef();
  }
  return nullptr;
}

// Borrow from IREE
MemRefType convertLegacyTypeToMemRef(Type type) {
  if (type.isIntOrIndexOrFloat()) {
    return MemRefType::get({}, type, {}, 0);
  } else if (auto tensorType = type.dyn_cast<RankedTensorType>()) {
    return MemRefType::get(tensorType.getShape(), tensorType.getElementType());
  } else if (auto memRefType = type.dyn_cast<MemRefType>()) {
    return MemRefType::get(memRefType.getShape(), memRefType.getElementType());
  } else {
    llvm_unreachable("Unconvertable type");
  }
}

// Borrow from IREE
bool insertLoad(BlockArgument *oldArg, BlockArgument *newArg, OpBuilder &builder,
                BlockAndValueMapping *mapping) {
  auto loc = oldArg->getOwner()->getParent()->getLoc();
  // If old arg was a memref we don't need to change anything. We still need
  // to remap so that the use lists match through coversion, though.
  if (oldArg->getType().isa<MemRefType>()) {
    mapping->map(oldArg, newArg);
    return false;
  } else if (oldArg->getType().isa<TensorType>()) {
    auto castOp = builder.create<TensorLoadOp>(loc, newArg);
    mapping->map(oldArg, castOp.getResult());
    return false;
  }

  // Insert the load we'll use to unbox the value.
  auto loadedValue = builder.create<LoadOp>(loc, newArg, ArrayRef<Value *>{});
  mapping->map(oldArg, loadedValue);

  return false;
}

// Borrow from IREE
Value* insertStore(Operation *oldOp, Value *oldValue, OpBuilder &builder,
                   BlockAndValueMapping *mapping) {
  auto *newValue = mapping->lookupOrNull(oldValue);
  if (!newValue) {
    return nullptr;
  }

  // If the previous value was already a memref we don't need to change
  // anything.
  if (oldValue->getType().isa<MemRefType>()) {
    return newValue;
  } else if (oldValue->getType().isa<TensorType>()) {
    // Allocate the memref to store the value.
    auto newStorage = builder.create<AllocOp>(
        oldOp->getLoc(), convertLegacyTypeToMemRef(oldValue->getType()));

    // Insert the store we'll use to box the value.
    // builder.create<StoreOp>(oldOp->getLoc(), newValue, newStorage,
    //                         ArrayRef<Value *>{});
    builder.create<TensorStoreOp>(oldOp->getLoc(), newValue, newStorage);
    return newStorage;
  }

  // Look back up and see if we can find the memref the value was loaded from.
  if (auto *sourceMemRef = resolveValueToSourceMemRef(oldValue, oldOp)) {
    return mapping->lookupOrNull(sourceMemRef);
  }

  // Allocate the memref to store the value.
  auto newStorage = builder.create<AllocOp>(
      oldOp->getLoc(), convertLegacyTypeToMemRef(oldValue->getType()));

  // Insert the store we'll use to box the value.
  builder.create<StoreOp>(oldOp->getLoc(), newValue, newStorage,
                          ArrayRef<Value *>{});

  return newStorage;
}

// Borrow from IREE
bool convertReturnOp(Operation *oldOp, OpBuilder &builder,
                     BlockAndValueMapping *mapping) {
  BlockAndValueMapping returnMapping;
  for (auto *oldArg : oldOp->getOperands()) {
    auto newArg = insertStore(oldOp, oldArg, builder, mapping);
    if (!newArg) {
      return true;
    }
    returnMapping.map(oldArg, newArg);
  }

  builder.clone(*oldOp, returnMapping);
  return false;
}

// Borrow from IREE
bool convertOperation(Operation *oldOp, OpBuilder &builder,
                      BlockAndValueMapping *mapping) {
  if (isa<ReturnOp>(oldOp)) {
    return convertReturnOp(oldOp, builder, mapping);
  } else {
    builder.clone(*oldOp, *mapping);
    return false;
  }
}

// Borrow from IREE
FunctionType getMemRefFunctionType(FunctionType type) {
  Builder builder(type.getContext());
  llvm::SmallVector<Type, 8> replacementInputs;
  for (auto type_input : type.getInputs()) {
    auto memRefType = convertLegacyTypeToMemRef(type_input);
    if (!memRefType) {
      return nullptr;
    }
    replacementInputs.push_back(memRefType);
  }
  llvm::SmallVector<Type, 8> replacementResults;
  for (auto resultType : type.getResults()) {
    auto memRefType = convertLegacyTypeToMemRef(resultType);
    if (!memRefType) {
      return nullptr;
    }
    replacementResults.push_back(memRefType);
  }
  return builder.getFunctionType(replacementInputs, replacementResults);
}

// Borrow from IREE
bool convertFunction(FuncOp oldFunc, FuncOp newFunc) {
  OpBuilder builder(newFunc.getBody());
  BlockAndValueMapping mapping;

  // Creat new blocks matching the expected arguments of the old ones.
  // This sets up the block mappings to enable us to reference blocks forward
  // during conversion.
  newFunc.getBlocks().clear();
  for (auto &oldBlock : oldFunc.getBlocks()) {
    auto *newBlock = builder.createBlock(&newFunc.getBody());
    for (auto *oldArg : oldBlock.getArguments()) {
      // Replace the block args with memRefs.
      auto memRefType = convertLegacyTypeToMemRef(oldArg->getType());
      if (!memRefType) return true;
      auto *newArg = newBlock->addArgument(memRefType);
      
      // Insert loads to preserved type, if needed.
      // This will replace all uses of the oldArg with the loaded value from
      // newArg so that the block contents are still using unwrapped values.
      if (insertLoad(oldArg, newArg, builder, &mapping)) {
        return true;
      }
    }
    mapping.map(&oldBlock, newBlock);
  }

  // Convert all ops in the blocks.
  for (auto &oldBlock : oldFunc.getBlocks()) {
    builder.setInsertionPointToEnd(mapping.lookupOrNull(&oldBlock));
    for (auto &oldOp : oldBlock.getOperations()) {
      if (convertOperation(&oldOp, builder, &mapping)) {
        return true;
      }
    }
  }

  return false;
}

void ConvertFuncToMemRefPass::runOnModule() {
  auto module = getModule();
  std::vector<std::pair<FuncOp, FuncOp>> convertedFunctions;

  for (auto oldFunc : module.getOps<FuncOp>()) {
    auto functionType = getMemRefFunctionType(oldFunc.getType());
    if (!functionType) {
      return signalPassFailure();
    }

    auto newFunc = FuncOp::create(oldFunc.getLoc(), oldFunc.getName(),
                                  functionType, oldFunc.getDialectAttrs());
    convertedFunctions.push_back({oldFunc, newFunc});

    // Perform the actual body conversion now.
    if (convertFunction(oldFunc, newFunc)) {
      return signalPassFailure();
    }
  }

  // Replace functions in the module.
  for (auto &pair : convertedFunctions) {
    pair.first.erase();
    module.push_back(pair.second);
  }
}
} // end anonymous namespace

std::unique_ptr<OpPassBase<ModuleOp>> createConvertFuncToMemRefPass() {
  return std::make_unique<ConvertFuncToMemRefPass>();
}

static PassRegistration<ConvertFuncToMemRefPass>
    pass("convert-func-to-memref", "Convert func from TensorType to MemRefType");
