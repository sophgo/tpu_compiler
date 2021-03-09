//===- AssignNeuronAddress.cpp - assigned neuron address ------------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file assined neuron address
//
//===----------------------------------------------------------------------===//

#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUCompressUtil.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/DivideOpsToSubFunc.h"
#include "tpuc/GmemAllocator.hpp"
#include "tpuc/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tpuc/Support/TensorFile.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"

#define DEBUG_TYPE "assign_neuron_address"

using namespace mlir;

namespace {

static llvm::cl::opt<size_t>
    clNeuronAlignment("tpu-neuron-address-align",
                      llvm::cl::desc("Specify the alignment for neuron"),
                      llvm::cl::init(16));

static llvm::cl::opt<bool>
    clNeuronReuse("tpu-neuron-memory-reuse",
                  llvm::cl::desc("Reuse neuron's memory"),
                  llvm::cl::init(false));

static llvm::cl::opt<std::string> clNeuronMapFilename(
    "tpu-neuron-map-filename",
    llvm::cl::desc("record neuron offset with its name into a csv map file"),
    llvm::cl::init("-"));

static int32_t getOpDtypeSize(Operation *op) {
  int32_t dsize = 1;
  auto elementType =
      op->getResult(0).getType().template cast<TensorType>().getElementType();
  if (elementType.isF32()) {
    dsize = sizeof(float);
  } else if (elementType.isInteger(8)) {
    dsize = sizeof(int8_t);
  } else if (elementType.isBF16()) {
    dsize = sizeof(uint16_t);
  } else {
    llvm_unreachable("unsupported data type");
  }
  return dsize;
}

template <typename OpTy>
struct TlLgLoadNeuronAddressPattern : public RewritePattern {
  TlLgLoadNeuronAddressPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto castOp = cast<OpTy>(op);
    if (castOp.gaddr().hasValue()) {
      return failure();
    }

    auto curPos = getPreviousOpAddress(castOp);
    auto offset = (int)castOp.offset().getValue();
    setOpAddress(op, curPos + offset);
    return success();
  }
};

static bool isSliceOpSkip(Operation *op) {
  if (auto sliceOp = llvm::dyn_cast<tpu::TG_INT8_SliceOp>(op)) {
    auto type = sliceOp.getResult().getType().template cast<TensorType>();
    int axis = sliceOp.axis();
    std::vector<int64_t> shape = type.getShape();
    for (int index = 0; index < axis; index++) {
      if (shape[index] != 1) {
        return false;
      }
    }
  } else if (auto sliceOp = llvm::dyn_cast<tpu::TG_BF16_SliceOp>(op)) {
    auto type = sliceOp.getResult().getType().template cast<TensorType>();
    int axis = sliceOp.axis();
    std::vector<int64_t> shape = type.getShape();
    for (int index = 0; index < axis; index++) {
      if (shape[index] != 1) {
        return false;
      }
    }
  } else {
    return false;
  }
  return true;
}

template <typename OpTy>
struct TgSliceAddressPattern : public RewritePattern {
  TgSliceAddressPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto castOp = cast<OpTy>(op);
    if (castOp.gaddr().hasValue()) {
      return failure();
    }
    int axis = castOp.axis();
    int offset = castOp.offset();

    std::vector<int64_t> shape = getTensorShape(castOp.input());

    if (false == isSliceOpSkip(op)) {
      return failure();
    }
    auto opd = op->getOperand(0).getDefiningOp();
    if (auto formerOp = llvm::dyn_cast<OpTy>(opd)) {
      if (formerOp.gaddr().hasValue() == false) {
        return failure();
      }
    }

    int32_t dsize = getOpDtypeSize(op);
    int64_t isz = 1;
    for (unsigned i = axis + 1; i < shape.size(); i++) {
      isz *= shape[i];
    }
    size_t offset_bytes = offset * isz * dsize;

    if (castOp.load_compr_act().hasValue() && castOp.load_compr_act().getValue()) {
      auto c_step = castOp.load_compr_act_param().getValue().c_step().getInt();
      auto step_size = castOp.load_compr_act_param().getValue().step_size().getInt();
      auto h = shape[2];
      offset_bytes = (offset / c_step) * h * step_size;
    }

    auto curPos = getPreviousOpAddress(castOp);
    setOpAddress(op, curPos + offset_bytes);

    if (opd->getAttr("buffer_reused")) {
      castOp->setAttr("buffer_reused", rewriter.getBoolAttr(true));
    }
    return success();
  }
};

struct TgConcatNAddressPattern : public RewritePattern {
  TgConcatNAddressPattern(MLIRContext *context)
      : RewritePattern("tpu.tg_concat_n", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto castOp = cast<tpu::TG_ConcatNOp>(op);
    auto concatGAddr = castOp.getGAddr();
    int64_t offset = 0;
    for (auto opd : op->getOperands()) {
      auto defOp = opd.getDefiningOp();
      std::vector<int64_t> shape = getTensorShape(defOp->getResult(0));
      int32_t dsize = getOpDtypeSize(defOp);
      int64_t isz = 1;
      for (unsigned i = 0; i < shape.size(); i++) {
        isz *= shape[i];
      }
      auto updatedGAddr = concatGAddr + offset;
      setOpAddress(defOp, updatedGAddr);
      offset += isz * dsize;
    }
    return success();
  }
};

template <typename OpTy>
struct TlLgStoreAddressNeuronPattern : public RewritePattern {
  TlLgStoreAddressNeuronPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto castOp = cast<OpTy>(op);
    if (castOp.gaddr().hasValue()) {
      return failure();
    }

    auto nextOp = getNextOp(op);
    auto curPos = getOpAddress(nextOp);
    auto offset = (int)castOp.offset().getValue();
    setOpAddress(op, curPos + offset);
    return success();
  }
};

static bool isInPlaceOp(Operation *op) {
  if (isSliceOpSkip(op)) {
    return true;
  } else if (isa<tpu::ReshapeOp>(op)) {
    return true;
  }
  return false;
}

static uint32_t getOpLine(Operation *op) {
  auto loc = op->getLoc().cast<FileLineColLoc>();
  return loc.getLine();
}

static void findInPlaceOpMaxUsePosition(Operation *op, uint32_t &maxPosition) {
  for (auto &use : op->getResult(0).getUses()) {
    Operation *next = use.getOwner();
    if (isInPlaceOp(next)) {
      findInPlaceOpMaxUsePosition(next, maxPosition);
    } else {
      uint32_t curPosition = getOpLine(next) + 1;
      if (maxPosition < curPosition) {
        maxPosition = curPosition;
      }
    }
  }
}


static void
updateLiveRangeOfOps(FuncOp &fn, std::vector<Operation *> &chosenOps,
                     std::map<Operation *, std::vector<uint32_t>> &liveRange) {

  auto updateOperandsLiveRange = [&](Operation *op, uint32_t endPosition) {
    for (uint32_t i = 0; i < op->getNumOperands(); i++) {
      auto opd = op->getOperand(i).getDefiningOp();
      if (liveRange.find(opd) != liveRange.end()) {
        if (liveRange[opd][1] == 0xFFFFFFFF || liveRange[opd][1] < endPosition) {
          liveRange[opd][1] = endPosition;
        }
      }
    }
  };

  fn.walk([&](Operation *op) {
    uint32_t endPosition = getOpLine(op) + 1;
    if (isa<tpu::TL_LW_Conv2DOp>(op) || isa<tpu::TL_ScaleOp>(op) ||
        isa<tpu::TL_EltwiseAddOp>(op) || isa<tpu::TL_EltwiseMulOp>(op) ||
        isa<tpu::TL_PoolAvg2DOp>(op) || isa<tpu::TL_LutOp>(op) ||
        isa<tpu::TL_PixelShuffleOp>(op)) {
      bool store = op->getAttr("tl_store_flag").cast<BoolAttr>().getValue();
      if (store) {
        chosenOps.push_back(op);
        liveRange[op] = {getOpLine(op), 0xFFFFFFFF};
      }
      updateOperandsLiveRange(op, endPosition);
    } else if (isInPlaceOp(op)) {
      uint32_t maxPosition = getOpLine(op) + 1;
      findInPlaceOpMaxUsePosition(op, maxPosition);
      updateOperandsLiveRange(op, maxPosition);
    } else if (isa<tpu::TL_LG_StoreOp>(op)) {
      auto joinOp = getNextOp(op);
      if (liveRange.find(joinOp) == liveRange.end()) {
        chosenOps.push_back(joinOp);
        liveRange[joinOp] = {getOpLine(op), 0xFFFFFFFF};
      }
      updateOperandsLiveRange(op, endPosition);
    } else if (isa<tpu::InputOp>(op)) {
      auto nextOp = getNextOp(op);
      if (!nextOp ||
          (!isa<tpu::GenericCpuOp>(nextOp) &&
           !isa<tpu::ReshapeOp>(nextOp))) {
        chosenOps.push_back(op);
        liveRange[op] = {getOpLine(op), 0xFFFFFFFF};
        updateOperandsLiveRange(op, endPosition);
      }
    } else if (isa<tpu::GenericCpuOp>(op) ||
               llvm::dyn_cast<tpu::TpuTGOpCodegenInterface>(op)) {
      chosenOps.push_back(op);
      liveRange[op] = {getOpLine(op), 0xFFFFFFFF};
      updateOperandsLiveRange(op, endPosition);
    } else {
      updateOperandsLiveRange(op, endPosition);
    }
  });
}

class AssignNeuronAddressPass : public mlir::PassWrapper<AssignNeuronAddressPass, FunctionPass> {
public:
  explicit AssignNeuronAddressPass() {}

  void runOnFunction() override {
    auto fn = getFunction();
    auto *context = &getContext();
    // create a map file
    std::unique_ptr<llvm::ToolOutputFile> neuronMapFile = nullptr;
    if (clNeuronMapFilename != "-") {
      std::string errorMessage;
      neuronMapFile = openOutputFile(clNeuronMapFilename, &errorMessage);
      if (!neuronMapFile) {
        llvm::errs() << errorMessage << "\n";
        exit(1);
      }
    }

    // remove gaddr attribute of all op
    fn.walk([&](Operation *op) {
      if (llvm::dyn_cast<tpu::TpuOpCommonInterface>(op)) {
        op->removeAttr(Identifier::get("gaddr", context));
      }
    });

    /*
     * find all ops that needed to be assigned,
     * and update live ranges of all ops in the function.
     */
    std::vector<Operation *> chosenOps;
    std::map<Operation *, std::vector<uint32_t>> liveRange;
    updateLiveRangeOfOps(fn, chosenOps, liveRange);

    // devide chosen ops to defferent memory regions.
    std::vector<std::vector<Operation *>> opsInSharedMemoryRegions;
    std::vector<Operation *> opsInPrivateMemoryRegionForTPU;
    std::vector<Operation *> opsInPrivateMemoryRegionForCpu;
    std::vector<Operation *> opsInIOMemoryRegion;

    auto subFunctions = SubFunction::divideOpsToSubFunc(&fn);
    for (auto subFn : subFunctions) {
      if (subFn->tpu) {
        std::vector<Operation *> targetOps;
        for (auto op : subFn->ops) {
          if (isOpInVector(op, chosenOps)) {
            if (isOpBelongToSharedMemoryRegion(op, subFn->outputs)) {
              targetOps.push_back(op);
            } else if (isOpBelongToIOMemoryRegion(opsInIOMemoryRegion, op)) {
              opsInIOMemoryRegion.push_back(op);
            } else if (isOpBelongToTPUPrivateMemoryRegion(op)) {
              opsInPrivateMemoryRegionForTPU.push_back(op);
            } else {
              opsInPrivateMemoryRegionForCpu.push_back(op);
            }
          }
        }
        opsInSharedMemoryRegions.push_back(targetOps);
      } else {
        for (auto op : subFn->ops) {
          if (isOpInVector(op, chosenOps)) {
            if (isOpBelongToIOMemoryRegion(opsInIOMemoryRegion, op)) {
              opsInIOMemoryRegion.push_back(op);
            } else if (isOpBelongToTPUPrivateMemoryRegion(op)) {
              opsInPrivateMemoryRegionForTPU.push_back(op);
            } else {
              opsInPrivateMemoryRegionForCpu.push_back(op);
            }
          }
        }
      }
    }

    // assign gaddr for ops in defferent memory regions.
    int64_t sharedGmemOffset = 0;
    int64_t sharedGmemSize = 0;
    std::map<Operation *, int64_t> gaddrMap;
    // 1. Assign gaddr for ops in shared regions.
    for (auto &targetOps : opsInSharedMemoryRegions) {
      GmemAllocator allocator(gaddrMap, clNeuronAlignment);
      auto gmemUsed = allocator.assignGaddr(targetOps, liveRange,
                                            clNeuronReuse, sharedGmemOffset);
      if (!clNeuronReuse) {
        sharedGmemOffset += gmemUsed;
      }
      if (sharedGmemSize < sharedGmemOffset + gmemUsed) {
        sharedGmemSize = sharedGmemOffset + gmemUsed;
      }
    }

    // To solve concat opt when axis = 0, it need the operand should be
    // continuous global memory.
    auto assignedGaddr = sharedGmemSize;
    for (auto &targetOps : opsInSharedMemoryRegions) {
      for (auto &op : targetOps) {
        auto size  = GmemAllocator::assignSpecifiedGmemToOp(op, gaddrMap,
                                                       assignedGaddr,
                                                       clNeuronAlignment);
        assignedGaddr += size;
      }
    }
    sharedGmemSize = assignedGaddr ;

    int64_t baseGaddr = (((uint64_t)2) << 40);
    int64_t privateGmemSize = 0;
    // 2. Assign gaddr for ops in CPU private region.
    if (!opsInPrivateMemoryRegionForCpu.empty()) {
      GmemAllocator allocator(gaddrMap, clNeuronAlignment);
      privateGmemSize = allocator.assignGaddr(opsInPrivateMemoryRegionForCpu, liveRange,
                                              clNeuronReuse, baseGaddr);
    }
    // 3. Assign gaddr for ops in TPU private region.
    if (!opsInPrivateMemoryRegionForTPU.empty()) {
      GmemAllocator allocator(gaddrMap, clNeuronAlignment);
      privateGmemSize += allocator.assignGaddr(opsInPrivateMemoryRegionForTPU, liveRange,
                                               clNeuronReuse, baseGaddr + privateGmemSize);
    }
    // 4. Assign gaddr for ops in IO memory regin.
    for (int i = 0; i < (int)opsInIOMemoryRegion.size(); ++i) {
      gaddrMap[opsInIOMemoryRegion[i]] = (((uint64_t)3 + i) << 40);
    }

    fn->setAttr("private_gmem", Builder(context).getI64IntegerAttr(privateGmemSize));
    fn->setAttr("shared_gmem", Builder(context).getI64IntegerAttr(sharedGmemSize));

    std::set<Operation *> gmemReusedSet;
    GmemAllocator::markGmemReusedOp(chosenOps, gaddrMap, gmemReusedSet, clNeuronAlignment);

    fn.walk([&](Operation *op) {
      if (gaddrMap.find(op) != gaddrMap.end()) {
        if (!isa<tpu::ReshapeOp>(op)) {
          setOpAddress(op, gaddrMap[op]);
        }
        if (gmemReusedSet.find(op) != gmemReusedSet.end()) {
          auto castOp = llvm::dyn_cast<tpu::TpuOpCommonInterface>(op);
          castOp->setAttr("buffer_reused", Builder(context).getBoolAttr(true));
        }
        if (neuronMapFile) {
          auto dsize = getOpDtypeSize(op);
          std::string dtype = "int8";
          if (dsize == 4) {
            dtype = "fp32";
          } else if (dsize == 2) {
            dtype = "int16";
          }
          std::vector<int64_t> shape =
              op->getResult(0).getType().cast<TensorType>().getShape();
          while (shape.size() < 4) {
            shape.insert(shape.begin(), 1);
          }
          auto &os = neuronMapFile->os();
          os << mlir::getOpName(op) << "," << llvm::format_hex(gaddrMap[op], 10) << ","
             << dtype;
          for (unsigned i = 0; i < shape.size(); ++i)
            os << "," << shape[i];
          os << "\n";
        }
      }
    });

    OwningRewritePatternList patterns;
    patterns.insert<TgSliceAddressPattern<tpu::TG_INT8_SliceOp>,
                    TgSliceAddressPattern<tpu::TG_BF16_SliceOp>,
                    TgConcatNAddressPattern>(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));
    patterns.clear();
    patterns.insert<TlLgStoreAddressNeuronPattern<tpu::TL_LG_StoreOp>,
                    TlLgLoadNeuronAddressPattern<tpu::TL_LG_LoadNeuronOp>>(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));

    if (neuronMapFile) {
      neuronMapFile->keep();
    }

    for (auto &it : subFunctions) {
      delete(it);
    }
    subFunctions.clear();
    subFunctions.shrink_to_fit();

  }

private:
  bool isOpInVector(Operation *op, std::vector<Operation *> &ops) {
    for (auto candidate : ops) {
      if (candidate == op)
        return true;
    }
    return false;
  }

  bool isOpBelongToSharedMemoryRegion(Operation *op, std::vector<Operation *> &outputs) {
    if (isOpInVector(op, outputs)) {
      return false;
    }
    for (auto &use : op->getResult(0).getUses()) {
      Operation *next = use.getOwner();
      if (isInPlaceOp(next) && isOpInVector(next, outputs)) {
        return false;
      }
    }
    return true;
  }

  bool isOpBelongToTPUPrivateMemoryRegion(Operation *op) {
    if (!isa<tpu::GenericCpuOp>(op) && !isa<tpu::InputOp>(op)) {
      return true;
    }
    for (auto &use : op->getResult(0).getUses()) {
      Operation *next = use.getOwner();
      if (isa<tpu::GenericCpuOp>(next) || isa<ReturnOp>(next)) {
        return false;
      }
    }
    return true;
  }

  bool isOpBelongToIOMemoryRegion(std::vector<Operation *> &ioRegion, Operation *op) {
    // Warning, IO memory region can only has capacity to store 5 ops.
    if (ioRegion.size() >= 5) {
      return false;
    }
    if (isa<tpu::InputOp>(op)) {
      for (auto &use : op->getResult(0).getUses()) {
        Operation *next = use.getOwner();
        if (!isa<tpu::GenericCpuOp>(next)) {
          return true;
        }
      }
    } else if (!isa<tpu::GenericCpuOp>(op)) {
      auto next = getNextOp(op);
      if (next && isa<ReturnOp>(next)) {
        return true;
      }
    }
    return false;
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::createAssignNeuronAddressPass() {
  return std::make_unique<AssignNeuronAddressPass>();
}
