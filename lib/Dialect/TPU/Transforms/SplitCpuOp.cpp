//===- SplitCpuOp - Implementation of Layer id assignment -----------------===//
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
// This file implements the cpu function pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TensorFile.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#define DEBUG_TYPE "split_cpu_op"

using namespace mlir;

namespace {

void collectInOutInfo(std::vector<Operation *> &cpuOpVec,
                             std::vector<Value *> &inputs,
                             std::vector<Value *> &outputs) {
  std::vector<Value *> defValue;
  for (auto op : cpuOpVec) {
    for (unsigned int i = 0; i < op->getNumOperands(); i++) {
      auto it = find(defValue.begin(), defValue.end(), op->getOperand(i));
      if (it == defValue.end()) {
        auto iter = inputs.begin();
        for (; iter != inputs.end(); ++iter) {
          if (*iter == op->getOperand(i))
            break;
        }
        if (inputs.empty() || (iter == inputs.end()))
          inputs.push_back(op->getOperand(i));
      }
    }
    for (unsigned int i = 0; i < op->getNumResults(); i++)
      defValue.push_back(op->getResult(i));
  }

  for (auto value : defValue) {
    if (value->use_empty())
        outputs.push_back(value);
    for (auto it = value->use_begin(); it != value->use_end(); ++it) {
      auto defOp = it.getUser();
      auto opIt = find(cpuOpVec.begin(), cpuOpVec.end(), defOp);
      auto valueIt = find(outputs.begin(), outputs.end(), value);
      if ((valueIt == outputs.end()) && (opIt == cpuOpVec.end())) {
        outputs.push_back(value);
      }
    }
  }
}

void collectInOutInfo(Operation * op, std::vector<Operation *> &cpuOpVec,
                             std::vector<Value *> &inputs,
                             std::vector<Value *> &outputs) {
  Block* block = op->getBlock();
  bool foundOp = false;
  for (auto iter = block->begin(); iter != block->end(); ++iter) {
    if (&(*iter) == op) {
      foundOp = true;
    }
    if ((*iter).getAttr("deleted") != nullptr ) {
      cpuOpVec.push_back(&(*iter));
    } else {
      if (foundOp)
        break;
      else
        cpuOpVec.clear();
    }
  }
  collectInOutInfo(cpuOpVec, inputs, outputs);
}

class commonPattern : public RewritePattern {
public :
  void addCpuCall(Operation *op, PatternRewriter &rewriter) const {
    std::vector<Operation *> cpuOpVec;
    std::vector<Value *> inputs;
    std::vector<Value *> outputs;
    collectInOutInfo(op, cpuOpVec, inputs, outputs);

    std::vector<mlir::Type> resType;
    for (auto output : outputs) {
      resType.push_back(output->getType());
    }

    unsigned int minLayerId = -1;
    Operation* targetOp = nullptr;
    for (auto output : outputs) {
      for (auto &use : output->getUses()) {
        Operation *useOp = use.getOwner();
        if (isa<ReturnOp>(useOp)) {
          targetOp = useOp;
          break;
        }
        auto layerId = useOp->getAttr("layer_id").cast<IntegerAttr>().getInt();
        if (layerId < minLayerId) {
          targetOp = useOp;
          minLayerId = layerId;
        }
      }
    }

    // set gaddr
    long gaddr = 0;
    for (auto cpuOp : cpuOpVec) {
      auto gaddrAttr = cpuOp->getAttr("gaddr");
      if (gaddrAttr) {
        gaddr =gaddrAttr.cast<IntegerAttr>().getInt();
      }
    }
    auto gaddrAttr = rewriter.getI64IntegerAttr(gaddr);
    auto layerIdAttr = rewriter.getI32IntegerAttr(0);
    auto callee = cpuOpVec[0]->getAttr("callee").cast<StringAttr>().getValue();
    auto callOp = rewriter.create<tpu::TG_CallOp>(op->getLoc(),
                           ArrayRef<mlir::Type>{resType}, gaddrAttr, "call",
                           layerIdAttr, callee, inputs);

    for (unsigned int i  = 0; i <  outputs.size(); i++) {
      auto outputOp = outputs[i]->getDefiningOp();
      // set the result tensor name of call op
      std::string attrName = "name";
      if (outputs.size() > 1) {
        std::string nameSuffix = std::string("_") + std::to_string(i);
        attrName = "name" + nameSuffix;
      }
      auto tensorName = outputOp->getAttr("name");
      callOp.setAttr(attrName, tensorName);
      // earse op
      auto iter = find(cpuOpVec.begin(), cpuOpVec.end(), outputOp);
      if (iter != cpuOpVec.end()) {
        cpuOpVec.erase(iter);
      }
      rewriter.replaceOp(outputOp, callOp.getResult(i));
    }
    if (cpuOpVec.size() > 1) {
      for (unsigned int i = cpuOpVec.size() - 1; i > 1; i--) {
        rewriter.eraseOp(cpuOpVec[i]);
      }
    }
    if (targetOp)
      callOp.getOperation()->moveBefore(targetOp);
  }

protected :
  commonPattern(StringRef opName, int benefit, MLIRContext *context)
      : RewritePattern(opName, benefit, context) {}
};

template<typename OpTy>
struct CpuOpPattern : public commonPattern {
  CpuOpPattern(MLIRContext *context)
      : commonPattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    if (op->getAttr("deleted") != nullptr ) {
      LLVM_DEBUG(llvm::errs() << OpTy::getOperationName() << "\n");
      std::string op_name =
          op->getAttrOfType<StringAttr>("name").getValue().str();
      LLVM_DEBUG(llvm::errs() << "tensor name: " << op_name << "\n");
      addCpuCall(op, rewriter);
      return matchSuccess();
    }
    return matchFailure();
  }
};

struct GenericCpuOpPattern : public commonPattern {
  GenericCpuOpPattern(MLIRContext *context)
      : commonPattern("tpu.generic_cpu", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    if (op->getAttr("deleted") != nullptr ) {
      std::string op_name =
          op->getAttrOfType<StringAttr>("name").getValue().str();
      LLVM_DEBUG(llvm::errs() << "tensor name: " << op_name << "\n");
      addCpuCall(op, rewriter);
      return matchSuccess();
    }
    return matchFailure();
  }
};

class GenCpuCallPass : public FunctionPass<GenCpuCallPass> {
public:
  explicit GenCpuCallPass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<CpuOpPattern<tpu::SoftmaxOp>>(context);
    patterns.insert<CpuOpPattern<tpu::DetectionOutputOp>>(context);
    patterns.insert<CpuOpPattern<tpu::QuantOp>>(context);
    patterns.insert<GenericCpuOpPattern>(context);
    applyPatternsGreedily(fn, patterns);
  }

private:
  llvm::raw_ostream &os;
};

class SplitCpuOpPass : public ModulePass<SplitCpuOpPass> {
public:
  explicit SplitCpuOpPass() : fnIndex(0){}
  void runOnModule() override;
  void createCpuFunc(std::vector<FuncOp> &cpuFuncVec,
                     std::vector<Operation *> &origCpuOpVec,
                     std::vector<Operation *> &maybeIncludedVec);
  bool isCpuOp(Operation *op);
  bool isMixedOp(Operation *op);
  int fnIndex;
};

void SplitCpuOpPass::runOnModule() {
  std::vector<FuncOp> cpuFuncVec;
  std::vector<Operation *> maybeIncludedVec;
  std::vector<Operation *> origCpuOpVec;

  BlockAndValueMapping mapper;
  bool hasCpuFunc = true;
  auto module = getModule();
  for (FuncOp fn : module.getOps<FuncOp>()) {
    fn.walk([&](Operation *op) {
      if (isCpuOp(op) || ((!hasCpuFunc) && isMixedOp(op))) {
        origCpuOpVec.push_back(op);
        hasCpuFunc = false;
      } else if (!hasCpuFunc){
        createCpuFunc(cpuFuncVec, origCpuOpVec, maybeIncludedVec);
        hasCpuFunc = true;
        origCpuOpVec.clear();
        fnIndex++;
      }
      if (hasCpuFunc) {
        if (isMixedOp(op)) {
          maybeIncludedVec.push_back(op);
        } else {
          maybeIncludedVec.clear();
        }
      }
    });
  }
  for (FuncOp fn : cpuFuncVec) {
    module.push_back(fn);
  }
}

// Create a new cpu function directly through the copy cpu instructions.
// Deleted copied instuctions in add-cpu-call pass.
void SplitCpuOpPass::createCpuFunc(std::vector<FuncOp> &cpuFuncVec,
                                   std::vector<Operation *> &origCpuOpVec,
                                   std::vector<Operation *> &maybeIncludedVec) {
  std::vector<mlir::Type> argType;
  std::vector<mlir::Type> resType;
  std::vector<Value* > inputs;
  std::vector<Value *> outputs;
  std::vector<Operation *> cpuOpVec;

  auto *context = &getContext();
  Builder builder(context);
  auto cpuFuncName = std::string("cpu_func") + std::string("_") +
                     std::to_string(fnIndex);

  BlockAndValueMapping mapper;
  for (auto opIter = maybeIncludedVec.rbegin(),
       opRend = maybeIncludedVec.rend(); opIter != opRend; ++opIter) {
    origCpuOpVec.insert(origCpuOpVec.begin(), *opIter);
  }
  for (auto &op : origCpuOpVec) {
    auto newOp = op->cloneWithoutRegions(mapper);
    cpuOpVec.push_back(newOp);
  }

  collectInOutInfo(cpuOpVec, inputs, outputs);

  for (auto input : inputs) {
    argType.push_back(input->getType());
  }

  for (auto output : outputs) {
    resType.push_back(output->getType());
  }

  auto fnType = builder.getFunctionType(llvm::ArrayRef<mlir::Type>{argType},
                                        llvm::ArrayRef<mlir::Type>{resType});
  auto cpuFn = FuncOp::create(builder.getUnknownLoc(), cpuFuncName, fnType);
  auto block = cpuFn.addEntryBlock();

  for (unsigned int i = 0; i < inputs.size(); i++) {
    // set tensor name the arg of cpu function
    auto defOp = inputs[i]->getDefiningOp();
    auto tensorName = defOp->getAttr("name");
    cpuFn.setArgAttr(i, "tpu.tensor_name", tensorName);

    // replaced the use with input value
    auto arg = block->getArgument(i);
    for (auto cpuOp : cpuOpVec) {
      for (unsigned int index = 0; index < cpuOp->getNumOperands(); index++) {
        if (cpuOp->getOperand(index) == inputs[i]) {
          cpuOp->setOperand(index, arg);
        }
      }
    }
  }

  for (auto op : cpuOpVec) {
    block->push_back(op);
  }
  OpBuilder(block).create<ReturnOp>(builder.getUnknownLoc(),
                                    llvm::ArrayRef<mlir::Value *>{outputs});
  cpuFuncVec.push_back(cpuFn);

  // add callee and deleted attribute
  for (unsigned int i = 0; i < origCpuOpVec.size(); i++) {
    if (0 == i) {
      StringRef fnSymbolName = SymbolTable::getSymbolAttrName();
      auto fnName = cpuFn.getAttr(fnSymbolName);
      origCpuOpVec[i]->setAttr("callee", fnName);
      origCpuOpVec[i]->setAttr("deleted", builder.getBoolAttr(true));
    } else {
      origCpuOpVec[i]->setAttr("deleted", builder.getBoolAttr(true));
    }
  }
}

bool SplitCpuOpPass::isCpuOp(Operation *op) {
  if (isa<tpu::DetectionOutputOp>(op) ||
      isa<tpu::SoftmaxOp>(op) ||
      isa<tpu::QuantOp>(op) ||
      isa<tpu::GenericCpuOp>(op)) {
    return true;
  } else
    return false;
}

bool SplitCpuOpPass::isMixedOp(Operation *op) {
 if (isa<tpu::ReshapeOp>(op)) {
    return true;
  } else
    return false;
}
} // namespace

std::unique_ptr<OpPassBase<ModuleOp>> mlir::createSplitCpuOpPass() {
  return std::make_unique<SplitCpuOpPass>();
}

static PassRegistration<SplitCpuOpPass>
    pass_1("split-cpu-op",
           "split cpu and tpu op");

std::unique_ptr<OpPassBase<FuncOp>> mlir::createAddCpuCallPass() {
  return std::make_unique<GenCpuCallPass>();
}

static PassRegistration<GenCpuCallPass>
    pass_2("gen-cpu-call",
           "generate call operation to call cpu function");
