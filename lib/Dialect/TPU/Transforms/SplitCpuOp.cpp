//===- SplitCpuOp - Implementation of Layer id assignment --------------===//
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
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include <iostream>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#define DEBUG_TYPE "split_cpu_op"

using namespace mlir;

namespace {
static void collectInOutInfo(std::vector<Operation *> &cpuOpVec,
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
      if (opIt == cpuOpVec.end()) {
        outputs.push_back(value);
      }
    }
  }
}

static void collectInOutInfo(Operation * op, std::vector<Operation *> &cpuOpVec,
                             std::vector<Value *> &inputs,
                             std::vector<Value *> &outputs) {
  Block* block = op->getBlock();
  cpuOpVec.push_back(op);
  bool foundOp = false;
  for (auto iter = block->begin(); iter != block->end(); ++iter) {
    if (&(*iter) == op) {
      foundOp = true;
      continue;
    }
    if (foundOp) {
      if ((*iter).getAttr("deleted") != nullptr ) {
        cpuOpVec.push_back(&(*iter));
      } else {
        break;
      }
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

    auto callOp = rewriter.create<CallOp>(op->getLoc(), op->getAttr("callee").cast<StringAttr>().getValue(),
                                          ArrayRef<mlir::Type>{resType}, inputs);

    for (unsigned int i  = 0; i <  outputs.size(); i++) {
      auto outputOp = outputs[i]->getDefiningOp();
      rewriter.replaceOp(outputOp, callOp.getResult(i));
    }

    if (cpuOpVec.size() > 1) {
      for (unsigned int i = 1; i < cpuOpVec.size(); i++) {
        rewriter.eraseOp(cpuOpVec[i]);
      }
    }
  }

protected :
  commonPattern(StringRef opName, int benefit, MLIRContext *context)
      : RewritePattern(opName, benefit, context) {}
};


struct CpuSoftmaxOpPattern : public commonPattern {
  CpuSoftmaxOpPattern(MLIRContext *context)
      : commonPattern("tpu.softmax", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto softmaxOp = cast<tpu::SoftmaxOp>(op);
    if (op->getAttr("callee") != nullptr ) {
      assert(op->getNumOperands() == 1);
      LLVM_DEBUG(llvm::errs() << softmaxOp.getOperationName() << "\n";);

      std::string op_name = softmaxOp.getAttrOfType<StringAttr>("name").getValue().str();
      LLVM_DEBUG(llvm::errs() << "Softmax Op: " << op_name << "\n";);
      addCpuCall(op, rewriter);
    }
    return matchSuccess();
  }
};

struct CpuDetectionOutputOpPattern : public commonPattern {
  CpuDetectionOutputOpPattern(MLIRContext *context)
      : commonPattern("tpu.detectionoutput", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto detectionOutputOp = cast<tpu::DetectionOutputOp>(op);
    if (op->getAttr("callee") != nullptr ) {
      assert(op->getNumOperands() == 3);
      LLVM_DEBUG(llvm::errs() << detectionOutputOp.getOperationName() << "\n";);
      // op_name
      std::string op_name = detectionOutputOp.getAttrOfType<StringAttr>("name").getValue().str();
      LLVM_DEBUG(llvm::errs() << "DetectionOutput Op: " << op_name << "\n";);
      addCpuCall(op, rewriter);
    }
    return matchSuccess();
  }
};

class AddCpuCallPass : public FunctionPass<AddCpuCallPass> {
public:
  explicit AddCpuCallPass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    OwningRewritePatternList patterns;
    auto *context = &getContext();
    patterns.insert<CpuSoftmaxOpPattern>(context);
    patterns.insert<CpuDetectionOutputOpPattern>(context);
    applyPatternsGreedily(fn, patterns);
  }

private:
  llvm::raw_ostream &os;
};

class SplitCpuOpPass : public ModulePass<SplitCpuOpPass> {
public:
  explicit SplitCpuOpPass() : fnIndex(0){}
  void runOnModule() override;
  void createCpuFunc(std::vector<Operation *> &cpuOpVec,
                     std::vector<FuncOp> &cpuFuncVec,
                     std::vector<Operation *> &origCpuOpVec);
  bool isCpuOp(Operation *op);
  bool isMixedOp(Operation *op);
  int fnIndex;
};

void SplitCpuOpPass::runOnModule() {
  std::vector<FuncOp> cpuFuncVec;
  std::vector<Operation *> cpuOpVec;
  std::vector<Operation *> origCpuOpVec;
  std::vector<Operation *> replacedOpVec;
  bool hasCpuFunc = true;
  auto module = getModule();
  BlockAndValueMapping mapper;
  for (FuncOp fn : module.getOps<FuncOp>()) {
    fn.walk([&](Operation *op) {
      if (isCpuOp(op) || ((!hasCpuFunc) && isMixedOp(op))) {
        auto newOp = op->cloneWithoutRegions(mapper);
        cpuOpVec.push_back(newOp);
        origCpuOpVec.push_back(op);
        hasCpuFunc = false;
      } else if (!hasCpuFunc){
        createCpuFunc(cpuOpVec, cpuFuncVec, origCpuOpVec);
        hasCpuFunc = true;
        replacedOpVec.push_back(origCpuOpVec[0]);
        cpuOpVec.clear();
        origCpuOpVec.clear();
        fnIndex++;
      }
    });
  }
  for (FuncOp fn : cpuFuncVec) {
    module.push_back(fn);
  }
}


// Create a new cpu function directly through the copy cpu instructions.
// Deleted copied instuctions in add-cpu-call pass.
void SplitCpuOpPass::createCpuFunc(std::vector<Operation *> &cpuOpVec,
                                   std::vector<FuncOp> &cpuFuncVec,
                                   std::vector<Operation *> &origCpuOpVec) {
  auto *context = &getContext();
  Builder builder(context);

  auto cpuFuncName = std::string("cpu_func") + std::string("_") + std::to_string(fnIndex);

  std::vector<mlir::Type> argType;
  std::vector<mlir::Type> resType;
  std::vector<Value* > inputs;
  std::vector<Value *> outputs;
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

  // replaced the use with input value
  for (unsigned int i = 0; i < inputs.size(); i++) {
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
    } else {
      origCpuOpVec[i]->setAttr("deleted", builder.getBoolAttr(true));
    }
  }
}

bool SplitCpuOpPass::isCpuOp(Operation *op) {
  if (isa<tpu::DetectionOutputOp>(op) ||
      isa<tpu::SoftmaxOp>(op)) {
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
  return std::make_unique<AddCpuCallPass>();
}

static PassRegistration<AddCpuCallPass>
    pass_2("add-cpu-call",
           "Add call operation to call cpu function");
