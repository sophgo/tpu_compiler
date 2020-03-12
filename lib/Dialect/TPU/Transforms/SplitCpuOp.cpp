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
using namespace mlir;

namespace {

struct CpuSoftmaxOpPattern : public RewritePattern {
  CpuSoftmaxOpPattern(MLIRContext *context)
      : RewritePattern("tpu.softmax", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto softmaxOp = cast<tpu::SoftmaxOp>(op);
    if (op->getAttr("callee") != nullptr ) {
      assert(op->getNumOperands() == 1);
      llvm::errs() << softmaxOp.getOperationName() << "\n";

      std::string op_name = softmaxOp.getAttrOfType<StringAttr>("name").getValue().str();
      llvm::errs() << "Softmax Op: " << op_name << "\n";

      auto callOp = rewriter.create<CallOp>(op->getLoc(), op->getAttr("callee").cast<StringAttr>().getValue(),
                             ArrayRef<mlir::Type>{op->getResult(0)->getType()}, op->getOperands());
      rewriter.replaceOp(op, callOp.getResults());
    }
    return matchSuccess();
  }
};

struct CpuDetectionOutputOpPattern : public RewritePattern {
  CpuDetectionOutputOpPattern(MLIRContext *context)
      : RewritePattern("tpu.detectionoutput", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto detectionOutputOp = cast<tpu::DetectionOutputOp>(op);
    if (op->getAttr("callee") != nullptr ) {
      assert(op->getNumOperands() == 3);
      llvm::errs() << detectionOutputOp.getOperationName() << "\n";
      // op_name
      std::string op_name = detectionOutputOp.getAttrOfType<StringAttr>("name").getValue().str();
      llvm::errs() << "DetectionOutput Op: " << op_name << "\n";

      auto callOp = rewriter.create<CallOp>(op->getLoc(), op->getAttr("callee").cast<StringAttr>().getValue(),
                             ArrayRef<mlir::Type>{op->getResult(0)->getType()}, op->getOperands());
      rewriter.replaceOp(op, callOp.getResults());
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
  explicit SplitCpuOpPass() {}

  void runOnModule() override {
    auto module = getModule();
    auto *context = &getContext();
    Builder builder(context);

    std::vector<FuncOp> cpuFuncVec;
    std::vector<Operation *> cpuOpVec;
    std::vector<Operation *> origCpuOpVec;
    std::vector<Operation *> replacedOpVec;
    uint32_t index = 0;
    bool hasCpuFunc = true;

    for (FuncOp fn : module.getOps<FuncOp>()) {
      fn.walk([&](Operation *op) {
        if (isCpuOp(op)) {
          auto newOp = op->cloneWithoutRegions();
          cpuOpVec.push_back(newOp);
          origCpuOpVec.push_back(op);
          hasCpuFunc = false;
        } else if (!hasCpuFunc){
          createCpuFunc(builder, cpuOpVec, cpuFuncVec, index);
          index++;
          hasCpuFunc = true;
          replacedOpVec.push_back(origCpuOpVec[0]);
          cpuOpVec.clear();
          origCpuOpVec.clear();
        }
      });
    }
    for (FuncOp fn : cpuFuncVec) {
      module.push_back(fn);
    }

    // add callee attribute
    for (unsigned int i = 0; i < cpuFuncVec.size(); i++) {
      auto fn = cpuFuncVec[i];
      StringRef fnSymbolName = SymbolTable::getSymbolAttrName();
      auto fnName = fn.getAttr(fnSymbolName);
      replacedOpVec[i]->setAttr("callee", fnName);
    }
  }

  // Create a new cpu function directly through the copy cpu instructions.
  // Deleted copied instuctions in add-cpu-call pass.
  void createCpuFunc(Builder &builder, std::vector<Operation *> &cpuOpVec,
                     std::vector<FuncOp> &cpuFuncVec, int index) {
    auto firstOp = cpuOpVec[0];
    auto lastOp = cpuOpVec[cpuOpVec.size() - 1];
    auto cpuFuncName = std::string("cpu_func") + std::string("_") + std::to_string(index);

    Value* resValue = lastOp->getResult(0);
    auto resType = resValue->getType();
    std::vector<mlir::Type> args;
    for (auto input : firstOp->getOperands()) {
      args.push_back(input->getType());
    }
    auto fnType = builder.getFunctionType(llvm::ArrayRef<mlir::Type>{args},
                                           llvm::ArrayRef<mlir::Type>{resType});
    auto cpuFn = FuncOp::create(builder.getUnknownLoc(), cpuFuncName, fnType);
    auto block = cpuFn.addEntryBlock();

    for (unsigned int i = 0; i < args.size(); i++) {
      auto arg = block->getArgument(i);
      firstOp->setOperand(i, arg);
    }

    for (auto op : cpuOpVec) {
      block->push_back(op);
    }
    OpBuilder(block).create<ReturnOp>(builder.getUnknownLoc(),
                                      llvm::ArrayRef<mlir::Value *>{resValue});
    cpuFuncVec.push_back(cpuFn);
  }

  bool isCpuOp(Operation *op) {
    if (isa<tpu::DetectionOutputOp>(op) ||
        isa<tpu::SoftmaxOp>(op)) {
      return true;
    } else
      return false;
  }
};

} // namespace

std::unique_ptr<OpPassBase<ModuleOp>> mlir::createSplitCpuOpPass() {
  return std::make_unique<SplitCpuOpPass>();
}

static PassRegistration<SplitCpuOpPass>
    pass_1("split-cpuop",
           "split cpu and tpu op");

std::unique_ptr<OpPassBase<FuncOp>> mlir::createAddCpuCallPass() {
  return std::make_unique<AddCpuCallPass>();
}

static PassRegistration<AddCpuCallPass>
    pass_2("add-cpu-call",
           "Add call operation to call cpu function");
