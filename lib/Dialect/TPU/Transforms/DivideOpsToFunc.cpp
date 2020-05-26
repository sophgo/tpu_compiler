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
#include <stdlib.h>
#include <time.h>

#define DEBUG_TYPE "devide_ops_to_func"

using namespace mlir;

namespace {

class SubFunction {
public:
  SubFunction(bool tpu) : tpu(tpu) {}
  FuncOp fnOp;
  bool tpu = false;
  std::string fnName;
  std::vector<Operation *> ops;
  std::vector<Value *> inputs;
  std::vector<Value *> outputs;
  std::map<Operation *, Operation *> mapping;
};

static bool isReshapeOpConnectToTpuOp(Operation *op) {
  for (auto &use : op->getResult(0)->getUses()) {
    auto nextOp = use.getOwner();
    if (llvm::dyn_cast<tpu::TpuTGOpCodegenInterface>(nextOp) ||
        llvm::dyn_cast<tpu::TpuTLOpCodegenInterface>(nextOp)) {
      return true;
    }
  }
  return false;
}

template<typename OpTy>
struct EliminateReshapeOpPattern : public RewritePattern {
  EliminateReshapeOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto castOp = llvm::cast<OpTy>(op);
    auto prevOp = castOp.getOperand()->getDefiningOp();
    if (!llvm::isa<tpu::GenericCpuOp>(prevOp)) {
      return matchFailure();
    }
    if (!prevOp->getResult(0)->hasOneUse()) {
      return matchFailure();
    }
    auto type = castOp.getResult()->getType();
    prevOp->getResult(0)->setType(type);

    for (auto &use : castOp.getResult()->getUses()) {
      auto nextOp = use.getOwner();
      for (uint32_t i = 0; i < nextOp->getNumOperands(); i++) {
        auto opd = nextOp->getOperand(i)->getDefiningOp();
        if (opd == op) {
          nextOp->setOperand(i, castOp.getOperand());
        }
      }
    }
    return matchSuccess();
  }
};

struct SinkCpuOPsToBottomPattern : public RewritePattern {
  SinkCpuOPsToBottomPattern(MLIRContext *context)
      : RewritePattern(ReturnOp::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto insertPoint = op;
    for (int i = (int)op->getNumOperands() - 1; i >= 0; --i) {
      auto opd = op->getOperand(i)->getDefiningOp();
      if (!isa<tpu::GenericCpuOp>(op)) {
        continue;
      }
      opd->moveBefore(insertPoint);
      insertPoint = opd;
    }
    return matchSuccess();
  }
};

struct MoveCpuOPsToCloseConsumerPattern : public RewritePattern {
  MoveCpuOPsToCloseConsumerPattern(MLIRContext *context)
      : RewritePattern(tpu::GenericCpuOp::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto myName = [](Operation *op) {
      if (!op->getAttr("name"))
        return op->getName().getStringRef();
      return op->getAttr("name").cast<StringAttr>().getValue();
    };

    auto getLastUse = [](Operation *op) {
      auto it = op->getResult(0)->use_begin();
      auto last = it;
      while (++it != op->getResult(0)->use_end()) {
        last = it;
      }
      return last->getOwner();
    };

    LLVM_DEBUG(llvm::errs() << "MoveCpuOPs, find generic_cpu:" << myName(op) << "\n");
    auto nextOp = getLastUse(op);
    LLVM_DEBUG(llvm::errs() << "MoveCpuOPs, find use:" << myName(nextOp) << "\n");
    auto insertPoint = nextOp;
    for (int i = 0; i < (int)nextOp->getNumOperands(); i++) {
      auto opd = nextOp->getOperand(i)->getDefiningOp();
      if (opd == op) {
        if (i - 1 >= 0) {
          insertPoint = nextOp->getOperand(i - 1)->getDefiningOp();
        }
        op->moveBefore(insertPoint);
        break;
      }
    }
    return matchSuccess();
  }
};

class DivideOpsToFuncPass : public ModulePass<DivideOpsToFuncPass> {
public:
  void runOnModule() override {
    auto module = getModule();
    auto *context = &getContext();
    for (auto fn : module.getOps<FuncOp>()) {
      OwningRewritePatternList patterns;
      patterns.insert<
          EliminateReshapeOpPattern<tpu::ReshapeOp>
          >(context);
      applyPatternsGreedily(fn, patterns);
      patterns.clear();
      patterns.insert<
          SinkCpuOPsToBottomPattern
          >(context);
      applyPatternsGreedily(fn, patterns);
      patterns.clear();
      patterns.insert<
          MoveCpuOPsToCloseConsumerPattern
          >(context);
      applyPatternsGreedily(fn, patterns);
    }

    std::vector<SubFunction *> tpuFuncs;
    SubFunction *tf = nullptr;

    for (auto fn : module.getOps<FuncOp>()) {
      fn.walk([&](Operation *op) {
        if (op->getName().getDialect().str() != "tpu"
           || isa<tpu::LoadWeightOp>(op)
           || isa<tpu::WeightFileOp>(op)
           || isa<tpu::NoneOp>(op)
           || isa<tpu::InputOp>(op)
           || isa<ReturnOp>(op)
           || isa<FuncOp>(op)) {
          // continue
        }
        if (isa<tpu::ReshapeOp>(op)) {
          // if ReshapeOp's next node is tg/tl op, it should be treat as tpu op
          if (!tf) {
            tf = new SubFunction(isReshapeOpConnectToTpuOp(op));
          }
          tf->ops.push_back(op);
        } else if (llvm::dyn_cast<tpu::TpuTGOpCodegenInterface>(op) ||
                   llvm::dyn_cast<tpu::TpuTLOpCodegenInterface>(op)) {
          if (!tf) {
            tf = new SubFunction(true);
          }
          // if op is tpu op, push it to group.
          for (unsigned i = 0; i < op->getNumOperands(); i++) {
            auto opd = op->getOperand(i)->getDefiningOp();
            if (isa<tpu::LoadWeightOp>(opd) || isa<tpu::NoneOp>(opd)) {
              tf->ops.push_back(opd);
            }
          }
          tf->ops.push_back(op);
        } else if (isa<tpu::GenericCpuOp>(op)) {
          if (tf) {
            // if op is not tpu op, skip it and create tpu
            // function with ops in group.
            addSubFunction(tf);
            tpuFuncs.push_back(tf);
            tf = new SubFunction(false);
            // if op is cpu op, push it to group.
            for (unsigned i = 0; i < op->getNumOperands(); i++) {
              auto opd = op->getOperand(i)->getDefiningOp();
              if (isa<tpu::LoadWeightOp>(opd) || isa<tpu::NoneOp>(opd)) {
                tf->ops.push_back(opd);
              }
            }
            tf->ops.push_back(op);
            addSubFunction(tf);
            tpuFuncs.push_back(tf);
            tf = nullptr;
          } else {
            tf = new SubFunction(false);
            // if op is cpu op, push it to group.
            for (unsigned i = 0; i < op->getNumOperands(); i++) {
              auto opd = op->getOperand(i)->getDefiningOp();
              if (isa<tpu::LoadWeightOp>(opd) || isa<tpu::NoneOp>(opd)) {
                tf->ops.push_back(opd);
              }
            }
            tf->ops.push_back(op);
            addSubFunction(tf);
            tpuFuncs.push_back(tf);
            tf = nullptr;
          }
        }
      });
    }

    for (auto tf : tpuFuncs) {
      module.push_back(tf->fnOp);
      delete tf;
    }
  }

private:
  int fnIdx_ = 0;

  void addSubFunction(SubFunction *tf) {
    std::vector<Operation *> fnOps;
    std::vector<Value *> fnInputs;
    std::vector<Value *> fnOutputs;

    BlockAndValueMapping mapper;
    for (auto op : tf->ops) {
      auto newOp = op->cloneWithoutRegions(mapper);
      tf->mapping[newOp] = op;
      fnOps.push_back(newOp);
    }

    getInputsOutputs(tf->ops, tf->inputs, tf->outputs);
    getInputsOutputs(fnOps, fnInputs, fnOutputs);

    std::vector<mlir::Type> argType;
    std::vector<mlir::Type> resType;
    for (auto input : fnInputs) {
      argType.push_back(input->getType());
    }
    for (auto output : fnOutputs) {
      resType.push_back(output->getType());
    }

    auto genUniqueCode = []() {
      srand(time(0));
      std::stringstream stream;
      stream << std::setfill ('0') << std::setw(sizeof(uint32_t) * 2)
             << std::hex << (uint32_t)random();
      return stream.str();
    };

    tf->fnName = tf->tpu ? std::string("tpu_") : std::string("cpu_");
    tf->fnName += std::string("subfunc") + std::to_string(fnIdx_++)
                  + "_" + genUniqueCode();

    Builder builder(&getContext());
    auto fnType = builder.getFunctionType(llvm::ArrayRef<mlir::Type>{argType},
                                          llvm::ArrayRef<mlir::Type>{resType});
    tf->fnOp = FuncOp::create(builder.getUnknownLoc(), tf->fnName, fnType);
    auto block = tf->fnOp.addEntryBlock();

    // replaced the use with input value
    for (unsigned i = 0; i < fnInputs.size(); i++) {
      auto arg = block->getArgument(i);
      for (auto op : fnOps) {
        for (unsigned j = 0; j < op->getNumOperands(); j++) {
          if (op->getOperand(j) == fnInputs[i]) {
            op->setOperand(j, arg);
          }
        }
      }
    }
    for (auto op : fnOps) {
      block->push_back(op);
    }

    OpBuilder(block).create<ReturnOp>(builder.getUnknownLoc(),
                                      llvm::ArrayRef<mlir::Value *>{fnOutputs});
    // add fn attribute
    for (unsigned i = 0; i < tf->ops.size(); i++) {
      tf->ops[i]->setAttr("fn", builder.getStringAttr(tf->fnName));
    }
  }

  void getInputsOutputs(std::vector<Operation *> &ops, std::vector<Value *> &inputs,
                        std::vector<Value *> &outputs) {
    std::vector<Value *> defValue;
    for (auto op : ops) {
      for (unsigned int i = 0; i < op->getNumOperands(); i++) {
        auto it = find(defValue.begin(), defValue.end(), op->getOperand(i));
        if (it == defValue.end()) {
          auto iter = inputs.begin();
          for (; iter != inputs.end(); ++iter) {
            if (*iter == op->getOperand(i)) {
              break;
            }
          }
          if (inputs.empty() || (iter == inputs.end())) {
            inputs.push_back(op->getOperand(i));
          }
        }
      }
      for (unsigned int i = 0; i < op->getNumResults(); i++) {
        defValue.push_back(op->getResult(i));
      }
    }

    for (auto value : defValue) {
      if (value->use_empty()) {
        outputs.push_back(value);
      }
      for (auto it = value->use_begin(); it != value->use_end(); ++it) {
        auto defOp = it.getUser();
        auto opIt = find(ops.begin(), ops.end(), defOp);
        auto valueIt = find(outputs.begin(), outputs.end(), value);
        if ((valueIt == outputs.end()) && (opIt == ops.end())) {
          outputs.push_back(value);
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<OpPassBase<ModuleOp>> createDivideOpsToFuncPass() {
  return std::make_unique<DivideOpsToFuncPass>();
}

static PassRegistration<DivideOpsToFuncPass> pass("divide-ops-to-func",
                                                  "divide ops into functions");
