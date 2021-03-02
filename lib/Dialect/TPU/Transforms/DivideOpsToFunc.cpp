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

#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
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
  std::vector<Value> inputs;
  std::vector<Value> outputs;
};

static bool isReshapeOpConnectToTpuOp(Operation *op) {
  for (auto &use : op->getResult(0).getUses()) {
    auto nextOp = use.getOwner();
    if (llvm::dyn_cast<tpu::TpuTGOpCodegenInterface>(nextOp) ||
        llvm::dyn_cast<tpu::TpuTLOpCodegenInterface>(nextOp)) {
      return true;
    }
  }
  return false;
}

class DivideOpsToFuncPass : public mlir::PassWrapper<DivideOpsToFuncPass, OperationPass<ModuleOp>> {
public:
  void runOnOperation() override {
    auto module = getOperation();
    // auto *context = &getContext();

    std::vector<SubFunction *> subFuncs;
    SubFunction *sf = nullptr;

    for (auto fn : module.getOps<FuncOp>()) {
      fn.walk([&](Operation *op) {
        if (op->getName().getDialect()->getNamespace() != "tpu"
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
          if (!sf) {
            sf = new SubFunction(isReshapeOpConnectToTpuOp(op));
          }

          sf->ops.push_back(op);
        } else if (llvm::dyn_cast<tpu::TpuTGOpCodegenInterface>(op) ||
                   llvm::dyn_cast<tpu::TpuTLOpCodegenInterface>(op)) {
          if (!sf) {
            sf = new SubFunction(true);
          }
          // if op is tpu op, push it to group.
          for (unsigned i = 0; i < op->getNumOperands(); i++) {
            auto opd = op->getOperand(i).getDefiningOp();
            if (isa<tpu::LoadWeightOp>(opd) || isa<tpu::NoneOp>(opd)) {
              sf->ops.push_back(opd);
            }
          }
          sf->ops.push_back(op);
        } else if (isa<tpu::GenericCpuOp>(op)) {
          if (sf) {
            // if op is not tpu op, skip it and create tpu
            // function with ops in group.
            if (sf->tpu) {
              addSubFunction(sf);
              subFuncs.push_back(sf);
              sf = new SubFunction(false);
            }
            // if op is cpu op, push it to group.
            for (unsigned i = 0; i < op->getNumOperands(); i++) {
              auto opd = op->getOperand(i).getDefiningOp();
              if (isa<tpu::LoadWeightOp>(opd) || isa<tpu::NoneOp>(opd)) {
                sf->ops.push_back(opd);
              }
            }
            sf->ops.push_back(op);
            addSubFunction(sf);
            subFuncs.push_back(sf);
            sf = nullptr;
          } else {
            sf = new SubFunction(false);
            // if op is cpu op, push it to group.
            for (unsigned i = 0; i < op->getNumOperands(); i++) {
              auto opd = op->getOperand(i).getDefiningOp();
              if (isa<tpu::LoadWeightOp>(opd) || isa<tpu::NoneOp>(opd)) {
                sf->ops.push_back(opd);
              }
            }
            sf->ops.push_back(op);
            addSubFunction(sf);
            subFuncs.push_back(sf);
            sf = nullptr;
          }
        }
      });
    }

    // for allOp are tpuOp case
    if (sf != nullptr) {
      addSubFunction(sf);
      subFuncs.push_back(sf);
    }

    for (auto sf : subFuncs) {
      module.push_back(sf->fnOp);
      delete sf;
    }
  }

private:
  int fnIdx_ = 0;

  void addSubFunction(SubFunction *sf) {
    // std::vector<Operation *> fnOps;
    std::vector<Value> fnInputs;
    std::vector<Value> fnOutputs;

    //for (auto op : sf->ops) {
    //  fnOps.push_back(newOp);
    //}

    // getInputsOutputs(sf->ops, sf->inputs, sf->outputs);
    //getInputsOutputs(fnOps, fnInputs, fnOutputs);
    getInputsOutputs(sf->ops, fnInputs, fnOutputs);
    std::vector<mlir::Type> argType;
    std::vector<mlir::Type> resType;
    for (auto input : fnInputs) {
      argType.push_back(input.getType());
    }
    for (auto output : fnOutputs) {
      resType.push_back(output.getType());
    }

    auto genUniqueCode = []() {
      srand(time(0));
      std::stringstream stream;
      stream << std::setfill ('0') << std::setw(sizeof(uint32_t) * 2)
             << std::hex << (uint32_t)random();
      return stream.str();
    };

    sf->fnName = sf->tpu ? std::string("tpu_") : std::string("cpu_");
    sf->fnName += std::string("subfunc") + std::to_string(fnIdx_++)
                  + "_" + genUniqueCode();

    OpBuilder builder(&getContext());
    auto fnType = builder.getFunctionType(llvm::ArrayRef<mlir::Type>{argType},
                                          llvm::ArrayRef<mlir::Type>{resType});
    sf->fnOp = FuncOp::create(builder.getUnknownLoc(), sf->fnName, fnType);
    auto block = sf->fnOp.addEntryBlock();
    builder.setInsertionPointToStart(block);

    BlockAndValueMapping mapper;
    for (int i = 0; i < (int)fnInputs.size(); i++) {
      auto arg = block->getArgument(i);
      mapper.map(fnInputs[i], arg);
    }

    for (auto op : sf->ops) {
      builder.clone(*op, mapper);
    }

    SmallVector<Value, 4> terminatorOperands;
    for (auto &val : fnOutputs) {
      terminatorOperands.push_back(mapper.lookup(val));
    }
    builder.create<ReturnOp>(builder.getUnknownLoc(), terminatorOperands);

    // add fn attribute
    for (unsigned i = 0; i < sf->ops.size(); i++) {
      sf->ops[i]->setAttr("fn", builder.getStringAttr(sf->fnName));
    }
  }

  void getInputsOutputs(std::vector<Operation *> &ops, std::vector<Value> &inputs,
                        std::vector<Value> &outputs) {
    std::vector<Value> defValue;
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
      if (value.use_empty()) {
        outputs.push_back(value);
      }
      for (auto it = value.use_begin(); it != value.use_end(); ++it) {
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

std::unique_ptr<mlir::Pass> mlir::createDivideOpsToFuncPass() {
  return std::make_unique<DivideOpsToFuncPass>();
}
