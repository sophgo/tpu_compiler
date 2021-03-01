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
#include <set>
#include "tpuc/DivideOpsToSubFunc.h"

namespace mlir {

SubFunction::SubFunction(bool tpu, std::vector<Operation *>& ops, int &idx)
  : tpu(tpu), ops(ops) {
  generateFuncName(idx++);

  std::set<Operation *> producers;
  std::set<Operation *> consumers;
  std::set<Operation *> returnOpds;
  for (auto op : ops) {
    producers.insert(op);
    for (int i = 0; i < (int)op->getNumOperands(); i++) {
      auto opd = op->getOperand(i).getDefiningOp();
      consumers.insert(opd);
    }
    for (auto &use : op->getResult(0).getUses()) {
      auto nextOp = use.getOwner();
      if (std::find(ops.begin(), ops.end(), nextOp) == ops.end()) {
        returnOpds.insert(op); // already get outputs
      }
    }
  }
  std::set_difference(consumers.begin(), consumers.end(), producers.begin(),
                      producers.end(), std::inserter(inputs, inputs.begin()));
  std::set_difference(producers.begin(), producers.end(), consumers.begin(),
                      consumers.end(), std::inserter(outputs, outputs.begin()));

  for (auto op : returnOpds) {
    if (std::find(outputs.begin(), outputs.end(), op) == outputs.end()) {
      outputs.push_back(op);
    }
  }
}

void SubFunction::generateFuncName(int idx) {
  auto genUniqueCode = []() {
    srand(time(0));
    std::stringstream stream;
    stream << std::setfill('0') << std::setw(sizeof(uint32_t) * 2) << std::hex
           << (uint32_t)random();
    return stream.str();
  };
  name = tpu ? std::string("tpu_") : std::string("cpu_");
  name += std::string("subfunc") + std::to_string(idx) + "_" + genUniqueCode();
}

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

std::vector<SubFunction *> SubFunction::divideOpsToSubFunc(FuncOp *fn) {
  int idx = 0;
  bool tpuSubFunc = true;
  std::vector<Operation *> ops;
  std::vector<SubFunction *> subFuncs;

  // Gather all InputOp to one SubFunction.
  fn->walk([&](Operation *op) {
    if (isa<tpu::InputOp>(op)) {
      ops.push_back(op);
    }
  });
  subFuncs.push_back(new SubFunction(false, ops, idx));
  ops.clear();

  fn->walk([&](Operation *op) {
    if (op->getName().getDialect().str() != "tpu" ||
        isa<tpu::LoadWeightOp>(op) ||
        isa<tpu::WeightFileOp>(op) ||
        isa<tpu::NoneOp>(op) ||
        isa<tpu::InputOp>(op) ||
        isa<ReturnOp>(op) ||
        isa<FuncOp>(op)) {
      // continue
    } else if (isa<tpu::ReshapeOp>(op)) {
      // if ReshapeOp's next node is tg/tl op, it should be treat as tpu op
      if (ops.empty()) {
        tpuSubFunc = isReshapeOpConnectToTpuOp(op);
      }
      ops.push_back(op);
    } else if (llvm::dyn_cast<tpu::TpuTGOpCodegenInterface>(op) ||
               llvm::dyn_cast<tpu::TpuTLOpCodegenInterface>(op)) {
      if (ops.empty()) {
        tpuSubFunc = true;
      }
      // if op is tpu op, push it to group.
      for (unsigned i = 0; i < op->getNumOperands(); i++) {
        auto opd = op->getOperand(i).getDefiningOp();
        if (isa<tpu::LoadWeightOp>(opd) || isa<tpu::NoneOp>(opd)) {
          ops.push_back(opd);
        }
      }
      ops.push_back(op);
    } else if (isa<tpu::GenericCpuOp>(op)) {
      if (!ops.empty()) {
        // if op is not tpu op, skip it and create tpu
        // function with ops in group.
        if (tpuSubFunc) {
          subFuncs.push_back(new SubFunction(tpuSubFunc, ops, idx));
          ops.clear();
        }
      }
      // if op is cpu op, push it to group.
      for (unsigned i = 0; i < op->getNumOperands(); i++) {
        auto opd = op->getOperand(i).getDefiningOp();
        if (isa<tpu::LoadWeightOp>(opd) || isa<tpu::NoneOp>(opd)) {
          ops.push_back(opd);
        }
      }
      ops.push_back(op);
      subFuncs.push_back(new SubFunction(false, ops, idx));
      ops.clear();
    }
  });

  // for allOp are tpuOp case
  if (!ops.empty()) {
    subFuncs.push_back(new SubFunction(tpuSubFunc, ops, idx));
    ops.clear();
  }
  return subFuncs;
}

} // namespace mlir
