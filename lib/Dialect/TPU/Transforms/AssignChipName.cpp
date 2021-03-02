//===- AssignChipName - Implementation of Chip Name assignment --------------===//
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
// This file implements the TPU chip name pass.
//
//===----------------------------------------------------------------------===//

#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUOperationSupport.h"
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

#define DEBUG_TYPE "chip_name"

using namespace mlir;

/// set static run-chip
// TODO: enable by default for now, should set default "cv183x"
llvm::cl::opt<std::string> clRunChipType(
     "chipname",
     llvm::cl::desc("set chip type"),
     llvm::cl::init("cv183x"));

namespace {
class AssignChipNamePass : public mlir::PassWrapper<AssignChipNamePass, FunctionPass> {
public:
  explicit AssignChipNamePass() {}

  void runOnFunction() override {
    auto fn = getFunction();
    auto *context = &getContext();
    fn->setAttr("chipname", Builder(context).getStringAttr(clRunChipType));
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::createAssignChipNamePass() {
  return std::make_unique<AssignChipNamePass>();
}
