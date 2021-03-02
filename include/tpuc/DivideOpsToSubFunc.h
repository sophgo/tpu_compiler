//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
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
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef DIVIDE_OPS_TO_SUBFUNC_H_
#define DIVIDE_OPS_TO_SUBFUNC_H_

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

namespace mlir {

class SubFunction {
public:
  bool tpu;
  std::string name;
  std::vector<Operation *> ops;
  std::vector<Operation *> inputs;
  std::vector<Operation *> outputs;

  static std::vector<SubFunction *> divideOpsToSubFunc(FuncOp *fn);

private:
  SubFunction(bool tpu, std::vector<Operation *>& ops, int &idx);
  void generateFuncName(int idx);
};


} // namespace mlir

#endif // DIVIDE_OPS_TO_SUBFUNC_H_
