//===- KernelOutlining.cpp - Implementation of TPU outling ---------===//
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
// This file implements the TPU dialect outlining pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/StandardOps/Ops.h"

using namespace mlir;

namespace {

class TpuKernelOutliningPass : public ModulePass<TpuKernelOutliningPass> {
public:
  void runOnModule() override {
    ModuleManager moduleManager(getModule());
    for (auto func : getModule().getOps<FuncOp>()) {
      //func.walk<mlir::tpu::LaunchOp>([&](mlir::tpu::LaunchOp op) {
        //FuncOp outlinedFunc = outlineKernelFunc(op);
        //moduleManager.insert(outlinedFunc);
        //convertToLaunchFuncOp(op, outlinedFunc);
      //});
    }
  }
};

} // namespace

ModulePassBase *mlir::createTpuKernelOutliningPass() {
  return new TpuKernelOutliningPass();
}

static PassRegistration<TpuKernelOutliningPass>
    pass("tpu-kernel-outlining",
         "Outline tpu.launch bodies to kernel functions.");
