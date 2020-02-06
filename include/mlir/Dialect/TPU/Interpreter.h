//===- Interpreter.h - Interpreter driver ------------------------------*- C++ -*-===//
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
// This header file defines prototypes that expose interpreter constructors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TPU_INTERPRETER_H_
#define MLIR_DIALECT_TPU_INTERPRETER_H_

#include "mlir/IR/Module.h"
#include "mlir/Dialect/TPU/ModuleInterpreter.h"

namespace mlir {

class ModuleOp;

LogicalResult runTpuModule(ModuleOp m,
    std::vector<int64_t> input_shape, std::vector<float> &input_vec,
    std::map<std::string, std::vector<float> > *results,
    std::map<std::string, std::vector<int64_t> > *shapeMap,
    std::map<std::string, std::vector<float> > *allTensorMap);

LogicalResult runTpuModule(ModuleOp m, ModuleInterpreter *interpreter,
    std::vector<int64_t> input_shape, std::vector<float> &input_vec,
    std::map<std::string, std::vector<float> > *results,
    std::map<std::string, std::vector<int64_t> > *shapeMap,
    std::map<std::string, std::vector<float> > *allTensorMap);
LogicalResult InitGpuOp(ModuleOp m, std::map<std::string, BaseGpuOp*> &gpuOps);
} // namespace mlir

#endif // MLIR_DIALECT_TPU_INTERPRETER_H_
