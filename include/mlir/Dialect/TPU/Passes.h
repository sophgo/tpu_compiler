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

#ifndef MLIR_DIALECT_TPU_PASSES_H_
#define MLIR_DIALECT_TPU_PASSES_H_

namespace mlir {

class ModulePassBase;
class FunctionPassBase;

std::unique_ptr<ModulePassBase> createPrintTpuOpStatsPass();
std::unique_ptr<ModulePassBase> createPrintTpuOpStatsPass_v0();

std::unique_ptr<FunctionPassBase> createConvertBnToScalePass();
std::unique_ptr<FunctionPassBase> createFoldScalePass();
std::unique_ptr<FunctionPassBase> createMergeScaleIntoConvPass();
std::unique_ptr<FunctionPassBase> createFuseReluPass();
std::unique_ptr<FunctionPassBase> createFuseEltwisePass();

std::unique_ptr<FunctionPassBase> createImportCalibrationTablePass();
std::unique_ptr<FunctionPassBase> createQuantizeInt8Pass();
std::unique_ptr<FunctionPassBase> createQuantizeBf16Pass();

std::unique_ptr<FunctionPassBase> createAssignWeightAddressPass();
std::unique_ptr<FunctionPassBase> createAssignNeuronAddressPass();

std::unique_ptr<FunctionPassBase> createAssignLayerIdPass();

} // namespace mlir

#endif // MLIR_DIALECT_TPU_PASSES_H_
