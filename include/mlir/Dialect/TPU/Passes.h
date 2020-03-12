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

class ModuleOp;
class FuncOp;
template <typename T> class OpPassBase;

std::unique_ptr<OpPassBase<ModuleOp>> createPrintTpuOpPass();
std::unique_ptr<OpPassBase<ModuleOp>> createPrintTpuOpStatsPass();

std::unique_ptr<OpPassBase<ModuleOp>> createConvertFuncToMemRefPass();
std::unique_ptr<OpPassBase<ModuleOp>> createSplitCpuOpPass();

std::unique_ptr<OpPassBase<FuncOp>> createDecomposeNormalizePass();
std::unique_ptr<OpPassBase<FuncOp>> createConvertBnToScalePass();
std::unique_ptr<OpPassBase<FuncOp>> createFoldScalePass();
std::unique_ptr<OpPassBase<FuncOp>> createMergeScaleIntoConvPass();
std::unique_ptr<OpPassBase<FuncOp>> createConvertScaleToDWConvPass();
std::unique_ptr<OpPassBase<FuncOp>> createFuseReluPass();
std::unique_ptr<OpPassBase<FuncOp>> createFuseEltwisePass();

std::unique_ptr<OpPassBase<FuncOp>> createGenReciprocalTablePass();
//std::unique_ptr<OpPassBase<FuncOp>> createGenPowerWeightPass() ;
std::unique_ptr<OpPassBase<FuncOp>> createGenSigmoidTablePass();
std::unique_ptr<OpPassBase<FuncOp>> createGenSqrtTablePass();
//std::unique_ptr<OpPassBase<FuncOp>> createGenTanHTablePass();

std::unique_ptr<OpPassBase<FuncOp>> createImportCalibrationTablePass();
std::unique_ptr<OpPassBase<FuncOp>> createTpuQuantPass();

std::unique_ptr<OpPassBase<FuncOp>> createTpuLowerPass();

std::unique_ptr<OpPassBase<FuncOp>> createAssignWeightAddressPass();
std::unique_ptr<OpPassBase<FuncOp>> createAssignNeuronAddressPass();
std::unique_ptr<OpPassBase<FuncOp>> createAssignLayerIdPass();
std::unique_ptr<OpPassBase<FuncOp>> createAddCpuCallPass();

std::unique_ptr<OpPassBase<FuncOp>> createDeepFusionSimple();
std::unique_ptr<OpPassBase<FuncOp>> createDeepFusionTG2TL_LA();
std::unique_ptr<OpPassBase<FuncOp>> createDeepFusionTL_LA2LW();

std::unique_ptr<OpPassBase<FuncOp>> createConvertPriorBoxPass();
std::unique_ptr<OpPassBase<FuncOp>> createConvertLoadeweightConcatToLoadweightPass();


std::unique_ptr<OpPassBase<FuncOp>> createTgFuseLeakyReluPass();

std::unique_ptr<OpPassBase<FuncOp>> createConvertTgOpToMemRefPass();
std::unique_ptr<OpPassBase<FuncOp>> createConvertTgOpToTensorPass();
std::unique_ptr<OpPassBase<FuncOp>> createAssignNeuronAddressMemRefPass();

} // namespace mlir

#endif // MLIR_DIALECT_TPU_PASSES_H_
