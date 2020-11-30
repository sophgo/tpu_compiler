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

#include "mlir/Pass/Pass.h"

namespace mlir {

// class ModuleOp;
// class FuncOp;
// template <typename T> class OpPassBase;

std::unique_ptr<mlir::Pass> createPrintTpuOpPass();
std::unique_ptr<mlir::Pass> createPrintTpuOpStatsPass();
std::unique_ptr<mlir::Pass> createGenPseudoWeightNpzPass();
std::unique_ptr<mlir::Pass> createConvertFuncToMemRefPass();
std::unique_ptr<mlir::Pass> createDecomposeNormalizePass();
std::unique_ptr<mlir::Pass> createConvertBnToScalePass();
std::unique_ptr<mlir::Pass> createConvertPoolMaskPass();
std::unique_ptr<mlir::Pass> createConvertUpsampleToDeconvPass();
std::unique_ptr<mlir::Pass> createFoldScalePass();
std::unique_ptr<mlir::Pass> createMergeScaleIntoConvPass();
std::unique_ptr<mlir::Pass> createConvertScaleToDWConvPass();
std::unique_ptr<mlir::Pass> createConvertSwishToReLUPass();
std::unique_ptr<mlir::Pass> createClipAsRelu6Pass();
std::unique_ptr<mlir::Pass> createTpuQuantClipPass();
std::unique_ptr<mlir::Pass> createFuseAsymmetricZeroPointPass();
std::unique_ptr<mlir::Pass> createFuseReluPass();
std::unique_ptr<mlir::Pass> createFusePadPass();
std::unique_ptr<mlir::Pass> createFuseEltwisePass();
std::unique_ptr<mlir::Pass> createRefactorEltAndConvPass();
std::unique_ptr<mlir::Pass> createRefactorOddIcConvPass();

std::unique_ptr<mlir::Pass> createGenReciprocalTablePass();
//std::unique_ptr<mlir::Pass> createGenPowerWeightPass() ;
std::unique_ptr<mlir::Pass> createGenSigmoidTablePass();
std::unique_ptr<mlir::Pass> createGenSqrtTablePass();
//std::unique_ptr<mlir::Pass> createGenTanHTablePass();

std::unique_ptr<mlir::Pass> createImportCalibrationTablePass();
std::unique_ptr<mlir::Pass> createTpuQuantPass();

std::unique_ptr<mlir::Pass> createTpuLowerPass();

std::unique_ptr<mlir::Pass> createAssignWeightAddressPass();
std::unique_ptr<mlir::Pass> createAssignNeuronAddressPass();
std::unique_ptr<mlir::Pass> createAssignLayerIdPass();
std::unique_ptr<mlir::Pass> createAssignChipNamePass();
std::unique_ptr<mlir::Pass> createAddCpuCallPass();

std::unique_ptr<mlir::Pass> createDeepFusionSimple();
std::unique_ptr<mlir::Pass> createDeepFusionTG2TL_LA();
std::unique_ptr<mlir::Pass> createDeepFusionTL_LA2LW();

std::unique_ptr<mlir::Pass> createConvertPriorBoxPass();
std::unique_ptr<mlir::Pass> createConvertLoadeweightConcatToLoadweightPass();


std::unique_ptr<mlir::Pass> createTgFuseLeakyReluPass();

std::unique_ptr<mlir::Pass> createConvertTgOpToMemRefPass();
std::unique_ptr<mlir::Pass> createConvertTgOpToTensorPass();
std::unique_ptr<mlir::Pass> createAssignNeuronAddressMemRefPass();

std::unique_ptr<mlir::Pass> createCompressActivationPass();
std::unique_ptr<mlir::Pass> createCompressWeightPass();
std::unique_ptr<mlir::Pass> createDeepFusionGroupSlice();
std::unique_ptr<mlir::Pass> createDeepFusionOpt();

std::unique_ptr<mlir::Pass> createTgOpTilePass();

std::unique_ptr<mlir::Pass> createFuseReshapePass();

std::unique_ptr<mlir::Pass> createTpucCanonicalizerPass();
std::unique_ptr<mlir::Pass> createDivideOpsToFuncPass();
std::unique_ptr<mlir::Pass> createEliminateDeadcodePass();
std::unique_ptr<mlir::Pass> createGroupOpsPass(); 
std::unique_ptr<mlir::Pass> createReorderOpPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "tpuc/Dialect/TPU/Passes.h.inc"

} // namespace mlir

#endif // MLIR_DIALECT_TPU_PASSES_H_
