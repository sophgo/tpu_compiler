//===- TpuOpStats.cpp - Implementation of TPU Op Stats ---------===//
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
// This file implements the TPU dialect OP Stats pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/MathExtras.h"
#include "MachineInfo.h"

struct SimpleMemoryUsageAnalysis_details {
  uint64_t inputNeuronSizePerLane;
  uint64_t outputNeuronSizePerLane;
  uint64_t filterSizePerLane;
  uint64_t biasSizePerLane;
  uint64_t reluWorkingSizePerLane;
  uint64_t eltwiseInputSizePerLane;
  uint64_t eltwiseWorkingSizePerLane;
};

template <typename OpTy>
uint64_t SimpleConv2DMemoryUsageAnalysis(OpTy &op,
    struct SimpleMemoryUsageAnalysis_details *details);

template <typename OpTy>
uint64_t SimpleEltwiseAddMemoryUsageAnalysis(OpTy &op,
    struct SimpleMemoryUsageAnalysis_details *details);
