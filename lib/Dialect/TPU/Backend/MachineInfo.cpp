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

#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/MathExtras.h"
#include "tpuc/MachineInfo.h"

#include "backend/backend_tg_api.h"
#include "backend/backend_tl_api.h"

#define DEBUG_TYPE "run-chip"

using namespace mlir;

namespace mlir {

uint32_t MInfo::version{0};
uint32_t MInfo::eu_num{0};
uint32_t MInfo::lane_num{0};
uint64_t MInfo::lmem_per_lane{0};
uint32_t MInfo::lmem_bank_num{0};

int MInfo::MAX_TIU_BATCH{0};
int MInfo::MAX_TIU_CHANNEL{0};
int MInfo::MAX_TIU_HEIGHT{0};
int MInfo::MAX_TIU_WIDTH{0};

void MInfo::getChipInfo(std::string chipName) {
  CviBackendContext *backend_ctx = nullptr;
  backend_ctx = cvi_backend_create_context(chipName.c_str());

  version = cvi_backend_chip_context(backend_ctx, CVI_CHIP_VERSION);
  eu_num = cvi_backend_chip_context(backend_ctx, CVI_CHIP_EU_NUM);
  lane_num = cvi_backend_chip_context(backend_ctx, CVI_CHIP_LANE_NUM);
  lmem_per_lane = cvi_backend_chip_context(backend_ctx, CVI_CHIP_LMEM_SIZE);
  lmem_bank_num = cvi_backend_chip_context(backend_ctx, CVI_CHIP_LMEM_BANK);

  MAX_TIU_BATCH = 4095 - 32;
  MAX_TIU_CHANNEL = 4095 - 32;
  MAX_TIU_HEIGHT = 4095 - 32;
  MAX_TIU_WIDTH = 4095 - 32;

  LLVM_DEBUG(llvm::errs() << " version = " << version << "\n";);
  LLVM_DEBUG(llvm::errs() << " eu = " << eu_num << "\n";);
  LLVM_DEBUG(llvm::errs() << " lane = " << lane_num << "\n";);
  LLVM_DEBUG(llvm::errs() << " lane size= " << lmem_per_lane << "\n";);

  cvi_backend_delete_context(backend_ctx);
}

void MInfo::getChipInfo(FuncOp fn) {
  // get chipname form function attributes.
  std::string chipname = "cx183x";
  if (fn->getAttr("chipname")) {
    chipname = fn->getAttr("chipname").cast<StringAttr>().getValue().str();
  }
  getChipInfo(chipname);

}
} // namespace mlir