//===- TpuOpPrint.cpp - Implementation of TPU Op Print ---------===//
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
// This file implements the TPU dialect OP pass.
//
//===----------------------------------------------------------------------===//

#include <random>
#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

namespace {

class GenPseudoWeightNpzPass : public FunctionPass<GenPseudoWeightNpzPass> {
public:
  explicit GenPseudoWeightNpzPass() {}

  void runOnFunction() override {
    auto fn = getFunction();
    auto *context = &getContext();
    Builder builder(context);

    cnpy::npz_t npz;
    std::random_device rd{};
    std::mt19937 gen{rd()};
    StringRef npzFileName;

    fn.walk([&](Operation *op) {
      if (auto castOp = llvm::dyn_cast<tpu::WeightFileOp>(op)) {
        npzFileName = castOp.filename();
      } else if (auto castOp = llvm::dyn_cast<tpu::LoadWeightOp>(op)) {
        std::string op_name = castOp.name().str();
        llvm::errs() << op_name << "\n";
        auto resultShape = getTensorShape(castOp.getResult());
        std::vector<size_t> shape;
        for (int i = 0; i < (int)resultShape.size(); ++i) {
          shape.push_back(resultShape[i]);
        }
        auto count = std::accumulate(resultShape.begin(), resultShape.end(), 1,
                                std::multiplies<int>());
        auto elementType =
            op->getResult(0)->getType().template cast<TensorType>().getElementType();
        if (elementType.isF32()) {
          std::vector<float> data(count);
          std::normal_distribution<float> d{0.3, 0.2};
          for (int i = 0; i < (int)count; i++) {
            float rand = d(gen);
            rand = rand < 0 ? 0 : rand;
            rand = rand > 1 ? 1 : rand;
            data[i] = rand;
          }
          cnpy::npz_add_array<float>(npz, op_name, (float *)data.data(), shape);
        } else if (elementType.isBF16()) {
          std::vector<uint16_t> data(count);
          std::normal_distribution<float> d{0.3, 0.2};
          for (int i = 0; i < (int)count; i++) {
            float rand = d(gen);
            rand = rand < 0 ? 0 : rand;
            rand = rand > 1 ? 1 : rand;
            uint16_t ui16 = (uint16_t)((*(uint32_t *)&rand) >> 16);
            data[i] = ui16;
          }
          cnpy::npz_add_array<uint16_t>(npz, op_name, (uint16_t *)data.data(), shape);
        } else if (elementType.isInteger(8)) {
          std::vector<float> data(count);
          std::normal_distribution<float> d{50, 50};
          for (int i = 0; i < (int)count; i++) {
            float rand = std::round(d(gen));
            rand = rand < 0 ? 0 : rand;
            rand = rand > 127 ? 127 : rand;
            data[i] = (float)rand;
          }
          cnpy::npz_add_array<float>(npz, op_name, (float *)data.data(), shape);
        } else {
          llvm_unreachable("unsupported data type");
        }
      }
    });
    cnpy::npz_save_all(npzFileName.str(), npz);
  }
};

} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createGenPseudoWeightNpzPass() {
  return std::make_unique<GenPseudoWeightNpzPass>();
}

static PassRegistration<GenPseudoWeightNpzPass>
    pass("gen-pseudo-weight-npz", "Generic fake weight npz file if only have mlir file");
