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
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

static int64_t calcConv2DOpMacCount(tpu::Conv2DOp &op, const bool debug = false) {
  llvm::raw_ostream &os = (debug == true) ? llvm::errs() : llvm::nulls();

  auto dh = op.dilation_h_factor();  // APInt, use .getLimitedValue(); to get uint65_t
  auto dw = op.dilation_w_factor();
  auto sh = op.stride_h();
  auto sw = op.stride_w();
  os << "  >> " << "sh: " << sh << ", sw: " << sw << ", dh : " << dh << ", dw: " << dw << "\n";

  auto input_type = op.input()->getType().cast<TensorType>();
  std::vector<int64_t> i_s(input_type.getShape());
  os << "  >> " << "input shape  : "
      << i_s[0] << "," << i_s[1] << "," << i_s[2] << "," << i_s[3] << "\n";
  auto output_type = op.output()->getType().cast<TensorType>();
  std::vector<int64_t> o_s(output_type.getShape());
  os << "  >> " << "output shape : "
      << o_s[0] << "," << o_s[1] << "," << o_s[2] << "," << o_s[3] << "\n";
  auto filter_type = op.filter()->getType().cast<TensorType>();
  std::vector<int64_t> f_s(filter_type.getShape());
  os << "  >> " << "filter shape : "
      << f_s[0] << "," << f_s[1] << "," << f_s[2] << "," << f_s[3] << "\n";

  assert((i_s[0] == o_s[0]) && "input N not equal to output N");
  auto n = i_s[0];
  if (n == -1) {
    os << "  >> " << "No determined N, use batch size 1" << "\n";
    n = 1;
  }

  auto oc = f_s[0];
  auto ic = f_s[1];
  auto kh = f_s[2];
  auto kw = f_s[3];
  auto oh = o_s[2];
  auto ow = o_s[3];

  auto mac_count = ow * oh * kh * kw * ic * oc * n;
  os << "  >> " << "MAC count : " << mac_count << ", OP count : " << mac_count * 2 << "\n";

  return mac_count;
}

static int64_t calcFullyConnectedOpMacCount(tpu::FullyConnectedOp &op, const bool debug = false) {
  llvm::raw_ostream &os = (debug == true) ? llvm::errs() : llvm::nulls();

  auto input_type = op.input()->getType().cast<TensorType>();
  std::vector<int64_t> i_s(input_type.getShape());
  os << "  >> " << "input shape  : " << i_s[0] << "," << i_s[1] << "\n";
  auto output_type = op.output()->getType().cast<TensorType>();
  std::vector<int64_t> o_s(output_type.getShape());
  os << "  >> " << "output shape : " << o_s[0] << "," << o_s[1] << "\n";
  auto weight_type = op.weight()->getType().cast<TensorType>();
  std::vector<int64_t> w_s(weight_type.getShape());
  os << "  >> " << "weight shape : " << w_s[0] << "," << w_s[1] << "\n";

  assert((i_s[0] == o_s[0]) && "input M not equal to output M");
  auto M = i_s[0];
  if (M == -1) {
    os << "  >> " << "No determined N, use batch size 1" << "\n";
    M = 1;
  }
  assert((i_s[1] == w_s[0]) && "input K not equal to weight K");
  auto K = i_s[1];
  assert((w_s[1] == o_s[1]) && "weight N not equal to output N");
  auto N = o_s[1];

  auto mac_count = M * K * N;
  os << "  >> " << "MAC count : " << mac_count << ", OP count : " << mac_count * 2 << "\n";

  return mac_count;
}

namespace {

class PrintTpuOpStatsPass : public ModulePass<PrintTpuOpStatsPass> {
public:
  explicit PrintTpuOpStatsPass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnModule() override {
    ModuleManager moduleManager(getModule());

    os << "Modules:\n";
    os << "-----------------------\n";
    for (auto &module : getModule()) {
      module.walk([&](Operation *op) {
        FuncOp funcOp = llvm::dyn_cast_or_null<FuncOp>(op);
        if (funcOp) {
          os << op->getName() << " : " << funcOp.getName() << "\n";
          FunctionType type = funcOp.getType();
          //type.print(os);
          type.dump();
          os << "\n";
        } else {
          os << " > " << op->getName() << "\n";
        }
      });
    }
    os << "\n";

    os << "Funcs:\n";
    os << "-----------------------\n";
    for (auto func : getModule().getOps<FuncOp>()) {
      os << func.getName() << "\n";
      func.walk([&](Operation *op) {
        os << " > " << op->getName() << "\n";
      });
    }
    os << "\n";

    os << "Module walk Conv2DOp:\n";
    os << "-----------------------\n";
    for (auto &module : getModule()) {
      module.walk<mlir::tpu::Conv2DOp>([&](mlir::tpu::Conv2DOp op) {
        os << " > " << op.getOperationName() << "\n";
        //auto mac_count = calcConv2DOpMacCount(op, true);
        auto mac_count = calcConv2DOpMacCount(op);
        os << "  >> MAC: " << mac_count
            << ", OPs: " << mac_count * 2 << "\n";
        //op.dump();
        //os << "\n";
      });
    }
    os << "\n";

    os << "Funcs walk Conv2DOp:\n";
    os << "-----------------------\n";
    for (auto func : getModule().getOps<FuncOp>()) {
      int64_t tatal_mac_count = 0;
      os << func.getName() << "\n";
      func.walk<mlir::tpu::Conv2DOp>([&](mlir::tpu::Conv2DOp op) {
        os << " > " << op.getOperationName() << "\n";
        tatal_mac_count += calcConv2DOpMacCount(op);
      });
      func.walk<mlir::tpu::FullyConnectedOp>([&](mlir::tpu::FullyConnectedOp op) {
        os << " > " << op.getOperationName() << "\n";
        tatal_mac_count += calcFullyConnectedOpMacCount(op);
      });
      os << "func total MAC: " << tatal_mac_count
          << ", total OPs: " << tatal_mac_count * 2 << "\n";
    }
    os << "\n";
  }

private:
  llvm::raw_ostream &os;
};

} // namespace

ModulePassBase *mlir::createPrintTpuOpStatsPass() {
  return new PrintTpuOpStatsPass();
}

static PassRegistration<PrintTpuOpStatsPass>
    pass("print-tpu-op-stats",
         "Print statistics of TPU operations.");
