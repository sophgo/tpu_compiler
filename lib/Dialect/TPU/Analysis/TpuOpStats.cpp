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

using namespace mlir;

namespace {

static llvm::cl::opt<std::string> clOpStatsFilename(
    "tpu-op-stats-filename",
    llvm::cl::desc("dump tpu op statistics into a csv file"),
    llvm::cl::init("-"));

class PrintTpuOpStatsPass : public ModulePass<PrintTpuOpStatsPass> {
public:
  explicit PrintTpuOpStatsPass() {}

  void runOnModule() override {
    std::unique_ptr<llvm::ToolOutputFile> file = nullptr;
    if (clOpStatsFilename != "-") {
      std::string errorMessage;
      file = openOutputFile(clOpStatsFilename, &errorMessage);
      if (!file) {
        llvm::errs() << errorMessage << "\n";
        exit(1);
      }
      file->keep();
    }
    llvm::raw_ostream &os = file ? file->os() : llvm::errs();

    mlir::ModuleOp module = getModule();
    //mlir::SymbolTable moduleSymTable(module);

    os << "name" << "," << "n" << "," << "g" << ","
       << "ic" << "," << "ih" << "," << "iw" << ","
       << "oc" << "," << "oh" << "," << "ow" << ","
       << "kh" << "," << "kw" << "," << "sh" << "," << "sw" << ","
       << "dh" << "," << "dw" << "," << "ph" << "," << "pw" << ","
       << "mac_count"
       <<"\n";
    total_mac_count = 0;
    for (auto func : module.getOps<FuncOp>()) {
      func.walk([&](Operation *opInst) {
        if (auto op = dyn_cast<tpu::Conv2DOp>(opInst)) {
          dumpConv2DOpParam(op, os);
        } else if (auto op = dyn_cast<tpu::PoolAvg2DOp>(opInst)) {
          dumpPool2DOpParam<tpu::PoolAvg2DOp>(op, os, true);
        } else if (auto op = dyn_cast<tpu::PoolMax2DOp>(opInst)) {
          dumpPool2DOpParam<tpu::PoolMax2DOp>(op, os, false);
        } else if (auto op = dyn_cast<tpu::FullyConnectedOp>(opInst)) {
          dumpFullyConnectedOpParam(op, os);
        }
      });
    }
    llvm::errs() << "Total MAC Count: " << total_mac_count << "\n";
  }

private:
  uint64_t total_mac_count;

  void dumpConv2DOpParam(tpu::Conv2DOp &op, llvm::raw_ostream &os) {
    bool is_dw, with_bias, do_relu;
    int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, ph, pw, dh, dw;
    parseConvParam(op.param(), op.input(), op.output(), op.filter(),
                   n, ic, ih, iw, oc, oh, ow, g,
                   kh, kw, sh, sw, ph, pw, dh, dw, is_dw, with_bias, do_relu);

    uint64_t mac_count = ow * oh * kh * kw * g * (ic / g) * (oc / g) * n;
    total_mac_count += mac_count;

    os << op.name() << "," << n << "," << g << ","
       << ic << "," << ih << "," << iw << ","
       << oc << "," << oh << "," << ow << ","
       << kh << "," << kw << "," << sh << "," << sw << ","
       << dh << "," << dw << "," << ph << "," << pw << ","
       << mac_count
       <<"\n";
  }

  template <typename OpTy>
  void dumpPool2DOpParam(OpTy &op, llvm::raw_ostream &os,
      bool is_average) {
    bool is_global, do_relu;
    int n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr;
    parsePoolParam(op.param(), op.input(), op.output(),
                   n, c, ih, iw, oh, ow,
                   kh, kw, sh, sw, pt, pb, pl, pr,
                   is_global, do_relu);

    uint64_t mac_count = ow * oh * kh * kw * c * n;
    total_mac_count += mac_count;

    os << op.name() << "," << n << "," << ","
       << c << "," << ih << "," << iw << ","
       << "," << oh << "," << ow << ","
       << kh << "," << kw << "," << sh << "," << sw << ","
       << "," << "," << pt << "," << pb << "," << pl << "," << pr << ","
       << mac_count
       <<"\n";
  }

  void dumpFullyConnectedOpParam(tpu::FullyConnectedOp &op, llvm::raw_ostream &os) {
    bool with_transpose, with_bias, do_relu;
    int m, k, n;
    getFullyConnectedOpParam(op, with_transpose, m, k, n, with_bias, do_relu);

    uint64_t mac_count = m * k * n;
    total_mac_count += mac_count;

    os << op.name() << "," << m << "," << ","
       << k << "," << "," << ","
       << m << "," << "," << ","
       << "," << "," << "," << ","
       << "," << "," << "," << ","
       << mac_count
       <<"\n";
  }

};

} // namespace

std::unique_ptr<OpPassBase<ModuleOp>> mlir::createPrintTpuOpStatsPass() {
  return std::make_unique<PrintTpuOpStatsPass>();
}

static PassRegistration<PrintTpuOpStatsPass>
    pass("print-tpu-op-stats",
         "Print statistics of TPU operations.");
