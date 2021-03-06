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
#include "tpuc/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"

#define DEBUG_TYPE "tpu_op_stats"

using namespace mlir;

namespace {

static llvm::cl::opt<std::string> clOpStatsFilename(
    "tpu-op-stats-filename",
    llvm::cl::desc("dump tpu op statistics into a csv file"),
    llvm::cl::init("-"));

class PrintTpuOpStatsPass : public mlir::PassWrapper<PrintTpuOpStatsPass, FunctionPass> {
public:
  explicit PrintTpuOpStatsPass() {}

  void runOnFunction() override {
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

    //mlir::SymbolTable moduleSymTable(module);
    os << "name" << "," << "n" << "," << "g" << ","
       << "ic" << "," << "ih" << "," << "iw" << ","
       << "oc" << "," << "oh" << "," << "ow" << ","
       << "kh" << "," << "kw" << "," << "sh" << "," << "sw" << ","
       << "dh" << "," << "dw" << "," << "ph" << "," << "pw" << ","
       << "mac_count"
       << "\n";
    total_mac_count = 0;
    auto func = getFunction();
    func.walk([&](Operation *opInst) {
      if (auto op = dyn_cast<tpu::Conv2DOp>(opInst)) {
        dumpConv2DOpParam<tpu::Conv2DOp>(op, os);
      } else if (auto op = dyn_cast<tpu::DeConv2DOp>(opInst)) {
        dumpConv2DOpParam<tpu::DeConv2DOp>(op, os);
      } else if (auto op = dyn_cast<tpu::PoolAvg2DOp>(opInst)) {
        dumpPool2DOpParam<tpu::PoolAvg2DOp>(op, os, true);
      } else if (auto op = dyn_cast<tpu::PoolMax2DOp>(opInst)) {
        dumpPool2DOpParam<tpu::PoolMax2DOp>(op, os, false);
      } else if (auto op = dyn_cast<tpu::FullyConnectedOp>(opInst)) {
        dumpFullyConnectedOpParam(op, os);
      }
    });
    LLVM_DEBUG(llvm::dbgs() << "Total MAC Count: " << total_mac_count << "\n");
  }

private:
  uint64_t total_mac_count;

  template <typename OpTy>
  void dumpConv2DOpParam(OpTy &op, llvm::raw_ostream &os) {
    bool is_dw, with_bias;
    int n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb, pl, pr, dh, dw, pad_value;
    bool is_deconv = isa<tpu::DeConv2DOp>(op.getOperation());
    parseConvParam(op.param(), is_deconv, op.input(), op.output(),
                   n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb, pl, pr,
                   dh, dw, is_dw, with_bias, pad_value);

    uint64_t mac_count = (uint64_t)ow * oh * kh * kw * g * (ic / g) * (oc / g) * n;
    total_mac_count += mac_count;
    os << op.name() << "," << n << "," << g << ","
       << ic << "," << ih << "," << iw << ","
       << oc << "," << oh << "," << ow << ","
       << kh << "," << kw << "," << sh << "," << sw << ","
       << dh << "," << dw << "," << pt << "," << pb << ","
       << pl << "," << pr << ","
       << mac_count
       << "\n";
  }

  template <typename OpTy>
  void dumpPool2DOpParam(OpTy &op, llvm::raw_ostream &os,
      bool is_average) {
    bool is_global, count_include_pad;
    int n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
    parsePoolParam(op.param(), op.input(), op.output(), n, c, ih, iw, oh, ow,
                   kh, kw, sh, sw, pt, pb, pl, pr, pad_value, is_global,
                   count_include_pad);

    uint64_t mac_count = (uint64_t)ow * oh * kh * kw * c * n;
    total_mac_count += mac_count;

    os << op.name() << "," << n << "," << ","
       << c << "," << ih << "," << iw << ","
       << "," << oh << "," << ow << ","
       << kh << "," << kw << "," << sh << "," << sw << ","
       << "," << "," << pt << "," << pb << "," << pl << "," << pr << ","
       << mac_count
       << "\n";
  }

  void dumpFullyConnectedOpParam(tpu::FullyConnectedOp &op,
                                 llvm::raw_ostream &os) {
    int batch, batch_high, batch_low, m, k, n;
    parseFullyConnectedParam<tpu::FullyConnectedOp>(op.getOperation(), batch_high,
                             batch_low, m, k, n);
    batch = batch_high * batch_low;

    uint64_t mac_count = (uint64_t)batch * m * k * n;
    total_mac_count += mac_count;

    os << op.name() << "," << batch << "," << m << "," << ","
       << k << "," << "," << ","
       << m << "," << "," << ","
       << "," << "," << "," << ","
       << "," << "," << "," << ","
       << mac_count
       << "\n";
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::createPrintTpuOpStatsPass() {
  return std::make_unique<PrintTpuOpStatsPass>();
}

