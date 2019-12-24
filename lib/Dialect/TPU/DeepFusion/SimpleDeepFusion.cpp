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

#define DEBUG_TYPE "deep-fusion-simple"

using namespace mlir;

// TODO: move to backend
static const struct MachineInfo {
  const int lane_num = 32;
  const int eu_num = 16;
  const uint64_t lmem_per_lane = 32 * 1024;
} mInfo;


static llvm::cl::opt<std::string> clDeepFusionStatsFilename(
    "deep-fusion-simple-stats",
    llvm::cl::desc("dump simple deepfusion statistics into a csv file"),
    llvm::cl::init("-"));

namespace {

class DeepFusionSimpleStats {
public:
  explicit DeepFusionSimpleStats() {};
  DeepFusionSimpleStats(const DeepFusionSimpleStats &) = delete;
  DeepFusionSimpleStats& operator=(const DeepFusionSimpleStats&) = delete;
  ~DeepFusionSimpleStats() = default;

  void dump(void) {
    llvm::errs() << "Total MAC Count: " << totalMacCount << "\n";
    for (auto it = chains.begin(); it != chains.end(); ++it) {
      llvm::errs() << "Chain: size = " << (*it)->size() << "\n";
      for (auto v = (*it)->begin(); v != (*it)->end(); ++v) {
        auto opInst = (*v)->getDefiningOp();
        if (auto op = dyn_cast<mlir::tpu::Conv2DOp>(opInst)) {
          llvm::errs() << "  " << op.name() << "\n";
        } else if (auto op = dyn_cast<mlir::tpu::Pool2DOp>(opInst)) {
          llvm::errs() << "  " << op.name() << "\n";
        } else if (auto op = dyn_cast<mlir::tpu::FullyConnectedOp>(opInst)) {
          llvm::errs() << "  " << op.name() << "\n";
        } else {
          assert(0);
        }
      }
    }
  }

  void pushChain(Value *op) {
    if (!curChain) {
      curChain = std::make_unique<std::vector<Value *> >();
    }
    curChain->push_back(op);
  }

  void completeChain(void) {
    if (curChain && curChain->size() != 0) {
      chains.push_back(std::move(curChain));
    }
  }

  void increaseMacCount(uint64_t count) {
    totalMacCount += count;
  }

private:
  uint64_t totalMacCount = 0;
  std::vector<std::unique_ptr<std::vector<Value *> > > chains;
  std::unique_ptr<std::vector<Value *> > curChain;
};

class DeepFusionSimple : public FunctionPass<DeepFusionSimple> {
public:
  explicit DeepFusionSimple() {}

  void runOnFunction() override {
    std::unique_ptr<llvm::ToolOutputFile> file = nullptr;
    if (clDeepFusionStatsFilename != "-") {
      std::string errorMessage;
      file = openOutputFile(clDeepFusionStatsFilename, &errorMessage);
      if (!file) {
        llvm::errs() << errorMessage << "\n";
        exit(1);
      }
      file->keep();
    }
    llvm::raw_ostream &os = file ? file->os() : llvm::errs();

    auto func = getFunction();
    os << "name" << "," << "n" << "," << "g" << ","
       << "ic" << "," << "ih" << "," << "iw" << ","
       << "oc" << "," << "oh" << "," << "ow" << ","
       << "kh" << "," << "kw" << "," << "sh" << "," << "sw" << ","
       << "dh" << "," << "dw" << "," << "ph" << "," << "pw" << ","
       << "mac_count";
    os << "," << "lmem_i";
    os << "," << "lmem_o";
    os << "," << "lmem_f";
    os << "," << "lmem_b";
    os << "," << "lmem_e_i";
    os << "," << "lmem_e_w";
    os << "," << "lmem_total";
    os <<"\n";
    stats = new DeepFusionSimpleStats();
    func.walk([&](mlir::Operation *opInst) {
      if (auto op = dyn_cast<mlir::tpu::Conv2DOp>(opInst)) {
        analyzeConv2DOpParam(op, os);
      } else if (auto op = dyn_cast<mlir::tpu::Pool2DOp>(opInst)) {
        analyzePool2DOpParam(op, os);
      } else if (auto op = dyn_cast<mlir::tpu::FullyConnectedOp>(opInst)) {
        analyzeFullyConnectedOpParam(op, os);
      } else if (auto op = dyn_cast<mlir::tpu::LoadWeightOp>(opInst)) {
        // we do analysis in compute node, skip load node
      } else if (auto op = dyn_cast<mlir::tpu::ReshapeOp>(opInst)) {
        // reshape has no computation or load/store, skip
      } else {
        llvm::errs() << "DeepFusion: Unsupported Op " << opInst->getName()
                     << ", break fusion here\n";
        stats->completeChain();
      }
    });
    stats->dump();
    delete stats;
  }

private:
  DeepFusionSimpleStats *stats;

  uint64_t getSizePerLane(int n, int c, int h, int w) {
    uint64_t cPerLane = llvm::alignTo(c, mInfo.lane_num) / mInfo.lane_num;
    return n * cPerLane * h * w;
  }

  void analyzeConv2DOpParam(tpu::Conv2DOp &op, llvm::raw_ostream &os) {
    // supporat int8 multiplier mode only
    assert(op.quant() == "INT8_MULTIPLIER");

    bool with_bias, do_relu;
    int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, ph, pw, dh, dw;
    getConv2DOpParam(op, n, ic, ih, iw, oc, oh, ow, g,
                     kh, kw, sh, sw, ph, pw, dh, dw, with_bias, do_relu);

    uint64_t mac_count = ow * oh * kh * kw * g * (ic / g) * (oc / g) * n;
    stats->increaseMacCount(mac_count);

    uint64_t inputNeuronSizePerLane = getSizePerLane(n, ic, ih, iw);
    uint64_t outputNeuronSizePerLane = getSizePerLane(n, oc, oh, ow);
    uint64_t filterSizePerLane = 0;
    // filter working size *2 for double buffer
    if (g != oc) {
      // for non-dw conv, assuming oc_step = lane_num
      int oc_step = mInfo.lane_num;
      filterSizePerLane = getSizePerLane(ic, oc_step, kh, kw) * 2;
    } else {
      // for dw conv, load weight all in once
      filterSizePerLane = getSizePerLane(1, oc, kh, kw) * 2;
    }
    // load bias all in once, and always assume bias is enabled (elt size is 9)
    uint64_t biasSizePerLane = getSizePerLane(9, oc, 1, 1);
    // if eltwise sum is enabled, eltwise input size
    uint64_t eltwiseInputSizePerLane = 0;
    uint64_t eltwiseWorkingSizePerLane = 0;
    if (op.fused_eltwise_method() == "SUM") {
      eltwiseInputSizePerLane = outputNeuronSizePerLane;
      // assuming each lane handle mininum 16 elements in one command
      // intermediate result in 16-bit format
      eltwiseWorkingSizePerLane = 16 * 2;
    }
    uint64_t totalPerLane = inputNeuronSizePerLane + outputNeuronSizePerLane
                            + filterSizePerLane + biasSizePerLane
                            + eltwiseInputSizePerLane + eltwiseWorkingSizePerLane;
    if (totalPerLane <= mInfo.lmem_per_lane) {
      stats->pushChain(op.getResult());
    } else {
      stats->completeChain();
    }

    os << op.name() << "," << n << "," << g << ","
       << ic << "," << ih << "," << iw << ","
       << oc << "," << oh << "," << ow << ","
       << kh << "," << kw << "," << sh << "," << sw << ","
       << dh << "," << dw << "," << ph << "," << pw << ","
       << mac_count;
    os << "," << inputNeuronSizePerLane;
    os << "," << outputNeuronSizePerLane;
    os << "," << filterSizePerLane;
    os << "," << biasSizePerLane;
    os << "," << eltwiseInputSizePerLane;
    os << "," << eltwiseWorkingSizePerLane;
    os << "," << totalPerLane;
    os <<"\n";
  }

  void analyzePool2DOpParam(tpu::Pool2DOp &op, llvm::raw_ostream &os) {
    bool is_average_pool, do_relu;
    int n, c, ih, iw, oh, ow, kh, kw, sh, sw, ph, pw;
    getPool2DOpParam(op, is_average_pool, n, c, ih, iw, oh, ow,
                     kh, kw, sh, sw, ph, pw, do_relu);

    uint64_t mac_count = ow * oh * kh * kw * c * n;
    stats->increaseMacCount(mac_count);

    uint64_t inputNeuronSizePerLane = getSizePerLane(n, c, ih, iw);
    uint64_t outputNeuronSizePerLane = getSizePerLane(n, c, oh, ow);
    uint64_t totalPerLane = inputNeuronSizePerLane + outputNeuronSizePerLane;
    if (totalPerLane <= mInfo.lmem_per_lane) {
      stats->pushChain(op.getResult());
    } else {
      stats->completeChain();
    }

    os << op.name() << "," << n << "," << ","
       << c << "," << ih << "," << iw << ","
       << "," << oh << "," << ow << ","
       << kh << "," << kw << "," << sh << "," << sw << ","
       << "," << "," << ph << "," << pw << ","
       << mac_count;
    os << "," << inputNeuronSizePerLane;
    os << "," << outputNeuronSizePerLane;
    os << ",";
    os << ",";
    os << ",";
    os << ",";
    os << "," << totalPerLane;
    os << "\n";
  }

  void analyzeFullyConnectedOpParam(tpu::FullyConnectedOp &op, llvm::raw_ostream &os) {
    bool with_transpose, with_bias, do_relu;
    int m, k, n;
    getFullyConnectedOpParam(op, with_transpose, m, k, n, with_bias, do_relu);

    uint64_t mac_count = m * k * n;
    stats->increaseMacCount(mac_count);

    uint64_t inputNeuronSizePerLane = getSizePerLane(m, k, 1, 1);
    uint64_t outputNeuronSizePerLane = getSizePerLane(m, n, 1, 1);
    uint64_t totalPerLane = inputNeuronSizePerLane + outputNeuronSizePerLane;
    if (totalPerLane <= mInfo.lmem_per_lane) {
      stats->pushChain(op.getResult());
    } else {
      stats->completeChain();
    }

    os << op.name() << "," << m << "," << ","
       << k << "," << "," << ","
       << m << "," << "," << ","
       << "," << "," << "," << ","
       << "," << "," << "," << ","
       << mac_count;
    os << "," << inputNeuronSizePerLane;
    os << "," << outputNeuronSizePerLane;
    os << ",";
    os << ",";
    os << ",";
    os << ",";
    os << "," << totalPerLane;
    os << "\n";
  }

};

} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createDeepFusionSimple() {
  return std::make_unique<DeepFusionSimple>();
}

static PassRegistration<DeepFusionSimple>
    pass("deep-fusion-simple",
         "Apply simple deep fusion.");
