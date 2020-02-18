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

#define DEBUG_TYPE "deep-fusion-simple"

using namespace mlir;

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
        } else if (auto op = dyn_cast<mlir::tpu::PoolAvg2DOp>(opInst)) {
          llvm::errs() << "  " << op.name() << "\n";
        } else if (auto op = dyn_cast<mlir::tpu::PoolMax2DOp>(opInst)) {
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
      } else if (auto op = dyn_cast<tpu::PoolAvg2DOp>(opInst)) {
        analyzePool2DOpParam<tpu::PoolAvg2DOp>(op, os, true);
      } else if (auto op = dyn_cast<tpu::PoolMax2DOp>(opInst)) {
        analyzePool2DOpParam<tpu::PoolMax2DOp>(op, os, false);
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

  template <typename OpTy>
  void analyzeConv2DOpParam(OpTy &op, llvm::raw_ostream &os) {
    // supporat int8 multiplier mode only
    assert(op.quant().mode().getValue() == "INT8");
    assert(op.quant().is_perchannel().getValue() == true);

    bool is_dw, with_bias, do_relu;
    int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, ph, pw, dh, dw;
    parseConvParam(op.param(), op.input(), op.output(), op.filter(),
                   n, ic, ih, iw, oc, oh, ow, g,
                   kh, kw, sh, sw, ph, pw, dh, dw, is_dw, with_bias, do_relu);

    //bool do_eltwise = (op.fused_eltwise_method() == "SUM") ? true : false;
    bool do_eltwise = false;
    uint64_t mac_count = ow * oh * kh * kw * g * (ic / g) * (oc / g) * n;
    stats->increaseMacCount(mac_count);

    uint64_t inputNeuronSizePerLane = MInfo::getSizePerLane(n, ic, ih, iw, true);
    uint64_t outputNeuronSizePerLane = MInfo::getSizePerLane(n, oc, oh, ow, true);
    uint64_t filterSizePerLane = 0;
    // filter working size *2 for double buffer
    if (g != oc) {
      assert(g == 1);
      // for non-dw conv, assuming oc_step = lane_num
      int oc_step = MInfo::lane_num;
      filterSizePerLane = MInfo::getSizePerLane(ic, oc_step, kh, kw, false) * 2;
    } else {
      // for dw conv, load weight all in once
      filterSizePerLane = MInfo::getSizePerLane(1, oc, kh, kw, false) * 2;
    }
    // load bias all in once
    int bias_size = with_bias ? 9 : 5;
    uint64_t biasSizePerLane = MInfo::getSizePerLane(1, oc, 1, bias_size, false);
    // if eltwise sum is enabled, eltwise input size
    uint64_t eltwiseInputSizePerLane = 0;
    uint64_t eltwiseWorkingSizePerLane = 0;
    if (do_eltwise) {
      eltwiseInputSizePerLane = outputNeuronSizePerLane;
      #define MIN_eltwise_working_size    (32)
      eltwiseWorkingSizePerLane = MIN_eltwise_working_size * 2;
    }
    // total
    uint64_t totalPerLane = inputNeuronSizePerLane + outputNeuronSizePerLane
                            + filterSizePerLane + biasSizePerLane
                            + eltwiseInputSizePerLane + eltwiseWorkingSizePerLane;
    if (totalPerLane <= MInfo::lmem_per_lane) {
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
  template <typename Opty>
  void analyzePool2DOpParam(Opty &op, llvm::raw_ostream &os,
      bool is_average) {
    bool is_global, do_relu;
    int n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr;
    parsePoolParam(op.param(), op.input(), op.output(),
                   n, c, ih, iw, oh, ow,
                   kh, kw, sh, sw, pt, pb, pl, pr,
                   is_global, do_relu);

    uint64_t mac_count = ow * oh * kh * kw * c * n;
    stats->increaseMacCount(mac_count);

    uint64_t inputNeuronSizePerLane = MInfo::getSizePerLane(n, c, ih, iw, true);
    uint64_t outputNeuronSizePerLane = MInfo::getSizePerLane(n, c, oh, ow, true);
    uint64_t totalPerLane = inputNeuronSizePerLane + outputNeuronSizePerLane;
    if (totalPerLane <= MInfo::lmem_per_lane) {
      stats->pushChain(op.getResult());
    } else {
      stats->completeChain();
    }

    os << op.name() << "," << n << "," << ","
       << c << "," << ih << "," << iw << ","
       << "," << oh << "," << ow << ","
       << kh << "," << kw << "," << sh << "," << sw << ","
       << "," << "," << pt << "," << pb << "," << pl << "," << pr << ","
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

    uint64_t inputNeuronSizePerLane = MInfo::getSizePerLane(m, k, 1, 1, false);
    uint64_t outputNeuronSizePerLane = MInfo::getSizePerLane(m, n, 1, 1, false);
    uint64_t totalPerLane = inputNeuronSizePerLane + outputNeuronSizePerLane;
    if (totalPerLane <= MInfo::lmem_per_lane) {
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
