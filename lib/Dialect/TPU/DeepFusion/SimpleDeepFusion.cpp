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
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
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
#include "SimpleAnalysis.h"

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
    LLVM_DEBUG(
      llvm::errs() << "Total MAC Count: " << totalMacCount << "\n";
      for (auto it = chains.begin(); it != chains.end(); ++it) {
        llvm::errs() << "Chain: size = " << (*it)->size() << "\n";
        for (auto v = (*it)->begin(); v != (*it)->end(); ++v) {
          auto opInst = (*v)->getDefiningOp();
          if (auto op = dyn_cast<mlir::tpu::TG_INT8_PC_Conv2DOp>(opInst)) {
            llvm::errs() << "  " << op.name() << "\n";
          } else if (auto op = dyn_cast<mlir::tpu::TG_INT8_PC_DeConv2DOp>(opInst)) {
            llvm::errs() << "  " << op.name() << "\n";
          } else if (auto op = dyn_cast<mlir::tpu::TG_INT8_PoolAvg2DOp>(opInst)) {
            llvm::errs() << "  " << op.name() << "\n";
          } else if (auto op = dyn_cast<mlir::tpu::TG_INT8_PoolMax2DOp>(opInst)) {
            llvm::errs() << "  " << op.name() << "\n";
          } else if (auto op = dyn_cast<mlir::tpu::TG_INT8_FullyConnectedOp>(opInst)) {
            llvm::errs() << "  " << op.name() << "\n";
          } else if (auto op = dyn_cast<tpu::TG_INT8_LeakyReluOp>(opInst)) {
            llvm::errs() << "  " << op.name() << "\n";
          } else {
            std::string opName = opInst->getName().getStringRef();
            llvm_unreachable(("unsupported tg op " + opName + "\n").c_str());
          }
        }
      }
    );
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
    os << "," << "lmem_l_w";
    os << "," << "lmem_e_i";
    os << "," << "lmem_e_w";
    os << "," << "lmem_total";
    os <<"\n";
    stats = new DeepFusionSimpleStats();
    func.walk([&](mlir::Operation *opInst) {
      if (auto op = dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(opInst)) {
        analyzeConv2DOpParam<tpu::TG_INT8_PC_Conv2DOp>(op, os);
      } else if (auto op = dyn_cast<tpu::TG_INT8_PC_DeConv2DOp>(opInst)) {
        analyzeConv2DOpParam<tpu::TG_INT8_PC_DeConv2DOp>(op, os);
      } else if (auto op = dyn_cast<tpu::TG_INT8_PoolAvg2DOp>(opInst)) {
        analyzePool2DOpParam<tpu::TG_INT8_PoolAvg2DOp>(op, os, true);
      } else if (auto op = dyn_cast<tpu::TG_INT8_PoolMax2DOp>(opInst)) {
        analyzePool2DOpParam<tpu::TG_INT8_PoolMax2DOp>(op, os, false);
      } else if (auto op = dyn_cast<tpu::TG_INT8_FullyConnectedOp>(opInst)) {
        analyzeFullyConnectedOpParam(op, os);
      } else if (auto op = dyn_cast<tpu::TG_INT8_LeakyReluOp>(opInst)) {
        analyzeLeakyReluOpParam(op, os);
      } else if (auto op = dyn_cast<tpu::TG_INT8_LutOp>(opInst)) {
        analyzeLutOpParam(op, os);
      }else if (auto op = dyn_cast<tpu::LoadWeightOp>(opInst)) {
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
    bool is_dw, with_bias, do_relu;
    int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, ph, pw, dh, dw;
    bool is_deconv = isa<tpu::TG_INT8_PC_DeConv2DOp>(op.getOperation());
    parseConvParam(op.param(), is_deconv, op.input(), op.output(), op.filter(),
                   n, ic, ih, iw, oc, oh, ow, g,
                   kh, kw, sh, sw, ph, pw, dh, dw, is_dw, with_bias, do_relu);

    uint64_t mac_count = ow * oh * kh * kw * g * (ic / g) * (oc / g) * n;
    stats->increaseMacCount(mac_count);

    struct SimpleMemoryUsageAnalysis_details details;
    uint64_t totalPerLane = SimpleConv2DMemoryUsageAnalysis(op, &details);
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
    os << "," << details.inputNeuronSizePerLane;
    os << "," << details.outputNeuronSizePerLane;
    os << "," << details.filterSizePerLane;
    os << "," << details.biasSizePerLane;
    os << "," << details.reluWorkingSizePerLane;
    os << "," << details.eltwiseInputSizePerLane;
    os << "," << details.eltwiseWorkingSizePerLane;
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
    os << ",";
    os << "," << totalPerLane;
    os << "\n";
  }

  void analyzeFullyConnectedOpParam(tpu::TG_INT8_FullyConnectedOp &op,
      llvm::raw_ostream &os) {
    int m, k, n;
    parseFullyConnectedParam(op.input(), op.output(), op.filter(), m, k, n);

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
    os << ",";
    os << "," << totalPerLane;
    os << "\n";
  }

  void analyzeLeakyReluOpParam(tpu::TG_INT8_LeakyReluOp &op,
      llvm::raw_ostream &os) {
    std::vector<int64_t> shape;
    int64_t input_size, n, c, h, w;
    getTensorShapeAndSize(op.input(), shape, input_size);
    getNCHW(shape, n, c, h, w);
    uint64_t mac_count = n * c * h * w;

    uint64_t inputNeuronSizePerLane = MInfo::getSizePerLane(n, c, h, w, true);
    uint64_t outputNeuronSizePerLane = MInfo::getSizePerLane(n, c, h, w, true);
    uint64_t totalPerLane = inputNeuronSizePerLane + outputNeuronSizePerLane;
    if (totalPerLane <= MInfo::lmem_per_lane) {
      stats->pushChain(op.getResult());
    } else {
      stats->completeChain();
    }

    os << op.name() << "," << n << "," << ","
       << c << "," << h << "," << w << ","
       << "," << "," << ","
       << "," << "," << "," << ","
       << "," << "," << "," << "," << "," << ","
       << mac_count;
    os << "," << inputNeuronSizePerLane;
    os << "," << outputNeuronSizePerLane;
    os << ",";
    os << ",";
    os << ",";
    os << ",";
    os << ",";
    os << "," << totalPerLane;
    os << "\n";

  }

  void analyzeLutOpParam(tpu::TG_INT8_LutOp &op,
      llvm::raw_ostream &os) {
    std::vector<int64_t> shape;
    int64_t input_size, n, c, h, w;
    getTensorShapeAndSize(op.input(), shape, input_size);
    getNCHW(shape, n, c, h, w);
    uint64_t mac_count = n * c * h * w;

    uint64_t inputNeuronSizePerLane = MInfo::getSizePerLane(n, c, h, w, true);
    uint64_t outputNeuronSizePerLane = MInfo::getSizePerLane(n, c, h, w, true);
    uint64_t totalPerLane = inputNeuronSizePerLane + outputNeuronSizePerLane;
    if (totalPerLane <= MInfo::lmem_per_lane) {
      stats->pushChain(op.getResult());
    } else {
      stats->completeChain();
    }

    os << op.name() << "," << n << "," << ","
       << c << "," << h << "," << w << ","
       << "," << "," << ","
       << "," << "," << "," << ","
       << "," << "," << "," << "," << "," << ","
       << mac_count;
    os << "," << inputNeuronSizePerLane;
    os << "," << outputNeuronSizePerLane;
    os << ",";
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
