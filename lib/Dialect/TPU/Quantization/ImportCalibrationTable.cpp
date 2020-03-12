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
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

#include <sstream>
#include <fstream>
#include <regex>
using namespace mlir;

static llvm::cl::OptionCategory clOptionsCategory("calibration table options");

static llvm::cl::opt<std::string> clCalibrationTableFilename(
    "calibration-table",
    llvm::cl::desc("Specify the calibration table filename"),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<bool> clCaliOverwriteThresholdBackwardRelu(
    "enable-cali-overwrite-threshold-backward-relu",
    llvm::cl::desc("Backward overwrite Relu Ops' threshold"),
    llvm::cl::cat(clOptionsCategory), llvm::cl::init(true));

static llvm::cl::opt<bool> clCaliOverwriteThresholdForwardRelu(
    "enable-cali-overwrite-threshold-forward-relu",
    llvm::cl::desc("Forward overwrite Relu Ops' threshold"),
    llvm::cl::cat(clOptionsCategory), llvm::cl::init(false));

static llvm::cl::opt<bool> clCaliOverwriteThresholdBackwardConcat(
    "enable-cali-overwrite-threshold-backward-concat",
    llvm::cl::desc("Overwrite threshold value for the Ops prev to concat "
                   "with the concat's threhold, and backpropagate result "
                   "threshold to operand ops"),
    llvm::cl::cat(clOptionsCategory), llvm::cl::init(false));

namespace {

/// Backpropgate concat threshold by setting all operands' threshold_y same as
/// the concat threshold_y.
struct BackwardOverwriteThresholdConcatPattern : public OpRewritePattern<tpu::ConcatOp> {
  using OpRewritePattern<tpu::ConcatOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(tpu::ConcatOp op,
                                     PatternRewriter &rewriter) const {
    float threshold_y = getOpThreshold(op);
    int written = 0;
    unsigned nInputs = op.getNumInputs();
    for (unsigned i = 0; i < nInputs; ++i) {
      auto formerOp = op.getOperand(i)->getDefiningOp();
      float threshold_x = getOpThreshold(formerOp);
      if (threshold_x == threshold_y) {
          continue;
      }
      if (auto cast_op = llvm::dyn_cast_or_null<tpu::BroadcastMulOp>(formerOp)) {
        setOpThreshold(formerOp, threshold_y);
      } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::ConcatOp>(formerOp)) {
        setOpThreshold(formerOp, threshold_y);
      } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::Conv2DOp>(formerOp)) {
        setOpThreshold(formerOp, threshold_y);
      } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::Conv2DOp>(formerOp)) {
        setOpThreshold(formerOp, threshold_y);
      } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::DeConv2DOp>(formerOp)) {
        setOpThreshold(formerOp, threshold_y);
      } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::EltwiseAddOp>(formerOp)) {
        setOpThreshold(formerOp, threshold_y);
      } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::EltwiseMaxOp>(formerOp)) {
        setOpThreshold(formerOp, threshold_y);
      } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::EltwiseMulOp>(formerOp)) {
        setOpThreshold(formerOp, threshold_y);
      } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::LeakyReluOp>(formerOp)) {
        setOpThreshold(formerOp, threshold_y);
      } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::ReluOp>(formerOp)) {
        setOpThreshold(formerOp, threshold_y);
      }  else if (auto cast_op = llvm::dyn_cast_or_null<tpu::ShuffleChannelOp>(formerOp)) {
        setOpThreshold(formerOp, threshold_y);
      }  else if (auto cast_op = llvm::dyn_cast_or_null<tpu::UpsampleOp>(formerOp)) {
        setOpThreshold(formerOp, threshold_y);
      } else {
        llvm::errs() << formerOp->getName() << ": behavior not defined\n";
        assert(false);
      }
      llvm::errs() << "Concat set prev " << formerOp->getName()
                   << " ["<< getOpName(formerOp) << "] threshold, from "
                   << std::to_string(threshold_x) << " to "
                   << std::to_string(threshold_y) << "\n";
      if (threshold_x < threshold_y * 0.5) {
        llvm::errs() << "  WARNING: prev threshold is too small to overwrite\n";
      }
      if (threshold_x > threshold_y * 2.0) {
        llvm::errs() << "  WARNING: prev threshold is too large to overwrite\n";
      }
      written++;
    }

    if (!written) {
      return matchFailure();
    } else {
      return matchSuccess();
    }
  }
};

template<typename TyOp>
struct BackendOverwriteThresholdDefaultPattern : public RewritePattern {
  BackendOverwriteThresholdDefaultPattern(MLIRContext *context)
      : RewritePattern(TyOp::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    float threshold_y = getOpThreshold(op);
    auto formerOp = op->getOperand(0)->getDefiningOp();
    float threshold_x = getPreviousOpThreshold(op);
    if (threshold_x == threshold_y) {
      return matchFailure();
    }
    if (auto cast_op = llvm::dyn_cast_or_null<tpu::BroadcastMulOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::ConcatOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::Conv2DOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::CropOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::DeConv2DOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::EltwiseAddOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::EltwiseMaxOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::EltwiseMulOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::FullyConnectedOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::LeakyReluOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::ReluOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::PermuteOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::ShuffleChannelOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::PermuteOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::PReluOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op =
                   llvm::dyn_cast_or_null<tpu::ShuffleChannelOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::ScaleOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op =
                   llvm::dyn_cast_or_null<tpu::UpsampleOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else {
      llvm::errs() << formerOp->getName() << ": behavior not defined\n";
      assert(false);
    }
    llvm::errs() << op->getName() << " [" << getOpName(op) << "] set prev "
                 << formerOp->getName() << " ["<< getOpName(formerOp) << "], "
                 "threshold from "
                 << std::to_string(threshold_x) << " to "
                 << std::to_string(threshold_y) << "\n";
    if (threshold_x < threshold_y * 0.5) {
      llvm::errs() << "  WARNING: prev threshold is too small to overwrite\n";
    }
    if (threshold_x > threshold_y * 2.0) {
      llvm::errs() << "  WARNING: prev threshold is too large to overwrite\n";
    }

    return matchSuccess();
  }
};

/// overwrite current Op threshold by prev Op threshold
/// for layers that we are not going to handle the threshold difference,
/// like max pooling, upsample, permute, crop.
template<typename TyOp>
struct ForwardOverwriteThresholdDefaultPattern : public RewritePattern {
  ForwardOverwriteThresholdDefaultPattern(MLIRContext *context)
      : RewritePattern(TyOp::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    auto op = cast<TyOp>(opInst);
    float threshold_x = getPreviousOpThreshold(op);
    float threshold_y = getOpThreshold(op);
    if (threshold_y == threshold_x) {
      // overwritten already
      return matchFailure();
    } else {
      setOpThreshold(opInst, threshold_x);
      llvm::errs() << opInst->getName() << " [" << op.name() << "] "
                   << "set threshold by prev Op threshold, from "
                   << std::to_string(threshold_y) << " to "
                   << std::to_string(threshold_x) << "\n";
      return matchSuccess();
    }
  }
};

/// bypass threshold from prev Op to current Op
/// for layers that has no threshold values, like slice
template<typename TyOp>
struct BypassThresholdDefaultPattern : public RewritePattern {
  BypassThresholdDefaultPattern(MLIRContext *context)
      : RewritePattern(TyOp::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    auto op = cast<TyOp>(opInst);
    if (getOpQuantParamType(op) == "THRESHOLD") {
      float threshold_x = getPreviousOpThreshold(opInst);
      float threshold_y = getOpThreshold(opInst);
      if (threshold_y == threshold_x) {
        // assigned already
        return matchFailure();
      }
    }

    /// be careful about call sequence
    /// since this is assuming previous Op has threshold_y already
    float threshold_x = getPreviousOpThreshold(op);
    setOpThreshold(opInst, threshold_x);
    setOpQuantParamType(op, "THRESHOLD");
    llvm::errs() << opInst->getName() << " [" << op.name() << "] "
                 << "set threshold by prev Op threshold "
                 << std::to_string(threshold_x) << "\n";
    return matchSuccess();
  }
};

/// force threshold for some Ops
template<typename TyOp>
struct ForceThresholdDefaultPattern : public RewritePattern {
  ForceThresholdDefaultPattern(MLIRContext *context, float threshold)
      : RewritePattern(TyOp::getOperationName(), 1, context),
        threshold_(threshold) {}

  PatternMatchResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    if (getOpQuantParamType(op) == "THRESHOLD") {
      if (getOpThreshold(op) == threshold_) {
        // assigned already
        return matchFailure();
      }
    }
    setOpThreshold(op, threshold_);
    setOpQuantParamType(op, "THRESHOLD");
    llvm::errs() << op->getName() << " [" << getOpName(op) << "] "
                 << "force threshold to "
                 << std::to_string(threshold_) << "\n";
    return matchSuccess();
  }

private:
  float threshold_;
};

/// force threshold for multiply operations
/// like eltwise_mul, broadcast_mul
template<typename TyOp>
struct ForceThresholdMulOpPattern : public RewritePattern {
  ForceThresholdMulOpPattern(MLIRContext *context)
      : RewritePattern(TyOp::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    auto op = cast<TyOp>(opInst);

    assert(opInst->getNumOperands() == 6);

    // skip pow() case
    if (opInst->getOperand(0)->getDefiningOp()
        == opInst->getOperand(1)->getDefiningOp()) {
      return matchFailure();
    }

    float threshold_x0 = getPreviousOpThreshold(opInst, 0);
    float threshold_x1 = getPreviousOpThreshold(opInst, 1);
    float threshold_y = getOpThreshold(opInst);
    float new_threshold_y = threshold_x0 * threshold_x1;

    // for swish, do backward overwrite
    //if (isa<tpu::EltwiseMulOp>(opInst)) {
    //  assert(isa<tpu::Conv2DOp>(opInst->getOperand(0)->getDefiningOp()));
    //  assert(isa<tpu::SigmoidOp>(opInst->getOperand(1)->getDefiningOp()));
    //  setOpThreshold(opInst->getOperand(0)->getDefiningOp(),
    //                 threshold_y);
    //  new_threshold_y = threshold_y;
    //}

    // for EltwiseMulOp, handle swish case only, other case needs more tuning
    if (isa<tpu::EltwiseMulOp>(opInst)) {
      bool is_swish = false;
      if (isa<tpu::Conv2DOp>(opInst->getOperand(0)->getDefiningOp())
          && isa<tpu::SigmoidOp>(opInst->getOperand(1)->getDefiningOp())) {
        is_swish = true;
      }
      if (!is_swish) {
        return matchFailure();
      }
    }

    if (getOpQuantParamType(op) == "THRESHOLD") {
      if (threshold_y == new_threshold_y) {
        // assigned already
        return matchFailure();
      }
    }


    setOpThreshold(opInst, new_threshold_y);
    setOpQuantParamType(op, "THRESHOLD");
    llvm::errs() << opInst->getName() << " [" << op.name() << "] "
                 << "set threshold by multiply threshold "
                 << std::to_string(new_threshold_y) << "\n";

    // for broadcast mul, update the ops after as well
    //if (isa<tpu::BroadcastMulOp>(opInst)) {
    //  float next_scale = sqrt(new_threshold_y / threshold_y);
    //  for (auto &use : opInst->getResult(0)->getUses()) {
    //    auto next_op = use.getOwner();
    //    float next_threashold_y = getOpThreshold(next_op);
    //    setOpThreshold(next_op, next_threashold_y * next_scale);
    //  }
    //}

    return matchSuccess();
  }
};

class ImportCalibrationTablePass : public FunctionPass<ImportCalibrationTablePass> {
public:
  explicit ImportCalibrationTablePass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    // load the table
    // Support symmetric quantization only
    std::map<std::string, float> threshold_map;
    os << "Calibration Table File : " << clCalibrationTableFilename << "\n";
    std::ifstream infile(clCalibrationTableFilename);
    std::string line;
    std::regex sym_pattern("[a-zA-Z0-9._/-]+ [-0-9.e]+");
    std::regex asym_pattern("[a-zA-Z0-9._/-]+ [-0-9.e]+ [-0-9.e]+");
    while (std::getline(infile, line)) {
      std::istringstream iss(line);
      std::string name;
      if (std::regex_match(line, sym_pattern)) {
        float threshold;
        if (!(iss >> name >> threshold)) { break; }
        llvm::errs() << "  name " << name << ", threshold "
                     << std::to_string(threshold) << "\n";
        threshold_map[name] = threshold;
      } else if (std::regex_match(line, asym_pattern)) {
        float min_threshold, max_threshold;
        if (!(iss >> name >> min_threshold >> max_threshold)) { break; }
          llvm::errs() << "  name " << name << ", min_threshold = "
                     << std::to_string(min_threshold) << ", max_threshold = "
                     << std::to_string(max_threshold) << "\n";
        // Not Support asymmetric quantization so far
        assert(false);
      } else {
        // Format of threshold table error
        llvm::errs() << line;
        llvm::errs() << "\n  => not match required format\n";
        assert(false);
      }
    }

    auto *context = &getContext();
    Builder builder(context);
    // assign the threshold_y to each op
    fn.walk([&](Operation *op) {
      os << op->getName() << "\n";

      if (op->getName().getDialect().str() != "tpu"
          || isa<tpu::WeightFileOp>(op)
          || isa<tpu::LoadWeightOp>(op)
          || isa<tpu::NoneOp>(op)) {
        // no need to assign
      } else if (isa<tpu::ReshapeOp>(op)) {
        // do not assign
      } else if (isa<tpu::SliceOp>(op)){
        // do not assign
      } else if ( !failed(setThresholdFromMap(op, threshold_map))) {
        // success
      } else if (isa<tpu::SoftmaxOp>(op)
                 || isa<tpu::DetectionOutputOp>(op)
                 || isa<tpu::RetinaFaceDetectionOp>(op)) {
        // doesn't matter assigned or not
      } else {
        llvm::errs() << "setThresholdFromMap didn't handle "
                     << op->getName() << "\n";
        assert(false);
      }
    });

    OwningRewritePatternList patterns;
    // apply force threshold to some ops
    //   SigmoidOp force to 1.0
    patterns.insert<
        ForceThresholdDefaultPattern<tpu::SigmoidOp>
        >(context, 1.0f);
    applyPatternsGreedily(fn, patterns);

    // apply default bypass for the ops that has no calibration threshold
    llvm::errs() << "Forword set bypass Ops threshold\n";
    patterns.clear();
    patterns.insert<
        BypassThresholdDefaultPattern<tpu::SliceOp>
        >(context);
    applyPatternsGreedily(fn, patterns);

    // apply multiply for mul ops
    llvm::errs() << "Force multiply Ops thresholds\n";
    patterns.clear();
    patterns.insert<
        ForceThresholdMulOpPattern<tpu::BroadcastMulOp>,
        ForceThresholdMulOpPattern<tpu::EltwiseMulOp>
        >(context);
    applyPatternsGreedily(fn, patterns);

    if (clCaliOverwriteThresholdBackwardRelu) {
      assert(!clCaliOverwriteThresholdForwardRelu);
      llvm::errs() << "Backward overwrite threshold for all\n";
      patterns.clear();
      patterns.insert<
          BackendOverwriteThresholdDefaultPattern<tpu::LeakyReluOp>,
          BackendOverwriteThresholdDefaultPattern<tpu::ReluOp>,
          BackendOverwriteThresholdDefaultPattern<tpu::UpsampleOp>,
          BackendOverwriteThresholdDefaultPattern<tpu::PermuteOp>,
          BackendOverwriteThresholdDefaultPattern<tpu::CropOp>,
          BackendOverwriteThresholdDefaultPattern<tpu::PoolMax2DOp>
          >(context);
      if (clCaliOverwriteThresholdBackwardConcat) {
        llvm::errs() << "Backward overwrite threshold for concat\n";
        patterns.insert<
            BackwardOverwriteThresholdConcatPattern
            >(context);
      }
      applyPatternsGreedily(fn, patterns);
    }
    if (clCaliOverwriteThresholdForwardRelu) {
      assert(!clCaliOverwriteThresholdBackwardRelu);
      llvm::errs() << "Forward overwrite threshold for all\n";
      patterns.clear();
      patterns.insert<
          ForwardOverwriteThresholdDefaultPattern<tpu::LeakyReluOp>,
          ForwardOverwriteThresholdDefaultPattern<tpu::ReluOp>,
          ForwardOverwriteThresholdDefaultPattern<tpu::UpsampleOp>,
          ForwardOverwriteThresholdDefaultPattern<tpu::PermuteOp>,
          ForwardOverwriteThresholdDefaultPattern<tpu::CropOp>,
          ForwardOverwriteThresholdDefaultPattern<tpu::PoolMax2DOp>
          >(context);
      applyPatternsGreedily(fn, patterns);
    }

    if (!clCaliOverwriteThresholdBackwardRelu
        && !clCaliOverwriteThresholdForwardRelu) {
      llvm::errs() << "Default backward overwrite\n";
      patterns.clear();
      patterns.insert<
          BackendOverwriteThresholdDefaultPattern<tpu::UpsampleOp>,
          BackendOverwriteThresholdDefaultPattern<tpu::PermuteOp>,
          BackendOverwriteThresholdDefaultPattern<tpu::CropOp>,
          BackendOverwriteThresholdDefaultPattern<tpu::PoolMax2DOp>
          >(context);
      applyPatternsGreedily(fn, patterns);
    }
  }

private:

  LogicalResult setThresholdFromMap(Operation *op,
      std::map<std::string, float> &threshold_map) {
    if (mlir::getOpName(op).empty()) {
      return failure();
    }

    std::string op_name = mlir::getOpName(op).str();
    if (threshold_map.find(op_name) == threshold_map.end()) {
      return failure();
    }

    auto tpuOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(op);
    if (!tpuOp) {
      return failure();
    }

    float threshold = threshold_map[op_name];
    auto ret = setOpThreshold(op, threshold);
    if (!failed(ret)) {
      setOpQuantParamType(op, "THRESHOLD");
      os << "  > " << op_name << ", " << std::to_string(threshold) << "\n";
    }
    return ret;
  }

  llvm::raw_ostream &os;
};

} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createImportCalibrationTablePass() {
  return std::make_unique<ImportCalibrationTablePass>();
}

static PassRegistration<ImportCalibrationTablePass>
    pass("import-calibration-table",
         "Import calibration table from external tools");
