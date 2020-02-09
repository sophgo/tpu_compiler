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
using namespace mlir;

static llvm::cl::OptionCategory clOptionsCategory("calibration table options");

static llvm::cl::opt<std::string> clCalibrationTableFilename(
    "calibration-table",
    llvm::cl::desc("Specify the calibration table filename"),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<bool> clCaliOverwriteThresholdForwardRelu(
    "enable-cali-overwrite-threshold-forward-relu",
    llvm::cl::desc("Overwrite threshold value for relu with prev Op's threshold"),
    llvm::cl::cat(clOptionsCategory), llvm::cl::init(false));

/// use relu-overwrite-backward as default
static llvm::cl::opt<bool> clCaliOverwriteThresholdBackwardRelu(
    "enable-cali-overwrite-threshold-backward-relu",
    llvm::cl::desc("Overwrite threshold value for the Op prev to relu "
                   "with the relu's threshold"),
    llvm::cl::cat(clOptionsCategory), llvm::cl::init(true));

static llvm::cl::opt<bool> clCaliOverwriteThresholdBackwardConcat(
    "enable-cali-overwrite-threshold-backward-concat",
    llvm::cl::desc("Overwrite threshold value for the Ops prev to concat "
                   "with the concat's threhold, and backpropagate result "
                   "threshold to operand ops"),
    llvm::cl::cat(clOptionsCategory), llvm::cl::init(false));

static llvm::cl::opt<bool> clCaliOverwriteThresholdMaxConcat(
    "enable-cali-overwrite-threshold-max-concat",
    llvm::cl::desc("Overwrite threshold value for concat and the Ops prev to "
                   "concat, by find the max threshold value along the chain"),
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
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto formerOp = op.getOperand(i)->getDefiningOp();
      float threshold_x = getOpThreshold(formerOp);
      if (threshold_x == threshold_y) {
          continue;
      }
      if (auto cast_op = llvm::dyn_cast_or_null<tpu::Conv2DOp>(formerOp)) {
        cast_op.setAttr("threshold_y", rewriter.getF32FloatAttr(threshold_y));
      } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::UpsampleOp>(formerOp)) {
        cast_op.setAttr("threshold_y", rewriter.getF32FloatAttr(threshold_y));
      } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::EltwiseOp>(formerOp)) {
        cast_op.setAttr("threshold_y", rewriter.getF32FloatAttr(threshold_y));
      } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::ReluOp>(formerOp)) {
        cast_op.setAttr("threshold_y", rewriter.getF32FloatAttr(threshold_y));
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

struct BackwardOverwriteThresholdReluPattern : public OpRewritePattern<tpu::ReluOp> {
  using OpRewritePattern<tpu::ReluOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(tpu::ReluOp op,
                                     PatternRewriter &rewriter) const {
    float threshold_y = getOpThreshold(op);
    auto formerOp = op.getOperand(0)->getDefiningOp();
    float threshold_x = getPreviousOpThreshold(op, 0);
    if (threshold_x == threshold_y) {
      return matchFailure();
    }
    if (auto cast_op = llvm::dyn_cast_or_null<tpu::Conv2DOp>(formerOp)) {
      cast_op.setAttr("threshold_y", rewriter.getF32FloatAttr(threshold_y));
    } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::FullyConnectedOp>(formerOp)) {
      cast_op.setAttr("threshold_y", rewriter.getF32FloatAttr(threshold_y));
    } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::ConcatOp>(formerOp)) {
      cast_op.setAttr("threshold_y", rewriter.getF32FloatAttr(threshold_y));
    } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::ScaleOp>(formerOp)) {
      cast_op.setAttr("threshold_y", rewriter.getF32FloatAttr(threshold_y));
    } else {
      llvm::errs() << formerOp->getName() << ": behavior not defined\n";
      assert(false);
    }
    llvm::errs() << "Relu set prev " << formerOp->getName()
                 << " ["<< getOpName(formerOp) << "] threshold, from "
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

struct BackwardOverwriteThresholdUpsamplePattern : public OpRewritePattern<tpu::UpsampleOp> {
  using OpRewritePattern<tpu::UpsampleOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(tpu::UpsampleOp op,
                                     PatternRewriter &rewriter) const {
    float threshold_y = getOpThreshold(op);
    auto formerOp = op.getOperand()->getDefiningOp();
    float threshold_x = getPreviousOpThreshold(op, 0);
    if (threshold_x == threshold_y) {
      return matchFailure();
    }
    if (auto cast_op = llvm::dyn_cast_or_null<tpu::Conv2DOp>(formerOp)) {
      cast_op.setAttr("threshold_y", rewriter.getF32FloatAttr(threshold_y));
    } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::ReluOp>(formerOp)) {
      cast_op.setAttr("threshold_y", rewriter.getF32FloatAttr(threshold_y));
    } else {
      llvm::errs() << formerOp->getName() << ": behavior not defined\n";
      assert(false);
    }
    llvm::errs() << "Upsample set prev " << formerOp->getName()
                 << " ["<< getOpName(formerOp) << "] threshold, from "
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

/// Max forward concat threshold by setting result's threshold_y as the largest
/// operands threshold.
struct MaxOverwriteThresholdConcatPattern : public OpRewritePattern<tpu::ConcatOp> {
  using OpRewritePattern<tpu::ConcatOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(tpu::ConcatOp op,
                                     PatternRewriter &rewriter) const {
    float threshold_y = getOpThreshold(op);
    float max_threshold = getOpThreshold(op);
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      float threshold_x = getPreviousOpThreshold(op, i);
      if (threshold_x > max_threshold) {
        max_threshold = threshold_x;
      }
    }
    if (max_threshold > threshold_y) {
      op.setAttr("threshold_y", rewriter.getF32FloatAttr(max_threshold));
      return matchSuccess();
    } else {
      return matchFailure();
    }
  }
};

/// bypass pool quantization by assigning threshold_y same as threshold_x.
/// for max pooling, threshold_y is always larger than threshold_x,
/// however, if one value exceed the quantization range, it has been saturated
/// already, an extra multiplier does not help.
/// for avg pooling, threshold_y is smaller than threshold_x,
/// we shall keep the quantization (i.e. multiply with a multiplier)
struct ForwardOverwriteThresholdMaxPool2DPattern : public OpRewritePattern<tpu::Pool2DOp> {
  using OpRewritePattern<tpu::Pool2DOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(tpu::Pool2DOp op,
                                     PatternRewriter &rewriter) const {
    if (op.pool() != "MAX") {
      return matchFailure();
    }
    float threshold_x = getPreviousOpThreshold(op);
    float threshold_y = getOpThreshold(op);
    if (threshold_y > threshold_x) {
      op.setAttr("threshold_y", rewriter.getF32FloatAttr(threshold_x));
      return matchSuccess();
    } else {
      llvm::errs() << "MAX Pool2D [" << op.name() << "] "
                   << "set threshold by prev Op threshold, from "
                   << std::to_string(threshold_y) << " to "
                   << std::to_string(threshold_x) << "\n";
      return matchFailure();
    }
  }
};

/// overwrite current Op threshold by prev Op threshold
/// for layers that we are not going to handle the threshold difference,
/// like upsample, permute, crop.
/// Pool2D is handled separately, because only max pooling needs overwriting
template<typename TyOp>
struct ForwardOverwriteThresholdDefaultPattern : public RewritePattern {
  ForwardOverwriteThresholdDefaultPattern(MLIRContext *context)
      : RewritePattern(TyOp::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    auto op = cast<TyOp>(opInst);
    assert(op.threshold_y().hasValue());
    float threshold_x = getPreviousOpThreshold(op);
    float threshold_y = getOpThreshold(op);
    if (threshold_y == threshold_x) {
      // overwritten already
      return matchFailure();
    } else {
      op.setAttr("threshold_y", rewriter.getF32FloatAttr(threshold_x));
      llvm::errs() << opInst->getName() << " [" << op.name() << "] "
                   << "set threshold by prev Op threshold, from "
                   << std::to_string(threshold_y) << " to "
                   << std::to_string(threshold_x) << "\n";
      return matchSuccess();
    }
  }
};

/// bypass threshold from prev Op to current Op
/// for layers that has no threshold values, like reshape, slice, etc
template<typename TyOp>
struct BypassThresholdDefaultPattern : public RewritePattern {
  BypassThresholdDefaultPattern(MLIRContext *context)
      : RewritePattern(TyOp::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    auto op = cast<TyOp>(opInst);
    if (op.threshold_y().hasValue()) {
      // assigned already
      return matchFailure();
    } else {
      /// be careful about call sequence
      /// since this is assuming previous Op has threshold_y already
      float threshold_x = getPreviousOpThreshold(op);
      op.setAttr("threshold_y", rewriter.getF32FloatAttr(threshold_x));
      llvm::errs() << opInst->getName() << " [" << op.name() << "] "
                   << "set threshold by prev Op threshold "
                   << std::to_string(threshold_x) << "\n";
      return matchSuccess();
    }
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
    while (std::getline(infile, line)) {
      std::istringstream iss(line);
      std::string name;
      float threshold;
      if (!(iss >> name >> threshold)) { break; }
      llvm::errs() << "  name " << name << ", threshold "
                   << std::to_string(threshold) << "\n";
      threshold_map[name] = threshold;
    }

    auto *context = &getContext();
    Builder builder(context);
    // assign the threshold_y to each op
    fn.walk([&](Operation *op) {
      os << op->getName() << "\n";

      addThresholdAttr<tpu::InputOp>(builder, threshold_map, op);
      addThresholdAttr<tpu::BatchNormOp>(builder, threshold_map, op);
      addThresholdAttr<tpu::ConcatOp>(builder, threshold_map, op);
      addThresholdAttr<tpu::Conv2DOp>(builder, threshold_map, op);
      addThresholdAttr<tpu::DeConv2DOp>(builder, threshold_map, op);
      addThresholdAttr<tpu::CropOp>(builder, threshold_map, op);
      addThresholdAttr<tpu::DivOp>(builder, threshold_map, op);
      addThresholdAttr<tpu::EltwiseOp>(builder, threshold_map, op);
      addThresholdAttr<tpu::FullyConnectedOp>(builder, threshold_map, op);
      addThresholdAttr<tpu::Pool2DOp>(builder, threshold_map, op);
      addThresholdAttr<tpu::PowerOp>(builder, threshold_map, op);
      addThresholdAttr<tpu::PReluOp>(builder, threshold_map, op);
      addThresholdAttr<tpu::ReluOp>(builder, threshold_map, op);
      addThresholdAttr<tpu::ScaleOp>(builder, threshold_map, op);
      addThresholdAttr<tpu::SigmoidOp>(builder, threshold_map, op);
      addThresholdAttr<tpu::SoftmaxOp>(builder, threshold_map, op);
      addThresholdAttr<tpu::SqrtOp>(builder, threshold_map, op);
      addThresholdAttr<tpu::UpsampleOp>(builder, threshold_map, op);
      addThresholdAttr<tpu::PermuteOp>(builder, threshold_map, op);
    });

    // apply default bypass
    OwningRewritePatternList patterns_1;
    patterns_1.insert<BypassThresholdDefaultPattern<tpu::ReshapeOp>,
                      BypassThresholdDefaultPattern<tpu::SliceOp>
                     >(context);
    applyPatternsGreedily(fn, patterns_1);

    // apply default forward overwrite
    OwningRewritePatternList patterns_2;
    patterns_2.insert<ForwardOverwriteThresholdDefaultPattern<tpu::UpsampleOp>,
                      ForwardOverwriteThresholdDefaultPattern<tpu::PermuteOp>,
                      ForwardOverwriteThresholdDefaultPattern<tpu::CropOp>,
                      ForwardOverwriteThresholdMaxPool2DPattern
                     >(context);
    applyPatternsGreedily(fn, patterns_2);

    if (clCaliOverwriteThresholdBackwardRelu) {
      llvm::errs() << "Backward overwrite threshold for relu\n";
      assert(!clCaliOverwriteThresholdForwardRelu);
      OwningRewritePatternList patterns_relu;
      patterns_relu.insert<BackwardOverwriteThresholdReluPattern>(context);
      applyPatternsGreedily(fn, patterns_relu);
    }
    if (clCaliOverwriteThresholdForwardRelu) {
      llvm::errs() << "Forward overwrite threshold for relu\n";
      assert(!clCaliOverwriteThresholdBackwardRelu);
      OwningRewritePatternList patterns_relu;
      patterns_relu.insert<ForwardOverwriteThresholdDefaultPattern<tpu::ReluOp>
                          >(context);
      applyPatternsGreedily(fn, patterns_relu);
    }

    if (clCaliOverwriteThresholdBackwardConcat) {
      llvm::errs() << "Backward overwrite threshold for concat\n";
      OwningRewritePatternList patterns_bp;
      patterns_bp.insert<BackwardOverwriteThresholdConcatPattern,
                         BackwardOverwriteThresholdReluPattern,
                         BackwardOverwriteThresholdUpsamplePattern
                        >(context);
      applyPatternsGreedily(fn, patterns_bp);
    } else if (clCaliOverwriteThresholdMaxConcat) {
      llvm::errs() << "Max overwrite threshold for concat\n";
      // forward bypass and max forward propergate first
      OwningRewritePatternList patterns_mx;
      patterns_mx.insert<MaxOverwriteThresholdConcatPattern>(context);
      applyPatternsGreedily(fn, patterns_mx);

      // then backpropagate the selected max threshold
      OwningRewritePatternList patterns_bp;
      patterns_bp.insert<BackwardOverwriteThresholdConcatPattern,
                         BackwardOverwriteThresholdReluPattern,
                         BackwardOverwriteThresholdUpsamplePattern
                        >(context);
      applyPatternsGreedily(fn, patterns_bp);
    }
  }

private:
  template<typename T>
  void addThresholdAttr(Builder &builder, std::map<std::string, float> &threshold_map,
      Operation *op) {
      auto cast_op = llvm::dyn_cast_or_null<T>(op);
      if (cast_op) {
        assert(cast_op.name().hasValue());
        std::string op_name = cast_op.name().getValue().str();
        float threshold;
        // FIXME: sometimes these is no softmax threshold present, use 1.0
        if (llvm::dyn_cast_or_null<tpu::SoftmaxOp>(op) && !threshold_map[op_name]) {
          threshold = 1.0f;
        } else {
          assert(threshold_map[op_name]);
          threshold = threshold_map[op_name];
        }
        os << " > " << op_name << ", " << std::to_string(threshold) << "\n";
        cast_op.setAttr("threshold_y", builder.getF32FloatAttr(threshold));
      }
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
