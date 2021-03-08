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
#include "tpuc/Passes.h"
#include "tpuc/TPUOperationSupport.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "tpuc/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

#include <sstream>
#include <fstream>
#include <regex>
#include <unordered_map>

#define DEBUG_TYPE "import_calibration_table"

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

  LogicalResult matchAndRewrite(tpu::ConcatOp op,
                                     PatternRewriter &rewriter) const {
    float threshold_y = getOpThreshold(op);
    int written = 0;
    unsigned nInputs = op.getNumInputs();
    for (unsigned i = 0; i < nInputs; ++i) {
      auto formerOp = op.getOperand(i).getDefiningOp();
      float threshold_x = getOpThreshold(formerOp);
      if (threshold_x == threshold_y) {
          continue;
      }
      if (auto cast_op = llvm::dyn_cast_or_null<tpu::AbsOp>(formerOp)) {
        setOpThreshold(formerOp, threshold_y);
      } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::BroadcastMulOp>(formerOp)) {
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
      } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::EltwiseMinOp>(formerOp)) {
        setOpThreshold(formerOp, threshold_y);
      }  else if (auto cast_op = llvm::dyn_cast_or_null<tpu::EltwiseMulOp>(formerOp)) {
        setOpThreshold(formerOp, threshold_y);
      } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::LeakyReluOp>(formerOp)) {
        setOpThreshold(formerOp, threshold_y);
      } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::CustomOp>(formerOp)) {
        setOpThreshold(formerOp, threshold_y);
      } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::ReluOp>(formerOp)) {
        setOpThreshold(formerOp, threshold_y);
      } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::UpsampleOp>(formerOp)) {
        setOpThreshold(formerOp, threshold_y);
      } else {
        llvm::errs() << formerOp->getName() << ": behavior not defined\n";
        assert(false);
      }
      LLVM_DEBUG(
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
      );
      written++;
    }

    if (!written) {
      return failure();
    } else {
      return success();
    }
  }
};

template<typename TyOp>
struct BackendOverwriteThresholdDefaultPattern : public RewritePattern {
  BackendOverwriteThresholdDefaultPattern(MLIRContext *context)
      : RewritePattern(TyOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    float threshold_y = getOpThreshold(op);
    auto formerOp = op->getOperand(0).getDefiningOp();
    float threshold_x = getPreviousOpThreshold(op);
    if (threshold_x == threshold_y) {
      return failure();
    }
    if (auto cast_op = llvm::dyn_cast_or_null<tpu::ReshapeOp>(formerOp)) {
      return failure(); // we skip reshape
    }
    if (auto cast_op = llvm::dyn_cast_or_null<tpu::BroadcastMulOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::ConcatOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::Conv2DOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::CropOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op =
                   llvm::dyn_cast_or_null<tpu::DeConv2DOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op =
                   llvm::dyn_cast_or_null<tpu::EltwiseAddOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op =
                   llvm::dyn_cast_or_null<tpu::EltwiseMaxOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op =
                   llvm::dyn_cast_or_null<tpu::EltwiseMinOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op =
                   llvm::dyn_cast_or_null<tpu::EltwiseMulOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op =
                   llvm::dyn_cast_or_null<tpu::FullyConnectedOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::InputOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op =
                   llvm::dyn_cast_or_null<tpu::LeakyReluOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::CustomOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::ReluOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::PadOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op =
                   llvm::dyn_cast_or_null<tpu::PoolMax2DOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op =
                   llvm::dyn_cast_or_null<tpu::PermuteOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::PReluOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::ScaleOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op =
                   llvm::dyn_cast_or_null<tpu::SigmoidOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op =
                   llvm::dyn_cast_or_null<tpu::UpsampleOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op =
                   llvm::dyn_cast_or_null<tpu::PoolAvg2DOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::CropOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::LrnOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else if (auto cast_op = llvm::dyn_cast_or_null<tpu::ExpOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);
    } else {
      llvm::errs() << formerOp->getName() << ": behavior not defined\n";
      assert(false);
    }
    LLVM_DEBUG(
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
    );

    return success();
  }
};

/// overwrite current Op threshold by prev Op threshold
/// for layers that we are not going to handle the threshold difference,
/// like max pooling, upsample, permute, crop.
template<typename TyOp>
struct ForwardOverwriteThresholdDefaultPattern : public RewritePattern {
  ForwardOverwriteThresholdDefaultPattern(MLIRContext *context)
      : RewritePattern(TyOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    auto op = cast<TyOp>(opInst);
    float threshold_x = getPreviousOpThreshold(op);
    float threshold_y = getOpThreshold(op);
    if (threshold_y == threshold_x) {
      // overwritten already
      return failure();
    } else {
      setOpThreshold(opInst, threshold_x);
      LLVM_DEBUG(llvm::errs() << opInst->getName() << " [" << op.name() << "] "
                   << "set threshold by prev Op threshold, from "
                   << std::to_string(threshold_y) << " to "
                   << std::to_string(threshold_x) << "\n";);
      return success();
    }
  }
};

/// bypass threshold from prev Op to current Op
/// for layers that has no threshold values, like slice
template<typename TyOp>
struct BypassThresholdDefaultPattern : public RewritePattern {
  BypassThresholdDefaultPattern(MLIRContext *context)
      : RewritePattern(TyOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    auto op = cast<TyOp>(opInst);
    if (getOpQuantParamType(op) == "THRESHOLD") {
      float threshold_x = getPreviousOpThreshold(opInst);
      float threshold_y = getOpThreshold(opInst);
      if (threshold_y == threshold_x) {
        // assigned already
        return failure();
      }
    }

    /// be careful about call sequence
    /// since this is assuming previous Op has threshold_y already
    float threshold_x = getPreviousOpThreshold(op);
    setOpThreshold(opInst, threshold_x);
    setOpQuantParamType(op, "THRESHOLD");
    LLVM_DEBUG(llvm::errs() << opInst->getName() << " [" << op.name() << "] "
                 << "set threshold by prev Op threshold "
                 << std::to_string(threshold_x) << "\n";);
    return success();
  }
};

/// force threshold for some Ops
template<typename TyOp>
struct ForceThresholdDefaultPattern : public RewritePattern {
  ForceThresholdDefaultPattern(MLIRContext *context, float threshold)
      : RewritePattern(TyOp::getOperationName(), 1, context),
        threshold_(threshold) {}

  LogicalResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    if (getOpQuantParamType(op) == "THRESHOLD") {
      if (getOpThreshold(op) == threshold_) {
        // assigned already
        return failure();
      }
    }
    setOpThreshold(op, threshold_);
    setOpQuantParamType(op, "THRESHOLD");
    LLVM_DEBUG(llvm::errs() << op->getName() << " [" << getOpName(op) << "] "
                 << "force threshold to "
                 << std::to_string(threshold_) << "\n";);
    return success();
  }

private:
  float threshold_;
};


/// force threshold for clip operations
/// use for rel6(relu + clip[0, 6])
/// relu will be fused in conv, we overwrite clip to this conv

template<typename TyOp>
struct ForceThresholdClipOpPattern : public RewritePattern {
  ForceThresholdClipOpPattern(MLIRContext *context)
      : RewritePattern(TyOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    float threshold_y;
    float threshold_x;

    if (isa<tpu::ClipOp>(opInst)) {
      threshold_x = getPreviousOpThreshold(opInst, 0);
      threshold_y = getOpThreshold(opInst);

    } else {
      return failure();
    }

    auto formerOp = opInst->getOperand(0).getDefiningOp();
    if (auto cast_op = llvm::dyn_cast_or_null<tpu::Conv2DOp>(formerOp)) {
      setOpThreshold(formerOp, threshold_y);

    } else {
      return failure();
    }

    setOpQuantParamType(opInst, "THRESHOLD");
    LLVM_DEBUG(llvm::errs()
                   << opInst->getName() << " [" << getOpName(opInst) << "] set prev "
                   << formerOp->getName() << " [" << getOpName(formerOp)
                   << "], "
                      "threshold from "
                   << std::to_string(threshold_x) << " to "
                   << std::to_string(threshold_y) << "\n";);

    // remove clip op
    rewriter.replaceOp(opInst, {opInst->getOperand(0)});
    return success();
  }
};

template<typename TyOp>
struct ForceThresholdCustomOpPattern : public RewritePattern {
  ForceThresholdCustomOpPattern(MLIRContext *context)
      : RewritePattern(TyOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto castOp = cast<TyOp>(op);
    auto overwrite = castOp.threshold_overwrite();
    llvm::errs() << "custom op overwrite:" << overwrite << "\n";
    if (overwrite == "none") {
      return failure();
    }

    float threshold_x = getPreviousOpThreshold(op);
    float threshold_y = getOpThreshold(op);

    if (getOpQuantParamType(op) == "THRESHOLD") {
      if (threshold_y == threshold_x) {
        // assigned already
        return failure();
      }
    }

    /// be careful about call sequence
    /// since this is assuming previous Op has threshold_y already
    if (overwrite == "forward") {
      setOpThreshold(op, threshold_x);
    } else if (overwrite == "backward") { // "backward"
      auto formerOp = op->getOperand(0).getDefiningOp();
      setOpThreshold(formerOp, threshold_y);
    }
    setOpQuantParamType(op, "THRESHOLD");
    return success();
  }
};

class ImportCalibrationTablePass : public mlir::PassWrapper<ImportCalibrationTablePass, FunctionPass> {
public:
  explicit ImportCalibrationTablePass() {}

  void runOnFunction() override {
    auto fn = getFunction();

    // load the table
    // Support symmetric quantization only
    std::map<std::string, float> threshold_map;
    std::unordered_map<std::string, std::vector<float>> weight_threshold_map;
    llvm::errs() << "Calibration Table File : " << clCalibrationTableFilename << "\n";
    std::ifstream infile(clCalibrationTableFilename);
    std::string line;
    std::regex old_pattern("[a-zA-Z0-9.:;@_\\/-]+ [-0-9.e]+");
    std::regex new_pattern(
        "[a-zA-Z0-9.:;@_\\/-]+ [-0-9.e]+ [-0-9.e]+ [-0-9.e]+");
    std::regex weight_pattern("weight [a-zA-Z0-9.:;@_\\/-]+ .*");
    std::regex info_pattern("#.*");
    while (std::getline(infile, line)) {
      std::istringstream iss(line);
      std::string name;
      if (std::regex_match(line, old_pattern)) {
        float threshold;
        if (!(iss >> name >> threshold)) { break; }
        LLVM_DEBUG(llvm::errs() << "  name " << name << ", threshold "
                     << std::to_string(threshold) << "\n";);
        threshold_map[name] = threshold;
      } else if (std::regex_match(line, new_pattern)) {
        float threshold, min_value, max_value;
        if (!(iss >> name >> threshold >> min_value >> max_value)) {
          break;
        }
        LLVM_DEBUG(llvm::errs() << "  name " << name
                                << ", threshold = " << std::to_string(threshold)
                                << ", min_value = " << std::to_string(min_value)
                                << ", max_value = " << std::to_string(max_value)
                                << "\n";);
        threshold_map[name] = threshold;

      } else if (std::regex_match(line, info_pattern)) {
        llvm::errs() << "\n  infomation  " << line << "\n";
      }else if(std::regex_match(line, weight_pattern)){
        std::vector<float> weight_threshold;
        iss.ignore(256, ' '); // skip "weight"
        iss >> name;
        float threshold;
        while (iss >> threshold) {
          weight_threshold.push_back(threshold);
          LLVM_DEBUG(llvm::errs() << "  name " << name << " weight threshold: " << threshold << "\n";);
        }
        weight_threshold_map[name] = weight_threshold;
      } else {
        // Format of threshold table error
        llvm::errs() << line;
        llvm_unreachable("\n  => not match required format\n");
      }
    }

    auto *context = &getContext();
    Builder builder(context);
    // assign the threshold_y to each op
    fn.walk([&](Operation *op) {
      LLVM_DEBUG(llvm::errs() << op->getName() << "\n";);

      if (op->getName().getDialect()->getNamespace() != "tpu" ||
          isa<tpu::WeightFileOp>(op) ||
          isa<tpu::NoneOp>(op)||
          isa<tpu::ReorgOp>(op) ||
          isa<tpu::ReshapeOp>(op) ||
          isa<tpu::ReverseOp>(op) ||
          isa<tpu::SliceOp>(op) ||
          isa<tpu::ShuffleChannelOp>(op) ||
          isa<tpu::SwapChannelOp>(op) ||
          isa<tpu::UpsampleOp>(op) ||
          isa<tpu::CscOp>(op) ||
          isa<tpu::ZeroMaskOp>(op)) {
      } else if(isa<tpu::LoadWeightOp>(op)) {
        setWeightThresholdFromMap(op, weight_threshold_map);
        // do not assign
      } else if (!failed(setThresholdFromMap(op, threshold_map))) {
        // success
      } else if (isa<tpu::DetectionOutputOp>(op)
                 || isa<tpu::FrcnDetectionOp>(op)
                 || isa<tpu::InputOp>(op)
                 || isa<tpu::ProposalOp>(op)
                 || isa<tpu::RetinaFaceDetectionOp>(op)
                 || isa<tpu::SoftmaxOp>(op)
                 || isa<tpu::SoftmaxCpuOp>(op)
                 || isa<tpu::SquareOp>(op)
                 || isa<tpu::QuadraticSumOp>(op)
                 || isa<tpu::TransposeOp>(op)
                 || isa<tpu::YoloDetectionOp>(op)
                 /*|| isa<tpu::CustomOp>(op)*/) {
        // doesn't matter assigned or not
      } else {
        std::string op_name = mlir::getOpName(op).str();
        llvm::errs() << "setThresholdFromMap didn't handle op " << op->getName()
                     << " layer name " << op_name << "\n";
        assert(false);
      }
    });

    OwningRewritePatternList patterns;

    // apply default bypass for the ops that has no calibration threshold
    LLVM_DEBUG(llvm::errs() << "Forword set bypass Ops threshold\n";);
    patterns.clear();
    patterns.insert<
        BypassThresholdDefaultPattern<tpu::ReorgOp>,
        BypassThresholdDefaultPattern<tpu::PadOp>,
        BypassThresholdDefaultPattern<tpu::PixelShuffleOp>,
        BypassThresholdDefaultPattern<tpu::ReverseOp>,
        BypassThresholdDefaultPattern<tpu::SliceOp>,
        BypassThresholdDefaultPattern<tpu::ShuffleChannelOp>,
        BypassThresholdDefaultPattern<tpu::SwapChannelOp>,
        BypassThresholdDefaultPattern<tpu::ReduceMaxOp>,
        BypassThresholdDefaultPattern<tpu::TileOp>,
        BypassThresholdDefaultPattern<tpu::UpsampleOp>,
        BypassThresholdDefaultPattern<tpu::CscOp>,
        BypassThresholdDefaultPattern<tpu::ZeroMaskOp>
        >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));

    if (clCaliOverwriteThresholdBackwardRelu) {
      //assert(!clCaliOverwriteThresholdForwardRelu);
      LLVM_DEBUG(llvm::errs() << "Backward overwrite threshold for all\n";);
      patterns.clear();
      patterns.insert<
          BackendOverwriteThresholdDefaultPattern<tpu::AbsOp>,
          BackendOverwriteThresholdDefaultPattern<tpu::LeakyReluOp>,
          BackendOverwriteThresholdDefaultPattern<tpu::ReluOp>,
          BackendOverwriteThresholdDefaultPattern<tpu::PadOp>,
          BackendOverwriteThresholdDefaultPattern<tpu::UpsampleOp>,
          BackendOverwriteThresholdDefaultPattern<tpu::PermuteOp>,
          BackendOverwriteThresholdDefaultPattern<tpu::CropOp>,
          BackendOverwriteThresholdDefaultPattern<tpu::PoolMax2DOp>
          >(context);
      if (clCaliOverwriteThresholdBackwardConcat) {
        LLVM_DEBUG(llvm::errs() << "Backward overwrite threshold for concat\n";);
        patterns.insert<
            BackwardOverwriteThresholdConcatPattern
            >(context);
      }
      applyPatternsAndFoldGreedily(fn, std::move(patterns));
    }
    if (clCaliOverwriteThresholdForwardRelu) {
      //assert(!clCaliOverwriteThresholdBackwardRelu);
      LLVM_DEBUG(llvm::errs() << "Forward overwrite threshold for all\n";);
      patterns.clear();
      patterns.insert<
          ForwardOverwriteThresholdDefaultPattern<tpu::AbsOp>,
          ForwardOverwriteThresholdDefaultPattern<tpu::LeakyReluOp>,
          ForwardOverwriteThresholdDefaultPattern<tpu::ReluOp>,
          ForwardOverwriteThresholdDefaultPattern<tpu::UpsampleOp>,
          ForwardOverwriteThresholdDefaultPattern<tpu::PadOp>,
          ForwardOverwriteThresholdDefaultPattern<tpu::PermuteOp>,
          ForwardOverwriteThresholdDefaultPattern<tpu::CropOp>,
          ForwardOverwriteThresholdDefaultPattern<tpu::PoolMax2DOp>
          >(context);
      applyPatternsAndFoldGreedily(fn, std::move(patterns));
    }

    if (!clCaliOverwriteThresholdBackwardRelu
        && !clCaliOverwriteThresholdForwardRelu) {
      LLVM_DEBUG(llvm::errs() << "Default backward overwrite\n";);
      patterns.clear();
      patterns.insert<
          BackendOverwriteThresholdDefaultPattern<tpu::UpsampleOp>,
          BackendOverwriteThresholdDefaultPattern<tpu::PermuteOp>,
          BackendOverwriteThresholdDefaultPattern<tpu::CropOp>,
          BackendOverwriteThresholdDefaultPattern<tpu::PoolMax2DOp>
          >(context);
      applyPatternsAndFoldGreedily(fn, std::move(patterns));
    }

    // apply clip overwrite conv threshold
    LLVM_DEBUG(llvm::errs() << "Default clip threshold overwrite\n";);
    patterns.clear();
    patterns.insert<
        ForceThresholdClipOpPattern<tpu::ClipOp>
        >(context);
    //move to ConvertClip.cpp for mix precision
    //applyPatternsAndFoldGreedily(fn, std::move(patterns));

    LLVM_DEBUG(llvm::errs() << "set CustomOp's threshold\n";);
    patterns.clear();
    patterns.insert<
        ForceThresholdCustomOpPattern<tpu::CustomOp>
        >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));
  }

private:

  LogicalResult setThresholdFromMap(Operation *op,
      std::map<std::string, float> &threshold_map) {
    if (mlir::getOpName(op).empty()) {
      return failure();
    }

    std::string op_name = mlir::getOpName(op).str();
    if (threshold_map.find(op_name) == threshold_map.end()) {
      llvm::errs() << "Failed to find " << op_name << " in calibration table\n";
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
      LLVM_DEBUG(llvm::errs() << "  > " << op_name << ", "
                   << std::to_string(threshold) << "\n";);
    }
    return ret;
  }
  LogicalResult setWeightThresholdFromMap(
      Operation *op,
      std::unordered_map<std::string, std::vector<float>> &threshold_map) {

    auto weightOp = llvm::dyn_cast<tpu::LoadWeightOp>(op);
    auto builder = Builder(op->getContext());
    std::string op_name = weightOp.name().str();
    if (!threshold_map.count(op_name)) {
      return failure();
    }
    std::vector<float> weight_thresholds = threshold_map[op_name];
    weightOp->setAttr("threshold", builder.getF32ArrayAttr(weight_thresholds));
    return success();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::createImportCalibrationTablePass() {
  return std::make_unique<ImportCalibrationTablePass>();
}
