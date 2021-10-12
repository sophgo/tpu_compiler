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
#include "llvm/Support/Format.h"
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

namespace {
template <typename TyOp>
struct BackwardOverwriteThresholdDefaultPattern : public RewritePattern {
  BackwardOverwriteThresholdDefaultPattern(MLIRContext *context)
      : RewritePattern(TyOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    float threshold_y = getOpThreshold(op);
    auto formerOp = op->getOperand(0).getDefiningOp();
    if (!llvm::dyn_cast<tpu::TpuOpQuantInterface>(formerOp)) {
      return failure();
    }
    float threshold_x = getOpThreshold(formerOp);
    if (threshold_x == threshold_y) {
      return failure();
    }

    setOpThreshold(formerOp, threshold_y);
    LLVM_DEBUG(llvm::errs()
                   << op->getName() << " [" << getOpName(op) << "] set prev "
                   << formerOp->getName() << " [" << getOpName(formerOp)
                   << "], "
                      "threshold from "
                   << std::to_string(threshold_x) << " to "
                   << std::to_string(threshold_y) << "\n";);
    if (threshold_x < threshold_y * 0.5) {
      llvm::errs() << "WARNING: prev threshold is too smaller, "
                   << op->getName() << ":" << getOpName(op) << llvm::format("<%f>", threshold_y)
                   << " => " << formerOp->getName() << ":" << getOpName(formerOp)
                   << llvm::format("<%f>\n", threshold_x);
    } else if (threshold_x > threshold_y * 2.0) {
      llvm::errs() << "WARNING: prev threshold is too large, "
                   << op->getName() << ":" << getOpName(op) << llvm::format("<%f>", threshold_y)
                   << " => " << formerOp->getName() << ":" << getOpName(formerOp)
                   << llvm::format("<%f>\n", threshold_x);
    }

    return success();
  }
};

struct PoolMaskThresholdPattern : public RewritePattern {
  PoolMaskThresholdPattern(MLIRContext *context)
      : RewritePattern(tpu::PoolMaskOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto nextOp = getNextOp(op);
    if (!nextOp) {
      return failure();
    }
    auto mulOp = dyn_cast<tpu::EltwiseMulOp>(nextOp);
    if (!mulOp) {
      return failure();
    }
    float threshold_y = getOpThreshold(nextOp);
    unsigned nInputs = mulOp.getNumInputs();
    if (nInputs != 2) {
      return failure();
    }

    float threshold_x;
    for (unsigned i = 0; i < nInputs; ++i) {
      auto opd = mulOp.getOperand(i).getDefiningOp();
      if (dyn_cast<tpu::PoolMaskOp>(opd) ||
          dyn_cast<tpu::LoadWeightOp>(opd)) {
        continue;
      } else {
        threshold_x = getOpThreshold(opd);
      }
    }
    if (threshold_x != threshold_y) {
      setOpThreshold(nextOp, threshold_x);
      return success();
    }
    return failure();
  }
};

struct BackwardThresholdConcatPattern : public RewritePattern {
  BackwardThresholdConcatPattern(MLIRContext *context)
      : RewritePattern("tpu.concat", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto concatOp = cast<tpu::ConcatOp>(op);
    float threshold_y = getOpThreshold(op);
    unsigned nInputs = concatOp.getNumInputs();
    bool match = false;
    for (unsigned i = 0; i < nInputs; ++i) {
      auto formerOp = concatOp.getOperand(i).getDefiningOp();
      if (!llvm::dyn_cast<tpu::TpuOpQuantInterface>(formerOp)) {
        return failure();
      }
      if (isa<tpu::PoolMax2DOp>(formerOp)) {
        return failure();
      }
      float threshold_x = getOpThreshold(formerOp);
      if (formerOp->getResult(0).hasOneUse() && threshold_x != threshold_y) {
        setOpThreshold(formerOp, threshold_y);
        match = true;
      }
    }
    return match ? success() : failure();
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
    auto formerOp = op->getOperand(0).getDefiningOp();
    if (!llvm::dyn_cast<tpu::TpuOpQuantInterface>(formerOp)) {
      return failure();
    }
    float threshold_x = getOpThreshold(formerOp);
    float threshold_y = getOpThreshold(op);
    if (threshold_y == threshold_x) {
      // overwritten already
      return failure();
    }
    setOpThreshold(opInst, threshold_x);
    LLVM_DEBUG(llvm::errs() << opInst->getName() << " [" << op.name() << "] "
                  << "set threshold by prev Op threshold, from "
                  << std::to_string(threshold_y) << " to "
                  << std::to_string(threshold_x) << "\n";);
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
    std::unordered_map<std::string, int> weight_bitwidth_map;
    LLVM_DEBUG(llvm::errs() << "Calibration Table File : " << clCalibrationTableFilename << "\n");
    std::ifstream infile(clCalibrationTableFilename);
    std::string line;
    std::regex old_pattern("\\S+\\s+[-0-9.e]+");
    std::regex new_pattern(
        "\\S+\\s+[-0-9.e]+\\s+[-0-9.e]+\\s+[-0-9.e]+");
    std::regex weight_pattern("weight \\S+ .*");
    std::regex bitwidth_pattern("bitwidth \\S+\\s+\\d+");
    std::regex info_pattern("#.*");
    while (std::getline(infile, line)) {
      if (line.back() == '\r') {
        line.pop_back();
      }
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
        LLVM_DEBUG(llvm::errs() << "\n  infomation  " << line << "\n");
      } else if(std::regex_match(line, weight_pattern)){
        std::vector<float> weight_threshold;
        iss.ignore(256, ' '); // skip "weight"
        iss >> name;
        float threshold;
        while (iss >> threshold) {
          weight_threshold.push_back(threshold);
          LLVM_DEBUG(llvm::errs() << "  name " << name << " weight threshold: " << threshold << "\n";);
        }
        weight_threshold_map[name] = weight_threshold;
      } else if(std::regex_match(line, bitwidth_pattern)){
        iss.ignore(256, ' '); // skip "bitwidth"
        int bitwidth;
        iss >> name >> bitwidth;
        weight_bitwidth_map[name] = bitwidth;
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
      if (isa<tpu::LoadWeightOp>(op)) {
        setWeightThresholdFromMap(op, weight_threshold_map);
        setWeightBitwidthFromMap(op, weight_bitwidth_map);
      } else if (llvm::dyn_cast<tpu::PoolMaskOp>(op)) {
        setOpThreshold(op, 127.0f);
        setOpQuantParamType(op, "THRESHOLD");
      } else if (llvm::dyn_cast<tpu::SigmoidOp>(op)) {
        setOpThreshold(op, 1.0f);
        setOpQuantParamType(op, "THRESHOLD");
      } else if (llvm::dyn_cast<tpu::TpuOpQuantInterface>(op)) {
        setThresholdFromMap(op, threshold_map);
      } else {
        // do nothing
      }
    });

    OwningRewritePatternList patterns;

    // backward concat op
    LLVM_DEBUG(llvm::errs() << "Backward overwrite threshold for concat\n";);
    patterns.clear();
    patterns.insert<BackwardThresholdConcatPattern>(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));

    // backward first, these ops don't do quantization, and theshold_x != threshold_y
    // be careful, if you need threhsold_x == threshold_y and don't make sure whether do backward, just do forward
    LLVM_DEBUG(llvm::errs() << "Backward overwrite threshold for all\n";);
    patterns.clear();
    patterns.insert<
        BackwardOverwriteThresholdDefaultPattern<tpu::LeakyReluOp>,
        BackwardOverwriteThresholdDefaultPattern<tpu::ReluOp>,
        BackwardOverwriteThresholdDefaultPattern<tpu::CropOp>,
        BackwardOverwriteThresholdDefaultPattern<tpu::PoolMax2DOp>
        >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));

    // forward, make sure  theshold_x == threshold_y in all ops that no do quantization
    LLVM_DEBUG(llvm::errs() << "Forward overwrite threshold for all\n";);
    patterns.clear();
    patterns.insert<
        ForwardOverwriteThresholdDefaultPattern<tpu::AbsOp>,
        ForwardOverwriteThresholdDefaultPattern<tpu::CropOp>,
        ForwardOverwriteThresholdDefaultPattern<tpu::LeakyReluOp>,
        ForwardOverwriteThresholdDefaultPattern<tpu::ReluOp>,
        ForwardOverwriteThresholdDefaultPattern<tpu::PadOp>,
        ForwardOverwriteThresholdDefaultPattern<tpu::PixelShuffleOp>,
        ForwardOverwriteThresholdDefaultPattern<tpu::PermuteOp>,
        ForwardOverwriteThresholdDefaultPattern<tpu::PoolMax2DOp>,
        ForwardOverwriteThresholdDefaultPattern<tpu::ReorgOp>,
        ForwardOverwriteThresholdDefaultPattern<tpu::ReverseOp>,
        ForwardOverwriteThresholdDefaultPattern<tpu::ReduceMaxOp>,
        ForwardOverwriteThresholdDefaultPattern<tpu::ReduceSumOp>,
        ForwardOverwriteThresholdDefaultPattern<tpu::SliceOp>,
        ForwardOverwriteThresholdDefaultPattern<tpu::ShuffleChannelOp>,
        ForwardOverwriteThresholdDefaultPattern<tpu::SwapChannelOp>,
        ForwardOverwriteThresholdDefaultPattern<tpu::TileOp>,
        ForwardOverwriteThresholdDefaultPattern<tpu::UpsampleOp>
        >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));

    LLVM_DEBUG(llvm::errs() << "set CustomOp's threshold\n";);
    patterns.clear();
    patterns.insert<
        ForceThresholdCustomOpPattern<tpu::CustomOp>
        >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));

    LLVM_DEBUG(llvm::errs() << "set PoolMask threshold\n";);
    patterns.clear();
    patterns.insert<PoolMaskThresholdPattern>(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));

    // make sure all ops have threshold
    int count = 0;
    fn.walk([&](Operation *op) {
      if (llvm::dyn_cast<tpu::TpuOpQuantInterface>(op)) {
        if (getOpQuantParamType(op) != "THRESHOLD") {
           llvm::errs() << "Warning:" << mlir::getOpName(op).str()
                        << " has no calibartion threshold\n";
           count++;
        }
      }
    });
    // assert(count == 0);
  }

private:

  LogicalResult setThresholdFromMap(Operation *op,
      std::map<std::string, float> &threshold_map) {
    if (mlir::getOpName(op).empty()) {
      return failure();
    }
    if (isa<tpu::ReshapeOp>(op)) {
      return failure();
    }

    std::string op_name = mlir::getOpName(op).str();
    if (threshold_map.find(op_name) == threshold_map.end()) {
      llvm::errs() << "Failed to find " << op_name << " in calibration table\n";
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

  LogicalResult setWeightBitwidthFromMap(
      Operation *op,
      std::unordered_map<std::string, int> &bitwidth_map) {
    auto weightOp = llvm::dyn_cast<tpu::LoadWeightOp>(op);
    auto builder = Builder(op->getContext());
    std::string op_name = weightOp.name().str();
    if (!bitwidth_map.count(op_name)) {
      return failure();
    }
    int bitwidth = bitwidth_map[op_name];
    if (bitwidth > 32) {
      llvm_unreachable((op_name + " quant_bitwidth too large").c_str());
    }
    weightOp->setAttr("quant_bitwidth", builder.getI32IntegerAttr(bitwidth));
    return success();
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
