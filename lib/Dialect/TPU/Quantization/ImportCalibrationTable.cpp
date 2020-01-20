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

namespace {

/// bypass pool quantization by assigning threshold_y same as threshold_x.
/// for max pooling, threshold_y is always larger than threshold_x,
/// however, if one value exceed the quantization range, it has been saturated
/// already, an extra multiplier does not help.
/// for avg pooling, threshold_y is smaller than threshold_x,
/// we shall keep the quantization (i.e. multiply with a multiplier)
struct BypassPoolQuantPattern : public OpRewritePattern<tpu::Pool2DOp> {
  using OpRewritePattern<tpu::Pool2DOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(tpu::Pool2DOp op,
                                     PatternRewriter &rewriter) const {
    float threshold_x = getPreviousOpThreshold(op);
    float threshold_y = op.threshold_y().getValue().convertToFloat();
    if (threshold_y > threshold_x) {
      op.setAttr("threshold_y", rewriter.getF32FloatAttr(threshold_x));
      return matchSuccess();
    } else {
      return matchFailure();
    }
  }
};

struct BypassPReluQuantPattern : public OpRewritePattern<tpu::PReluOp> {
  using OpRewritePattern<tpu::PReluOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(tpu::PReluOp op,
                                     PatternRewriter &rewriter) const {
    float threshold_x = getPreviousOpThreshold(op);
    op.setAttr("threshold_y", rewriter.getF32FloatAttr(threshold_x));

    return matchSuccess();
  }
};

/// bypass relu quantization by assigning threshold_y same as threshold_x.
/// for same reason as bypassing max pool.
struct BypassReluQuantPattern : public OpRewritePattern<tpu::ReluOp> {
  using OpRewritePattern<tpu::ReluOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(tpu::ReluOp op,
                                     PatternRewriter &rewriter) const {
    float threshold_x = getPreviousOpThreshold(op);
    op.setAttr("threshold_y", rewriter.getF32FloatAttr(threshold_x));

    return matchSuccess();
  }
};

/// there is no calibration result for reshape op, assign threshold_y as threshold_x
struct AssignReshapeThresholdPattern : public OpRewritePattern<tpu::ReshapeOp> {
  using OpRewritePattern<tpu::ReshapeOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(tpu::ReshapeOp op,
                                     PatternRewriter &rewriter) const {
    float threshold_x = getPreviousOpThreshold(op);
    op.setAttr("threshold_y", rewriter.getF32FloatAttr(threshold_x));


    return matchSuccess();
  }
};

/// bypass slice quantization by assigning threshold_y same as threshold_x.
struct BypassSliceQuantPattern : public OpRewritePattern<tpu::SliceOp> {
  using OpRewritePattern<tpu::SliceOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(tpu::SliceOp op,
                                     PatternRewriter &rewriter) const {
    float threshold_x = getPreviousOpThreshold(op);
    op.setAttr("threshold_y", rewriter.getF32FloatAttr(threshold_x));

    return matchSuccess();
  }
};

class ImportCalibrationTablePass : public FunctionPass<ImportCalibrationTablePass> {
public:
  explicit ImportCalibrationTablePass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    // load the table
    std::map<std::string, float> threshold_map;
    os << "Calibration Table File : " << clCalibrationTableFilename << "\n";
    std::ifstream infile(clCalibrationTableFilename);
    std::string line;
    while (std::getline(infile, line)) {
      std::istringstream iss(line);
      std::string name;
      float threshold;
      if (!(iss >> name >> threshold)) { break; }
      std::cout << "  name " << name << ", threshold " << threshold << "\n";
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

    OwningRewritePatternList patterns;
    //auto *context = &getContext();
    patterns.insert<BypassPoolQuantPattern, BypassReluQuantPattern, BypassSliceQuantPattern,
        AssignReshapeThresholdPattern>(context);
    applyPatternsGreedily(fn, patterns);
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
        os << " > " << op_name << ", " << threshold << "\n";
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
