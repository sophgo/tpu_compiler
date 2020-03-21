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
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
#include "mlir/Dialect/TPU/QuantizationArithmetic.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TensorFile.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <sstream>
#include <fstream>

#define DEBUG_TYPE "quantization"

using namespace mlir;

static llvm::cl::OptionCategory clOptionsCategory("TPU quantization options");

static llvm::cl::opt<bool> clQuantInt8PerTensor(
    "quant-int8-per-tensor",
    llvm::cl::desc("Disable per channel for convolution quantization"),
    llvm::cl::init(false),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<bool> clQuantInt8RshiftOnly(
    "quant-int8-rshift-only",
    llvm::cl::desc("Disable multipler for convolution quantization"),
    llvm::cl::init(false),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<bool> clQuantBf16(
    "quant-full-bf16",
    llvm::cl::desc("Quant to bf16 for all TPU ops"),
    llvm::cl::init(false),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<bool> clQuantMixTable(
    "quant-int8-mix-bf16-table",
    llvm::cl::desc("Enable bf16 mix-presion from a table specifying mode for each TPU Op"),
    llvm::cl::init(false),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<bool> clQuantMixSigmoid(
    "quant-int8-mix-bf16-sigmoid",
    llvm::cl::desc("Enable bf16 mix-presion on sigmoid Ops"),
    llvm::cl::init(false),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<bool> clQuantMixBroadcastMul(
    "quant-int8-mix-bf16-broadcastmul",
    llvm::cl::desc("Enable bf16 mix-presion on BroadcastMul Ops"),
    llvm::cl::init(false),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<bool> clQuantMixEltwiseMul(
    "quant-int8-mix-bf16-eltwisemul",
    llvm::cl::desc("Enable bf16 mix-presion on EltwiseMul Ops"),
    llvm::cl::init(false),
    llvm::cl::cat(clOptionsCategory));

template<typename OpTy>
struct TpuAddInt8QuantOpBeforeOpPattern : public RewritePattern {
  TpuAddInt8QuantOpBeforeOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    if (op->getOperand(0)->getDefiningOp()
        && isa<tpu::QuantOp>(op->getOperand(0)->getDefiningOp())) {
      // added already
      return matchFailure();
    }

    auto type = op->getResult(0)->getType();
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("from",
        rewriter.getStringAttr("NONE")));
    attrs.push_back(rewriter.getNamedAttr("to",
        rewriter.getStringAttr("INT8")));
    attrs.push_back(rewriter.getNamedAttr("threshold",
        rewriter.getF32FloatAttr(getOpThreshold(op))));
    attrs.push_back(rewriter.getNamedAttr("name",
        rewriter.getStringAttr(getOpName(op).str() + "_quant")));
    attrs.push_back(rewriter.getNamedAttr("layer_id",
        rewriter.getI32IntegerAttr(getOpLayerId(op))));
    auto quantOp = rewriter.create<tpu::QuantOp>(op->getLoc(), type,
        ArrayRef<Value *>{op->getOperand(0)}, ArrayRef<NamedAttribute>{attrs});
    setOpResultType(quantOp.getOperation(), StandardTypes::Integer, 8);

    op->setOperand(0, quantOp.getResult());

    return matchSuccess();
  }
};

template<typename OpTy>
struct TpuAddInt8DequantOpBeforeOpPattern : public RewritePattern {
  TpuAddInt8DequantOpBeforeOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    if (isa<tpu::QuantOp>(op->getOperand(0)->getDefiningOp())) {
      // added already
      return matchFailure();
    }

    for (unsigned i = 0; i < op->getNumOperands(); i++) {
      auto prev_op = op->getOperand(i)->getDefiningOp();
      if (getOpQuant(prev_op) != "INT8") {
        continue;
      }
      auto type = op->getOperand(i)->getType();
      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr("from",
          rewriter.getStringAttr("INT8")));
      attrs.push_back(rewriter.getNamedAttr("to",
          rewriter.getStringAttr("NONE")));
      attrs.push_back(rewriter.getNamedAttr("threshold",
          rewriter.getF32FloatAttr(getOpThreshold(prev_op))));
      attrs.push_back(rewriter.getNamedAttr("name",
          rewriter.getStringAttr(getOpName(prev_op).str() + "_dequant")));
      attrs.push_back(rewriter.getNamedAttr("layer_id",
          rewriter.getI32IntegerAttr(getOpLayerId(prev_op))));
      auto quantOp = rewriter.create<tpu::QuantOp>(prev_op->getLoc(), type,
          ArrayRef<Value *>{op->getOperand(i)}, ArrayRef<NamedAttribute>{attrs});
      setOpResultType(quantOp.getOperation(), StandardTypes::F32);
      op->setOperand(i, quantOp.getResult());
    }

    return matchSuccess();
  }
};

template<typename OpTy>
struct TpuAddBf16QuantOpBeforeOpPattern : public RewritePattern {
  TpuAddBf16QuantOpBeforeOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    if (op->getOperand(0)->getDefiningOp()
        && isa<tpu::QuantOp>(op->getOperand(0)->getDefiningOp())) {
      // added already
      return matchFailure();
    }

    auto type = op->getResult(0)->getType();
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("from",
        rewriter.getStringAttr("NONE")));
    attrs.push_back(rewriter.getNamedAttr("to",
        rewriter.getStringAttr("BF16")));
    attrs.push_back(rewriter.getNamedAttr("name",
        rewriter.getStringAttr(getOpName(op).str() + "_quant")));
    attrs.push_back(rewriter.getNamedAttr("layer_id",
        rewriter.getI32IntegerAttr(getOpLayerId(op))));
    auto quantOp = rewriter.create<tpu::QuantOp>(op->getLoc(), type,
        ArrayRef<Value *>{op->getOperand(0)}, ArrayRef<NamedAttribute>{attrs});

    op->setOperand(0, quantOp.getResult());

    return matchSuccess();
  }
};

template<typename OpTy>
struct TpuAddBf16DequantOpBeforeOpPattern : public RewritePattern {
  TpuAddBf16DequantOpBeforeOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    if (isa<tpu::QuantOp>(op->getOperand(0)->getDefiningOp())) {
      // added already
      return matchFailure();
    }

    for (unsigned i = 0; i < op->getNumOperands(); i++) {
      auto prev_op = op->getOperand(i)->getDefiningOp();
      if (getOpQuant(prev_op) != "BF16") {
        continue;
      }
      auto type = op->getOperand(i)->getType();
      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr("from",
          rewriter.getStringAttr("BF16")));
      attrs.push_back(rewriter.getNamedAttr("to",
          rewriter.getStringAttr("NONE")));
      attrs.push_back(rewriter.getNamedAttr("threshold",
          rewriter.getF32FloatAttr(getOpThreshold(prev_op))));
      attrs.push_back(rewriter.getNamedAttr("name",
          rewriter.getStringAttr(getOpName(prev_op).str() + "_dequant")));
      attrs.push_back(rewriter.getNamedAttr("layer_id",
          rewriter.getI32IntegerAttr(getOpLayerId(prev_op))));
      auto quantOp = rewriter.create<tpu::QuantOp>(prev_op->getLoc(), type,
          ArrayRef<Value *>{op->getOperand(i)}, ArrayRef<NamedAttribute>{attrs});
      setOpResultType(quantOp.getOperation(), StandardTypes::F32);
      op->setOperand(i, quantOp.getResult());
    }

    return matchSuccess();
  }
};

template<typename OpTy>
struct TpuInsertQuantOpPattern : public RewritePattern {
  TpuInsertQuantOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    if (isa<tpu::WeightFileOp>(op)
        || isa<tpu::LoadWeightOp>(op)
        || isa<tpu::NoneOp>(op)
        || isa<tpu::NoneOp>(op)) {
      assert(false);
      return matchFailure();
    }

    LLVM_DEBUG(llvm::errs() << "insertQuant: " << OpTy::getOperationName()
                 << " [" << getOpName(op) << "]\n";);

    int changed = 0;
    StringRef curr_quant;
    if (isa<ReturnOp>(op)) {
      curr_quant = "NONE";
    } else {
      curr_quant = getOpQuant(op);
    }
    for (unsigned i = 0; i < op->getNumOperands(); i++) {
      auto prev_op = op->getOperand(i)->getDefiningOp();
      StringRef prev_quant;
      if (!prev_op) {
        assert(isa<tpu::InputOp>(op));
        prev_quant = "NONE";
      } else if (isa<tpu::QuantOp>(prev_op)
                 || isa<tpu::LoadWeightOp>(prev_op)
                 || isa<tpu::NoneOp>(prev_op)) {
        continue;
      } else {
        prev_quant = getOpQuant(prev_op);
      }

      // insert quant if prev and curr have different quant mode
      if (curr_quant != prev_quant) {
        std::vector<NamedAttribute> attrs;
        attrs.push_back(rewriter.getNamedAttr("from",
            rewriter.getStringAttr(prev_quant)));
        attrs.push_back(rewriter.getNamedAttr("to",
            rewriter.getStringAttr(curr_quant)));
        float threshold = 0.0f;
        std::string name;
        int layer_id = -1;
        if (curr_quant == "INT8") {
          if (prev_quant == "BF16") {
            threshold = getOpThreshold(prev_op);
          } else {
            // TODO: should move quant after Input Op
            // assert(isa<tpu::InputOp>(op) || isa<tpu::PreprocessOp>(op));
            threshold = getOpThreshold(prev_op);
          }
          name = getOpName(prev_op).str() + "_quant";
          layer_id = getOpLayerId(op);
        } else if (prev_quant == "INT8") {
          threshold = getOpThreshold(prev_op);
          name = getOpName(prev_op).str() + "_dequant";
          layer_id = getOpLayerId(prev_op);
        } else if (curr_quant == "BF16") {
          name = getOpName(op).str() + "_quant";
          layer_id = getOpLayerId(op);
        } else if (prev_quant == "BF16") {
          name = getOpName(prev_op).str() + "_dequant";
          layer_id = getOpLayerId(prev_op);
        }
        // check if prev op has inserted quant/dequant op
        if (prev_op) {
          bool found = false;
          for (auto &use : prev_op->getResult(0)->getUses()) {
            auto nextOp = use.getOwner();
            if (getOpName(nextOp) == name) {
              op->setOperand(i, nextOp->getResult(0));
              LLVM_DEBUG(llvm::errs() << "  opd " << i << ", " << name << ", "
                        << prev_quant << " => " << curr_quant << "\n";);
              found = true;
              nextOp->moveBefore(op);
              break;
            }
          }
          if (found) {
            changed++;
            continue;
          }
        }

        attrs.push_back(rewriter.getNamedAttr("threshold",
            rewriter.getF32FloatAttr(threshold)));
        attrs.push_back(rewriter.getNamedAttr("name",
            rewriter.getStringAttr(name)));
        attrs.push_back(rewriter.getNamedAttr("layer_id",
            rewriter.getI32IntegerAttr(layer_id)));

        auto shape = op->getOperand(i)->getType().cast<TensorType>().getShape();
        Type eltType;
        if (curr_quant == "INT8") {
          eltType = IntegerType::get(8, rewriter.getContext());
        } else if (curr_quant == "BF16") {
          eltType = FloatType::getBF16(rewriter.getContext());
        } else {
          eltType = FloatType::getF32(rewriter.getContext());
        }
        auto type = RankedTensorType::get(shape, eltType);
        auto quantOp = rewriter.create<tpu::QuantOp>(op->getLoc(), type,
            ArrayRef<Value *>{op->getOperand(i)}, ArrayRef<NamedAttribute>{attrs});

        op->setOperand(i, quantOp.getResult());

        LLVM_DEBUG(llvm::errs() << "  opd " << i << ", " << name << ", "
                   << prev_quant << " => " << curr_quant << "\n";);

        changed++;
      }
    }
    if (changed) {
      return matchSuccess();
    }
    return matchFailure();
  }
};

class TpuQuantPass : public FunctionPass<TpuQuantPass> {
public:
  explicit TpuQuantPass() {}

  void runOnFunction() override {
    auto fn = getFunction();
    auto *context = &getContext();

    // mark quant mode
    fn.walk([&](Operation *op) {
      if (op->getName().getDialect().str() != "tpu"
          || isa<tpu::WeightFileOp>(op)
          || isa<tpu::LoadWeightOp>(op)
          || isa<tpu::NoneOp>(op)) {
      } else if (isa<tpu::ReshapeOp>(op)
                 || isa<tpu::InputOp>(op)
                 || isa<tpu::PreprocessOp>(op)
                 || isa<tpu::TransposeOp>(op)) {
        // no need to quant
      } else if (auto quantOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(op)) {
        if (clQuantMixTable) {
          //setOpQuant(op, quant_mix_table[getOpName(op)]);
          assert(false);
        } else if (!clQuantBf16) {
          if (!isa<tpu::InputOp>(op))
            setOpQuant(op, "INT8");
          if (isa<tpu::Conv2DOp>(op) || isa<tpu::DeConv2DOp>(op)) {
            if (clQuantInt8PerTensor) {
              setOpQuantPerchannel(op, false);
              setOpQuantParamType(op, "RSHIFT_ONLY");
            } else {
              setOpQuantPerchannel(op, true);
              if (clQuantInt8RshiftOnly) {
                setOpQuantParamType(op, "RSHIFT_ONLY");
              } else {
                setOpQuantParamType(op, "RSHIFT_AND_M_I32");
              }
            }
          }
          // mix-bf16 options
          if (clQuantMixSigmoid && isa<tpu::SigmoidOp>(op)) {
            setOpQuant(op, "BF16");
          }
          if (clQuantMixBroadcastMul && isa<tpu::BroadcastMulOp>(op)) {
            setOpQuant(op, "BF16");
          }
          if (clQuantMixEltwiseMul && isa<tpu::EltwiseMulOp>(op)) {
            setOpQuant(op, "BF16");
          }
        } else {
          setOpQuant(op, "BF16");
        }
      } else if (isa<tpu::DetectionOutputOp>(op)
                 || isa<tpu::PriorBoxOp>(op)
                 || isa<tpu::RetinaFaceDetectionOp>(op)
                 || isa<tpu::SoftmaxOp>(op)) {
        // cpu Ops that has no quant support
      } else {
        llvm::errs() << "quant assignment didn't handle " << op->getName() << "\n";
        assert(false);
      }
    });

    // do quant
    fn.walk([&](Operation *op) {
      if (op->getName().getDialect().str() != "tpu"
          || isa<tpu::WeightFileOp>(op)
          || isa<tpu::LoadWeightOp>(op)
          || isa<tpu::NoneOp>(op)) {
      } else if (isa<tpu::ReshapeOp>(op)
                 || isa<tpu::QuantOp>(op)
                 || isa<tpu::InputOp>(op)
                 || isa<tpu::PreprocessOp>(op)
                 || isa<tpu::TransposeOp>(op)) {
        // no need to quant
      } else if (auto quantOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(op)) {
        if (getOpQuant(op) == "INT8") {
          auto ret = quantOp.quantizeInt8();
          assert(!failed(ret));
        } else if (getOpQuant(op) == "BF16") {
          auto ret = quantOp.quantizeBf16();
          assert(!failed(ret));
        } else {
          assert(false);
        }
      } else if (isa<tpu::DetectionOutputOp>(op)
                 || isa<tpu::PriorBoxOp>(op)
                 || isa<tpu::RetinaFaceDetectionOp>(op)
                 || isa<tpu::SoftmaxOp>(op)) {
        // cpu Ops that has no quant support
      } else {
        llvm::errs() << "quant didn't handle " << op->getName() << "\n";
        assert(false);
      }
    });

    // To make ReshapeOp's result element type same as
    // operand's after quantization
    fn.walk([&](tpu::ReshapeOp op) {
      auto _op = op.getOperation();
      auto eltType = _op->getOperand(0)->getType().cast<TensorType>().getElementType();
      auto shape = _op->getResult(0)->getType().cast<TensorType>().getShape();
      auto type = RankedTensorType::get(shape, eltType);
      _op->getResult(0)->setType(type);
    });

    // insert quant/dequant
    OwningRewritePatternList patterns;
    patterns.insert<
        TpuInsertQuantOpPattern<ReturnOp>,
        TpuInsertQuantOpPattern<tpu::BroadcastMulOp>,
        TpuInsertQuantOpPattern<tpu::BatchNormOp>,
        TpuInsertQuantOpPattern<tpu::ConcatOp>,
        TpuInsertQuantOpPattern<tpu::Conv2DOp>,
        TpuInsertQuantOpPattern<tpu::CropOp>,
        TpuInsertQuantOpPattern<tpu::DeConv2DOp>,
        TpuInsertQuantOpPattern<tpu::DetectionOutputOp>,
        TpuInsertQuantOpPattern<tpu::EltwiseAddOp>,
        TpuInsertQuantOpPattern<tpu::EltwiseMaxOp>,
        TpuInsertQuantOpPattern<tpu::EltwiseMulOp>,
        TpuInsertQuantOpPattern<tpu::FullyConnectedOp>,
        //TpuInsertQuantOpPattern<tpu::InputOp>,
        TpuInsertQuantOpPattern<tpu::LeakyReluOp>,
        TpuInsertQuantOpPattern<tpu::NormalizeOp>,
        TpuInsertQuantOpPattern<tpu::PermuteOp>,
        TpuInsertQuantOpPattern<tpu::PixelShuffleOp>,
        TpuInsertQuantOpPattern<tpu::PoolAvg2DOp>,
        TpuInsertQuantOpPattern<tpu::PoolMax2DOp>,
        TpuInsertQuantOpPattern<tpu::PowerOp>,
        TpuInsertQuantOpPattern<tpu::PReluOp>,
        TpuInsertQuantOpPattern<tpu::PriorBoxOp>,
        TpuInsertQuantOpPattern<tpu::ReciprocalOp>,
        TpuInsertQuantOpPattern<tpu::ReluOp>,
        TpuInsertQuantOpPattern<tpu::ReshapeOp>,
        TpuInsertQuantOpPattern<tpu::RetinaFaceDetectionOp>,
        TpuInsertQuantOpPattern<tpu::ScaleOp>,
        TpuInsertQuantOpPattern<tpu::ShuffleChannelOp>,
        TpuInsertQuantOpPattern<tpu::SliceOp>,
        TpuInsertQuantOpPattern<tpu::SoftmaxOp>,
        TpuInsertQuantOpPattern<tpu::SigmoidOp>,
        TpuInsertQuantOpPattern<tpu::SqrtOp>,
        TpuInsertQuantOpPattern<tpu::SwapChannelOp>,
        TpuInsertQuantOpPattern<tpu::TanHOp>,
        TpuInsertQuantOpPattern<tpu::UpsampleOp>
    >(context);
    applyPatternsGreedily(fn, patterns);
  }
};

std::unique_ptr<OpPassBase<FuncOp>> mlir::createTpuQuantPass() {
  return std::make_unique<TpuQuantPass>();
}

static PassRegistration<TpuQuantPass>
    pass("tpu-quant",
         "Do quantization on TPU Ops");
