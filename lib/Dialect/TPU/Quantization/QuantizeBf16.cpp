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
#include "mlir/Dialect/TPU/QuantizationArithmetic.h"
#include "mlir/Dialect/StandardOps/Ops.h"
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
#include <math.h>

using namespace mlir;

namespace {

static void addWeightTensorAndUpdateWeightOp(Value* opd,
    std::vector<bfloat16> &weight, std::vector<int64_t> &shape,
    PatternRewriter &rewriter, TensorFile *wTF) {
  auto weightOp = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
      opd->getDefiningOp());
  auto name = weightOp.name().getValue().str() + "_quant_bf16";
  llvm::errs() << "  new_weight : " << name << "\n";
  auto type = RankedTensorType::get(shape,
      FloatType::getBF16(rewriter.getContext()));
  wTF->addTensor<uint16_t>(name, &weight, type);
  weightOp.setAttr("name", rewriter.getStringAttr(name));
  weightOp.setAttr("storage", rewriter.getStringAttr("BF16"));
  weightOp.getResult()->setType(type);
}

static std::unique_ptr<std::vector<float> > readAndDeleteWeightTensor(
    Value *opd, TensorFile *wTF) {
  auto weightOp = llvm::dyn_cast_or_null<tpu::LoadWeightOp>(
      opd->getDefiningOp());
  assert(weightOp);
  assert(weightOp.name().hasValue());
  auto name = weightOp.name().getValue();
  llvm::errs() << "  weight : " << name << "\n";
  auto type = weightOp.getResult()->getType().cast<TensorType>();
  auto T = wTF->readTensor<float>(name, type);
  // delete the tensor from the weight file
  wTF->deleteTensor<float>(name);
  return std::move(T);
}

struct TpuQuantConv2DOpPattern : public RewritePattern {
  TpuQuantConv2DOpPattern(MLIRContext *context, TensorFile *weightTensorFile,
      Value* weightFileVar)
      : RewritePattern("tpu.conv_2d", 1, context),
        weightTensorFile_(weightTensorFile),
        weightFileVar_(weightFileVar) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto convOp = cast<tpu::Conv2DOp>(op);
    //auto loc = op->getLoc();

    if (convOp.quant() != "NONE") {
      llvm::errs() << convOp.name() << " quantized already\n";
      return matchFailure();
    }
    assert(convOp.per_channel_info_is_aggregated() == false);

    // quantize filter
    auto filter = readAndDeleteWeightTensor(convOp.getOperand(1),
        weightTensorFile_);
    auto filterType = convOp.filter()->getType().cast<TensorType>();
    std::vector<int64_t> filterShape(filterType.getShape());
    int64_t filterSize = std::accumulate(std::begin(filterShape),
        std::end(filterShape), 1, std::multiplies<>());
    assert(filterSize == (int64_t)filter->size());
    // create new tensors
    auto new_filter = std::make_unique<std::vector<bfloat16> >(filterSize);
    // quantization
    FloatToBFloat16(filter->data(), new_filter->data(), filterSize);
    // update op
    addWeightTensorAndUpdateWeightOp(convOp.getOperand(1),
        *new_filter, filterShape, rewriter, weightTensorFile_);

    // quantize bias
    if (convOp.with_bias()) {
      auto bias = readAndDeleteWeightTensor(convOp.getOperand(2),
          weightTensorFile_);
      auto biasType = convOp.getOperand(2)->getType().cast<TensorType>();
      std::vector<int64_t> biasShape(biasType.getShape());
      int64_t biasSize = std::accumulate(std::begin(biasShape),
          std::end(biasShape), 1, std::multiplies<>());
      int64_t oc = 0;
      if (filterShape.size() == 4) {
        oc = filterShape[0];
      } else if (filterShape.size() == 5) {
        assert(convOp.group() != 1);
        // g, oc/g, ic/g, kh, kw
        oc = filterShape[0] * filterShape[1];
      } else {
        assert(0);
      }
      assert(biasSize == oc);
      assert(biasSize == (int64_t)bias->size());
      // create new tensors
      auto new_bias = std::make_unique<std::vector<bfloat16> >(biasSize);
      // quantization
      FloatToBFloat16(bias->data(), new_bias->data(), biasSize);
      // update op
      addWeightTensorAndUpdateWeightOp(convOp.getOperand(2),
          *new_bias, biasShape, rewriter, weightTensorFile_);
    }

    convOp.setAttr("quant", rewriter.getStringAttr("BF16"));

    return matchSuccess();
  }

  TensorFile *weightTensorFile_;
  Value* weightFileVar_;
};

struct TpuQuantFullyConnectedOpPattern : public RewritePattern {
  TpuQuantFullyConnectedOpPattern(MLIRContext *context, TensorFile *weightTensorFile,
      Value* weightFileVar)
      : RewritePattern("tpu.fully_connected", 1, context),
        weightTensorFile_(weightTensorFile),
        weightFileVar_(weightFileVar) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto fcOp = cast<tpu::FullyConnectedOp>(op);
    //auto loc = op->getLoc();

    if (fcOp.quant() != "NONE") {
      llvm::errs() << fcOp.name() << " quantized already\n";
      return matchFailure();
    }

    // quantize filter
    auto filter = readAndDeleteWeightTensor(fcOp.getOperand(1),
        weightTensorFile_);
    auto filterType = fcOp.filter()->getType().cast<TensorType>();
    std::vector<int64_t> filterShape(filterType.getShape());
    int64_t filterSize = std::accumulate(std::begin(filterShape),
        std::end(filterShape), 1, std::multiplies<>());
    assert(filterSize == (int64_t)filter->size());
    // create new tensors
    auto new_filter = std::make_unique<std::vector<bfloat16> >(filterSize);
    // quantization
    FloatToBFloat16(filter->data(), new_filter->data(), filterSize);
    // update op
    addWeightTensorAndUpdateWeightOp(fcOp.getOperand(1),
        *new_filter, filterShape, rewriter, weightTensorFile_);

    // quantize bias
    if (fcOp.with_bias()) {
      auto bias = readAndDeleteWeightTensor(fcOp.getOperand(2),
          weightTensorFile_);
      auto biasType = fcOp.getOperand(2)->getType().cast<TensorType>();
      std::vector<int64_t> biasShape(biasType.getShape());
      int64_t biasSize = std::accumulate(std::begin(biasShape),
          std::end(biasShape), 1, std::multiplies<>());
      int64_t n = filterShape[0];
      assert(biasSize == n);
      assert(biasSize == (int64_t)bias->size());
      // create new tensors
      auto new_bias = std::make_unique<std::vector<bfloat16> >(biasSize);
      // quantization
      FloatToBFloat16(bias->data(), new_bias->data(), biasSize);
      // update op
      addWeightTensorAndUpdateWeightOp(fcOp.getOperand(2),
          *new_bias, biasShape, rewriter, weightTensorFile_);
    }

    fcOp.setAttr("quant", rewriter.getStringAttr("BF16"));

    return matchSuccess();
  }

  TensorFile *weightTensorFile_;
  Value* weightFileVar_;
};

struct TpuQuantPool2DOpPattern : public RewritePattern {
  TpuQuantPool2DOpPattern(MLIRContext *context, TensorFile *weightTensorFile,
      Value* weightFileVar)
      : RewritePattern("tpu.pool_2d", 1, context),
        weightTensorFile_(weightTensorFile),
        weightFileVar_(weightFileVar) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto poolOp = cast<tpu::Pool2DOp>(op);
    std::string op_name = poolOp.getAttrOfType<StringAttr>("name").getValue().str();
    //auto loc = op->getLoc();

    if (poolOp.quant() != "NONE") {
      llvm::errs() << poolOp.name() << " quantized already\n";
      return matchFailure();
    }

    poolOp.setAttr("quant", rewriter.getStringAttr("BF16"));

    return matchSuccess();
  }

  TensorFile *weightTensorFile_;
  Value* weightFileVar_;
};

struct TpuQuantEltwiseOpPattern : public RewritePattern {
  TpuQuantEltwiseOpPattern(MLIRContext *context, TensorFile *weightTensorFile,
      Value* weightFileVar)
      : RewritePattern("tpu.eltwise", 1, context),
        weightTensorFile_(weightTensorFile),
        weightFileVar_(weightFileVar) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto eltOp = cast<tpu::EltwiseOp>(op);
    std::string op_name = eltOp.getAttrOfType<StringAttr>("name").getValue().str();
    //auto loc = op->getLoc();

    if (eltOp.quant() != "NONE") {
      llvm::errs() << eltOp.name() << " quantized already\n";
      return matchFailure();
    }

    eltOp.setAttr("quant", rewriter.getStringAttr("BF16"));

    return matchSuccess();
  }

  TensorFile *weightTensorFile_;
  Value* weightFileVar_;
};

template<typename T>
static void addQuantOpAfterOp(PatternRewriter &rewriter,
    T &op, std::string op_name) {
  auto loc = op.getLoc();

  auto *inst = op.getOperation();
  OpBuilder builder(inst);
  auto clonedOp = cast<T>(builder.clone(*inst));

  auto type = op.getResult()->getType();
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(op_name)));
  attrs.push_back(rewriter.getNamedAttr("quant", rewriter.getStringAttr("BF16")));

  auto quantOp = rewriter.create<tpu::QuantizationOp>(loc, type,
      ArrayRef<Value *>{clonedOp.getResult()}, ArrayRef<NamedAttribute>{attrs});
  rewriter.replaceOp(op, {quantOp});
}

template<typename T>
static void addDequantOpBeforeOp(PatternRewriter &rewriter,
    T &op, std::string op_name) {
  auto loc = op.getLoc();

  auto type = op.getOperation()->getOperand(0)->getType();
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("name", rewriter.getStringAttr(op_name)));
  attrs.push_back(rewriter.getNamedAttr("quant", rewriter.getStringAttr("BF16")));
  auto dequantOp = rewriter.create<tpu::DequantizationOp>(loc, type,
      ArrayRef<Value *>{op.getOperation()->getOperand(0)}, ArrayRef<NamedAttribute>{attrs});
  op.getOperation()->setOperand(0, dequantOp);
}

// insert Quant Op after input Op
struct TpuAddQuantAfterInputOpPattern : public OpRewritePattern<tpu::InputOp> {
  using OpRewritePattern<tpu::InputOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(tpu::InputOp op,
                                     PatternRewriter &rewriter) const {
    for (auto &use : op.getResult()->getUses()) {
      Operation *operandOp = use.getOwner();
      if (auto cast_op = llvm::dyn_cast_or_null<tpu::QuantizationOp>(operandOp)) {
        llvm::errs() << op.name() << " quantized already\n";
        return matchFailure();
      }
    }

    llvm::errs() << op.name() << " add quantization op after Input\n";
    std::string op_name = op.getAttrOfType<StringAttr>("name").getValue().str();
    addQuantOpAfterOp<tpu::InputOp>(rewriter, op, op_name + "_quant");

    return matchSuccess();
  }
};

// insert Dequant Op before return Op
struct TpuAddQuantBeforeReturnOpPattern : public OpRewritePattern<ReturnOp> {
  using OpRewritePattern<ReturnOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(ReturnOp op,
                                     PatternRewriter &rewriter) const {
    auto formerOp = op.getOperand(0)->getDefiningOp();
    if (matchPattern(formerOp, m_Op<tpu::DequantizationOp>())) {
      llvm::errs() << "return dequantized already\n";
      return matchFailure();
    }

    llvm::errs() << " add dequantization op defore Return\n";
    addDequantOpBeforeOp<ReturnOp>(rewriter, op, "return");

    return matchSuccess();
  }
};

struct TpuAddQuantAndDequantForSoftmaxOpPattern : public OpRewritePattern<tpu::SoftmaxOp> {
  using OpRewritePattern<tpu::SoftmaxOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(tpu::SoftmaxOp op,
                                     PatternRewriter &rewriter) const {
    auto formerOp = op.getOperand()->getDefiningOp();
    if (matchPattern(formerOp, m_Op<tpu::DequantizationOp>())) {
      llvm::errs() << op.name() << " insert quant and dequant already\n";
      return matchFailure();
    }

    llvm::errs() << op.name() << " insert quant and dequant\n";
    std::string op_name = op.getAttrOfType<StringAttr>("name").getValue().str();

    addDequantOpBeforeOp<tpu::SoftmaxOp>(rewriter, op, op_name + "_quant");
    addQuantOpAfterOp<tpu::SoftmaxOp>(rewriter, op, op_name + "_dequant");

    return matchSuccess();
  }
};

struct TpuSimplifyQuantDequantPattern : public OpRewritePattern<tpu::DequantizationOp> {
  using OpRewritePattern<tpu::DequantizationOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(tpu::DequantizationOp op,
                                     PatternRewriter &rewriter) const {
    auto formerOp = op.getOperand()->getDefiningOp();
    if (!matchPattern(formerOp, m_Op<tpu::QuantizationOp>())) {
      llvm::errs() << op.name() << " simplified quant and dequant already\n";
      return matchFailure();
    }

    llvm::errs() << " simplify quant and dequant\n";
    rewriter.replaceOp(op, formerOp->getOperand(0));

    return matchSuccess();
  }
};

class QuantizeBf16Pass : public FunctionPass<QuantizeBf16Pass> {
public:
  explicit QuantizeBf16Pass(llvm::raw_ostream &os = llvm::errs()) : os(os) {}

  void runOnFunction() override {
    auto fn = getFunction();

    // find tensorFile and Value
    llvm::StringRef filename;
    Value* weightFileVar;
    fn.walk([&](tpu::LoadFileOp op) {
      filename = op.filename();
      llvm::errs() << "LoadFileOp filename " << filename << "\n";
      weightFileVar = op.getResult();
    });
    auto weightTensorFile = openTensorFile(filename);

    auto *context = &getContext();

    OwningRewritePatternList patterns_w;
    patterns_w.insert<TpuQuantConv2DOpPattern>(context,
        weightTensorFile.get(), weightFileVar);
    patterns_w.insert<TpuQuantFullyConnectedOpPattern>(context,
        weightTensorFile.get(), weightFileVar);
    patterns_w.insert<TpuQuantPool2DOpPattern>(context,
        weightTensorFile.get(), weightFileVar);
    patterns_w.insert<TpuQuantEltwiseOpPattern>(context,
        weightTensorFile.get(), weightFileVar);
    applyPatternsGreedily(fn, patterns_w);

    OwningRewritePatternList patterns_q;
    // add Quant after Input
    patterns_q.insert<TpuAddQuantAfterInputOpPattern>(context);
    // add Dequant before Result
    patterns_q.insert<TpuAddQuantBeforeReturnOpPattern>(context);
    // add Quant and Dequant before and after any cpu layer
    patterns_q.insert<TpuAddQuantAndDequantForSoftmaxOpPattern>(context);
    applyPatternsGreedily(fn, patterns_q);

    OwningRewritePatternList patterns_s;
    // Fold and remove consecutive Dequant and Quant
    patterns_s.insert<TpuSimplifyQuantDequantPattern>(context);
    applyPatternsGreedily(fn, patterns_s);

    std::string newName;
    weightTensorFile->keep(true, &newName);
    fn.walk([&](tpu::LoadFileOp op) {
      OpBuilder opBuilder(context);
      op.setAttr("filename", opBuilder.getStringAttr(newName));
      llvm::errs() << "LoadFileOp filename updated to " << newName << "\n";
    });
  }

private:

  llvm::raw_ostream &os;
};

} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createQuantizeBf16Pass() {
  return std::make_unique<QuantizeBf16Pass>();
}

static PassRegistration<QuantizeBf16Pass>
    pass("quant-bf16",
         "Quantization to bf16");
