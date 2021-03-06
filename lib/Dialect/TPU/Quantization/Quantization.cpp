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
#include "tpuc/TPUTensorSupport.h"
#include "tpuc/QuantizationArithmetic.h"
#include "tpuc/NativeCpuImplementation.h"
#include "tpuc/MachineInfo.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "tpuc/CustomOpPlugin.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "tpuc/Support/TensorFile.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/DynamicLibrary.h"
#include <sstream>
#include <fstream>
#include "tpuc/MachineInfo.h"
#include <regex>

#define DEBUG_TYPE "quantization"

using namespace mlir;

static llvm::cl::OptionCategory clOptionsCategory("TPU quantization options");

static llvm::cl::opt<std::string> clQuantMode(
    "quant-mode",
    llvm::cl::desc("Quant mode for all TPU ops:BF16/INT8/MIX_BF16"),
    llvm::cl::init("INT8"),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<bool> clQuantMixSoftmax(
    "quant-bf16-softmax",
    llvm::cl::desc("Enable bf16 Softmax Ops"),
    llvm::cl::init(true),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<std::string> clQuantLayerByFile(
    "quant-mix-layers-file",
    llvm::cl::desc("Enable bf16 mix-presion on specify layers by file"),
    llvm::cl::cat(clOptionsCategory));

typedef enum {
  QUANT_INT8,
  QUANT_BF16,
  QUANT_MIX_BF16,
  QUANT_UNKNOWN,
} quant_mode_t;

static quant_mode_t qmode(const std::string &mode) {
  std::string tmp = toupper(mode);
  if (tmp == "INT8") {
    return QUANT_INT8;
  }
  if (tmp == "BF16") {
    return QUANT_BF16;
  }
  if (tmp == "MIX_BF16") {
    return QUANT_MIX_BF16;
  }
  llvm::errs() << "Error, unknown quant mode: " << mode << "\n";
  assert(false);
  return QUANT_UNKNOWN;
}

static std::map<std::string, quant_mode_t> gQuantLayers;
static quant_mode_t gDefaultQuant = QUANT_BF16;

static inline bool is_fix8b(const StringRef &quant) {
  return quant == "INT8" || quant == "UINT8";
}

static void insertQuantOp(Operation *op) {
  auto builder = OpBuilder(op);
  StringRef curr_quant;
  if (isa<tpu::ReshapeOp>(op)) {
    if (!isa<tpu::InputOp>(op->getOperand(0).getDefiningOp())) {
      return;
    }
    auto nextOp = getNextOp(op);
    if (nextOp == nullptr || isa<tpu::ReshapeOp>(nextOp)) {
      return;
    }
    curr_quant = isa<ReturnOp>(nextOp) ? "NONE" : getOpQuant(nextOp);
  } else {
    curr_quant = isa<ReturnOp>(op) ? "NONE" : getOpQuant(op);
  }

  if (isa<tpu::ZeroMaskOp>(op)) {
    auto input = op->getOperand(0);
    for (auto &use : input.getUses()) {
      auto useOp = use.getOwner();
      if (isa<tpu::QuantOp>(useOp)) {
        op->setOperand(0, useOp->getResult(0));
        return;
      }
      if (isa<tpu::EmbeddingOp>(useOp)) {
        curr_quant = "UINT16";
        break;
      }
    }
  }

  for (unsigned i = 0; i < op->getNumOperands(); i++) {
    auto prev_op = op->getOperand(i).getDefiningOp();
    assert(prev_op);
    if (isa<tpu::QuantOp>(prev_op)
        || isa<tpu::LoadWeightOp>(prev_op)
        || isa<tpu::NoneOp>(prev_op)) {
      continue;
    }
    if (isa<tpu::ReshapeOp>(prev_op) &&
        isa<tpu::QuantOp>(prev_op->getOperand(0).getDefiningOp())) {
      continue;
    }
    if (isa<tpu::EmbeddingOp>(op)) {
      curr_quant = (i == 0 ? "UINT16" : "BF16");
    }
    if (isa<tpu::MatchTemplateOp>(op)) {
      curr_quant =  "UINT8";
    }

    StringRef prev_quant;
    auto prev_name = getOpName(prev_op);
    if (isa<tpu::ReshapeOp>(prev_op)) {
      prev_op = prev_op->getOperand(0).getDefiningOp();
    }
    prev_quant = getOpQuant(prev_op);
    if (auto castOp = dyn_cast<tpu::QuadraticSumOp>(prev_op)) {
      if (castOp.high_precision()) {
        prev_quant = "NONE";
      }
    }
    if (auto castOp = dyn_cast<tpu::ArgMaxOp>(prev_op)) {
      prev_quant = "NONE";
    }

    if (prev_quant == "INT8" && isa<tpu::CscOp>(prev_op)) {
      prev_quant = "UINT8";
    }

    // if cur and prev op has same quant type, return directly.
    if (curr_quant == prev_quant) {
      continue;
    }
    // Not to insert quant op if cur and prev op
    // are same int8 quant type but different in signness.
    if ((curr_quant == "UINT8" || curr_quant == "INT8") &&
        (prev_quant == "UINT8" || prev_quant == "INT8")) {
      continue;
    }

    // insert quant if prev and curr have different quant mode
    float scale = 1.0f;
    std::string name;
    if (curr_quant == "INT8" || curr_quant == "UINT8") {
      // FP32|BF16 => INT8|UINT8
      int max_val = (curr_quant == "INT8") ? 128 : 256;
      scale = max_val / getOpThreshold(prev_op);
      name = prev_name.str() + "_quant_i8";
    } else if (prev_quant == "INT8" || prev_quant == "UINT8") {
      // INT8/UINT8 ==> FP32|BF16
      int max_val = (prev_quant == "INT8") ? 128 : 256;
      scale = getOpThreshold(prev_op) / max_val;
      if (curr_quant == "BF16") {
        name = prev_name.str() + "_dequant_bf16";
      } else {
        name = prev_name.str() + "_dequant";
      }
    } else if (curr_quant == "BF16") {
      // FP32 => BF16
      name = prev_name.str() + "_quant_bf16";
    } else if (prev_quant == "BF16") {
      // BF16 => FP32
      name = prev_name.str() + "_dequant";
    } else if (curr_quant == "INT16") {
      name = prev_name.str() + "_quant_i16";
    } else if (curr_quant == "UINT16") {
      name = prev_name.str() + "_quant_u16";
    }
    // check if prev op has inserted quant/dequant op
    auto opd = op->getOperand(i).getDefiningOp();
    if (opd) {
      bool found = false;
      for (auto &use : opd->getResult(0).getUses()) {
        auto nextOp = use.getOwner();
        if (getOpName(nextOp) == name) {
          op->setOperand(i, nextOp->getResult(0));
          LLVM_DEBUG(llvm::errs() << "  opd " << i << ", " << name << ", "
                    << prev_quant << " => " << curr_quant << "\n";);
          found = true;
          break;
        }
      }
      if (found) {
        continue;
      }
    }

    std::vector<NamedAttribute> attrs;
    attrs.push_back(builder.getNamedAttr("from",
        builder.getStringAttr(prev_quant)));
    attrs.push_back(builder.getNamedAttr("to",
        builder.getStringAttr(curr_quant)));
    attrs.push_back(builder.getNamedAttr("scale",
        builder.getF32FloatAttr(scale)));
    attrs.push_back(builder.getNamedAttr("name",
        builder.getStringAttr(name)));

    auto shape = op->getOperand(i).getType().cast<TensorType>().getShape();
    Type eltType;
    if (curr_quant == "INT8") {
      eltType = IntegerType::get(builder.getContext(), 8);
    } else if (curr_quant == "BF16") {
      eltType = FloatType::getBF16(builder.getContext());
    } else {
      eltType = FloatType::getF32(builder.getContext());
    }
    auto type = RankedTensorType::get(shape, eltType);
    auto quantOp = builder.create<tpu::QuantOp>(op->getLoc(), type,
        ArrayRef<Value>{op->getOperand(i)}, ArrayRef<NamedAttribute>{attrs});

    op->setOperand(i, quantOp.getResult());
    if (auto shape_op = dyn_cast_or_null<tpu::ReshapeOp>(op)) {
      std::string name = shape_op.name().str();
      shape_op->setAttr("name", builder.getStringAttr(name + "_quant"));
    }
  }
}

struct TpuConvertSoftmaxToSoftmaxCpu : public RewritePattern {
  TpuConvertSoftmaxToSoftmaxCpu(MLIRContext *context)
      : RewritePattern("tpu.softmax", 1, context) {}
      LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    if(!clQuantMixSoftmax){
      std::vector<Value> operands;
      const int nInputs =  1;
      for (auto i = 0; i < nInputs; ++i) {
        operands.push_back(op->getOperand(i));
      }

      // Return same opValue
      auto loc = op->getLoc();
      auto newOp = rewriter.create<tpu::SoftmaxCpuOp>(loc,
        op->getResult(0).getType(),
        operands,
        op->getAttrs());

      rewriter.replaceOp(op, {newOp});
      return success();
    }

    return failure();
  }
};

// remove lrn_one/lrn_two/lrn_three
struct TpuMergeLrnPattern : public RewritePattern {
  TpuMergeLrnPattern(MLIRContext *context)
      : RewritePattern("tpu.lrn", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto builder = Builder(op->getContext());
    auto lrnOp = cast<tpu::LrnOp>(op);

    auto lrnThreeOp = lrnOp.getOperand(3).getDefiningOp();
    if (false == isa<tpu::LrnThreeOp>(lrnThreeOp)) {
      return failure();
    }
    auto lrnTwoOp = lrnThreeOp->getOperand(0).getDefiningOp();
    assert(isa<tpu::LrnTwoOp>(lrnTwoOp));
    auto lrnOneOp = lrnTwoOp->getOperand(0).getDefiningOp();
    assert(isa<tpu::LrnOneOp>(lrnOneOp));

    // remote operand 3, not use any more
    auto none_op = lrnOp.getOperand(1);
    assert(isa<tpu::NoneOp>(none_op.getDefiningOp()));
    lrnOp.setOperand(3, none_op);
    if (lrnOp.getOpQuant() == "INT8") {
      float sq_thy = getOpThreshold(lrnOneOp);
      float sumsq_thy = getOpThreshold(lrnTwoOp);
      float scale_thy = getOpThreshold(lrnThreeOp);
      lrnOp->setAttr("threshold_parts", builder.getF32ArrayAttr(ArrayRef<float>(
                                            {sq_thy, sumsq_thy, scale_thy})));
    }
    rewriter.replaceOp(lrnThreeOp, {lrnOp});
    rewriter.replaceOp(lrnTwoOp, {lrnOp});
    rewriter.replaceOp(lrnOneOp, {lrnOp});

    return success();
  }
};

struct ConvertClipOpToIdentityOpPattern : public RewritePattern {
  ConvertClipOpToIdentityOpPattern(MLIRContext *context)
      : RewritePattern(tpu::ClipOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto clipOp = cast<tpu::ClipOp>(op);
    auto curr_quant = getOpQuant(op);
    float threshold_y = getOpThreshold(op);
    auto formerOp = clipOp.input().getDefiningOp();

    if (curr_quant != "INT8" ) {
      return failure();
    }
    if (clipOp.fused_relu() == false || isa<tpu::ReluOp>(formerOp)) {
      setOpQuant(op, "BF16");
      return failure();
    }
    setOpThreshold(formerOp, threshold_y);
    formerOp->setAttr("name", clipOp.nameAttr());
    rewriter.replaceOp(op, {clipOp.input()});
    return success();
  }
};

struct TpuQuantInputPassPattern : public RewritePattern {
  TpuQuantInputPassPattern(MLIRContext *context)
      : RewritePattern("tpu.input", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto inputOp = llvm::dyn_cast<tpu::InputOp>(op);
    if (!inputOp) {
      return failure();
    }
    if (nullptr != getNextOp(op)) {
      // only one use
      return failure();
    }
    bool hasBf16 = false;
    bool hasOther = false;
    for (auto &use : op->getResult(0).getUses()) {
      auto nextOp = use.getOwner();
      while (isa<tpu::ReshapeOp>(nextOp)) {
        nextOp = getNextOp(nextOp);
        if (nextOp == nullptr) {
          llvm::errs() << "Warning: after reshape, have no nextop";
          return failure();
        }
      }
      auto quant = getOpQuant(nextOp);
      if (quant == "BF16") {
        hasBf16 = true;
      } else {
        hasOther = true;
      }
      if (hasBf16 && hasOther) {
        break;
      }
    }
    if (hasBf16 && hasOther) {
      // all quant to BF16
      for (auto &use : op->getResult(0).getUses()) {
        auto nextOp = use.getOwner();
        while (isa<tpu::ReshapeOp>(nextOp)) {
          nextOp = getNextOp(nextOp);
        }
        auto quant = getOpQuant(nextOp);
        if (quant == "BF16") {
          continue;
        }
        setOpQuant(nextOp, "BF16");
      }
      return success();
    }

    return failure();
  }
};

struct TpuConvertDilationWeightPattern : public RewritePattern {
  TpuConvertDilationWeightPattern(MLIRContext *context)
      : RewritePattern("tpu.conv_2d", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto convOp = cast<tpu::Conv2DOp>(op);
    LLVM_DEBUG(llvm::errs() << convOp.getOperationName() << ":"
                            << getOpName(op)<< "\n";);

    auto dh = convOp.param().dilation_h().getInt();
    auto dw = convOp.param().dilation_w().getInt();
    const int DILATION_H_MAX = 15;
    const int DILATION_W_MAX = 15;
    if (dh <= DILATION_H_MAX && dw <= DILATION_W_MAX)
      return failure();

    TensorFile *wTF = getWeightTensorFile(op);
    auto filter = readAndDeleteWeightTensor<float>(convOp.filter(), wTF);
    std::vector<int64_t> filterShape;
    filterShape = getTensorShape(convOp.filter());

    int64_t oc = 0;
    int64_t ic = 0;
    int64_t kh = 0;
    int64_t kw = 0;
    if (filterShape.size() == 4) {
      oc = filterShape[0];
      ic = filterShape[1];
      kh = filterShape[2];
      kw = filterShape[3];
    } else if (filterShape.size() == 5) {
      // g, oc/g, ic/g, kh, kw
      oc = filterShape[0] * filterShape[1];
      ic = filterShape[2];
      kh = filterShape[3];
      kw = filterShape[4];
    } else {
      assert(0);
    }

    int insertNumH = 0;
    int insertNumW = 0;
    int newDilationH = dh;
    int newDilationW = dw;
    while(1) {
      insertNumH++;
      newDilationH = (dh - 1 - insertNumH) / (insertNumH + 1) + 1;
      if (((dh - 1 - insertNumH) % (insertNumH + 1) == 0) &&
         newDilationH < DILATION_H_MAX)
        break;
    }

    if (dw > 1) {
      while(1) {
        insertNumW++;
        newDilationW = (dw - 1 - insertNumW) / (insertNumW + 1) + 1;
        if (((dw - 1 - insertNumW) % (insertNumW + 1) == 0) &&
          newDilationW < DILATION_W_MAX)
          break;
      }
    }

    int k_ext_h = (insertNumH + 1) * (kh - 1) + 1;
    int k_ext_w = (insertNumW + 1) * (kw - 1) + 1;
    filterShape[2] = k_ext_h;
    filterShape[3] = k_ext_w;
    auto filterSize = oc * ic * k_ext_h * k_ext_w;
    std::vector<float> newFilter(filterSize, 0);
    for (int i = 0; i < oc * ic; i++) {
      for (int j = 0; j < kh; j++) {
        for (int k = 0; k < kw; k++) {
          auto old_offset = i * kh * kw + j * kw + k;
          auto new_offset = i * k_ext_h * k_ext_w +
                            j * (insertNumH + 1) * k_ext_w +
                            k * (insertNumW + 1);
          newFilter[new_offset] = filter->data()[old_offset];
        }
      }
    }

    // update op
    if (getOpQuant(op) == "INT8")
      addWeightTensorAndUpdateWeightOp<float>(convOp.getOperand(1),
          "dilation", newFilter, filterShape, "INT8", wTF);
    else if (getOpQuant(op) == "BF16")
      addWeightTensorAndUpdateWeightOp<float>(convOp.getOperand(1),
          "dilation", newFilter, filterShape, "BF16", wTF);
    // rewrite pad
    convOp->setAttr("param",
           tpu::ConvParam::get(
                rewriter.getI32IntegerAttr(k_ext_h),
                rewriter.getI32IntegerAttr(k_ext_w),
                convOp.param().stride_h(),
                convOp.param().stride_w(),
                convOp.param().padding(),
                rewriter.getI32IntegerAttr(newDilationH),
                rewriter.getI32IntegerAttr(newDilationW),
                convOp.param().padding_t(),
                convOp.param().padding_b(),
                convOp.param().padding_l(),
                convOp.param().padding_r(),
                convOp.param().group(),
                convOp.param().is_dw(),
                convOp.param().with_bias(),
                convOp.param().ins(),
                convOp.param().pad_value(),
                rewriter.getContext()));

    return success();
  }
};

static void init_quant_layers() {
  gQuantLayers.clear();
  if (clQuantLayerByFile.empty()) {
    return;
  }
  std::regex pattern0("\\S+\\s+\\S+");
  std::regex pattern1("\\S+");
  std::regex info_pattern("#.*");
  std::ifstream infile(clQuantLayerByFile);
  std::string line;
  while (std::getline(infile, line)) {
    if (line.back() == '\r') {
      line.pop_back();
    }
    std::istringstream iss(line);
    std::string name;
    std::string mode;
    if (std::regex_match(line, info_pattern)) {
      LLVM_DEBUG(llvm::errs() << "\n  infomation  " << line << "\n");
      continue;
    }
    if (std::regex_match(line, pattern0)) {
      iss >> name;
      iss >> mode;
      gQuantLayers[name] = qmode(mode);
      continue;
    }
    if (std::regex_match(line, pattern1)) {
      iss >> name;
      gQuantLayers[name] = QUANT_BF16;
      continue;
    }
    if (std::regex_match(line, info_pattern)) {
      continue;
    }
    llvm::errs() << "Error, mix quant file [" << line << "]\n";
    assert(false);
  }
}

static bool quant_no_need(Operation *op) {
  auto quantOp = llvm::dyn_cast_or_null<tpu::TpuOpQuantInterface>(op);
  if (!quantOp) {
    return true;
  }
  if (op->getName().getDialect()->getNamespace() != "tpu" ||
      isa<tpu::ReshapeOp>(op) || isa<tpu::InputOp>(op) ||
      isa<tpu::InstanceNormOp>(op) || isa<tpu::ROIPoolingOp>(op) ||
      isa<tpu::SoftmaxCpuOp>(op)) {
    return true;
  } else if (isa<tpu::CustomOp>(op) && !cast<tpu::CustomOp>(op).do_quant()) {
    return true;
  } else if ((!clQuantMixSoftmax) && isa<tpu::SoftmaxOp>(op)) {
    return true;
  }
  return false;
}

template <typename T>
static bool has_const(Operation *op) {
  auto cast_op = dyn_cast_or_null<T>(op);
  if (cast_op == nullptr) {
    return false;
  }
  for (auto input : cast_op.inputs()) {
    if (isa<tpu::LoadWeightOp>(input.getDefiningOp())) {
      return true;
    }
  }
  return false;
}

static void quant_for_special(Operation *op) {
  if (getOpQuant(op) == "BF16") {
    // no op only use INT8
    return;
  }
  if (has_const<tpu::EltwiseMulOp>(op) || has_const<tpu::EltwiseAddOp>(op) ||
      has_const<tpu::ConcatOp>(op) || has_const<tpu::BroadcastMulOp>(op) ||
      has_const<tpu::BroadcastAddOp>(op)) {
  } else if (clQuantMixSoftmax && isa<tpu::SoftmaxOp>(op)) {
  } else if (isa<tpu::CustomOp>(op) && cast<tpu::CustomOp>(op).tpu() == true) {
  } else if (isa<tpu::LayerNormOp>(op) || isa<tpu::ConvFcOp>(op) ||
             isa<tpu::GruOp>(op) || isa<tpu::LstmOp>(op) ||
             isa<tpu::StdOp>(op) || isa<tpu::EmbeddingOp>(op) ||
             isa<tpu::QuadraticSumOp>(op) || isa<tpu::Conv3DOp>(op) ||
             isa<tpu::ReduceL2Op>(op) || isa<tpu::MatchTemplateOp>(op)) {
  } else {
    return;
  }
  setOpQuant(op, "BF16");
  setOpQuantParamType(op, "MIX_BF16");
}

static void quant_by_layers(Operation *op) {
  auto tpuOp = llvm::dyn_cast_or_null<tpu::TpuOpCommonInterface>(op);
  if (!tpuOp) {
    return;
  }
  std::string op_name = tpuOp.getOpName().str();

  if (gQuantLayers.end() == gQuantLayers.find(op_name)) {
    return;
  }
  auto qmode = gQuantLayers[op_name];
  if (qmode == gDefaultQuant) {
    return;
  }
  switch (qmode) {
  case QUANT_INT8:
    setOpQuant(op, "INT8");
    break;
  case QUANT_BF16:
    setOpQuant(op, "BF16");
    break;
  case QUANT_MIX_BF16:
    setOpQuant(op, "BF16");
    setOpQuantParamType(op, "MIX_BF16");
    break;
  default:
    llvm_unreachable("unknown mode");
  }
}

static void quant_by_default(Operation *op) {
  switch (gDefaultQuant) {
  case QUANT_INT8:
    setOpQuant(op, "INT8");
    break;
  case QUANT_BF16:
    setOpQuant(op, "BF16");
    break;
  case QUANT_MIX_BF16:
    setOpQuant(op, "BF16");
    setOpQuantParamType(op, "MIX_BF16");
    break;
  default:
    llvm_unreachable("unknown mode");
    break;
  }
}

class TpuQuantPass : public mlir::PassWrapper<TpuQuantPass, FunctionPass> {

public:
  explicit TpuQuantPass() {}

  void runOnFunction() override {
    MInfo::getChipInfo(getFunction());
    assert(MInfo::version && "refer to set-chip");

    auto fn = getFunction();
    auto *context = &getContext();

    // read mix precision from file, seperated by \n
    init_quant_layers();
    gDefaultQuant = qmode(clQuantMode);

    // mark quant mode
    fn.walk([&](Operation *op) {
      if (quant_no_need(op)) {
      } else {
        quant_by_default(op);
        quant_by_layers(op);
        quant_for_special(op);
      }
    });

    OwningRewritePatternList patterns;

    patterns.insert<ConvertClipOpToIdentityOpPattern>(context);
    // patch for dialation > 15
    patterns.insert<TpuConvertDilationWeightPattern>(context);
    patterns.insert<TpuMergeLrnPattern>(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));

    // if input has more than one use, and do different quant,
    // then quant to bf16 all.
    patterns.clear();
    patterns.insert<TpuQuantInputPassPattern>(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));

    // do quant
    fn.walk([&](Operation *op) {
      if (op->getName().getDialect()->getNamespace() != "tpu"
          || isa<tpu::InputOp>(op)
          || isa<tpu::QuantOp>(op)
          || isa<tpu::InstanceNormOp>(op)
          || isa<tpu::ReshapeOp>(op)
          || isa<tpu::ROIPoolingOp>(op)
          || isa<tpu::SoftmaxCpuOp>(op)) {
        // pass
      } else if (isa<tpu::CustomOp>(op) && cast<tpu::CustomOp>(op).tpu() == false) {
        assert(getOpQuant(op) == "BF16");
        setOpResultType(op->getResult(0), FloatType::getF32(op->getContext()));
      }
#if 0
// custom op use default quantbf16 now
      else if (auto castOp = llvm::dyn_cast<tpu::CustomOp>(op)) {
        auto builder = Builder(context);
        if (getOpQuant(op) != "NONE") {
          cvi::OpParam param, quant;
          auto operation_name = castOp.operation_name().str();
          float prevThreshold = getPreviousOpThreshold(op);
          convertAttributesToOpParam(castOp.param(), param);
          convertAttributesToOpParam(castOp.quant(), quant);
          cvi::CustomOpPlugin *plugin = cvi::CustomOpPlugin::load();
          assert(plugin);
          if (getOpQuant(op) == "INT8") {
            plugin->int8Quant(operation_name.c_str(), param, &quant, prevThreshold);
            setOpResultType(op->getResult(0), IntegerType::get(op->getContext(), 8, IntegerType::Signed));
          } else if (getOpQuant(op) == "BF16") {
            plugin->bf16Quant(operation_name.c_str(), param, &quant, prevThreshold);
            setOpResultType(op->getResult(0), FloatType::getF32(op->getContext()));
          }
          std::vector<NamedAttribute> newParam, newQuant;
          convertOpParamToAttributes(builder, param, newParam);
          convertOpParamToAttributes(builder, quant, newQuant);
          castOp->setAttr("param", DictionaryAttr::get(context, newParam));
          castOp->setAttr("quant", DictionaryAttr::get(context, newQuant));
        }
      }
#endif
      else if (auto quantOp = llvm::dyn_cast<tpu::TpuOpQuantInterface>(op)) {
        if (getOpQuant(op) == "INT8" || getOpQuant(op) == "UINT8") {
          auto ret = quantOp.quantizeInt8();
          assert(!failed(ret));
        } else if (getOpQuant(op) == "BF16") {
          auto ret = quantOp.quantizeBf16();
          assert(!failed(ret));
        } else {
          llvm::errs() << "assert:" << op->getName() << "\n";
          assert(false);
        }
      }
    });

    // insert QuantOp if quant types don't equal.
    fn.walk([&](Operation *op) {
      if ((op->getName().getDialect()->getNamespace() != "tpu"
           && !isa<ReturnOp>(op))
          || isa<tpu::InputOp>(op)
          || isa<tpu::WeightFileOp>(op)
          || isa<tpu::LoadWeightOp>(op)
          || isa<tpu::NoneOp>(op)) {
      } else {
        insertQuantOp(op);
      }
    });

    // To make ReshapeOp's result element type same as
    // operand's after quantization

    fn.walk([&](tpu::ReshapeOp op) {
      auto _op = op.getOperation();
      auto eltType = _op->getOperand(0).getType().cast<TensorType>().getElementType();
      auto shape = _op->getResult(0).getType().cast<TensorType>().getShape();
      auto type = RankedTensorType::get(shape, eltType);
      _op->getResult(0).setType(type);
    });

    // gen special operations
    patterns.clear();
    patterns.insert<
      TpuConvertSoftmaxToSoftmaxCpu
    >(context);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));
  }
};

std::unique_ptr<mlir::Pass> mlir::createTpuQuantPass() {
  return std::make_unique<TpuQuantPass>();
}
