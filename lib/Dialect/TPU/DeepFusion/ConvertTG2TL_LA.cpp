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
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/Dialect/TPU/MachineInfo.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/MathExtras.h"
#include "SimpleAnalysis.h"

#define DEBUG_TYPE "deep-fusion-tg2tl-la"

using namespace mlir;

namespace {

static int64_t getBatchSize(Operation *op) {
  auto resultType = op->getResult(0)->getType();
  auto tensorType = resultType.dyn_cast<RankedTensorType>();
  auto batch = tensorType.getShape()[0];
  return (int64_t)batch;
}

struct TpuTG2TLConv2DOpPattern : public RewritePattern {
  TpuTG2TLConv2DOpPattern(MLIRContext *context)
      : RewritePattern("tpu.tg_int8_pc_conv_2d", 1, context) {}
  PatternMatchResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    if (getBatchSize(opInst) != 1) {
      return matchFailure();
    }
    auto op = cast<tpu::TG_INT8_PC_Conv2DOp>(opInst);
    assert(op);

    uint64_t totalPerLane = SimpleConv2DMemoryUsageAnalysis(op, nullptr);
    if (totalPerLane > MInfo::lmem_per_lane) {
      LLVM_DEBUG(llvm::errs() << "TG2TL_LA: " << op.name()
                   << ", layer ID " << op.layer_id()
                   << ", SKIP, lmem " << totalPerLane
                   << " needed\n";);
      return matchFailure();
    }

    LLVM_DEBUG(llvm::errs() << "TG2TL_LA: " << op.name()
                 << ", layer ID " << op.layer_id() << "\n";);

    // break leaky relu fuse
    if (opInst->getResult(0)->hasOneUse()) {
      auto next_op = getNextOp(opInst);
      if (auto lreluOp = dyn_cast<tpu::TG_INT8_LeakyReluOp>(next_op)) {
        lreluOp.setAttr("fuse_prev", rewriter.getBoolAttr(false));
      }
    }

    // convert to TL_LA_Conv2DOp
    assert(op.getNumOperands() == 3 && "support 3 inputs only");
    std::vector<Value *> newOperands;
    newOperands.push_back(op.getOperand(0));
    newOperands.push_back(op.getOperand(1));
    newOperands.push_back(op.getOperand(2));

    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("param", op.paramAttr()));
    attrs.push_back(rewriter.getNamedAttr("gaddr", op.gaddrAttr()));
    attrs.push_back(rewriter.getNamedAttr("name", op.nameAttr()));
    attrs.push_back(rewriter.getNamedAttr("layer_id", op.layer_idAttr()));
    if(op.do_ic_alignment().hasValue()){
      attrs.push_back(rewriter.getNamedAttr("do_ic_alignment", rewriter.getBoolAttr(op.do_ic_alignment().getValue())));
    }

    if (op.fuse_next()) {
      attrs.push_back(rewriter.getNamedAttr("do_leaky_relu", op.fuse_nextAttr()));
      if (op.rshift_pos().hasValue())
        attrs.push_back(rewriter.getNamedAttr("rshift_pos", op.rshift_posAttr()));
      if (op.m_i8_pos().hasValue())
        attrs.push_back(rewriter.getNamedAttr("m_i8_pos", op.m_i8_posAttr()));
      if (op.rshift_neg().hasValue())
        attrs.push_back(rewriter.getNamedAttr("rshift_neg", op.rshift_negAttr()));
      if (op.m_i8_neg().hasValue())
        attrs.push_back(rewriter.getNamedAttr("m_i8_neg", op.m_i8_negAttr()));
    }

    if (op.buffer_reused().hasValue())
      attrs.push_back(rewriter.getNamedAttr("buffer_reused", op.buffer_reusedAttr()));

    rewriter.replaceOpWithNewOp<tpu::TL_LA_Conv2DOp>(
        op, op.getResult()->getType(),
        ArrayRef<Value *>{newOperands}, ArrayRef<NamedAttribute>{attrs});

    return matchSuccess();
  }
};

template<typename OpTy, typename OpTy2>
struct TpuTG2TLElewiseOpPattern : public RewritePattern {
  TpuTG2TLElewiseOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    if (getBatchSize(opInst) != 1) {
      return matchFailure();
    }
    auto op = cast<OpTy>(opInst);
    assert(op);

    uint64_t totalPerLane = SimpleEltwiseMemoryUsageAnalysis(op, nullptr);
    if (totalPerLane > MInfo::lmem_per_lane) {
      LLVM_DEBUG(llvm::errs() << "TG2TL_LA: " << op.name()
                   << ", layer ID " << op.layer_id()
                   << ", SKIP, lmem " << totalPerLane
                   << " needed\n";);
      return matchFailure();
    }

    // Check whether operand ConvOp has enough memory
    for (auto operand : opInst->getOperands()) {
      auto operandOp = operand->getDefiningOp();
      if (auto convOp = dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(operandOp)) {
        uint64_t totalPerLane =
            SimpleConv2DMemoryUsageAnalysis(convOp, nullptr);
        if (totalPerLane > MInfo::lmem_per_lane) {
          LLVM_DEBUG(llvm::errs() << "TG2TL_LA: " << op.name()
                     << ", layer ID " << op.layer_id()
                     << ", operandOp " << convOp.name()
                     << ", SKIP, lmem " << totalPerLane
                     << " needed\n";);
          return matchFailure();
        }
      }
    }

    if (1) {
      LLVM_DEBUG(llvm::errs() << "TG2TL_LA: " << op.name()
                   << ", layer ID " << op.layer_id() << "\n";);

      assert(op.getNumOperands() == 2 && "support 2 inputs only");
      std::vector<Value *> newOperands;
      newOperands.push_back(op.getOperand(0));
      newOperands.push_back(op.getOperand(1));

      std::vector<NamedAttribute> attrs;

      attrs.push_back(rewriter.getNamedAttr("rshift", op.rshiftAttr()));
      if(op.m_i8_inputs().hasValue()) {
        attrs.push_back(rewriter.getNamedAttr("m_i8_inputs", op.m_i8_inputsAttr()));
      }

      attrs.push_back(rewriter.getNamedAttr("do_relu",
          rewriter.getBoolAttr(op.do_relu())));
      if (op.do_early_stride()) {
        attrs.push_back(rewriter.getNamedAttr("do_early_stride", rewriter.getBoolAttr(true)));
        attrs.push_back(rewriter.getNamedAttr("early_stride_h", op.early_stride_hAttr()));
        attrs.push_back(rewriter.getNamedAttr("early_stride_w", op.early_stride_wAttr()));
      }

      uint32_t la_invalid = 0xffffffff;
      attrs.push_back(rewriter.getNamedAttr("lm_layout", rewriter.getStringAttr("NONE")));
      attrs.push_back(rewriter.getNamedAttr("la_input", rewriter.getI32IntegerAttr(la_invalid)));
      attrs.push_back(rewriter.getNamedAttr("la_working", rewriter.getI32IntegerAttr(la_invalid)));
      attrs.push_back(rewriter.getNamedAttr("la_output", rewriter.getI32IntegerAttr(la_invalid)));
      attrs.push_back(rewriter.getNamedAttr("tl_load_flag", rewriter.getBoolAttr(true)));
      attrs.push_back(rewriter.getNamedAttr("tl_store_flag", rewriter.getBoolAttr(true)));

      attrs.push_back(rewriter.getNamedAttr("gaddr", op.gaddrAttr()));
      attrs.push_back(rewriter.getNamedAttr("name", op.nameAttr()));
      attrs.push_back(rewriter.getNamedAttr("layer_id", op.layer_idAttr()));

      if (op.buffer_reused().hasValue())
        attrs.push_back(rewriter.getNamedAttr("buffer_reused", op.buffer_reusedAttr()));
      if(op.m_i32_output().hasValue())
        attrs.push_back(rewriter.getNamedAttr("m_i32_output", op.m_i32_outputAttr()));

      rewriter.replaceOpWithNewOp<OpTy2>(
          op, op.getResult()->getType(),
          ArrayRef<Value *>{newOperands}, ArrayRef<NamedAttribute>{attrs});
      return matchSuccess();
    }
  }
};

struct TpuTG2TLLutOpPattern : public RewritePattern {
  TpuTG2TLLutOpPattern(MLIRContext *context)
      : RewritePattern("tpu.tg_int8_lut", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    if (getBatchSize(opInst) != 1) {
      return matchFailure();
    }
    auto op = cast<tpu::TG_INT8_LutOp>(opInst);
    assert(op);

    uint64_t totalPerLane = SimpleLutMemoryUsageAnalysis(op, nullptr);
    if (totalPerLane > MInfo::lmem_per_lane) {
      LLVM_DEBUG(llvm::errs() << "TG2TL_LA: " << op.name()
                   << ", layer ID " << op.layer_id()
                   << ", SKIP, lmem " << totalPerLane
                   << " needed\n";);
      return matchFailure();
    }

    // Check whether operand ConvOp has enough memory
    for (auto operand : opInst->getOperands()) {
      auto operandOp = operand->getDefiningOp();
      if (auto convOp = dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(operandOp)) {
        uint64_t totalPerLane =
            SimpleConv2DMemoryUsageAnalysis(convOp, nullptr);
        if (totalPerLane > MInfo::lmem_per_lane) {
          LLVM_DEBUG(llvm::errs() << "TG2TL_LA: " << op.name()
                     << ", layer ID " << op.layer_id()
                     << ", operandOp " << convOp.name()
                     << ", SKIP, lmem " << totalPerLane
                     << " needed\n";);
          return matchFailure();
        }
      }
    }

    if (1) {
      LLVM_DEBUG(llvm::errs() << "TG2TL_LA: " << op.name()
                   << ", layer ID " << op.layer_id() << "\n";);

      assert(op.getNumOperands() == 3);
      std::vector<Value *> newOperands;
      newOperands.push_back(op.getOperand(0));
      newOperands.push_back(op.getOperand(1));
      newOperands.push_back(op.getOperand(2));

      std::vector<NamedAttribute> attrs;

      uint32_t la_invalid = 0xffffffff;
      attrs.push_back(rewriter.getNamedAttr("lm_layout", rewriter.getStringAttr("NONE")));
      attrs.push_back(rewriter.getNamedAttr("la_input", rewriter.getI32IntegerAttr(la_invalid)));
      attrs.push_back(rewriter.getNamedAttr("la_working", rewriter.getI32IntegerAttr(la_invalid)));
      attrs.push_back(rewriter.getNamedAttr("la_output", rewriter.getI32IntegerAttr(la_invalid)));
      attrs.push_back(rewriter.getNamedAttr("tl_load_flag", rewriter.getBoolAttr(true)));
      attrs.push_back(rewriter.getNamedAttr("tl_store_flag", rewriter.getBoolAttr(true)));

      attrs.push_back(rewriter.getNamedAttr("gaddr", op.gaddrAttr()));
      attrs.push_back(rewriter.getNamedAttr("name", op.nameAttr()));
      attrs.push_back(rewriter.getNamedAttr("layer_id", op.layer_idAttr()));

      if (op.buffer_reused().hasValue())
        attrs.push_back(rewriter.getNamedAttr("buffer_reused", op.buffer_reusedAttr()));

      rewriter.replaceOpWithNewOp<tpu::TL_LutOp>(
          op, op.getResult()->getType(),
          ArrayRef<Value *>{newOperands}, ArrayRef<NamedAttribute>{attrs});
      return matchSuccess();
    }
  }
};

struct TpuTG2TLBroadcastMulOpPattern : public RewritePattern {
  TpuTG2TLBroadcastMulOpPattern(MLIRContext *context)
      : RewritePattern("tpu.tg_int8_broadcast_mul", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    if (getBatchSize(opInst) != 1) {
      return matchFailure();
    }
    auto op = cast<tpu::TG_INT8_BroadcastMulOp>(opInst);
    assert(op);

    uint64_t totalPerLane = SimpleBroadcastMulMemoryUsageAnalysis(op, nullptr);
    if (totalPerLane > MInfo::lmem_per_lane) {
      LLVM_DEBUG(llvm::errs() << "TG2TL_LA: " << op.name()
                   << ", layer ID " << op.layer_id()
                   << ", SKIP, lmem " << totalPerLane
                   << " needed\n";);
      return matchFailure();
    }

    // Check whether operand ConvOp has enough memory
    for (auto operand : opInst->getOperands()) {
      auto operandOp = operand->getDefiningOp();
      if (auto convOp = dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(operandOp)) {
        uint64_t totalPerLane =
            SimpleConv2DMemoryUsageAnalysis(convOp, nullptr);
        if (totalPerLane > MInfo::lmem_per_lane) {
          LLVM_DEBUG(llvm::errs() << "TG2TL_LA: " << op.name()
                     << ", layer ID " << op.layer_id()
                     << ", operandOp " << convOp.name()
                     << ", SKIP, lmem " << totalPerLane
                     << " needed\n";);
          return matchFailure();
        }
      }
    }

    if (1) {
      LLVM_DEBUG(llvm::errs() << "TG2TL_LA: " << op.name()
                   << ", layer ID " << op.layer_id() << "\n";);

      assert(op.getNumOperands() == 3);
      std::vector<Value *> newOperands;
      newOperands.push_back(op.getOperand(0));
      newOperands.push_back(op.getOperand(1));
      newOperands.push_back(op.getOperand(2));

      std::vector<NamedAttribute> attrs;

      uint32_t la_invalid = 0xffffffff;
      attrs.push_back(rewriter.getNamedAttr("lm_layout", rewriter.getStringAttr("NONE")));
      attrs.push_back(rewriter.getNamedAttr("la_input", rewriter.getI32IntegerAttr(la_invalid)));
      attrs.push_back(rewriter.getNamedAttr("la_working", rewriter.getI32IntegerAttr(la_invalid)));
      attrs.push_back(rewriter.getNamedAttr("la_output", rewriter.getI32IntegerAttr(la_invalid)));
      attrs.push_back(rewriter.getNamedAttr("tl_load_flag", rewriter.getBoolAttr(true)));
      attrs.push_back(rewriter.getNamedAttr("tl_store_flag", rewriter.getBoolAttr(true)));

      attrs.push_back(rewriter.getNamedAttr("param",
        tpu::ConvParam::get(
            rewriter.getI32IntegerAttr(1),
            rewriter.getI32IntegerAttr(1),
            rewriter.getStringAttr("VALID"),
            rewriter.getI32IntegerAttr(1),
            rewriter.getI32IntegerAttr(1),
            rewriter.getI32IntegerAttr(0), // pd_t
            rewriter.getI32IntegerAttr(0), // pd_b
            rewriter.getI32IntegerAttr(0), // pd_l
            rewriter.getI32IntegerAttr(0), // pd_r
            rewriter.getI32IntegerAttr(1),
            rewriter.getBoolAttr(true),    // is_dw
            rewriter.getBoolAttr(false),   // with_bias
            rewriter.getBoolAttr(false),   // do_relu
            rewriter.getContext())));
      attrs.push_back(rewriter.getNamedAttr("gaddr", op.gaddrAttr()));
      attrs.push_back(rewriter.getNamedAttr("name", op.nameAttr()));
      attrs.push_back(rewriter.getNamedAttr("layer_id", op.layer_idAttr()));

      if (op.buffer_reused().hasValue())
        attrs.push_back(rewriter.getNamedAttr("buffer_reused", op.buffer_reusedAttr()));

      rewriter.replaceOpWithNewOp<tpu::TL_BroadcastMulOp>(
          op, op.getResult()->getType(),
          ArrayRef<Value *>{newOperands}, ArrayRef<NamedAttribute>{attrs});
      return matchSuccess();
    }
  }
};

template<typename OpTy, typename OpTy2>
struct TpuTG2TLPoolOpPattern : public RewritePattern {
  TpuTG2TLPoolOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    if (getBatchSize(opInst) != 1) {
      return matchFailure();
    }
    auto op = cast<OpTy>(opInst);
    assert(op);

    uint64_t totalPerLane = SimpleIOMemoryUsageAnalysis(op, nullptr);
    if (totalPerLane > MInfo::lmem_per_lane) {
      LLVM_DEBUG(llvm::errs() << "TG2TL_LA: " << op.name()
                   << ", layer ID " << op.layer_id()
                   << ", SKIP, lmem " << totalPerLane
                   << " needed\n";);
      return matchFailure();
    }

    // Check whether operand ConvOp has enough memory
    // ????
    for (auto operand : opInst->getOperands()) {
      auto operandOp = operand->getDefiningOp();
      if (auto convOp = dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(operandOp)) {
        uint64_t totalPerLane =
            SimpleConv2DMemoryUsageAnalysis(convOp, nullptr);
        if (totalPerLane > MInfo::lmem_per_lane) {
          LLVM_DEBUG(llvm::errs() << "TG2TL_LA: " << op.name()
                     << ", layer ID " << op.layer_id()
                     << ", operandOp " << convOp.name()
                     << ", SKIP, lmem " << totalPerLane
                     << " needed\n";);
          return matchFailure();
        }
      }
    }

    if (1) {
      LLVM_DEBUG(llvm::errs() << "TG2TL_LA: " << op.name()
                   << ", layer ID " << op.layer_id() << "\n";);

      std::vector<Value *> newOperands;
      newOperands.push_back(op.getOperand());

      std::vector<NamedAttribute> attrs;

      if(op.rshift().hasValue()) {
        attrs.push_back(rewriter.getNamedAttr("rshift", op.rshiftAttr()));
      }
      if(op.m_i8().hasValue()) {
        attrs.push_back(rewriter.getNamedAttr("m_i8", op.m_i8Attr()));
      }
      attrs.push_back(rewriter.getNamedAttr("param", op.paramAttr()));

      uint32_t la_invalid = 0xffffffff;
      attrs.push_back(rewriter.getNamedAttr("lm_layout", rewriter.getStringAttr("NONE")));
      attrs.push_back(rewriter.getNamedAttr("la_input", rewriter.getI32IntegerAttr(la_invalid)));
      attrs.push_back(rewriter.getNamedAttr("la_working", rewriter.getI32IntegerAttr(la_invalid)));
      attrs.push_back(rewriter.getNamedAttr("la_output", rewriter.getI32IntegerAttr(la_invalid)));
      attrs.push_back(rewriter.getNamedAttr("tl_load_flag", rewriter.getBoolAttr(true)));
      attrs.push_back(rewriter.getNamedAttr("tl_store_flag", rewriter.getBoolAttr(true)));

      attrs.push_back(rewriter.getNamedAttr("gaddr", op.gaddrAttr()));
      attrs.push_back(rewriter.getNamedAttr("name", op.nameAttr()));
      attrs.push_back(rewriter.getNamedAttr("layer_id", op.layer_idAttr()));

      if (op.buffer_reused().hasValue())
        attrs.push_back(rewriter.getNamedAttr("buffer_reused", op.buffer_reusedAttr()));

      rewriter.replaceOpWithNewOp<OpTy2>(
          op, op.getResult()->getType(),
          ArrayRef<Value *>{newOperands}, ArrayRef<NamedAttribute>{attrs});
      return matchSuccess();
    }
  }
};

class DeepFusionTG2TL_LA : public FunctionPass<DeepFusionTG2TL_LA> {
public:
  explicit DeepFusionTG2TL_LA() {}

  void runOnFunction() override {
    std::string getRunChipType;
    MInfo Machineinfo;
    get_cvichip_name(getRunChipType);
    Machineinfo.getChipInfo(getRunChipType.c_str());
    assert(MInfo::version && "refer to set-chip");
    auto fn = getFunction();
    auto *context = &getContext();
    OwningRewritePatternList patterns;
    patterns.insert<
      TpuTG2TLConv2DOpPattern,
      TpuTG2TLElewiseOpPattern<tpu::TG_INT8_EltwiseAddOp, tpu::TL_EltwiseAddOp>,
      TpuTG2TLElewiseOpPattern<tpu::TG_INT8_EltwiseMulOp, tpu::TL_EltwiseMulOp>,
      TpuTG2TLLutOpPattern,
      TpuTG2TLPoolOpPattern<tpu::TG_INT8_PoolAvg2DOp, tpu::TL_PoolAvg2DOp>,
      TpuTG2TLBroadcastMulOpPattern
    >(context);
    applyPatternsGreedily(fn, patterns);
  }
};

} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createDeepFusionTG2TL_LA() {
  return std::make_unique<DeepFusionTG2TL_LA>();
}

static PassRegistration<DeepFusionTG2TL_LA>
    pass("deep-fusion-tg2tl-la",
         "convert Ops from TG to TL, "
         "this is a trivial conversion, yielding no improvement at all");
