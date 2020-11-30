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
#include "tpuc/TPUOperationSupport.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "tpuc/Passes.h"
#include "tpuc/MachineInfo.h"
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
#include "tpuc/SimpleAnalysis.h"

#define DEBUG_TYPE "deep-fusion-tg2tl-la"

using namespace mlir;

namespace {

struct TpuTG2TLConv2DOpPattern : public RewritePattern {
  TpuTG2TLConv2DOpPattern(MLIRContext *context)
      : RewritePattern("tpu.tg_int8_pc_conv_2d", 1, context) {}

  LogicalResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    auto op = cast<tpu::TG_INT8_PC_Conv2DOp>(opInst);
    assert(op);

    int64_t input_size;
    std::vector<int64_t> shape;
    getTensorShapeAndSize(op.getOperand(0), shape, input_size);
    if (shape.size() > 1 && shape[1] == 1) {
      // workaround: input_c == 1 not support
      return failure();
    }

    uint64_t totalPerLane = SimpleConv2DMemoryUsageAnalysis(op, nullptr);
    if (totalPerLane > MInfo::lmem_per_lane) {
      LLVM_DEBUG(llvm::errs() << "TG2TL_LA: " << op.name()
                   << ", layer ID " << getOpLayerId(opInst)
                   << ", SKIP, lmem " << totalPerLane
                   << " needed\n";);
      return failure();
    }

    LLVM_DEBUG(llvm::errs() << "TG2TL_LA: " << op.name()
                 << ", layer ID " << getOpLayerId(opInst) << "\n";);

    // convert to TL_LA_Conv2DOp
    assert(op.getNumOperands() == 3 && "support 3 inputs only");
    std::vector<Value> newOperands;
    newOperands.push_back(op.getOperand(0));
    newOperands.push_back(op.getOperand(1));
    newOperands.push_back(op.getOperand(2));

    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("param", op.paramAttr()));
    attrs.push_back(rewriter.getNamedAttr("name", op.nameAttr()));
    if(op.do_ic_alignment().hasValue()){
      attrs.push_back(rewriter.getNamedAttr("do_ic_alignment", rewriter.getBoolAttr(op.do_ic_alignment().getValue())));
    }

    if (op.do_leaky_relu()) {
      attrs.push_back(rewriter.getNamedAttr("do_leaky_relu", op.do_leaky_reluAttr()));
      if (op.rshift_pos().hasValue())
        attrs.push_back(rewriter.getNamedAttr("rshift_pos", op.rshift_posAttr()));
      if (op.m_i8_pos().hasValue())
        attrs.push_back(rewriter.getNamedAttr("m_i8_pos", op.m_i8_posAttr()));
      if (op.rshift_neg().hasValue())
        attrs.push_back(rewriter.getNamedAttr("rshift_neg", op.rshift_negAttr()));
      if (op.m_i8_neg().hasValue())
        attrs.push_back(rewriter.getNamedAttr("m_i8_neg", op.m_i8_negAttr()));
    }

    rewriter.replaceOpWithNewOp<tpu::TL_LA_Conv2DOp>(
        op, op.getResult().getType(),
        ArrayRef<Value>{newOperands}, ArrayRef<NamedAttribute>{attrs});

    return success();
  }
};

template<typename OpTy, typename OpTy2>
struct TpuTG2TLElewiseOpPattern : public RewritePattern {
  TpuTG2TLElewiseOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    auto op = cast<OpTy>(opInst);
    assert(op);

    uint64_t totalPerLane = SimpleEltwiseMemoryUsageAnalysis(op, nullptr);
    if (totalPerLane > MInfo::lmem_per_lane) {
      LLVM_DEBUG(llvm::errs() << "TG2TL_LA: " << op.name()
                   << ", layer ID " << getOpLayerId(opInst)
                   << ", SKIP, lmem " << totalPerLane
                   << " needed\n";);
      return failure();
    }

    // workaround: if add is after mul, not support now
    if (isa<tpu::TG_INT8_EltwiseMulOp>(opInst)) {
      auto next_op = getNextOp(opInst);
      if (next_op != nullptr && isa<tpu::TG_INT8_EltwiseAddOp>(next_op)) {
        return failure();
      }
    } else if(isa<tpu::TG_INT8_EltwiseAddOp>(opInst)) {
      for (auto operand : opInst->getOperands()) {
        auto operandOp = operand.getDefiningOp();
        if(operandOp != nullptr && isa<tpu::TG_INT8_EltwiseMulOp>(operandOp)){
          return failure();
        }
      }
    }

    // Check whether operand ConvOp has enough memory
    for (auto operand : opInst->getOperands()) {
      auto operandOp = operand.getDefiningOp();
      if (auto convOp = dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(operandOp)) {
        uint64_t totalPerLane =
            SimpleConv2DMemoryUsageAnalysis(convOp, nullptr);
        if (totalPerLane > MInfo::lmem_per_lane) {
          LLVM_DEBUG(llvm::errs() << "TG2TL_LA: " << op.name()
                     << ", layer ID " << getOpLayerId(opInst)
                     << ", operandOp " << convOp.name()
                     << ", SKIP, lmem " << totalPerLane
                     << " needed\n";);
          return failure();
        }
      }
    }

    if (1) {
      LLVM_DEBUG(llvm::errs() << "TG2TL_LA: " << op.name()
                   << ", layer ID " << getOpLayerId(opInst) << "\n";);

      assert(op.getNumOperands() == 2 && "support 2 inputs only");
      std::vector<Value> newOperands;
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

      attrs.push_back(rewriter.getNamedAttr("name", op.nameAttr()));

      if(op.m_i32_output().hasValue())
        attrs.push_back(rewriter.getNamedAttr("m_i32_output", op.m_i32_outputAttr()));

      rewriter.replaceOpWithNewOp<OpTy2>(
          op, op.getResult().getType(),
          ArrayRef<Value>{newOperands}, ArrayRef<NamedAttribute>{attrs});
      return success();
    }
  }
};

struct TpuTG2TLLutOpPattern : public RewritePattern {
  TpuTG2TLLutOpPattern(MLIRContext *context)
      : RewritePattern("tpu.tg_int8_lut", 1, context) {}

  LogicalResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    auto op = cast<tpu::TG_INT8_LutOp>(opInst);
    assert(op);

    uint64_t totalPerLane = SimpleLutMemoryUsageAnalysis(op, nullptr);
    if (totalPerLane > MInfo::lmem_per_lane) {
      LLVM_DEBUG(llvm::errs() << "TG2TL_LA: " << op.name()
                   << ", layer ID " << getOpLayerId(opInst)
                   << ", SKIP, lmem " << totalPerLane
                   << " needed\n";);
      return failure();
    }

    // Check whether operand ConvOp has enough memory
    for (auto operand : opInst->getOperands()) {
      auto operandOp = operand.getDefiningOp();
      if (auto convOp = dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(operandOp)) {
        uint64_t totalPerLane =
            SimpleConv2DMemoryUsageAnalysis(convOp, nullptr);
        if (totalPerLane > MInfo::lmem_per_lane) {
          LLVM_DEBUG(llvm::errs() << "TG2TL_LA: " << op.name()
                     << ", layer ID " << getOpLayerId(opInst)
                     << ", operandOp " << convOp.name()
                     << ", SKIP, lmem " << totalPerLane
                     << " needed\n";);
          return failure();
        }
      }
    }

    if (1) {
      LLVM_DEBUG(llvm::errs() << "TG2TL_LA: " << op.name()
                   << ", layer ID " << getOpLayerId(opInst) << "\n";);

      assert(op.getNumOperands() == 3);
      std::vector<Value> newOperands;
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

      attrs.push_back(rewriter.getNamedAttr("name", op.nameAttr()));

      rewriter.replaceOpWithNewOp<tpu::TL_LutOp>(
          op, op.getResult().getType(),
          ArrayRef<Value>{newOperands}, ArrayRef<NamedAttribute>{attrs});
      return success();
    }
  }
};

struct TpuTG2TLBroadcastMulOpPattern : public RewritePattern {
  TpuTG2TLBroadcastMulOpPattern(MLIRContext *context)
      : RewritePattern("tpu.tg_int8_broadcast_mul", 1, context) {}

  LogicalResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    auto op = cast<tpu::TG_INT8_BroadcastMulOp>(opInst);
    assert(op);

    uint64_t totalPerLane = SimpleBroadcastMulMemoryUsageAnalysis(op, nullptr);
    if (totalPerLane > MInfo::lmem_per_lane) {
      LLVM_DEBUG(llvm::errs() << "TG2TL_LA: " << op.name()
                   << ", layer ID " << getOpLayerId(opInst)
                   << ", SKIP, lmem " << totalPerLane
                   << " needed\n";);
      return failure();
    }

    // Check whether operand ConvOp has enough memory
    for (auto operand : opInst->getOperands()) {
      auto operandOp = operand.getDefiningOp();
      if (auto convOp = dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(operandOp)) {
        uint64_t totalPerLane =
            SimpleConv2DMemoryUsageAnalysis(convOp, nullptr);
        if (totalPerLane > MInfo::lmem_per_lane) {
          LLVM_DEBUG(llvm::errs() << "TG2TL_LA: " << op.name()
                     << ", layer ID " << getOpLayerId(opInst)
                     << ", operandOp " << convOp.name()
                     << ", SKIP, lmem " << totalPerLane
                     << " needed\n";);
          return failure();
        }
      }
    }

    if (1) {
      LLVM_DEBUG(llvm::errs() << "TG2TL_LA: " << op.name()
                   << ", layer ID " << getOpLayerId(opInst) << "\n";);

      assert(op.getNumOperands() == 3);
      std::vector<Value> newOperands;
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
            rewriter.getI32ArrayAttr(ArrayRef<int32_t>({})), // [0]ins_w/[1]ins_h
            rewriter.getI32IntegerAttr(0), //pad_value
            rewriter.getContext())));
      attrs.push_back(rewriter.getNamedAttr("name", op.nameAttr()));

      rewriter.replaceOpWithNewOp<tpu::TL_BroadcastMulOp>(
          op, op.getResult().getType(),
          ArrayRef<Value>{newOperands}, ArrayRef<NamedAttribute>{attrs});
      return success();
    }
  }
};

template<typename OpTy, typename OpTy2>
struct TpuTG2TLPoolOpPattern : public RewritePattern {
  TpuTG2TLPoolOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *opInst,
                                     PatternRewriter &rewriter) const override {
    auto op = cast<OpTy>(opInst);
    assert(op);

    uint64_t totalPerLane = SimpleIOMemoryUsageAnalysis(op, nullptr);
    if (totalPerLane > MInfo::lmem_per_lane) {
      LLVM_DEBUG(llvm::errs() << "TG2TL_LA: " << op.name()
                   << ", layer ID " << getOpLayerId(opInst)
                   << ", SKIP, lmem " << totalPerLane
                   << " needed\n";);
      return failure();
    }

    // Check whether operand ConvOp has enough memory
    // ????
    for (auto operand : opInst->getOperands()) {
      auto operandOp = operand.getDefiningOp();
      if (auto convOp = dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(operandOp)) {
        uint64_t totalPerLane =
            SimpleConv2DMemoryUsageAnalysis(convOp, nullptr);
        if (totalPerLane > MInfo::lmem_per_lane) {
          LLVM_DEBUG(llvm::errs() << "TG2TL_LA: " << op.name()
                     << ", layer ID " << getOpLayerId(opInst)
                     << ", operandOp " << convOp.name()
                     << ", SKIP, lmem " << totalPerLane
                     << " needed\n";);
          return failure();
        }
      }
    }

    if (1) {
      LLVM_DEBUG(llvm::errs() << "TG2TL_LA: " << op.name()
                   << ", layer ID " << getOpLayerId(opInst) << "\n";);

      std::vector<Value> newOperands;
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

      attrs.push_back(rewriter.getNamedAttr("name", op.nameAttr()));

      rewriter.replaceOpWithNewOp<OpTy2>(
          op, op.getResult().getType(),
          ArrayRef<Value>{newOperands}, ArrayRef<NamedAttribute>{attrs});
      return success();
    }
  }
};

class DeepFusionTG2TL_LA : public mlir::PassWrapper<DeepFusionTG2TL_LA, FunctionPass> {
public:
  explicit DeepFusionTG2TL_LA() {}

  void runOnFunction() override {
    MInfo::getChipInfo(getFunction());
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
    applyPatternsAndFoldGreedily(fn, std::move(patterns));
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::createDeepFusionTG2TL_LA() {
  return std::make_unique<DeepFusionTG2TL_LA>();
}
