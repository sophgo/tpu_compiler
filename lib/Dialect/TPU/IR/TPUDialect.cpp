//===- TPUDialect.cpp - MLIR Dialect for TPU implementation -------===//
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
// This file implements the TPU dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUOperationSupport.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::tpu;

TPUDialect::TPUDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<TPUDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "tpuc/Dialect/TPU/TPUOps.cpp.inc"
      >();
  #if 0
  addInterfaces<
    TpuOpCommonInterface,
    TpuOpQuantInterface,
    TpuOpLowerInterface,
    TpuTGOpCodegenInterface,
    TpuTLSimpleOpCodegenInterface>();
  #endif

}

static ParseResult parseTG_CallOp(OpAsmParser &parser, OperationState &result) {
  FlatSymbolRefAttr calleeAttr;
  FunctionType calleeType;
  SmallVector<OpAsmParser::OperandType, 4> operands;
  auto calleeLoc = parser.getNameLoc();
  if (parser.parseAttribute(calleeAttr, "callee", result.attributes) ||
      parser.parseOperandList(operands, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(calleeType) ||
      parser.addTypesToList(calleeType.getResults(), result.types) ||
      parser.resolveOperands(operands, calleeType.getInputs(), calleeLoc,
                             result.operands))
    return failure();
  return success();
}

static void print(OpAsmPrinter &p, TG_CallOp op) {
  #if 0
  SmallVector<Type, 4> resultTypes(op.getResultTypes());
  SmallVector<Type, 8> argTypes(op.getOperandTypes());
  auto callType = FunctionType::get(argTypes, resultTypes, op.getContext());

  p << "tpu.tg_call " << op->getAttr("callee") << '(';
  p.printOperands(op.getOperands());
  p << ')';
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"callee"});
  p << " : ";
  p.printType(callType);
  #endif
}


namespace mlir {
#define GET_OP_CLASSES
#include "tpuc/Dialect/TPU/TPUOps.cpp.inc"

//===----------------------------------------------------------------------===//
// TPU OpInterface definitions
//===----------------------------------------------------------------------===//
#include "tpuc/Dialect/TPU/TPUInterface.cpp.inc"

//===----------------------------------------------------------------------===//
// TPU Struct Attribute definitions
//===----------------------------------------------------------------------===//

#include "tpuc/Dialect/TPU/TPUAttribute.cpp.inc"

} // namespace mlir

//
// Implementation of TpuOpCommont Interface
//
#define DECLARE_GET_OP_NAME_METHOD(OP) \
    StringRef OP::getOpName() {return name();}

#define DECLARE_GET_LAYER_ID_METHOD(OP) \
    int OP::getLayerId() { \
      auto op = getOperation(); \
      auto loc = op->getLoc().cast<FileLineColLoc>(); \
      return loc.getLine() - 3; \
    }

#define DECLARE_ALL_COMMON_INTERFACE_METHODS(OP) \
    DECLARE_GET_OP_NAME_METHOD(OP) \
    DECLARE_GET_LAYER_ID_METHOD(OP)

// TPU Ops
DECLARE_ALL_COMMON_INTERFACE_METHODS(AbsOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(ArgMaxOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(BroadcastMulOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(BroadcastAddOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(BroadcastSubOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(BatchNormOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(ClipOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(ConcatOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(Conv2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(Conv3DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(ConvFcOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(CopyOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(CropOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(CscOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(CustomOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(DeConv2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(DetectionOutputOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(DilateOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(EmbeddingOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(EltwiseAddOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(EltwiseMaxOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(EltwiseMinOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(EltwiseMulOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(MulConstOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(EluOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(ExpOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(FullyConnectedOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(FrcnDetectionOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(GenericCpuOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(GruOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(InputOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(InterpOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(InstanceNormOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(LayerNormOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(LogOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(LeakyReluOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(LrnOneOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(LrnTwoOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(LrnThreeOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(LrnOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(LstmOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(MatchTemplateOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(MatMulOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(NormalizeOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(MishOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(PadOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(PermuteOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(PixelShuffleOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(PoolAvg2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(PoolMax2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(PoolMax3DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(PoolMaskOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(PowOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(PReluOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(PriorBoxOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(ProposalOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(ReflectionPadOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(ReluOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(ReshapeOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(ReorgOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(RetinaFaceDetectionOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(ROIPoolingOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(ReverseOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(ScaleOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(ScaleLutOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(ShuffleChannelOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(SoftmaxOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(SoftmaxCpuOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(SigmoidOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(StdOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(SwishOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(QuadraticSumOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(SwapChannelOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TanHOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TileOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(UpsampleOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(YoloDetectionOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(ReduceL2Op)
DECLARE_ALL_COMMON_INTERFACE_METHODS(ReduceMeanOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(ReduceMaxOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(ReduceMinOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(ReduceSumOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(SoftPlusOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(ZeroMaskOp)
// TPU Support Ops
DECLARE_ALL_COMMON_INTERFACE_METHODS(QuantOp)
// TPU TG Ops
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_AbsOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_AbsOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_ArgMaxOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_ArgMaxOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_BroadcastMulOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_BroadcastMulOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_BroadcastAddOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_BroadcastAddOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_BroadcastSubOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_BroadcastSubOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_ConcatOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_ConcatOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_Conv2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_Conv2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_Conv3DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_CopyOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_CopyOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_CropOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_CropOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_DeConv2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_DeConv2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_DilateOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_DilateOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_EmbeddingOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_EmbeddingOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_EltwiseAddOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_MulConstOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_EltwiseMaxOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_EltwiseMinOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_EltwiseMulOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_EltwiseAddOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_MulConstOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_EltwiseMaxOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_EltwiseMinOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_EltwiseMulOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_FullyConnectedOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_FullyConnectedOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_GenericTpuOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_InterpOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_InterpOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_LayerNormOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_LeakyReluOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_LeakyReluOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_LrnOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_LrnOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_LutOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_LutOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_PermuteOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_PermuteOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_PoolAvg2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_PoolMax2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_PoolMax3DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_PoolAvg2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_PoolMax2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_PoolMax3DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_PoolMaskOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_PoolMaskOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_ReverseOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_ReverseOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_ShuffleChannelOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_ShuffleChannelOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_ScaleOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_ScaleOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_ScaleLutOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_StdOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_PixelShuffleOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_PixelShuffleOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_ClipOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_ClipOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_PReluOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_PReluOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_ReluOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_ReluOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_ReorgOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_ReorgOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_SwapChannelOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_SwapChannelOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_TileOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_TileOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_UpsampleOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_UpsampleOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_CallOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_QuantOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_DequantOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_PadOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_PadOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_ReflectionPadOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_ReflectionPadOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_ReduceMeanOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_ReduceMeanOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_ReduceMaxOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_ReduceMaxOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_ReduceMinOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_ReduceMinOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_ReduceSumOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_ReduceSumOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_ReduceL2Op)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_GruOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_GruOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_LstmOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_LstmOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_SoftmaxOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_SoftmaxOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_QuadraticSumOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_Yuv420CscOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_MatchTemplateOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_MatMulOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_MatMulOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_ConvFcOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_ConcatNOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_MergeConvConvPoolOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_ZeroMaskOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_ZeroMaskOp)

// TPU TL Ops for layer group
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_INT8_AbsOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_INT8_Conv2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_INT8_DeConv2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_INT8_EltwiseAddOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_INT8_EltwiseMulOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_LoadNeuronOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_LoadCoeffOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_StoreOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_CopyOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_QuantOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_JoinOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_INT8_PoolAvg2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_INT8_PoolMax2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_INT8_LrnOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_INT8_ScaleOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_INT8_ScaleLutOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_INT8_UpsampleOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_INT8_LeakyReluOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_INT8_LutOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_INT8_MulConstOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_INT8_PReluOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_INT8_ConcatOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_INT8_PadOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_INT8_CropOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_INT8_ReluOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_INT8_SwapChannelOp)

DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_BF16_AbsOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_BF16_Conv2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_BF16_DeConv2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_BF16_EltwiseAddOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_BF16_EltwiseMulOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_BF16_PoolAvg2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_BF16_PoolMax2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_BF16_LayerNormOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_BF16_LrnOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_BF16_MulConstOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_BF16_ScaleOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_BF16_UpsampleOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_BF16_LeakyReluOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_BF16_LutOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_BF16_PReluOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_BF16_ConcatOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_BF16_PadOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_BF16_CropOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_BF16_ReluOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LG_BF16_SwapChannelOp)

//
// Implementation of TpuOpQuant Interface
//
// quant().mode()
#define DECLARE_GET_OP_QUANT_MODE_METHOD(OP) \
    StringRef OP::getOpQuant() {return quant().mode().getValue();}
#define DECLARE_SET_OP_QUANT_MODE_METHOD(OP) \
    LogicalResult OP::setOpQuantMode(StringRef &mode) { \
      (*this)->setAttr("quant", \
          tpu::QuantParam::get( \
              Builder(getOperation()->getContext()).getStringAttr(mode), \
              quant().param_type(), \
              quant().threshold(), \
              getOperation()->getContext())); \
      return success(); \
    }

// quant().param_type()
#define DECLARE_GET_OP_QUANT_PARAM_TYPE_METHOD(OP) \
    StringRef OP::getOpQuantParamType() {return quant().param_type().getValue();}
#define DECLARE_SET_OP_QUANT_PARAM_TYPE_METHOD(OP) \
    LogicalResult OP::setOpQuantParamType(StringRef &type) { \
      (*this)->setAttr("quant", \
          tpu::QuantParam::get( \
              quant().mode(), \
              Builder(getOperation()->getContext()).getStringAttr(type), \
              quant().threshold(), \
              getOperation()->getContext())); \
      return success(); \
    }

// quant().threshold()
#define DECLARE_GET_OP_QUANT_THRESHOLD_METHOD(OP) \
    float OP::getOpQuantThreshold() { \
      return quant().threshold().getValue().convertToFloat(); \
    }
#define DECLARE_SET_OP_QUANT_THRESHOLD_METHOD(OP) \
    LogicalResult OP::setOpQuantThreshold(float threshold) { \
      (*this)->setAttr("quant", \
          tpu::QuantParam::get( \
              quant().mode(), \
              quant().param_type(), \
              Builder(getOperation()->getContext()).getF32FloatAttr(threshold), \
              getOperation()->getContext())); \
      return success(); \
    }


// declare quant methods
#define DECLARE_ALL_QUANT_INTERFACE_METHODS(OP) \
    DECLARE_GET_OP_QUANT_MODE_METHOD(OP) \
    DECLARE_SET_OP_QUANT_MODE_METHOD(OP) \
    DECLARE_GET_OP_QUANT_PARAM_TYPE_METHOD(OP) \
    DECLARE_SET_OP_QUANT_PARAM_TYPE_METHOD(OP) \
    DECLARE_GET_OP_QUANT_THRESHOLD_METHOD(OP) \
    DECLARE_SET_OP_QUANT_THRESHOLD_METHOD(OP)



DECLARE_ALL_QUANT_INTERFACE_METHODS(AbsOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(ArgMaxOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(BroadcastMulOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(BroadcastAddOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(BroadcastSubOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(ClipOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(ConcatOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(Conv2DOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(Conv3DOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(ConvFcOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(CopyOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(CropOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(CscOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(CustomOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(DeConv2DOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(DilateOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(EmbeddingOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(EltwiseAddOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(MulConstOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(EltwiseMaxOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(EltwiseMinOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(EltwiseMulOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(EluOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(ExpOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(FullyConnectedOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(GruOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(InputOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(InterpOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(InstanceNormOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(LayerNormOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(LeakyReluOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(LogOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(LrnOneOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(LrnTwoOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(LrnThreeOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(LrnOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(LstmOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(MatchTemplateOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(MatMulOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(MishOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(PadOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(PermuteOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(PixelShuffleOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(PoolAvg2DOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(PoolMax2DOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(PoolMax3DOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(PoolMaskOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(PowOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(PReluOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(ReflectionPadOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(ReluOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(ReorgOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(ROIPoolingOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(ReduceL2Op)
DECLARE_ALL_QUANT_INTERFACE_METHODS(ReduceMeanOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(ReduceMaxOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(ReduceMinOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(ReduceSumOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(ReverseOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(ShuffleChannelOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(ScaleLutOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(SigmoidOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(SwishOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(SoftmaxOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(SoftmaxCpuOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(StdOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(QuadraticSumOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(SwapChannelOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(TanHOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(TileOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(UpsampleOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(SoftPlusOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(ZeroMaskOp)

//
// Implementation of TpuTGOpCodegen Interface
//
#define DECLARE_GET_TG_OP_GADDR_METHOD(OP) \
    uint64_t OP::getGAddr() { \
      return gaddr().getValue(); \
    }
#define DECLARE_SET_TG_OP_GADDR_METHOD(OP) \
    LogicalResult OP::setGAddr(uint64_t gaddr) { \
      (*this)->setAttr("gaddr", Builder(getOperation()->getContext()).getI64IntegerAttr(gaddr)); \
      return success(); \
    }
#define DECLARE_SET_TG_OP_BUFFER_REUSED_METHOD(OP) \
    LogicalResult OP::setBufferReused(bool flag) { \
      if (flag) \
        (*this)->setAttr("buffer_reused", Builder(getOperation()->getContext()).getBoolAttr(flag)); \
      else \
        (*this)->removeAttr("buffer_reused"); \
      return success(); \
    }

//
// Implementation of TpuTGOpCodegen Interface
//
#define DECLARE_GET_TG_OP_ENABLE_PARALLEL_METHOD(OP) \
    bool OP::getEnableParallel() { \
      return enable_parallel(); \
    }
#define DECLARE_SET_TG_OP_ENABLE_PARALLEL_METHOD(OP) \
    LogicalResult OP::setEnableParallel(bool b_flag) { \
      (*this)->setAttr("enable_parallel", \
              Builder(getOperation()->getContext()).getBoolAttr(b_flag)); \
      return success(); \
    }

//
// Implementation of TpuTGOpCodegen Interface
//
#define DECLARE_GET_TG_OP_DISABLE_PARALLEL_METHOD(OP) \
    bool OP::getDisableParallel() { \
      return disable_parallel(); \
    }
#define DECLARE_SET_TG_OP_DISABLE_PARALLEL_METHOD(OP) \
    LogicalResult OP::setDisableParallel(bool b_flag) { \
      (*this)->setAttr("disable_parallel", \
              Builder(getOperation()->getContext()).getBoolAttr(b_flag)); \
      return success(); \
    }

// declare TG Codegen methods
#define DECLARE_ALL_CODEGEN_INTERFACE_METHODS(OP) \
    DECLARE_GET_TG_OP_GADDR_METHOD(OP) \
    DECLARE_SET_TG_OP_GADDR_METHOD(OP) \
    DECLARE_SET_TG_OP_BUFFER_REUSED_METHOD(OP)

#define DECLARE_TL_CODEGEN_INTERFACE_METHODS(OP) \
    DECLARE_GET_TG_OP_GADDR_METHOD(OP) \
    DECLARE_SET_TG_OP_GADDR_METHOD(OP) \
    DECLARE_GET_TG_OP_ENABLE_PARALLEL_METHOD(OP) \
    DECLARE_SET_TG_OP_ENABLE_PARALLEL_METHOD(OP) \
    DECLARE_GET_TG_OP_DISABLE_PARALLEL_METHOD(OP) \
    DECLARE_SET_TG_OP_DISABLE_PARALLEL_METHOD(OP)


// TG Ops
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_AbsOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_AbsOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_ArgMaxOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_ArgMaxOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_BroadcastMulOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_BroadcastMulOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_BroadcastAddOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_BroadcastAddOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_BroadcastSubOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_BroadcastSubOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_ConcatOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_ConcatOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_Conv2DOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_Conv2DOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_Conv3DOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_CopyOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_CopyOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_CropOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_CropOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_DeConv2DOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_DeConv2DOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_DilateOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_DilateOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_EmbeddingOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_EmbeddingOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_EltwiseAddOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_MulConstOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_EltwiseMaxOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_EltwiseMinOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_EltwiseMulOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_EltwiseAddOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_MulConstOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_EltwiseMaxOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_EltwiseMinOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_EltwiseMulOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_FullyConnectedOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_FullyConnectedOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_GenericTpuOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_InterpOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_InterpOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_LeakyReluOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_LeakyReluOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_LayerNormOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_LrnOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_LrnOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_LutOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_LutOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_PermuteOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_PermuteOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_PoolAvg2DOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_PoolMax2DOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_PoolMax3DOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_PoolAvg2DOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_PoolMax2DOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_PoolMax3DOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_PoolMaskOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_PoolMaskOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_ReverseOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_ReverseOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_ShuffleChannelOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_ShuffleChannelOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_PixelShuffleOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_PixelShuffleOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_ClipOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_ClipOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_PReluOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_PReluOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_ReluOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_ReluOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_ReorgOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_ReorgOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_ScaleOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_ScaleOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_ScaleLutOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_SwapChannelOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_SwapChannelOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_TileOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_TileOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_UpsampleOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_UpsampleOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_CallOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_QuantOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_DequantOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_PadOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_PadOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_ReflectionPadOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_ReflectionPadOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_ReduceMeanOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_ReduceMeanOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_ReduceMaxOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_ReduceMaxOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_ReduceMinOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_ReduceMinOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_ReduceSumOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_ReduceSumOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_ReduceL2Op)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_GruOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_GruOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_LstmOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_LstmOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_SoftmaxOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_SoftmaxOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_QuadraticSumOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_Yuv420CscOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_MatchTemplateOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_MatMulOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_MatMulOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_ConvFcOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_StdOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_ConcatNOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_MergeConvConvPoolOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_ZeroMaskOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_ZeroMaskOp)

// TL Ops for layer group
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_INT8_AbsOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_INT8_Conv2DOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_INT8_DeConv2DOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_INT8_EltwiseAddOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_INT8_EltwiseMulOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_LoadNeuronOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_LoadCoeffOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_StoreOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_CopyOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_JoinOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_QuantOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_INT8_PoolAvg2DOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_INT8_PoolMax2DOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_INT8_LrnOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_INT8_MulConstOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_INT8_ScaleOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_INT8_ScaleLutOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_INT8_UpsampleOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_INT8_LeakyReluOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_INT8_LutOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_INT8_PReluOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_INT8_ConcatOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_INT8_PadOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_INT8_CropOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_INT8_ReluOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_INT8_SwapChannelOp)

DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_BF16_AbsOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_BF16_Conv2DOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_BF16_DeConv2DOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_BF16_EltwiseAddOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_BF16_EltwiseMulOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_BF16_PoolAvg2DOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_BF16_PoolMax2DOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_BF16_LayerNormOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_BF16_LrnOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_BF16_MulConstOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_BF16_ScaleOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_BF16_UpsampleOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_BF16_LeakyReluOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_BF16_LutOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_BF16_PReluOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_BF16_ConcatOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_BF16_PadOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_BF16_CropOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_BF16_ReluOp)
DECLARE_TL_CODEGEN_INTERFACE_METHODS(TL_LG_BF16_SwapChannelOp)

// Reshape Op
uint64_t ReshapeOp::getGAddr() {
  auto prev_op = this->getOperand().getDefiningOp();
  return mlir::getOpAddress(prev_op);
}

LogicalResult ReshapeOp::setGAddr(uint64_t gaddr) {
  assert(false);
  auto prev_op = this->getOperand().getDefiningOp();
  return mlir::setOpAddress(prev_op, gaddr);
}

StringRef ReshapeOp::getOpQuant() {
  auto prev_op = this->getOperand().getDefiningOp();
  return mlir::getOpQuant(prev_op);
}

LogicalResult ReshapeOp::setOpQuantMode(StringRef &mode) {
  auto prev_op = this->getOperand().getDefiningOp();
  return mlir::setOpQuant(prev_op, mode);
}

StringRef ReshapeOp::getOpQuantParamType() {
  auto prev_op = this->getOperand().getDefiningOp();
  return mlir::getOpQuantParamType(prev_op);
}

LogicalResult ReshapeOp::setOpQuantParamType(StringRef &type) {
  auto prev_op = this->getOperand().getDefiningOp();
  return mlir::setOpQuantParamType(prev_op, type);
}

float ReshapeOp::getOpQuantThreshold() {
  auto prev_op = this->getOperand().getDefiningOp();
  return mlir::getOpThreshold(prev_op);
}

LogicalResult ReshapeOp::setOpQuantThreshold(float threshold) {
  auto prev_op = this->getOperand().getDefiningOp();
  return mlir::setOpThreshold(prev_op, threshold);
}

LogicalResult ReshapeOp::setBufferReused(bool flag) {
  return mlir::setOpBufferReused(this->getOperation(), flag);
}
