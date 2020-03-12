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

#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;
using namespace mlir::tpu;

TPUDialect::TPUDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
      WeightFileOp,
#define GET_OP_LIST
#include "mlir/Dialect/TPU/TPUOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/TPU/TPUOps.cpp.inc"

//===----------------------------------------------------------------------===//
// TPU OpInterface definitions
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/TPU/TPUInterface.cpp.inc"

//===----------------------------------------------------------------------===//
// TPU Struct Attribute definitions
//===----------------------------------------------------------------------===//
namespace mlir {

#include "mlir/Dialect/TPU/TPUAttribute.cpp.inc"

} // namespace mlir

//
// Implementation of TpuOpCommont Interface
//
#define DECLARE_GET_OP_NAME_METHOD(OP) \
    StringRef OP::getOpName() {return name();}
#define DECLARE_GET_LAYER_ID_METHOD(OP) \
    int OP::getOpLayerId() { \
      if (layer_id().hasValue()) { \
        return layer_id().getValue().getLimitedValue(); \
      } else { \
        llvm::errs() << name() << " has no layer_id assigned\n"; \
        assert(false); \
        return -1; \
      } \
    }
#define DECLARE_SET_LAYER_ID_METHOD(OP) \
    LogicalResult OP::setOpLayerId(int id) { \
      setAttr("layer_id", \
          Builder(getOperation()->getContext()).getI32IntegerAttr(id)); \
      return success(); \
    }

#define DECLARE_ALL_COMMON_INTERFACE_METHODS(OP) \
    DECLARE_GET_OP_NAME_METHOD(OP) \
    DECLARE_GET_LAYER_ID_METHOD(OP) \
    DECLARE_SET_LAYER_ID_METHOD(OP)
// TPU Ops
DECLARE_ALL_COMMON_INTERFACE_METHODS(BroadcastMulOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(BatchNormOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(ConcatOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(Conv2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(CropOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(DeConv2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(DetectionOutputOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(EltwiseAddOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(EltwiseMaxOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(EltwiseMulOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(FullyConnectedOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(InputOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(LeakyReluOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(NormalizeOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(PermuteOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(PixelShuffleOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(PoolAvg2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(PoolMax2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(PowerOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(PReluOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(PriorBoxOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(ReciprocalOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(ReluOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(ReshapeOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(RetinaFaceDetectionOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(ScaleOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(ShuffleChannelOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(SliceOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(SoftmaxOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(SigmoidOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(SqrtOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TanHOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(UpsampleOp)
// TPU Support Ops
DECLARE_ALL_COMMON_INTERFACE_METHODS(QuantOp)
// TPU TG Ops
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_BroadcastMulOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_BroadcastMulOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_ConcatOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_ConcatOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_PT_Conv2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_PC_Conv2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_Conv2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_CropOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_CropOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_PT_DeConv2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_PC_DeConv2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_DeConv2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_EltwiseAddOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_EltwiseMaxOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_EltwiseMulOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_EltwiseAddOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_EltwiseMaxOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_EltwiseMulOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_FullyConnectedOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_FullyConnectedOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_InputOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_InputOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_LeakyReluOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_LeakyReluOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_LutOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_LutOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_PermuteOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_PermuteOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_PoolAvg2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_PoolMax2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_PoolAvg2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_PoolMax2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_ShuffleChannelOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_ShuffleChannelOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_PixelShuffleOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_PixelShuffleOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_PReluOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_PReluOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_ReluOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_ReluOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_SliceOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_SliceOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_INT8_UpsampleOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_BF16_UpsampleOp)

// TPU TG MemRef Ops
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_MemRef_INT8_PC_Conv2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_MemRef_INT8_EltwiseAddOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_MemRef_INT8_FullyConnectedOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_MemRef_INT8_InputOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_MemRef_INT8_PoolAvg2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_MemRef_INT8_PoolMax2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_MemRef_QuantOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TG_MemRef_ReshapeOp)

// TPU TL Ops
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LA_Conv2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_LW_Conv2DOp)
DECLARE_ALL_COMMON_INTERFACE_METHODS(TL_EltwiseAddOp)

//
// Implementation of TpuOpQuant Interface
//
// quant().mode()
#define DECLARE_GET_OP_QUANT_MODE_METHOD(OP) \
    StringRef OP::getOpQuant() {return quant().mode().getValue();}
#define DECLARE_SET_OP_QUANT_MODE_METHOD(OP) \
    LogicalResult OP::setOpQuantMode(StringRef &mode) { \
      setAttr("quant", \
          tpu::QuantParam::get( \
              Builder(getOperation()->getContext()).getStringAttr(mode), \
              quant().param_type(), \
              quant().is_perchannel(), \
              quant().is_asymmetric(), \
              quant().threshold_max(), \
              quant().threshold_min(), \
              getOperation()->getContext())); \
      return success(); \
    }

// quant().param_type()
#define DECLARE_GET_OP_QUANT_PARAM_TYPE_METHOD(OP) \
    StringRef OP::getOpQuantParamType() {return quant().param_type().getValue();}
#define DECLARE_SET_OP_QUANT_PARAM_TYPE_METHOD(OP) \
    LogicalResult OP::setOpQuantParamType(StringRef &type) { \
      setAttr("quant", \
          tpu::QuantParam::get( \
              quant().mode(), \
              Builder(getOperation()->getContext()).getStringAttr(type), \
              quant().is_perchannel(), \
              quant().is_asymmetric(), \
              quant().threshold_max(), \
              quant().threshold_min(), \
              getOperation()->getContext())); \
      return success(); \
    }

// quant().is_perchannel()
#define DECLARE_GET_OP_QUANT_IS_PERCHANNEL_METHOD(OP) \
    bool OP::isOpQuantPerchannel() {return quant().is_perchannel().getValue();}
#define DECLARE_SET_OP_QUANT_IS_PERCHANNEL_METHOD(OP) \
    LogicalResult OP::setOpQuantPerchannel(bool flag) { \
      setAttr("quant", \
          tpu::QuantParam::get( \
              quant().mode(), \
              quant().param_type(), \
              Builder(getOperation()->getContext()).getBoolAttr(flag), \
              quant().is_asymmetric(), \
              quant().threshold_max(), \
              quant().threshold_min(), \
              getOperation()->getContext())); \
      return success(); \
    }

// quant().is_asymmetric()
#define DECLARE_GET_OP_QUANT_IS_ASYMMETRIC_METHOD(OP) \
    bool OP::isOpQuantAsymmetric() {return quant().is_asymmetric().getValue();}
#define DECLARE_SET_OP_QUANT_IS_ASYMMETRIC_METHOD(OP) \
    LogicalResult OP::setOpQuantAsymmetric(bool flag) { \
      setAttr("quant", \
          tpu::QuantParam::get( \
              quant().mode(), \
              quant().param_type(), \
              quant().is_perchannel(), \
              Builder(getOperation()->getContext()).getBoolAttr(flag), \
              quant().threshold_max(), \
              quant().threshold_min(), \
              getOperation()->getContext())); \
      return success(); \
    }

// quant().threshold()
#define DECLARE_GET_OP_QUANT_THRESHOLD_METHOD(OP) \
    float OP::getOpQuantThreshold() { \
      return quant().threshold_max().getValue().convertToFloat(); \
    }
#define DECLARE_SET_OP_QUANT_THRESHOLD_METHOD(OP) \
    LogicalResult OP::setOpQuantThreshold(float threshold) { \
      assert( !quant().is_asymmetric().getValue() ); \
      setAttr("quant", \
          tpu::QuantParam::get( \
              quant().mode(), \
              quant().param_type(), \
              quant().is_perchannel(), \
              quant().is_asymmetric(), \
              Builder(getOperation()->getContext()).getF32FloatAttr(threshold), \
              quant().threshold_min(), \
              getOperation()->getContext())); \
      return success(); \
    }

// declare quant methods
#define DECLARE_ALL_QUANT_INTERFACE_METHODS(OP) \
    DECLARE_GET_OP_QUANT_MODE_METHOD(OP) \
    DECLARE_SET_OP_QUANT_MODE_METHOD(OP) \
    DECLARE_GET_OP_QUANT_PARAM_TYPE_METHOD(OP) \
    DECLARE_SET_OP_QUANT_PARAM_TYPE_METHOD(OP) \
    DECLARE_GET_OP_QUANT_IS_PERCHANNEL_METHOD(OP) \
    DECLARE_SET_OP_QUANT_IS_PERCHANNEL_METHOD(OP) \
    DECLARE_GET_OP_QUANT_IS_ASYMMETRIC_METHOD(OP) \
    DECLARE_SET_OP_QUANT_IS_ASYMMETRIC_METHOD(OP) \
    DECLARE_GET_OP_QUANT_THRESHOLD_METHOD(OP) \
    DECLARE_SET_OP_QUANT_THRESHOLD_METHOD(OP)
DECLARE_ALL_QUANT_INTERFACE_METHODS(BroadcastMulOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(ConcatOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(Conv2DOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(CropOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(DeConv2DOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(EltwiseAddOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(EltwiseMaxOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(EltwiseMulOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(FullyConnectedOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(InputOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(LeakyReluOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(PermuteOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(PixelShuffleOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(PoolAvg2DOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(PoolMax2DOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(PowerOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(PReluOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(ReciprocalOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(ReluOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(ShuffleChannelOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(SigmoidOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(SliceOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(SqrtOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(TanHOp)
DECLARE_ALL_QUANT_INTERFACE_METHODS(UpsampleOp)

//
// Implementation of TpuTGOpCodegen Interface
//
#define DECLARE_GET_TG_OP_GADDR_METHOD(OP) \
    uint64_t OP::getGAddr() { \
      return gaddr().getValue().getLimitedValue(); \
    }
#define DECLARE_SET_TG_OP_GADDR_METHOD(OP) \
    LogicalResult OP::setGAddr(uint64_t gaddr) { \
      setAttr("gaddr", Builder(getOperation()->getContext()).getI64IntegerAttr(gaddr)); \
      return success(); \
    }

// declare TG Codegen methods
#define DECLARE_ALL_CODEGEN_INTERFACE_METHODS(OP) \
    DECLARE_GET_TG_OP_GADDR_METHOD(OP) \
    DECLARE_SET_TG_OP_GADDR_METHOD(OP)
// TG Ops
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_BroadcastMulOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_BroadcastMulOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_ConcatOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_ConcatOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_PT_Conv2DOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_PC_Conv2DOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_Conv2DOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_CropOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_CropOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_PT_DeConv2DOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_PC_DeConv2DOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_DeConv2DOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_EltwiseAddOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_EltwiseMaxOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_EltwiseMulOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_EltwiseAddOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_EltwiseMaxOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_EltwiseMulOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_FullyConnectedOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_FullyConnectedOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_InputOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_InputOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_LeakyReluOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_LeakyReluOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_LutOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_LutOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_PermuteOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_PermuteOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_PoolAvg2DOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_PoolMax2DOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_PoolAvg2DOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_PoolMax2DOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_ShuffleChannelOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_ShuffleChannelOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_PixelShuffleOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_PixelShuffleOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_PReluOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_PReluOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_ReluOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_ReluOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_SliceOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_SliceOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_INT8_UpsampleOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_BF16_UpsampleOp)

// TG MemRef Op
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_MemRef_INT8_PC_Conv2DOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_MemRef_INT8_EltwiseAddOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_MemRef_INT8_FullyConnectedOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_MemRef_INT8_InputOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_MemRef_INT8_PoolAvg2DOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_MemRef_INT8_PoolMax2DOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_MemRef_QuantOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TG_MemRef_ReshapeOp)

// TL Ops
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TL_LA_Conv2DOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TL_LW_Conv2DOp)
DECLARE_ALL_CODEGEN_INTERFACE_METHODS(TL_EltwiseAddOp)



// Reshape Op
uint64_t ReshapeOp::getGAddr() {
  auto prev_op = this->getOperand()->getDefiningOp();
  return mlir::getOpAddress(prev_op);
}

LogicalResult ReshapeOp::setGAddr(uint64_t gaddr) {
  assert(false);
  auto prev_op = this->getOperand()->getDefiningOp();
  return mlir::setOpAddress(prev_op, gaddr);
}

StringRef ReshapeOp::getOpQuant() {
  auto prev_op = this->getOperand()->getDefiningOp();
  return mlir::getOpQuant(prev_op);
}

LogicalResult ReshapeOp::setOpQuantMode(StringRef &mode) {
  assert(false);
  auto prev_op = this->getOperand()->getDefiningOp();
  return mlir::setOpQuant(prev_op, mode);
}

StringRef ReshapeOp::getOpQuantParamType() {
  assert(false);
  return StringRef();
}

LogicalResult ReshapeOp::setOpQuantParamType(StringRef &type) {
  assert(false);
  return failure();
}

bool ReshapeOp::isOpQuantPerchannel() {
  assert(false);
  return false;
}
LogicalResult ReshapeOp::setOpQuantPerchannel(bool flag) {
  assert(false);
  return failure();
}

bool ReshapeOp::isOpQuantAsymmetric() {
  assert(false);
  return false;
}

LogicalResult ReshapeOp::setOpQuantAsymmetric(bool flag) {
  assert(false);
  return failure();
}

float ReshapeOp::getOpQuantThreshold() { \
  auto prev_op = this->getOperand()->getDefiningOp();
  return mlir::getOpThreshold(prev_op);
}

LogicalResult ReshapeOp::setOpQuantThreshold(float threshold) {
  assert(false);
  return failure();
}
