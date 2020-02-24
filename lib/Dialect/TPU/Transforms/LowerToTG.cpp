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
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"

#include <sstream>
#include <fstream>
#include <math.h>

#define DEBUG_TYPE "convert_to_tg"

namespace mlir {

Value* tpu::BroadcastMulOp::convertToTG(void *info) {
  llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *weightTF_ = (TensorFile *)info;
  assert(weightTF_);
  assert(this->axis() == 1);

  std::vector<Value *> operands;
  operands.push_back(input());
  operands.push_back(multiplier());
  // This is a little tricky, as there is no bias() operand to reuse
  // we reuse the quant_rshift() to carry the packed per-channel info
  operands.push_back(quant_rshift());

  std::vector<NamedAttribute> attrs;
  // only do_relu is useful for now
  attrs.push_back(builder.getNamedAttr("param",
      tpu::ConvParam::get(
          builder.getI32IntegerAttr(1),
          builder.getI32IntegerAttr(1),
          builder.getStringAttr("VALID"),
          builder.getI32IntegerAttr(1),
          builder.getI32IntegerAttr(1),
          builder.getI32IntegerAttr(1),
          builder.getBoolAttr(true),    // is_dw
          builder.getBoolAttr(false),   // with_bias
          builder.getBoolAttr(this->do_relu()),   // do_relu
          builder.getContext())));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));
  if (getOpQuant() == "INT8") {
    // somehow, existing backend implementation is using per-channel mode
    // to do a per-tensor operation. which means, it needs to copy 1 rshift
    // value to a oc sized vector, so does the 1 multiplier value, then pack
    // these two tensors into one as if this is a per-channel multiplier mode
    // convolution.
    // TODO: the right way maybe doing a `REAL` per-channel multiplier mode
    // convolution. to put the scale tensor as multiplier rather than filter
    // and the multiplier is by nature per-channel.
    assert(getOpQuantParamType() == "RSHIFT_AND_M_I32");
    assert( !isTensorNone(quant_rshift()) );
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_BroadcastMulOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    assert( isTensorNone(quant_rshift()) );
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_BroadcastMulOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  assert(false);
  return nullptr;
}

Value* tpu::ConcatOp::convertToTG(void *info) {
  llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *weightTF_ = (TensorFile *)info;
  assert(weightTF_);

  const unsigned nInputs = this->getNumInputs();
  std::vector<Value *> operands;
  for (unsigned i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));
  attrs.push_back(builder.getNamedAttr("axis", axisAttr()));

  if (getOpQuant() == "INT8") {
    if (getOpQuantParamType() == "NONE") {
      // the quant is bypassed (threshold for input and output are the same)
      // do nothing
    } else {
      assert(getOpQuantParamType() == "RSHIFT_AND_M_I8");
      // ADD
      // rshift
      auto rshift = readAndDeleteWeightTensor<float>(quant_rshift(), weightTF_);
      assert(rshift->size() == 1);
      attrs.push_back(builder.getNamedAttr("rshift",
          builder.getI8IntegerAttr(static_cast<int8_t>(rshift->at(0)))));

      // m_i8_inputs
      auto multiplier = readAndDeleteWeightTensor<float>(quant_multiplier(),
                                                       weightTF_);
      std::vector<int32_t> m_i8_inputs_array(nInputs);
      for (unsigned i = 0; i < nInputs; ++i) {
        m_i8_inputs_array[i] = static_cast<int32_t>(multiplier->at(i));
      }
      attrs.push_back(builder.getNamedAttr("m_i8_inputs",
          builder.getI32ArrayAttr(ArrayRef<int32_t>({m_i8_inputs_array}))));
    }
    // create op
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_ConcatOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_ConcatOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  assert(false);
  return nullptr;
}

Value* tpu::Conv2DOp::convertToTG(void *info) {
  llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *weightTF_ = (TensorFile *)info;
  assert(weightTF_);

  std::vector<Value *> operands;
  operands.push_back(input());
  operands.push_back(filter());
  operands.push_back(bias());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("param", paramAttr()));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));
  if (getOpQuant() == "INT8") {
    if (isOpQuantPerchannel()) {
      // per-channel, rshift and mulitplier are in weight .bin
      assert(getOpQuantParamType() == "RSHIFT_AND_M_I32");
      auto newOp = OpBuilder(op).create<tpu::TG_INT8_PC_Conv2DOp>(op->getLoc(),
          getResult()->getType(), ArrayRef<Value *>{operands},
          ArrayRef<NamedAttribute>{attrs});
     return newOp.getResult();
    } else {
      // per-tensor, rshift only mode
      assert(getOpQuantParamType() == "RSHIFT_ONLY");
      assert( !isTensorNone(quant_rshift()) );
      auto rshift = readAndDeleteWeightTensor<float>(quant_rshift(), weightTF_);
      assert(rshift->size() == 1);
      attrs.push_back(builder.getNamedAttr("pt_rshift",
          builder.getI8IntegerAttr(static_cast<int8_t>(rshift->at(0)))));
      auto newOp = OpBuilder(op).create<tpu::TG_INT8_PT_Conv2DOp>(op->getLoc(),
          getResult()->getType(), ArrayRef<Value *>{operands},
          ArrayRef<NamedAttribute>{attrs});
      return newOp.getResult();
    }
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_Conv2DOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  assert(false);
  return nullptr;
}

Value* tpu::CropOp::convertToTG(void *info) {
  llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *weightTF_ = (TensorFile *)info;
  assert(weightTF_);

  const unsigned nInputs = op->getNumOperands();
  std::vector<Value *> operands;
  for (unsigned i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));
  attrs.push_back(builder.getNamedAttr("crop_shape", crop_shapeAttr()));
  attrs.push_back(builder.getNamedAttr("crop_offset", crop_offsetAttr()));

  if (getOpQuant() == "INT8") {
    // create op
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_CropOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_CropOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  assert(false);
  return nullptr;
}

Value* tpu::DeConv2DOp::convertToTG(void *info) {
  llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *weightTF_ = (TensorFile *)info;
  assert(weightTF_);

  std::vector<Value *> operands;
  operands.push_back(input());
  operands.push_back(filter());
  operands.push_back(bias());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("param", paramAttr()));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));
  if (getOpQuant() == "INT8") {
    if (isOpQuantPerchannel()) {
      // per-channel, rshift and mulitplier are in weight .bin
      assert(getOpQuantParamType() == "RSHIFT_AND_M_I32");
      auto newOp = OpBuilder(op).create<tpu::TG_INT8_PC_DeConv2DOp>(op->getLoc(),
          getResult()->getType(), ArrayRef<Value *>{operands},
          ArrayRef<NamedAttribute>{attrs});
     return newOp.getResult();
    } else {
      // per-tensor, rshift only mode
      assert(getOpQuantParamType() == "RSHIFT_ONLY");
      assert( !isTensorNone(quant_rshift()) );
      auto rshift = readAndDeleteWeightTensor<float>(quant_rshift(), weightTF_);
      assert(rshift->size() == 1);
      attrs.push_back(builder.getNamedAttr("pt_rshift",
          builder.getI8IntegerAttr(static_cast<int8_t>(rshift->at(0)))));
      auto newOp = OpBuilder(op).create<tpu::TG_INT8_PT_DeConv2DOp>(op->getLoc(),
          getResult()->getType(), ArrayRef<Value *>{operands},
          ArrayRef<NamedAttribute>{attrs});
      return newOp.getResult();
    }
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_DeConv2DOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  assert(false);
  return nullptr;
}

Value* tpu::EltwiseAddOp::convertToTG(void *info) {
  llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *weightTF_ = (TensorFile *)info;
  assert(weightTF_);

  const unsigned nInputs = this->getNumInputs();
  std::vector<Value *> operands;
  for (unsigned i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));
  attrs.push_back(builder.getNamedAttr("do_relu",
      builder.getBoolAttr(do_relu())));

  if (getOpQuant() == "INT8") {
    if (getOpQuantParamType() == "NONE") {
      // the quant is bypassed (threshold for input and output are the same)
      // do nothing
    } else {
      assert(getOpQuantParamType() == "RSHIFT_AND_M_I8");
      // ADD
      // rshift
      auto rshift = readAndDeleteWeightTensor<float>(quant_rshift(), weightTF_);
      assert(rshift->size() == 1);
      attrs.push_back(builder.getNamedAttr("rshift",
          builder.getI8IntegerAttr(static_cast<int8_t>(rshift->at(0)))));

      // m_i8_inputs
      auto multiplier = readAndDeleteWeightTensor<float>(quant_multiplier(),
                                                       weightTF_);
      std::vector<int32_t> m_i8_inputs_array(nInputs);
      for (unsigned i = 0; i < nInputs; ++i) {
        m_i8_inputs_array[i] = static_cast<int32_t>(multiplier->at(i));
      }
      attrs.push_back(builder.getNamedAttr("m_i8_inputs",
          builder.getI32ArrayAttr(ArrayRef<int32_t>({m_i8_inputs_array}))));
    }

    // create op
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_EltwiseAddOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_EltwiseAddOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  assert(false);
  return nullptr;
}

Value* tpu::EltwiseMaxOp::convertToTG(void *info) {
  llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *weightTF_ = (TensorFile *)info;

  const unsigned nInputs = this->getNumInputs();
  std::vector<Value *> operands;
  for (unsigned i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));
  attrs.push_back(builder.getNamedAttr("do_relu", do_reluAttr()));

  if (getOpQuant() == "INT8") {
    if (getOpQuantParamType() == "NONE") {
      // the quant is bypassed (threshold for input and output are the same)
      // do nothing
    } else {
      assert(getOpQuantParamType() == "RSHIFT_AND_M_I8");
      // MAX
      // rshift
      auto rshift = readAndDeleteWeightTensor<float>(quant_rshift(), weightTF_);
      assert(rshift->size() == 1);
      attrs.push_back(builder.getNamedAttr("rshift",
          builder.getI8IntegerAttr(static_cast<int8_t>(rshift->at(0)))));

      // m_i8_inputs
      auto multiplier = readAndDeleteWeightTensor<float>(quant_multiplier(),
                                                       weightTF_);
      std::vector<int32_t> m_i8_inputs_array(nInputs);
      for (unsigned i = 0; i < nInputs; ++i) {
        m_i8_inputs_array[i] = static_cast<int32_t>(multiplier->at(i));
      }
      attrs.push_back(builder.getNamedAttr("m_i8_inputs",
          builder.getI32ArrayAttr(ArrayRef<int32_t>({m_i8_inputs_array}))));
    }

    // create op
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_EltwiseMaxOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_EltwiseMaxOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  assert(false);
  return nullptr;
}

Value* tpu::EltwiseMulOp::convertToTG(void *info) {
  llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *weightTF_ = (TensorFile *)info;

  const unsigned nInputs = this->getNumInputs();
  std::vector<Value *> operands;
  for (unsigned i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));

  if (getOpQuant() == "INT8") {
    assert(getOpQuantParamType() == "RSHIFT_AND_M_I8");
    // MUL
    // rshift
    auto rshift = readAndDeleteWeightTensor<float>(quant_rshift(), weightTF_);
    assert(rshift->size() == 1);
    attrs.push_back(builder.getNamedAttr("rshift",
        builder.getI8IntegerAttr(static_cast<int8_t>(rshift->at(0)))));

    // m_i8_output
    auto multiplier = readAndDeleteWeightTensor<float>(quant_multiplier(),
                                                     weightTF_);
    assert(multiplier->size() == 1);
    attrs.push_back(builder.getNamedAttr("m_i8_output",
        builder.getI8IntegerAttr(static_cast<int8_t>(multiplier->at(0)))));

    // create op
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_EltwiseMulOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_EltwiseMulOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  assert(false);
  return nullptr;
}

Value *tpu::FullyConnectedOp::convertToTG(void *info) {
  llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *weightTF_ = (TensorFile *)info;

  std::vector<Value *> operands;
  operands.push_back(input());
  operands.push_back(filter());
  operands.push_back(bias());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("do_relu",
      builder.getBoolAttr(do_relu())));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));

  if (getOpQuant() == "INT8") {
    assert(getOpQuantParamType() == "RSHIFT_ONLY");
    // rshift
    auto rshift = readAndDeleteWeightTensor<float>(quant_rshift(), weightTF_);
    assert(rshift->size() == 1);
    attrs.push_back(builder.getNamedAttr("rshift",
        builder.getI8IntegerAttr(static_cast<int8_t>(rshift->at(0)))));

    // create op
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_FullyConnectedOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_FullyConnectedOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  assert(false);
  return nullptr;
}

Value* tpu::LeakyReluOp::convertToTG(void *info) {
  llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *weightTF_ = (TensorFile *)info;

  std::vector<Value *> operands;
  operands.push_back(op->getOperand(0));

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));
  attrs.push_back(builder.getNamedAttr("negative_slope", negative_slopeAttr()));

  if (getOpQuant() == "INT8") {
    assert(getOpQuantParamType() == "RSHIFT_AND_M_I8");

    auto rshift_pos     = readAndDeleteWeightTensor<float>(
                              quant_pos_rshift(), weightTF_);
    auto multiplier_pos = readAndDeleteWeightTensor<float>(
                              quant_pos_multiplier(), weightTF_);
    auto rshift_neg     = readAndDeleteWeightTensor<float>(
                              quant_neg_rshift(), weightTF_);
    auto multiplier_neg = readAndDeleteWeightTensor<float>(
                              quant_neg_multiplier(), weightTF_);

    bool do_pos_scale = (multiplier_pos->at(0) != 0.0) ? true : false;

    if (do_pos_scale) {
      LLVM_DEBUG(llvm::errs() << "    do_pos_scale\n";);
      attrs.push_back(builder.getNamedAttr("rshift_pos",
          builder.getI8IntegerAttr(static_cast<int8_t>(rshift_pos->at(0)))));
      attrs.push_back(builder.getNamedAttr("m_i8_pos",
          builder.getI8IntegerAttr(static_cast<int8_t>(multiplier_pos->at(0)))));
    } else {
      LLVM_DEBUG(llvm::errs() << "    NO pos_scale\n";);
    }
    attrs.push_back(builder.getNamedAttr("rshift_neg",
        builder.getI8IntegerAttr(static_cast<int8_t>(rshift_neg->at(0)))));
    attrs.push_back(builder.getNamedAttr("m_i8_neg",
        builder.getI8IntegerAttr(static_cast<int8_t>(multiplier_neg->at(0)))));

    // create op
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_LeakyReluOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_LeakyReluOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }

  assert(false);
  return nullptr;
}

Value* tpu::PoolAvg2DOp::convertToTG(void *info) {
  llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  TensorFile *weightTF_ = (TensorFile *)info;

  std::vector<Value *> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("param", paramAttr()));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));

  if (getOpQuant() == "INT8") {
    assert(getOpQuantParamType() == "RSHIFT_AND_M_I8");

    assert( !isTensorNone(quant_rshift()) );
    auto rshift = readAndDeleteWeightTensor<float>(quant_rshift(), weightTF_);
    assert(rshift->size() == 1);
    attrs.push_back(builder.getNamedAttr("rshift",
        builder.getI8IntegerAttr(static_cast<int8_t>(rshift->at(0)))));

    assert( !isTensorNone(quant_multiplier()) );
    auto multiplier = readAndDeleteWeightTensor<float>(quant_multiplier(), weightTF_);
    assert(multiplier->size() == 1);
    attrs.push_back(builder.getNamedAttr("m_i8",
        builder.getI8IntegerAttr(static_cast<int8_t>(multiplier->at(0)))));

    auto newOp = OpBuilder(op).create<tpu::TG_INT8_PoolAvg2DOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_PoolAvg2DOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  assert(false);
  return nullptr;
}

Value* tpu::PoolMax2DOp::convertToTG(void *info) {
  llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  //TensorFile *weightTF_ = (TensorFile *)info;

  std::vector<Value *> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("param", paramAttr()));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));

  if (getOpQuant() == "INT8") {
    assert(getOpQuantParamType() == "NONE");
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_PoolMax2DOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_PoolMax2DOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  assert(false);
  return nullptr;
}

Value *tpu::PReluOp::convertToTG(void *info) {
  llvm::errs() << "lowerToTG: " << getOperationName() << " [" << getOpName()
               << "]\n";
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());

  std::vector<Value *> operands;
  // operands.push_back(input());
  // operands.push_back(negative_slope());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));
  if (getOpQuant() == "INT8") {
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_PReluOp>(
        op->getLoc(), getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_PReluOp>(

        op->getLoc(), getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  assert(false);
  return nullptr;
}

Value *tpu::ShuffleChannelOp::convertToTG(void *info) {
  llvm::errs() << "lowerToTG: " << getOperationName() << " [" << getOpName()
               << "]\n";
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  // TensorFile *weightTF_ = (TensorFile *)info;

  std::vector<Value *> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("group", groupAttr()));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));

  if (getOpQuant() == "INT8") {
    assert(getOpQuantParamType() == "NONE");
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_ShuffleChannelOp>(
        op->getLoc(), getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  assert(false);
  return nullptr;
}

Value *tpu::SigmoidOp::convertToTG(void *info) {
  llvm::errs() << "lowerToTG: " << getOperationName() << " [" << getOpName()
               << "]\n";
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());


  int nInputs = 2; // input and table
  std::vector<Value *> operands;
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));

  if (getOpQuant() == "INT8") {
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_SigmoidOp>(
        op->getLoc(), getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_SigmoidOp>(
        op->getLoc(), getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  assert(false);
  return nullptr;
}

Value* tpu::UpsampleOp::convertToTG(void *info) {
  llvm::errs() << "lowerToTG: " << getOperationName()
               << " [" << getOpName() << "]\n";
  Operation *op = this->getOperation();
  auto builder = Builder(op->getContext());
  //TensorFile *weightTF_ = (TensorFile *)info;

  std::vector<Value *> operands;
  operands.push_back(input());

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("scale", scaleAttr()));
  attrs.push_back(builder.getNamedAttr("name", nameAttr()));
  attrs.push_back(builder.getNamedAttr("layer_id", layer_idAttr()));

  if (getOpQuant() == "INT8") {
    assert(getOpQuantParamType() == "NONE");
    auto newOp = OpBuilder(op).create<tpu::TG_INT8_UpsampleOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  } else if (getOpQuant() == "BF16") {
    auto newOp = OpBuilder(op).create<tpu::TG_BF16_UpsampleOp>(op->getLoc(),
        getResult()->getType(), ArrayRef<Value *>{operands},
        ArrayRef<NamedAttribute>{attrs});
    return newOp.getResult();
  }
  assert(false);
  return nullptr;
}

template<typename OpTy>
struct DefaultToTGPattern : public RewritePattern {
  DefaultToTGPattern(MLIRContext *context, TensorFile *weightTF,
      Value* weightFV)
      : RewritePattern(OpTy::getOperationName(), 1, context),
        weightTF_(weightTF),
        weightFV_(weightFV) {}

  PatternMatchResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    auto tpuOp = llvm::dyn_cast<tpu::TpuOpLowerInterface>(op);
    if (!tpuOp) {
      return matchFailure();
    }
    auto newValue = tpuOp.convertToTG((void *)weightTF_);
    if (!newValue) {
      return matchFailure();
    }
    rewriter.replaceOp(op, {newValue});
    return matchSuccess();
  }

  TensorFile *weightTF_;
  Value* weightFV_;
};

template<typename OpTy>
struct DefaultErasePattern : public RewritePattern {
  DefaultErasePattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, {op->getOperand(0)});
    return matchSuccess();
  }
};

struct FoldReshapePattern : public RewritePattern {
  FoldReshapePattern(MLIRContext *context)
      : RewritePattern("tpu.reshape", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto laterReshapeOp = cast<tpu::ReshapeOp>(op);

    auto formerOp = laterReshapeOp.getOperand()->getDefiningOp();
    if (!matchPattern(formerOp, m_Op<tpu::ReshapeOp>())) {
      return matchFailure();
    }
    auto formerScaleOp = cast<tpu::ReshapeOp>(formerOp);

    laterReshapeOp.getOperation()->setOperand(0, formerScaleOp.getOperand());
    return matchSuccess();
  }

  TensorFile *weightTensorFile_;
};

static std::unique_ptr<std::vector<uint8_t> > packWeight(
    std::vector<float> *bias, std::vector<float> *rshift,
    std::vector<float> *multiplier, int64_t oc,
    std::vector<int64_t> &shape) {
  if (bias)
    assert(bias->size() == (size_t)oc);
  assert(rshift->size() == (size_t)oc);
  assert(multiplier->size() == (size_t)oc);

  int64_t isz = bias ? 9 : 5;
  shape = std::vector<int64_t>{oc, 1, isz};

  auto packed = std::make_unique<std::vector<uint8_t> >(oc * isz);

  uint8_t *ptr = packed->data();
  for (int i = 0; i < oc; i++) {
    if (bias) {
      uint32_t val = (uint32_t)(*bias)[i];
      *ptr = (uint8_t)(val & 0xff);
      ptr++;
      *ptr = (uint8_t)((val >> 8) & 0xff);
      ptr++;
      *ptr = (uint8_t)((val >> 16) & 0xff);
      ptr++;
      *ptr = (uint8_t)((val >> 24) & 0xff);
      ptr++;
    }

    {
      uint32_t val = (uint32_t)(*multiplier)[i];
      *ptr = (uint8_t)(val & 0xff);
      ptr++;
      *ptr = (uint8_t)((val >> 8) & 0xff);
      ptr++;
      *ptr = (uint8_t)((val >> 16) & 0xff);
      ptr++;
      *ptr = (uint8_t)((val >> 24) & 0xff);
      ptr++;
    }

    {
      uint8_t val = (uint8_t)(*rshift)[i];
      *ptr = (uint8_t)val;
      ptr++;
    }
  }

  return std::move(packed);
}

template <typename OpTy>
struct PackWeightConv2DOpPattern : public RewritePattern {
  PackWeightConv2DOpPattern(MLIRContext *context, TensorFile *weightTF,
      Value* weightFV)
      : RewritePattern(OpTy::getOperationName(), 1, context),
        weightTF_(weightTF),
        weightFV_(weightFV) {}

  PatternMatchResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    auto convOp = cast<OpTy>(op);
    if (getOpQuant(op) != "INT8" || !isOpQuantPerchannel(op)
        || getOpQuantParamType(op) != "RSHIFT_AND_M_I32") {
      // for perchannel multiplier mode only
      return matchFailure();
    }
    if ( !isTensorNone(convOp.bias()) ) {
      auto biasOp = cast<tpu::LoadWeightOp>(convOp.bias()->getDefiningOp());
      if (biasOp.lowered()) {
        // packed already
        return matchFailure();
      }
    }
    assert( !isTensorNone(convOp.quant_rshift()) );
    assert( !isTensorNone(convOp.quant_multiplier()) );
    llvm::errs() << "Pack Weight for Conv2D: " << getOpName(op) << "\n";

    // get param
    auto filter_type = convOp.filter()->getType().template cast<TensorType>();
    std::vector<int64_t> filter_shape(filter_type.getShape());
    int64_t oc;
    auto g = convOp.param().group().getValue().getLimitedValue();
    if (g != 1) {
      assert(filter_shape.size() == 5);
      oc = filter_shape[0] * filter_shape[1];
    } else {
      assert(filter_shape.size() == 4);
      oc = filter_shape[0];
    }

    // get tensor
    std::unique_ptr<std::vector<float> > bias = nullptr;
    if ( !isTensorNone(convOp.bias()) ) {
      bias = readAndDeleteWeightTensor<float>(convOp.bias(), weightTF_);
    }
    auto rshift = readAndDeleteWeightTensor<float>(convOp.quant_rshift(), weightTF_);
    auto multiplier = readAndDeleteWeightTensor<float>(convOp.quant_multiplier(), weightTF_);

    // pack the weights
    std::vector<int64_t> packedShape;
    auto packed = packWeight(bias.get(), rshift.get(), multiplier.get(), oc,
                             packedShape);

    // store to the packed per_channel operand in "UINT8"
    if (bias) {
      addWeightTensorAndUpdateWeightOp<uint8_t>(convOp.bias(),
          "pack", *packed, packedShape, "UINT8", weightTF_);
    } else {
      auto packed_op = addWeightTensorAndCreateWeightOp<uint8_t>(
          op, "pack", *packed, packedShape, "UINT8",
          weightTF_, weightFV_);
      convOp.setOperand(2, packed_op);
    }
    auto biasOp = cast<tpu::LoadWeightOp>(convOp.bias()->getDefiningOp());
    biasOp.setAttr("lowered", rewriter.getBoolAttr(true));

    // erase quant_rshift and quant_multiplier tensor
    auto NoneOp = OpBuilder(op).create<tpu::NoneOp>(
        rewriter.getUnknownLoc(), rewriter.getNoneType());
    convOp.setOperand(5, NoneOp);
    convOp.setOperand(6, NoneOp);

    return matchSuccess();
  }

  TensorFile *weightTF_;
  Value* weightFV_;
};

// somehow, existing backend implementation is using per-channel mode
// to do a per-tensor operation. which means, it needs to copy 1 rshift
// value to a oc sized vector, so does the 1 multiplier value, then pack
// these two tensors into one as if this is a per-channel multiplier mode
// convolution.
// TODO: the right way maybe doing a `REAL` per-channel multiplier mode
// convolution. to put the scale tensor as multiplier rather than filter
// and the multiplier is by nature per-channel.
struct PackWeightBroadcastMulOpPattern : public RewritePattern {
  PackWeightBroadcastMulOpPattern(MLIRContext *context, TensorFile *weightTF,
      Value* weightFV)
      : RewritePattern(tpu::BroadcastMulOp::getOperationName(), 1, context),
        weightTF_(weightTF),
        weightFV_(weightFV) {}

  PatternMatchResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    auto castOp = cast<tpu::BroadcastMulOp>(op);
    if (getOpQuant(op) != "INT8") {
      return matchFailure();
    }

    // after quantizeInt8, the quantparam is "RSHIFT_AND_M_I8"
    // after packing, will mark it as
    auto rshiftOp = cast<tpu::LoadWeightOp>(castOp.quant_rshift()->getDefiningOp());
    if ( getOpQuantParamType(op) == "RSHIFT_AND_M_I32" ) {
      assert (rshiftOp.lowered());
      // packed already
      return matchFailure();
    }
    assert( !rshiftOp.lowered() );
    assert( getOpQuantParamType(op) == "RSHIFT_AND_M_I8" );
    assert( !isTensorNone(castOp.quant_rshift()) );
    assert( !isTensorNone(castOp.quant_multiplier()) );
    llvm::errs() << "Pack Weight for BroadcastMul: " << getOpName(op) << "\n";

    // get param
    int64_t oc = getTensorSize(castOp.multiplier());

    // get tensor
    std::unique_ptr<std::vector<float> > pc_info = nullptr;
    auto rshift = readAndDeleteWeightTensor<float>(castOp.quant_rshift(), weightTF_);
    auto multiplier = readAndDeleteWeightTensor<float>(castOp.quant_multiplier(), weightTF_);

    // expand
    auto rshift_perchannel = std::make_unique<std::vector<float>>(oc, rshift->at(0));
    auto multiplier_perchannel =
          std::make_unique<std::vector<float>>(oc, multiplier->at(0));

    // pack the weights
    std::vector<int64_t> packedShape;
    auto packed = packWeight(nullptr, rshift_perchannel.get(),
        multiplier_perchannel.get(), oc, packedShape);

    // this is tricky, as where is no bias() to reuse, use quant_rshift() instead
    // store to the packed per_channel operand in "UINT8"
    addWeightTensorAndUpdateWeightOp<uint8_t>(castOp.quant_rshift(),
        "pack", *packed, packedShape, "UINT8", weightTF_);
    rshiftOp.setAttr("lowered", rewriter.getBoolAttr(true));

    // erase quant_multiplier tensor
    auto NoneOp = OpBuilder(op).create<tpu::NoneOp>(
        rewriter.getUnknownLoc(), rewriter.getNoneType());
    castOp.setOperand(5, NoneOp);

    setOpQuantParamType(op, "RSHIFT_AND_M_I32");
    return matchSuccess();
  }

  TensorFile *weightTF_;
  Value* weightFV_;
};

template<typename T>
static void transposeConvolutionFilter(std::vector<T> &w,
    std::vector<int64_t> &s) {
  int64_t oc, ic, ks;
  if (s.size() == 4) {
    oc = s[0];
    ic = s[1];
    ks = s[2] * s[3];
  } else if (s.size() == 5) {
    // g, oc/g, ic/g, kh, kw
    oc = s[0] * s[1];
    ic = s[2];
    ks = s[3] * s[4];
  } else {
    assert(false);
  }

  std::vector<T> w_t(w.size());
  if (ks == 1 || ic == 1) {
    return;
  } else {
    // for other conv, transpose ic <-> kh*kw
    for (int64_t i = 0; i < oc; i++) {
      for (int64_t j = 0; j < ic; j++) {
        for (int64_t k = 0; k < ks; k++) {
          w_t[i * ic * ks + k * ic + j] = w[i * ic * ks + j * ks + k];
        }
      }
    }
  }
  w.assign(w_t.begin(), w_t.end());
}

template<typename T>
static void transposeFullyConnectedFilter(std::vector<T> &w,
    std::vector<int64_t> &s) {
  assert(s.size() == 2);
  int row = s[0];
  int col = s[1];
  std::vector<T> w_t(w.size());
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      w_t[j * row + i] = w[i * col  + j];
    }
  }
  w.assign(w_t.begin(), w_t.end());
}

static void transposeBiasInt16(std::vector<int16_t> &w_int16) {
  int8_t *ptr = reinterpret_cast<int8_t *>(w_int16.data());
  std::vector<int8_t> w(ptr, ptr + w_int16.size() * sizeof(int16_t));
  std::vector<int8_t> w_t(w.size());
  for (size_t i = 0; i < w_int16.size(); i++) {
    for (size_t j = 0; j < 2; j++) {
      w_t[j * w_int16.size() + i] = w[i * 2 + j];
    }
  }
  memcpy(ptr, w_t.data(), w_t.size());
}

struct LowerWeightConv2DOpPattern : public RewritePattern {
  LowerWeightConv2DOpPattern(MLIRContext *context, TensorFile *weightTF)
      : RewritePattern("tpu.conv_2d", 1, context),
        weightTF_(weightTF) {}

  PatternMatchResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    auto convOp = cast<tpu::Conv2DOp>(op);
    auto filterOp = cast<tpu::LoadWeightOp>(convOp.filter()->getDefiningOp());
    if (filterOp.lowered()) {
      // lowered already
      return matchFailure();
    }
    llvm::errs() << "Lower Weight for Conv2D: " << getOpName(op) << "\n";

    if (getOpQuant(op) == "INT8") {
      // lower filter
      {
        assert(filterOp.storage() == "INT8");
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(convOp.filter(), shape, size);
        auto filter = readAndDeleteWeightTensor<float>(convOp.filter(), weightTF_);
        std::vector<int8_t> filter_int8(filter->begin(), filter->end());
        // transpose ic <-> kh*kw
        // if kh*kw == 1 or ic/g == 1, transposeConvolutionFilter() will do nothing
        assert(shape.size() == 4 || shape.size() == 5);
        transposeConvolutionFilter<int8_t>(filter_int8, shape);

        // save it
        addWeightTensorAndUpdateWeightOp<int8_t>(convOp.filter(),
            "lowered", filter_int8, shape, "INT8", weightTF_);
        filterOp.setAttr("lowered", rewriter.getBoolAttr(true));
      }

      // lower bias
      if ( !isTensorNone(convOp.bias()) ) {
        auto biasOp = cast<tpu::LoadWeightOp>(convOp.bias()->getDefiningOp());
        if (isOpQuantPerchannel(op)
            && getOpQuantParamType(op) == "RSHIFT_AND_M_I32") {
          // lowered already, in pack
          assert(biasOp.lowered());
          assert(biasOp.storage() == "UINT8");
        } else if (isOpQuantPerchannel(op)) {
          // per-channel mode, bias is INT32
          assert(biasOp.storage() == "INT32");
          assert(false && "REMINDER: NOT sure if per-channel bias needs transpose");

          // TODO:

          // save it
          //StringRef storageType = "INT32";
          //addWeightTensorAndUpdateWeightOp<int32_t>(convOp.bias(),
          //    "lowered", bias_int16, shape, storageType, weightTF_);
          biasOp.setAttr("lowered", rewriter.getBoolAttr(true));
        } else {
          // per-tensor mode, bias is INT16
          assert(biasOp.storage() == "INT16");
          std::vector<int64_t> shape;
          int64_t size;
          getTensorShapeAndSize(convOp.bias(), shape, size);
          auto bias = readAndDeleteWeightTensor<float>(convOp.bias(), weightTF_);
          std::vector<int16_t> bias_int16(bias->begin(), bias->end());
          transposeBiasInt16(bias_int16);
          std::vector<uint16_t> bias_uint16(size);
          memcpy(bias_uint16.data(), bias_int16.data(), size * sizeof(int16_t));

          // save it
          // after transpose, this is not INT16 anymore, it is 2 stripes of UINT8
          // we save it as UINT16, to carry the eltment bitwidth, so we don`t need
          // to change the shape.
          addWeightTensorAndUpdateWeightOp<uint16_t>(convOp.bias(),
              "lowered", bias_uint16, shape, "UINT16", weightTF_);
          biasOp.setAttr("lowered", rewriter.getBoolAttr(true));
        }
      }
    } else if (getOpQuant(op) == "BF16") {
      // lower filter
      {
        assert(filterOp.storage() == "BF16");
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(convOp.filter(), shape, size);
        auto filter = readAndDeleteWeightTensor<bfloat16>(convOp.filter(), weightTF_);
        std::vector<uint16_t> filter_bf16(filter->begin(), filter->end());

        // transpose ic <-> kh*kw
        // if kh*kw == 1 or ic/g == 1, transposeConvolutionFilter() will do nothing
        assert(shape.size() == 4 || shape.size() == 5);
        transposeConvolutionFilter<uint16_t>(filter_bf16, shape);

        // save it
        StringRef storageType = "BF16";
        addWeightTensorAndUpdateWeightOp<uint16_t>(convOp.filter(),
            "lowered", filter_bf16, shape, storageType, weightTF_);
        filterOp.setAttr("lowered", rewriter.getBoolAttr(true));
      }

      // lower bias
      if ( !isTensorNone(convOp.bias()) ) {
        auto biasOp = cast<tpu::LoadWeightOp>(convOp.bias()->getDefiningOp());
        assert(biasOp.storage() == "BF16");
        // NOTE: for 1880v2, bias is fp32, rather than bf16
        // however, for simplicity, in quantizeBf16, we quantize all tensor into bf16
        // before lowering to hardware, we need to expand the bf16 to fp32 first
        // then transpose into 2 stripes of uint16_t
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(convOp.bias(), shape, size);
        auto bias = readAndDeleteWeightTensor<bfloat16>(convOp.bias(), weightTF_);
        std::vector<uint16_t> bias_bf16(bias->begin(), bias->end());
        // rather than expand to fp32, then transpose, we simply add a new stripe
        // of uint16_t with all 0x0000
        size_t sz = bias_bf16.size();
        for (size_t i = 0; i < sz; ++i) {
          bias_bf16.push_back(0x0000);
        }
        // then copy into uint32_t
        std::vector<uint32_t> bias_uint32(sz);
        memcpy(bias_uint32.data(), bias_bf16.data(), sz * sizeof(uint32_t));

        // save it
        // after expand to FB32 and transpose, this is not FB32 anymore
        // it is 2 stripes of UINT16(BF16)
        // we save it as UINT32, to carry the eltment bitwidth, so we don`t need
        // to change the shape
        StringRef storageType = "UINT32";
        addWeightTensorAndUpdateWeightOp<uint32_t>(convOp.bias(),
            "lowered", bias_uint32, shape, storageType, weightTF_);
        biasOp.setAttr("lowered", rewriter.getBoolAttr(true));
      }
    }

    return matchSuccess();
  }

  TensorFile *weightTF_;
};

struct LowerWeightFullyConnectedOpPattern : public RewritePattern {
  LowerWeightFullyConnectedOpPattern(MLIRContext *context, TensorFile *weightTF)
      : RewritePattern("tpu.fully_connected", 1, context),
        weightTF_(weightTF) {}

  PatternMatchResult matchAndRewrite(Operation *op,
      PatternRewriter &rewriter) const override {
    auto fcOp = cast<tpu::FullyConnectedOp>(op);
    auto filterOp = cast<tpu::LoadWeightOp>(fcOp.filter()->getDefiningOp());
    if (filterOp.lowered()) {
      // lowered already
      return matchFailure();
    }
    llvm::errs() << "Lower Weight for FullyConnectedOp: " << getOpName(op) << "\n";

    if (getOpQuant(op) == "INT8") {
      // lower filter
      {
        assert(filterOp.storage() == "INT8");
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(fcOp.filter(), shape, size);
        auto filter = readAndDeleteWeightTensor<float>(fcOp.filter(), weightTF_);
        std::vector<int8_t> filter_int8(filter->begin(), filter->end());
        // transpose k,n
        assert(shape.size() == 2);
        transposeFullyConnectedFilter<int8_t>(filter_int8, shape);

        // save it
        addWeightTensorAndUpdateWeightOp<int8_t>(fcOp.filter(),
            "lowered", filter_int8, shape, "INT8", weightTF_);
        filterOp.setAttr("lowered", rewriter.getBoolAttr(true));
      }

      // lower bias
      if ( !isTensorNone(fcOp.bias()) ) {
        auto biasOp = cast<tpu::LoadWeightOp>(fcOp.bias()->getDefiningOp());
        // per-tensor mode, bias is INT16
        assert(biasOp.storage() == "INT16");
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(fcOp.bias(), shape, size);
        auto bias = readAndDeleteWeightTensor<float>(fcOp.bias(), weightTF_);
        std::vector<int16_t> bias_int16(bias->begin(), bias->end());
        transposeBiasInt16(bias_int16);
        std::vector<uint16_t> bias_uint16(size);
        memcpy(bias_uint16.data(), bias_int16.data(), size * sizeof(int16_t));

        // save it
        // after transpose, this is not INT16 anymore, it is 2 stripes of UINT8
        // we save it as UINT16, to carry the eltment bitwidth, so we don`t need
        // to change the shape.
        addWeightTensorAndUpdateWeightOp<uint16_t>(fcOp.bias(),
            "lowered", bias_uint16, shape, "UINT16", weightTF_);
        biasOp.setAttr("lowered", rewriter.getBoolAttr(true));
      }
    } else if (getOpQuant(op) == "BF16") {
      // lower filter
      {
        assert(filterOp.storage() == "BF16");
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(fcOp.filter(), shape, size);
        auto filter = readAndDeleteWeightTensor<bfloat16>(fcOp.filter(), weightTF_);
        std::vector<uint16_t> filter_bf16(filter->begin(), filter->end());
        // transpose h,n
        assert(shape.size() == 2);
        transposeFullyConnectedFilter<uint16_t>(filter_bf16, shape);

        // save it
        StringRef storageType = "BF16";
        addWeightTensorAndUpdateWeightOp<uint16_t>(fcOp.filter(),
            "lowered", filter_bf16, shape, storageType, weightTF_);
        filterOp.setAttr("lowered", rewriter.getBoolAttr(true));
      }

      // lower bias
      if ( !isTensorNone(fcOp.bias()) ) {
        auto biasOp = cast<tpu::LoadWeightOp>(fcOp.bias()->getDefiningOp());
        assert(biasOp.storage() == "BF16");
        // NOTE: for 1880v2, bias is fp32, rather than bf16
        // however, for simplicity, in quantizeBf16, we quantize all tensor into bf16
        // before lowering to hardware, we need to expand the bf16 to fp32 first
        // then transpose into 2 stripes of uint16_t
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(fcOp.bias(), shape, size);
        auto bias = readAndDeleteWeightTensor<bfloat16>(fcOp.bias(), weightTF_);
        std::vector<uint16_t> bias_bf16(bias->begin(), bias->end());
        // rather than expand to fp32, then transpose, we simply add a new stripe
        // of uint16_t with all 0x0000
        size_t sz = bias_bf16.size();
        for (size_t i = 0; i < sz; ++i) {
          bias_bf16.push_back(0x0000);
        }
        // then copy into uint32_t
        std::vector<uint32_t> bias_uint32(sz);
        memcpy(bias_uint32.data(), bias_bf16.data(), sz * sizeof(uint32_t));

        // save it
        // after expand to FB32 and transpose, this is not FB32 anymore
        // it is 2 stripes of UINT16(BF16)
        // we save it as UINT32, to carry the eltment bitwidth, so we don`t need
        // to change the shape
        StringRef storageType = "UINT32";
        addWeightTensorAndUpdateWeightOp<uint32_t>(fcOp.bias(),
            "lowered", bias_uint32, shape, storageType, weightTF_);
        biasOp.setAttr("lowered", rewriter.getBoolAttr(true));
      }
    }

    return matchSuccess();
  }

  TensorFile *weightTF_;
};

struct LowerWeightSigmoidOpPattern : public RewritePattern {
  LowerWeightSigmoidOpPattern(MLIRContext *context, TensorFile *weightTF)
      : RewritePattern("tpu.sigmoid", 1, context), weightTF_(weightTF) {
  }

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto sigOp = cast<tpu::SigmoidOp>(op);
    
    auto tableOp = cast<tpu::LoadWeightOp>(sigOp.getOperand(1)->getDefiningOp());
    if (tableOp.lowered()) {
      // lowered already
      return matchFailure();
    }
    llvm::errs() << "Lower Weight for SigmoidOp: " << getOpName(op)
                 << "\n";

    if (getOpQuant(op) == "INT8") {
      // lower filter
        assert(tableOp.storage() == "INT8");
        std::vector<int64_t> shape;
        int64_t size;
        getTensorShapeAndSize(sigOp.table(), shape, size);
        auto table = readAndDeleteWeightTensor<float>(tableOp, weightTF_);
        std::vector<int8_t> table_int8(table->begin(), table->end());
        // 1880 support 256 lookup table
        // because of 1880 hardware search table only on each local memory
        // we dupicate table to limit number <32>
        assert(shape[2] * shape[3] == 256);
        assert(shape[1] == 32);

        // save it
        addWeightTensorAndUpdateWeightOp<int8_t>(
            tableOp, "lowered", table_int8, shape, "INT8", weightTF_);
        tableOp.setAttr("lowered", rewriter.getBoolAttr(true));

    } else if (getOpQuant(op) == "BF16") {
      // lower filter
      assert(false && "TOTO BF16");
    }

    return matchSuccess();
  }

  TensorFile *weightTF_;
};

class TpuLowerPass : public FunctionPass<TpuLowerPass> {
public:
  void runOnFunction() override {
    auto *context = &getContext();
    auto fn = getFunction();

    // find tensorFile and Value
    llvm::StringRef filename;
    Value* weightFV;
    fn.walk([&](tpu::LoadFileOp op) {
      filename = op.filename();
      LLVM_DEBUG(llvm::errs() << "LoadFileOp filename " << filename << "\n";);
      weightFV = op.getResult();
    });
    auto weightTF = openTensorFile(filename);

    // first, merge conv rshift/multiplier/bias into one packed tensor
    OwningRewritePatternList patterns_pack;
    patterns_pack.insert<
        PackWeightConv2DOpPattern<tpu::Conv2DOp>,
        PackWeightConv2DOpPattern<tpu::DeConv2DOp>,
        PackWeightBroadcastMulOpPattern
        >(context, weightTF.get(), weightFV);
    applyPatternsGreedily(fn, patterns_pack);

    // second, do weight lower on weight tensors
    // lower means transpose and save as storageType (int8/bf16,etc)
    OwningRewritePatternList patterns_lower;
    patterns_lower.insert<
        LowerWeightConv2DOpPattern,
        LowerWeightSigmoidOpPattern,
        LowerWeightFullyConnectedOpPattern
        >(context, weightTF.get());
    applyPatternsGreedily(fn, patterns_lower);

    // do op lower
    OwningRewritePatternList patterns;
    patterns.insert<
        DefaultToTGPattern<tpu::BroadcastMulOp>,
        DefaultToTGPattern<tpu::ConcatOp>,
        DefaultToTGPattern<tpu::Conv2DOp>,
        DefaultToTGPattern<tpu::CropOp>,
        DefaultToTGPattern<tpu::DeConv2DOp>,
        DefaultToTGPattern<tpu::EltwiseAddOp>,
        DefaultToTGPattern<tpu::EltwiseMaxOp>,
        DefaultToTGPattern<tpu::EltwiseMulOp>,
        DefaultToTGPattern<tpu::FullyConnectedOp>,
        DefaultToTGPattern<tpu::LeakyReluOp>,
        DefaultToTGPattern<tpu::PoolAvg2DOp>,
        DefaultToTGPattern<tpu::PoolMax2DOp>,
        DefaultToTGPattern<tpu::ShuffleChannelOp>,
        DefaultToTGPattern<tpu::SigmoidOp>,
        DefaultToTGPattern<tpu::UpsampleOp>
    >(context, weightTF.get(), weightFV);
    applyPatternsGreedily(fn, patterns);

    // TODO: this is temporary
    // erase CPU ops, fold reshape
    patterns.clear();
    patterns.insert<
        DefaultErasePattern<tpu::SoftmaxOp>,
        //DefaultErasePattern<tpu::QuantizationOp>,
        DefaultErasePattern<tpu::DequantizationOp>,
        FoldReshapePattern
        >(context);
    applyPatternsGreedily(fn, patterns);

    // keep tensorfile
    std::string newName;
    weightTF->keep(true, &newName);
    fn.walk([&](tpu::LoadFileOp op) {
      OpBuilder opBuilder(context);
      op.setAttr("filename", opBuilder.getStringAttr(newName));
      llvm::errs() << "LoadFileOp filename updated to " << newName << "\n";
    });
  }
};

std::unique_ptr<OpPassBase<FuncOp>> createTpuLowerPass() {
  return std::make_unique<TpuLowerPass>();
}

static PassRegistration<TpuLowerPass>
    pass("tpu-lower",
         "Lower TPU Dialect from TPU Ops to TPU_TG Ops");

} // namespace mlir
