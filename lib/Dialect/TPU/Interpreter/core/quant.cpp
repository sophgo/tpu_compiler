#include "tpuc/Interpreter/cpu/quant.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"
#include "tpuc/QuantizationArithmetic.h"

#include <cmath>

static inline signed char tpu_float2int8(float v, int mode = 0) {

  int int32 = 0;
  float fraction, integer;
  float abs_v = std::abs(v);
  fraction = std::modf(abs_v, &integer);
  int32 = (int)integer;
  if (fraction > 0.5) {
    int32 = int32 + 1;
  } else if (fraction == 0.5) {
    if (int32 & 0x01) {
      int32 = int32 + 1;
    }
  }
  if (v < 0) {
    int32 = -int32;
  }
  if (int32 > 127) {
    return 127;
  }
  if (int32 < -128) {
    return -128;
  }
  return (signed char)int32;
}
// Quantize an Activation tensor into INT8, given threshold
static void quantizeFromFp32ToInt8(float *src, float *dst, int64_t size,
                                   float scale, int zero_point, bool tpu_mode) {
  if (tpu_mode) {
    bfloat16 bf_scale, bf_tmp, bf_zp;
    bf_scale = FloatToBFloat16(scale);
    scale = BFloat16ToFloat(bf_scale);
    bf_zp = FloatToBFloat16(zero_point);
    zero_point = FloatToBFloat16(bf_zp);
    for (int64_t i = 0; i < size; ++i) {
      float f_tmp = src[i];
      // remove [17:31] mantissa part
      bf_tmp = FloatToBFloat16(f_tmp);
      f_tmp = BFloat16ToFloat(bf_tmp);
      f_tmp = f_tmp * scale;
      // align backend
      bf_tmp = FloatToBFloat16(f_tmp);
      f_tmp = BFloat16ToFloat(bf_tmp);
      f_tmp = f_tmp + zero_point;
      bf_tmp = FloatToBFloat16(f_tmp);
      f_tmp = BFloat16ToFloat(bf_tmp);
      dst[i] = (float)tpu_float2int8(f_tmp, 1);
    }
  } else {
    for (int64_t i = 0; i < size; ++i) {
      int val = std::round(src[i] * scale) + zero_point;
      if (val > 127) {
        val = 127;
      } else if (val < -128) {
        val = -128;
      }
      dst[i] = (float)val;
    }
  }
}

/// Dequant an Int8 Activation tensor to Bf16, given threshold
/// Keep interpreter int8 quant align with TPU
void dequantizeFromInt8ToBf16(float *src, float *dst, int64_t size, float scale,
                              int zero_point) {
  bfloat16 bf_scale;
  bf_scale = FloatToBFloat16(scale);
  scale = BFloat16ToFloat(bf_scale);
  bfloat16 bf_zp;
  bf_zp = FloatToBFloat16(zero_point);
  zero_point = BFloat16ToFloat(bf_zp);
  for (int64_t i = 0; i < size; ++i) {
    bfloat16 out = FloatToBFloat16((src[i] + zero_point) * scale);
    dst[i] = (float)BFloat16ToFloat(out);
  }
}

/// DeQuantize an Activation tensor from INT8, given threshold
void dequantizeFromInt8ToFp32(float *src, float *dst, int64_t size, float scale,
                              int zero_point, bool tpu_mode) {
  if (tpu_mode) {
    bfloat16 bf_scale, bf_tmp;
    bf_scale = FloatToBFloat16(scale);
    scale = BFloat16ToFloat(bf_scale);

    for (int64_t i = 0; i < size; ++i) {
      // i8->bf16
      float fp_tmp = src[i];
      fp_tmp += zero_point;
      bf_tmp = FloatToBFloat16(fp_tmp);
      fp_tmp = BFloat16ToFloat(bf_tmp);
      // bf16 mul scale
      fp_tmp *= scale;
      bf_tmp = FloatToBFloat16(fp_tmp);
      fp_tmp = BFloat16ToFloat(bf_tmp);
      // bf16 -> fp32
      dst[i] = fp_tmp;
    }
  } else {
    for (int64_t i = 0; i < size; ++i) {
      dst[i] = (src[i] + zero_point) * scale;
    }
  }
}

namespace mlir {
QuantOpKernel::QuantOpKernel(Operation &op, value_map_t &valueMapping) {
  auto quantOp = cast<tpu::QuantOp>(op);
  assert(quantOp);
  llvm::outs() << " Quant op: [" << quantOp.name() << "]\n";

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = quantOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  llvm::outs() << "    =>required memory size: [" << size << "]\n";
  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

  this->name = quantOp.name().str();
  this->op_type = op.getName().getStringRef().str();
  this->scale = quantOp.scale().convertToFloat();
  this->zero_point = quantOp.zero_point();
  this->from = quantOp.from().str();
  this->to = quantOp.to().str();

  set_datatype(getOpQuant(&op).str());
  // get tensors
  input_data = opTensors[0];
  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}

void QuantOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " Quant op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> QuantOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void QuantOpKernel::invoke() {
  if (this->from == "NONE" && this->to == "INT8") {
    quantizeFromFp32ToInt8(input_data->data(), output_data->data(),
                           input_data->size(), scale, zero_point, false);
  } else if (this->from == "INT8" && this->to == "NONE") {
    dequantizeFromInt8ToFp32(input_data->data(), output_data->data(),
                             input_data->size(), scale, zero_point, false);
  } else if (this->from == "INT8" && this->to == "BF16") {
    dequantizeFromInt8ToBf16(input_data->data(), output_data->data(),
                             input_data->size(), scale, zero_point);
  } else if (this->from == "BF16" && this->to == "NONE") {
    output_data->assign(input_data->begin(), input_data->end());
  } else if (this->from == "NONE" && this->to == "BF16") {
    clean16bitmantissa(input_data->data(), output_data->data(),
                       output_data->size());
  } else if (this->from == "BF16" && this->to == "INT8") {
    quantizeActivationFromBf16ToInt8(output_data->data(), input_data->data(),
                                     output_data->size(), scale);
  }
}

void QuantOpKernel::dump() {
  OpKernel::dump();
  llvm::outs() << "\tScale: " << this->scale << "\n";
  llvm::outs() << "\tFrom: " << this->from << "\n";
  llvm::outs() << "\tTo: " << this->to << "\n";
}
} // namespace mlir