#include "tpuc/Interpreter/cpu/quant.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"
#include "tpuc/QuantizationArithmetic.h"

#include <cmath>

llvm::cl::opt<bool>
    clUseTPUQuantOp("use-tpu-quant-op",
                llvm::cl::desc("Quant op inference by tpu instead of cpu"),
                llvm::cl::init(true));

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
    bfloat16 bf_scale, bf_tmp;
    bf_scale = FloatToBFloat16(scale);
    scale = BFloat16ToFloat(bf_scale);
    for (int64_t i = 0; i < size; ++i) {
      float f_tmp = src[i];
      // remove [17:31] mantissa part
      // align \TgQuantKernel.cpp that we directly use high part
      // rather than convert it
      FloatToBFloat16(&f_tmp, &bf_tmp, /*size=*/1, /*rounding=*/0);

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

static void quantizeFromFp32ToInt16(float *src, float *dst, int64_t size,
                                     float scale) {
  for (int64_t i = 0; i < size; ++i) {
    int val = std::round(src[i] * scale);
    if (val > 32767) {
      val = 32767;
    } else if (val < -32768) {
      val = -32768;
    }
    dst[i] = (float)val;
  }
}

static void quantizeFromFp32ToUInt16(float *src, float *dst, int64_t size,
                                     float scale) {
  for (int64_t i = 0; i < size; ++i) {
    int val = std::round(src[i] * scale);
    val = std::max(0, std::min(val, 65535));
    dst[i] = (float)val;
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
QuantOpKernel::QuantOpKernel(Operation &op, value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  auto quantOp = cast<tpu::QuantOp>(op);
  this->scale = quantOp.scale().convertToFloat();
  this->zero_point = quantOp.zero_point();
  this->from = quantOp.from().str();
  this->to = quantOp.to().str();
  auto prevOp = quantOp.getOperand().getDefiningOp(); // input
  //this->useTpuQuant = isa<tpu::InputOp>(prevOp) ? false : clUseTPUQuantOp;
  if (isa<tpu::ReshapeOp>(prevOp)) {
    auto pprevOp = cast<tpu::ReshapeOp>(prevOp).getOperand().getDefiningOp();
    if (isa<tpu::InputOp>(pprevOp)) {
      this->useTpuQuant = false;
    } else {
      this->useTpuQuant = clUseTPUQuantOp;
    }
  } else {
    this->useTpuQuant = isa<tpu::InputOp>(prevOp) ? false : clUseTPUQuantOp;
  }
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void QuantOpKernel::invoke() {
  if (this->from == "NONE" && this->to == "INT8") {
    quantizeFromFp32ToInt8(input_data->data(), output_data->data(),
                           input_data->size(), scale, zero_point, useTpuQuant);
  } else if ((this->from == "INT8" || this->from == "UINT8") &&
             this->to == "NONE") {
    dequantizeFromInt8ToFp32(input_data->data(), output_data->data(),
                             input_data->size(), scale, zero_point, true);
  } else if ((this->from == "INT8" || this->from == "UINT8") &&
             this->to == "BF16") {
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
  } else if (this->from == "NONE" && this->to == "INT16") {
    quantizeFromFp32ToInt16(input_data->data(), output_data->data(),
                             input_data->size(), scale);
  } else if (this->from == "NONE" && this->to == "UINT16") {
    quantizeFromFp32ToUInt16(input_data->data(), output_data->data(),
                             input_data->size(), scale);
  } else {
    dump();
    llvm_unreachable("TODO");
  }
}

ReQuantOpKernel::ReQuantOpKernel(Operation &op, value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  auto requantOp = cast<tpu::ReQuantOp>(op);
  this->scale = requantOp.qscale().convertToFloat();
  this->input_offset = (float)-getPreviousOpZeroPoint(&op);
  this->output_offset = (float)getOpZeroPoint(&op);
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void ReQuantOpKernel::invoke() {
  std::vector<float> input_data_asym(input_data->begin(), input_data->end());
  // convert fp32 to bf16
  auto tensor_bf16 =
      std::make_unique<std::vector<bfloat16>>(output_data->size());

  clean16bitmantissa(&scale, &scale, 1);
  clean16bitmantissa(input_data_asym.data(), output_data->data(),
                     input_data_asym.size());

  for (size_t i = 0; i < tensor_bf16->size(); i++) {
    output_data->at(i) += (float)input_offset;
  }
  clean16bitmantissa(output_data->data(), output_data->data(),
                     output_data->size());

  for (size_t i = 0; i < tensor_bf16->size(); i++) {
    output_data->at(i) *= scale;
  }
  clean16bitmantissa(output_data->data(), output_data->data(),
                     output_data->size());
  for (size_t i = 0; i < tensor_bf16->size(); i++) {
    output_data->at(i) += (float)output_offset;
  }
  clean16bitmantissa(output_data->data(), output_data->data(),
                     output_data->size());
  // round
  for (size_t i = 0; i < tensor_bf16->size(); i++) {
    float sub_part;
    float int_part;
    sub_part = std::modf(output_data->at(i), &int_part);
    // subpart 0.5
    if (std::abs(std::abs(sub_part) - 0.5f) < 0.001) {
      output_data->at(i) = std::round(output_data->at(i) / 2) * 2;
    } else if (std::abs(sub_part) > 0.0f) {
      output_data->at(i) = std::nearbyint(output_data->at(i));
    }
  }
}

} // namespace mlir
