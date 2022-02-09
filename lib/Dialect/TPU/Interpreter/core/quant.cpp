#include "tpuc/Interpreter/cpu/quant.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/MlirModuleInterpreter.h"
#include "tpuc/QuantizationArithmetic.h"

#include <cmath>

// Quantize an Activation tensor into INT8, given threshold
static void quantizeFromFp32ToInt8(float *src, float *dst, int64_t size,
                                   float scale, int zero_point, bool tpu_mode) {
  if (tpu_mode) {
    scale = BF16(scale);
    for (int64_t i = 0; i < size; ++i) {
      // remove [17:31] mantissa part
      // align \TgQuantKernel.cpp that we directly use high part
      // rather than convert it
      float val = BF16(BF16(BF16(src[i], false) * scale) + zero_point,
                       (zero_point != 0));
      dst[i] = (float)F32ToInt8(val, 0);
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

static void quantizeFromFp32ToUInt8(float *src, float *dst, int64_t size) {
  for (int64_t i = 0; i < size; ++i) {
    float f32_val = src[i];
    dst[i] = (float)F32ToUint8(f32_val, 0);
  }
}

static void quantizeFromFp32ToBf16(float *src, float *dst, int64_t size) {
  for (int64_t i = 0; i < size; ++i) {
    float f32_val = src[i];
    uint32_t *u32_val = reinterpret_cast<uint32_t *>(&f32_val);
    uint32_t input = *u32_val;
    bfloat16 bf_val = (bfloat16)(input >> 16);

    uint16_t *q = reinterpret_cast<uint16_t *>(&dst[i]);
    q[0] = 0;
    q[1] = bf_val;
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
static void dequantizeFromInt8ToBf16(float *src, float *dst, int64_t size,
                                     float scale, int zero_point) {
  scale = BF16(scale);
  zero_point = BF16(zero_point);
  for (int64_t i = 0; i < size; ++i) {
    dst[i] = BF16(BF16(src[i] + zero_point, (zero_point != 0)) * scale);
  }
}

static void quantizeFromBf16ToInt8(float *output, float *input, int64_t size,
                                   float scale) {
  scale = BF16(scale);
  for (int64_t i = 0; i < size; ++i) {
    float val = BF16(input[i] * scale);
    output[i] = (float)F32ToInt8(val, 0);
  }
}

/// DeQuantize an Activation tensor from INT8, given threshold
static void dequantizeFromInt8ToFp32(float *src, float *dst, int64_t size,
                                     float scale, int zero_point,
                                     bool tpu_mode) {
  if (tpu_mode) {
    scale = BF16(scale);
    for (int64_t i = 0; i < size; ++i) {
      dst[i] = BF16(BF16(src[i] + zero_point, (zero_point != 0)) * scale);
    }
  } else {
    for (int64_t i = 0; i < size; ++i) {
      dst[i] = (src[i] + zero_point) * scale;
    }
  }
}

namespace mlir {
QuantOpKernel::QuantOpKernel(Operation &op, value_map_t &valueMapping,
                             weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  auto quantOp = cast<tpu::QuantOp>(op);
  this->scale = quantOp.scale().convertToFloat();
  this->zero_point = 0;
  this->from = quantOp.from().str();
  this->to = quantOp.to().str();
  auto prevOp = quantOp.getOperand().getDefiningOp(); // input
  if (isa<tpu::ReshapeOp>(prevOp)) {
    auto pprevOp = cast<tpu::ReshapeOp>(prevOp).getOperand().getDefiningOp();
    this->useTpuQuant = isa<tpu::InputOp>(pprevOp) ? false : true;
  } else {
    this->useTpuQuant = isa<tpu::InputOp>(prevOp) ? false : true;
  }
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void QuantOpKernel::invoke() {
  if (this->from == "NONE" && this->to == "INT8") {
    quantizeFromFp32ToInt8(input_data->data(), output_data->data(),
                           input_data->size(), scale, zero_point, useTpuQuant);
  }else if (this->from == "NONE" && this->to == "UINT8") {
    quantizeFromFp32ToUInt8(input_data->data(), output_data->data(),
                             input_data->size());
  }else if ((this->from == "INT8" || this->from == "UINT8") &&
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
    quantizeFromFp32ToBf16(input_data->data(), output_data->data(),
                           output_data->size());
  } else if (this->from == "BF16" && this->to == "INT8") {
    quantizeFromBf16ToInt8(output_data->data(), input_data->data(),
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

} // namespace mlir
