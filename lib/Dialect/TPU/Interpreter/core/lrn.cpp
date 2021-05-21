#include "tpuc/Interpreter/cpu/lrn.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"
#include "tpuc/NativeCpuImplementation.h"

template <typename Dtype>
static void array_mul(const int N, const Dtype *a, Dtype *b, Dtype *y) {
  for (int i = 0; i < N; i++) {
    y[i] = a[i] * b[i];
  }
}

template <typename Dtype>
static void array_axpy(const int N, const Dtype alpha, const Dtype *X,
                       Dtype *Y) {
  for (int i = 0; i < N; i++) {
    Y[i] = X[i] * alpha + Y[i];
  }
}

template <typename Dtype>
static void array_ax(const int N, const Dtype *X, const Dtype alpha, Dtype *Y) {
  for (int i = 0; i < N; i++) {
    Y[i] = X[i] * alpha;
  }
}

template <typename Dtype>
static void array_add(const int N, const Dtype *X, const Dtype alpha,
                      Dtype *Y) {
  for (int i = 0; i < N; i++) {
    Y[i] = X[i] + alpha;
  }
}

template <typename Dtype>
static void array_powx(const int N, const Dtype *a, const Dtype b, Dtype *y) {
  for (int i = 0; i < N; i++) {
    y[i] = std::pow(a[i], b);
  }
}

// lrn step one
void lrn_one(float *input, float *output, int n, int c, int h, int w,
             unsigned int local_size, float alpha) {
  int count = n * c * h * w;
  array_mul(count, input, input, output);
  array_ax(count, output, alpha / local_size, output);
}

// lrn step two
void lrn_two(float *input, float *output, int n, int c, int h, int w,
             unsigned int local_size) {
  int count = n * c * h * w;
  // start with the constant value
  for (int i = 0; i < count; ++i) {
    output[i] = 0;
  }
  int batch_size = c * h * w;
  int frame_size = h * w;
  int pre_pad = (local_size - 1) / 2;
  std::vector<float> padded_square((c + local_size - 1) * h * w, 0.0f);
  float *padded_square_data = padded_square.data();
  // go through the images
  for (int index_n = 0; index_n < n; ++index_n) {
    float *in_data = input + index_n * batch_size;
    float *out_data = output + index_n * batch_size;
    memcpy(padded_square_data + pre_pad * frame_size, in_data,
           batch_size * sizeof(float));
    for (uint32_t index_c = 0; index_c < local_size; ++index_c) {
      array_axpy(frame_size, 1.0f, padded_square_data + index_c * frame_size,
                 out_data);
    }
    for (int index_c = 1; index_c < c; ++index_c) {
      // copy previous scale
      memcpy(out_data + index_c * frame_size,
             out_data + (index_c - 1) * frame_size, frame_size * sizeof(float));
      // add head
      array_axpy(frame_size, 1.0f,
                 padded_square_data + (index_c + local_size - 1) * frame_size,
                 out_data + index_c * frame_size);
      // subtract tail
      array_axpy(frame_size, -1.0f,
                 padded_square_data + (index_c - 1) * frame_size,
                 out_data + index_c * frame_size);
    }
  }
}

// lrn step three
void lrn_three(float *input, float *output, int n, int c, int h, int w,
               float beta, float k) {
  int count = n * c * h * w;
  array_add(count, input, k, output);
  array_powx(count, output, -beta, output);
}

void lrn_main(float *input, float *scale, float *output, int n, int c, int h,
              int w) {
  int count = n * c * h * w;
  array_mul(count, scale, input, output);
}

void lrn_int8(float *input, float *output, int n, int c, int h, int w,
              unsigned int local_size, float *sqr_lut, float *power_lut,
              int sq_right_shift, int lrn_right_shift, int quant0, int quant1) {
  int count = n * c * h * w;
  int pre_pad = (local_size - 1) / 2;
  int padded_c = c + local_size - 1;
  int batch_size = c * h * w;
  int frame_size = h * w;
  std::vector<float> padded_square(padded_c * h * w, 0.0f);
  std::vector<float> scale(count, 0.0f);
  float *padded_square_data = padded_square.data();
  float *scale_data = scale.data();
  for (int i = 0; i < count; i++) {
    output[i] = 0;
  }
  float *square_data = padded_square_data + pre_pad * frame_size;
  for (int index_n = 0; index_n < n; index_n++) {
    float *in_ndata = input + index_n * batch_size;
    float *scale_ndata = scale_data + index_n * batch_size;
    for (int i = 0; i < batch_size; i++) {
      square_data[i] = sqr_lut[(uint8_t)in_ndata[i]];
    }
    for (uint32_t index_c = 0; index_c < local_size; ++index_c) {
      array_axpy(frame_size, (float)quant0,
                 padded_square_data + index_c * frame_size, scale_ndata);
    }
    for (int index_c = 1; index_c < c; ++index_c) {
      // copy previous scale
      memcpy(scale_ndata + index_c * frame_size,
             scale_ndata + (index_c - 1) * frame_size,
             frame_size * sizeof(float));
      // add head
      array_axpy(frame_size, (float)quant0,
                 padded_square_data + (index_c + local_size - 1) * frame_size,
                 scale_ndata + index_c * frame_size);
      // subtract tail
      array_axpy(frame_size, (float)-quant0,
                 padded_square_data + (index_c - 1) * frame_size,
                 scale_ndata + index_c * frame_size);
    }
  }
  float sq_scale = 1.0f / (1 << sq_right_shift);
  float lrn_scale = 1.0f / (1 << lrn_right_shift);
  for (int i = 0; i < count; i++) {
    scale_data[i] = std::floor(scale_data[i] * sq_scale + 0.5f);
    if (scale_data[i] < 0.0f) {
      scale_data[i] = 0.0f;
    } else if (scale_data[i] > 255.0f) {
      scale_data[i] = 255.0;
    }
    output[i] = power_lut[(uint8_t)scale_data[i]];
    output[i] *= input[i] * quant1 * lrn_scale;
    output[i] = std::floor(output[i] + 0.5f);
    if (output[i] < -128.0f) {
      output[i] = -128.0f;
    } else if (output[i] > 127.0f) {
      output[i] = 127.0f;
    }
  }
}

namespace mlir {
LrnOpKernel::LrnOpKernel(Operation &op, value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  auto lrnOp = cast<tpu::LrnOp>(op);
  this->input_shape = getTensorShape(op.getOperand(0));
  if (this->input_shape.size() < 4) {
    llvm_unreachable("input shape size less than 4");
  }
  this->local_size = lrnOp.local_size();
  this->alpha = lrnOp.alpha().convertToFloat();
  this->beta = lrnOp.beta().convertToFloat();
  this->k = lrnOp.k().convertToFloat();
  if (datatype == DataType::INT8) {
    sqr_lut_data = this->opdTensors[1];
    power_lut_data = this->opdTensors[2];
    this->sum_rshift = lrnOp.sum_rshift();
    this->lrn_rshift = lrnOp.lrn_rshift();
    this->quant_data0 = lrnOp.quant_data0();
    this->quant_data1 = lrnOp.quant_data1();
  }
  // get tensors
  input_data = this->opdTensors[0];
  scale_data = this->opdTensors[3];
  output_data = this->resTensor;
}

void LrnOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " Lrn op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};
std::vector<float> LrnOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void LrnOpKernel::invoke() {

  int n = input_shape[0];
  int c = input_shape[1];
  int h = input_shape[2];
  int w = input_shape[3];
  size_t input_size = n * c * h * w;
  if (datatype == DataType::FP32) {
    if (scale_data == nullptr) {
      scale_data = std::make_shared<std::vector<float>>(input_size);
      lrn_one(input_data->data(), scale_data->data(), n, c, h, w, local_size,
              alpha);

      lrn_two(scale_data->data(), output_data->data(), n, c, h, w, local_size);

      lrn_three(output_data->data(), scale_data->data(), n, c, h, w, beta, k);
    }
    lrn_main(input_data->data(), scale_data->data(), output_data->data(), n, c,
             h, w);

  } else if (datatype == DataType::INT8) {
    my_lrn_int8(input_data->data(), output_data->data(), n, c, h, w, local_size,
                sqr_lut_data->data(), power_lut_data->data(), sum_rshift,
                lrn_rshift, quant_data0, quant_data1);
  } else {
    if (scale_data == nullptr) {
      scale_data = std::make_shared<std::vector<float>>(input_size);
      lrn_one(input_data->data(), scale_data->data(), n, c, h, w, local_size,
              alpha);

      lrn_two(scale_data->data(), output_data->data(), n, c, h, w, local_size);

      lrn_three(output_data->data(), scale_data->data(), n, c, h, w, beta, k);
    }
    lrn_main(input_data->data(), scale_data->data(), output_data->data(), n, c,
             h, w);
    clean16bitmantissa(output_data->data(), output_data->data(),
                       output_data->size());
  }
}
void LrnOpKernel::dump() { OpKernel::dump(); }

LrnOneOpKernel::LrnOneOpKernel(Operation &op, value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  auto lrnOp = cast<tpu::LrnOneOp>(op);
  this->input_shape = getTensorShape(op.getOperand(0));
  if (this->input_shape.size() < 4) {
    llvm_unreachable("input shape size less than 4");
  }
  this->local_size = lrnOp.local_size();
  this->alpha = lrnOp.alpha().convertToFloat();
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void LrnOneOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " Lrn op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};
std::vector<float> LrnOneOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void LrnOneOpKernel::invoke() {

  int n = input_shape[0];
  int c = input_shape[1];
  int h = input_shape[2];
  int w = input_shape[3];
  lrn_one(input_data->data(), output_data->data(), n, c, h, w, local_size,
          alpha);
}
void LrnOneOpKernel::dump() { OpKernel::dump(); }

LrnTwoOpKernel::LrnTwoOpKernel(Operation &op, value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  auto lrnOp = cast<tpu::LrnTwoOp>(op);
  this->input_shape = getTensorShape(op.getOperand(0));
  if (this->input_shape.size() < 4) {
    llvm_unreachable("input shape size less than 4");
  }
  this->local_size = lrnOp.local_size();
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void LrnTwoOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " Lrn op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};
std::vector<float> LrnTwoOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void LrnTwoOpKernel::invoke() {

  int n = input_shape[0];
  int c = input_shape[1];
  int h = input_shape[2];
  int w = input_shape[3];

  lrn_two(input_data->data(), output_data->data(), n, c, h, w, local_size);
}
void LrnTwoOpKernel::dump() { OpKernel::dump(); }

LrnThreeOpKernel::LrnThreeOpKernel(Operation &op, value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  auto lrnOp = cast<tpu::LrnThreeOp>(op);
  this->input_shape = getTensorShape(op.getOperand(0));
  if (this->input_shape.size() < 4) {
    llvm_unreachable("input shape size less than 4");
  }
  this->k = lrnOp.k().convertToFloat();
  this->beta = lrnOp.beta().convertToFloat();
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void LrnThreeOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " Lrn op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};
std::vector<float> LrnThreeOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void LrnThreeOpKernel::invoke() {

  int n = input_shape[0];
  int c = input_shape[1];
  int h = input_shape[2];
  int w = input_shape[3];

  lrn_three(input_data->data(), output_data->data(), n, c, h, w, beta, k);
}
void LrnThreeOpKernel::dump() { OpKernel::dump(); }
} // namespace mlir