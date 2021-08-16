#include "tpuc/Interpreter/cpu/interpolation.hpp"
#include "internal.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"
#include "tpuc/NativeCpuImplementation.h"

static inline float coordinate_transform(float x_resized, float x_scale,
                                         float length_resized, bool pytorch) {
  // please refer NativeCpuImplementation.cpp for more details
  if (pytorch) {
    return length_resized > 1 ? ((x_resized + 0.5f) / x_scale - 0.5f) : 0.0f;
  } else {
    return (x_resized + 0.5f) / x_scale - 0.5f;
  }
}

// copy from caffe_cpu_interp2
void interp(const int channels, const float *data1, const int x1, const int y1,
            const int height1, const int width1, const int Height1,
            const int Width1, float *data2, const int x2, const int y2,
            const int height2, const int width2, const int Height2,
            const int Width2) {
  bool packed = false;

  assert(x1 >= 0 && y1 >= 0 && height1 > 0 && width1 > 0 && x2 >= 0 &&
         y2 >= 0 && height2 > 0 && width2 > 0);
  assert(Width1 >= width1 + x1 && Height1 >= height1 + y1 &&
         Width2 >= width2 + x2 && Height2 >= height2 + y2);

  // special case: just copy
  if (height1 == height2 && width1 == width2) {
    for (int h2 = 0; h2 < height2; ++h2) {
      const int h1 = h2;
      for (int w2 = 0; w2 < width2; ++w2) {
        const int w1 = w2;
        if (packed) {
          const float *pos1 =
              &data1[channels * ((y1 + h1) * Width1 + (x1 + w1))];
          float *pos2 = &data2[channels * ((y2 + h2) * Width2 + (x2 + w2))];
          for (int c = 0; c < channels; ++c) {
            pos2[0] = pos1[0];
            pos1++;
            pos2++;
          }
        } else {
          const float *pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
          float *pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
          for (int c = 0; c < channels; ++c) {
            pos2[0] = pos1[0];
            pos1 += Width1 * Height1;
            pos2 += Width2 * Height2;
          }
        }
      }
    }
    return;
  }
  const float rheight =
      (height2 > 1) ? static_cast<float>(height1 - 1) / (height2 - 1) : 0.f;
  const float rwidth =
      (width2 > 1) ? static_cast<float>(width1 - 1) / (width2 - 1) : 0.f;
  for (int h2 = 0; h2 < height2; ++h2) {
    const float h1r = rheight * h2;
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const float h1lambda = h1r - h1;
    const float h0lambda = float(1.) - h1lambda;
    for (int w2 = 0; w2 < width2; ++w2) {
      const float w1r = rwidth * w2;
      const int w1 = w1r;
      const int w1p = (w1 < width1 - 1) ? 1 : 0;
      const float w1lambda = w1r - w1;
      const float w0lambda = float(1.) - w1lambda;
      if (packed) {
        const float *pos1 = &data1[channels * ((y1 + h1) * Width1 + (x1 + w1))];
        float *pos2 = &data2[channels * ((y2 + h2) * Width2 + (x2 + w2))];
        for (int c = 0; c < channels; ++c) {
          pos2[0] =
              h0lambda *
                  (w0lambda * pos1[0] + w1lambda * pos1[channels * w1p]) +
              h1lambda * (w0lambda * pos1[channels * h1p * Width1] +
                          w1lambda * pos1[channels * (h1p * Width1 + w1p)]);
          pos1++;
          pos2++;
        }
      } else {
        const float *pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
        float *pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
        for (int c = 0; c < channels; ++c) {
          pos2[0] = h0lambda * (w0lambda * pos1[0] + w1lambda * pos1[w1p]) +
                    h1lambda * (w0lambda * pos1[h1p * Width1] +
                                w1lambda * pos1[h1p * Width1 + w1p]);
          pos1 += Width1 * Height1;
          pos2 += Width2 * Height2;
        }
      }
    }
  }
}

template <typename T>
void upsampleBilinear(int64_t batch_size, int64_t num_channels,
                      int64_t input_height, int64_t input_width,
                      float height_scale, float width_scale, const T *Xdata,
                      T *Ydata, bool pytorch = false) {
  int64_t output_width = static_cast<int64_t>(input_width * width_scale);
  int64_t output_height = static_cast<int64_t>(input_height * height_scale);

  for (int64_t n = 0; n < batch_size; ++n) {
    for (int64_t c = 0; c < num_channels; ++c) {
      for (int64_t y = 0; y < output_height; ++y) {
        float in_y =
            std::min(y / height_scale, static_cast<float>(input_height - 1));
        in_y = height_scale == 1
                   ? static_cast<float>(y)
                   : coordinate_transform(static_cast<float>(y), height_scale,
                                          static_cast<float>(output_height),
                                          pytorch);
        in_y = std::max(0.0f,
                        std::min(in_y, static_cast<float>(input_height - 1)));

        const int64_t in_y1 =
            std::min(static_cast<int64_t>(in_y), input_height - 1);
        const int64_t in_y2 = std::min(in_y1 + 1, input_height - 1);
        float dy1 = fabs(in_y - in_y1);
        float dy2 = fabs(in_y - in_y2);
        if (in_y1 == in_y2) {
          dy1 = 0.5f;
          dy2 = 0.5f;
        }

        const int64_t input_width_mul_y1 = input_width * in_y1;
        const int64_t input_width_mul_y2 = input_width * in_y2;

        for (int64_t x = 0; x < output_width; ++x) {
          float in_x =
              std::min(x / width_scale, static_cast<float>(input_width - 1));
          in_x = width_scale == 1
                     ? static_cast<float>(x)
                     : coordinate_transform(static_cast<float>(x), width_scale,
                                            static_cast<float>(output_width),
                                            pytorch);
          in_x = std::max(0.0f,
                          std::min(in_x, static_cast<float>(input_width - 1)));

          const int64_t in_x1 =
              std::min(static_cast<int64_t>(in_x), input_width - 1);
          const int64_t in_x2 = std::min(in_x1 + 1, input_width - 1);

          float dx1 = std::abs(in_x - in_x1);
          float dx2 = std::abs(in_x - in_x2);
          if (in_x1 == in_x2) {
            dx1 = 0.5f;
            dx2 = 0.5f;
          }

          T X11 = Xdata[input_width_mul_y1 + in_x1];
          T X21 = Xdata[input_width_mul_y1 + in_x2];
          T X12 = Xdata[input_width_mul_y2 + in_x1];
          T X22 = Xdata[input_width_mul_y2 + in_x2];

          Ydata[output_width * y + x] =
              static_cast<T>(dx2 * dy2 * X11 + dx1 * dy2 * X21 +
                             dx2 * dy1 * X12 + dx1 * dy1 * X22);
        }
      }
      Xdata += input_height * input_width;
      Ydata += output_width * output_height;
    }
  }
}

void interp_linear(float *input, float *output, int n, int c, int ih, int iw,
                   int oh, int ow, bool pytorch = false) {
  float height_scale = (float)oh / (float)ih;
  float width_scale = (float)ow / (float)iw;
  upsampleBilinear<float>(n, c, ih, iw, height_scale, width_scale, input,
                          output, pytorch);
}

void interp_neast(float *input, float *output, int n, int c, int ih, int iw,
                  int oh, int ow, bool half_pixel) {
  int nc = n * c;
  float scale_h = ((float)ih) / oh;
  float scale_w = ((float)iw) / ow;
#pragma omp parallel for schedule(static, omp_schedule(nc))
  for (int i = 0; i < nc; i++) {
    for (int h = 0; h < oh; h++) {
      for (int w = 0; w < ow; w++) {
        int o_index = i * oh * ow + h * ow + w;
        int h_resized = (int)(half_pixel ? std::ceil((h + 0.5) * scale_h - 1.0)
                                         : h * scale_h);
        int w_resized = (int)(half_pixel ? std::ceil((w + 0.5) * scale_w - 1.0)
                                         : w * scale_w);
        int i_index = i * ih * iw + h_resized * iw + w_resized;
        output[o_index] = input[i_index];
      }
    }
  }
}

static inline float value(float *input, int w, int ih, int iw) {
  return input[ih * w + iw];
}

static float value(float *input, int w, float fh, float fw) {
  int h0 = std::floor(fh);
  int h1 = std::ceil(fh);
  int w0 = std::floor(fw);
  int w1 = std::ceil(fw);
  if (h0 == fh && w0 == fw) {
    return value(input, w, h0, w0);
  }
  if (h0 == fh) {
    return value(input, w, h0, w0) * (w1 - fw) +
           value(input, w, h0, w1) * (fw - w0);
  }
  if (w0 == fw) {
    return value(input, w, h0, w0) * (h1 - fh) +
           value(input, w, h1, w0) * (fh - h0);
  }
  float scale0 = (w1 - fw) * (h1 - fh);
  float scale1 = (fw - w0) * (h1 - fh);
  float scale2 = (w1 - fw) * (fh - h0);
  float scale3 = (fw - w0) * (fh - h0);
  return value(input, w, h0, w0) * scale0 + value(input, w, h0, w1) * scale1 +
         value(input, w, h1, w0) * scale2 + value(input, w, h1, w1) * scale3;
}

void interp_asymmetric(float *input, float *output, int n, int c, int ih,
                       int iw, int oh, int ow) {
  int nc = n * c;
  float scale_h = (float)ih / oh;
  float scale_w = (float)iw / ow;
#pragma omp parallel for schedule(static, omp_schedule(nc))
  for (int i = 0; i < nc; i++) {
    for (int h = 0; h < oh; h++) {
      for (int w = 0; w < ow; w++) {
        int o_index = i * oh * ow + h * ow + w;
        float fh = std::min(h * scale_h, (float)(ih - 1));
        float fw = std::min(w * scale_w, (float)(iw - 1));
        output[o_index] = value(input + i * ih * iw, iw, fh, fw);
      }
    }
  }
}

namespace mlir {

InterpolationOpKernel::InterpolationOpKernel(Operation &op,
                                             value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  auto interpolationOp = cast<tpu::InterpOp>(op);
  auto input_type =
      interpolationOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();
  this->height = interpolationOp.height();
  this->width = interpolationOp.width();
  this->pad_beg = interpolationOp.pad_beg();
  this->pad_end = interpolationOp.pad_end();
  this->name = interpolationOp.name().str();
  this->shrink_factor = interpolationOp.shrink_factor();
  this->zoom_factor = interpolationOp.zoom_factor();
  this->coordinate_transformation_mode =
      interpolationOp.coordinate_transformation_mode().str();
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void InterpolationOpKernel::invoke() {
  int in = input_shape[0];
  int ic = input_shape[1];
  int ih = input_shape[2];
  int iw = input_shape[3];
  int oh = shape[2];
  int ow = shape[3];
  int height_in_ = ih;
  int width_in_ = iw;
  int height_in_eff_ = height_in_ + pad_beg + pad_end;
  int width_in_eff_ = width_in_ + pad_beg + pad_end;
  int height_out_ = -1;
  int width_out_ = -1;
  if (this->shrink_factor && !this->zoom_factor) {
    assert(shrink_factor >= 1 && "Shrink factor must be positive");
    height_out_ = (height_in_eff_ - 1) / shrink_factor + 1;
    width_out_ = (width_in_eff_ - 1) / shrink_factor + 1;
  } else if (this->zoom_factor && !this->shrink_factor) {
    assert(zoom_factor >= 1 && "Zoom factor must be positive");
    height_out_ = height_in_eff_ + (height_in_eff_ - 1) * (zoom_factor - 1);
    width_out_ = width_in_eff_ + (width_in_eff_ - 1) * (zoom_factor - 1);
  } else if (this->height && this->width) {
    height_out_ = this->height;
    width_out_ = this->width;
  } else if (this->zoom_factor && this->shrink_factor) {
    assert(shrink_factor >= 1 && "Shrink factor must be positive");
    assert(zoom_factor >= 1 && "Zoom factor must be positive");

    height_out_ = (height_in_eff_ - 1) / shrink_factor + 1;
    width_out_ = (width_in_eff_ - 1) / shrink_factor + 1;
    height_out_ = height_out_ + (height_out_ - 1) * (zoom_factor - 1);
    width_out_ = width_out_ + (width_out_ - 1) * (zoom_factor - 1);
  }
  if (coordinate_transformation_mode == "align_corners") {
    // TODO: verify pad_end_ > 0
    my_interp(in * ic, input_data->data(), -pad_beg, -pad_beg, height_in_eff_,
              width_in_eff_, height_in_, width_in_, output_data->data(), 0, 0,
              height_out_, width_out_, height_out_, width_out_);
  } else if (coordinate_transformation_mode == "half_pixel") {
    interp_linear(input_data->data(), output_data->data(), in, ic, ih, iw, oh,
                  ow);
  } else if (coordinate_transformation_mode == "pytorch_half_pixel") {
    interp_linear(input_data->data(), output_data->data(), in, ic, ih, iw, oh,
                  ow, true);
  } else if (coordinate_transformation_mode == "nearest_half_pixel") {
    interp_neast(input_data->data(), output_data->data(), in, ic, ih, iw, oh,
                 ow, true);
  } else if (coordinate_transformation_mode == "nearest") {
    interp_neast(input_data->data(), output_data->data(), in, ic, ih, iw, oh,
                 ow, false);
  } else if (coordinate_transformation_mode == "asymmetric") {
    interp_asymmetric(input_data->data(), output_data->data(), in, ic, ih, iw,
                      oh, ow);
  } else {
    llvm_unreachable("coordinate_transformation_model not support");
  }
}

} // namespace mlir