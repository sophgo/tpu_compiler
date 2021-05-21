#include "tpuc/Interpreter/cpu/csc.hpp"
#include "bmkernel/bm1880v2/1880v2_fp_convert.h"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Interpreter/cpu/crop.hpp"
#include "tpuc/ModuleInterpreter.h"
#include "tpuc/Interpreter/cpu/activation.hpp"

static inline int align_up(int x, int n) {
  if (n == 0 || n == 1) {
    return x;
  }
  return ((x + n - 1) / n) * n;
}

static inline float UINT8(float data) {
  return static_cast<float>(convert_fp32_u8(data));
}

void yuv420_csc(float *input, float *output, int n, int c, int h, int w,
                std::vector<int> &order, int quant_type) {
  int y_w_aligned = align_up(w, 32);
  int uv_w_aligned = align_up(w / 2, 32);
  int y_offset = 0;
  int u_offset = align_up(h * y_w_aligned, 0x1000);
  int v_offset = align_up(u_offset + h / 2 * uv_w_aligned, 0x1000);
  int n_stride = align_up(v_offset + h / 2 * uv_w_aligned, 0x1000);
  for (int idx_n = 0; idx_n < n; idx_n++) {
    for (int idx_h = 0; idx_h < h; idx_h++) {
      for (int idx_w = 0; idx_w < w; idx_w++) {
        int y_idx = y_offset + idx_n * n_stride + idx_h * y_w_aligned + idx_w;
        int u_idx =
            u_offset + idx_n * n_stride + idx_h / 2 * uv_w_aligned + idx_w / 2;
        int v_idx =
            v_offset + idx_n * n_stride + idx_h / 2 * uv_w_aligned + idx_w / 2;
        float y = input[y_idx];
        float u = input[u_idx];
        float v = input[v_idx];
        float r, g, b;
        if (quant_type == 0) {
          y = 1.164 * (y - 16.0f);
          u -= 128;
          v -= 128;
          // float:
          r = y + 1.596 * v;
          g = y - 0.813 * v - 0.391 * u;
          b = y + 2.018 * u;
        } else {
          // u8 or bf16

          y = (float)(uint8_t)y;
          u = (float)(uint8_t)u;
          v = (float)(uint8_t)v;

          y = BF16(BF16(1.164) * (y - 16.0f));
          u -= 128.0f;
          v -= 128.0f;
          r = BF16(y + BF16(1.596f) * v);
          g = BF16(BF16(y + BF16(-0.813f) * v) + BF16(-0.391f) * u);
          b = BF16(y + BF16(2.018f) * u);
        }
        r = UINT8(r);
        g = UINT8(g);
        b = UINT8(b);

        float color[3] = {b, g, r};
        int c0_idx = idx_n * 3 * h * w + idx_h * w + idx_w;
        int c1_idx = c0_idx + h * w;
        int c2_idx = c1_idx + h * w;
        output[c0_idx] = color[order[0]];
        output[c1_idx] = color[order[1]];
        output[c2_idx] = color[order[2]];
      }
    }
  }
}

namespace mlir {

CscOpKernel::CscOpKernel(Operation &op, value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  auto cscOp = cast<tpu::CscOp>(op);

  auto input_type = cscOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();
  this->pixel_format = cscOp.pixel_format().str();
  this->aligned = cscOp.aligned();
  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;
}

void CscOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " CscOp op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> CscOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void CscOpKernel::invoke() {
  if (shape.size() < 4 || input_shape.size() < 4) {
    dump();
    llvm_unreachable("wrong shape size");
  }
  int on = shape[0];
  int oc = shape[1];
  int oh = shape[2];
  int ow = shape[3];
  int ic = input_shape[1];
  int ih = input_shape[2];

  if (pixel_format == "YUV420_PLANAR") {
    std::vector<int> orders{0, 1, 2};
    yuv420_csc(input_data->data(), output_data->data(), on, oc, oh, ow, orders,
               datatype == DataType::FP32 ? 0 : 1);
  } else if (aligned) {
    // do crop to make data unaligned
    std::vector<int64_t> crop_shape(input_shape.begin(), input_shape.end());
    crop_shape[3] = (int)(oc * oh * ow / (ic * ih));
    std::vector<int> crop_offset{0, 0, 0, 0};
    std::vector<int> indices(4, 0);
    crop(input_data->data(), output_data->data(), input_shape.data(),
         crop_shape.data(), 0, crop_offset.data(), indices.data());
  } else {
    output_data->assign(input_data->begin(), input_data->end());
  }
}
void CscOpKernel::dump() { OpKernel::dump(); }
} // namespace mlir