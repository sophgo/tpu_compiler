#include "tpuc/Interpreter/cpu/pooling.hpp"
#include "internal.hpp"
#include "mkldnn.h"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Interpreter/cpu/conv.hpp"
#include "tpuc/Interpreter/cpu/slice.hpp"
#include "tpuc/MachineInfo.h"
#include "tpuc/MlirModuleInterpreter.h"

namespace mlir {

PoolingOpKernel::PoolingOpKernel(Operation &op, value_map_t &valueMapping,
                                 weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  tpu::PoolParam pool_param;
  bool is_avg;
  if (isa<tpu::PoolAvg2DOp>(op)) {
    auto poolavgOp = cast<tpu::PoolAvg2DOp>(op);
    pool_method = POOL_METHOD::AVG;
    pool_param = poolavgOp.param();
    is_avg = true;
  } else if (isa<tpu::PoolMax2DOp>(op)) {
    auto poolmaxOp = cast<tpu::PoolMax2DOp>(op);
    pool_method = POOL_METHOD::MAX;
    pool_param = poolmaxOp.param();
    is_avg = false;
  }

  this->input_shape = getTensorShape(op.getOperand(0));

  auto input_value = op.getOperand(0);
  auto result = op.getResult(0);
  parsePoolParam(pool_param, input_value, result, n, c, ih, iw, oh, ow, kh, kw,
                 sh, sw, pt, pb, pl, pr, pad_value, is_global, do_relu,
                 count_include_pad);
  assert(!do_relu);

  // get tensors
  input_data = this->opdTensors[0];
  output_data = this->resTensor;

  if (datatype == DataType::INT8 && pool_method == POOL_METHOD::AVG) {
    // in int8 average pool case:
    // SyQy = Avg(SxQx)
    // Qy = 1/Sy *  Sx * Qxi * 1 / (kh * kw)
    // mkldnn pool can not simulate this case,
    // we use detphwise conv,  use mutlipiler and rshift to handle 1 / (kh* kw)
    // clean mkldn and reset
    auto quant_rshift = this->opdTensors[3];
    auto quant_multiplier = this->opdTensors[4];
    if (!quant_rshift) {
      llvm_unreachable("quant_rshift is null!");
    }
    if (!quant_multiplier) {
      llvm_unreachable("quant_multiplier is null!");
    }
    this->rshift = quant_rshift->at(0);
    this->multiplier = quant_multiplier->at(0);
    int8_avg_pool.setup(n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr);
  } else {
    pool.setup(n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, is_avg,
               count_include_pad);
  }
}

void PoolingOpKernel::invoke() {

  if (datatype == DataType::FP32) {
    pool.run(input_data->data(), output_data->data());
  } else if (datatype == DataType::BF16) {
    pool.run(input_data->data(), output_data->data());
  } else {
    if (pool_method == POOL_METHOD::AVG) {
      int lmem_size = MInfo::lmem_per_lane;
      // why?, copy code from previous implement.
      int size = output_data->size();
      if ((ih * iw) > ((lmem_size - size) / 2) && kh == ih && kw == iw) {
        llvm::errs() << "split average pool\n";
        // In hardware limitation, we can not put avg pool with large kernel
        // if avg pool ih * iw > local memory, in our hardware
        // need to split it then sum
        std::vector<int> h_slices;
        int h_slice_size = (int)(((lmem_size - size) / iw) / 2);
        int total_h = ih;
        while (total_h > 0) {
          if (total_h > h_slice_size) {
            total_h -= h_slice_size;
            h_slices.push_back(h_slice_size);
          } else {
            h_slices.push_back(total_h);
            break;
          }
        }
        int offset = 0;
        std::vector<float> output_data_(size, 0);
        for (auto &s : h_slices) {
          int filter_shape = c * s * kw;
          int g = c;
          int oc = c;
          int dh = 1, dw = 1;
          int input_slice_size = n * c * s * kw;
          std::vector<float> conv_filter(filter_shape, 1);
          std::vector<float> input_slice(input_slice_size);
          std::vector<float> output_tmp_data(size);
          std::vector<int64_t> tmp_shape = {n, c, s, iw};
          slice(input_data->data(), input_slice.data(), 2, offset, input_shape,
                tmp_shape);
          llvm::errs() << "slice xxxxxxx, s:" << s << "\n";
          mkldnn_conv(input_slice.data(), conv_filter.data(), NULL,
                      output_tmp_data.data(), n, c, s, iw, oc, 1, 1, s, kw, sh,
                      sw, dh, dw, pt, pb, pl, pr, g, 0);
          offset += s;
          for (int64_t i = 0; i < size; ++i) {
            float sum = output_tmp_data[i];
            output_tmp_data[i] = (float)applyMultiplierAndRShiftAndSaturateInt8(
                sum, (uint32_t)rshift, (uint32_t)multiplier, false);
            output_data_[i] += output_tmp_data[i];
          }
        }
        for (int64_t i = 0; i < size; ++i) {
          output_data->at(i) = output_data_[i];
        }
      } else {
        int8_avg_pool.run(input_data->data(), output_data->data());
        size_t output_size = output_data->size();
#pragma omp parallel for schedule(static, omp_schedule(output_size))
        for (size_t i = 0; i < output_size; ++i) {
          output_data->at(i) = (float)applyMultiplierAndRShiftAndSaturateInt8(
              output_data->at(i), (uint32_t)rshift, (uint32_t)multiplier,
              false);
        }
      }
    } else {
      pool.run(input_data->data(), output_data->data());
    }
  }
}

} // namespace mlir