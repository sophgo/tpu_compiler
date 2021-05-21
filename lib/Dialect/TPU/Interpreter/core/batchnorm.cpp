#include "tpuc/Interpreter/cpu/batchnorm.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"
#include "tpuc/Interpreter/cpu/activation.hpp"

namespace mlir {
BatchNormOpKernel::BatchNormOpKernel(Operation &op, value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  auto bnOp = cast<tpu::BatchNormOp>(op);
  // get tensors
  input_data = this->opdTensors[0];
  mean = this->opdTensors[1];
  variance = this->opdTensors[2];
  scale = this->opdTensors[3];
  variance_epsilon = bnOp.variance_epsilon().convertToFloat();
  output_data = this->resTensor;
}

void BatchNormOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " BatchNorm op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
}

std::vector<float> BatchNormOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void BatchNormOpKernel::invoke() {
  if (datatype != DataType::FP32) {
    llvm_unreachable("except fp32, other mode please fused batchnorm");
  }
  int n = this->shape.at(0);
  int c = this->shape.at(1);
  int h = this->shape.size() > 2 ? this->shape.at(2) : 1;
  int w = this->shape.size() > 3 ? this->shape.at(3) : 1;

  float scale_factor = 1 / scale->at(0);
#pragma omp parallel for schedule(static, omp_schedule(c))
  for (int i = 0; i < c; ++i) {
    mean->at(i) = mean->at(i) * scale_factor;
    variance->at(i) = variance->at(i) * scale_factor;
  }
  int planner = h * w;
#pragma omp parallel for collapse(3)
  for (int ni = 0; ni < n; ++ni) {
    for (int ci = 0; ci < c; ++ci) {
      for (int i = 0; i < planner; ++i) {
        auto x = input_data->at(ni * c * h * w + ci * h * w + i) - mean->at(ci);
        auto d = sqrt(variance->at(ci) + variance_epsilon);
        output_data->at(ni * c * h * w + ci * h * w + i) = x / d;
        if (fabs(variance->at(ci)) <= variance_epsilon &&
            fabs(mean->at(ci)) <= 1e-8 &&
            fabs(input_data->at(ni * c * h * w + ci * h * w + i)) >= 1.0e-4 &&
            fabs(output_data->at(ni * c * h * w + ci * h * w + i)) >= 1.0e-2) {
          llvm::errs() << "WARNING: BN: var too small, i=" << i
                       << ", v=" << std::to_string(variance->at(ci))
                       << ", m=" << std::to_string(mean->at(ci))
                       << "\n               "
                       << ", i="
                       << std::to_string(
                              input_data->at(ni * c * h * w + ci * h * w + i))
                       << ", x=" << std::to_string(x)
                       << ", d=" << std::to_string(d) << ", o="
                       << std::to_string(
                              output_data->at(ni * c * h * w + ci * h * w + i))
                       << "\n";
        }
      }
    }
  }
#pragma omp parallel for schedule(static, omp_schedule(c))
  for (int i = 0; i < c; ++i) {
    mean->at(i) = mean->at(i) * scale->at(0);
    variance->at(i) = variance->at(i) * scale->at(0);
  }
}
void BatchNormOpKernel::dump() { OpKernel::dump(); }
} // namespace mlir