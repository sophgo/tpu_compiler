#include "tpuc/Interpreter/cpu/instancenorm.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {
InstanceNormOpKernel::InstanceNormOpKernel(Operation &op,
                                           value_map_t &valueMapping) {
  auto inOp = cast<tpu::InstanceNormOp>(op);
  assert(inOp);
  LLVM_DEBUG(llvm::outs() << " InstanceNorm op: [" << inOp.name() << "]\n";);

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = inOp.getResult();
  auto size = getTensorSize(result);
  LLVM_DEBUG(llvm::outs() << "    =>required memory size: [" << size << "]\n";);
  auto resultTensor = std::make_shared<std::vector<float>>(size);

  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();
  this->name = inOp.name().str();
  this->op_type = op.getName().getStringRef().str();
  // only cpu layer
  set_datatype("NONE");
  // get tensors

  // gamma_value * (x - mean_value) / np.sqrt(var_value + epsilon) + beta_value
  // where `mean` and `variance` are computed per instance per channel
  // mean = np.mean(x, axis=axis, keepdims=True)
  // var = np.var(x, axis=axis, keepdims=True)

  // input
  input_data = opTensors[0];
  scale = opTensors[1];
  bias = opTensors[2];
  variance_epsilon = inOp.variance_epsilon().convertToFloat();

  // output
  output_data = resultTensor;

  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}

void InstanceNormOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " InstanceNorm op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};
std::vector<float> InstanceNormOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void InstanceNormOpKernel::invoke() {
  if (datatype != DataType::FP32) {
    llvm_unreachable("except fp32, other mode please fused instancenorm");
  }
  int n = this->shape.at(0);
  int c = this->shape.at(1);
  int h = this->shape.size() > 2 ? this->shape.at(2) : 1;
  int w = this->shape.size() > 3 ? this->shape.at(3) : 1;

  // gamma_value * (x - mean_value) / np.sqrt(var_value + epsilon) + beta_value
  // epsilon default is 1e-5
  // please reference onnx
  // [implementation](https://github.com/onnx/onnx/blob/master/docs/Operators.md#InstanceNormalization)
  // caffe2 cpu
  // [implementation](https://caffe2.ai/doxygen-c/html/instance__norm__op_8cc_source.html)

  std::vector<float> _mean(c);
  std::vector<float> _variance(c);
  int hw = h * w;

  for (int ni = 0; ni < n; ++ni) {
    for (int ci = 0; ci < c; ++ci) {
      int channel_shift = ni * c * h * w + ci * h * w;
      auto start = input_data->begin() + channel_shift;
      _mean[ci] = accumulate(start, start + hw, 0.0) / hw;

      float var = 0;

      for (int i = 0; i < hw; ++i) {
        var +=
            pow(input_data->at(ni * c * h * w + ci * h * w + i) - _mean[ci], 2);
      }
      var = (var) / hw;
      _variance[ci] = var;
    }
  }

  auto mean = &_mean;
  auto variance = &_variance;

  // duplicate code from bn
  float scale_factor = 1 / scale->at(0);
  for (int i = 0; i < c; ++i) {
    mean->at(i) = mean->at(i) * scale_factor;
    variance->at(i) = variance->at(i) * scale_factor;
  }
  for (int ni = 0; ni < n; ++ni) {
    for (int ci = 0; ci < c; ++ci) {
      for (int i = 0; i < h * w; ++i) {
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
  for (int i = 0; i < c; ++i) {
    mean->at(i) = mean->at(i) * scale->at(0);
    variance->at(i) = variance->at(i) * scale->at(0);
  }
}

void InstanceNormOpKernel::dump() { OpKernel::dump(); }

} // namespace mlir
