#ifndef INTERPRETER_CPU_PREPROCESS_H
#define INTERPRETER_CPU_PREPROCESS_H

#include "tpuc/Interpreter/cpukernel.h"

#include <memory>
namespace mlir {
void preprocess(float *input, float *output, const std::vector<int64_t> &shape,
                const std::vector<int> &channel_order,
                const std::vector<float> &mean, const std::vector<float> &std,
                float raw_scale, float input_scale);

class PreprocessOpKernel : public CPUOpKernel<PreprocessOpKernel> {
public:
  static constexpr const char *OpName = "CPUPreprocessOp";

  PreprocessOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
  void set_tensor(const std::vector<float> &data) override;
  std::vector<float> get_tensor() override;
  void dump() override;

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedDataShape input_shape;

  // param
  std::vector<unsigned int> transpose_orders;
  std::vector<float> means;
  std::vector<float> stds;
  std::vector<int> crop_offset;
  std::vector<int> color_orders;
  float raw_scale;
  float input_scale;
};
} // namespace mlir

#endif
