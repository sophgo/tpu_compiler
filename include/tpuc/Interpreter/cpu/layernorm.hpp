#ifndef INTERPRETER_CPU_LAYERNORM_H
#define INTERPRETER_CPU_LAYERNORM_H

#include "tpuc/Interpreter/cpukernel.h"

#include <memory>
namespace mlir {

class LayerNormOpKernel : public CPUOpKernel<LayerNormOpKernel> {
public:
  static constexpr const char *OpName = "CPULayerNormOp";
  LayerNormOpKernel(Operation &op, value_map_t &valueMapping);
  void invoke() override;
  void set_tensor(const std::vector<float> &data) override;
  std::vector<float> get_tensor() override;
  void dump() override;

private:
  void normalize_fp32(float *src, float *dst, int size);
  void normalize_bf16(float *src, float *dst, int size);

private:
  SyncedData input_data;
  SyncedData output_data;
  SyncedData scale_data;
  SyncedData bias_data;
  SyncedData lut;
  SyncedData mantissa_lut;
  // param
  int32_t normalized_size;
  int32_t batch_size;
  float eps;
  bool affine;
  std::vector<int32_t> normalized_shape;
};
} // namespace mlir

#endif
