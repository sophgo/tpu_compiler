#ifndef INTERPRETER_CPU_MATCHTEMPLATE_H
#define INTERPRETER_CPU_MATCHTEMPLATE_H

#include "tpuc/Interpreter/cpukernel.h"

#include <memory>
namespace mlir {

class MatchTemplateOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUMatchTemplateOp";
  MatchTemplateOpKernel(Operation &op, value_map_t &valueMapping,
                    weight_map_t &weightMapping);
  void invoke() override;

private:
  void ccoeff_normed_fp32(float *input, float *tmplate, float *dst, int size);
  void ccoeff_normed_bf16(float *input, float *tmplate, float *dst, int size);
  void sqiff_fp32(float *input, float *tmplate, float *dst, int size);
  void sqiff_bf16(float *input, float *tmplate, float *dst, int size);

private:
  SyncedData input_data;
  SyncedData template_data;
  SyncedData output_data;
  SyncedData lut;
  SyncedData mantissa_lut;
  SyncedDataShape input_shape;
  SyncedDataShape template_shape;

  // param
  std::string mode;
  int32_t outer_size;
  int32_t match_size;
  int32_t n, c, h, w, stride;
};
} // namespace mlir

#endif
