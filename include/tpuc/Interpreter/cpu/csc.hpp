#ifndef INTERPRETER_CPU_CSC_H
#define INTERPRETER_CPU_CSC_H

#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {

class CscOpKernel : public CPUOpKernel {
public:
  enum YuvType {
    YUV_UNKNOWN = 0,
    YUV420_PLANAR = 1,
    YUV_NV12 = 2,
    YUV_NV21 = 3
  };

  static constexpr const char *OpName = "CPUCscOp";

  CscOpKernel(Operation &op, value_map_t &valueMapping,
              weight_map_t &weightMapping);

  void invoke() override;

private:
  void yuv_csc(float *input, float *output, int n, int c, int h, int w,
                  std::vector<int> &order, int quant_type, YuvType pixel_type);
  private:
    SyncedData input_data;
    SyncedData output_data;
    SyncedDataShape input_shape;

    // param
    std::string pixel_format;
    int aligned;
    int y_align, w_align, channel_align;
  };
} // namespace mlir
#endif