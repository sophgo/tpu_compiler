#ifndef INTERPRETER_CPU_DETECTION_OUTPUT_H
#define INTERPRETER_CPU_DETECTION_OUTPUT_H

#include "tpuc/CpuLayer_DetectionOutput.h"
#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {

class DetectionOutputOpKernel : public CPUOpKernel<DetectionOutputOpKernel> {
public:
  static constexpr const char *OpName = "CPUDetectionOutputOp";

  DetectionOutputOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
  void set_tensor(const std::vector<float> &data) override;
  std::vector<float> get_tensor() override;
  void dump() override;

private:
  SyncedData loc_data;
  SyncedData conf_data;
  SyncedData prior_data;
  SyncedData output_data;
  SyncedDataShape loc_shape;
  SyncedDataShape conf_shape;
  SyncedDataShape prior_shape;

  // param
  int keep_top_k;
  float confidence_threshold;
  float nms_threshold;
  int top_k;
  int num_classes;
  bool share_location;
  int background_label_id;
  Decode_CodeType code_type;
};
} // namespace mlir
#endif