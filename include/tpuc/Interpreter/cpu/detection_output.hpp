#ifndef INTERPRETER_CPU_DETECTION_OUTPUT_H
#define INTERPRETER_CPU_DETECTION_OUTPUT_H

#include "tpuc/CpuLayer_DetectionOutput.h"
#include "tpuc/Interpreter/cpukernel.h"

namespace mlir {

class DetectionOutputOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUDetectionOutputOp";

  DetectionOutputOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

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

class YoloDetectionOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUYoloDetectionOpKernel";

  YoloDetectionOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  std::vector<SyncedData> inputs_data;

  SyncedData output_data;
  std::vector<SyncedDataShape> inputs_shape;

  // param
  int net_input_h;
  int net_input_w;
  float obj_threshold;
  float nms_threshold;
  int input_count;

  int keep_topk;
  int num_classes;
  bool tiny;
  bool share_location;
  bool yolo_v4;
  int class_num;

  std::vector<float> vec_anchors;
};

class FrcnDetectionOpKernel : public CPUOpKernel {
public:
  static constexpr const char *OpName = "CPUFrcnDetectionOp";

  FrcnDetectionOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;

private:
  SyncedData bbox_deltas;
  SyncedData scores;
  SyncedData rois;
  SyncedData output_data;
  SyncedDataShape rois_shape;
  SyncedDataShape scores_shape;
  SyncedDataShape bbox_deltas_shape;

  // param
  int keep_topk;
  int class_num;
  float nms_threshold;
  float obj_threshold;
};
} // namespace mlir
#endif