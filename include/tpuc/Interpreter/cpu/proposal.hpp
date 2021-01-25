#ifndef INTERPRETER_CPU_PROPOSAL_H
#define INTERPRETER_CPU_PROPOSAL_H

#include "tpuc/Interpreter/cpukernel.h"

#include <memory>
namespace mlir {

class ProposalOpKernel : public CPUOpKernel<ProposalOpKernel> {
public:
  static constexpr const char *OpName = "CPUProposalOp";

  ProposalOpKernel(Operation &op, value_map_t &valueMapping);

  void invoke() override;
  void set_tensor(const std::vector<float> &data) override;
  std::vector<float> get_tensor() override;
  void dump() override;

private:
  SyncedData output_data;
  SyncedData score;
  SyncedData bbox_deltas;
  SyncedDataShape score_shape;
  SyncedDataShape bbox_shape;

  // param
  int net_input_h;
  int net_input_w;
  int feat_stride;
  int anchor_base_size;
  float rpn_obj_threshold;
  float rpn_nms_threshold;
  float rpn_nms_post_top_n;

  std::vector<float> anchor_scale = {8, 16, 32};
  std::vector<float> anchor_ratio = {0.5, 1, 2};
  std::vector<float> anchor_boxes;
};
} // namespace mlir

#endif
