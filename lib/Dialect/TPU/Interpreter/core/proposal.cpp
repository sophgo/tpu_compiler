#include "tpuc/Interpreter/cpu/proposal.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"

static void _mkanchors(std::vector<float> ctrs, std::vector<float> &anchors) {
  anchors.push_back(ctrs[2] - 0.5 * (ctrs[0] - 1));
  anchors.push_back(ctrs[3] - 0.5 * (ctrs[1] - 1));
  anchors.push_back(ctrs[2] + 0.5 * (ctrs[0] - 1));
  anchors.push_back(ctrs[3] + 0.5 * (ctrs[1] - 1));
}

static void _whctrs(std::vector<float> anchor, std::vector<float> &ctrs) {
  float w = anchor[2] - anchor[0] + 1;
  float h = anchor[3] - anchor[1] + 1;
  float x_ctr = anchor[0] + 0.5 * (w - 1);
  float y_ctr = anchor[1] + 0.5 * (h - 1);
  ctrs.push_back(w);
  ctrs.push_back(h);
  ctrs.push_back(x_ctr);
  ctrs.push_back(y_ctr);
}

static void _ratio_enum(std::vector<float> anchor,
                        std::vector<float> anchor_ratio,
                        std::vector<float> &ratio_anchors) {
  std::vector<float> ctrs;
  _whctrs(anchor, ctrs);
  float size = ctrs[0] * ctrs[1];
  int ratio_num = anchor_ratio.size();
  for (int i = 0; i < ratio_num; i++) {
    float ratio = size / anchor_ratio[i];
    int ws = int(std::round(std::sqrt(ratio)));
    int hs = int(std::round(ws * anchor_ratio[i]));
    std::vector<float> ctrs_in;
    ctrs_in.push_back(ws);
    ctrs_in.push_back(hs);
    ctrs_in.push_back(ctrs[2]);
    ctrs_in.push_back(ctrs[3]);
    _mkanchors(ctrs_in, ratio_anchors);
  }
}

static void _scale_enum(std::vector<float> ratio_anchors,
                        std::vector<float> anchor_scale,
                        std::vector<float> &anchor_boxes) {
  int anchors_ratio_num = ratio_anchors.size() / 4;
  for (int i = 0; i < anchors_ratio_num; i++) {
    std::vector<float> anchor;
    anchor.push_back(ratio_anchors[i * 4]);
    anchor.push_back(ratio_anchors[i * 4 + 1]);
    anchor.push_back(ratio_anchors[i * 4 + 2]);
    anchor.push_back(ratio_anchors[i * 4 + 3]);
    std::vector<float> ctrs;
    _whctrs(anchor, ctrs);
    int scale_num = anchor_scale.size();
    for (int j = 0; j < scale_num; j++) {
      float ws = ctrs[0] * anchor_scale[j];
      float hs = ctrs[1] * anchor_scale[j];
      std::vector<float> ctrs_in;
      ctrs_in.push_back(ws);
      ctrs_in.push_back(hs);
      ctrs_in.push_back(ctrs[2]);
      ctrs_in.push_back(ctrs[3]);
      _mkanchors(ctrs_in, anchor_boxes);
    }
  }
}

static void generate_anchors(int anchor_base_size,
                             std::vector<float> anchor_scale,
                             std::vector<float> anchor_ratio,
                             std::vector<float> &anchor_boxes) {
  std::vector<float> base_anchor = {0, 0, (float)(anchor_base_size - 1),
                                    (float)(anchor_base_size - 1)};
  std::vector<float> ratio_anchors;
  _ratio_enum(base_anchor, anchor_ratio, ratio_anchors);
  _scale_enum(ratio_anchors, anchor_scale, anchor_boxes);
}

static void
anchor_box_transform_inv(float img_width, float img_height,
                         std::vector<std::vector<float>> bbox,
                         std::vector<std::vector<float>> select_anchor,
                         std::vector<std::vector<float>> &pred) {
  int num = bbox.size();
  for (int i = 0; i < num; i++) {
    float dx = bbox[i][0];
    float dy = bbox[i][1];
    float dw = bbox[i][2];
    float dh = bbox[i][3];
    float pred_ctr_x = select_anchor[i][0] + select_anchor[i][2] * dx;
    float pred_ctr_y = select_anchor[i][1] + select_anchor[i][3] * dy;
    float pred_w = select_anchor[i][2] * std::exp(dw);
    float pred_h = select_anchor[i][3] * std::exp(dh);
    std::vector<float> tmp_pred;
    tmp_pred.push_back(
        std::max(std::min((float)(pred_ctr_x - 0.5 * pred_w), img_width - 1),
                 (float)0.0));
    tmp_pred.push_back(
        std::max(std::min((float)(pred_ctr_y - 0.5 * pred_h), img_height - 1),
                 (float)0.0));
    tmp_pred.push_back(
        std::max(std::min((float)(pred_ctr_x + 0.5 * pred_w), img_width - 1),
                 (float)0.0));
    tmp_pred.push_back(
        std::max(std::min((float)(pred_ctr_y + 0.5 * pred_h), img_height - 1),
                 (float)0.0));
    pred.push_back(tmp_pred);
  }
}

static void anchor_box_nms(std::vector<std::vector<float>> &pred_boxes,
                           std::vector<float> &confidence,
                           float nms_threshold) {
  for (size_t i = 0; i < pred_boxes.size() - 1; i++) {
    float s1 = (pred_boxes[i][2] - pred_boxes[i][0] + 1) *
               (pred_boxes[i][3] - pred_boxes[i][1] + 1);
    for (size_t j = i + 1; j < pred_boxes.size(); j++) {
      float s2 = (pred_boxes[j][2] - pred_boxes[j][0] + 1) *
                 (pred_boxes[j][3] - pred_boxes[j][1] + 1);

      float x1 = std::max(pred_boxes[i][0], pred_boxes[j][0]);
      float y1 = std::max(pred_boxes[i][1], pred_boxes[j][1]);
      float x2 = std::min(pred_boxes[i][2], pred_boxes[j][2]);
      float y2 = std::min(pred_boxes[i][3], pred_boxes[j][3]);

      float width = x2 - x1;
      float height = y2 - y1;
      if (width > 0 && height > 0) {
        float IOU = width * height / (s1 + s2 - width * height);
        if (IOU > nms_threshold) {
          if (confidence[i] >= confidence[j]) {
            pred_boxes.erase(pred_boxes.begin() + j);
            confidence.erase(confidence.begin() + j);
            j--;
          } else {
            pred_boxes.erase(pred_boxes.begin() + i);
            confidence.erase(confidence.begin() + i);
            i--;
            break;
          }
        }
      }
    }
  }
}

namespace mlir {

ProposalOpKernel::ProposalOpKernel(Operation &op, value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  auto proposalOp = cast<tpu::ProposalOp>(op);
  this->score_shape = op.getOperand(0).getType().cast<TensorType>().getShape();
  this->bbox_shape = op.getOperand(1).getType().cast<TensorType>().getShape();

  this->net_input_h = proposalOp.net_input_h();
  this->net_input_w = proposalOp.net_input_w();
  this->feat_stride = proposalOp.feat_stride();
  this->anchor_base_size = proposalOp.anchor_base_size();
  this->rpn_obj_threshold = proposalOp.rpn_obj_threshold().convertToFloat();
  this->rpn_nms_threshold = proposalOp.rpn_nms_threshold().convertToFloat();
  this->rpn_nms_post_top_n = proposalOp.rpn_nms_post_top_n();
  generate_anchors(anchor_base_size, anchor_scale, anchor_ratio, anchor_boxes);

  // get tensors
  score = this->opdTensors[0];
  bbox_deltas = this->opdTensors[1];
  output_data = this->resTensor;
}

void ProposalOpKernel::set_tensor(const std::vector<float> &data) {
  llvm_unreachable("TODO!");
};

std::vector<float> ProposalOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void ProposalOpKernel::invoke() {
  int batch = score_shape[0];
  int channel = score_shape[1];
  int height = score_shape[2];
  int width = score_shape[3];

  float thresh = rpn_obj_threshold;

  for (int b = 0; b < batch; ++b) {
    auto batched_score = score->data() + b * channel * height * width;
    auto batched_bbox_deltas =
        bbox_deltas->data() + b * bbox_shape[1] * bbox_shape[2] * bbox_shape[3];
    std::vector<std::vector<float>> select_anchor;
    std::vector<float> confidence;
    std::vector<std::vector<float>> bbox;
    int anchor_num = anchor_scale.size() * anchor_ratio.size();

    for (int k = 0; k < anchor_num; k++) {
      float w = anchor_boxes[4 * k + 2] - anchor_boxes[4 * k] + 1;
      float h = anchor_boxes[4 * k + 3] - anchor_boxes[4 * k + 1] + 1;
      float x_ctr = anchor_boxes[4 * k] + 0.5 * (w - 1);
      float y_ctr = anchor_boxes[4 * k + 1] + 0.5 * (h - 1);

      for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
          if (batched_score[anchor_num * height * width +
                            (k * height + i) * width + j] >= thresh) {
            std::vector<float> tmp_anchor;
            std::vector<float> tmp_bbox;

            tmp_anchor.push_back(j * feat_stride + x_ctr);
            tmp_anchor.push_back(i * feat_stride + y_ctr);
            tmp_anchor.push_back(w);
            tmp_anchor.push_back(h);
            select_anchor.push_back(tmp_anchor);
            confidence.push_back(batched_score[anchor_num * height * width +
                                               (k * height + i) * width + j]);
            tmp_bbox.push_back(
                batched_bbox_deltas[(4 * k * height + i) * width + j]);
            tmp_bbox.push_back(
                batched_bbox_deltas[((4 * k + 1) * height + i) * width + j]);
            tmp_bbox.push_back(
                batched_bbox_deltas[((4 * k + 2) * height + i) * width + j]);
            tmp_bbox.push_back(
                batched_bbox_deltas[((4 * k + 3) * height + i) * width + j]);
            bbox.push_back(tmp_bbox);
          }
        }
      }
    }
    std::vector<std::vector<float>> pred_boxes;
    anchor_box_transform_inv(net_input_w, net_input_h, bbox, select_anchor,
                             pred_boxes);
    anchor_box_nms(pred_boxes, confidence, rpn_nms_threshold);
    int num = pred_boxes.size() > rpn_nms_post_top_n ? rpn_nms_post_top_n
                                                     : pred_boxes.size();

    auto batched_output =
        output_data->data() + b * shape[1] * shape[2] * shape[3];
    for (int i = 0; i < num; i++) {
      batched_output[5 * i] = b;
      batched_output[5 * i + 1] = pred_boxes[i][0];
      batched_output[5 * i + 2] = pred_boxes[i][1];
      batched_output[5 * i + 3] = pred_boxes[i][2];
      batched_output[5 * i + 4] = pred_boxes[i][3];
    }
  }
}

void ProposalOpKernel::dump() { OpKernel::dump(); }
} // namespace mlir