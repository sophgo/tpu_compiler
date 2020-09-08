#ifndef CPU_LAYER_FASTER_RCNN_H
#define CPU_LAYER_FASTER_RCNN_H

#include <vector>

typedef struct {
    float x1, y1, x2, y2;
} coord;

typedef struct  {
    coord bbox;
    int cls;
    float score;
} detections;

void _mkanchors(std::vector<float> ctrs, std::vector<float> &anchors) {
  anchors.push_back(ctrs[2] - 0.5*(ctrs[0] - 1));
  anchors.push_back(ctrs[3] - 0.5*(ctrs[1] - 1));
  anchors.push_back(ctrs[2] + 0.5*(ctrs[0] - 1));
  anchors.push_back(ctrs[3] + 0.5*(ctrs[1] - 1));
}

void _whctrs(std::vector<float> anchor, std::vector<float> &ctrs) {
  float w = anchor[2] - anchor[0] + 1;
  float h = anchor[3] - anchor[1] + 1;
  float x_ctr = anchor[0] + 0.5 * (w - 1);
  float y_ctr = anchor[1] + 0.5 * (h - 1);
  ctrs.push_back(w);
  ctrs.push_back(h);
  ctrs.push_back(x_ctr);
  ctrs.push_back(y_ctr);
}

void _ratio_enum(std::vector<float> anchor, std::vector<float> anchor_ratio,
                 std::vector<float> &ratio_anchors) {
  std::vector<float> ctrs;
  _whctrs(anchor, ctrs);
  float size = ctrs[0] * ctrs[1];
  int ratio_num = anchor_ratio.size();
  for (int i = 0; i < ratio_num; i++)
  {
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

void _scale_enum(std::vector<float> ratio_anchors, std::vector<float> anchor_scale,
                 std::vector<float> &anchor_boxes) {
  int anchors_ratio_num = ratio_anchors.size() / 4;
  for (int i = 0; i < anchors_ratio_num; i++)
  {
    std::vector<float> anchor;
    anchor.push_back(ratio_anchors[i * 4]);
    anchor.push_back(ratio_anchors[i * 4 + 1]);
    anchor.push_back(ratio_anchors[i * 4 + 2]);
    anchor.push_back(ratio_anchors[i * 4 + 3]);
    std::vector<float> ctrs;
    _whctrs(anchor, ctrs);
    int scale_num = anchor_scale.size();
    for (int j = 0; j < scale_num; j++)
    {
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

void generate_anchors(int anchor_base_size, std::vector<float> anchor_scale,
                    std::vector<float> anchor_ratio, std::vector<float> &anchor_boxes) {
  std::vector<float> base_anchor = {0, 0, (float)anchor_base_size - 1, (float)anchor_base_size - 1};
  std::vector<float> ratio_anchors;
  _ratio_enum(base_anchor, anchor_ratio, ratio_anchors);
  _scale_enum(ratio_anchors, anchor_scale, anchor_boxes);
}

void anchor_box_transform_inv(float img_width, float img_height, std::vector<std::vector<float>> bbox,
                    std::vector<std::vector<float>> select_anchor, std::vector<std::vector<float>> &pred)
{
  int num = bbox.size();
  for (int i = 0; i< num; i++)
  {
    float dx = bbox[i][0];
    float dy = bbox[i][1];
    float dw = bbox[i][2];
    float dh = bbox[i][3];
    float pred_ctr_x = select_anchor[i][0] + select_anchor[i][2] * dx;
    float pred_ctr_y = select_anchor[i][1] + select_anchor[i][3] * dy;
    float pred_w = select_anchor[i][2] * std::exp(dw);
    float pred_h = select_anchor[i][3] * std::exp(dh);
    std::vector<float> tmp_pred;
    tmp_pred.push_back(std::max(std::min((float)(pred_ctr_x - 0.5* pred_w), img_width - 1), (float)0.0));
    tmp_pred.push_back(std::max(std::min((float)(pred_ctr_y - 0.5* pred_h), img_height - 1), (float)0.0));
    tmp_pred.push_back(std::max(std::min((float)(pred_ctr_x + 0.5* pred_w), img_width - 1), (float)0.0));
    tmp_pred.push_back(std::max(std::min((float)(pred_ctr_y + 0.5* pred_h), img_height - 1), (float)0.0));
    pred.push_back(tmp_pred);
  }
}

void anchor_box_nms(std::vector<std::vector<float>> &pred_boxes, std::vector<float> &confidence, float nms_threshold)
{
  for (int i = 0; i < (int)pred_boxes.size()-1; i++)
  {
    float s1 = (pred_boxes[i][2] - pred_boxes[i][0] + 1) *(pred_boxes[i][3] - pred_boxes[i][1] + 1);
    for (int j = i + 1; j < (int)pred_boxes.size(); j++)
    {
      float s2 = (pred_boxes[j][2] - pred_boxes[j][0] + 1) *(pred_boxes[j][3] - pred_boxes[j][1] + 1);

      float x1 = std::max(pred_boxes[i][0], pred_boxes[j][0]);
      float y1 = std::max(pred_boxes[i][1], pred_boxes[j][1]);
      float x2 = std::min(pred_boxes[i][2], pred_boxes[j][2]);
      float y2 = std::min(pred_boxes[i][3], pred_boxes[j][3]);

      float width = x2 - x1;
      float height = y2 - y1;
      if (width > 0 && height > 0)
      {
        float IOU = width * height / (s1 + s2 - width * height);
        if (IOU > nms_threshold)
        {
          if (confidence[i] >= confidence[j])
          {
            pred_boxes.erase(pred_boxes.begin() + j);
            confidence.erase(confidence.begin() + j);
            j--;
          }
          else
          {
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

void bbox_transform_inv(const float* boxes, const float* deltas, float* pred, int num, int class_num)
{
  for (int i = 0; i < num; ++i) {
    float height = boxes[i*4+3] - boxes[i*4+1] + 1;
    float width = boxes[i*4+2] - boxes[i*4+0] + 1;
    float ctr_x = boxes[i*4+0] + width * 0.5;
    float ctr_y = boxes[i*4+1] + height * 0.5;

    for (int j = 0; j < class_num; ++j) {
      float dx = deltas[i*class_num*4 + j*4 + 0];
      float dy = deltas[i*class_num*4 + j*4 + 1];
      float dw = deltas[i*class_num*4 + j*4 + 2];
      float dh = deltas[i*class_num*4 + j*4 + 3];

      float pred_ctr_x = dx * width + ctr_x;
      float pred_ctr_y = dy * height + ctr_y;
      float pred_w = std::exp(dw) * width;
      float pred_h = std::exp(dh) * height;

      pred[i*class_num*4 + j*4 + 0] = pred_ctr_x - pred_w / 2;
      pred[i*class_num*4 + j*4 + 1] = pred_ctr_y - pred_h / 2;
      pred[i*class_num*4 + j*4 + 2] = pred_ctr_x + pred_w / 2;
      pred[i*class_num*4 + j*4 + 3] = pred_ctr_y + pred_h / 2;
    }
  }
}

void nms(detections *dets, int num, float nms_threshold)
{
  for (int i = 0; i < num; i++) {
    if (dets[i].score == 0) {
      // erased already
      continue;
    }

    float s1 = (dets[i].bbox.x2 - dets[i].bbox.x1 + 1) * (dets[i].bbox.y2 - dets[i].bbox.y1 + 1);
    for (int j = i + 1; j < num; j++) {
      if (dets[j].score == 0) {
        // erased already
        continue;
      }
      if (dets[i].cls != dets[j].cls) {
        // not the same class
        continue;
      }

      float s2 = (dets[j].bbox.x2 - dets[j].bbox.x1 + 1) * (dets[j].bbox.y2 - dets[j].bbox.y1 + 1);

      float x1 = std::max(dets[i].bbox.x1, dets[j].bbox.x1);
      float y1 = std::max(dets[i].bbox.y1, dets[j].bbox.y1);
      float x2 = std::min(dets[i].bbox.x2, dets[j].bbox.x2);
      float y2 = std::min(dets[i].bbox.y2, dets[j].bbox.y2);

      float width = x2 - x1;
      float height = y2 - y1;
      if (width > 0 && height > 0) {
        float iou = width * height / (s1 + s2 - width * height);
        assert(iou <= 1.0f);
        if (iou > nms_threshold) {
          // overlapped, select one to erase
          if (dets[i].score < dets[j].score) {
            dets[i].score = 0;
          } else {
            dets[j].score = 0;
          }
        }
      }
    }
  }
}

#endif