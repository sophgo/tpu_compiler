#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/TPU/CpuLayer_DetectionOutput.h"


namespace mlir {

bool SortScoreCmp0 (const pair<float,int> &pair1, const pair<float,int> &pair2) {
  return pair1.first > pair2.first;
}

bool SortScoreCmp1 (const pair<float, pair<int, int>>& pair1, const pair<float, pair<int, int>>& pair2) {
  return pair1.first > pair2.first;
}

void GetConfidenceScores_opt (const float* conf_data, const int num,
      const int num_preds_per_class, const int num_classes, const float score_threshold,
      vector<map<int, vector<pair<float ,int>> > >* conf_preds) {
  conf_preds->clear();
  conf_preds->resize(num);
  for (int i = 0; i < num; ++i) {
    map<int, vector<pair<float ,int>> >& label_scores = (*conf_preds)[i];
    for (int p = 0; p < num_preds_per_class; ++p) {
      int start_idx = p * num_classes;
      for (int c = 0; c < num_classes; ++c) {
        if (conf_data[start_idx + c] > score_threshold) {
          label_scores[c].push_back(std::make_pair(conf_data[start_idx + c],p));
        }
      }
    }
    conf_data += num_preds_per_class * num_classes;
  }

}



void GetLocPredictions_opt (const float* loc_data, const int num,
      const int num_preds_per_class, const int num_loc_classes,
      const bool share_location, float *decode_index,
      vector<LabelBBox_l>* loc_preds) {
  loc_preds->clear();
  if (share_location) {
    assert(num_loc_classes==1);
  }
  loc_preds->resize(num);
  float * decode_pos = decode_index;
  for (int i = 0; i < num; ++i) {
    if (share_location) {
      decode_pos = decode_index + i*num_preds_per_class;
    }
    LabelBBox_l& label_bbox = (*loc_preds)[i];
    for (int p = 0; p < num_preds_per_class; ++p) {
      int start_idx = p * num_loc_classes * 4;
      for (int c = 0; c < num_loc_classes; ++c) {
        if (!share_location) {
          decode_pos = decode_index + num_preds_per_class*num_loc_classes*i + num_preds_per_class*c;
        }
        int label = share_location ? -1 : c;
        if (label_bbox.find(label) == label_bbox.end()) {
          label_bbox[label].resize(num_preds_per_class);
        }
        if (decode_pos[p]!=1) {
          continue;
        }
        label_bbox[label][p].xmin = loc_data[start_idx + c * 4];
        label_bbox[label][p].ymin = loc_data[start_idx + c * 4 + 1];
        label_bbox[label][p].xmax = loc_data[start_idx + c * 4 + 2];
        label_bbox[label][p].ymax = loc_data[start_idx + c * 4 + 3];
      }
    }
    loc_data += num_preds_per_class * num_loc_classes * 4;
  }
}

void DecodeBBoxesAll_opt (const vector<LabelBBox_l>& all_loc_preds,
    int num_priors, const float* prior_data,
    const int num, const bool share_location,
    const int num_loc_classes, const int background_label_id,
    const CodeType code_type, const bool variance_encoded_in_target,
    const bool clip, float *decode_index ,
    vector<LabelBBox_l>* all_decode_bboxes) {
  assert(all_loc_preds.size()==num);
  all_decode_bboxes->clear();
  all_decode_bboxes->resize(num);
  float * decode_pos = decode_index;
  for (int i = 0; i < num; ++i) {
    if (share_location) {
      decode_pos = decode_index + i*num_priors;
    }
    // Decode predictions into bboxes.
    for (int c = 0; c < num_loc_classes; ++c) {
      int label = share_location ? -1 : c;
      if (label == background_label_id) {
        // Ignore background class.
        continue;
      }
      if (all_loc_preds[i].find(label) == all_loc_preds[i].end()) {
       llvm::errs() << "Could not find location predictions for label " << label;
      }
      const vector<BBox_l>& bboxes = all_loc_preds[i].find(label)->second;
      LabelBBox_l& decode_bboxes = (*all_decode_bboxes)[i];
      vector<BBox_l>* p = &(decode_bboxes[label]);
      p->clear();

      if (!share_location) {
        decode_pos = decode_index + num_priors*num_loc_classes*i + num_priors*c;
      }
      for (int k = 0; k < num_priors; ++k) {
        //NormalizedBBox decode_bbox;
        BBox_l decode_bbox;
        if (decode_pos[k] != 1) {
          p->push_back(decode_bbox);
          continue;
        }
        //opt CENTER_SIZE
        assert (code_type==PriorBoxParameter_CodeType_CENTER_SIZE);
        //prior_bboxes
        int start_idx = k * 4;
        const float *p0 = prior_data + start_idx;
        const float *p1 = prior_data + start_idx + 4*num_priors;
        float prior_width = p0[2] - p0[0];
        assert(prior_width > 0);
        float prior_height = p0[3] - p0[1];
        assert(prior_height > 0);
        float prior_center_x = (p0[0] + p0[2]) * 0.5;
        float prior_center_y = (p0[1] + p0[3]) * 0.5;

        float decode_bbox_center_x, decode_bbox_center_y;
        float decode_bbox_width, decode_bbox_height;
        if (variance_encoded_in_target) {
          // variance is encoded in target, we simply need to retore the offset
          // predictions.
          decode_bbox_center_x = bboxes[k].xmin * prior_width + prior_center_x;
          decode_bbox_center_y = bboxes[k].ymin * prior_height + prior_center_y;
          decode_bbox_width = exp(bboxes[k].xmax) * prior_width;
          decode_bbox_height = exp(bboxes[k].ymax) * prior_height;
        } else {
          // variance is encoded in bbox, we need to scale the offset accordingly.
          decode_bbox_center_x = p1[0] * bboxes[k].xmin * prior_width + prior_center_x;
          decode_bbox_center_y = p1[1] * bboxes[k].ymin * prior_height + prior_center_y;
          decode_bbox_width = exp(p1[2] * bboxes[k].xmax) * prior_width;
          decode_bbox_height = exp(p1[3] * bboxes[k].ymax) * prior_height;
        }
        decode_bbox.xmin = decode_bbox_center_x - decode_bbox_width * 0.5;
        decode_bbox.ymin = decode_bbox_center_y - decode_bbox_height * 0.5;
        decode_bbox.xmax = decode_bbox_center_x + decode_bbox_width * 0.5;
        decode_bbox.ymax = decode_bbox_center_y + decode_bbox_height * 0.5;
        decode_bbox.CalcSize();
        p->push_back(decode_bbox);
      }
    }
  }
}

void ApplyNMSFast_opt (const vector<BBox_l>& bboxes, const vector<pair<float ,int>> & conf_score ,
     const float score_threshold, const float nms_threshold, const float eta, int top_k,
     vector<pair<float,int>>* indices) {
  // Do nms.
  float adaptive_threshold = nms_threshold;
  int i = 0;
  int length = (top_k < (int)conf_score.size()) ? top_k : conf_score.size();
    while (length != i) {
    bool keep = true;
    for (int k = 0; k < (int)indices->size(); ++k) {
      if (keep) {
        const int kept_idx = (*indices)[k].second;
        const BBox_l & b1 = bboxes[conf_score[i].second];
        const BBox_l & b2 = bboxes[kept_idx];
        if (b2.xmin > b1.xmax || b2.xmax < b1.xmin ||
            b2.ymin > b1.ymax || b2.ymax < b1.ymin) {
          keep = true;
        }
        else {
          const float inter_xmin = std::max(b1.xmin, b2.xmin);
          const float inter_ymin = std::max(b1.ymin, b2.ymin);
          const float inter_xmax = std::min(b1.xmax, b2.xmax);
          const float inter_ymax = std::min(b1.ymax, b2.ymax);
          const float inter_width = inter_xmax - inter_xmin;
          const float inter_height = inter_ymax - inter_ymin;
          const float inter_size = inter_width * inter_height;
          const float total_size = b1.size + b2.size;
          keep = (inter_size*(adaptive_threshold+1) <= total_size*adaptive_threshold) ? true : false;
        }
      } else {
        break;
      }
    }
    if (keep) {
      indices->push_back(conf_score[i]);
    }
    if (keep && eta < 1 && adaptive_threshold > 0.5) {
      adaptive_threshold *= eta;
    }
    i++;
  }
}

} // namespace
