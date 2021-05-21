#include "tpuc/Interpreter/cpu/detection_output.hpp"

#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Interpreter/cpu/crop.hpp"
#include "tpuc/Interpreter/cpu/permute.hpp"
#include "tpuc/ModuleInterpreter.h"

#define GET_INDEX(cell_idx, box_idx_in_cell, data_idx, num_cell, class_num)    \
  (box_idx_in_cell * (class_num + 5) * num_cell + data_idx * num_cell +        \
   cell_idx)

constexpr int MAX_DET = 200;
constexpr int MAX_DET_RAW = 500;

static inline float exp_fast(float x) {
  union {
    unsigned int i;
    float f;
  } v;
  v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);

  return v.f;
}

static inline float _sigmoid(float x, bool fast) {
  if (fast)
    return 1.0f / (1.0f + exp_fast(-x));
  else
    return 1.0f / (1.0f + exp(-x));
}

static inline float _softmax(float *probs, float *data, int input_stride,
                             int num_of_class, int *max_cls, bool fast) {
  float x[num_of_class];
  float max_x = -INFINITY;
  float min_x = INFINITY;
  for (int i = 0; i < num_of_class; i++) {
    x[i] = data[i * input_stride];
    if (x[i] > max_x) {
      max_x = x[i];
    }
    if (x[i] < min_x) {
      min_x = x[i];
    }
  }
  const float t = -100.0f;
  float exp_x[num_of_class];
  float sum = 0;
  for (int i = 0; i < num_of_class; i++) {
    x[i] = x[i] - max_x;
    if (min_x < t)
      x[i] = x[i] / min_x * t;
    if (fast)
      exp_x[i] = exp_fast(x[i]);
    else
      exp_x[i] = exp(x[i]);
    sum += exp_x[i];
  }
  float max_prob = 0;
  for (int i = 0; i < num_of_class; i++) {
    probs[i] = exp_x[i] / sum;
    if (probs[i] > max_prob) {
      max_prob = probs[i];
      *max_cls = i;
    }
  }
  return max_prob;
}

typedef struct box_ {
  float x, y, w, h;
} box;

typedef struct detection_ {
  box bbox;
  int cls;
  float score;
} detection;

typedef struct {
  float x1, y1, x2, y2;
} coord;

typedef struct {
  coord bbox;
  int cls;
  float score;
} detections;

// https : // github.com/ChenYingpeng/caffe-yolov3/blob/master/box.cpp
static float overlap(float x1, float w1, float x2, float w2) {
  float l1 = x1 - w1 / 2;
  float l2 = x2 - w2 / 2;
  float left = l1 > l2 ? l1 : l2;
  float r1 = x1 + w1 / 2;
  float r2 = x2 + w2 / 2;
  float right = r1 < r2 ? r1 : r2;
  return right - left;
}

static float box_intersection(box a, box b) {
  float w = overlap(a.x, a.w, b.x, b.w);
  float h = overlap(a.y, a.h, b.y, b.h);
  if (w < 0 || h < 0)
    return 0;
  float area = w * h;
  return area;
}

static float box_union(box a, box b) {
  float i = box_intersection(a, b);
  float u = a.w * a.h + b.w * b.h - i;
  return u;
}

//
// more aboud iou
//   https://github.com/ultralytics/yolov3/blob/master/utils/utils.py
// IoU = inter / (a + b - inter), can't handle enclosure issue
// GIoU, DIoU, CIoU?
//
static float box_iou(box a, box b) {
  return box_intersection(a, b) / box_union(a, b);
}
static void nms(detection *det, int num, float nms_threshold) {
  for (int i = 0; i < num; i++) {
    if (det[i].score == 0) {
      // erased already
      continue;
    }
    for (int j = i + 1; j < num; j++) {
      if (det[j].score == 0) {
        // erased already
        continue;
      }
      if (det[i].cls != det[j].cls) {
        // not the same class
        continue;
      }
      float iou = box_iou(det[i].bbox, det[j].bbox);
      assert(iou <= 1.0f);
      if (iou > nms_threshold) {
        // overlapped, select one to erase
        if (det[i].score < det[j].score) {
          det[i].score = 0;
        } else {
          det[j].score = 0;
        }
      }
    }
  }
}

static void process_feature(detection *det, int *det_idx, float *feature,
                            std::vector<int> grid_size, float *anchor,
                            std::vector<int> yolo_size, int num_of_class,
                            float obj_threshold) {
  int yolo_w = yolo_size[1];
  int yolo_h = yolo_size[0];
  std::cout << "grid_h: " << grid_size[0] << std::endl;
  std::cout << "grid_w: " << grid_size[1] << std::endl;
  std::cout << "obj_threshold: " << obj_threshold << std::endl;
  int num_boxes_per_cell = 3;
  // assert(num_of_class == 80);

  // 255 = 3 * (5 + 80)
  // feature in shape [3][5+80][grid_size][grid_size]
  constexpr int COORD_X_INDEX = 0;
  constexpr int COORD_Y_INDEX = 1;
  constexpr int COORD_W_INDEX = 2;
  constexpr int COORD_H_INDEX = 3;
  constexpr int CONF_INDEX = 4;
  constexpr int CLS_INDEX = 5;
  int num_cell = grid_size[0] * grid_size[1];
  // int box_dim = 5 + num_of_class;

  int idx = *det_idx;
  int hit = 0, hit2 = 0;
  ;
  for (int i = 0; i < num_cell; i++) {
    for (int j = 0; j < num_boxes_per_cell; j++) {
      float box_confidence = _sigmoid(
          feature[GET_INDEX(i, j, CONF_INDEX, num_cell, num_of_class)], false);
      if (box_confidence < obj_threshold) {
        continue;
      }
      hit++;
      float box_class_probs[80];
      int box_max_cls;
      float box_max_prob =
          _softmax(box_class_probs,
                   &feature[GET_INDEX(i, j, CLS_INDEX, num_cell, num_of_class)],
                   num_cell, num_of_class, &box_max_cls, false);
      float box_max_score = box_confidence * box_max_prob;
      if (box_max_score < obj_threshold) {
        continue;
      }
      // get coord now
      int grid_x = i % grid_size[1];
      int grid_y = i / grid_size[1];
      float box_x = _sigmoid(
          feature[GET_INDEX(i, j, COORD_X_INDEX, num_cell, num_of_class)],
          false);
      box_x += grid_x;
      box_x /= grid_size[1];
      float box_y = _sigmoid(
          feature[GET_INDEX(i, j, COORD_Y_INDEX, num_cell, num_of_class)],
          false);
      box_y += grid_y;
      box_y /= grid_size[0];
      // anchor is in shape [3][2]
      float box_w =
          exp(feature[GET_INDEX(i, j, COORD_W_INDEX, num_cell, num_of_class)]);
      box_w *= anchor[j * 2];
      box_w /= yolo_w;
      float box_h =
          exp(feature[GET_INDEX(i, j, COORD_H_INDEX, num_cell, num_of_class)]);
      box_h *= anchor[j * 2 + 1];
      box_h /= yolo_h;
      hit2++;
      // DBG("  hit2 %d, conf = %f, cls = %d, coord = [%f, %f, %f, %f]\n",
      //    hit2, box_max_score, box_max_cls, box_x, box_y, box_w, box_h);
      det[idx].bbox = box{box_x, box_y, box_w, box_h};
      det[idx].score = box_max_score;
      det[idx].cls = box_max_cls;
      idx++;
      assert(idx <= MAX_DET);
    }
  }
  *det_idx = idx;
}

static void bbox_transform_inv(const float *boxes, const float *deltas,
                               float *pred, int num, int class_num) {
  for (int i = 0; i < num; ++i) {
    float height = boxes[i * 4 + 3] - boxes[i * 4 + 1] + 1;
    float width = boxes[i * 4 + 2] - boxes[i * 4 + 0] + 1;
    float ctr_x = boxes[i * 4 + 0] + width * 0.5;
    float ctr_y = boxes[i * 4 + 1] + height * 0.5;

    for (int j = 0; j < class_num; ++j) {
      float dx = deltas[i * class_num * 4 + j * 4 + 0];
      float dy = deltas[i * class_num * 4 + j * 4 + 1];
      float dw = deltas[i * class_num * 4 + j * 4 + 2];
      float dh = deltas[i * class_num * 4 + j * 4 + 3];

      float pred_ctr_x = dx * width + ctr_x;
      float pred_ctr_y = dy * height + ctr_y;
      float pred_w = std::exp(dw) * width;
      float pred_h = std::exp(dh) * height;

      pred[i * class_num * 4 + j * 4 + 0] = pred_ctr_x - pred_w / 2;
      pred[i * class_num * 4 + j * 4 + 1] = pred_ctr_y - pred_h / 2;
      pred[i * class_num * 4 + j * 4 + 2] = pred_ctr_x + pred_w / 2;
      pred[i * class_num * 4 + j * 4 + 3] = pred_ctr_y + pred_h / 2;
    }
  }
}

static void nms(detections *dets, int num, float nms_threshold) {
  for (int i = 0; i < num; i++) {
    if (dets[i].score == 0) {
      // erased already
      continue;
    }

    float s1 = (dets[i].bbox.x2 - dets[i].bbox.x1 + 1) *
               (dets[i].bbox.y2 - dets[i].bbox.y1 + 1);
    for (int j = i + 1; j < num; j++) {
      if (dets[j].score == 0) {
        // erased already
        continue;
      }
      if (dets[i].cls != dets[j].cls) {
        // not the same class
        continue;
      }

      float s2 = (dets[j].bbox.x2 - dets[j].bbox.x1 + 1) *
                 (dets[j].bbox.y2 - dets[j].bbox.y1 + 1);

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
namespace mlir {

DetectionOutputOpKernel::DetectionOutputOpKernel(Operation &op,
                                                 value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  auto detection_outputOp = cast<tpu::DetectionOutputOp>(op);

  auto loc_type =
      detection_outputOp.input()[0].getType().template cast<TensorType>();
  this->loc_shape = loc_type.getShape();

  auto conf_type =
      detection_outputOp.input()[1].getType().template cast<TensorType>();
  this->conf_shape = conf_type.getShape();

  auto prior_type =
      detection_outputOp.input()[2].getType().template cast<TensorType>();
  this->prior_shape = prior_type.getShape();

  this->keep_top_k = detection_outputOp.keep_top_k();
  this->confidence_threshold =
      detection_outputOp.confidence_threshold().convertToFloat();
  this->nms_threshold = detection_outputOp.nms_threshold().convertToFloat();
  this->top_k = detection_outputOp.top_k();
  this->num_classes = detection_outputOp.num_classes();
  this->share_location = detection_outputOp.share_location();
  this->background_label_id = detection_outputOp.background_label_id();
  if (detection_outputOp.code_type() == "CORNER") {
    this->code_type = PriorBoxParameter_CodeType_CORNER;
  } else if (detection_outputOp.code_type() == "CENTER_SIZE") {
    this->code_type = PriorBoxParameter_CodeType_CENTER_SIZE;
  } else if (detection_outputOp.code_type() == "CORNER_SIZE") {
    this->code_type = PriorBoxParameter_CodeType_CORNER_SIZE;
  } else {
    llvm_unreachable("code type wrong");
  }
  // get tensors
  loc_data = this->opdTensors[0];
  conf_data = this->opdTensors[1];
  prior_data = this->opdTensors[2];

  output_data = this->resTensor;
}

void DetectionOutputOpKernel::set_tensor(const std::vector<float> &data) {
  llvm_unreachable("TODO");
};

std::vector<float> DetectionOutputOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void DetectionOutputOpKernel::invoke() {
  int num = loc_shape[0];
  int num_priors = prior_shape[2] / 4;
  int num_loc_classes = share_location ? 1 : num_classes;
  float eta = 1.0;
  bool variance_encoded_in_target = false;
  std::vector<std::map<int, std::vector<std::pair<float, int>>>>
      all_conf_scores;
  GetConfidenceScores_opt(conf_data->data(), num, num_priors, num_classes,
                          confidence_threshold, &all_conf_scores);
  for (int i = 0; i < num; ++i) {
    for (int c = 0; c < num_classes; ++c) {
      if (all_conf_scores[i].find(c) == all_conf_scores[i].end()) {
        LLVM_DEBUG(std::cout << "class with no score idx = %d," << c << "\n";);
        continue;
      }
      std::vector<std::pair<float, int>> &scores =
          all_conf_scores[i].find(c)->second;

      if (top_k < (int)scores.size()) {
        std::partial_sort(scores.begin(), scores.begin() + top_k, scores.end(),
                          SortScoreCmp0);
      } else {
        std::sort(scores.begin(), scores.end(), SortScoreCmp0);
      }
    }
  }

  // build keep for decode ,recode vilad index
  float *decode_keep_index;
  int buf_length = 0;
  if (share_location) {
    buf_length = num * num_priors;
  } else {
    buf_length = num * num_priors * num_classes;
  }
  decode_keep_index = new float[buf_length];
  memset(decode_keep_index, 0, buf_length * 4);
  float *p = decode_keep_index;
  for (int i = 0; i < num; ++i) {
    if (share_location) {
      p = decode_keep_index + num_priors * i;
    }
    for (int c = 0; c < num_classes; ++c) {
      if (!share_location) {
        p = decode_keep_index + num_priors * num_classes * i + num_priors * c;
      }
      if (c == background_label_id) {
        // Ignore background class.
        continue;
      }

      if (all_conf_scores[i].find(c) == all_conf_scores[i].end())
        continue;
      std::vector<std::pair<float, int>> &scores =
          all_conf_scores[i].find(c)->second;
      int length = top_k < (int)scores.size() ? top_k : scores.size();
      for (int k = 0; k < length; ++k) {
        p[scores[k].second] = 1;
      }
    }
  }

  // Retrieve all location predictions.
  std::vector<LabelBBox_l> all_loc_preds;
  GetLocPredictions_opt(loc_data->data(), num, num_priors, num_loc_classes,
                        share_location, decode_keep_index, &all_loc_preds);

  // Decode all loc predictions to bboxes.
  std::vector<LabelBBox_l> all_decode_bboxes;
  const bool clip_bbox = false;
  DecodeBBoxesAll_opt(all_loc_preds, num_priors, prior_data->data(), num,
                      share_location, num_loc_classes, background_label_id,
                      code_type, variance_encoded_in_target, clip_bbox,
                      decode_keep_index, &all_decode_bboxes);
  delete[] decode_keep_index;

  int num_kept = 0;
  std::vector<std::map<int, std::vector<std::pair<float, int>>>> all_indices;
  for (int i = 0; i < num; ++i) {
    const LabelBBox_l &decode_bboxes = all_decode_bboxes[i];
    const std::map<int, std::vector<std::pair<float, int>>> &conf_scores =
        all_conf_scores[i];
    std::map<int, std::vector<std::pair<float, int>>> indices;
    int num_det = 0;
    for (int c = 0; c < num_classes; ++c) {
      if (c == background_label_id) {
        // Ignore background class.
        continue;
      }
      if (conf_scores.find(c) == conf_scores.end())
        continue;
      int label = share_location ? -1 : c;
      if (decode_bboxes.find(label) == decode_bboxes.end()) {
        // Something bad happened if there are no predictions for current label.
        llvm::errs() << "Could not find location predictions for label "
                     << label;
        continue;
      }
      const std::vector<BBox_l> &bboxes = decode_bboxes.find(label)->second;
      const std::vector<std::pair<float, int>> &aa =
          conf_scores.find(c)->second;
      ApplyNMSFast_opt(bboxes, aa, confidence_threshold, nms_threshold, eta,
                       top_k, &(indices[c]));

      num_det += indices[c].size();
    }

    if (keep_top_k > -1 && num_det > keep_top_k) {
      std::vector<std::pair<float, std::pair<int, int>>> score_index_pairs;
      for (auto it = indices.begin(); it != indices.end(); ++it) {
        int label = it->first;

        const std::vector<std::pair<float, int>> &label_indices = it->second;
        for (int j = 0; j < (int)label_indices.size(); ++j) {
          score_index_pairs.push_back(
              std::make_pair(label_indices[j].first,
                             std::make_pair(label, label_indices[j].second)));
        }
      }
      // Keep top k results per image.
      std::sort(score_index_pairs.begin(), score_index_pairs.end(),
                SortScoreCmp1);
      score_index_pairs.resize(keep_top_k);
      // Store the new indices.
      std::map<int, std::vector<std::pair<float, int>>> new_indices;
      for (int j = 0; j < (int)score_index_pairs.size(); ++j) {

        int label = score_index_pairs[j].second.first;
        int idx = score_index_pairs[j].second.second;
        float s = score_index_pairs[j].first;

        new_indices[label].push_back(std::make_pair(s, idx));
      }
      all_indices.push_back(new_indices);
      num_kept += keep_top_k;
    } else {
      all_indices.push_back(indices);
      num_kept += num_det;
    }
  }
  // float *top_data = (float *)opdT[0]->data();

  float *top_data = (float *)output_data->data();

  int output_size = num * keep_top_k * 1 * 1 * 7;
  // init output buf
  for (int i = 0; i < output_size; ++i) {
    top_data[i] = -1;
  }

  if (num_kept == 0) {
    LLVM_DEBUG(llvm::errs() << "Couldn't find any detections";);
    // Generate fake results per image.
    for (int i = 0; i < num; ++i) {
      top_data[0] = i;
      top_data += 7;
    }
  } else {
    int count = 0;
    for (int i = 0; i < num; ++i) {
      const LabelBBox_l &decode_bboxes = all_decode_bboxes[i];
      for (auto it = all_indices[i].begin(); it != all_indices[i].end(); ++it) {
        int label = it->first;
        int loc_label = share_location ? -1 : label;
        if (decode_bboxes.find(loc_label) == decode_bboxes.end()) {
          // Something bad happened if there are no predictions for current
          // label.
          llvm::errs() << "Could not find location predictions for "
                       << loc_label;
          continue;
        }
        const std::vector<BBox_l> &bboxes =
            decode_bboxes.find(loc_label)->second;
        std::vector<std::pair<float, int>> &indices = it->second;
        for (int j = 0; j < (int)indices.size(); ++j) {

          int idx = indices[j].second;
          top_data[count * 7] = i;
          top_data[count * 7 + 1] = label;
          top_data[count * 7 + 2] = indices[j].first;
          const BBox_l &bbox = bboxes[idx];
          top_data[count * 7 + 3] = bbox.xmin;
          top_data[count * 7 + 4] = bbox.ymin;
          top_data[count * 7 + 5] = bbox.xmax;
          top_data[count * 7 + 6] = bbox.ymax;
          ++count;
        }
      }
    }
  }
}

void DetectionOutputOpKernel::dump() { OpKernel::dump(); }

YoloDetectionOpKernel::YoloDetectionOpKernel(Operation &op,
                                             value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  auto yoOp = cast<tpu::YoloDetectionOp>(op);
  this->net_input_h = yoOp.net_input_h();
  this->net_input_w = yoOp.net_input_w();
  this->obj_threshold = yoOp.obj_threshold().convertToFloat();
  this->nms_threshold = yoOp.nms_threshold().convertToFloat();
  this->keep_topk = yoOp.keep_topk();
  this->tiny = yoOp.tiny();
  this->yolo_v4 = yoOp.yolo_v4();
  this->class_num = yoOp.class_num();

  std::string str_anchors = yoOp.anchors().str();
  std::stringstream iss(str_anchors);
  std::string s;
  while (std::getline(iss, s, ',')) {
    this->vec_anchors.push_back(atof(s.c_str()));
  }
  this->input_count = this->opdTensors.size();
  for (int i = 0; i < input_count; i++) {
    inputs_shape.push_back(getTensorShape(op.getOperand(i)));
  }

  if (tiny) {
    if (this->vec_anchors.size() == 0) {
      this->vec_anchors = {
          10, 14, 23,  27,  37,  58, // layer23-conv (26*26)
          81, 82, 135, 169, 344, 319 // layer16-conv (13*13)
      };
    }
  } else {
    if (this->vec_anchors.size() == 0) {
      if (yolo_v4) {
        this->vec_anchors = {
            142, 110, 192, 243, 459, 401, // layer161-conv
            36,  75,  76,  55,  72,  146, // layer150-conv
            12,  16,  19,  36,  40,  28,  // layer139-conv
        };
      } else {
        this->vec_anchors = {
            10,  13, 16,  30,  33,  23,  // layer106-conv (52*52)
            30,  61, 62,  45,  59,  119, // layer94-conv  (26*26)
            116, 90, 156, 198, 373, 326  // layer82-conv  (13*13)
        };
      }
    }
  }

  // get tensors
  inputs_data = this->opdTensors;
  output_data = this->resTensor;
}

void YoloDetectionOpKernel::set_tensor(const std::vector<float> &data) {
  llvm_unreachable("TODO");
};

std::vector<float> YoloDetectionOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void YoloDetectionOpKernel::invoke() {
  float anchors[input_count][6];
  assert((int)vec_anchors.size() == input_count * 6);

  for (int i = 0; i < input_count; ++i) {
    for (int j = 0; j < 6; ++j) {
      anchors[i][j] = vec_anchors[i * 6 + j];
    }
  }
  int batch = this->shape.at(0);
  for (int b = 0; b < batch; ++b) {
    std::vector<std::vector<int>> grid_size;
    std::vector<std::vector<float>> features;

    for (int i = 0; i < input_count; ++i) {
      auto shape = inputs_shape[i];
      grid_size.push_back(std::vector<int>{(int)shape[2], (int)shape[3]});
      auto data = inputs_data[i]->data() + b * shape[1] * shape[2] * shape[3];
      auto size = inputs_data[i]->size() / batch;
      std::vector<float> bottom_data(data, data + size);
      features.push_back(bottom_data);
    }

    detection det_raw[MAX_DET_RAW];
    detection dets[MAX_DET];
    int det_raw_idx = 0;
    for (size_t i = 0; i < features.size(); i++) {
      process_feature(det_raw, &det_raw_idx, features[i].data(), grid_size[i],
                      &anchors[i][0], {net_input_h, net_input_w}, class_num,
                      obj_threshold);
    }
    nms(det_raw, det_raw_idx, nms_threshold);
    int det_idx = 0;
    for (int i = 0; i < det_raw_idx; i++) {
      if (det_raw[i].score > 0) {
        dets[det_idx] = det_raw[i];
        det_idx++;
      } else {
        // std::cout << "erased: " << det_raw[i].cls << std::endl;
      }
    }

    if (keep_topk > det_idx)
      keep_topk = det_idx;

    long long count = 0;
    float *batched_output_data =
        output_data->data() + b * shape[1] * shape[2] * shape[3];
    for (int i = 0; i < (int)keep_topk; ++i) {
      batched_output_data[count++] = dets[i].bbox.x;
      batched_output_data[count++] = dets[i].bbox.y;
      batched_output_data[count++] = dets[i].bbox.w;
      batched_output_data[count++] = dets[i].bbox.h;
      batched_output_data[count++] = dets[i].cls;
      batched_output_data[count++] = dets[i].score;

      LLVM_DEBUG(llvm::errs()
                     << "x= " << dets[i].bbox.x << ",y= " << dets[i].bbox.y
                     << ",w= " << dets[i].bbox.w << ",h= " << dets[i].bbox.h
                     << ", class= " << dets[i].cls
                     << ", score= " << dets[i].score << "\n";);
    }
  }
}
void YoloDetectionOpKernel::dump() { OpKernel::dump(); }

FrcnDetectionOpKernel::FrcnDetectionOpKernel(Operation &op,
                                             value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {
  auto frcndOp = cast<tpu::FrcnDetectionOp>(op);
  this->rois_shape = op.getOperand(2).getType().cast<TensorType>().getShape();
  this->class_num = frcndOp.class_num();
  this->keep_topk = frcndOp.keep_topk();
  this->nms_threshold = frcndOp.nms_threshold().convertToFloat();
  this->obj_threshold = frcndOp.obj_threshold().convertToFloat();

  // get tensors
  bbox_deltas = this->opdTensors[0];
  scores = this->opdTensors[1];
  rois = this->opdTensors[2];
  output_data = this->resTensor;
}

void FrcnDetectionOpKernel::set_tensor(const std::vector<float> &data) {
  llvm_unreachable("TODO");
};

std::vector<float> FrcnDetectionOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void FrcnDetectionOpKernel::invoke() {
  int batch = rois_shape[0];
  int num = rois_shape[2];

  for (int b = 0; b < batch; ++b) {
    auto batched_bbox_deltas = bbox_deltas->data() + b * num * class_num * 4;
    auto batched_scores = scores->data() + b * num * class_num;
    auto batched_rois = rois->data() + b * num * 5;

    std::vector<float> boxes(num * 4, 0);
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < 4; ++j) {
        boxes[i * 4 + j] = batched_rois[i * 5 + j + 1];
      }
    }

    std::vector<float> pred(num * class_num * 4, 0);
    float *pred_data = pred.data();
    std::vector<float> deltas(batched_bbox_deltas,
                              batched_bbox_deltas + num * class_num * 4);
    bbox_transform_inv(boxes.data(), deltas.data(), pred_data, num, class_num);

    int det_num = 0;
    auto dets = new detections[num];

    for (int i = 0; i < num; ++i) {
      for (int j = 1; j < (int)class_num; ++j) {
        if (batched_scores[i * class_num + j] > obj_threshold) {
          dets[det_num].bbox.x1 = pred[i * class_num * 4 + j * 4 + 0];
          dets[det_num].bbox.y1 = pred[i * class_num * 4 + j * 4 + 1];
          dets[det_num].bbox.x2 = pred[i * class_num * 4 + j * 4 + 2];
          dets[det_num].bbox.y2 = pred[i * class_num * 4 + j * 4 + 3];
          dets[det_num].cls = j;
          dets[det_num].score = batched_scores[i * class_num + j];
          det_num++;
        }
      }
    }

    nms(dets, det_num, nms_threshold);
    auto dets_nms = new detections[det_num];
    int det_idx = 0;
    for (int i = 0; i < det_num; i++) {
      if (dets[i].score > 0) {
        dets_nms[det_idx] = dets[i];
        det_idx++;
      }
    }

    if (keep_topk > det_idx)
      keep_topk = det_idx;

    long long count = 0;
    auto batched_output =
        output_data->data() + b * shape[1] * shape[2] * shape[3];
    for (int i = 0; i < keep_topk; ++i) {
      batched_output[count++] = dets_nms[i].bbox.x1;
      batched_output[count++] = dets_nms[i].bbox.y1;
      batched_output[count++] = dets_nms[i].bbox.x2;
      batched_output[count++] = dets_nms[i].bbox.y2;
      batched_output[count++] = dets_nms[i].cls;
      batched_output[count++] = dets_nms[i].score;
      // printf("x1: %f, y1: %f, x2: %f, y2: %f, cls: %d, score: %f\n",
      //     dets_nms[i].bbox.x1, dets_nms[i].bbox.y1, dets_nms[i].bbox.x2,
      //     dets_nms[i].bbox.y2, dets_nms[i].cls, dets_nms[i].score);
    }
    delete[] dets_nms;
    delete[] dets;
  }
}
void FrcnDetectionOpKernel::dump() { OpKernel::dump(); }
} // namespace mlir