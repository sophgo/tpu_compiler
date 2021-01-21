#include "tpuc/Interpreter/cpu/detection_output.hpp"

#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Interpreter/cpu/crop.hpp"
#include "tpuc/Interpreter/cpu/permute.hpp"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {

DetectionOutputOpKernel::DetectionOutputOpKernel(Operation &op,
                                                 value_map_t &valueMapping) {
  auto detection_outputOp = cast<tpu::DetectionOutputOp>(op);
  assert(detection_outputOp);
  llvm::outs() << " DetectionOutputOp op: [" << detection_outputOp.name()
               << "]\n";

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = detection_outputOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  llvm::outs() << "    =>required memory size: [" << size << "]\n";
  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

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
  this->name = detection_outputOp.name().str();
  this->op_type = op.getName().getStringRef().str();
  set_datatype(getOpQuant(&op).str());

  // get tensors
  loc_data = opTensors[0];
  conf_data = opTensors[1];
  prior_data = opTensors[2];

  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
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
} // namespace mlir