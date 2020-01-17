//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TPU_OPERATION_CPUOP_SUPPORT_H_
#define MLIR_DIALECT_TPU_OPERATION_CPUOP_SUPPORT_H_

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

#include <utility>
#include <vector>
#include <map>

namespace mlir {

enum Decode_CodeType {
  PriorBoxParameter_CodeType_CORNER = 1,
  PriorBoxParameter_CodeType_CENTER_SIZE = 2,
  PriorBoxParameter_CodeType_CORNER_SIZE = 3
  };
class BBox_l {
  public:
  float xmin;
  float ymin;
  float xmax;
  float ymax;
  float size;

  void CalcSize() {
    if (xmax < xmin || ymax < ymin) {
      size = 0;
    } else {
      float width = xmax - xmin;
      float height = ymax - ymin;
      size = width * height;
    }
  }
};

typedef Decode_CodeType CodeType;
typedef std::map<int, std::vector<BBox_l> > LabelBBox_l;

void GetConfidenceScores_opt (const float* conf_data, const int num,
      const int num_preds_per_class, const int num_classes, const float score_threshold,
      std::vector<std::map<int, std::vector<std::pair<float ,int>> > >* conf_preds);

void GetLocPredictions_opt (const float* loc_data, const int num,
      const int num_preds_per_class, const int num_loc_classes,
      const bool share_location, float *decode_index,
      std::vector<LabelBBox_l>* loc_preds);

void DecodeBBoxesAll_opt (const std::vector<LabelBBox_l>& all_loc_preds,
    int num_priors, const float* prior_data,
    const int num, const bool share_location,
    const int num_loc_classes, const int background_label_id,
    const CodeType code_type, const bool variance_encoded_in_target,
    const bool clip, float *decode_index ,
    std::vector<LabelBBox_l>* all_decode_bboxes);
void ApplyNMSFast_opt (const std::vector<BBox_l>& bboxes,
    const std::vector<std::pair<float ,int>> & conf_score ,
    const float score_threshold, const float nms_threshold, const float eta, int top_k,
    std::vector<std::pair<float,int>>* indices);
bool SortScoreCmp0 (const std::pair<float,int> &pair1,
    const std::pair<float,int> &pair2);

bool SortScoreCmp1 (const std::pair<float, std::pair<int, int>>& pair1,
    const std::pair<float, std::pair<int, int>>& pair2) ;

}

#endif