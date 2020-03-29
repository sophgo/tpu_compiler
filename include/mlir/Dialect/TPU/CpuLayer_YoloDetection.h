#ifndef CPULAYER_YOLO_DETECTION_H
#define CPULAYER_YOLO_DETECTION_H

#define MAX_DET 200
#define MAX_DET_RAW 500

typedef struct box_ {
    float x, y, w, h;
} box;

typedef struct detection_ {
    box bbox;
    int cls;
    float score;
} detection;

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
  float x[80];
  float max_x = -INFINITY;
  float min_x = INFINITY;
  for (int i = 0; i < 80; i++) {
    x[i] = data[i * input_stride];
    if (x[i] > max_x) {
      max_x = x[i];
    }
    if (x[i] < min_x) {
      min_x = x[i];
    }
  }
  #define t (-100.0f)
  float exp_x[80];
  float sum = 0;
  for (int i = 0; i < 80; i++) {
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
  for (int i = 0; i < 80; i++) {
    probs[i] =exp_x[i] / sum;
    if (probs[i] > max_prob) {
      max_prob = probs[i];
      *max_cls = i;
    }
  }
  return max_prob;
}

// feature in shape [3][5+80][grid_size][grid_size]
#define GET_INDEX(cell_idx, box_idx_in_cell, data_idx, num_cell) \
    (box_idx_in_cell * 85 * num_cell + data_idx * num_cell + cell_idx)

static void process_feature(detection *det, int *det_idx, float *feature,
    std::vector<int> grid_size, const float* anchor,
    std::vector<int> yolo_size, int num_of_class, float obj_threshold) {
  int yolo_w = yolo_size[1];
  int yolo_h = yolo_size[0];
  std::cout << "grid_h: " <<  grid_size[0] << std::endl;
  std::cout << "grid_w: " <<  grid_size[1] << std::endl;
  std::cout << "obj_threshold: " << obj_threshold << std::endl;
  int num_boxes_per_cell = 3;
  assert(num_of_class == 80);

  // 255 = 3 * (5 + 80)
  // feature in shape [3][5+80][grid_size][grid_size]
  #define COORD_X_INDEX (0)
  #define COORD_Y_INDEX (1)
  #define COORD_W_INDEX (2)
  #define COORD_H_INDEX (3)
  #define CONF_INDEX    (4)
  #define CLS_INDEX     (5)
  int num_cell = grid_size[0] * grid_size[1];
  //int box_dim = 5 + num_of_class;

  int idx = *det_idx;
  int hit = 0, hit2 = 0;;
  for (int i = 0; i < num_cell; i++) {
    for (int j = 0; j < num_boxes_per_cell; j++) {
      float box_confidence = _sigmoid(feature[GET_INDEX(i, j, CONF_INDEX, num_cell)], false);
      if (box_confidence < obj_threshold) {
        continue;
      }
      hit ++;
      float box_class_probs[80];
      int box_max_cls;
      float box_max_prob = _softmax(box_class_probs,
              &feature[GET_INDEX(i, j, CLS_INDEX, num_cell)],
              num_cell, num_of_class, &box_max_cls, false);
      float box_max_score = box_confidence * box_max_prob;
      if (box_max_score < obj_threshold) {
        continue;
      }
      // get coord now
      int grid_x = i % grid_size[1];
      int grid_y = i / grid_size[1];
      float box_x = _sigmoid(feature[GET_INDEX(i, j, COORD_X_INDEX, num_cell)], false);
      box_x += grid_x;
      box_x /= grid_size[1];
      float box_y = _sigmoid(feature[GET_INDEX(i, j, COORD_Y_INDEX, num_cell)], false);
      box_y += grid_y;
      box_y /= grid_size[0];
      // anchor is in shape [3][2]
      float box_w = exp(feature[GET_INDEX(i, j, COORD_W_INDEX, num_cell)]);
      box_w *= anchor[j*2];
      box_w /= yolo_w;
      float box_h = exp(feature[GET_INDEX(i, j, COORD_H_INDEX, num_cell)]);
      box_h *= anchor[j*2 + 1];
      box_h /= yolo_h;
      hit2 ++;
      //DBG("  hit2 %d, conf = %f, cls = %d, coord = [%f, %f, %f, %f]\n",
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

// https://github.com/ChenYingpeng/caffe-yolov3/blob/master/box.cpp
static float overlap(float x1, float w1, float x2, float w2) {
  float l1 = x1 - w1/2;
  float l2 = x2 - w2/2;
  float left = l1 > l2 ? l1 : l2;
  float r1 = x1 + w1/2;
  float r2 = x2 + w2/2;
  float right = r1 < r2 ? r1 : r2;
  return right - left;
}

static float box_intersection(box a, box b) {
  float w = overlap(a.x, a.w, b.x, b.w);
  float h = overlap(a.y, a.h, b.y, b.h);
  if(w < 0 || h < 0) return 0;
  float area = w*h;
  return area;
}

static float box_union(box a, box b) {
  float i = box_intersection(a, b);
  float u = a.w*a.h + b.w*b.h - i;
  return u;
}

//
// more aboud iou
//   https://github.com/ultralytics/yolov3/blob/master/utils/utils.py
// IoU = inter / (a + b - inter), can't handle enclosure issue
// GIoU, DIoU, CIoU?
//
static float box_iou(box a, box b) {
  return box_intersection(a, b)/box_union(a, b);
}

static void nms(detection *det, int num, float nms_threshold) {
  for(int i = 0; i < num; i++) {
    if (det[i].score == 0) {
      // erased already
      continue;
    }
    for(int j = i + 1; j < num; j++) {
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

#endif
