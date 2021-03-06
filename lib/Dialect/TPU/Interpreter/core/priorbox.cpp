#include "tpuc/Interpreter/cpu/priorbox.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Interpreter/cpu/crop.hpp"
#include "tpuc/Interpreter/cpu/permute.hpp"
#include "tpuc/MlirModuleInterpreter.h"

namespace mlir {

PriorBoxOpKernel::PriorBoxOpKernel(Operation &op, value_map_t &valueMapping,
                                   weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {
  auto priorboxOp = cast<tpu::PriorBoxOp>(op);
  this->input_shape = op.getOperand(0).getType().cast<TensorType>().getShape();

  this->input_image_shape =
      op.getOperand(1).getType().cast<TensorType>().getShape();

  this->clip = priorboxOp.clip();
  this->use_default_aspect_ratio = priorboxOp.use_default_aspect_ratio();
  this->offset = priorboxOp.offset().convertToFloat();
  this->step_w = priorboxOp.step_w().convertToFloat();
  this->step_h = priorboxOp.step_h().convertToFloat();
  this->num_priors = priorboxOp.num_priors();
  this->img_width = priorboxOp.img_w();
  this->img_height = priorboxOp.img_h();
  arrayAttrToVector(priorboxOp.min_size(), this->min_size);
  arrayAttrToVector(priorboxOp.max_size(), this->max_size);
  arrayAttrToVector(priorboxOp.aspect_ratios(), this->aspect_ratios);
  arrayAttrToVector(priorboxOp.variance(), this->variance);

  output_data = this->resTensor;
}

void PriorBoxOpKernel::invoke() {
  assert(input_image_shape.size() == 4 && input_shape.size() == 4);

  int layer_height = input_shape[2];
  int layer_width = input_shape[3];

  if (img_width == 0 || img_height == 0) {
    img_height = input_image_shape[2];
    img_width = input_image_shape[3];
  }

  if (step_w == 0 || step_h == 0) {
    step_w = static_cast<float>(img_width) / layer_width;
    step_h = static_cast<float>(img_height) / layer_height;
  }

  float *top_data = (float *)output_data->data();

  int dim = layer_height * layer_width * num_priors * 4;
  int idx = 0;
  for (int h = 0; h < layer_height; ++h) {
    for (int w = 0; w < layer_width; ++w) {
      float center_x = (w + offset) * step_w;
      float center_y = (h + offset) * step_h;
      float box_width, box_height;
      for (size_t s = 0; s < min_size.size(); ++s) {
        int min_size_ = min_size[s];
        if (use_default_aspect_ratio) {
          // first prior: aspect_ratio = 1, size = min_size
          box_width = box_height = min_size_;
          // xmin
          top_data[idx++] = (center_x - box_width / 2.) / img_width;
          // ymin
          top_data[idx++] = (center_y - box_height / 2.) / img_height;
          // xmax
          top_data[idx++] = (center_x + box_width / 2.) / img_width;
          // ymax
          top_data[idx++] = (center_y + box_height / 2.) / img_height;
        }

        if (max_size.size() > 0) {
          int max_size_ = max_size[s];
          // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
          box_width = box_height = sqrt(min_size_ * max_size_);
          // xmin
          top_data[idx++] = (center_x - box_width / 2.) / img_width;
          // ymin
          top_data[idx++] = (center_y - box_height / 2.) / img_height;
          // xmax
          top_data[idx++] = (center_x + box_width / 2.) / img_width;
          // ymax
          top_data[idx++] = (center_y + box_height / 2.) / img_height;
        }

        // rest of priors
        for (size_t r = 0; r < aspect_ratios.size(); ++r) {
          float ar = aspect_ratios[r];
          if (fabs(ar - 1.) < 1e-6) {
            continue;
          }
          box_width = min_size_ * sqrt(ar);
          box_height = min_size_ / sqrt(ar);
          // xmin
          top_data[idx++] = (center_x - box_width / 2.) / img_width;
          // ymin
          top_data[idx++] = (center_y - box_height / 2.) / img_height;
          // xmax
          top_data[idx++] = (center_x + box_width / 2.) / img_width;
          // ymax
          top_data[idx++] = (center_y + box_height / 2.) / img_height;
        }
      }
    }
  }
  // clip the prior's coordidate such that it is within [0, 1]
  if (clip) {
    for (int d = 0; d < dim; ++d) {
      top_data[d] = std::min<float>(std::max<float>(top_data[d], 0.), 1.);
    }
  }

  std::vector<int64_t> o_s = shape;

  // set the variance.
  top_data += (o_s[2]);

  int count = 0;
  for (int h = 0; h < layer_height; ++h) {
    for (int w = 0; w < layer_width; ++w) {
      for (int i = 0; i < num_priors; ++i) {
        for (int j = 0; j < 4; ++j) {
          top_data[count] = variance[j];
          ++count;
        }
      }
    }
  }
}

} // namespace mlir