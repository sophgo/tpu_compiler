#include "tpuc/Interpreter/cpu/roi_pooling.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {

ROIPoolingOpKernel::ROIPoolingOpKernel(Operation &op,
                                       value_map_t &valueMapping) {
  auto roi_poolingOp = cast<tpu::ROIPoolingOp>(op);
  assert(roi_poolingOp);
  llvm::outs() << " ROIPoolingOp op: [" << roi_poolingOp.name() << "]\n";

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = roi_poolingOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  llvm::outs() << "    =>required memory size: [" << size << "]\n";
  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

  this->input_shape = op.getOperand(0).getType().cast<TensorType>().getShape();
  this->roi_shape = op.getOperand(1).getType().cast<TensorType>().getShape();
  this->name = roi_poolingOp.name().str();
  this->pooled_h = roi_poolingOp.pooled_h();
  this->pooled_w = roi_poolingOp.pooled_w();
  this->spatial_scale = roi_poolingOp.spatial_scale().convertToFloat();

  this->op_type = op.getName().getStringRef().str();
  set_datatype(getOpQuant(&op).str());
  // get tensors
  input_data = opTensors[0];
  rois = opTensors[1];
  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}
void ROIPoolingOpKernel::set_tensor(const std::vector<float> &data) {
  llvm_unreachable("TODO!");
};

std::vector<float> ROIPoolingOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void ROIPoolingOpKernel::invoke() {
  int batch = input_shape[0];
  int channel = input_shape[1];
  int height = input_shape[2];
  int width = input_shape[3];
  int num_rois = roi_shape[2];

  for (int b = 0; b < batch; ++b) {
    auto batched_rois = rois->data() + b * num_rois * 5;
    auto batched_output =
        output_data->data() + b * num_rois * channel * pooled_h * pooled_w;
    for (int n = 0; n < num_rois; ++n) {
      int roi_batch_ind = batched_rois[0];
      int roi_start_w = std::round(batched_rois[1] * spatial_scale);
      int roi_start_h = std::round(batched_rois[2] * spatial_scale);
      int roi_end_w = std::round(batched_rois[3] * spatial_scale);
      int roi_end_h = std::round(batched_rois[4] * spatial_scale);
      assert(roi_batch_ind < batch);

      int roi_height = std::max(roi_end_h - roi_start_h + 1, 1);
      int roi_width = std::max(roi_end_w - roi_start_w + 1, 1);
      const float bin_size_h =
          static_cast<float>(roi_height) / static_cast<float>(pooled_h);
      const float bin_size_w =
          static_cast<float>(roi_width) / static_cast<float>(pooled_w);

      float *batch_data =
          input_data->data() + roi_batch_ind * channel * height * width;

      for (int c = 0; c < channel; ++c) {
        for (int ph = 0; ph < pooled_h; ++ph) {
          for (int pw = 0; pw < pooled_w; ++pw) {
            // Compute pooling region for this output unit:
            //  start (included) = floor(ph * roi_height / pooled_height_)
            //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
            int hstart = static_cast<int>(
                std::floor(static_cast<float>(ph) * bin_size_h));
            int wstart = static_cast<int>(
                std::floor(static_cast<float>(pw) * bin_size_w));
            int hend = static_cast<int>(
                std::ceil(static_cast<float>(ph + 1) * bin_size_h));
            int wend = static_cast<int>(
                std::ceil(static_cast<float>(pw + 1) * bin_size_w));

            hstart = std::min(std::max(hstart + roi_start_h, 0), height);
            hend = std::min(std::max(hend + roi_start_h, 0), height);
            wstart = std::min(std::max(wstart + roi_start_w, 0), width);
            wend = std::min(std::max(wend + roi_start_w, 0), width);

            bool is_empty = (hend <= hstart) || (wend <= wstart);

            const int pool_index = ph * pooled_w + pw;
            if (is_empty) {
              batched_output[pool_index] = 0;
            }

            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int index = h * width + w;
                if (batch_data[index] > batched_output[pool_index]) {
                  batched_output[pool_index] = batch_data[index];
                }
              }
            }
          }
        }
        batch_data += height * width;
        batched_output += pooled_h * pooled_w;
      }
      batched_rois += 5;
    }
  }
}

void ROIPoolingOpKernel::dump() { OpKernel::dump(); }
} // namespace mlir