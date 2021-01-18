#include "tpuc/Interpreter/cpu/preprocess.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Interpreter/cpu/crop.hpp"
#include "tpuc/Interpreter/cpu/permute.hpp"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {
void preprocess(float *input, float *output, const std::vector<int64_t> &shape,
                const std::vector<int> &channel_order,
                const std::vector<float> &mean, const std::vector<float> &std,
                float raw_scale, float input_scale) {
  int n = shape[0];
  int c = shape[1];
  int h = shape[2];
  int w = shape[3];
  int csz = h * w;
  int isz = c * h * w;
  int count = n * c * h * w;
  float *p = input;
  float *q = output;

  if (channel_order.size()) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < c; j++) {
        memcpy(q + channel_order[j] * csz, p + j * csz, csz * sizeof(float));
      }
      p += isz;
      q += isz;
    }
    p = q = output;
  }

  for (int i = 0; i < count; i++) {
    float val = *p++;
    if (raw_scale != 0) {
      val *= (raw_scale / 255);
    }
    if (mean.size()) {
      val -= mean[(i / csz) % c];
    }
    if (std.size()) {
      val /= std[(i / csz) % c];
    }
    if (input_scale != 0) {
      val *= input_scale;
    }
    *q++ = val;
  }
}

PreprocessOpKernel::PreprocessOpKernel(Operation &op,
                                       value_map_t &valueMapping) {
  auto preprocessOp = cast<tpu::PreprocessOp>(op);
  assert(preprocessOp);
  llvm::outs() << " PreprocessOp op: [" << preprocessOp.name() << "]\n";

  auto opTensors = getOperandTensors(&op, valueMapping);
  auto result = preprocessOp.getResult();
  auto size = getTensorSize(result);
  auto resultTensor = std::make_shared<std::vector<float>>(size);
  llvm::outs() << "    =>required memory size: [" << size << "]\n";
  auto type = result.getType().cast<TensorType>();
  this->shape = type.getShape();

  auto input_type = preprocessOp.input().getType().template cast<TensorType>();
  this->input_shape = input_type.getShape();

  for (auto o : llvm::enumerate(preprocessOp.color_order().getValue())) {
    auto attr = o.value().dyn_cast<IntegerAttr>();
    color_orders.push_back(attr.getInt());
  }

  for (auto m : llvm::enumerate(preprocessOp.transpose_order().getValue())) {
    auto attr = m.value().dyn_cast<IntegerAttr>();
    transpose_orders.push_back(attr.getInt());
  }

  for (auto m : llvm::enumerate(preprocessOp.mean().getValue())) {
    auto attr = m.value().dyn_cast<FloatAttr>();
    means.push_back((float)attr.getValueAsDouble());
  }

  for (auto s : llvm::enumerate(preprocessOp.std().getValue())) {
    auto attr = s.value().dyn_cast<FloatAttr>();
    stds.push_back((float)attr.getValueAsDouble());
  }

  for (auto m : llvm::enumerate(preprocessOp.crop_offset().getValue())) {
    auto attr = m.value().dyn_cast<IntegerAttr>();
    crop_offset.push_back(attr.getInt());
  }
  this->raw_scale = preprocessOp.raw_scale().convertToFloat();
  this->input_scale = preprocessOp.scale().convertToFloat();

  this->name = preprocessOp.name().str();
  this->op_type = op.getName().getStringRef().str();
  set_datatype(getOpQuant(&op).str());
  // get tensors
  input_data = opTensors[0];
  output_data = resultTensor;
  // record mapping table for next op connecting
  valueMapping[result] = std::move(resultTensor);
}
void PreprocessOpKernel::set_tensor(const std::vector<float> &data) {
  if (data.size() != this->input_data->capacity()) {
    llvm::errs() << " PreprocessOp op: [" << this->name
                 << "] required memsize :" << this->input_data->capacity()
                 << "\n";
    llvm::errs() << " input data size: " << data.size() << "\n";
    llvm_unreachable(" size not same!");
  }
  this->input_data->assign(data.begin(), data.end());
};

std::vector<float> PreprocessOpKernel::get_tensor() {
  // deep copy
  std::vector<float> ret(this->output_data->begin(), this->output_data->end());
  return ret;
}

void PreprocessOpKernel::invoke() {

  std::vector<float> tmp_data(input_data->begin(), input_data->end());
  std::vector<int64_t> tmp_shape(input_shape.begin(), input_shape.end());

  if (transpose_orders != std::vector<unsigned int>{0, 1, 2, 3}) {
    int64_t t_on, t_oc, t_oh, t_ow;
    t_on = input_shape.at(transpose_orders.at(0));
    t_oc = input_shape.at(transpose_orders.at(1));
    t_oh = input_shape.at(transpose_orders.at(2));
    t_ow = input_shape.at(transpose_orders.at(3));

    permute(tmp_data.data(), tmp_data.data(), tmp_shape, transpose_orders);
    tmp_shape = {t_on, t_oc, t_oh, t_ow};
  }
  if (tmp_data.size() > output_data->size()) {
    std::vector<float> crop_data(output_data->size());
    std::vector<int> indices(tmp_shape.size(), 0);
    std::vector<int64_t> crop_shape(shape.begin(), shape.end());
    crop(tmp_data.data(), crop_data.data(), tmp_shape.data(), crop_shape.data(),
         0, crop_offset.data(), indices.data());
    tmp_data.assign(crop_data.begin(), crop_data.end());
  }

  preprocess(tmp_data.data(), output_data->data(), shape, color_orders,
             means, stds, raw_scale, input_scale);
}

void PreprocessOpKernel::dump() {
  OpKernel::dump();
  std::string color_order_str, transpose_order_str, mean_str, std_str,
      crop_offset_str;
  for (auto &i : this->color_orders) {
    color_order_str = color_order_str + std::to_string(i) + " ";
  }
  for (auto &i : this->transpose_orders) {
    transpose_order_str = transpose_order_str + std::to_string(i) + " ";
  }
  for (auto &i : this->means) {
    mean_str = mean_str + std::to_string(i) + " ";
  }
  for (auto &i : this->stds) {
    std_str = std_str + std::to_string(i) + " ";
  }
  for (auto &i : this->crop_offset) {
    crop_offset_str = crop_offset_str + std::to_string(i) + " ";
  }
  llvm::outs() << "\tColor Order: " << color_order_str << "\n";
  llvm::outs() << "\tTranspose Order: " << transpose_order_str << "\n";
  llvm::outs() << "\tMean: " << mean_str << "\n";
  llvm::outs() << "\tStd: " << std_str << "\n";
  llvm::outs() << "\tCrop Offset: " << crop_offset_str << "\n";
  llvm::outs() << "\tRaw scale: " << std::to_string(raw_scale) << "\n";
  llvm::outs() << "\tInput scale: " << std::to_string(input_scale) << "\n";
}
} // namespace mlir