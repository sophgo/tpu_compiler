#include "tpuc/Interpreter/cpu/broadcast.hpp"
#include "internal.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/MlirModuleInterpreter.h"
#include "tpuc/TPUTensorSupport.h"

namespace mlir {

BroadcastOpKernel::BroadcastOpKernel(Operation &op, value_map_t &valueMapping,
                                     weight_map_t &weightMapping)
    : CPUOpKernel(op, valueMapping, weightMapping) {

  auto input_shape0 = getTensorShape(op.getOperand(0));
  auto input_shape1 = getTensorShape(op.getOperand(1));
  bool align_right = true;
  if (op.hasAttr("align_right")) {
    align_right = op.getAttr("align_right").cast<BoolAttr>().getValue();
  }

  getNCHW(input_shape0, n0, c0, h0, w0, align_right);
  getNCHW(input_shape1, n1, c1, h1, w1, align_right);

  do_relu = false;
  if (op.hasAttr("do_relu")) {
    do_relu = op.getAttr("do_relu").cast<BoolAttr>().getValue();
  }

  on = std::max(n0, n1);
  oc = std::max(c0, c1);
  oh = std::max(h0, h1);
  ow = std::max(w0, w1);

  if (datatype == DataType::INT8) {
    auto quant_rshift = this->opdTensors[4];
    auto quant_multiplier = this->opdTensors[5];
    assert(quant_rshift && quant_multiplier);
    rshift = quant_rshift->at(0);
    multiplier.assign(quant_multiplier->begin(), quant_multiplier->end());
  }
  // get tensors
  inputs_data = this->opdTensors;
  output_data = this->resTensor;
}

void BroadcastAddOpKernel::invoke() {
  std::vector<std::vector<float>> inputs(2);
  for (size_t i = 0; i < 2; ++i) {
    inputs[i].assign(inputs_data[i]->begin(), inputs_data[i]->end());
  }
  if (datatype == DataType::INT8) {
    for (size_t i = 0; i < 2; ++i) {
      for (size_t j = 0; j < inputs[i].size(); ++j) {
        inputs[i][j] *= (int8_t)multiplier.at(i);
      }
    }
  }
  int idx = 0;
  for (int n = 0; n < on; ++n) {
    for (int c = 0; c < oc; ++c) {
      for (int h = 0; h < oh; ++h) {
        for (int w = 0; w < ow; ++w) {
          int idx0 = (n0 == 1 ? 0 : n) * c0 * h0 * w0 +
                     (c0 == 1 ? 0 : c) * h0 * w0 + (h0 == 1 ? 0 : h) * w0 +
                     (w0 == 1 ? 0 : w);
          int idx1 = (n1 == 1 ? 0 : n) * c1 * h1 * w1 +
                     (c1 == 1 ? 0 : c) * h1 * w1 + (h1 == 1 ? 0 : h) * w1 +
                     (w1 == 1 ? 0 : w);
          output_data->at(idx++) = inputs[0][idx0] + inputs[1][idx1];
        }
      }
    }
  }
  if (do_relu) {
    relu(output_data->data(), output_data->data(), output_data->size());
  }
  if (datatype == DataType::INT8) {
    for (size_t i = 0; i < output_data->size(); ++i) {
      output_data->at(i) = (float)applyRShiftAndSaturateInt8(output_data->at(i),
                                                             (uint32_t)rshift);
    }
  } else if (datatype == DataType::BF16) {
    BF16(output_data->data(), output_data->data(), output_data->size());
  }
}

void BroadcastMulOpKernel::invoke() {
  int idx = 0;
  for (int n = 0; n < on; ++n) {
    for (int c = 0; c < oc; ++c) {
      for (int h = 0; h < oh; ++h) {
        for (int w = 0; w < ow; ++w) {
          int idx0 = (n0 == 1 ? 0 : n) * c0 * h0 * w0 +
                     (c0 == 1 ? 0 : c) * h0 * w0 + (h0 == 1 ? 0 : h) * w0 +
                     (w0 == 1 ? 0 : w);
          int idx1 = (n1 == 1 ? 0 : n) * c1 * h1 * w1 +
                     (c1 == 1 ? 0 : c) * h1 * w1 + (h1 == 1 ? 0 : h) * w1 +
                     (w1 == 1 ? 0 : w);
          output_data->at(idx++) =
              inputs_data[0]->at(idx0) * inputs_data[1]->at(idx1);
        }
      }
    }
  }
  if (do_relu) {
    relu(output_data->data(), output_data->data(), output_data->size());
  }
  if (datatype == DataType::INT8) {
    for (size_t i = 0; i < output_data->size(); ++i) {
      output_data->at(i) = (float)applyMultiplierAndRShiftAndSaturateInt8(
          output_data->at(i), (uint32_t)rshift, (uint32_t)multiplier[0], true);
    }
  } else if (datatype == DataType::BF16) {
    BF16(output_data->data(), output_data->data(), output_data->size());
  }
}

void BroadcastSubOpKernel::invoke() {
  std::vector<std::vector<float>> inputs(2);
  for (size_t i = 0; i < 2; ++i) {
    inputs[i].assign(inputs_data[i]->begin(), inputs_data[i]->end());
  }
  if (datatype == DataType::INT8) {
    for (size_t i = 0; i < 2; ++i) {
      for (size_t j = 0; j < inputs[i].size(); ++j) {
        inputs[i][j] *= (int8_t)multiplier.at(i);
      }
    }
  }
  int idx = 0;
  for (int n = 0; n < on; ++n) {
    for (int c = 0; c < oc; ++c) {
      for (int h = 0; h < oh; ++h) {
        for (int w = 0; w < ow; ++w) {
          int idx0 = (n0 == 1 ? 0 : n) * c0 * h0 * w0 +
                     (c0 == 1 ? 0 : c) * h0 * w0 + (h0 == 1 ? 0 : h) * w0 +
                     (w0 == 1 ? 0 : w);
          int idx1 = (n1 == 1 ? 0 : n) * c1 * h1 * w1 +
                     (c1 == 1 ? 0 : c) * h1 * w1 + (h1 == 1 ? 0 : h) * w1 +
                     (w1 == 1 ? 0 : w);
          output_data->at(idx++) = inputs[0][idx0] - inputs[1][idx1];
        }
      }
    }
  }
  if (do_relu) {
    relu(output_data->data(), output_data->data(), output_data->size());
  }
  if (datatype == DataType::INT8) {
    for (size_t i = 0; i < output_data->size(); ++i) {
      output_data->at(i) = (float)applyRShiftAndSaturateInt8(output_data->at(i),
                                                             (uint32_t)rshift);
    }
  } else if (datatype == DataType::BF16) {
    BF16(output_data->data(), output_data->data(), output_data->size());
  }
}

} // namespace mlir