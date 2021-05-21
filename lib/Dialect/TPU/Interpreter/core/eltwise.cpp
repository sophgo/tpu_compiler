#include "tpuc/Interpreter/cpu/eltwise.hpp"
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/Interpreter/cpu/activation.hpp"
#include "tpuc/ModuleInterpreter.h"

namespace mlir {
EltwiseAddOpKernel::EltwiseAddOpKernel(Operation &op,
                                       value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {

  auto elt_addOp = cast<tpu::EltwiseAddOp>(op);
  const unsigned nInputs = op.getNumOperands() - 4;
  this->do_quant = getOpQuantParamType(&op) != "NONE";
  this->is_asymmetric = isOpQuantAsymmetric(&op);

  this->do_relu = elt_addOp.do_relu();
  if (elt_addOp.coeff().hasValue()) {
    arrayAttrToVector(elt_addOp.coeff().getValue(), coeff);
  } else {
    coeff.assign(nInputs, 1.0f);
  }
  if (datatype == DataType::INT8) {
    auto quant_rshift = this->opdTensors[nInputs + 2];
    auto quant_multiplier = this->opdTensors[nInputs + 3];
    if (do_quant) {
      assert(quant_rshift);
      assert(quant_multiplier);
      this->rshift.assign(quant_rshift->begin(), quant_rshift->end());
      this->multiplier.assign(quant_multiplier->begin(),
                              quant_multiplier->end());
    }

    if (is_asymmetric) {
      this->inputs_offset.resize(nInputs);
      for (size_t i = 0; i < nInputs; ++i) {
        this->inputs_offset[i] = -getPreviousOpZeroPoint(&op, i);
      }
      this->output_offset = getOpZeroPoint(&op);
    }
  }
  this->opdTensors.erase(this->opdTensors.begin() + nInputs, this->opdTensors.end());
  // get tensors
  inputs_data = this->opdTensors;
  output_data = this->resTensor;
}

void EltwiseAddOpKernel::fp32_invoke() {

  std::fill(output_data->begin(), output_data->end(), 0);
  for (size_t ni = 0; ni < inputs_data.size(); ++ni) {
    for (size_t i = 0; i < output_data->size(); ++i) {
      output_data->at(i) += coeff[ni] * inputs_data[ni]->at(i);
    }
  }
  if (do_relu) {
    relu(output_data->data(), output_data->data(), output_data->size());
  }
}

void EltwiseAddOpKernel::i8_invoke() {
  int in = this->shape.at(0);
  int ic = this->shape.at(1);
  int ih = shape.size() > 2 ? this->shape.at(2) : 1;
  int iw = shape.size() > 3 ? this->shape.at(3) : 1;
  size_t input_number = inputs_data.size();
  size_t size = in * ic * ih * iw;

  std::vector<std::vector<float>> i8_inputs(input_number);
  for (size_t i = 0; i < input_number; ++i) {
    i8_inputs[i].assign(inputs_data[i]->begin(), inputs_data[i]->end());

    if (is_asymmetric) {
      for (auto &data : i8_inputs[i]) {
        data += inputs_offset[i];
      }
    }
  }

  for (size_t i = 0; i < input_number; ++i) {
    for (size_t j = 0; j < size; ++j) {
      i8_inputs[i][j] *= (int8_t)multiplier.at(i);
    }
  }

  std::fill(output_data->begin(), output_data->end(), 0);
  for (size_t ni = 0; ni < input_number; ++ni) {
    for (size_t i = 0; i < size; ++i) {
      output_data->at(i) += i8_inputs[ni].at(i);
    }
  }
  if (do_relu && !is_asymmetric) {
    relu(output_data->data(), output_data->data(), output_data->size());
  }

  for (size_t i = 0; i < size; ++i) {
    output_data->at(i) = (float)applyRShiftAndSaturateInt8(
        output_data->at(i), (uint32_t)rshift.at(0), output_offset);
  }
}

void EltwiseAddOpKernel::invoke() {
  if (datatype == DataType::FP32) {
    fp32_invoke();
  } else if (datatype == DataType::INT8) {
    if (do_quant) {
      i8_invoke();
    } else {
      fp32_invoke();
    }
  } else {
    fp32_invoke();
    clean16bitmantissa(output_data->data(), output_data->data(),
                       output_data->size());
  }
};

EltwiseMaxOpKernel::EltwiseMaxOpKernel(Operation &op,
                                       value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {

  auto elt_addOp = cast<tpu::EltwiseMaxOp>(op);
  const unsigned nInputs = op.getNumOperands() - 4;
  this->do_quant = getOpQuantParamType(&op) != "NONE";
  this->do_relu = elt_addOp.do_relu();
  if (datatype == DataType::INT8) {
    auto quant_rshift = this->opdTensors[nInputs + 2];
    auto quant_multiplier = this->opdTensors[nInputs + 3];
    if (do_quant) {
      assert(quant_rshift);
      assert(quant_multiplier);
      this->rshift.assign(quant_rshift->begin(), quant_rshift->end());
      this->multiplier.assign(quant_multiplier->begin(),
                              quant_multiplier->end());
    }
  }
  this->opdTensors.erase(this->opdTensors.begin() + nInputs, this->opdTensors.end());
  // get tensors
  inputs_data = this->opdTensors;
  output_data = this->resTensor;
}

void EltwiseMaxOpKernel::fp32_invoke() {

  output_data->assign(inputs_data[0]->begin(), inputs_data[0]->end());
  for (size_t ni = 1; ni < inputs_data.size(); ++ni) {
    for (size_t i = 0; i < output_data->size(); ++i) {
      output_data->at(i) = output_data->at(i) > inputs_data[ni]->at(i)
                               ? output_data->at(i)
                               : inputs_data[ni]->at(i);
    }
  }
  if (do_relu) {
    relu(output_data->data(), output_data->data(), output_data->size());
  }
}

void EltwiseMaxOpKernel::i8_invoke() {
  int in = this->shape.at(0);
  int ic = this->shape.at(1);
  int ih = shape.size() > 2 ? this->shape.at(2) : 1;
  int iw = shape.size() > 3 ? this->shape.at(3) : 1;
  size_t input_number = inputs_data.size();
  size_t size = in * ic * ih * iw;

  std::vector<std::vector<float>> i8_inputs(input_number);
  for (size_t i = 0; i < input_number; ++i) {
    i8_inputs[i].assign(inputs_data[i]->begin(), inputs_data[i]->end());
  }

  for (size_t i = 0; i < input_number; ++i) {
    for (size_t j = 0; j < size; ++j) {
      i8_inputs[i][j] *= (int8_t)multiplier.at(i);
    }
  }

  output_data->assign(i8_inputs[0].begin(), i8_inputs[0].end());
  for (size_t ni = 1; ni < input_number; ++ni) {
    for (size_t i = 0; i < size; ++i) {
      output_data->at(i) = output_data->at(i) > i8_inputs[ni].at(i)
                               ? output_data->at(i)
                               : i8_inputs[ni].at(i);
    };
  }

  if (do_relu) {
    relu(output_data->data(), output_data->data(), output_data->size());
  }

  for (size_t i = 0; i < size; ++i) {
    output_data->at(i) = (float)applyRShiftAndSaturateInt8(
        output_data->at(i), (uint32_t)rshift.at(0));
  }
}

void EltwiseMaxOpKernel::invoke() {
  if (datatype == DataType::FP32) {
    fp32_invoke();
  } else if (datatype == DataType::INT8) {
    if (do_quant) {
      i8_invoke();
    } else {
      fp32_invoke();
    }
  } else {
    fp32_invoke();
    clean16bitmantissa(output_data->data(), output_data->data(),
                       output_data->size());
  }
};

EltwiseMinOpKernel::EltwiseMinOpKernel(Operation &op,
                                       value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {

  auto elt_addOp = cast<tpu::EltwiseMinOp>(op);
  const unsigned nInputs = op.getNumOperands() - 4;
  this->do_quant = getOpQuantParamType(&op) != "NONE";

  this->do_relu = elt_addOp.do_relu();
  if (datatype == DataType::INT8) {
    auto quant_rshift = this->opdTensors[nInputs + 2];
    auto quant_multiplier = this->opdTensors[nInputs + 3];
    if (do_quant) {
      assert(quant_rshift);
      assert(quant_multiplier);
      this->rshift.assign(quant_rshift->begin(), quant_rshift->end());
      this->multiplier.assign(quant_multiplier->begin(),
                              quant_multiplier->end());
    }
  }
  this->opdTensors.erase(this->opdTensors.begin() + nInputs, this->opdTensors.end());
  // get tensors
  inputs_data = this->opdTensors;
  output_data = this->resTensor;
}

void EltwiseMinOpKernel::fp32_invoke() {

  output_data->assign(inputs_data[0]->begin(), inputs_data[0]->end());
  for (size_t ni = 1; ni < inputs_data.size(); ++ni) {
    for (size_t i = 0; i < output_data->size(); ++i) {
      output_data->at(i) = output_data->at(i) < inputs_data[ni]->at(i)
                               ? output_data->at(i)
                               : inputs_data[ni]->at(i);
    }
  }
  if (do_relu) {
    relu(output_data->data(), output_data->data(), output_data->size());
  }
}

void EltwiseMinOpKernel::i8_invoke() {
  int in = this->shape.at(0);
  int ic = this->shape.at(1);
  int ih = shape.size() > 2 ? this->shape.at(2) : 1;
  int iw = shape.size() > 3 ? this->shape.at(3) : 1;
  size_t input_number = inputs_data.size();
  size_t size = in * ic * ih * iw;

  std::vector<std::vector<float>> i8_inputs(input_number);
  for (size_t i = 0; i < input_number; ++i) {
    i8_inputs[i].assign(inputs_data[i]->begin(), inputs_data[i]->end());
  }

  for (size_t i = 0; i < input_number; ++i) {
    for (size_t j = 0; j < size; ++j) {
      i8_inputs[i][j] *= (int8_t)multiplier.at(i);
    }
  }

  output_data->assign(i8_inputs[0].begin(), i8_inputs[0].end());
  for (size_t ni = 1; ni < input_number; ++ni) {
    for (size_t i = 0; i < size; ++i) {
      output_data->at(i) = output_data->at(i) < i8_inputs[ni].at(i)
                               ? output_data->at(i)
                               : i8_inputs[ni].at(i);
    };
  }

  if (do_relu) {
    relu(output_data->data(), output_data->data(), output_data->size());
  }

  for (size_t i = 0; i < size; ++i) {
    output_data->at(i) = (float)applyRShiftAndSaturateInt8(
        output_data->at(i), (uint32_t)rshift.at(0));
  }
}

void EltwiseMinOpKernel::invoke() {
  if (datatype == DataType::FP32) {
    fp32_invoke();
  } else if (datatype == DataType::INT8) {
    if (do_quant) {
      i8_invoke();
    } else {
      fp32_invoke();
    }
  } else {
    fp32_invoke();
    clean16bitmantissa(output_data->data(), output_data->data(),
                       output_data->size());
  }
};

EltwiseMulOpKernel::EltwiseMulOpKernel(Operation &op,
                                       value_map_t &valueMapping)
    : CPUOpKernel(op, valueMapping) {

  auto elt_mulOp = cast<tpu::EltwiseMulOp>(op);
  const unsigned nInputs = op.getNumOperands() - 4;

  this->do_quant = getOpQuantParamType(&op) != "NONE";
  this->is_asymmetric = isOpQuantAsymmetric(&op);
  if (is_asymmetric) {
    llvm_unreachable("No Asymmetric mode");
  }
  this->do_relu = elt_mulOp.do_relu();
  if (datatype == DataType::INT8) {
    auto quant_rshift = this->opdTensors[nInputs + 2];
    auto quant_multiplier = this->opdTensors[nInputs + 3];
    if (do_quant) {
      assert(quant_rshift);
      assert(quant_multiplier);
      this->rshift.assign(quant_rshift->begin(), quant_rshift->end());
      this->multiplier.assign(quant_multiplier->begin(),
                              quant_multiplier->end());
    }
  }

  this->opdTensors.erase(this->opdTensors.begin() + nInputs, this->opdTensors.end());
  // get tensors
  inputs_data = this->opdTensors;
  output_data = this->resTensor;
}

void EltwiseMulOpKernel::fp32_invoke() {
  int in = this->shape.at(0);
  int ic = this->shape.at(1);
  int ih = shape.size() > 2 ? this->shape.at(2) : 1;
  int iw = shape.size() > 3 ? this->shape.at(3) : 1;
  std::fill(output_data->begin(), output_data->end(), 1);

  for (size_t ni = 0; ni < inputs_data.size(); ++ni) {
    for (size_t i = 0; i < (size_t)(in * ic * ih * iw); ++i) {
      output_data->at(i) *= inputs_data[ni]->at(i);
    }
  }
  if (do_relu) {
    relu(output_data->data(), output_data->data(), output_data->size());
  }
}

void EltwiseMulOpKernel::i8_invoke() {
  int in = this->shape.at(0);
  int ic = this->shape.at(1);
  int ih = shape.size() > 2 ? this->shape.at(2) : 1;
  int iw = shape.size() > 3 ? this->shape.at(3) : 1;
  size_t input_number = inputs_data.size();
  size_t size = in * ic * ih * iw;

  std::fill(output_data->begin(), output_data->end(), 1);
  for (size_t ni = 0; ni < input_number; ++ni) {
    for (size_t i = 0; i < size; ++i) {
      output_data->at(i) *= inputs_data[ni]->at(i);
    }
  }
  if (do_relu) {
    relu(output_data->data(), output_data->data(), output_data->size());
  }
  if (rshift.at(0) != 0 || multiplier.at(0) != 0) {
    for (size_t i = 0; i < output_data->size(); ++i) {
      output_data->at(i) = (float)applyMultiplierAndRShiftAndSaturateInt8(
          output_data->at(i), (uint32_t)rshift.at(0),
          (uint32_t)multiplier.at(0), true);
    }
  }
}

void EltwiseMulOpKernel::invoke() {
  if (datatype == DataType::FP32) {
    fp32_invoke();
  } else if (datatype == DataType::INT8) {
    if (do_quant) {
      i8_invoke();
    } else {
      fp32_invoke();
    }
  } else {
    fp32_invoke();
    clean16bitmantissa(output_data->data(), output_data->data(),
                       output_data->size());
  }
};

} // namespace mlir