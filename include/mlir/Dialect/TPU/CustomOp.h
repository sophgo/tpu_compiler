#ifndef CVI_CUSTOM_OP_H
#define CVI_CUSTOM_OP_H

#include <assert.h>
#include <iostream>
#include <map>
#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <typeinfo>
#include <stdint.h>
#include "mlir/Dialect/TPU/CustomOpParam.h"

namespace cvi {

class CustomOp {
public:
  CustomOp(OpParam &param) : param(param) {}
  virtual ~CustomOp() {}

  virtual void
  interpretInt8(std::vector<std::shared_ptr<std::vector<float>>> &operand_tensors,
                std::vector<std::vector<int64_t>> &operand_shapes,
                std::shared_ptr<std::vector<float>> &result_tensor,
                std::vector<int64_t> &result_shape) {
    std::cout << "interpretInt8() isn't implemented.\n";
    assert(0);
  }

  virtual void
  interpretFp32(std::vector<std::shared_ptr<std::vector<float>>> &operand_tensors,
                std::vector<std::vector<int64_t>> &operand_shapes,
                std::shared_ptr<std::vector<float>> &result_tensor,
                std::vector<int64_t> &result_shape) {
    std::cout << "interpretFp32() isn't implemented.\n";
    assert(0);
  }

  virtual void
  interpretBf16(std::vector<std::shared_ptr<std::vector<float>>> &operand_tensors,
                std::vector<std::vector<int64_t>> &operand_shapes,
                std::shared_ptr<std::vector<float>> &result_tensor,
                std::vector<int64_t> &result_shape) {
    std::cout << "interpretBf16() isn't implemented.\n";
    assert(0);
  }

  virtual void quantizeInt8() {
    std::cout << "quantizeInt8() isn't implemented.\n";
    assert(0);
  }

  virtual void quantizeBf16() {
    std::cout << "quantizeBf16() isn't implemented.\n";
    assert(0);
  }

  virtual void codeGenInt8(void *ctx,
                       std::vector<std::vector<int64_t>> &operand_shapes,
                       std::vector<uint64_t> &operand_gaddrs,
                       std::vector<int64_t> &result_shape, uint64_t result_gaddr,
                       int layer_id) {
    std::cout << "codegen() isn't implemented.\n";
    assert(0);
  }

  virtual void codeGenBf16(void *ctx,
                       std::vector<std::vector<int64_t>> &operand_shapes,
                       std::vector<uint64_t> &operand_gaddrs,
                       std::vector<int64_t> &result_shape, uint64_t result_gaddr,
                       int layer_id) {
    std::cout << "codegen() isn't implemented.\n";
    assert(0);
  }

  void setQuantParam(OpParam *quant, float prev_threshold) {
    this->quant = quant;
    prev_threshold_ = prev_threshold;
  }

protected:
  OpParam &param;

  void setOpQuantParamType(std::string type) {
    for (auto &avail : quant_type_) {
      if (type == avail) {
        quant->put<std::string>("param_type", type);
        return;
      }
    }
    std::cout << "not supported quant type:" << type << "\n";
    assert(0);
  }

  void setOpQuantPerchannel(bool yes) { quant->put<bool>("is_perchannel", yes); }

  float getOpThreshold() { return quant->get<float>("threshold_max"); }

  float getPrevOpThreshold() { return prev_threshold_; }


private:
  OpParam *quant;
  float prev_threshold_;
  std::vector<std::string> quant_type_ = {
      "NONE",        "THRESHOLD",        "SCALE",
      "RSHIFT_ONLY", "RSHIFT_AND_M_I32", "RSHIFT_AND_M_I8",
      "LUT_INT8",    "LUT_BF16"};
};

#define RegisterCustomOp(N, X)                                    \
  extern "C" CustomOp *CustomOp##N##Create(cvi::OpParam &param) { \
    return new X(param);                                \
  }

}


#endif
