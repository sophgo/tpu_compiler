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
#include "tpuc/CustomOpParam.h"

namespace cvi {

class CustomOp {
public:
  CustomOp(OpParam &param) : param(param) {}
  virtual ~CustomOp() {}

  virtual void
  interpretFp32(std::vector<std::shared_ptr<std::vector<float>>> &operand_tensors,
                std::vector<std::vector<int64_t>> &operand_shapes,
                std::shared_ptr<std::vector<float>> &result_tensor,
                std::vector<int64_t> &result_shape) {
    std::cout << "interpretFp32() isn't implemented.\n";
    assert(0);
  }

  virtual void quantizeBf16() {
    std::cout << "quantizeBf16() isn't implemented.\n";
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

protected:
  OpParam &param;
};

#define RegisterCustomOp(N, X)                                    \
  extern "C" CustomOp *CustomOp##N##Create(cvi::OpParam &param) { \
    return new X(param);                                \
  }

}


#endif
