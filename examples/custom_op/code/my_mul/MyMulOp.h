/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 */
#ifndef MYMUL_OP_H_
#define MYMUL_OP_H_

#include "tpuc/CustomOp.h"
#include <cvikernel/cvikernel.h>

namespace cvi {

class MyMulOp : public CustomOp {
public:
  MyMulOp(OpParam &param) : CustomOp(param) {}

  void interpretFp32(
      std::vector<std::shared_ptr<std::vector<float>>> &operand_tensors,
      std::vector<std::vector<int64_t>> &operand_shapes,
      std::shared_ptr<std::vector<float>> &result_tensor,
      std::vector<int64_t> &result_shape);

  void codeGenBf16(void *ctx, std::vector<std::vector<int64_t>> &operand_shapes,
                   std::vector<uint64_t> &operand_gaddrs,
                   std::vector<int64_t> &result_shape, uint64_t result_gaddr,
                   int layer_id);

private:
  typedef struct tiling_info {
    uint32_t n;
    uint32_t c;
    uint32_t h;
    uint32_t w;
    uint64_t offset; // gmem offset
  } tiling_info_t;
  std::vector<tiling_info_t> tiles;
  void tiling(void *ctx, int64_t total);
};

} // namespace cvi
#endif
