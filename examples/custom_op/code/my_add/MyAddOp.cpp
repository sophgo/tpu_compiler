/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 */
#include "MyAddOp.h"
#include <cvikernel/cvikernel.h>
#include <numeric>
#include <string.h>

#define NPU_NUM 32
#define EU_NUM 8
#define LOCAL_MEM_SIZE (1 << 15)
#define MAX_TIU_NUM (4095 - 32)

namespace cvi {
static inline uint8_t getTdmaBaseSelectIndexFromGaddr(uint64_t gaddr) {
  return (uint8_t)((((uint64_t)gaddr) >> 40) & 0x07);
}

void MyAddOp::interpretFp32(
    std::vector<std::shared_ptr<std::vector<float>>> &operand_tensors,
    std::vector<std::vector<int64_t>> &operand_shapes,
    std::shared_ptr<std::vector<float>> &result_tensor,
    std::vector<int64_t> &result_shape) {
  assert(operand_tensors.size() == 2);
  assert(operand_shapes.size() == 2);
  auto &shape0 = operand_shapes[0];
  auto &shape1 = operand_shapes[1];
  auto &shape = result_shape;
  assert(shape0 == shape1 && shape0 == shape);

  float *data0 = operand_tensors[0]->data();
  float *data1 = operand_tensors[1]->data();
  float *output = result_tensor->data();
  int64_t total = std::accumulate(shape.begin(), shape.end(), 1,
                                  std::multiplies<int64_t>());

  for (int64_t i = 0; i < total; i++) {
    output[i] = data0[i] + data1[i];
  }
}

static inline int ceiling_func(int numerator, int denominator) {
  return (numerator + denominator - 1) / denominator;
}

void MyAddOp::tiling(void *ctx, int64_t total) {
  cvk_context_t *context = (cvk_context_t *)ctx;
  tiling_info_t tile;
  memset(&tile, 0, sizeof(tile));
  tile.n = 1;
  tile.c = NPU_NUM;
  tile.w = EU_NUM;
  tile.h = std::min(ceiling_func(total, tile.c * tile.w), MAX_TIU_NUM);
  bool lmem_ok = false;
  tiles.clear();
  while (total > 0) {
    int64_t count = tile.n * tile.c * tile.h * tile.w;
    cvk_tl_shape_t tl_shape = {
        .n = tile.n, .c = tile.c, .h = tile.h, .w = tile.w};
    if (lmem_ok == false) {
      uint32_t lsize = 2 * context->ops->lmem_tensor_to_size(context, tl_shape,
                                                             CVK_FMT_BF16, 1);
      lmem_ok = (lsize <= (uint32_t)LOCAL_MEM_SIZE);
    }
    if (count > total || lmem_ok == false) {
      if (tile.h > 1) {
        tile.h--;
      } else if (tile.w > 1) {
        tile.w--;
      } else if (tile.c > 1) {
        tile.c--;
      } else {
        assert(0 && "lmem is not enough");
      }
    } else {
      tiles.emplace_back(tile);
      total -= count;
      tile.offset += count * 2;
    }
  }
  assert(total == 0 && "tiling error");
  return;
}

void MyAddOp::codeGenBf16(void *ctx,
                          std::vector<std::vector<int64_t>> &operand_shapes,
                          std::vector<uint64_t> &operand_gaddrs,
                          std::vector<int64_t> &result_shape,
                          uint64_t result_gaddr, int layer_id) {

  uint64_t ga_input0 = operand_gaddrs[0];
  uint64_t ga_input1 = operand_gaddrs[1];
  uint64_t ga_output = result_gaddr;
  // tiling
  auto &shape = result_shape;
  int64_t total = std::accumulate(shape.begin(), shape.end(), 1,
                                  std::multiplies<int64_t>());
  tiling(ctx, total);
  cvk_context_t *context = (cvk_context_t *)ctx;
  for (auto &tile : tiles) {
    cvk_tl_shape_t tl_shape = {
        .n = tile.n, .c = tile.c, .h = tile.h, .w = tile.w};
    auto tl_input0 =
        context->ops->lmem_alloc_tensor(context, tl_shape, CVK_FMT_BF16, 1);
    auto tl_input1 =
        context->ops->lmem_alloc_tensor(context, tl_shape, CVK_FMT_BF16, 1);
    // load input 0
    cvk_tg_t tg_i0 = {0};
    tg_i0.fmt = CVK_FMT_BF16;
    tg_i0.start_address = ga_input0 + tile.offset;
    tg_i0.base_reg_index = getTdmaBaseSelectIndexFromGaddr(ga_input0);
    tg_i0.shape = {tile.n, tile.c, tile.h, tile.w};
    tg_i0.stride =
        context->ops->tg_default_stride(context, tg_i0.shape, CVK_FMT_BF16);

    cvk_tdma_g2l_tensor_copy_param_t p0 = {0};
    p0.src = &tg_i0;
    p0.dst = tl_input0;
    p0.layer_id = layer_id;
    context->ops->tdma_g2l_bf16_tensor_copy(context, &p0);

    // load input 1
    cvk_tg_t tg_i1 = {0};
    tg_i1.fmt = CVK_FMT_BF16;
    tg_i1.start_address = ga_input1 + tile.offset;
    tg_i1.base_reg_index = getTdmaBaseSelectIndexFromGaddr(ga_input1);
    tg_i1.shape = {tile.n, tile.c, tile.h, tile.w};
    tg_i1.stride =
        context->ops->tg_default_stride(context, tg_i1.shape, CVK_FMT_BF16);

    cvk_tdma_g2l_tensor_copy_param_t p1 = {0};
    p1.src = &tg_i1;
    p1.dst = tl_input1;
    p1.layer_id = layer_id;
    context->ops->tdma_g2l_bf16_tensor_copy(context, &p1);

    // add input 0 and input 1 => input0
    cvk_tiu_add_param_t p2 = {0};
    p2.res_low = tl_input0;
    p2.a_low = tl_input0;
    p2.b.low = tl_input1;
    p2.layer_id = layer_id;
    context->ops->tiu_add(context, &p2);

    // store
    cvk_tg_t tg_dst = {0};
    tg_dst.fmt = CVK_FMT_BF16;
    tg_dst.start_address = ga_output + tile.offset;
    tg_dst.base_reg_index = getTdmaBaseSelectIndexFromGaddr(ga_output);
    tg_dst.shape = {tile.n, tile.c, tile.h, tile.w};
    tg_dst.stride =
        context->ops->tg_default_stride(context, tg_dst.shape, CVK_FMT_BF16);

    cvk_tdma_l2g_tensor_copy_param_t p3 = {0};
    p3.src = tl_input0;
    p3.dst = &tg_dst;
    p3.layer_id = layer_id;
    context->ops->tdma_l2g_bf16_tensor_copy(context, &p3);

    context->ops->lmem_free_tensor(context, tl_input1);
    context->ops->lmem_free_tensor(context, tl_input0);
  }
}

RegisterCustomOp(MyAdd, MyAddOp);

} // namespace cvi