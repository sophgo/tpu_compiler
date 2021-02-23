/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgFcKernel.cpp
 * Description:
 */

#include "TgFcKernel.hpp"

#define DEBUG_TYPE "fc_kernel"

void TgFcKernel::init(uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_weight,
                      gaddr_t ga_bias, gaddr_t ga_output, int M, int K, int N,
                      bool do_bias, bool do_relu, int rshift_width,
                      int multiplier, std::vector<int> compressed_pos,
                      cvk_fmt_t fmt) {

  this->layer_id = layer_id;
  this->M = static_cast<uint32_t>(M);
  this->K = static_cast<uint32_t>(K);
  this->N = static_cast<uint32_t>(N);
  this->ga_input = ga_input;
  this->ga_weight = ga_weight;
  this->ga_bias = ga_bias;
  this->ga_output = ga_output;
  this->do_bias = do_bias;
  this->do_relu = do_relu;
  this->rshift_width = rshift_width;
  this->multiplier = multiplier;
  this->compressed_pos = compressed_pos;
  this->fmt = fmt;
  this->total_steps = 1;
  this->fmt_size = ctx.bytesize_of_fmt(fmt);
  TOTAL_EU = NPU_NUM * ctx.tiu_eu_num(fmt);
  tile_N = this->N;
  tile_K = this->K;
  tile_M = this->M;
  ctx.set_layer_id(layer_id);
}

uint32_t TgFcKernel::lmem_matrix_size(uint32_t row, uint32_t col,
                                      bool ps32) const {
  auto shape = ctx.ml_default_shape(row, col, fmt);
  if (ps32 == false) {
    return ctx.lmem_matrix_to_size(shape, fmt, 1);
  } else {
    return ctx.lmem_ps32_matrix_to_size(shape, fmt, 1);
  }
}

TgFcKernel::lmem_size_t TgFcKernel::get_lmem_size() const {
  lmem_size_t size;
  size.B = do_bias ? lmem_matrix_size(4 / fmt_size, tile_N) : 0;
  size.L = lmem_matrix_size(tile_M, tile_K);
  size.R = lmem_matrix_size(tile_K, tile_N);
  size.Y = lmem_matrix_size(tile_M, tile_N, K != tile_K);

  size.blob_L = 1;
  size.blob_R = 1;
  size.blob_B = 1;
  size.blob_Y = 1;
  if (do_parallel == true && !(tile_K == K && tile_N == N)) {
    size.blob_L = (slice_k() > 1 ? 2 : 1);
    size.blob_R = 2;
    size.blob_B = (slice_n() > 1 ? 2 : 1);
    size.blob_Y = slice_n();
  }
  return size;
}

uint32_t TgFcKernel::total_lmem_size() const {
  auto lmem_size = get_lmem_size();
  return lmem_size.blob_L * lmem_size.L + lmem_size.blob_R * lmem_size.R +
         lmem_size.blob_B * lmem_size.B + lmem_size.blob_Y * lmem_size.Y;
}

void TgFcKernel::update_laddr() {
  auto lmem_size = get_lmem_size();
  uint32_t last_laddr = 0;
  for (uint32_t y = 0; y < lmem_size.blob_Y; y++) {
    Y_laddr.push_back(last_laddr);
    last_laddr += lmem_size.Y;
  }
  L_laddr[0] = L_laddr[1] = last_laddr;
  last_laddr += lmem_size.L;
  if (lmem_size.blob_L > 1) {
    L_laddr[1] = last_laddr;
    last_laddr += lmem_size.L;
  }
  R_laddr[0] = R_laddr[1] = last_laddr;
  last_laddr += lmem_size.R;
  if (lmem_size.blob_R > 1) {
    R_laddr[1] = last_laddr;
    last_laddr += lmem_size.R;
  }
  B_laddr[0] = B_laddr[1] = last_laddr;
  last_laddr += lmem_size.B;
  if (lmem_size.blob_B > 1) {
    B_laddr[1] = last_laddr;
  }
}

void TgFcKernel::selectTilePolicy() {
  tile_info_t info;
  memset(&info, 0, sizeof(info));
  uint32_t max_tiu = (1 << 12) - 1; // 1880v2: 12 bit
  uint32_t maxM = std::min(M, max_tiu);
  uint32_t maxK = std::min(K, max_tiu);
  uint32_t maxN = std::min(N, max_tiu);
  do_parallel = (maxM == M);
  // try parallel first
  for (tile_M = maxM; tile_M > 0;) {
    for (tile_K = maxK; tile_K > 0; tile_K--) {
      for (tile_N = maxN; tile_N > 0;) {
        if (total_lmem_size() <= (uint32_t)LOCAL_MEM_SIZE) {
          goto tiling_exit;
        }
        if (tile_N % TOTAL_EU) {
          tile_N -= (tile_N % TOTAL_EU);
        } else {
          tile_N -= TOTAL_EU;
        }
      }
    }
    if (do_parallel) {
      do_parallel = false;
    } else {
      tile_M--;
    }
  }
tiling_exit:
  if (tile_M == 0) {
    llvm::errs() << llvm::format("Tilling FC failed, M:%d,K:%d,N:%d, fmt:%d\n",
                                 M, K, N, fmt);
    assert(0);
  }

  if (compressed_pos.empty() == false) {
    assert(slice_n() * slice_k() == compressed_pos.size());
  }

  if (tile_M == M && tile_N == N && tile_K == K) {
    do_parallel = false;
  }

  if (do_parallel) {
    for (uint32_t k_idx = 0, pos_k = 0; pos_k < K; k_idx++, pos_k += tile_K) {
      for (uint32_t n_idx = 0, pos_n = 0; pos_n < N; n_idx++, pos_n += tile_N) {
        info.n = std::min(N - pos_n, tile_N);
        info.k = std::min(K - pos_k, tile_K);
        info.m = M;
        info.pos_n = pos_n;
        info.pos_k = pos_k;
        info.pos_m = 0;
        info.Y_idx = n_idx;
        info.compress_idx = n_idx * slice_k() + k_idx;
        tiles.emplace_back(info);
        info.RB_idx = 1 - info.RB_idx;
      }
      info.L_idx = 1 - info.L_idx;
    }
  } else {
    for (uint32_t n_idx = 0, pos_n = 0; pos_n < N; n_idx++, pos_n += tile_N) {
      for (uint32_t pos_m = 0; pos_m < M; pos_m += tile_M) {
        for (uint32_t k_idx = 0, pos_k = 0; pos_k < K; k_idx++, pos_k += tile_K) {
          info.n = std::min(N - pos_n, tile_N);
          info.k = std::min(K - pos_k, tile_K);
          info.m = std::min(M - pos_m, tile_M);
          info.pos_n = pos_n;
          info.pos_k = pos_k;
          info.pos_m = pos_m;
          info.compress_idx = n_idx * slice_k() + k_idx;
          tiles.emplace_back(info);
        }
      }
    }
  }
  total_steps = tiles.size();
  update_laddr();
}

void TgFcKernel::update_tl_matrix(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  ctx.lmem_init_matrix(&tl_L, ctx.ml_default_shape(tile.m, tile.k, fmt), fmt,
                       1);
  tl_L.start_address = L_laddr[tile.L_idx];
  ctx.lmem_init_matrix(&tl_R, ctx.ml_default_shape(tile.k, tile.n, fmt), fmt,
                       1);
  tl_R.start_address = R_laddr[tile.RB_idx];
  if (do_bias) {
    ctx.lmem_init_matrix(&tl_B, ctx.ml_default_shape(4 / fmt_size, tile.n, fmt),
                         fmt, 1);
    tl_B.start_address = B_laddr[tile.RB_idx];
  }
  ctx.lmem_init_matrix(&tl_Y, ctx.ml_default_shape(tile.m, tile.n, fmt), fmt,
                       1);
  tl_Y.start_address = Y_laddr[tile.Y_idx];
}

void TgFcKernel::matrix_for_tiu() {
  if (tl_Y.shape.w >= ctx.tiu_eu_num(fmt)) {
    return;
  }
  tl_Y.shape.w = ctx.tiu_eu_num(fmt);
  tl_Y.stride = ctx.ml_default_stride(tl_Y.shape, fmt, 1);
  tl_R.shape.w = ctx.tiu_eu_num(fmt);
  tl_R.stride = ctx.ml_default_stride(tl_R.shape, fmt, 1);
  if (do_bias) {
    tl_B.shape.w = ctx.tiu_eu_num(fmt);
    tl_B.stride = ctx.ml_default_stride(tl_B.shape, fmt, 1);
  }
}

void TgFcKernel::compute(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  update_tl_matrix(step_idx);
  matrix_for_tiu();
  bool is_last = is_last_k(step_idx);
  uint32_t ps32_mode = 0;   // normal mode
  uint32_t relu_enable = 0; // 1880v2 relu can be used in ps32_mode
  if (tile_K < K) {
    if (tile.pos_k == 0) { // first tile
      ps32_mode = 2;       // write 32b result at the first time
    } else if (is_last) {  // last tile
      ps32_mode = 1;       // load previous 32-bit result
    } else {
      ps32_mode = 3; // init & write 32bits partial sum
    }
  }

  // No tiling or last tile
  if ((ps32_mode == 0 || ps32_mode == 1) && do_relu) {
    relu_enable = 1;
  }
  const cvk_ml_t *p_bias = nullptr;
  if (is_last && do_bias) {
    p_bias = &tl_B;
  }

  // New multiplier and 32bit bias are only used in final post data
  // processing stage.
  // So, only set chan_quan = 1 if no tiling or last tile.
  // And when chan_quan is enabled, des_opt_res0_int8 must be 1
  if (multiplier != 0) {
    cvk_tiu_matrix_multiplication_qm_param_t p = {0};
    p.res = &tl_Y;
    p.left = &tl_L;
    p.right = &tl_R;
    p.bias = p_bias;
    p.rshift_bits = is_last ? rshift_width : 0; // quantization down
    p.relu_enable = relu_enable;
    p.ps32_mode = ps32_mode;
    p.quan_m = multiplier;
    p.layer_id = layer_id;
    p.res_is_int8 = 1;
    ctx.tiu_matrix_multiplication_qm(&p);
  } else {
    cvk_tiu_matrix_multiplication_param_t p = {0};
    p.res = &tl_Y;
    p.left = &tl_L;
    p.right = &tl_R;
    p.bias = p_bias;
    p.lshift_bits = 0;                          // deprecated
    p.rshift_bits = is_last ? rshift_width : 0; // quantization down
    p.res_is_int8 = is_last ? 1 : 0;            // output 8bit
    p.add_result = 0;                           // deprecated
    p.relu_enable = relu_enable;
    p.ps32_mode = ps32_mode;
    p.layer_id = layer_id;
    ctx.tiu_matrix_multiplication(&p);
  }
}

bool TgFcKernel::is_last_k(int32_t step_idx) const {
  if (step_idx >= total_steps - 1) {
    return true;
  }
  auto &tile = tiles[step_idx];
  return tile.pos_k + tile.k == K;
}

void TgFcKernel::load(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  update_tl_matrix(step_idx);
  // load L
  if (tile.pos_n == 0 || do_parallel == false) {
    ctx.tdma_load_stride(&tl_L,
                         ga_input + (tile.pos_m * K + tile.pos_k) * fmt_size,
                         {K * fmt_size});
  }
  // load R
  if (compressed_pos.empty()) {
    ctx.tdma_load_stride(&tl_R,
                         ga_weight + (tile.pos_k * N + tile.pos_n) * fmt_size,
                         {N * fmt_size});
  } else {
    cvi_backend_ml_load_stride(ctx, layer_id,
                               ga_weight + compressed_pos[tile.compress_idx],
                               tl_R.start_address, tile.k, tile.n, tile.n,
                               false, true, fmt, fmt, true);
  }
  // load B
  if (do_bias && is_last_k(step_idx)) {
    ctx.tdma_load_stride(&tl_B, ga_bias + tile.pos_n * fmt_size,
                         {N * fmt_size});
  }
}

void TgFcKernel::store(int32_t step_idx) {
  if (false == is_last_k(step_idx)) {
    return;
  }
  auto &tile = tiles[step_idx];
  update_tl_matrix(step_idx);
  ctx.tdma_store_stride(&tl_Y,
                        ga_output + (tile.pos_m * N + tile.pos_n) * fmt_size,
                        {N * fmt_size});
}

void TgFcKernel::schedule() {
  LLVM_DEBUG(llvm::errs() << llvm::format(
                 "Tilling FC, M:%d,K:%d,N:%d,tile_M:%d,tile_K:%d,tile_N:%d,fmt:%d\n", M,
                 K, N, tile_M, tile_K, tile_N, fmt););
  if (do_parallel) {
    for (int step = 0; step < total_steps + 2; step++) {
      ctx.parallel_enable();
      if (step > 0 && step - 1 < total_steps) {
        compute(step - 1);
      }
      if (step < total_steps) {
        load(step);
      }
      if (step > 1) {
        store(step - 2);
      }
      ctx.parallel_disable();
    }
  } else {
    for (int step = 0; step < total_steps; step++) {
      load(step);
      compute(step);
      store(step);
    }
  }
}

void cvi_backend_tg_fixed_fc_kernel(const CviBackendContext &ctx,
                                    uint32_t layer_id, gaddr_t ga_input,
                                    gaddr_t ga_weight, gaddr_t ga_bias,
                                    gaddr_t ga_output, int M, int K, int N,
                                    bool do_bias, bool do_relu,
                                    int rshift_width, int multiplier,
                                    std::vector<int> compressed_pos) {
  LLVM_DEBUG(
      llvm::errs() << llvm::format("cvi_backend_tg_fixed_fc_kernel\n"
                                   "M:%d,K:%d,N:%d, do_bias %d, do_relu %d\n",
                                   M, K, N, do_bias, do_relu));

  TgFcKernel kernel(ctx);
  kernel.init(layer_id, ga_input, ga_weight, ga_bias, ga_output, M, K, N,
              do_bias, do_relu, rshift_width, multiplier, compressed_pos,
              CVK_FMT_I8);

  kernel.selectTilePolicy();
  kernel.schedule();
}

void cvi_backend_tg_bf16_fc_kernel(const CviBackendContext &ctx,
                                   uint32_t layer_id, gaddr_t ga_input,
                                   gaddr_t ga_weight, gaddr_t ga_bias,
                                   gaddr_t ga_output, int M, int K, int N,
                                   bool do_bias, bool do_relu,
                                   std::vector<int> compressed_pos) {
  LLVM_DEBUG(
      llvm::errs() << llvm::format("cvi_backend_tg_bf16_fc_kernel\n"
                                   "M:%d,K:%d,N:%d, do_bias %d, do_relu %d\n",
                                   M, K, N, do_bias, do_relu));
  TgFcKernel kernel(ctx);
  kernel.init(layer_id, ga_input, ga_weight, ga_bias, ga_output, M, K, N,
              do_bias, do_relu, 0, 0, compressed_pos, CVK_FMT_BF16);

  kernel.selectTilePolicy();
  kernel.schedule();
}
