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
                      bool do_bias, bool do_relu,
                      std::vector<int> *rshift_width,
                      std::vector<int> *multiplier, int batch_high,
                      int batch_low, bool lstride, bool rstride, bool ostride,
                      std::vector<int> compressed_pos, cvk_fmt_t fmt) {

  this->layer_id = layer_id;
  this->M = static_cast<uint32_t>(M);
  this->K = static_cast<uint32_t>(K);
  this->N = static_cast<uint32_t>(N);
  this->ga_i = ga_input;
  this->ga_w = ga_weight;
  this->ga_b = ga_bias;
  this->ga_o = ga_output;
  this->do_bias = do_bias;
  this->do_relu = do_relu;
  this->compressed_pos = compressed_pos;
  this->fmt = fmt;
  this->total_steps = 1;
  this->fmt_size = ctx.bytesize_of_fmt(fmt);
  this->compress_offset = 0;
  TOTAL_EU = NPU_NUM * ctx.tiu_eu_num(fmt);
  uint32_t max_tiu = MAX_TIU_CHL; // 1880v2: 12 bit
  this->maxM = std::min(this->M, max_tiu);
  this->maxK = std::min(this->K, max_tiu);
  this->maxN = std::min(this->N, max_tiu);
  this->tile_N = this->maxN;
  this->tile_K = this->maxK;
  this->tile_M = this->maxM;
  this->lstride = lstride;
  this->rstride = rstride;
  this->ostride = ostride;
  this->batch_high = batch_high > 0 ? batch_high : 1;
  this->batch_low = batch_low > 0 ? batch_low : 1;
  this->batch = this->batch_high * this->batch_low;
  this->cur_multiplier = 0;
  this->cur_rshift = 0;
  left_gstride.row = K * fmt_size * (lstride ? batch_low : 1);
  right_gstride.row = N * fmt_size * (rstride ? batch_low : 1);
  output_gstride.row = N * fmt_size * (ostride ? batch_low : 1);
  size_t batch = this->batch_high * this->batch_low;
  if (rshift_width != nullptr) {
    if (rshift_width->size() == 1) {
      this->rshift.assign(batch, rshift_width->at(0));
    } else if (rshift_width->size() == batch) {
      this->rshift.assign(rshift_width->begin(), rshift_width->end());
    } else {
      llvm_unreachable("rshift size error");
    }
  }
  if (multiplier != nullptr) {
    if (multiplier->size() == 1) {
      this->multiplier.assign(batch, multiplier->at(0));
    } else if (multiplier->size() == batch) {
      this->multiplier.assign(multiplier->begin(), multiplier->end());
    } else {
      llvm_unreachable("multiplier size error");
    }
  }
  if (compressed_pos.empty() == false) {
    this->ga_weight = ga_weight;
  }
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
  switch (mode) {
  case FC_GROUP_PARALLEL:
    size.blob_L = 2;
    size.blob_R = 2;
    size.blob_B = 2;
    size.blob_Y = 2;
    break;
  case FC_PARALLEL:
    size.blob_L = (slice_k() > 1 ? 2 : 1);
    size.blob_R = 2;
    size.blob_B = (slice_n() > 1 ? 2 : 1);
    size.blob_Y = (slice_k() > 1 ? slice_n() : 2);
    break;
  case FC_NO_TILING:
  case FC_NO_PARALLEL:
    size.blob_L = 1;
    size.blob_R = 1;
    size.blob_B = 1;
    size.blob_Y = 1;
    break;
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

// tiling N, for each group
bool TgFcKernel::try_tiling_group_parallel() {
  mode = FC_GROUP_PARALLEL;
  if (batch == 1 || maxK != K || maxM != M) {
    return false;
  }
  tile_K = K;
  tile_M = M;
  for (tile_N = maxN; tile_N > 0;) {
    if (total_lmem_size() <= (uint32_t)LOCAL_MEM_SIZE) {
      goto tiling_group_parallel_exit;
    }
    if (tile_N % TOTAL_EU) {
      tile_N -= (tile_N % TOTAL_EU);
    } else {
      tile_N -= TOTAL_EU;
    }
  }
tiling_group_parallel_exit:
  if (tile_N == 0) {
    return false;
  }
  tile_info_t info = {0};
  info.k = K;
  info.m = M;
  for (info.batch_high = 0; info.batch_high < batch_high; info.batch_high++) {
    for (info.batch_low = 0; info.batch_low < batch_low; info.batch_low++) {
      for (info.pos_n = 0; info.pos_n < N; info.pos_n += tile_N) {
        info.n = std::min(tile_N, N - info.pos_n);
        info.compress_idx = info.pos_n / tile_N;
        tiles.emplace_back(info);
        info.RB_idx = 1 - info.RB_idx;
        info.Y_idx = 1 - info.Y_idx;
      }
      info.L_idx = 1 - info.L_idx;
    }
  }
  return true;
}

bool TgFcKernel::try_no_tiling() {
  mode = FC_NO_TILING;
  if (batch > 1 || maxM != M || maxN != N || maxK != K) {
    return false;
  }
  if (total_lmem_size() > (uint32_t)LOCAL_MEM_SIZE) {
    return false;
  }
  tile_info_t info = {0};
  info.n = N;
  info.k = K;
  info.m = M;
  info.batch_high = 1;
  info.batch_low = 1;
  info.compress_idx = 0;
  tiles.emplace_back(info);
  return true;
}

bool TgFcKernel::try_tiling_parallel() {
  mode = FC_PARALLEL;
  if (maxM != M) {
    return false;
  }

  for (tile_K = maxK; tile_K > 0; tile_K--) {
    for (tile_N = maxN; tile_N > 0;) {
      if (total_lmem_size() <= (uint32_t)LOCAL_MEM_SIZE) {
        goto tiling_parallel_exit;
      }
      if (tile_N % TOTAL_EU) {
        tile_N -= (tile_N % TOTAL_EU);
      } else {
        tile_N -= TOTAL_EU;
      }
    }
  }
tiling_parallel_exit:
  if (tile_K == 0) {
    return false;
  }

  auto size = get_lmem_size();
  tile_info_t info = {0};
  info.m = M;
  for (uint32_t k_idx = 0, pos_k = 0; pos_k < K; k_idx++, pos_k += tile_K) {
    for (uint32_t n_idx = 0, pos_n = 0; pos_n < N; n_idx++, pos_n += tile_N) {
      info.n = std::min(N - pos_n, tile_N);
      info.k = std::min(K - pos_k, tile_K);
      info.pos_n = pos_n;
      info.pos_k = pos_k;
      info.Y_idx = n_idx % size.blob_Y;
      info.compress_idx = n_idx * slice_k() + k_idx;
      tiles.emplace_back(info);
      info.RB_idx = 1 - info.RB_idx;
    }
    info.L_idx = 1 - info.L_idx;
  }
  return true;
}

bool TgFcKernel::try_tiling_no_parallel() {
  mode = FC_NO_PARALLEL;
  // try parallel first
  for (tile_M = maxM; tile_M > 0; tile_M--) {
    for (tile_K = maxK; tile_K > 0; tile_K--) {
      for (tile_N = maxN; tile_N > 0;) {
        if (total_lmem_size() <= (uint32_t)LOCAL_MEM_SIZE) {
          goto tiling_no_parallel_exit;
        }
        if (tile_N % TOTAL_EU) {
          tile_N -= (tile_N % TOTAL_EU);
        } else {
          tile_N -= TOTAL_EU;
        }
      }
    }
  }
tiling_no_parallel_exit:
  if (tile_M == 0) {
    return false;
  }

  tile_info_t info = {0};
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
  return true;
}

void TgFcKernel::selectTilePolicy() {
  if (try_tiling_group_parallel()) {
  } else if (try_no_tiling()) {
  } else if (try_tiling_parallel()) {
  } else if (try_tiling_no_parallel()) {
  } else {
    llvm::errs() << llvm::format("Tilling FC failed, M:%d,K:%d,N:%d, fmt:%d\n",
                                 M, K, N, fmt);
    assert(0);
  }
  total_steps = tiles.size();
  update_laddr();
  if (compressed_pos.empty() == false) {
    assert(batch * slice_n() * slice_k() == compressed_pos.size());
  }
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
  if (mode == FC_GROUP_PARALLEL) {
    update_batch_info(tile.batch_high, tile.batch_low);
  }
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
  if (cur_multiplier != 0) {
    cvk_tiu_matrix_multiplication_qm_param_t p = {0};
    p.res = &tl_Y;
    p.left = &tl_L;
    p.right = &tl_R;
    p.bias = p_bias;
    p.rshift_bits = is_last ? cur_rshift : 0; // quantization down
    p.relu_enable = relu_enable;
    p.ps32_mode = ps32_mode;
    p.quan_m = cur_multiplier;
    p.layer_id = layer_id;
    p.res_is_int8 = 1;
    ctx.tiu_matrix_multiplication_qm(&p);
  } else {
    cvk_tiu_matrix_multiplication_param_t p = {0};
    p.res = &tl_Y;
    p.left = &tl_L;
    p.right = &tl_R;
    p.bias = p_bias;
    p.lshift_bits = 0;                        // deprecated
    p.rshift_bits = is_last ? cur_rshift : 0; // quantization down
    p.res_is_int8 = is_last ? 1 : 0;          // output 8bit
    p.add_result = 0;                         // deprecated
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
  if (mode == FC_GROUP_PARALLEL) {
    update_batch_info(tile.batch_high, tile.batch_low);
  }
  // load L
  if (tile.pos_n == 0 || mode == FC_NO_PARALLEL) {
    ctx.tdma_load_stride(
        &tl_L, ga_input + tile.pos_m * left_gstride.row + tile.pos_k * fmt_size,
        left_gstride);
  }
  // load R
  if (compressed_pos.empty()) {
    ctx.tdma_load_stride(&tl_R,
                         ga_weight + tile.pos_k * right_gstride.row +
                             tile.pos_n * fmt_size,
                         right_gstride);
  } else {
    cvi_backend_ml_load_stride(
        ctx, layer_id,
        ga_weight + compressed_pos[tile.compress_idx + compress_offset],
        tl_R.start_address, tile.k, tile.n, tile.n, false, true, fmt, fmt,
        true);
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
  if (mode == FC_GROUP_PARALLEL) {
    update_batch_info(tile.batch_high, tile.batch_low);
  }
  ctx.tdma_store_stride(&tl_Y,
                        ga_output + tile.pos_m * output_gstride.row +
                            tile.pos_n * fmt_size,
                        output_gstride);
}

void TgFcKernel::update_batch_info(int high_idx, int low_idx) {
  int batch_idx = high_idx * batch_low + low_idx;
  if (lstride) {
    ga_input = ga_i + high_idx * M * left_gstride.row + low_idx * K * fmt_size;
  } else {
    ga_input = ga_i + batch_idx * M * left_gstride.row;
  }
  if (compressed_pos.empty()) {
    if (rstride) {
      ga_weight =
          ga_w + high_idx * K * right_gstride.row + low_idx * N * fmt_size;
    } else {
      ga_weight = ga_w + batch_idx * K * right_gstride.row;
    }
  }
  if (ostride) {
    ga_output =
        ga_o + high_idx * M * output_gstride.row + low_idx * N * fmt_size;
  } else {
    ga_output = ga_o + batch_idx * M * output_gstride.row;
  }
  if (do_bias) {
    ga_bias = ga_b + batch_idx * N * 4;
  }
  compress_offset = batch_idx * slice_k() * slice_n();
  if (multiplier.empty() || rshift.empty()) {
    return;
  }
  cur_multiplier = multiplier[batch_idx];
  cur_rshift = rshift[batch_idx];
}

void TgFcKernel::schedule_group_parallel() {
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
}

void TgFcKernel::schedule_parallel() {
  for (int b0 = 0; b0 < batch_high; b0++) {
    for (int b1 = 0; b1 < batch_low; b1++) {
      update_batch_info(b0, b1);
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
    }
  }
}

void TgFcKernel::schedule_no_parallel() {
  for (int b0 = 0; b0 < batch_high; b0++) {
    for (int b1 = 0; b1 < batch_low; b1++) {
      update_batch_info(b0, b1);
      for (int step = 0; step < total_steps; step++) {
        load(step);
        compute(step);
        store(step);
      }
    }
  }
}

void TgFcKernel::schedule() {
  switch (mode) {
  case FC_GROUP_PARALLEL:
    schedule_group_parallel();
    break;
  case FC_PARALLEL:
    schedule_parallel();
    break;
  case FC_NO_PARALLEL:
  case FC_NO_TILING:
    schedule_no_parallel();
    break;
  default:
    assert(0);
  }
}

void cvi_backend_tg_fixed_fc_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_input,
    gaddr_t ga_weight, gaddr_t ga_bias, gaddr_t ga_output, int M, int K, int N,
    bool do_bias, bool do_relu, std::vector<int> rshift_width,
    std::vector<int> multiplier, std::vector<int> compressed_pos,
    int batch_high, int batch_low, bool lstride, bool rstride, bool ostride) {
  TgFcKernel kernel(ctx);
  kernel.init(layer_id, ga_input, ga_weight, ga_bias, ga_output, M, K, N,
              do_bias, do_relu, &rshift_width, &multiplier, batch_high,
              batch_low, lstride, rstride, ostride, compressed_pos, CVK_FMT_I8);

  kernel.selectTilePolicy();
  kernel.schedule();
}

void cvi_backend_tg_bf16_fc_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_input,
    gaddr_t ga_weight, gaddr_t ga_bias, gaddr_t ga_output, int M, int K, int N,
    bool do_bias, bool do_relu, std::vector<int> compressed_pos, int batch_high,
    int batch_low, bool lstride, bool rstride, bool ostride) {
  TgFcKernel kernel(ctx);
  kernel.init(layer_id, ga_input, ga_weight, ga_bias, ga_output, M, K, N,
              do_bias, do_relu, nullptr, nullptr, batch_high, batch_low,
              lstride, rstride, ostride, compressed_pos, CVK_FMT_BF16);

  kernel.selectTilePolicy();
  kernel.schedule();
}
