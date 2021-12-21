/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgFcKernel.cpp
 * Description:
 */

#include <map>
#include "TgFcKernel.hpp"
#include "tpuc/TPUCompressUtil.h"

#define DEBUG_TYPE "fc_kernel"

void TgFcKernel::init(
    uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_weight, gaddr_t ga_bias,
    gaddr_t ga_output, int M, int K, int N, bool do_bias, bool do_relu,
    std::vector<int> *rshift_width, std::vector<int> *multiplier,
    int batch_high, int batch_low, bool lstride, bool rstride, bool ostride,
    bool need_compress, const std::vector<uint8_t> *weight,
    std::vector<int> *compr_weight_poss, std::vector<uint8_t> *compr_weight,
    cvk_fmt_t fmt, bool do_quant_bf16, gaddr_t ga_scale, gaddr_t ga_zeropoint) {

  this->layer_id = layer_id;
  this->M = static_cast<uint32_t>(M);
  this->K = static_cast<uint32_t>(K);
  this->N = static_cast<uint32_t>(N);
  this->ga_i = ga_input;
  this->ga_w = ga_weight;
  this->ga_b = ga_bias;
  this->ga_o = ga_output;
  this->ga_scale = ga_scale;
  this->ga_zeropoint = ga_zeropoint;
  this->do_quant_bf16 = do_quant_bf16;
  this->do_bias = do_bias;
  this->do_relu = do_relu;
  this->need_compress = need_compress;
  this->filter = weight;
  this->compressed_pos = compr_weight_poss;
  this->compressed_weight = compr_weight;
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
  this->bias_loaded = false;
  this->quant_loaded = false;
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
  if (need_compress) {
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
  size.Q = do_quant_bf16 ? lmem_matrix_size(1, tile_N) : 0;
  switch (mode) {
  case FC_GROUP_PARALLEL:
    size.blob_L = 2;
    size.blob_R = 2;
    size.blob_B = 2;
    size.blob_Y = 2;
    size.blob_Q = (slice_n() > 1 ? 4 : 2);
    break;
  case FC_PARALLEL_KN:
    // M not slice
    size.blob_L = (slice_k() > 1 ? 2 : 1);
    size.blob_R = 2;
    size.blob_B = (slice_n() > 1 ? 2 : 1);
    size.blob_Y = (slice_k() > 1 ? slice_n() : 2);
    size.blob_Q = (slice_n() > 1 ? 4 : 2);
    break;
  case FC_PARALLEL_MN:
    // K not slice
    size.blob_L = (slice_m() > 1 ? 2 : 1);
    size.blob_R = (slice_n() > 1 ? 2 : 1);
    size.blob_B = (slice_n() > 1 ? 2 : 1);
    size.blob_Y = 2;
    size.blob_Q = (slice_n() > 1 ? 4 : 2);
    break;
  case FC_NO_TILING:
  case FC_NO_PARALLEL:
    size.blob_L = 1;
    size.blob_R = 1;
    size.blob_B = 1;
    size.blob_Y = 1;
    size.blob_Q = 2;
    break;
  }
  return size;
}

uint32_t TgFcKernel::total_lmem_size() const {
  auto lmem_size = get_lmem_size();
  return lmem_size.blob_L * lmem_size.L + lmem_size.blob_R * lmem_size.R +
         lmem_size.blob_B * lmem_size.B + lmem_size.blob_Y * lmem_size.Y +
         lmem_size.blob_Q * lmem_size.Q;
}

void TgFcKernel::set_laddr() {
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
    last_laddr += lmem_size.B;
  }
  Q_laddr[0] = Q_laddr[1] = last_laddr;
  last_laddr += lmem_size.Q;
  Q_laddr[2] = Q_laddr[3] = last_laddr;
  last_laddr += lmem_size.Q;
  if (lmem_size.blob_Q == 4) {
    Q_laddr[1] = last_laddr;
    last_laddr += lmem_size.Q;
    Q_laddr[3] = last_laddr;
    last_laddr += lmem_size.Q;
  }
  assert(last_laddr <= (uint32_t)LOCAL_MEM_SIZE);
}

// tiling N, for each group
bool TgFcKernel::try_tiling_group_parallel() {
  mode = FC_GROUP_PARALLEL;
  if (batch == 1 || maxK != K || maxM != M) {
    return false;
  }
  tile_K = maxK;
  tile_M = maxM;
  for (tile_N = maxN; tile_N > 0; tile_N--) {
    if (total_lmem_size() <= (uint32_t)LOCAL_MEM_SIZE) {
      goto tiling_group_parallel_exit;
    }
  }
  return false;
tiling_group_parallel_exit:
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
  if (maxM != M || maxN != N || maxK != K) {
    return false;
  }
  tile_K = maxK;
  tile_M = maxM;
  tile_N = maxN;
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

bool TgFcKernel::try_tiling_parallel_kn() {
  mode = FC_PARALLEL_KN;
  if (maxM != M) {
    return false;
  }
  tile_M = maxM;
  for (tile_K = maxK; tile_K > 0; tile_K--) {
    for (tile_N = maxN; tile_N > 0; tile_N--) {
      if (total_lmem_size() <= (uint32_t)LOCAL_MEM_SIZE) {
        goto parallel_kn_success;
      }
    }
  }
  return false;
parallel_kn_success:
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

bool TgFcKernel::try_tiling_parallel_mn() {
  mode = FC_PARALLEL_MN;
  if (maxK != K) {
    return false;
  }
  tile_K = maxK;
  for (tile_M = maxM; tile_M > 0; tile_M--) {
    for (tile_N = maxN; tile_N > 0; tile_N--) {
      if (total_lmem_size() <= (uint32_t)LOCAL_MEM_SIZE) {
        goto parallel_mn_success;
      }
    }
  }
  return false;
parallel_mn_success:
  tile_info_t info = {0};
  info.k = K;
  for (uint32_t n_idx = 0, pos_n = 0; pos_n < N; n_idx++, pos_n += tile_N) {
    for (uint32_t m_idx = 0, pos_m = 0; pos_m < M; m_idx++, pos_m += tile_M) {
      info.m = std::min(M - pos_m, tile_M);
      info.n = std::min(N - pos_n, tile_N);
      info.pos_m = pos_m;
      info.pos_n = pos_n;
      info.compress_idx = n_idx;
      tiles.emplace_back(info);
      info.L_idx = 1 - info.L_idx;
      info.Y_idx = 1 - info.Y_idx;
    }
    info.RB_idx = 1 - info.RB_idx;
  }
  return true;
}

bool TgFcKernel::try_tiling_no_parallel() {
  mode = FC_NO_PARALLEL;
  // try parallel first
  for (tile_M = maxM; tile_M > 0; tile_M--) {
    for (tile_K = maxK; tile_K > 0; tile_K--) {
      for (tile_N = maxN; tile_N > 0; tile_N--) {
        if (total_lmem_size() <= (uint32_t)LOCAL_MEM_SIZE) {
          goto tiling_no_parallel_exit;
        }
      }
    }
  }
  return false;
tiling_no_parallel_exit:
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

template <typename T>
static void stridedMatrixMemcpy(T *dstPtr, const T *srcPtr, int srcStride, int H,
                                int W) {
  for (int i = 0; i < H; ++i) {
    for (int j = 0; j < W; ++j) {
      dstPtr[i * W + j] = srcPtr[i * srcStride + j];
    }
  }
}

void TgFcKernel::compress_weight() {
  int dstOffset = 0;
  int filterSize = filter->size();

  // remove redundant pos_k, pos_n
  // [pos_n, [pos_k, {n_size, k_size, compress_idx}]]
  std::map<uint32_t, std::map<uint32_t, std::tuple<uint32_t, uint32_t, int> > > kn_map;

  int split_num = 0;
  for (auto &tile : tiles) {
    if (kn_map.find(tile.pos_n) != kn_map.end() &&
        kn_map[tile.pos_n].find(tile.pos_k) != kn_map[tile.pos_n].end()) {
      assert(std::get<0>(kn_map[tile.pos_n][tile.pos_k]) == tile.n &&
             std::get<1>(kn_map[tile.pos_n][tile.pos_k]) == tile.k &&
             std::get<2>(kn_map[tile.pos_n][tile.pos_k]) == tile.compress_idx);
    } else {
      kn_map[tile.pos_n][tile.pos_k] = std::make_tuple(tile.n, tile.k, tile.compress_idx);
      ++split_num;
    }
  }

  compressed_pos->resize(split_num * batch);
  int compress_idx_offset = split_num;
  for (int b = 0; b < batch; ++b) {
    for (auto n_iter : kn_map) {
      for (auto k_iter : n_iter.second) {
        int n_pos = n_iter.first;
        int k_pos = k_iter.first;
        int n_size = 0;
        int k_size = 0;
        int compress_idx = 0;
        std::tie(n_size, k_size, compress_idx) = k_iter.second;
        compress_idx += compress_idx_offset * b;

        int srcOffset = b * N * K + k_pos * N + n_pos;
        int stepSize = n_size * k_size * fmt_size;
        auto plainData = std::make_unique<std::vector<uint8_t>>(stepSize);
        bool isBf16Flt = false;
        if (fmt_size == 1) {
          stridedMatrixMemcpy<uint8_t>(
              (uint8_t *)plainData->data(),
              (const uint8_t *)filter->data() + srcOffset, N, k_size, n_size);
        } else {
          isBf16Flt = true;
          stridedMatrixMemcpy<uint16_t>((uint16_t *)plainData->data(),
                                        (uint16_t *)filter->data() + srcOffset,
                                        N, k_size, n_size);
        }
        // Calculate compress parameter first.
        mlir::CompressCommandInfo cmdInfo;
        std::memset(&cmdInfo, 0, sizeof(cmdInfo));
        cmdInfo.signedness = isBf16Flt ? 0 : 1;
        cmdInfo.is_bfloat16 = isBf16Flt ? 1 : 0;
        cmdInfo.bias0 = isBf16Flt ? 127 : 0;
        getCompressParameter(plainData->data(), stepSize, cmdInfo.signedness,
                             cmdInfo.is_bfloat16, &cmdInfo);

        // Create Compress data.
        int requiredSize =
            mlir::getCompressedDataSize(stepSize, isBf16Flt ? 1 : 0);
        auto compressedData =
            std::make_unique<std::vector<uint8_t>>(requiredSize);
        int compressedSize = requiredSize;

        if (isBf16Flt)
          mlir::compressBf16Data(plainData->data(), stepSize,
                                 compressedData->data(), &compressedSize,
                                 &cmdInfo);
        else
          mlir::compressInt8Data(plainData->data(), stepSize,
                                 compressedData->data(), &compressedSize,
                                 &cmdInfo);

        if ((dstOffset + compressedSize) > filterSize) {
          llvm::errs() << "  compressed size exceed, dstOffset " << dstOffset
                       << " + " << compressedSize << " > " << filterSize
                       << "\n";
          need_compress = false;
          compressed_weight->clear();
          compressed_pos->clear();
          break;
        }
        {
          // debug
          LLVM_DEBUG(llvm::errs() << "n_pos:" << n_pos << "  k_pos:" << k_pos
                                  << "  n_size:" << n_size
                                  << "  k_size:" << k_size << "\n";
                     llvm::errs()
                     << "weight length:" << stepSize
                     << " compressed size:" << compressedSize << "\n";);
        }

        // Fill compressed data.
        std::memcpy(compressed_weight->data() + dstOffset,
                    compressedData->data(), compressedSize);
        (*compressed_pos)[compress_idx] = dstOffset;
        dstOffset += compressedSize;
      }
    }
  }
  LLVM_DEBUG(
    if (need_compress) {
      llvm::errs() << "weight size:" << filterSize
                 << " after compress:" << dstOffset << "\n";
    }          
  );
}

void TgFcKernel::selectTilePolicy() {
  if (try_tiling_group_parallel()) {
  } else if (try_no_tiling()) {
  } else if (try_tiling_parallel_mn()) {
  } else if (try_tiling_parallel_kn()) {
  } else if (try_tiling_no_parallel()) {
  } else {
    llvm::errs() << llvm::format("Tilling FC failed, M:%d,K:%d,N:%d, fmt:%d\n",
                                 M, K, N, fmt);
    assert(0);
  }
  total_steps = tiles.size();
  if (need_compress) {
    compress_weight();
  }
  set_laddr();
  if (compressed_pos->empty() == false) {
    assert(batch * slice_n() * slice_k() == compressed_pos->size());
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
  if (do_quant_bf16) {
    ctx.lmem_init_matrix(&tl_scale, ctx.ml_default_shape(1, tile.n, fmt), fmt,
                         1);
    ctx.lmem_init_matrix(&tl_zeropoint, ctx.ml_default_shape(1, tile.n, fmt),
                         fmt, 1);
    tl_scale.start_address = Q_laddr[tile.RB_idx];
    tl_zeropoint.start_address = Q_laddr[2 + tile.RB_idx];
  }
  if (mode == FC_GROUP_PARALLEL) {
    update_batch_info(tile.batch_high, tile.batch_low);
  }
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

void TgFcKernel::matrix_to_tensor(cvk_tl_t *tensor, const cvk_ml_t &matrix) {
  cvk_tl_shape_t shape = {matrix.shape.n, matrix.shape.c, 1, matrix.shape.w};
  ctx.lmem_init_tensor(tensor, shape, fmt, 1);
  tensor->start_address = matrix.start_address;
}

void TgFcKernel::quant_bf16(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  if (do_quant_bf16 == false) {
    return;
  }
  if (mode == FC_PARALLEL_MN && tile.pos_m != 0) {
    return;
  }
  cvk_tl_t scale_tensor, zeropoint_tensor, filter_tensor;
  matrix_to_tensor(&scale_tensor, tl_scale);
  matrix_to_tensor(&zeropoint_tensor, tl_zeropoint);
  matrix_to_tensor(&filter_tensor, tl_R);
  scale_tensor.shape.n = tl_R.shape.n;
  zeropoint_tensor.shape.n = tl_R.shape.n;
  scale_tensor.stride.n = 0;
  zeropoint_tensor.stride.n = 0;
  cvk_tiu_mul_param_t p = {0};
  p.res_high = nullptr;
  p.res_low = &filter_tensor;
  p.a = &filter_tensor;
  p.b_is_const = 0;
  p.b = &scale_tensor;
  p.rshift_bits = 0;
  p.layer_id = layer_id;
  p.relu_enable = 0;
  ctx.tiu_mul(&p);

  cvk_tiu_add_param_t p1 = {0};
  p1.res_high = nullptr;
  p1.res_low = &filter_tensor;
  p1.a_high = nullptr;
  p1.a_low = &filter_tensor;
  p1.b_is_const = false;
  p1.b.high = nullptr;
  p1.b.low = &zeropoint_tensor;
  p1.rshift_bits = 0;
  p1.layer_id = layer_id;
  p1.relu_enable = 0;
  ctx.tiu_add(&p1);
}

void TgFcKernel::compute(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  update_tl_matrix(step_idx);
  quant_bf16(step_idx);

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

void TgFcKernel::load_L(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  if ((mode == FC_PARALLEL_KN || mode == FC_GROUP_PARALLEL) &&
      tile.pos_n != 0) {
    return;
  }

  ctx.tdma_load_stride(
      &tl_L, ga_input + tile.pos_m * left_gstride.row + tile.pos_k * fmt_size,
      left_gstride);
}

void TgFcKernel::load_R(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  if (mode == FC_PARALLEL_MN && tile.pos_m != 0) {
    return;
  }
  if (do_quant_bf16) {
    int row_stride = N * (rstride ? batch_low : 1);
    cvk_mg_t mg_src = {0};
    mg_src.start_address = ga_weight + tile.pos_k * row_stride + tile.pos_n;
    mg_src.base_reg_index =
        ctx.getTdmaBaseSelectIndexFromGaddr(mg_src.start_address);
    mg_src.fmt = CVK_FMT_I8;
    mg_src.shape = {tile.k, tile.n};
    mg_src.stride = {N};
    cvk_tdma_g2l_matrix_copy_param_t param = {0};
    param.src = &mg_src;
    param.dst = &tl_R;
    param.layer_id = layer_id;
    ctx.tdma_g2l_matrix_copy(&param);
  } else if (compressed_pos->empty()) {
    ctx.tdma_load_stride(&tl_R,
                         ga_weight + tile.pos_k * right_gstride.row +
                             tile.pos_n * fmt_size,
                         right_gstride);
  } else {
    cvi_backend_ml_load_stride(
        ctx, layer_id,
        ga_weight + (*compressed_pos)[tile.compress_idx + compress_offset],
        tl_R.start_address, tile.k, tile.n, tile.n, false, true, fmt, fmt,
        true);
  }
}

void TgFcKernel::load_B(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  if (do_bias == false || bias_loaded == true) {
    return;
  }
  if (is_last_k(step_idx) == false) {
    return;
  }
  ctx.tdma_load_stride(&tl_B, ga_bias + tile.pos_n * fmt_size, {N * fmt_size});
  if (slice_n() == 1 && mode != FC_GROUP_PARALLEL) {
    bias_loaded = true;
  }
}

void TgFcKernel::load_Q(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  if (do_quant_bf16 == false || quant_loaded == true) {
    return;
  }
  if (mode == FC_PARALLEL_MN && tile.pos_m != 0) {
    return;
  }
  ctx.tdma_load_stride(&tl_scale, ga_scale + tile.pos_n * fmt_size,
                       {N * fmt_size});
  ctx.tdma_load_stride(&tl_zeropoint, ga_zeropoint + tile.pos_n * fmt_size,
                       {N * fmt_size});
  if (slice_n() == 1) {
    quant_loaded = true;
  }
}

void TgFcKernel::load(int32_t step_idx) {
  update_tl_matrix(step_idx);
  // load L
  load_L(step_idx);
  // load R
  load_R(step_idx);
  // load B
  load_B(step_idx);
  // load quant
  load_Q(step_idx);
}

void TgFcKernel::store(int32_t step_idx) {
  if (false == is_last_k(step_idx)) {
    return;
  }
  auto &tile = tiles[step_idx];
  update_tl_matrix(step_idx);
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
  if (compressed_pos->empty()) {
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
  case FC_PARALLEL_KN:
  case FC_PARALLEL_MN:
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
    std::vector<int> multiplier, bool need_compress,
    const std::vector<uint8_t> *weight, std::vector<int> *compr_weight_poss,
    std::vector<uint8_t> *compr_weight, int batch_high, int batch_low,
    bool lstride, bool rstride, bool ostride) {
  TgFcKernel kernel(ctx);
  kernel.init(layer_id, ga_input, ga_weight, ga_bias, ga_output, M, K, N,
              do_bias, do_relu, &rshift_width, &multiplier, batch_high,
              batch_low, lstride, rstride, ostride, need_compress, weight,
              compr_weight_poss, compr_weight, CVK_FMT_I8);

  kernel.selectTilePolicy();
  kernel.schedule();
}

void cvi_backend_tg_bf16_fc_kernel(
    const CviBackendContext &ctx, uint32_t layer_id, gaddr_t ga_input,
    gaddr_t ga_weight, gaddr_t ga_bias, gaddr_t ga_output, int M, int K, int N,
    bool do_bias, bool do_relu, bool need_compress,
    const std::vector<uint8_t> *weight, std::vector<int> *compr_weight_poss,
    std::vector<uint8_t> *compr_weight, int batch_high, int batch_low,
    bool lstride, bool rstride, bool ostride, bool do_quant_bf16,
    gaddr_t ga_scale, gaddr_t ga_zeropoint) {
  TgFcKernel kernel(ctx);
  kernel.init(layer_id, ga_input, ga_weight, ga_bias, ga_output, M, K, N,
              do_bias, do_relu, nullptr, nullptr, batch_high, batch_low,
              lstride, rstride, ostride, need_compress, weight,
              compr_weight_poss, compr_weight, CVK_FMT_BF16, do_quant_bf16,
              ga_scale, ga_zeropoint);

  kernel.selectTilePolicy();
  kernel.schedule();
}
