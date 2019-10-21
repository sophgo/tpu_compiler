/*
 * Copyright (C) Bitmain Technologies Inc.
 * All Rights Reserved.
 */
//#include <support/Debug.h>
//#include <support/Format.h>
//#include <targets/plat-bm188x/BM1880v2BackendContext.hpp>
#include <bmkernel/bm_kernel.h>
#include <bmkernel/bm_kernel_legacy.h>
#include "BM1880v2BackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>

#define DEBUG_TYPE "bmnet_bm1880v2_dma_legacy"
#define DEBUG_BMNET(x) LLVM_DEBUG(x)
#define LOG(x) llvm::errs()

// namespace bmnet {

//
// Implement 1880 gdma_load_stride, tensor format
//
// Note:
//   1880 bmk keeps eu-aligned info and calculate stride info.
//   1880v2 bmk does not keep eu-aligned info, use need to calculate stride info.
//
//<! FIXME: move to LIR
bool BM1880v2BackendContext::is_compress(int n, int c, int h, int w, int stride_n, int stride_c,
                                         int stride_h, u64 ga_src, ctrl_t ctrl) const {
  u64 offset = ga_src;
  int len = n * c * h * w * INT8_SIZE;
  bool is_compress = false;
  bool is_continue = stride_n == c * h * w && stride_c == h * w && stride_h == w;
  bool is_align_eu = !(ga_src % 0x10);  //<! hw bus width align
  //bool is_cic_check = !this->get_weight_optimized();
  static std::vector<u64> compressed_ga;  //<! FIXME: not declare here
  bool is_compressed =
      std::find(compressed_ga.begin(), compressed_ga.end(), offset) != compressed_ga.end();

  //if (is_cic_check) {
  //  llvm::errs() << "offset (" << offset << ") large than weight_size_(" << weight_size_ << "), 81";
  //  return false;
  //}

  if (is_compressed) {
    llvm::errs() << "offset/len  (" << offset << "/" << len << ") has compressed, 81";
    return true;
  }

  if (is_continue && is_align_eu) {
    int8_t *weight = reinterpret_cast<int8_t *>(weight_);
    int is_signed = 1;
    int compress_md = 0;
    compress_addr_info compressed_data;
    u8 *res = compress(reinterpret_cast<u8 *>(&weight[offset]), len, compress_md, is_signed,
                       &compressed_data);

    if (compressed_data.total_size >= len) {
      llvm::errs() << "compressed sz(" << compressed_data.total_size << ") large than before(" << len
                << "), 81";
      return false;
    }

    // update original buffer to compressed
    memcpy((void *)(&weight[offset]), (void *)res, compressed_data.total_size);

    is_compress = true;
    compressed_ga.push_back(offset);

    float compress_per = ((len - compressed_data.total_size) / static_cast<float>(len)) * 100;
    llvm::errs() << "compress percentage:" << compress_per << "%";
  }
  return is_compress;
}

void BM1880v2BackendContext::tdma_load_stride(bmk1880v2_tensor_lmem_t *tlp, u64 ga_src,
                                              bmk1880v2_tensor_tgmem_stride_t ts_stride,
                                              ctrl_t ctrl) const {
  assert(tlp != nullptr);

  bool DoTranspose = (ctrl & CTRL_TP) ? true : false;
  bool isNeuron = (ctrl & CTRL_NEURON) ? true : false;

  // tensor in system memory
  //
  // Constraint:
  //   assert_tl_tg_same_size()
  //   Global_N == Local_N
  //
  // 1. Global channel != local channel
  //    Eg.
  //     alexnet: src (, 256, 1, 1), dst (2, 128, 1, 1)
  //
  // 2. Global shape != local shape
  //    Eg.
  //     alexnet conv5 relu
  //     src (, 384, 13, 13), dst (1, 384, 8, 13)

  // tensor in system memory
  // Global shape use local shape
  bmk1880v2_tensor_tgmem_t ts_data;
  ts_data.base_reg_index =
      isNeuron ? BM1880v2BackendContext::NEURON_MEMORY : BM1880v2BackendContext::WEIGHT_MEMORY;
  ts_data.fmt = tlp->fmt;
  ts_data.start_address = ga_src;
  ts_data.shape = {tlp->shape.n, tlp->shape.c, tlp->shape.h, tlp->shape.w};
  ts_data.stride = ts_stride;

  if (DoTranspose) {
    bmk1880v2_tdma_tg2l_tensor_copy_nc_transposed_param_t p1;
    p1.src = &ts_data;
    p1.dst = tlp;
    this->tdma_g2l_tensor_copy_nc_transposed(&p1);
  } else {
    bmk1880v2_tdma_tg2l_tensor_copy_param_t p1;
    p1.src = &ts_data;
    p1.dst = tlp;

    //<! FIXME: also apply in neuron
    if (!isNeuron && is_compress(tlp->shape.n, tlp->shape.c, tlp->shape.h, tlp->shape.w,
                                 ts_stride.n, ts_stride.c, ts_stride.h, ga_src, ctrl)) {
      bmk1880v2_tdma_tg2l_tensor_copy_decompressed_param_t p2;
      bmk1880v2_compressed_tensor_tgmem_t ts_data1;
      ts_data1.base_reg_index = ts_data.base_reg_index;
      ts_data1.start_address = ts_data.start_address;
      ts_data1.shape = ts_data.shape;
      ts_data1.stride = ts_data.stride;
      ts_data1.bit_length = 8;  // 1 byte

      p2.src = &ts_data1;
      p2.dst = tlp;
      this->tdma_g2l_tensor_copy_decompressed(&p2);
    } else {
      this->tdma_g2l_tensor_copy(&p1);
    }
  }
}

//
// Implement 1880 gdma_load, tensor format
//
// Note:
//   1880 bmk keeps eu-aligned info and calculate stride info.
//   1880v2 bmk does not keep eu-aligned info, use need to calculate stride info.
//
void BM1880v2BackendContext::tdma_load(bmk1880v2_tensor_lmem_t *tlp, u64 ga_src,
                                       ctrl_t ctrl) const {
  assert(tlp != nullptr);

  bool DoTranspose = (ctrl & CTRL_TP) ? true : false;
  bool isNeuron = (ctrl & CTRL_NEURON) ? true : false;

  // Global memory shape == local memory shape
  // Gobal memory stride from local memory shape
  bmk1880v2_tensor_tgmem_t ts_data;
  ts_data.base_reg_index =
      isNeuron ? BM1880v2BackendContext::NEURON_MEMORY : BM1880v2BackendContext::WEIGHT_MEMORY;
  ts_data.fmt = tlp->fmt;
  ts_data.start_address = ga_src;
  ts_data.shape = {tlp->shape.n, tlp->shape.c, tlp->shape.h, tlp->shape.w};
  ts_data.stride = bmk1880v2_tensor_tgmem_default_stride(ts_data.shape);

  if (DoTranspose) {
    bmk1880v2_tdma_tg2l_tensor_copy_nc_transposed_param_t p1;
    p1.src = &ts_data;
    p1.dst = tlp;
    this->tdma_g2l_tensor_copy_nc_transposed(&p1);
  } else {
    bmk1880v2_tdma_tg2l_tensor_copy_param_t p1;
    p1.src = &ts_data;
    p1.dst = tlp;
    this->tdma_g2l_tensor_copy(&p1);
  }
}

//
// Implement 1880 gdma_store_stride, tensor format
//
// Note:
//   1880 bmk keeps eu-aligned info and calculate stride info.
//   1880v2 bmk does not keep eu-aligned info, use need to calculate stride info.
//
void BM1880v2BackendContext::tdma_store_stride(bmk1880v2_tensor_lmem_t *tlp, u64 ga_dst,
                                               bmk1880v2_tensor_tgmem_stride_t ts_stride,
                                               ctrl_t ctrl) const {
  assert(tlp != nullptr);

  bool DoTranspose = (ctrl & CTRL_TP) ? true : false;
  bool isNeuron = (ctrl & CTRL_NEURON) ? true : false;

  // tensor in system memory
  // Global shape use local shape
  // Global shape used for stride calculation
  bmk1880v2_tensor_tgmem_t ts_data;
  ts_data.base_reg_index =
      isNeuron ? BM1880v2BackendContext::NEURON_MEMORY : BM1880v2BackendContext::WEIGHT_MEMORY;
  ts_data.fmt = tlp->fmt;
  ts_data.start_address = ga_dst;
  ts_data.shape = {tlp->shape.n, tlp->shape.c, tlp->shape.h, tlp->shape.w};
  ts_data.stride = ts_stride;

  if (DoTranspose) {
    bmk1880v2_tdma_l2tg_tensor_copy_nc_transposed_param_t p1;
    p1.src = tlp;
    p1.dst = &ts_data;
    this->tdma_l2g_tensor_copy_nc_transposed(&p1);
  } else {
    bmk1880v2_tdma_l2tg_tensor_copy_param_t p1;
    p1.src = tlp;
    p1.dst = &ts_data;
    this->tdma_l2g_tensor_copy(&p1);
  }
}

//
// Implement 1880 gdma_store, tensor format
//
// Note:
//   1880 bmk keeps eu-aligned info and calculate stride info.
//   1880v2 bmk does not keep eu-aligned info, use need to calculate stride info.
//
void BM1880v2BackendContext::tdma_store(bmk1880v2_tensor_lmem_t *tlp, u64 ga_dst,
                                        ctrl_t ctrl) const {
  assert(tlp != nullptr);

  bool DoTranspose = (ctrl & CTRL_TP) ? true : false;
  bool isNeuron = (ctrl & CTRL_NEURON) ? true : false;

  // tensor in system memory
  // Global shape use local shape
  // Gobal memory stride from local memory shape
  bmk1880v2_tensor_tgmem_t ts_data;
  ts_data.base_reg_index =
      isNeuron ? BM1880v2BackendContext::NEURON_MEMORY : BM1880v2BackendContext::WEIGHT_MEMORY;
  ts_data.fmt = tlp->fmt;
  ts_data.start_address = ga_dst;
  ts_data.shape = {tlp->shape.n, tlp->shape.c, tlp->shape.h, tlp->shape.w};
  ts_data.stride = bmk1880v2_tensor_tgmem_default_stride(ts_data.shape);

  if (DoTranspose) {
    bmk1880v2_tdma_l2tg_tensor_copy_nc_transposed_param_t p1;
    p1.src = tlp;
    p1.dst = &ts_data;
    this->tdma_l2g_tensor_copy_nc_transposed(&p1);
  } else {
    bmk1880v2_tdma_l2tg_tensor_copy_param_t p1;
    p1.src = tlp;
    p1.dst = &ts_data;
    this->tdma_l2g_tensor_copy(&p1);
  }
}

//
// Implement 1880 gdma_tg_copy, tensor format
//   No split
//
//   Condition:
//     size <= local LANE size
//
bool BM1880v2BackendContext::_tdma_tg_copy_no_split(bmk1880v2_tensor_tgmem_t *tg_dst,
                                                    bmk1880v2_tensor_tgmem_t *tg_src) const {
  bmk1880v2_tensor_lmem_t tl_data;
  tl_data.start_address = 0;  // Start of LMEM
  tl_data.fmt = FMT_I8;       // don't care
  tl_data.shape = {tg_src->shape.n, tg_src->shape.c, tg_src->shape.h, tg_src->shape.w};
  tl_data.stride = this->tensor_lmem_default_stride(tl_data.shape, /*eu_aligned=*/0);

  // Check required lane size
  u32 required_lane_size = tl_data.shape.n * tl_data.stride.n;
  if (required_lane_size > this->hw.local_mem_size) {
    return false;
  }

  // G2L
  bmk1880v2_tdma_tg2l_tensor_copy_param_t p1;
  p1.src = tg_src;
  p1.dst = &tl_data;
  this->tdma_g2l_tensor_copy(&p1);

  // L2G
  bmk1880v2_tdma_l2tg_tensor_copy_param_t p2;
  p2.src = &tl_data;
  p2.dst = tg_dst;
  this->tdma_l2g_tensor_copy(&p2);

  return true;
}

//
// Implement 1880 gdma_tg_copy, tensor format
//   Split along N, H axis.
//
//   Condition:
//     src shape == dst shape.
//     H*W > one lane size.
//     C close to lane number.
//
bool BM1880v2BackendContext::_tdma_tg_copy_split_nh(bmk1880v2_tensor_tgmem_t *tg_dst,
                                                    bmk1880v2_tensor_tgmem_t *tg_src) const {
  u64 input_gaddr = tg_src->start_address;
  u32 input_n = tg_src->shape.n;
  u32 input_c = tg_src->shape.c;
  u32 input_h = tg_src->shape.h;
  u32 input_w = tg_src->shape.w;
  u64 output_gaddr = tg_dst->start_address;
  u32 output_n = tg_dst->shape.n;
  u32 output_c = tg_dst->shape.c;
  u32 output_h = tg_dst->shape.h;
  u32 output_w = tg_dst->shape.w;

  DEBUG_BMNET(llvm::errs() << llvm::format("_tg_copy_split_nh:\n"
                                        "    src shape (%d, %d, %d, %d), stride (%d, %d, %d)\n"
                                        "    dst shape (%d, %d, %d, %d), stride (%d, %d, %d)\n",
                                        tg_src->shape.n, tg_src->shape.c, tg_src->shape.h,
                                        tg_src->shape.w, tg_src->stride.n, tg_src->stride.c,
                                        tg_src->stride.h, tg_dst->shape.n, tg_dst->shape.c,
                                        tg_dst->shape.h, tg_dst->shape.w, tg_dst->stride.n,
                                        tg_dst->stride.c, tg_dst->stride.h));

  // Shape must be equal
  if (input_n != output_n || input_c != output_c || input_h != output_h || input_w != output_w) {
    DEBUG_BMNET(llvm::errs() << "<= _tg_copy_split_nh, shape not equal" << "\n");
    return false;
  }

  // Split along H based on LMEM size
  u32 lmem_size = this->hw.local_mem_size * this->hw.npu_num;
  u32 chw_size = input_c * input_h * input_w;
  u32 max_tiled_h = (chw_size <= lmem_size) ? input_h : lmem_size / input_c / input_w;

  // Find max tiled_h to fit local memory
  do {
    bmk1880v2_tensor_lmem_shape_t shape = {1, input_c, max_tiled_h, input_w};
    bmk1880v2_tensor_lmem_stride_t stride =
        this->tensor_lmem_default_stride(shape, /*eu_aligned=*/0);
    u32 required_lane_size = stride.n;

    if (required_lane_size <= this->hw.local_mem_size) {
      break;
    }
  } while (--max_tiled_h);

  // Something wrong !
  if (!max_tiled_h) {
    DEBUG_BMNET(llvm::errs() << "<= _tg_copy_split_nh, unable to split h" << "\n");
    return false;
  }

  // Split along N axis
  for (u32 i = 0; i < input_n; i++) {
    // Split along H axis
    for (u32 h_offset = 0; h_offset < input_h; h_offset += max_tiled_h) {
      u32 tiled_h = (h_offset + max_tiled_h) <= input_h ? max_tiled_h : (input_h - h_offset);

      // Calculate offset with stride(original shape), position (ni, 0, h_offset, 0)
      u64 input_offset = i * tg_src->stride.n + h_offset * tg_src->stride.h;
      u64 output_offset = i * tg_dst->stride.n + h_offset * tg_dst->stride.h;

      DEBUG_BMNET(
          llvm::errs() << llvm::format("    [%d] h_offset %d, src tiled shape(1, %d, %d, %d)\n", i,
                                    h_offset, input_c, tiled_h, input_w));

      // G2L tensor copy
      {
        // Tiled shape and original stride
        bmk1880v2_tensor_tgmem_t src;
        src.base_reg_index = tg_src->base_reg_index;
        src.start_address = input_gaddr + input_offset;
        src.fmt = FMT_I8;  // don't care
        src.shape = {1, input_c, tiled_h, input_w};
        src.stride = tg_src->stride;

        // Follow source shape
        bmk1880v2_tensor_lmem_t dst;
        dst.start_address = 0;  // start of lmem
        dst.fmt = src.fmt;
        dst.shape = {1, input_c, tiled_h, input_w};
        dst.stride = this->tensor_lmem_default_stride(dst.shape, /*eu_aligned=*/0);

        bmk1880v2_tdma_tg2l_tensor_copy_param_t p1;
        p1.src = &src;
        p1.dst = &dst;

        DEBUG_BMNET(llvm::errs() << llvm::format(
                        "    [%d] G2L tensor copy: h_offset %d, tiled_h %d\n"
                        "         src addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d)\n"
                        "         dst addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n",
                        i, h_offset, tiled_h, p1.src->start_address, p1.src->shape.n,
                        p1.src->shape.c, p1.src->shape.h, p1.src->shape.w, p1.src->stride.n,
                        p1.src->stride.c, p1.src->stride.h, p1.dst->start_address, p1.dst->shape.n,
                        p1.dst->shape.c, p1.dst->shape.h, p1.dst->shape.w, p1.dst->stride.n,
                        p1.dst->stride.c, p1.dst->stride.h, p1.dst->stride.w));

        this->tdma_g2l_tensor_copy(&p1);
      }

      // L2G tensor copy
      {
        bmk1880v2_tensor_lmem_t src;
        src.start_address = 0;
        src.fmt = FMT_I8;  // don't care
        src.shape = {1, input_c, tiled_h, input_w};
        src.stride = this->tensor_lmem_default_stride(src.shape, /*eu_aligned=*/0);

        // tilted shape and original stride
        bmk1880v2_tensor_tgmem_t dst;
        dst.base_reg_index = tg_dst->base_reg_index;
        dst.start_address = output_gaddr + output_offset;
        dst.fmt = FMT_I8;  // don't care
        dst.shape = {1, output_c, tiled_h, output_w};
        dst.stride = tg_dst->stride;

        bmk1880v2_tdma_l2tg_tensor_copy_param_t p2;
        p2.src = &src;
        p2.dst = &dst;

        DEBUG_BMNET(llvm::errs() << llvm::format(
                        "    [%d] L2G: h_offset %d, tiled_h %d\n"
                        "         src addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d, %d)\n"
                        "         dst addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d)\n",
                        i, h_offset, tiled_h, p2.src->start_address, p2.src->shape.n,
                        p2.src->shape.c, p2.src->shape.h, p2.src->shape.w, p2.src->stride.n,
                        p2.src->stride.c, p2.src->stride.h, p2.src->stride.w, p2.dst->start_address,
                        p2.dst->shape.n, p2.dst->shape.c, p2.dst->shape.h, p2.dst->shape.w,
                        p2.dst->stride.n, p2.dst->stride.c, p2.dst->stride.h));

        this->tdma_l2g_tensor_copy(&p2);
      }

    }  // for (u32 h_offset = 0; h_offset < input_h; h_offset+=max_tiled_h)
  }    // for (u32 i = 0; i < input_n; i++

  DEBUG_BMNET(llvm::errs() << "<= _tg_copy_split_nh" << "\n");
  return true;
}

//
// Implement 1880 gdma_tg_copy, tensor format
//
void BM1880v2BackendContext::_tdma_tg_copy(bmk1880v2_tensor_tgmem_t *dst,
                                           bmk1880v2_tensor_tgmem_t *src) const {
  bmk1880v2_tensor_lmem_shape_t tl_shape = {src->shape.n, src->shape.c, src->shape.h, src->shape.w};
  bmk1880v2_tensor_lmem_stride_t tl_stride =
      this->tensor_lmem_default_stride(tl_shape, /*eu_aligned=*/0);
  u32 required_lane_size = tl_shape.n * tl_stride.n;
  bool result = false;

  // Case 1: total size <= TPU LMEM size, no need to split N or C
  result = this->_tdma_tg_copy_no_split(dst, src);
  if (result) {
    return;
  }

  // Case 2: split along N, H axis
  result = this->_tdma_tg_copy_split_nh(dst, src);
  if (result) {
    return;
  }

  // Shoul not reach here
  LOG(ERROR) << "tdma_tg_copy: fail to split";
}

//
// Implement 1880 gdma_tg_copy, tensor format
//
void BM1880v2BackendContext::tdma_tg_copy(bmk1880v2_tensor_tgmem_t *dst,
                                          bmk1880v2_tensor_tgmem_t *src, ctrl_t ctrl) const {
  assert(dst != nullptr && src != nullptr);

  bool DoTranspose = (ctrl & CTRL_TP) ? true : false;
  bool isNeuron = (ctrl & CTRL_NEURON) ? true : false;

  src->base_reg_index =
      isNeuron ? BM1880v2BackendContext::NEURON_MEMORY : BM1880v2BackendContext::WEIGHT_MEMORY;

  dst->base_reg_index =
      isNeuron ? BM1880v2BackendContext::NEURON_MEMORY : BM1880v2BackendContext::WEIGHT_MEMORY;

  if (DoTranspose) {
    // TDMA does not support G2G nc transpose

    // DO NOT know how to tile
    assert((src->shape.n * src->shape.c * src->shape.h * src->shape.w) <=
           (this->hw.local_mem_size * this->hw.npu_num));

    // 1. G2L tensor copy
    bmk1880v2_tensor_lmem_t tl_data;
    tl_data.start_address = 0;
    tl_data.fmt = src->fmt;
    tl_data.shape = {src->shape.n, src->shape.c, src->shape.h, src->shape.w};
    tl_data.stride = this->tensor_lmem_default_stride(tl_data.shape, /*eu_aligned=*/0);

    bmk1880v2_tdma_tg2l_tensor_copy_param_t p1;
    p1.src = src;
    p1.dst = &tl_data;
    this->tdma_g2l_tensor_copy(&p1);

    // 2. L2G nc transpose
    bmk1880v2_tdma_l2tg_tensor_copy_nc_transposed_param_t p2;
    p2.src = &tl_data;
    p2.dst = dst;
    this->tdma_l2g_tensor_copy_nc_transposed(&p2);
  } else {
    if (0) {
      LOG(INFO) << "use g2g";
      bmk1880v2_tdma_tg2tg_tensor_copy_param_t p;
      p.src = src;
      p.dst = dst;
      this->tdma_g2g_tensor_copy(&p);
      // return bmk1880v2_tdma_g2g_tensor_copy(bmk_, p);
    } else {
      this->_tdma_tg_copy(dst, src);
    }
  }
}

//
// Implement 1880 gdma_load_stride, matrix format
//
void BM1880v2BackendContext::tdma_load_stride(bmk1880v2_matrix_lmem_t *tlp, u64 ga_src,
                                              bmk1880v2_matrix_tgmem_stride_t ts_stride,
                                              ctrl_t ctrl) const {
  assert(tlp != nullptr);

  bool DoTranspose = (ctrl & CTRL_TP) ? true : false;
  bool isNeuron = (ctrl & CTRL_NEURON) ? true : false;

  // Global memory from reshaped local memory
  bmk1880v2_matrix_tgmem_t ts_data;
  ts_data.base_reg_index =
      isNeuron ? BM1880v2BackendContext::NEURON_MEMORY : BM1880v2BackendContext::WEIGHT_MEMORY;
  ts_data.start_address = ga_src;
  ts_data.stride = ts_stride;

  if (DoTranspose) {
    ts_data.shape = {tlp->shape.col, tlp->shape.n};  // Explicit transpose shape !!!

    bmk1880v2_tdma_tg2l_matrix_copy_row_col_transposed_param_t p1;
    p1.src = &ts_data;
    p1.dst = tlp;

    DEBUG_BMNET(llvm::errs() << llvm::format(
                    "tdma_load_stride(matrix): src (%d, %d), dst(n=%d, c=%d, w=%d,col= %d)\n",
                    p1.src->shape.row, p1.src->shape.col, p1.dst->shape.n, p1.dst->shape.c,
                    p1.dst->shape.w, p1.dst->shape.col));

    this->tdma_g2l_matrix_copy_row_col_transposed(&p1);
  } else {
    ts_data.shape = {tlp->shape.n, tlp->shape.col};

    bmk1880v2_tdma_tg2l_matrix_copy_param_t p1;
    p1.src = &ts_data;
    p1.dst = tlp;

    // if (is_compress(1, 1, p1.src->shape.row, p1.src->shape.col,
    //      1, 1, ts_stride.row,
    //      ga_src, ctrl)){
    //  //TODO matrix to tensor
    //  this->tdma_g2l_matrix_copy(&p1);
    //}
    // else
    { this->tdma_g2l_matrix_copy(&p1); }
  }
}

//
// Implement 1880 gdma_load, matrix format
//
void BM1880v2BackendContext::tdma_load(bmk1880v2_matrix_lmem_t *tlp, u64 ga_src,
                                       ctrl_t ctrl) const {
  assert(tlp != nullptr);

  bool DoTranspose = (ctrl & CTRL_TP) ? true : false;
  bool isNeuron = (ctrl & CTRL_NEURON) ? true : false;

  // Global memory from reshaped local memory
  bmk1880v2_matrix_tgmem_t ts_data;
  ts_data.base_reg_index =
      isNeuron ? BM1880v2BackendContext::NEURON_MEMORY : BM1880v2BackendContext::WEIGHT_MEMORY;
  ts_data.start_address = ga_src;
  ts_data.shape = {tlp->shape.n, tlp->shape.col};
  ts_data.stride = {tlp->shape.col};

  if (DoTranspose) {
    ts_data.shape = {tlp->shape.col, tlp->shape.n};  // Explicit transpose shape !!!

    bmk1880v2_tdma_tg2l_matrix_copy_row_col_transposed_param_t p1;
    p1.src = &ts_data;
    p1.dst = tlp;
    this->tdma_g2l_matrix_copy_row_col_transposed(&p1);
  } else {
    bmk1880v2_tdma_tg2l_matrix_copy_param_t p1;
    p1.src = &ts_data;
    p1.dst = tlp;
    this->tdma_g2l_matrix_copy(&p1);
  }
}

//
// Implement 1880 gdma_store, matrix format
//
void BM1880v2BackendContext::tdma_store(bmk1880v2_matrix_lmem_t *tlp, u64 ga_dst,
                                        ctrl_t ctrl) const {
  bool DoTranspose = (ctrl & CTRL_TP) ? true : false;
  bool isNeuron = (ctrl & CTRL_NEURON) ? true : false;

  assert(DoTranspose == false);

  // tensor in system memory
  // Global shape use local shape
  // Gobal memory stride from local memory shape
  bmk1880v2_matrix_tgmem_t ts_data;
  ts_data.base_reg_index =
      isNeuron ? BM1880v2BackendContext::NEURON_MEMORY : BM1880v2BackendContext::WEIGHT_MEMORY;
  ts_data.start_address = ga_dst;
  ts_data.shape = {tlp->shape.n, tlp->shape.col};
  ts_data.stride = {tlp->shape.col};

  bmk1880v2_tdma_l2tg_matrix_copy_param_t p1;
  p1.src = tlp;
  p1.dst = &ts_data;
  this->tdma_l2g_matrix_copy(&p1);
}

//
// Implement 1880 gdma_store_stride, matrix format
//
void BM1880v2BackendContext::tdma_store_stride(bmk1880v2_matrix_lmem_t *tlp, u64 ga_dst,
                                               bmk1880v2_matrix_tgmem_stride_t ts_stride,
                                               ctrl_t ctrl) const {
  bool DoTranspose = (ctrl & CTRL_TP) ? true : false;
  bool isNeuron = (ctrl & CTRL_NEURON) ? true : false;

  assert(DoTranspose == false);

  // tensor in system memory
  // Global shape use local shape
  // Global shape used for stride calculation
  bmk1880v2_matrix_tgmem_t ts_data;
  ts_data.base_reg_index =
      isNeuron ? BM1880v2BackendContext::NEURON_MEMORY : BM1880v2BackendContext::WEIGHT_MEMORY;
  ts_data.start_address = ga_dst;
  ts_data.shape = {tlp->shape.n, tlp->shape.col};
  ts_data.stride = ts_stride;

  bmk1880v2_tdma_l2tg_matrix_copy_param_t p1;
  p1.src = tlp;
  p1.dst = &ts_data;
  this->tdma_l2g_matrix_copy(&p1);
}

//}  // namespace bmnet
