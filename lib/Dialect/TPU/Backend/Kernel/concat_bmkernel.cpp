/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: concat_bmkernel.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include <cmath>


//#include <bmnet/support/Debug.h>
//#include <bmnet/support/Format.h>
//#include <bmnet/targets/Target.hpp>

#define ASSERT(x) assert(x)

#define DEBUG_TYPE "bmnet_bm1880v2_bmkernel_concat"
#define DEBUG_SPLIT "bmnet_bm1880v2_bmkernel_concat_split"

static int concat_size_lmem(const CviBackendContext &ctx,
                            const cvk_tg_shape_t p) {
  return tensor_size_lmem(ctx, p.n, p.c, p.h, p.w);
}

static int split_concat_forward(const CviBackendContext &ctx,
                                const cvk_tg_shape_t _p, int *step_n, int *step_c,
                                int *step_h) {
  int target_size = NPU_NUM * LOCAL_MEM_SIZE;
  int npu_num = NPU_NUM;
  cvk_tg_shape_t p = _p;

  *step_n = p.n;
  *step_c = p.c;
  *step_h = p.h;
  // c < 0x1000 is hw constraint
  int max_c = 0x1000;
  if (concat_size_lmem(ctx, p) <= target_size && *step_c < max_c) {
    return 0;
  }

  if (p.n > 1) {
    *step_n = p.n = 1;
    int size = concat_size_lmem(ctx, p);
    if (size <= target_size) {
      *step_n = target_size / size;
      return 0;
    }
  }

  if (p.c > (uint32_t)npu_num) {
    *step_c = p.c = npu_num;
    int size = concat_size_lmem(ctx, p);
    if (size <= target_size) {
      // FIXME: check boundry
      int step = 1;
      do {
        *step_c = (target_size / size / step) * npu_num;
        step++;
      }
      while (*step_c >= max_c);

      return 0;
    }
  }

  *step_h = p.h = 1;
  int size = concat_size_lmem(ctx, p);
  if (size <= target_size) {
    *step_h = target_size / size;
    return 0;
  }

  return -1;
}

void cvi_backend_tg_fixed_concat_kernel(
    const CviBackendContext &ctx, uint32_t stream_id, uint32_t inst_id,
    uint32_t layer_id, const uint32_t *depends, uint32_t depends_len,
    gaddr_t input_gaddrs[], gaddr_t output_gaddr, int input_dims[],
    int input_num, int concat_axis, int output_dim_size, int *output_dim,
    bool do_relu, const int need_quantize_num, const int *right_shift_width,
    const int *threshold_x_quantized) {
  LLVM_DEBUG(llvm::errs() << llvm::format(
                  "cvi_backend_tg_fixed_concat_kernel:\n"
                  "    output_gaddr 0x%lx\n"
                  "    input_num %d, concat_axis %d, output_dim_size %d, need_quantize_num %d\n",
                  output_gaddr, input_num, concat_axis, output_dim_size, need_quantize_num););

  cvk_tg_stride_t stride_dst;
  cvk_tg_stride_t stride_src;
  cvk_tg_shape_t shape_;
  uint64_t offset = 0;

  if (need_quantize_num == 0 && false == do_relu) {
    if (concat_axis == 0) {
      ASSERT(output_dim_size == 4);
      stride_dst = {static_cast<uint32_t>(output_dim[1]) * output_dim[2] * output_dim[3],
                    static_cast<uint32_t>(output_dim[2]) * output_dim[3],
                    static_cast<uint32_t>(output_dim[3])};
      // Concat N.
      for (int i = 0; i < input_num; i++) {
        stride_src = {static_cast<uint32_t>(output_dim[1]) * output_dim[2] * output_dim[3],
                      static_cast<uint32_t>(output_dim[2]) * output_dim[3],
                      static_cast<uint32_t>(output_dim[3])};
        shape_ = {static_cast<uint32_t>(input_dims[i]), static_cast<uint32_t>(output_dim[1]),
                  static_cast<uint32_t>(output_dim[2]), static_cast<uint32_t>(output_dim[3])};

        cvk_tg_t src;
        src.start_address = input_gaddrs[i];
        src.fmt = CVK_FMT_U8;
        src.shape = shape_;
        src.stride = stride_src;

        cvk_tg_t dst;
        dst.start_address = output_gaddr + offset;
        dst.fmt = CVK_FMT_U8;
        dst.shape = shape_;
        dst.stride = stride_dst;

        LLVM_DEBUG(llvm::errs() << llvm::format(
                        "    [%d] 1 tdma_tg_copy:\n"
                        "         src addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d)\n"
                        "         dst addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d)\n",
                        i, src.start_address, src.shape.n, src.shape.c, src.shape.h, src.shape.w,
                        src.stride.n, src.stride.c, src.stride.h, dst.start_address, dst.shape.n,
                        dst.shape.c, dst.shape.h, dst.shape.w, dst.stride.n, dst.stride.c,
                        dst.stride.h));

        ctx.tdma_tg_copy(&dst, &src);

        offset += input_dims[i] * output_dim[1] * output_dim[2] * output_dim[3] * sizeof(uint8_t);
      }
    } else if (concat_axis == 1) {
      // Concat C.
      int* _output_dim = output_dim; // keep for restore
      int output_dims[4];
      memcpy(&output_dims, output_dim, output_dim_size * sizeof(int));
      output_dim = &output_dims[0];

      switch (output_dim_size) {
        case 3:
          // we extend from <n, c, h> dim to <n, c, h, 1> to leverage concat with c
          output_dim[3] = 1; // extend w to 1
          /* fall through */

        case 4:

          stride_dst = {static_cast<uint32_t>(output_dim[1]) * output_dim[2] * output_dim[3],
                        static_cast<uint32_t>(output_dim[2]) * output_dim[3],
                        static_cast<uint32_t>(output_dim[3])};
          for (int i = 0; i < input_num; i++) {
            if (input_gaddrs[i] != GA_INVALID) {
              stride_src = {static_cast<uint32_t>(input_dims[i]) * output_dim[2] * output_dim[3],
                            static_cast<uint32_t>(output_dim[2]) * output_dim[3],
                            static_cast<uint32_t>(output_dim[3])};
              shape_ = {static_cast<uint32_t>(output_dim[0]), static_cast<uint32_t>(input_dims[i]),
                        static_cast<uint32_t>(output_dim[2]), static_cast<uint32_t>(output_dim[3])};

              cvk_tg_t src;
              src.start_address = input_gaddrs[i];
              src.fmt = CVK_FMT_U8;
              src.shape = shape_;
              src.stride = stride_src;

              cvk_tg_t dst;
              dst.start_address = output_gaddr + offset;
              dst.fmt = CVK_FMT_U8;
              dst.shape = shape_;
              dst.stride = stride_dst;

              LLVM_DEBUG(
                  llvm::errs() << llvm::format(
                      "    [%d] 2 tdma_tg_copy:\n"
                      "         src addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d)\n"
                      "         dst addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d)\n",
                      i, src.start_address, src.shape.n, src.shape.c, src.shape.h, src.shape.w,
                      src.stride.n, src.stride.c, src.stride.h, dst.start_address, dst.shape.n,
                      dst.shape.c, dst.shape.h, dst.shape.w, dst.stride.n, dst.stride.c,
                      dst.stride.h));

              ctx.tdma_tg_copy(&dst, &src);
            }
            offset += input_dims[i] * output_dim[2] * output_dim[3] * sizeof(uint8_t);
          }
          break;
        case 2:
          // shape n,c,1,1
          stride_dst = {static_cast<uint32_t>(output_dim[1]), 1, 1};
          for (int i = 0; i < input_num; i++) {
            stride_src = {static_cast<uint32_t>(input_dims[i]), 1, 1};
            if (input_dims[i] < 65536) {
              shape_ = {static_cast<uint32_t>(output_dim[0]), static_cast<uint32_t>(input_dims[i]), 1, 1};

              cvk_tg_t src;
              src.start_address = input_gaddrs[i];
              src.fmt = CVK_FMT_U8;
              src.shape = shape_;
              src.stride = stride_src;

              cvk_tg_t dst;
              dst.start_address = output_gaddr + offset;
              dst.fmt = CVK_FMT_U8;
              dst.shape = shape_;
              dst.stride = stride_dst;

              LLVM_DEBUG(
                  llvm::errs() << llvm::format(
                      "    [%d] 3 tdma_tg_copy:\n"
                      "         src addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d)\n"
                      "         dst addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d)\n",
                      i, src.start_address, src.shape.n, src.shape.c, src.shape.h, src.shape.w,
                      src.stride.n, src.stride.c, src.stride.h, dst.start_address, dst.shape.n,
                      dst.shape.c, dst.shape.h, dst.shape.w, dst.stride.n, dst.stride.c,
                      dst.stride.h));

              ctx.tdma_tg_copy(&dst, &src);
              offset += input_dims[i] * sizeof(uint8_t);
            } else {
              // We need slice the C.
              int c_slice = (input_dims[i] + 65534) / 65535;
              uint64_t soffset = 0;
              for (int j = 0; j < c_slice - 1; j++) {
                shape_ = {static_cast<uint32_t>(output_dim[0]), 65535, 1, 1};

                cvk_tg_t src;
                src.start_address = input_gaddrs[i] + soffset;
                src.fmt = CVK_FMT_U8;
                src.shape = shape_;
                src.stride = stride_src;

                cvk_tg_t dst;
                dst.start_address = output_gaddr + offset;
                dst.fmt = CVK_FMT_U8;
                dst.shape = shape_;
                dst.stride = stride_dst;

                LLVM_DEBUG(
                    llvm::errs() << llvm::format(
                        "    [%d] 4 tdma_tg_copy:\n"
                        "         src addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d)\n"
                        "         dst addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d)\n",
                        i, src.start_address, src.shape.n, src.shape.c, src.shape.h, src.shape.w,
                        src.stride.n, src.stride.c, src.stride.h, dst.start_address, dst.shape.n,
                        dst.shape.c, dst.shape.h, dst.shape.w, dst.stride.n, dst.stride.c,
                        dst.stride.h));

                ctx.tdma_tg_copy(&dst, &src);

                offset += 65535 * sizeof(uint8_t);
                soffset += 65535 * sizeof(uint8_t);
              }
              if (input_dims[i] % 65535 != 0) {
                shape_ = {static_cast<uint32_t>(output_dim[0]), static_cast<uint32_t>(input_dims[i]) % 65535,
                          1, 1};

                cvk_tg_t src;
                src.start_address = input_gaddrs[i] + soffset;
                src.fmt = CVK_FMT_U8;
                src.shape = shape_;
                src.stride = stride_src;

                cvk_tg_t dst;
                dst.start_address = output_gaddr + offset;
                dst.fmt = CVK_FMT_U8;
                dst.shape = shape_;
                dst.stride = stride_dst;

                LLVM_DEBUG(
                    llvm::errs() << llvm::format(
                        "    [%d] 5 tdma_tg_copy:\n"
                        "         src addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d)\n"
                        "         dst addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d)\n",
                        i, src.start_address, src.shape.n, src.shape.c, src.shape.h, src.shape.w,
                        src.stride.n, src.stride.c, src.stride.h, dst.start_address, dst.shape.n,
                        dst.shape.c, dst.shape.h, dst.shape.w, dst.stride.n, dst.stride.c,
                        dst.stride.h));

                ctx.tdma_tg_copy(&dst, &src);

                offset += (input_dims[i] % 65535) * sizeof(uint8_t);
              }
            }
          }  // End of for loop
          break;
        default:
          LLVM_DEBUG(llvm::errs() << "concat can't support this shape"
                                << "\n");
          ASSERT(0);
          break;
      }
      output_dim = _output_dim;
    } else if (concat_axis == 2) {
      if (output_dim_size == 4) {
        ASSERT(output_dim[3] == 1);
      } else {
        ASSERT(output_dim_size == 3);
      }
      // shape n,c,h,1
      stride_dst = {static_cast<uint32_t>(output_dim[1]) * output_dim[2],
                    static_cast<uint32_t>(output_dim[2]), 1};
      for (int i = 0; i < input_num; i++) {
        ASSERT(input_dims[i] < 65536);
        stride_src = {static_cast<uint32_t>(output_dim[1]) * input_dims[i],
                      static_cast<uint32_t>(input_dims[i]), 1};
        shape_ = {static_cast<uint32_t>(output_dim[0]), static_cast<uint32_t>(output_dim[1]),
                  static_cast<uint32_t>(input_dims[i]), 1};

        cvk_tg_t src;
        src.start_address = input_gaddrs[i];
        src.fmt = CVK_FMT_U8;
        src.shape = shape_;
        src.stride = stride_src;

        cvk_tg_t dst;
        dst.start_address = output_gaddr + offset;
        dst.fmt = CVK_FMT_U8;
        dst.shape = shape_;
        dst.stride = stride_dst;

        LLVM_DEBUG(llvm::errs() << llvm::format(
                        "    [%d] 6 tdma_tg_copy:\n"
                        "         src addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d)\n"
                        "         dst addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d)\n",
                        i, src.start_address, src.shape.n, src.shape.c, src.shape.h, src.shape.w,
                        src.stride.n, src.stride.c, src.stride.h, dst.start_address, dst.shape.n,
                        dst.shape.c, dst.shape.h, dst.shape.w, dst.stride.n, dst.stride.c,
                        dst.stride.h));

        ctx.tdma_tg_copy(&dst, &src);
        offset += input_dims[i] * sizeof(uint8_t);
      }
    } else if(concat_axis == 3) {
      // concat w
      assert(output_dim_size == 4);
      stride_dst = {static_cast<uint32_t>(output_dim[1] * output_dim[2] * output_dim[3]),
                    static_cast<uint32_t>(output_dim[2] * output_dim[3]),
                    static_cast<uint32_t>(output_dim[3])};
      shape_ = {static_cast<uint32_t>(output_dim[0]), static_cast<uint32_t>(output_dim[1]),
                static_cast<uint32_t>(output_dim[2]), 1};
      for (int i = 0; i < input_num; i++) {
        ASSERT(input_dims[i] < 65536);
        shape_.w = static_cast<uint32_t>(input_dims[i]);
        stride_src = ctx.tg_default_stride(shape_, CVK_FMT_U8);
        cvk_tg_t src;
        src.start_address = input_gaddrs[i];
        src.fmt = CVK_FMT_U8;
        src.shape = shape_;
        src.stride = stride_src;
        cvk_tg_t dst;
        dst.start_address = output_gaddr + offset;
        dst.fmt = CVK_FMT_U8;
        dst.shape = shape_;
        dst.stride = stride_dst;
        ctx.tdma_tg_copy(&dst, &src);
        offset += input_dims[i] * sizeof(uint8_t);
      }

    } else {
      LLVM_DEBUG(llvm::errs() << "concat can't support this concat_axis"
                            << "\n");
      ASSERT(0);
    }
  } else {
    cvk_fmt_t fmt = CVK_FMT_I8;
    gaddr_t top_data = output_gaddr;
    int input_nchw[4] = {1, 1, 1, 1}; // nchw
    int _output_dim[4] = {1, 1, 1, 1}; // nchw
    memcpy(_output_dim, output_dim, sizeof(int) * output_dim_size);
    memcpy(input_nchw, _output_dim, sizeof(int) * output_dim_size);

    // check is it need tiling
    int is_need_tiled = 0;
    int eu_align = 0; // pack load/store

    for (int i = 0; i < input_num; ++i) {
      const int bottom_concat_axis = input_dims[i];
      input_nchw[concat_axis] = bottom_concat_axis;
      cvk_tl_shape_t if_shape = {
        static_cast<uint32_t>(input_nchw[0]), static_cast<uint32_t>(input_nchw[1]),
        static_cast<uint32_t>(input_nchw[2]), static_cast<uint32_t>(input_nchw[3])};
      int required_size = ctx.lmem_tensor_to_size(if_shape, fmt, eu_align);
      if (required_size > LOCAL_MEM_SIZE) {
        is_need_tiled = 1;
        break;
      }

      if (if_shape.n < 0x1000 &&
          if_shape.c < 0x1000 &&
          if_shape.h <= (4095-32) &&
          if_shape.w <= (4095-32)) {
        // valid setting
      }
      else {
        is_need_tiled = 1;
        break;
      }
    }

    if (!is_need_tiled) {
      // TODO: experiment code that merge all axis case, consider with tiling
      // get copy count
      int num_concats_ = 1;
      for (int i = 0; i < concat_axis; ++i) {
        num_concats_ *= _output_dim[i];
      }

      int concat_input_size_ = 1;
      for (int i = concat_axis + 1; i < output_dim_size; ++i) {
        concat_input_size_ *= _output_dim[i];
      }

      int offset_concat_axis = 0;
      for (int i = 0; i < input_num; ++i) {
        const gaddr_t bottom_data = input_gaddrs[i];
        const int bottom_concat_axis = input_dims[i];
        //for (int n = 0; n < num_concats_; ++n) {
        //  caffe_copy(bottom_concat_axis * concat_input_size_,
        //      bottom_data + n * bottom_concat_axis * concat_input_size_,
        //      top_data + (n * top_concat_axis + offset_concat_axis)
        //      * concat_input_size_);
        //}
        //offset_concat_axis += bottom_concat_axis;

        // copy data from ddr to local for multiply rshift

        input_nchw[concat_axis] = bottom_concat_axis;
        cvk_tl_t _ifmap;
        cvk_tl_t* ifmap = &_ifmap;
        tdma_g2l_tensor_copy(
            ctx,
            &ifmap,
            input_nchw[0], input_nchw[1], input_nchw[2], input_nchw[3],
            bottom_data, fmt, eu_align);

        // apply quantize int 8 mode
        apply_qi8(
            ctx,
            ifmap,
            layer_id,
            do_relu ? 1 : 0,
            need_quantize_num > 0 ? right_shift_width[i] : 0,
            need_quantize_num > 0 ? threshold_x_quantized[i] : 1);

        cvk_tg_stride_t stride_dst = {
          static_cast<uint32_t>(_output_dim[1]) * _output_dim[2] * _output_dim[3],
          static_cast<uint32_t>(_output_dim[2]) * _output_dim[3],
          static_cast<uint32_t>(_output_dim[3])};

        ctx.tdma_store_stride(ifmap, top_data + offset_concat_axis, stride_dst);

        ctx.lmem_free_tensor(ifmap);
        offset_concat_axis += bottom_concat_axis * concat_input_size_;
      }
      // success concat, directlt return
      return;
    }

    ASSERT(need_quantize_num == 0 || need_quantize_num == input_num);
    int fixed_output_dims[4] = {0};
    if (concat_axis == 3 && output_dim_size == 4) {
      // convert to 2 dims
      concat_axis = 1;
      output_dim_size = 2;
      fixed_output_dims[0] = output_dim[0] * output_dim[1] * output_dim[2];
      fixed_output_dims[1] = output_dim[3];
      output_dim = &fixed_output_dims[0];
    }
    if (concat_axis == 1) {
      // Concat C.
      int* _output_dim = output_dim; // keep for restore
      int output_dims[4];
      memcpy(&output_dims, output_dim, output_dim_size * sizeof(int));
      output_dim = &output_dims[0];

      switch (output_dim_size) {
        case 2:
          {
            int _input_dim[] = {1, 1, 1, 1};
            cvk_fmt_t fmt = CVK_FMT_I8;
            uint64_t ofmap_offset = output_gaddr;
            LLVM_DEBUG(llvm::errs() << "concat dim size = 2 case"
                            << "\n");
            // we could reshape <h, w> to <h, ?, ?, 1>, reshape c first for avoid channel limitation > 0x1000
            for (int n = 0; n < output_dim[0]; n++) {
              // it could directly concat with channel(concat_axis = 1)
              for (int i = 0; i < input_num; i++) {

                // channel c slicing is first priority
                _input_dim[1] = std::__gcd(input_dims[i], NPU_NUM);
                assert(_input_dim[1] && "slicing c fail, change __gcd strategy");

                int residual = input_dims[i] / _input_dim[1];

                // try to slice w align EU_NUM
                _input_dim[3] = std::__gcd(residual, EU_NUM);
                if (_input_dim[3] != 1) {
                  _input_dim[2] = residual / _input_dim[3];
                  if (_input_dim[3] < EU_NUM) {
                      _input_dim[2] = std::__gcd(residual, EU_NUM);
                      _input_dim[3] = residual / _input_dim[2];
                  }
                }
                else {
                  // cant slice w, e.g: gcd(17, EU_NUM). we direct set h = 1, w = 17
                  _input_dim[2] = 1;
                  _input_dim[3] = residual;
                }

                LLVM_DEBUG(llvm::errs() << llvm::format(
                  "input[%d] nchw is %d %d %d %d:\n",
                  i, _input_dim[0], _input_dim[1], _input_dim[2], _input_dim[3]));

                // retry tiling
                int require_shape = shape_size(1, _input_dim[1], _input_dim[2], _input_dim[3], fmt);
                std::vector<std::pair<cvk_tl_shape_t, gaddr_t> > tiling_info;
                tiling_packing(ctx, require_shape, 0, 1, fmt, &tiling_info);

                offset = 0;
                int n_offset = shape_size(n, _input_dim[1], _input_dim[2], _input_dim[3], fmt);
                for (uint64_t j = 0; j < tiling_info.size(); j++) {
                  int n = tiling_info[j].first.n;
                  int c = tiling_info[j].first.c;
                  int h = tiling_info[j].first.h;
                  int w = tiling_info[j].first.w;
                  offset = tiling_info[j].second;

                  uint64_t ifmap_offset = input_gaddrs[i] + n_offset + offset;

                  LLVM_DEBUG(llvm::errs() << llvm::format(
                      "slice input[%d] ifoff %lu nchw is %d %d %d %d:, "
                      " ofmap_offset %lu, n_offset %d, right_shift_width %d"
                      " threshold_x_quantized %d\n",
                      i, ifmap_offset, n, c, h, w,
                      ofmap_offset, n_offset,
                      right_shift_width[i],
                      threshold_x_quantized[i]));


                  // copy data from ddr to local for multiply rshift
                  cvk_tl_t _ifmap;
                  cvk_tl_t* ifmap = &_ifmap;
                  tdma_g2l_tensor_copy(
                      ctx,
                      &ifmap,
                      n, c, h, w,
                      ifmap_offset, fmt, eu_align);

                  // apply quantize int 8 mode
                  apply_qi8(
                      ctx,
                      ifmap,
                      layer_id,
                      do_relu ? 1 : 0,
                      need_quantize_num > 0 ? right_shift_width[i] : 0,
                      need_quantize_num > 0 ? threshold_x_quantized[i] : 1);

                  ctx.tdma_store(ifmap, ofmap_offset);
                  ctx.lmem_free_tensor(ifmap);
                  ofmap_offset += shape_size(n, c, h, w, fmt);
                }
              }
            }
          }
          break;

        case 3:
          // we extend from <n, c, h> dim to <n, c, h, 1> to concat with c
          output_dim[3] = 1; // extend w to 1
          /* fall through */

        case 4:
          stride_dst = {static_cast<uint32_t>(output_dim[1]) * output_dim[2] * output_dim[3],
                        static_cast<uint32_t>(output_dim[2]) * output_dim[3],
                        static_cast<uint32_t>(output_dim[3])};
          for (int i = 0; i < input_num; i++) {
            if (input_gaddrs[i] != GA_INVALID) {
              LLVM_DEBUG(llvm::errs() << "concat threshold_x_quantized=" << threshold_x_quantized[i]
                                    << ",right_shift_width=" << right_shift_width[i] << "\n");

              cvk_tg_shape_t if_shape = {
                  static_cast<uint32_t>(output_dim[0]), static_cast<uint32_t>(input_dims[i]),
                  static_cast<uint32_t>(output_dim[2]), static_cast<uint32_t>(output_dim[3])};
              int step_n = output_dim[0], step_c = input_dims[i], step_h = output_dim[2];
              int err = split_concat_forward(ctx, if_shape, &step_n, &step_c, &step_h);
              //LLVM_DEBUG_WITH_TYPE(
              //    DEBUG_SPLIT, std::cout << llvm::format(
              //                     "[Concat::split], n_slices = %d,c_slices = %d, h_slices=%d\n",
              //                     step_n, step_c, step_h));
              ASSERT(err == 0);

              for (int n_pos = 0; n_pos < output_dim[0]; n_pos += step_n) {
                int cur_n = std::min(output_dim[0] - n_pos, step_n);
                for (int c_pos = 0; c_pos < input_dims[i]; c_pos += step_c) {
                  int cur_c = std::min(step_c, input_dims[i] - c_pos);
                  for (int h_pos = 0; h_pos < output_dim[2]; h_pos += step_h) {
                    int cur_h = std::min(step_h, output_dim[2] - h_pos);

                    uint64_t ifmap_offset =
                        input_gaddrs[i] +
                        (n_pos * input_dims[i] * output_dim[2] * output_dim[3] +
                         c_pos * output_dim[2] * output_dim[3] + h_pos * output_dim[3]) *
                            sizeof(uint8_t);

                    cvk_tl_t *ifmap = ctx.lmem_alloc_tensor(
                        ctx.shape_t4(cur_n, cur_c, cur_h, output_dim[3]), CVK_FMT_I8, 1);  // EU-aligned

                    stride_src = {static_cast<uint32_t>(input_dims[i]) * output_dim[2] * output_dim[3],
                                  static_cast<uint32_t>(output_dim[2]) * output_dim[3],
                                  static_cast<uint32_t>(output_dim[3])};

                    ctx.tdma_load_stride(ifmap, ifmap_offset, stride_src);

                    cvk_tiu_mul_param_t p = {0};
                    p.res_high = nullptr;
                    p.res_low = ifmap;
                    p.a = ifmap;
                    p.b_const.val = need_quantize_num > 0 ? threshold_x_quantized[i] : 1;
                    p.b_const.is_signed = false;
                    p.b_is_const = 1;
                    p.rshift_bits = need_quantize_num > 0 ? right_shift_width[i] : 0;
                    p.layer_id = layer_id;
                    p.relu_enable = do_relu ? 1 : 0;
                    ctx.tiu_mul(&p);

                    uint64_t ofmap_offset =
                        output_gaddr + offset +
                        (n_pos * output_dim[1] * output_dim[2] * output_dim[3] +
                         c_pos * output_dim[2] * output_dim[3] + h_pos * output_dim[3]) *
                            sizeof(uint8_t);
                    cvk_tg_stride_t stride_dst = {
                        static_cast<uint32_t>(output_dim[1]) * output_dim[2] * output_dim[3],
                        static_cast<uint32_t>(output_dim[2]) * output_dim[3],
                        static_cast<uint32_t>(output_dim[3])};

                    ctx.tdma_store_stride(ifmap, ofmap_offset, stride_dst);

                    ctx.lmem_free_tensor(ifmap);
                  }
                }
              }
            }
            offset += input_dims[i] * output_dim[2] * output_dim[3] * sizeof(uint8_t);
          }
          break;
        default:
          LLVM_DEBUG(llvm::errs() << "concat can't support this shape"
                                << "\n");
          ASSERT(0);
          break;
      }
      output_dim = _output_dim;
    } else {
      LLVM_DEBUG(llvm::errs() << "concat can't support this concat_axis"
                            << "\n");
      ASSERT(0);
    }
  }

  LLVM_DEBUG(llvm::errs() << "<= cvi_backend_tg_fixed_concat_kernel" << "\n");
}
//}  // namespace bmnet
