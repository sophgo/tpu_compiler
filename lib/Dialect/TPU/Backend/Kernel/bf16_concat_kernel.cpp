/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: bf16_concat_kernel.cpp
 * Description:
 */

#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include <cmath>

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
  if (concat_size_lmem(ctx, p) <= target_size) {
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
      *step_c = (target_size / size) * npu_num;
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

void cvi_backend_tg_bf16_concat_kernel(const CviBackendContext &ctx, uint32_t stream_id,
                                         uint32_t inst_id, uint32_t layer_id, const uint32_t *depends,
                                         uint32_t depends_len, gaddr_t input_gaddrs[],
                                         gaddr_t output_gaddr, int input_dims[], int input_num,
                                         int concat_axis, int output_dim_size, int *output_dim,
                                         bool do_relu, const int need_quantize_num,
                                         const int *threshold_x_quantized) {
  LLVM_DEBUG(llvm::errs() << llvm::format(
                  "cvi_backend_tg_bf16_concat_kernel:\n"
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
      stride_dst = {static_cast<uint32_t>(output_dim[1]) * output_dim[2] * output_dim[3] * 2,
                    static_cast<uint32_t>(output_dim[2]) * output_dim[3] * 2,
                    static_cast<uint32_t>(output_dim[3]) * 2};
      // Concat N.
      for (int i = 0; i < input_num; i++) {
        stride_src = {static_cast<uint32_t>(output_dim[1]) * output_dim[2] * output_dim[3] * 2,
                      static_cast<uint32_t>(output_dim[2]) * output_dim[3] * 2,
                      static_cast<uint32_t>(output_dim[3]) * 2};
        shape_ = {static_cast<uint32_t>(input_dims[i]), static_cast<uint32_t>(output_dim[1]),
                  static_cast<uint32_t>(output_dim[2]), static_cast<uint32_t>(output_dim[3])};

        cvk_tg_t src;
        src.start_address = input_gaddrs[i];
        src.fmt = CVK_FMT_BF16;
        src.shape = shape_;
        src.stride = stride_src;

        cvk_tg_t dst;
        dst.start_address = output_gaddr + offset;
        dst.fmt = CVK_FMT_BF16;
        dst.shape = shape_;
        dst.stride = stride_dst;

        LLVM_DEBUG(llvm::errs() << llvm::format(
                        "    [%d] 1 g2g:\n"
                        "         src addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d)\n"
                        "         dst addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d)\n",
                        i, src.start_address, src.shape.n, src.shape.c, src.shape.h, src.shape.w,
                        src.stride.n, src.stride.c, src.stride.h, dst.start_address, dst.shape.n,
                        dst.shape.c, dst.shape.h, dst.shape.w, dst.stride.n, dst.stride.c,
                        dst.stride.h));
        cvk_tdma_g2g_tensor_copy_param_t p = {0};
        p.src = &src;
        p.dst = &dst;
        ctx.tdma_g2g_bf16_tensor_copy(&p);

        offset += input_dims[i] * output_dim[1] * output_dim[2] * output_dim[3] * sizeof(uint16_t);
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
          stride_dst = {static_cast<uint32_t>(output_dim[1]) * output_dim[2] * output_dim[3] * 2,
                        static_cast<uint32_t>(output_dim[2]) * output_dim[3] * 2,
                        static_cast<uint32_t>(output_dim[3]) * 2};
         LLVM_DEBUG(llvm::errs() << llvm::format("output_dim n= %d c=%d h=%d w=%d\n", output_dim[0], output_dim[1],output_dim[2],output_dim[3]););
         LLVM_DEBUG(llvm::errs() << llvm::format("input_num = %d\n", input_num););
          for (int i = 0; i < input_num; i++) {
            if (input_gaddrs[i] != GA_INVALID) {
              stride_src = {static_cast<uint32_t>(input_dims[i]) * output_dim[2] * output_dim[3] * 2,
                            static_cast<uint32_t>(output_dim[2]) * output_dim[3] * 2,
                            static_cast<uint32_t>(output_dim[3]) * 2};
              shape_ = {static_cast<uint32_t>(output_dim[0]), static_cast<uint32_t>(input_dims[i]),
                        static_cast<uint32_t>(output_dim[2]), static_cast<uint32_t>(output_dim[3])};

              cvk_tg_t src;
              src.start_address = input_gaddrs[i];
              src.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(src.start_address);
              src.fmt = CVK_FMT_BF16;
              src.shape = shape_;
              src.stride = stride_src;

              cvk_tg_t dst;
              dst.start_address = output_gaddr + offset;
              dst.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(dst.start_address);
              dst.fmt = CVK_FMT_BF16;
              dst.shape = shape_;
              dst.stride = stride_dst;

              LLVM_DEBUG(
                  llvm::errs() << llvm::format(
                      "    [%d] 2 g2g:\n"
                      "         src addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d)\n"
                      "         dst addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d)\n",
                      i, src.start_address, src.shape.n, src.shape.c, src.shape.h, src.shape.w,
                      src.stride.n, src.stride.c, src.stride.h, dst.start_address, dst.shape.n,
                      dst.shape.c, dst.shape.h, dst.shape.w, dst.stride.n, dst.stride.c,
                      dst.stride.h);
              );
              cvk_tdma_g2g_tensor_copy_param_t p = {0};
              p.src = &src;
              p.dst = &dst;
              ctx.tdma_g2g_bf16_tensor_copy(&p);
            }
            offset += input_dims[i] * output_dim[2] * output_dim[3] * sizeof(uint16_t);
          }
          break;
        case 2:
          //assert(0);
          // shape n,c,1,1
          stride_dst = {static_cast<uint32_t>(output_dim[1]) * 2, 1 * 2, 1 * 2};
          for (int i = 0; i < input_num; i++) {
            stride_src = {static_cast<uint32_t>(input_dims[i]) * 2, 1 * 2, 1 * 2};
            if (input_dims[i] < 65536) {
              shape_ = {static_cast<uint32_t>(output_dim[0]), static_cast<uint32_t>(input_dims[i]), 1, 1};

              cvk_tg_t src;
              src.start_address = input_gaddrs[i];
              src.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(src.start_address);
              src.fmt = CVK_FMT_BF16;
              src.shape = shape_;
              src.stride = stride_src;

              cvk_tg_t dst;
              dst.start_address = output_gaddr + offset;
              dst.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(dst.start_address);
              dst.fmt = CVK_FMT_BF16;
              dst.shape = shape_;
              dst.stride = stride_dst;

              LLVM_DEBUG(
                  llvm::errs() << llvm::format(
                      "    [%d] 3 g2g:\n"
                      "         src addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d)\n"
                      "         dst addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d)\n",
                      i, src.start_address, src.shape.n, src.shape.c, src.shape.h, src.shape.w,
                      src.stride.n, src.stride.c, src.stride.h, dst.start_address, dst.shape.n,
                      dst.shape.c, dst.shape.h, dst.shape.w, dst.stride.n, dst.stride.c,
                      dst.stride.h));
              cvk_tdma_g2g_tensor_copy_param_t p = {0};
              p.src = &src;
              p.dst = &dst;
              ctx.tdma_g2g_bf16_tensor_copy(&p);

              offset += input_dims[i] * sizeof(uint16_t);
            } else {
              //assert(0);
              // We need slice the C.
              int c_slice = (input_dims[i] + 65534) / 65535;
              uint64_t soffset = 0;
              for (int j = 0; j < c_slice - 1; j++) {
                shape_ = {static_cast<uint32_t>(output_dim[0]), 65535, 1, 1};

                cvk_tg_t src;
                src.start_address = input_gaddrs[i] + soffset;
                src.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(src.start_address);
                src.fmt = CVK_FMT_BF16;
                src.shape = shape_;
                src.stride = stride_src;

                cvk_tg_t dst;
                dst.start_address = output_gaddr + offset;
                dst.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(dst.start_address);
                dst.fmt = CVK_FMT_BF16;
                dst.shape = shape_;
                dst.stride = stride_dst;

                LLVM_DEBUG(
                    llvm::errs() << llvm::format(
                        "    [%d] 4 g2g:\n"
                        "         src addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d)\n"
                        "         dst addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d)\n",
                        i, src.start_address, src.shape.n, src.shape.c, src.shape.h, src.shape.w,
                        src.stride.n, src.stride.c, src.stride.h, dst.start_address, dst.shape.n,
                        dst.shape.c, dst.shape.h, dst.shape.w, dst.stride.n, dst.stride.c,
                        dst.stride.h));

                cvk_tdma_g2g_tensor_copy_param_t p = {0};
                p.src = &src;
                p.dst = &dst;
                ctx.tdma_g2g_bf16_tensor_copy(&p);

                offset += 65535 * sizeof(uint16_t);
                soffset += 65535 * sizeof(uint16_t);
              }
              if (input_dims[i] % 65535 != 0) {
                shape_ = {static_cast<uint32_t>(output_dim[0]), static_cast<uint32_t>(input_dims[i]) % 65535,
                          1, 1};

                cvk_tg_t src;
                src.start_address = input_gaddrs[i] + soffset;
                src.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(src.start_address);
                src.fmt = CVK_FMT_BF16;
                src.shape = shape_;
                src.stride = stride_src;

                cvk_tg_t dst;
                dst.start_address = output_gaddr + offset;
                dst.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(dst.start_address);
                dst.fmt = CVK_FMT_BF16;
                dst.shape = shape_;
                dst.stride = stride_dst;

                LLVM_DEBUG(
                    llvm::errs() << llvm::format(
                        "    [%d] 5 g2g:\n"
                        "         src addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d)\n"
                        "         dst addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d)\n",
                        i, src.start_address, src.shape.n, src.shape.c, src.shape.h, src.shape.w,
                        src.stride.n, src.stride.c, src.stride.h, dst.start_address, dst.shape.n,
                        dst.shape.c, dst.shape.h, dst.shape.w, dst.stride.n, dst.stride.c,
                        dst.stride.h));

                cvk_tdma_g2g_tensor_copy_param_t p = {0};
                p.src = &src;
                p.dst = &dst;
                ctx.tdma_g2g_bf16_tensor_copy(&p);

                offset += (input_dims[i] % 65535) * sizeof(uint16_t);
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
      stride_dst = {static_cast<uint32_t>(output_dim[1]) * output_dim[2] * 2,
                    static_cast<uint32_t>(output_dim[2]) * 2, 1 * 2};
      for (int i = 0; i < input_num; i++) {
        ASSERT(input_dims[i] < 65536);
        stride_src = {static_cast<uint32_t>(output_dim[1]) * input_dims[i] * 2,
                      static_cast<uint32_t>(input_dims[i]) * 2, 1 * 2};
        shape_ = {static_cast<uint32_t>(output_dim[0]), static_cast<uint32_t>(output_dim[1]),
                  static_cast<uint32_t>(input_dims[i]), 1};

        cvk_tg_t src;
        src.start_address = input_gaddrs[i];
        src.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(src.start_address);
        src.fmt = CVK_FMT_BF16;
        src.shape = shape_;
        src.stride = stride_src;

        cvk_tg_t dst;
        dst.start_address = output_gaddr + offset;
        dst.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(dst.start_address);
        dst.fmt = CVK_FMT_BF16;
        dst.shape = shape_;
        dst.stride = stride_dst;

        LLVM_DEBUG(llvm::errs() << llvm::format(
                        "    [%d] 6 g2g:\n"
                        "         src addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d)\n"
                        "         dst addr 0x%lx, shape(%d, %d, %d, %d), stride(%d, %d, %d)\n",
                        i, src.start_address, src.shape.n, src.shape.c, src.shape.h, src.shape.w,
                        src.stride.n, src.stride.c, src.stride.h, dst.start_address, dst.shape.n,
                        dst.shape.c, dst.shape.h, dst.shape.w, dst.stride.n, dst.stride.c,
                        dst.stride.h));
        cvk_tdma_g2g_tensor_copy_param_t p = {0};
        p.src = &src;
        p.dst = &dst;
        ctx.tdma_g2g_bf16_tensor_copy(&p);

        offset += input_dims[i] * sizeof(uint16_t);
      }
    }  else if(concat_axis == 3) {
      // concat w
      assert(output_dim_size == 4);
      stride_dst = {static_cast<uint32_t>(output_dim[1] * output_dim[2] * output_dim[3] * 2),
                    static_cast<uint32_t>(output_dim[2] * output_dim[3] * 2),
                    static_cast<uint32_t>(output_dim[3] * 2)};
      shape_ = {static_cast<uint32_t>(output_dim[0]), static_cast<uint32_t>(output_dim[1]),
                static_cast<uint32_t>(output_dim[2]), 1};
      for (int i = 0; i < input_num; i++) {
        ASSERT(input_dims[i] < 65536);
        shape_.w = static_cast<uint32_t>(input_dims[i]);
        stride_src = ctx.tg_default_stride(shape_, CVK_FMT_BF16);
        cvk_tg_t src;
        src.start_address = input_gaddrs[i];
        src.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(src.start_address);
        src.fmt = CVK_FMT_BF16;
        src.shape = shape_;
        src.stride = stride_src;

        cvk_tg_t dst;
        dst.start_address = output_gaddr + offset;
        dst.base_reg_index = ctx.getTdmaBaseSelectIndexFromGaddr(dst.start_address);
        dst.fmt = CVK_FMT_BF16;
        dst.shape = shape_;
        dst.stride = stride_dst;
        cvk_tdma_g2g_tensor_copy_param_t p = {0};
        p.src = &src;
        p.dst = &dst;
        ctx.tdma_g2g_bf16_tensor_copy(&p);
        offset += input_dims[i] * sizeof(uint16_t);
      }

    } else {
      LLVM_DEBUG(llvm::errs() << "concat can't support this concat_axis"  << "\n");
      ASSERT(0);
    }
  } else {
    ASSERT(need_quantize_num == 0 || need_quantize_num == input_num);
    if (concat_axis == 1) {
      // Concat C.
      switch (output_dim_size) {
        case 4:
          stride_dst = {static_cast<uint32_t>(output_dim[1]) * output_dim[2] * output_dim[3] * 2,
                        static_cast<uint32_t>(output_dim[2]) * output_dim[3] * 2,
                        static_cast<uint32_t>(output_dim[3]) * 2};
          for (int i = 0; i < input_num; i++) {
            if (input_gaddrs[i] != GA_INVALID) {
              LLVM_DEBUG(llvm::errs() << llvm::format("concat threshold_x_quantized= %d", threshold_x_quantized[i]););

              cvk_tg_shape_t if_shape = {
                  static_cast<uint32_t>(output_dim[0]), static_cast<uint32_t>(input_dims[i]),
                  static_cast<uint32_t>(output_dim[2]), static_cast<uint32_t>(output_dim[3])};
              int step_n = output_dim[0], step_c = input_dims[i], step_h = output_dim[2];
              int err = split_concat_forward(ctx, if_shape, &step_n, &step_c, &step_h);
//              LLVM_DEBUG_WITH_TYPE(
//                  DEBUG_SPLIT, llvm::errs() << llvm::format(
//                                   "[Concat::split], n_slices = %d,c_slices = %d, h_slices=%d\n",
//                                   step_n, step_c, step_h));
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
                            sizeof(uint16_t);

                    cvk_tl_t *ifmap = ctx.lmem_alloc_tensor(
                        ctx.shape_t4(cur_n, cur_c, cur_h, output_dim[3]), CVK_FMT_BF16, 1);  // EU-aligned

                    stride_src = {static_cast<uint32_t>(input_dims[i]) * output_dim[2] * output_dim[3] * 2,
                                  static_cast<uint32_t>(output_dim[2]) * output_dim[3] * 2,
                                  static_cast<uint32_t>(output_dim[3]) * 2};
                    if(1)
                      assert(0);
                    ctx.tdma_load_stride(ifmap, ifmap_offset, stride_src);

                    cvk_tiu_mul_param_t p = {0};
                    p.res_high = nullptr;
                    p.res_low = ifmap;
                    p.a = ifmap;
                    p.b_const.val = need_quantize_num > 0 ? threshold_x_quantized[i] : 1;
                    p.b_const.is_signed = false;
                    p.b_is_const = 1;
//                    p.rshift_bits = right_shift_width[i];
                    p.layer_id = layer_id;
                    p.relu_enable = do_relu ? 1 : 0;
                    ctx.tiu_mul(&p);

                    uint64_t ofmap_offset =
                        output_gaddr + offset +
                        (n_pos * output_dim[1] * output_dim[2] * output_dim[3] +
                         c_pos * output_dim[2] * output_dim[3] + h_pos * output_dim[3]) *
                            sizeof(uint16_t);
                    cvk_tg_stride_t stride_dst = {
                        static_cast<uint32_t>(output_dim[1]) * output_dim[2] * output_dim[3] * 2,
                        static_cast<uint32_t>(output_dim[2]) * output_dim[3] * 2,
                        static_cast<uint32_t>(output_dim[3]) * 2};

                    ctx.tdma_store_stride(ifmap, ofmap_offset, stride_dst);

                    ctx.lmem_free_tensor(ifmap);
                  }
                }
              }
            }
            offset += input_dims[i] * output_dim[2] * output_dim[3] * sizeof(uint16_t);
          }
          break;
        default:
          LLVM_DEBUG(llvm::errs() << "concat can't support this shape\n");
          ASSERT(0);
          break;
      }
    } else {
      LLVM_DEBUG(llvm::errs() << "concat can't support this concat_axis\n");
      ASSERT(0);
    }
  }

  LLVM_DEBUG(llvm::errs() << "<= cvi_backend_tg_fixed_concat_kernel");
}
