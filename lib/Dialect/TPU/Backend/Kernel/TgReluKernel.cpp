/*
 * Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
 *
 * File Name: TgReluKernel.cpp
 *
 * refined 2020-11-18
 */

#include "TgReluKernel.hpp"

#define DEBUG_TYPE "kernel_relu"

void TgReluKernel::init(uint32_t layer_id, int32_t n, int32_t c, int32_t h,
                        int32_t w, gaddr_t ga_input, gaddr_t ga_output,
                        gaddr_t ga_slope, float negative_slope, int GT_rshift,
                        int GT_scale, int LE_rshift, int LE_scale, int input_offset, int output_offset,
                        cvk_fmt_t fmt, mode_t mode) {
  this->layer_id = layer_id;
  this->n = n;
  this->c = c;
  this->h = h;
  this->w = w;
  this->ga_input = ga_input;
  this->ga_output = ga_output;
  this->ga_slope = ga_slope;
  this->negative_slope = negative_slope;
  this->GT_rshift = GT_rshift;
  this->GT_scale = GT_scale;
  this->LE_rshift = LE_rshift;
  this->LE_scale = LE_scale;
  this->input_offset = input_offset;
  this->output_offset = output_offset;
  this->is_asymmetric = (input_offset != 0 || output_offset != 0);
  this->fmt = fmt;
  this->mode = mode;
  gstride = ctx.tg_default_stride(c, h, w, fmt);

  ctx.set_layer_id(layer_id);
}

void TgReluKernel::selectTilePolicy() {
  // blob_num = 4, input/output with flip
  // in asymmetric, we need more 3 space calculating
  int blob_num = is_asymmetric ? 7 : 4;
  if (mode == PRELU) {
    auto slope_size = ctx.lmem_tensor_to_size(1, c, 1, 1, fmt, 1);
    ctx.tiling_packing(tiles, n, c, h, w, fmt, blob_num, slope_size,
                       CviBackendContext::TilingDimNHW, true);

  } else {
    ctx.tiling_packing(tiles, n, c, h, w, fmt, blob_num, 0,
                       CviBackendContext::TilingDimAll, true);
  }
}

void TgReluKernel::allocLmem() {
  if (mode == PRELU) {
    cvk_tl_shape_t slope_shape = ctx.tl_shape_t4(1, c, 1, 1);
    tl_slope = ctx.lmem_alloc_tensor(slope_shape, fmt, 1);
    ctx.tdma_load(tl_slope, ga_slope);
  }

  cvk_tl_shape_t tile_shape =
      ctx.tl_shape_t4(tiles[0].n, tiles[0].c, tiles[0].h, tiles[0].w);

  tl_input[0] = ctx.lmem_alloc_tensor(tile_shape, fmt, 1);
  tl_input[1] = ctx.lmem_alloc_tensor(tile_shape, fmt, 1);
  tl_output[0] = ctx.lmem_alloc_tensor(tile_shape, fmt, 1);
  tl_output[1] = ctx.lmem_alloc_tensor(tile_shape, fmt, 1);
  if(is_asymmetric){
    tl_pos_neg_map = ctx.lmem_alloc_tensor(tile_shape, fmt, 1);
    tl_working[0] = ctx.lmem_alloc_tensor(tile_shape, fmt, 1);
    tl_working[1] = ctx.lmem_alloc_tensor(tile_shape, fmt, 1);
  }
}

void TgReluKernel::deallocLmem() {
  if (is_asymmetric) {
    ctx.lmem_free_tensor(tl_working[1]);
    ctx.lmem_free_tensor(tl_working[0]);
    ctx.lmem_free_tensor(tl_pos_neg_map);
  }
  ctx.lmem_free_tensor(tl_output[1]);
  ctx.lmem_free_tensor(tl_output[0]);
  ctx.lmem_free_tensor(tl_input[1]);
  ctx.lmem_free_tensor(tl_input[0]);
  if (mode == PRELU) {
    ctx.lmem_free_tensor(tl_slope);
  }
}

cvk_tl_t TgReluKernel::get_input(int32_t step_idx, int32_t flip) {
  auto &tile = tiles[step_idx];
  auto tl_ifmap = *tl_input[flip];
  tl_ifmap.shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  tl_ifmap.stride = ctx.tl_default_stride(tl_ifmap.shape, fmt, 1);
  return tl_ifmap;
}

cvk_tl_t TgReluKernel::get_output(int32_t step_idx, int32_t flip) {
  auto &tile = tiles[step_idx];
  auto tl_ofmap = *tl_output[flip];
  tl_ofmap.shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  tl_ofmap.stride = ctx.tl_default_stride(tl_ofmap.shape, fmt, 1);
  return tl_ofmap;
}

void TgReluKernel::change_workspace_size(int32_t step_idx) {
  auto &tile = tiles[step_idx];
  for(size_t i = 0; i < 2; i++){
    tl_working[i]->shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
    tl_working[i]->stride = ctx.tl_default_stride(tl_working[i]->shape, fmt, 1);
  }
  tl_pos_neg_map->shape = ctx.tl_shape_t4(tile.n, tile.c, tile.h, tile.w);
  tl_pos_neg_map->stride = ctx.tl_default_stride(tl_pos_neg_map->shape, fmt, 1);
}

void TgReluKernel::load(int32_t step_idx, int32_t flip) {
  auto &tile = tiles[step_idx];
  auto tl_ifmap = get_input(step_idx, flip);
  if (mode == PRELU) {
    ctx.tdma_load_stride(&tl_ifmap, ga_input + tile.offset, gstride);
  } else {
    ctx.tdma_load(&tl_ifmap, ga_input + tile.offset);
  }
}

void TgReluKernel::compute_relu(int32_t step_idx, int32_t flip) {
  auto tl_ifmap = get_input(step_idx, flip);
  auto tl_ofmap = get_output(step_idx, flip);

  cvk_tiu_max_param_t p = {0};
  p.max = &tl_ofmap;
  p.a = &tl_ifmap;
  p.b_is_const = 1;
  p.layer_id = layer_id;

  if (fmt == CVK_FMT_BF16) {
    p.b_const.val = ctx.convert_fp32_to_bf16(0);
  } else {
    p.b_const.val = (0);
    if (fmt == CVK_FMT_I8) {
      p.b_const.is_signed = 1;
    } else if (fmt == CVK_FMT_U8) {
      p.b_const.is_signed = 0;
    } else {
      assert(0 && "fmt not supported");
    }
  }
  ctx.tiu_max(&p);
}
void TgReluKernel::reset(cvk_tl_t *tl) {
  cvk_tiu_mul_param_t p = {0};
  p.res_high = nullptr;
  p.res_low = tl;
  p.a = tl;
  p.b_const.val = 0;
  p.b_const.is_signed = 0;
  p.b_is_const = 1;
  p.rshift_bits = 0;
  p.layer_id = layer_id;
  p.relu_enable = 0;
  ctx.tiu_mul(&p);
}

/*
  LeakyRelu in asymmetric quantization
  There are two cases in leaky relu, postive and negative
  postive cases:
    Qy = Sx/Sy(Qx - Zx) + Qy
  negative cases:
    Qy = alpha * Sx/Sy * (Qx - Zx) + Qy

  in scale Sx/Sy, we make it to 2^rshift * mutlipiler(8bit)
  therefore,
    Qy = 2^rshift * multiplier * (Qx - Zx) + Qy

  output = ((Qx+offset)) * multiplier) >> rshift
*/

void TgReluKernel::compute_leaky_relu_fixed_asym(int32_t step_idx,
                                                int32_t flip) {
  auto tl_ifmap = get_input(step_idx, flip);
  auto tl_ofmap = get_output(step_idx, flip);
  change_workspace_size(step_idx);


  // Postive
  // Find (Qx - Zx) > 0 case
  // Find Qx > Zx
  cvk_tiu_max_param_t p_pos = {0};
  p_pos.max = &tl_ofmap;
  p_pos.a = &tl_ifmap;
  p_pos.b_is_const = 1;
  p_pos.b_const.is_signed = 1;
  p_pos.b_const.val = -input_offset;
  p_pos.layer_id = layer_id;
  ctx.tiu_max(&p_pos);

  // save a map keeping element is postive or negitave
  /* like [ 0 1 0 0
            1 0 0 1
            1 1 0 0 ]
      0 is negivate, 1 is postive
  */
  reset(tl_pos_neg_map);
  reset(tl_working[0]);
  reset(tl_working[1]);

  cvk_tiu_mac_param_t p_toi16 = {0};
  p_toi16.res_high = tl_working[1];
  p_toi16.res_low = tl_working[0];
  p_toi16.a = &tl_ofmap;
  p_toi16.res_is_int8 = false;
  p_toi16.b_const.val = 1;
  p_toi16.b_is_const = 1;
  p_toi16.b_const.is_signed = true;
  p_toi16.lshift_bits = 0;
  p_toi16.rshift_bits = 0;
  p_toi16.layer_id = layer_id;
  p_toi16.relu_enable = 0;
  ctx.tiu_mac(&p_toi16);

  cvk_tiu_add_param_t p_pos_neg_map = {0};
  p_pos_neg_map.res_high = tl_working[1];
  p_pos_neg_map.res_low = tl_pos_neg_map;
  p_pos_neg_map.a_high = tl_working[1];
  p_pos_neg_map.a_low = tl_working[0];
  p_pos_neg_map.b_is_const = 1;
  p_pos_neg_map.b_const.val = input_offset;
  p_pos_neg_map.b_const.is_signed = 1;
  p_pos_neg_map.rshift_bits = 0;
  p_pos_neg_map.layer_id = layer_id;
  p_pos_neg_map.relu_enable = 0;
  ctx.tiu_add(&p_pos_neg_map);

  cvk_tiu_mac_param_t p_toi8 = {0};
  p_toi8.res_high = tl_working[1];
  p_toi8.res_low = tl_pos_neg_map;
  p_toi8.a = tl_working[0];
  p_toi8.res_is_int8 = true;
  p_toi8.b_const.val = 0;
  p_toi8.b_is_const = 1;
  p_toi8.b_const.is_signed = true;
  p_toi8.lshift_bits = 0;
  p_toi8.rshift_bits = 0;
  p_toi8.layer_id = layer_id;
  p_toi8.relu_enable = 0;
  ctx.tiu_mac(&p_toi8);

  cvk_tiu_min_param_t p_maketo1 = {0};
  p_maketo1.min = tl_pos_neg_map;
  p_maketo1.a = tl_pos_neg_map;
  p_maketo1.b_is_const = 1;
  p_maketo1.b_const.is_signed = 1;
  p_maketo1.b_const.val = 1;
  p_maketo1.layer_id = layer_id;
  ctx.tiu_min(&p_maketo1);

  cvk_tiu_max_param_t p_maketo0 = {0};
  p_maketo0.max = tl_pos_neg_map;
  p_maketo0.a = tl_pos_neg_map;
  p_maketo0.b_is_const = 1;
  p_maketo0.b_const.is_signed = 1;
  p_maketo0.b_const.val = 0;
  p_maketo0.layer_id = layer_id;
  ctx.tiu_max(&p_maketo0);

  // Do Leaky Relu function
  reset(tl_working[0]);
  reset(tl_working[1]);

  // Do (Qx - Zx)(16bit)
  // make input from int8 to int16, use mac
  // put tmp result to tl_working space
  p_toi16.res_high = tl_working[1];
  p_toi16.res_low = tl_working[0];
  p_toi16.a = &tl_ofmap;
  p_toi16.res_is_int8 = false;
  p_toi16.b_const.val = 1;
  p_toi16.b_is_const = 1;
  p_toi16.b_const.is_signed = true;
  p_toi16.lshift_bits = 0;
  p_toi16.rshift_bits = 0;
  p_toi16.layer_id = layer_id;
  p_toi16.relu_enable = 0;
  ctx.tiu_mac(&p_toi16);

  reset(&tl_ofmap);
  // add input offset, and replace tl_working
  // (Qx-Zx) 16bit

  cvk_tiu_add_param_t p_offset = {0};
  p_offset.res_high = tl_working[1];
  p_offset.res_low = tl_working[0];
  p_offset.a_high = tl_working[1];
  p_offset.a_low = tl_working[0];
  p_offset.b_is_const = 1;
  p_offset.b_const.val = input_offset;
  p_offset.b_const.is_signed = 1;
  p_offset.rshift_bits = 0;
  p_offset.layer_id = layer_id;
  p_offset.relu_enable = 0;
  ctx.tiu_add(&p_offset);

  // This step:
  //    Sx/Sy(Qx-Zx)
  // we already make Sx/Sy = 2^rshift * multiplier(8bit)
  // rshift will be done at final, here we do
  // (Qx-Zx) * multiplier, beacause of (Qx-Zx) is 16bit
  // we mul high 8bit first, get Qx'_high
  cvk_tiu_mul_param_t p_high_scale = {0};
  p_high_scale.res_high = nullptr;
  p_high_scale.res_low = tl_working[1];
  p_high_scale.a = tl_working[1];
  p_high_scale.b_is_const = 1;
  p_high_scale.b_const.is_signed = 1;
  p_high_scale.b_const.val = GT_scale;
  p_high_scale.rshift_bits = 0;
  p_high_scale.layer_id = layer_id;
  p_high_scale.relu_enable = 0;
  ctx.tiu_mul(&p_high_scale);

  // then we do mac with low 8bit
  // and add Qx'_high with put in res_high( high 8bit)
  // ((Qx_low) * multiplier + (Qx'_high << 8)) >> rshift = ((Qx-Zx) * multiplier) >> rshift
  cvk_tiu_mac_param_t p_low_scale = {0};
  tl_working[0]->fmt = CVK_FMT_U8;
  p_low_scale.res_high = tl_working[1];
  p_low_scale.res_low = &tl_ofmap;
  p_low_scale.a = tl_working[0];
  p_low_scale.b_const.val = GT_scale;
  p_low_scale.b_is_const = 1;
  p_low_scale.b_const.is_signed = true;
  p_low_scale.lshift_bits = 0;
  p_low_scale.rshift_bits = GT_rshift;
  p_low_scale.layer_id = layer_id;
  p_low_scale.relu_enable = 0;
  ctx.tiu_mac(&p_low_scale);
  tl_working[0]->fmt = tl_ifmap.fmt;

  // reset to 0
  reset(tl_working[0]);

  // In next step, we use mac (1) * output_offset + ((Qx-Zx) * multiplier) >> rshift
  // Here we make 1 input
  cvk_tiu_add_param_t p9 = {0};
  p9.res_high = tl_working[1];
  p9.res_low = tl_working[0];
  p9.a_high = tl_working[1];
  p9.a_low = tl_working[0];
  p9.b_const.val = 1;
  p9.b_is_const = 1;
  p9.rshift_bits = 0;
  p9.relu_enable = 0;
  ctx.tiu_add(&p9);

  reset(tl_working[1]);
  // final add output offset,
  // Postive done
  // mac (1) * output_offset + ((Qx-Zx) * multiplier) >> rshift
  cvk_tiu_mac_param_t p_output_offset = {0};
  p_output_offset.res_high = tl_working[1];
  p_output_offset.res_low = &tl_ofmap;
  p_output_offset.a = tl_working[0];
  p_output_offset.res_is_int8 = true;
  p_output_offset.b_const.val = output_offset;
  p_output_offset.b_is_const = 1;
  p_output_offset.b_const.is_signed = true;
  p_output_offset.lshift_bits = 0;
  p_output_offset.rshift_bits = 0;
  p_output_offset.layer_id = layer_id;
  p_output_offset.relu_enable = 0;
  ctx.tiu_mac(&p_output_offset);

  reset(tl_working[0]);
  reset(tl_working[1]);

  // Negative, just scale different, all step as same as Postive
  cvk_tiu_min_param_t p5 = {0};
  p5.min = &tl_ifmap;
  p5.a = &tl_ifmap;
  p5.b_is_const = 1;
  p5.b_const.val = -input_offset;
  p5.b_const.is_signed = 1;
  p5.layer_id = layer_id;
  ctx.tiu_min(&p5);

  p_toi16.res_high = tl_working[1];
  p_toi16.res_low = tl_working[0];
  p_toi16.a = &tl_ifmap;
  p_toi16.res_is_int8 = false;
  p_toi16.b_const.val = 1;
  p_toi16.b_is_const = 1;
  p_toi16.b_const.is_signed = true;
  p_toi16.lshift_bits = 0;
  p_toi16.rshift_bits = 0;
  p_toi16.layer_id = layer_id;
  p_toi16.relu_enable = 0;
  ctx.tiu_mac(&p_toi16);

  p_offset.res_high = tl_working[1];
  p_offset.res_low = tl_working[0];
  p_offset.a_high = tl_working[1];
  p_offset.a_low = tl_working[0];
  p_offset.b_is_const = 1;
  p_offset.b_const.val = input_offset;
  p_offset.b_const.is_signed = 1;
  p_offset.rshift_bits = 0;
  p_offset.layer_id = layer_id;
  p_offset.relu_enable = 0;
  ctx.tiu_add(&p_offset);

  p_high_scale.res_high = nullptr;
  p_high_scale.res_low = tl_working[1];
  p_high_scale.a = tl_working[1];
  p_high_scale.b_is_const = 1;
  p_high_scale.b_const.is_signed = 1;
  p_high_scale.b_const.val = LE_scale;
  p_high_scale.rshift_bits = 0;
  p_high_scale.layer_id = layer_id;
  p_high_scale.relu_enable = 0;
  ctx.tiu_mul(&p_high_scale);

  reset(&tl_ifmap);
  tl_working[0]->fmt = CVK_FMT_U8;
  p_low_scale.res_high = tl_working[1];
  p_low_scale.res_low = &tl_ifmap;
  p_low_scale.a = tl_working[0];
  p_low_scale.res_is_int8 = false;
  p_low_scale.b_const.val = LE_scale;
  p_low_scale.b_is_const = 1;
  p_low_scale.b_const.is_signed = true;
  p_low_scale.lshift_bits = 0;
  p_low_scale.rshift_bits = LE_rshift;
  p_low_scale.layer_id = layer_id;
  p_low_scale.relu_enable = 0;
  ctx.tiu_mac(&p_low_scale);
  tl_working[0]->fmt = tl_ifmap.fmt;

  // reset to 0
  reset(tl_working[0]);
  p9.res_high = tl_working[1];
  p9.res_low = tl_working[0];
  p9.a_high = tl_working[1];
  p9.a_low = tl_working[0];
  p9.b_const.val = 1;
  p9.b_is_const = 1;
  p9.rshift_bits = 0;
  p9.relu_enable = 0;
  ctx.tiu_add(&p9);

  // Negative Done
  p_output_offset.res_high = tl_working[1];
  p_output_offset.res_low = &tl_ifmap;
  p_output_offset.a = tl_working[0];
  p_output_offset.res_is_int8 = true;
  p_output_offset.b_const.val = output_offset;
  p_output_offset.b_is_const = 1;
  p_output_offset.b_const.is_signed = true;
  p_output_offset.lshift_bits = 0;
  p_output_offset.rshift_bits = 0;
  p_output_offset.layer_id = layer_id;
  p_output_offset.relu_enable = 0;
  ctx.tiu_mac(&p_output_offset);

  // final merge
  /*      [ 0 1 0 0
           1 0 0 1
           1 1 0 0 ]
     0 is negivate, 1 is postive
  */
  // postive mul map to output postive section
  cvk_tiu_mul_param_t p_postive_section = {0};
  p_postive_section.res_high = nullptr;
  p_postive_section.res_low = &tl_ofmap;
  p_postive_section.a = &tl_ofmap;
  p_postive_section.b_is_const = 0;
  p_postive_section.b = tl_pos_neg_map;
  p_postive_section.rshift_bits = 0;
  p_postive_section.layer_id = layer_id;
  p_postive_section.relu_enable = 0;
  ctx.tiu_mul(&p_postive_section);

  reset(tl_working[1]);

  // flip table 0->1, 1->0
  // use (table -1) * -1
  cvk_tiu_add_param_t p_flip = {0};
  p_flip.res_high = tl_working[1];
  p_flip.res_low = tl_pos_neg_map;
  p_flip.a_high = tl_working[1];
  p_flip.a_low = tl_pos_neg_map;
  p_flip.b_is_const = 1;
  p_flip.b_const.val = -1;
  p_flip.b_const.is_signed = 1;
  p_flip.rshift_bits = 0;
  p_flip.layer_id = layer_id;
  p_flip.relu_enable = 0;
  ctx.tiu_add(&p_flip);

  cvk_tiu_mul_param_t p_flip_mul = {0};
  p_flip_mul.res_high = nullptr;
  p_flip_mul.res_low = tl_pos_neg_map;
  p_flip_mul.a = tl_pos_neg_map;
  p_flip_mul.b_is_const = 1;
  p_flip_mul.b_const.val = -1;
  p_flip_mul.b_const.is_signed = 1;
  p_flip_mul.rshift_bits = 0;
  p_flip_mul.layer_id = layer_id;
  p_flip_mul.relu_enable = 0;
  ctx.tiu_mul(&p_flip_mul);

  // do negative section
  cvk_tiu_mul_param_t p_negivate_section = {0};
  p_negivate_section.res_high = nullptr;
  p_negivate_section.res_low = &tl_ifmap;
  p_negivate_section.a = &tl_ifmap;
  p_negivate_section.b_is_const = 0;
  p_negivate_section.b = tl_pos_neg_map;
  p_negivate_section.rshift_bits = 0;
  p_negivate_section.layer_id = layer_id;
  p_negivate_section.relu_enable = 0;
  ctx.tiu_mul(&p_negivate_section);

  // merge
  cvk_tiu_or_int8_param_t p_merge = {0};
  p_merge.res = &tl_ofmap;
  p_merge.a = &tl_ofmap;
  p_merge.b = &tl_ifmap;
  p_merge.layer_id = layer_id;
  ctx.tiu_or_int8(&p_merge);
}

void TgReluKernel::compute_leaky_relu_fixed_sym(int32_t step_idx, int32_t flip) {
  auto tl_ifmap = get_input(step_idx, flip);
  auto tl_ofmap = get_output(step_idx, flip);

  bool isIgnorePosPart = (GT_scale == 0);
  bool isSlopeSmallerThanOne = ((LE_scale >> LE_rshift) == 0);

  if (isIgnorePosPart) {
    cvk_tiu_mul_param_t p1 = {0};
    p1.res_high = nullptr;
    p1.res_low = &tl_ofmap;
    p1.a = &tl_ifmap;
    p1.b_const.val = LE_scale;
    p1.b_const.is_signed = true;
    p1.b_is_const = 1;
    p1.rshift_bits = LE_rshift;
    p1.layer_id = layer_id;
    p1.relu_enable = 0;
    ctx.tiu_mul(&p1);

    if (isSlopeSmallerThanOne) {
      cvk_tiu_max_param_t p2 = {0};
      p2.max = &tl_ofmap;
      p2.a = &tl_ofmap;
      p2.b = &tl_ifmap;
      p2.b_is_const = 0;
      p2.layer_id = layer_id;
      ctx.tiu_max(&p2);
    } else {
      cvk_tiu_min_param_t p2 = {0};
      p2.min = &tl_ofmap;
      p2.a = &tl_ofmap;
      p2.b = &tl_ifmap;
      p2.b_is_const = 0;
      p2.layer_id = layer_id;
      ctx.tiu_min(&p2);
    }
  } else {
    cvk_tiu_max_param_t p3 = {0};
    p3.max = &tl_ofmap;
    p3.a = &tl_ifmap;
    p3.b_is_const = 1;
    p3.b_const.is_signed = 1;
    p3.b_const.val = 0;
    p3.layer_id = layer_id;
    ctx.tiu_max(&p3);

    // tl_ofmap = (tl_ofmap * GT_scale) >> GT_rshift
    cvk_tiu_mul_param_t p4 = {0};
    p4.res_high = nullptr;
    p4.res_low = &tl_ofmap;
    p4.a = &tl_ofmap;
    p4.b_const.val = GT_scale;
    p4.b_const.is_signed = true;
    p4.b_is_const = 1;
    p4.rshift_bits = GT_rshift;
    p4.layer_id = layer_id;
    p4.relu_enable = 0;
    ctx.tiu_mul(&p4);

    // tl_ifmap = min(0, tl_ifmap)
    cvk_tiu_min_param_t p5 = {0};
    p5.min = &tl_ifmap;
    p5.a = &tl_ifmap;
    p5.b_is_const = 1;
    p5.b_const.val = 0;
    p5.b_const.is_signed = 1;
    p5.layer_id = layer_id;
    ctx.tiu_min(&p5);

    // tl_ifmap = (tl_ifmap * slope) >> LE_rshift
    cvk_tiu_mul_param_t p6 = {0};
    p6.res_high = nullptr;
    p6.res_low = &tl_ifmap;
    p6.a = &tl_ifmap;
    p6.b_const.val = LE_scale;
    p6.b_const.is_signed = true;
    p6.b_is_const = 1;
    p6.rshift_bits = LE_rshift;
    p6.layer_id = layer_id;
    p6.relu_enable = 0;
    ctx.tiu_mul(&p6);

    // tl_ofmap = or(tl_ofmap, tl_ifmap)
    cvk_tiu_or_int8_param_t p7 = {0};
    p7.res = &tl_ofmap;
    p7.a = &tl_ofmap;
    p7.b = &tl_ifmap;
    p7.layer_id = layer_id;
    ctx.tiu_or_int8(&p7);
  }
}

void TgReluKernel::compute_leaky_relu_bf16(int32_t step_idx, int32_t flip) {
  auto tl_ifmap = get_input(step_idx, flip);
  auto tl_ofmap = get_output(step_idx, flip);
  cvk_tiu_mul_param_t p1 = {0};
  p1.res_high = nullptr; // useless
  p1.res_low = &tl_ofmap;
  p1.a = &tl_ifmap;
  p1.b_const.val = ctx.convert_fp32_to_bf16(negative_slope);
  p1.b_const.is_signed = true;
  p1.b_is_const = true;
  p1.rshift_bits = 0;
  p1.layer_id = layer_id;
  p1.relu_enable = 0;
  ctx.tiu_mul(&p1);

  // 1. relu = max(tl_ifmap, relu)
  if (negative_slope <= 1) {
    cvk_tiu_max_param_t p2 = {0};
    p2.max = &tl_ofmap;
    p2.a = &tl_ifmap;
    p2.b_is_const = 0;
    p2.b_const.is_signed = 1;
    p2.b = &tl_ofmap;
    p2.layer_id = layer_id;
    ctx.tiu_max(&p2);
  } else {
    cvk_tiu_min_param_t p3 = {0};
    p3.min = &tl_ofmap;
    p3.a = &tl_ifmap;
    p3.b_is_const = 0;
    p3.b_const.is_signed = 1;
    p3.b = &tl_ofmap;
    p3.layer_id = layer_id;
    ctx.tiu_min(&p3);
  }
}

void TgReluKernel::compute_prelu_fixed(int32_t step_idx, int32_t flip) {
  auto tl_ifmap = get_input(step_idx, flip);
  auto tl_ofmap = get_output(step_idx, flip);
  cvk_tiu_max_param_t p1 = {0};
  p1.max = &tl_ofmap;
  p1.a = &tl_ifmap;
  p1.b_is_const = 1;
  p1.b_const.is_signed = 1;
  p1.b_const.val = 0;
  p1.layer_id = layer_id;
  ctx.tiu_max(&p1);

  // 1. relu = (relu * GT_scale) >> GT_rshift
  cvk_tiu_mul_param_t p2 = {0};
  p2.res_high = nullptr;
  p2.res_low = &tl_ofmap;
  p2.a = &tl_ofmap;
  p2.b_const.val = GT_scale;
  p2.b_const.is_signed = true;
  p2.b_is_const = 1;
  p2.rshift_bits = GT_rshift;
  p2.layer_id = layer_id;
  p2.relu_enable = 0;
  ctx.tiu_mul(&p2);

  // 2. neg = neg(0, botom)
  cvk_tiu_min_param_t p3 = {0};
  p3.min = &tl_ifmap;
  p3.a = &tl_ifmap;
  p3.b_is_const = 1;
  p3.b_const.val = 0;
  p3.b_const.is_signed = 1;
  p3.layer_id = layer_id;
  ctx.tiu_min(&p3);

  // 3. neg (n,c,h,w) = (neg(n,c,h,w) * slope(1,c,1,1)) >>
  // LE_rshift
  cvk_tiu_depthwise_pt_convolution_param_t p4 = {0};
  p4.ins_h = 0;
  p4.ins_last_h = 0;
  p4.ins_w = 0;
  p4.ins_last_w = 0;
  p4.pad_top = 0;
  p4.pad_bottom = 0;
  p4.pad_left = 0;
  p4.pad_right = 0;
  p4.stride_h = 1;
  p4.stride_w = 1;
  p4.dilation_h = 1;
  p4.dilation_w = 1;
  p4.ofmap = &tl_ifmap;
  p4.ifmap = &tl_ifmap;
  p4.weight = tl_slope;
  p4.bias = nullptr;
  p4.rshift_bits = LE_rshift;
  p4.relu_enable = 0;
  p4.layer_id = layer_id;
  p4.ins_val = 0;                            // symmetric quantization
  p4.ins_fp = ctx.convert_fp32_to_bf16(0.0); // symmetric quantization
  ctx.tiu_pt_depthwise_convolution(&p4);

  // 4. tl_ifmap = or relu, neg
  cvk_tiu_or_int8_param_t p5 = {0};
  p5.res = &tl_ofmap;
  p5.a = &tl_ofmap;
  p5.b = &tl_ifmap;
  p5.layer_id = layer_id;
  ctx.tiu_or_int8(&p5);
}

void TgReluKernel::compute_prelu_bf16(int32_t step_idx, int32_t flip) {
  auto tl_ifmap = get_input(step_idx, flip);
  auto tl_ofmap = get_output(step_idx, flip);
  cvk_tiu_min_param_t p1 = {0};
  p1.min = &tl_ofmap;
  p1.a = &tl_ifmap;
  p1.b_is_const = 1;
  p1.b_const.val = ctx.convert_fp32_to_bf16(0.0);
  p1.b_const.is_signed = 1;
  p1.layer_id = layer_id;
  ctx.tiu_min(&p1);

  // 2. neg (n,c,h,w) = (neg(n,c,h,w) * slope(1,c,1,1)) >> LE_rshift
  cvk_tiu_depthwise_pt_convolution_param_t p2 = {0};
  p2.ins_h = 0;
  p2.ins_last_h = 0;
  p2.ins_w = 0;
  p2.ins_last_w = 0;
  p2.pad_top = 0;
  p2.pad_bottom = 0;
  p2.pad_left = 0;
  p2.pad_right = 0;
  p2.stride_h = 1;
  p2.stride_w = 1;
  p2.dilation_h = 1;
  p2.dilation_w = 1;
  p2.ofmap = &tl_ofmap;
  p2.ifmap = &tl_ofmap;
  p2.weight = tl_slope;
  p2.bias = nullptr;
  p2.rshift_bits = 0;
  p2.relu_enable = 0;
  p2.layer_id = layer_id;
  p2.ins_val = 0;                            // symmetric quantization
  p2.ins_fp = ctx.convert_fp32_to_bf16(0.0); // symmetric quantization
  ctx.tiu_pt_depthwise_convolution(&p2);

  // 3. relu = relu(tl_ifmap), dirty it
  cvk_tiu_max_param_t p3 = {0};
  p3.max = &tl_ifmap;
  p3.a = &tl_ifmap;
  p3.b_is_const = 1;
  p3.b_const.is_signed = 1;
  p3.b_const.val = ctx.convert_fp32_to_bf16(0.0);
  p3.layer_id = layer_id;
  ctx.tiu_max(&p3);

  cvk_tiu_add_param_t p4 = {0};
  p4.res_high = nullptr;
  p4.res_low = &tl_ofmap;
  p4.a_high = nullptr;
  p4.a_low = &tl_ifmap;
  p4.b_is_const = false;
  p4.b.high = nullptr;
  p4.b.low = &tl_ofmap;
  p4.rshift_bits = 0;
  p4.layer_id = layer_id;
  p4.relu_enable = 0;
  ctx.tiu_add(&p4);
}

void TgReluKernel::compute(int32_t step_idx, int32_t flip) {
  switch (mode) {
  case RELU:
    compute_relu(step_idx, flip);
    break;
  case LEAKY_RELU:
    if (fmt == CVK_FMT_BF16) {
      compute_leaky_relu_bf16(step_idx, flip);
    } else {
      if (is_asymmetric) {
        compute_leaky_relu_fixed_asym(step_idx, flip);
      } else {
        compute_leaky_relu_fixed_sym(step_idx, flip);
      }
    }
    break;
  case PRELU:
    if (fmt == CVK_FMT_BF16) {
      compute_prelu_bf16(step_idx, flip);
    } else {
      compute_prelu_fixed(step_idx, flip);
    }
    break;
  default:
    break;
  }
}

void TgReluKernel::store(int32_t step_idx, int32_t flip) {
  auto &tile = tiles[step_idx];
  auto tl_ofmap = get_output(step_idx, flip);
  if (mode == PRELU) {
    ctx.tdma_store_stride(&tl_ofmap, ga_output + tile.offset, gstride);
  } else {
    ctx.tdma_store(&tl_ofmap, ga_output + tile.offset);
  }
}

void TgReluKernel::schedule() {
  allocLmem();
  int32_t total_steps = tiles.size();
  for (int32_t i = 0; i < total_steps + 2; i++) {
    ctx.parallel_enable();

    if (i - 1 >= 0 && i - 1 < total_steps) {
      compute(i - 1, 1 - flip);
    }
    if (i < total_steps) {
      load(i, flip);
    }
    if (i - 2 >= 0) {
      store(i - 2, flip);
    }
    flip = 1 - flip;
    ctx.parallel_disable();
  }
  deallocLmem();
}

// i8/bf16 relu
void cvi_backend_tg_relu_kernel(const CviBackendContext &ctx, uint32_t layer_id,
                                uint64_t ga_input, uint64_t ga_output, int n,
                                int c, int h, int w, cvk_fmt_t fmt) {
  TgReluKernel kernel(ctx);
  kernel.init(layer_id, n, c, h, w, ga_input, ga_output, 0, 0, 0, 0, 0, 0, 0, 0, fmt,
              TgReluKernel::RELU);
  kernel.selectTilePolicy();
  kernel.schedule();
}

// i8 leakyrelu
void cvi_backend_tg_fixed_leakyrelu_kernel(const CviBackendContext &ctx,
                                           uint32_t layer_id, uint64_t ga_input,
                                           uint64_t ga_output, int n, int c,
                                           int h, int w, int GT_rshift,
                                           int LE_rshift, int GT_scale,
                                           int LE_scale, int input_offset, int output_offset) {
  TgReluKernel kernel(ctx);
  kernel.init(layer_id, n, c, h, w, ga_input, ga_output, 0, 0, GT_rshift,
              GT_scale, LE_rshift, LE_scale, input_offset, output_offset, CVK_FMT_I8,
              TgReluKernel::LEAKY_RELU);
  kernel.selectTilePolicy();
  kernel.schedule();
}

// bf16 leakyrelu
void cvi_backend_tg_bf16_leakyrelu_kernel(const CviBackendContext &ctx,
                                          uint32_t layer_id, gaddr_t ga_input,
                                          gaddr_t ga_output,
                                          float negative_slope, int n, int c,
                                          int h, int w) {
  TgReluKernel kernel(ctx);
  kernel.init(layer_id, n, c, h, w, ga_input, ga_output, 0, negative_slope, 0,
              0, 0, 0, 0, 0, CVK_FMT_BF16, TgReluKernel::LEAKY_RELU);
  kernel.selectTilePolicy();
  kernel.schedule();
}

// i8 prelu
void cvi_backend_tg_fixed_prelu_kernel(const CviBackendContext &ctx,
                                       uint32_t layer_id, uint64_t ga_input,
                                       uint64_t ga_output,
                                       uint64_t negative_scope_gaddr, int n,
                                       int c, int h, int w, int GT_rshift,
                                       int GT_scale, int LE_rshift) {
  TgReluKernel kernel(ctx);
  kernel.init(layer_id, n, c, h, w, ga_input, ga_output, negative_scope_gaddr,
              0, GT_rshift, GT_scale, LE_rshift, 0, 0, 0, CVK_FMT_I8,
              TgReluKernel::PRELU);
  kernel.selectTilePolicy();
  kernel.schedule();
}

void cvi_backend_tg_bf16_prelu_kernel(const CviBackendContext &ctx,
                                      uint32_t layer_id, gaddr_t ga_input,
                                      gaddr_t ga_output, gaddr_t ga_slope,
                                      int n, int c, int h, int w) {
  TgReluKernel kernel(ctx);
  kernel.init(layer_id, n, c, h, w, ga_input, ga_output, ga_slope, 0, 0, 0, 0,
              0, 0,
              0, CVK_FMT_BF16, TgReluKernel::PRELU);
  kernel.selectTilePolicy();
  kernel.schedule();
}