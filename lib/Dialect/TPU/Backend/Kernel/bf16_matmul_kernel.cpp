#include "CviBackendContext.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include <cmath>

#define DEBUG_TYPE "bmnet_bf16_matmul_kernel"


void cvi_backend_tg_bf16_matmul_kernel(
    const CviBackendContext &ctx, uint32_t layer_id,
    gaddr_t bottom_0_data_gaddr,
    gaddr_t bottom_1_data_gaddr,
    gaddr_t top_data_gaddr,
    int in_row, int in_col,
    int out_col) {

  cvi_backend_tg_bf16_fc_kernel(
      ctx,
      layer_id, // layer_id
      bottom_0_data_gaddr, // input_data_gaddr
      bottom_1_data_gaddr, // weight_data_gaddr
      GA_INVALID, // bias_data_gaddr
      top_data_gaddr, // output_data_gaddr
      in_row, // int in_row
      in_col, // int in_col
      out_col, // in out_col,
      false, // has_bias
      false, // do_activation
      0);
}

