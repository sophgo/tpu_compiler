/*
 * Copyright (C) Bitmain Technologies Inc.
 * All Rights Reserved.
 */
#ifndef LAYERGROUP_TDMACYCLE_H
#define LAYERGROUP_TDMACYCLE_H

#include "NetGraph.hpp"
#include "Tensor.hpp"
#include "utils.hpp"
#include "Cycle.hpp"

namespace mlir {

typedef struct {
  uint32_t vld;
  uint32_t compress_en;
  uint32_t eod;
  uint32_t intp_en;
  uint32_t bar_en;
  uint32_t check_bf16_value;
  uint32_t trans_dir;
  uint32_t rsv00;
  uint32_t trans_fmt;
  uint32_t transpose_md;
  uint32_t rsv01;
  uint32_t outstanding_en;
  uint32_t cmd_id;
  uint32_t spec_func;
  uint32_t dst_fmt;
  uint32_t src_fmt;
  uint32_t cmprs_fmt;
  uint32_t sys_dtype;
  uint32_t rsv2_1;
  uint32_t int8_sign;
  uint32_t compress_zero_guard;
  uint32_t int8_rnd_mode;
  uint32_t wait_id_tpu;
  uint32_t wait_id_other_tdma;
  uint32_t wait_id_sdma;
  uint32_t const_val;
  uint32_t src_base_reg_sel;
  uint32_t mv_lut_idx;
  uint32_t dst_base_reg_sel;
  uint32_t mv_lut_base;
  uint32_t rsv4_5;
  uint32_t dst_h_stride;
  uint32_t dst_c_stride;
  uint32_t dst_n_stride;
  uint32_t src_h_stride;
  uint32_t src_c_stride;
  uint32_t src_n_stride;
  uint32_t dst_c;
  uint32_t src_c;
  uint32_t dst_w;
  uint32_t dst_h;
  uint32_t src_w;
  uint32_t src_h;
  uint32_t dst_base_addr_low;
  uint32_t src_base_addr_low;
  uint32_t src_n;
  uint32_t dst_base_addr_high;
  uint32_t src_base_addr_high;
  uint32_t compress_bias0;
  uint32_t compress_bias1;
  uint32_t layer_ID;
  // extra
  uint64_t dram_byte_count;
  uint64_t sram_byte_count;
} tdma_inst_t;

class TdmaCycle {
 public:
  TdmaCycle(NetGraph * net_graph) {
    net_graph_ = net_graph;
    inst_ = &inst;
    setup_hw_config();
  }

  ~TdmaCycle() {}

  int get_cycle(const TENSOR_STEP& step);
  int get_cycle_load();
  int get_cycle_store();
  int init(const TENSOR_STEP& step);
  void setup_hw_config();
 protected:
 private:
  struct dim {int n; int c; int h; int w;};
  dim tg_default_stride(int n, int c, int h, int w, int unit_size) {
    uint32_t data_type_size = unit_size;
    dim stride;
    stride.h = w * data_type_size;
    stride.c = h * stride.h;
    stride.n = c * stride.c;
    return stride;
  }

  dim tl_default_stride(int n, int c, int h, int w, int unit_size, int eu_align) {
    dim stride;
    uint32_t eu_num = EU_NUM;
    uint32_t npu_num = NPU_NUM;
    uint32_t fmt = unit_size;
    stride.w = fmt;
    stride.h = w * fmt;
    if (eu_align)
      stride.c = align_up(h * w * fmt, eu_num);
    else
      stride.c = h * w * fmt;

    stride.n = stride.c * ceiling_func(c, npu_num);
    return stride;
  }

  bool transpose = false;
  bool aligned = false;
  int tensor_dim[4];
  int local_shape[4];
  NetGraph* net_graph_;
  Tensor* tensor;
  int tensor_id;

  uint64_t dram_bw_;
  uint64_t sram_bw_;

  tdma_inst_t* inst_;
  tdma_inst_t inst;
  int counter_;

  void get_tdma_cycle(uint64_t baseAddr, uint64_t data_size, bool isStore);
  uint64_t calByteCnt(uint64_t baseAddr, uint64_t size);
  uint64_t calSramCycle(tdma_inst_t* _inst);
  void cal_load_cycle();
  void cal_store_cycle();
};
}
#endif
