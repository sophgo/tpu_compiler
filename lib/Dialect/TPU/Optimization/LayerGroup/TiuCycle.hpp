/*
 * Copyright (C) Bitmain Technologies Inc.
 * All Rights Reserved.
 */
#ifndef GROUPOPS_TIUOPSTATS_H
#define GROUPOPS_TIUOPSTATS_H

#include "NetGraph.hpp"
#include "Tensor.hpp"
#include "utils.hpp"
#include "Cycle.hpp"

#define LOCAL_MEM_ADDRWIDTH 15 //32KB shift
#define LOCAL_MEM_START_ADDR 0x00000000
#define LANE_NUMBER 32
#define EU_NUMBER 16

namespace mlir {

// copy from bm1880v2_tiu_reg.h
enum Des_tsk_typ {
    Conv2D,
    Pooling,
    MatrixMul,
    TensorArithmetic,
    MatrixMul2
};

typedef struct {
  uint32_t cmd_en;
  uint32_t cmd_end;
  uint32_t cmd_id_en;
  uint32_t cmd_id_tpu;
  uint32_t cmd_id_gdma;
  uint32_t cmd_keep;
  uint32_t cmd_intr_en;
  uint32_t tsk_typ;
  uint32_t tsk_eu_typ;
  uint32_t tsk_opd_num;
  uint32_t opt_right_shift;
  uint32_t opt_left_shift;
  uint32_t opt_shift_typ;
  uint32_t opt_rshift_typ;
  uint32_t opt_res_add;
  uint32_t opt_relu;
  uint32_t opt_left_tran;
  uint32_t opt_chl_quan;
  uint32_t tens_mdsum;
  uint32_t tens_lookup;
  uint32_t opt_res0_sign;
  uint32_t opt_opd0_sign;
  uint32_t opt_opd1_sign;
  uint32_t opt_opd2_sign;
  uint32_t opt_res0_int8;
  uint32_t opt_opd0_int8;
  uint32_t opt_opd1_int8;
  uint32_t opt_opd2_int8;
  uint32_t opt_opd0_const;
  uint32_t opt_opd1_const;
  uint32_t opt_opd2_const;
  uint32_t short_nchwstr_same;
  uint32_t short_res0_str;
  uint32_t short_opd0_str;
  uint32_t short_opd1_str;
  uint32_t short_opd2_str;
  uint32_t conv_opd0_x_ins0;
  uint32_t conv_opd0_y_ins0;
  uint32_t conv_opd0_x_ins0_last;
  uint32_t conv_opd0_y_ins0_last;
  uint32_t conv_opd1_x_ins0;
  uint32_t conv_opd1_y_ins0;
  uint32_t opd0_ins_val;
  uint32_t ps32_md;
  uint32_t double_conv;
  uint32_t rsvd0;
  uint32_t res0_n;
  uint32_t res0_c;
  uint32_t res0_h;
  uint32_t res0_w;
  uint32_t res0_addr;
  uint32_t opd0_addr;
  uint32_t opd1_addr;
  uint32_t rsvd1;
  uint32_t opd2_addr;
  uint32_t opd0_c;
  uint32_t opd0_h;
  uint32_t opd0_w;
  uint32_t opd1_h;
  uint32_t opd1_w;
  uint32_t conv_opd0_up_pad;
  uint32_t conv_opd0_dn_pad;
  uint32_t conv_opd0_lf_pad;
  uint32_t conv_opd0_rt_pad;
  uint32_t conv_op_x_str;
  uint32_t conv_op_y_str;
  uint32_t opd0_ins_fp;
  uint32_t rsvd2;
  uint32_t opd0_n;
  uint32_t opd1_n;
  uint32_t opd1_c;
  uint32_t opd2_n;
  uint32_t opd2_c;
  uint32_t opd2_h;
  uint32_t opd2_w;
  uint32_t quan_m;
  uint32_t opd_typ;
  uint32_t fp_round_typ;
  uint32_t rsvd7;
  uint32_t rsvd3;
  uint32_t res0_n_str;
  uint32_t res0_c_str;
  uint32_t res0_h_str;
  uint32_t res0_w_str;
  uint32_t res0_b_str;
  uint32_t opd0_n_str;
  uint32_t opd0_c_str;
  uint32_t rsvd4;
  uint32_t opd0_h_str;
  uint32_t opd0_w_str;
  uint32_t opd0_b_str;
  uint32_t opd1_n_str;
  uint32_t opd1_c_str;
  uint32_t opd1_h_str;
  uint32_t opd1_w_str;
  uint32_t rsvd5;
  uint32_t opd1_b_str;
  uint32_t opd2_n_str;
  uint32_t opd2_c_str;
  uint32_t opd2_h_str;
  uint32_t opd2_w_str;
  uint32_t opd2_b_str;
  uint32_t layer_info;
  uint32_t rsvd6;
} tiu_inst_t;


class TiuCycle {
 public:
  TiuCycle(NetGraph* net_graph) {
    net_graph_ = net_graph;
    setup_hw_config();
  }

  ~TiuCycle() {}

  int get_cycle(int cur_layer);
  uint64_t calCycle(tiu_inst_t task);
  void setup_hw_config();
  int getTensorArithmeticMode(int taskType, bool is8BitMode);
  float getEltwiseLatency(int taskType, bool is8BitMode, bool isOpd1Const, int mode);

 private:
  void set_tl_conv_param();
  void set_tl_deconv_param();
  void set_tl_eltwise(bool is_h_split);
  void set_tl_pooling_param();
  void set_tl_lrn(bool is_h_split);
  void set_tl_broadcast_mul(bool is_h_split);
  void set_tl_activation(bool is_h_split);
  void set_tl_upsample(bool is_h_split);
  void set_tl_leaky_relu(bool is_h_split);
  void set_tl_prelu(bool is_h_split);
  void set_tl_concat(bool is_h_split);
  void set_tl_pad(bool is_h_split);
  void set_tl_crop(bool is_h_split);
  void set_tl_relu(bool is_h_split);
  void set_tl_quant(bool is_h_split);
  void set_tl_zero_mask(bool is_h_split);

  Operation *op;
  int layer_id_;
  bool isPerChannelQuan;
  bool isBfloat16;
  tiu_inst_t inst;
  NetGraph* net_graph_;
  int tpu_frequency_;
};

}
#endif
