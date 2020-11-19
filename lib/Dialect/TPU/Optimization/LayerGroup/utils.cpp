#include "utils.hpp"

namespace mlir {

void getConvParam(Operation *p,
                  int &n, int &ic, int &ih, int &iw,
                  int &oc, int &oh, int &ow, int &g,
                  int &kh, int &kw,
                  int &sh, int &sw, int &pt, int &pb,
                  int &pl, int &pr, int &dh, int &dw,
                  bool &is_dw, bool &with_bias,
                  bool &do_relu,
                  bool &do_ic_align,
                  bool &do_leaky_relu,
                  int &pad_value) {
  if (auto op = dyn_cast<tpu::TG_INT8_PC_Conv2DOp>(p)) {
    bool is_deconv = false;
    parseConvParam(op.param(), is_deconv, op.input(), op.output(), op.filter(),
                   n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, pt, pb, pl, pr,
                   dh, dw, is_dw, with_bias, do_relu, pad_value);
    do_ic_align = op.do_ic_alignment().hasValue() ?
                  op.do_ic_alignment().getValue() : false;
    do_leaky_relu = op.do_leaky_relu();
  } else if (auto op = dyn_cast<tpu::TG_BF16_Conv2DOp>(p)) {
    bool is_deconv = false;
    parseConvParam(op.param(), is_deconv, op.input(), op.output(), op.filter(),
                   n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, pt, pb, pl, pr,
                   dh, dw, is_dw, with_bias, do_relu, pad_value);
    do_ic_align = op.do_ic_alignment().hasValue() ?
                  op.do_ic_alignment().getValue() : false;
    do_leaky_relu = op.do_leaky_relu();
  } else if (auto op = dyn_cast<tpu::TG_INT8_PC_DeConv2DOp>(p)) {
    bool is_deconv = true;
    parseConvParam(op.param(), is_deconv, op.input(), op.output(), op.filter(),
                   n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, pt, pb, pl, pr,
                   dh, dw, is_dw, with_bias, do_relu, pad_value);
    do_ic_align = op.do_ic_alignment().hasValue() ?
                  op.do_ic_alignment().getValue() : false;
    do_leaky_relu = op.do_leaky_relu();
  }else if (auto op = dyn_cast<tpu::TG_BF16_DeConv2DOp>(p)) {
    bool is_deconv = true;
    parseConvParam(op.param(), is_deconv, op.input(), op.output(), op.filter(),
                   n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, pt, pb, pl, pr,
                   dh, dw, is_dw, with_bias, do_relu, pad_value);
    do_ic_align = op.do_ic_alignment().hasValue() ?
                  op.do_ic_alignment().getValue() : false;
    do_leaky_relu = op.do_leaky_relu();
  } else {
    assert(!"Only support INT8/BF16 Conv in LayerGroup");
  }
}

void getConcatParam(Operation *op,
                           int &axis, bool &do_relu) {
  if (isa<tpu::TG_INT8_ConcatOp>(op)) {
    auto concat_op = dyn_cast<tpu::TG_INT8_ConcatOp>(op);
    axis = concat_op.axis();
    do_relu = concat_op.do_relu();
  } else if (isa<tpu::TG_BF16_ConcatOp>(op)){
    auto concat_op = dyn_cast<tpu::TG_BF16_ConcatOp>(op);
    axis = concat_op.axis();
    do_relu = concat_op.do_relu();
  } else {
    assert(!"Only support INT8/BF16 Concat in LayerGroup");
  }
}

void getSliceParam(Operation * op,
                  int &axis) {
  if (isa<tpu::TG_INT8_SliceOp>(op)) {
    auto slice_op = dyn_cast<tpu::TG_INT8_SliceOp>(op);
    axis = slice_op.axis();
  } else if (isa<tpu::TG_BF16_SliceOp>(op)) {
    auto slice_op = dyn_cast<tpu::TG_BF16_SliceOp>(op);
    axis = slice_op.axis();
  } else {
    assert(!"Only support INT8/BF16 Slice in LayerGroup");
  }
}

void getUpsampleParam(Operation * op,
                      int &scale_h, int &scale_w) {
  if (isa<tpu::TG_INT8_UpsampleOp>(op)) {
    auto upsample_op = dyn_cast<tpu::TG_INT8_UpsampleOp>(op);
    scale_h = upsample_op.scale_h();
    scale_w = upsample_op.scale_w();
  } else if (isa<tpu::TG_BF16_UpsampleOp>(op)) {
    auto upsample_op = dyn_cast<tpu::TG_BF16_UpsampleOp>(op);
    scale_h = upsample_op.scale_h();
    scale_w = upsample_op.scale_w();
  } else {
    assert(!"Only support INT8/BF16 Upsample in LayerGroup");
  }
}

void getPoolingParam(Operation * op,
                    int &n, int &c, int &ih, int &iw,
                    int &oh, int &ow,
                    int &kh, int &kw, int &sh, int &sw,
                    int &pt, int &pb, int &pl, int &pr,
                    bool &is_global, bool &do_relu,
                    bool &count_include_pad) {
  if (isa<tpu::TG_INT8_PoolAvg2DOp>(op)) {
    auto pooling_op = cast<tpu::TG_INT8_PoolAvg2DOp>(op);
    parsePoolParam(pooling_op.param(), pooling_op.input(),
                   pooling_op.output(),
                   n, c, ih, iw, oh, ow,
                   kh, kw, sh, sw, pt, pb, pl, pr,
                   is_global, do_relu, count_include_pad);
  } else if (isa<tpu::TG_INT8_PoolMax2DOp>(op)) {
    auto pooling_op = cast<tpu::TG_INT8_PoolMax2DOp>(op);
    parsePoolParam(pooling_op.param(), pooling_op.input(),
                   pooling_op.output(),
                   n, c, ih, iw, oh, ow,
                   kh, kw, sh, sw, pt, pb, pl, pr,
                   is_global, do_relu, count_include_pad);
  } else if (isa<tpu::TG_BF16_PoolAvg2DOp>(op)) {
    auto pooling_op = cast<tpu::TG_BF16_PoolAvg2DOp>(op);
    parsePoolParam(pooling_op.param(), pooling_op.input(),
                   pooling_op.output(),
                   n, c, ih, iw, oh, ow,
                   kh, kw, sh, sw, pt, pb, pl, pr,
                   is_global, do_relu, count_include_pad);
  } else if (isa<tpu::TG_BF16_PoolMax2DOp>(op)) {
    auto pooling_op = cast<tpu::TG_BF16_PoolMax2DOp>(op);
    parsePoolParam(pooling_op.param(), pooling_op.input(),
                   pooling_op.output(),
                   n, c, ih, iw, oh, ow,
                   kh, kw, sh, sw, pt, pb, pl, pr,
                   is_global, do_relu, count_include_pad);
  } else {
    assert(!"Only support INT8/BF16 Pooling in LayerGroup");
  }
}

void getEltwiseAddParam(Operation * op,
                        bool &do_early_stride,
                        int &h_stride, int &w_stride) {
  if (isa<tpu::TG_INT8_EltwiseAddOp>(op)) {
    auto eltwise_op = dyn_cast<tpu::TG_INT8_EltwiseAddOp>(op);
    do_early_stride = eltwise_op.do_early_stride();
    h_stride = eltwise_op.early_stride_h();
    w_stride = eltwise_op.early_stride_w();
  } else if(isa<tpu::TG_BF16_EltwiseAddOp>(op)) {
    auto eltwise_op = dyn_cast<tpu::TG_BF16_EltwiseAddOp>(op);
    do_early_stride = eltwise_op.do_early_stride();
    h_stride = eltwise_op.early_stride_h();
    w_stride = eltwise_op.early_stride_w();
  } else {
    assert(!"Unsupport eltwise add op in Layergroup.");
  }
}

void getEltwiseReluParam(Operation * op,
                         bool &do_relu) {
  if (auto eltwise_op = dyn_cast<tpu::TG_INT8_EltwiseAddOp>(op)) {
    do_relu = eltwise_op.do_relu();
  } else if(auto eltwise_op = dyn_cast<tpu::TG_BF16_EltwiseAddOp>(op)) {
    do_relu = eltwise_op.do_relu();
  } else if(auto eltwise_op = dyn_cast<tpu::TG_INT8_EltwiseMulOp>(op)) {
    do_relu = eltwise_op.do_relu();
  } else if(auto eltwise_op = dyn_cast<tpu::TG_BF16_EltwiseMulOp>(op)) {
    do_relu = eltwise_op.do_relu();
  } else {
    assert(!"Unsupport eltwise op in Layergroup.");
  }
}

void getLrnParam(Operation * op,
                  uint32_t &local_size,
                  int &sum_rshift, int &lrn_rshift,
                  int &quant_data0, int &quant_data1,
                  float &alpha, float &k) {
  if (auto lrn_op = dyn_cast<tpu::TG_INT8_LrnOp>(op)) {
    local_size = lrn_op.local_size();
    sum_rshift = lrn_op.sum_rshift();
    lrn_rshift = lrn_op.lrn_rshift();
    quant_data0 = lrn_op.quant_data0();
    quant_data1 = lrn_op.quant_data1();
  } else if(auto lrn_op = dyn_cast<tpu::TG_BF16_LrnOp>(op)) {
    local_size = lrn_op.local_size();
    alpha = lrn_op.alpha().convertToFloat();
    k = lrn_op.k().convertToFloat();
    sum_rshift = 0;
    lrn_rshift = 0;
    quant_data0 = 0;
    quant_data1 = 0;
  } else {
    assert(!"Unsupport eltwise add op in Layergroup.");
  }
}

}
