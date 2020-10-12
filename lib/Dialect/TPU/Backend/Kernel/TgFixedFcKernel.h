#ifndef TG_FIXED_FC_KERNEL_H
#define TG_FIXED_FC_KERNEL_H

// refined 2020-10-12
class TgFcKernel {
public:
  TgFcKernel(const CviBackendContext &ctx, int input_row, int input_col,
             int output_col, bool with_bias, cvk_fmt_t fmt)
      : ctx(ctx), input_row(input_row), input_col(input_col),
        output_col(output_col), with_bias(with_bias), fmt(fmt) {
    dataTypeSize = (fmt == CVK_FMT_BF16) ? 2 : 1;
  }

// Y(M, N) = L(M, K) * R(K, N)
struct TileInfo {
  int m_step;
  int n_step;
  int k_step;
};

  TileInfo getTileSizes();
  int getLmSizePerLane(int tileM, int tileK, int tileN, bool hasB);

  const CviBackendContext &ctx;
  int input_row = {1};
  int input_col = {1};
  int output_col = {1};
  bool with_bias = {false};
  bool do_relu = {false};
  cvk_fmt_t fmt = {CVK_FMT_I8};
  int64_t ga_ifmap = {0};
  int64_t ga_weight = {0};
  int64_t ga_bias = {0};
  int64_t ga_ofmap = {0};
  int quant_rshift = {0};
  uint32_t quant_multiplier = {1};
  int dataTypeSize = {1};
};

#endif // TG_FIXED_FC_KERNEL_H
