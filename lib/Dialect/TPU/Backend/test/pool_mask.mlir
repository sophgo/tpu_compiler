

module {
  func @tpu_func(%arg0: tensor<8x64x58x61xf32>) -> tensor<8x64x60x63xf32> {
    %1 = "tpu.input"(%arg0) {name = "data", quant = {mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32}} : (tensor<8x64x58x61xf32>) -> tensor<8x64x58x61xf32>
    %2 = "tpu.pool_mask"(%1) {name = "pool_mask", quant = {mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32}, scale = 3 : i32} : (tensor<8x64x58x61xf32>) -> tensor<8x64x60x63xf32>
    return %2 : tensor<8x64x60x63xf32>
  }
}
