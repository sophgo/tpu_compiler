

module {
  func @tpu_func(%arg0: tensor<8x6x200x200xf32>) -> tensor<8x3x400x400xf32> {
    %1 = "tpu.input"(%arg0) {name = "data", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<8x6x200x200xf32>) -> tensor<8x6x200x200xf32>
    %2 = "tpu.yuv420_csc"(%1) {name = "yuv", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<8x6x200x200xf32>) -> tensor<8x3x400x400xf32>
    return %2 : tensor<8x3x400x400xf32>
  }
}
