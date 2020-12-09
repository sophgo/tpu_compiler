

module {
  func @tpu_func(%arg0: tensor<4x64x227x227xf32>) -> tensor<4x64x227x227xf32> {
    %1 = "tpu.input"(%arg0) {name = "data", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i8}} : (tensor<4x64x227x227xf32>) -> tensor<4x64x227x227xf32>
    %2 = "tpu.reverse"(%1) {axis = 0 : i32, name = "reverse", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<4x64x227x227xf32>) -> tensor<4x64x227x227xf32>
    return %2 : tensor<4x64x227x227xf32>
  }
}
