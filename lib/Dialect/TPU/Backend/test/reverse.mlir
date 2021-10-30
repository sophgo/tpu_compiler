

module {
  func @tpu_func(%arg0: tensor<4x64x227x227xf32>) -> tensor<4x64x227x227xf32> {
    %1 = "tpu.input"(%arg0) {name = "data", quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<4x64x227x227xf32>) -> tensor<4x64x227x227xf32>
    %2 = "tpu.reverse"(%1) {axis = 0 : i32, name = "reverse", quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<4x64x227x227xf32>) -> tensor<4x64x227x227xf32>
    return %2 : tensor<4x64x227x227xf32>
  }
}
