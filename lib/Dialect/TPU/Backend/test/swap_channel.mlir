
module {
  func @tpu_func(%arg0: tensor<64x3x256x256xf32>) -> (tensor<64x3x256x256xf32>,tensor<64x3x256x256xf32>) {
    %1 = "tpu.input"(%arg0) {name = "data", quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<64x3x256x256xf32>) -> tensor<64x3x256x256xf32>
    %2 = "tpu.swap_channel"(%1) {name = "older201", channel_order = [2 : i32, 0 : i32, 1 : i32], quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<64x3x256x256xf32>) -> tensor<64x3x256x256xf32>
    %3 = "tpu.swap_channel"(%1) {name = "older120", channel_order = [1 : i32, 2 : i32, 0 : i32], quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<64x3x256x256xf32>) -> tensor<64x3x256x256xf32>
    return %2,%3 : tensor<64x3x256x256xf32>,tensor<64x3x256x256xf32>
  }
}
