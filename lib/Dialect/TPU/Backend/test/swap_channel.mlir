
module {
  func @tpu_func(%arg0: tensor<64x3x256x256xf32>) -> (tensor<64x3x256x256xf32>,tensor<64x3x256x256xf32>) {
    %1 = "tpu.input"(%arg0) {name = "data", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i8}} : (tensor<64x3x256x256xf32>) -> tensor<64x3x256x256xf32>
    %2 = "tpu.swap_channel"(%1) {name = "older201", channel_order = [2 : i32, 0 : i32, 1 : i32], quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i8}} : (tensor<64x3x256x256xf32>) -> tensor<64x3x256x256xf32>
    %3 = "tpu.swap_channel"(%1) {name = "older120", channel_order = [1 : i32, 2 : i32, 0 : i32], quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i8}} : (tensor<64x3x256x256xf32>) -> tensor<64x3x256x256xf32>
    return %2,%3 : tensor<64x3x256x256xf32>,tensor<64x3x256x256xf32>
  }
}
