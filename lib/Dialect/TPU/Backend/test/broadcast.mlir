

module {
  func @tpu_func(%arg0: tensor<4x4092x28x28xf32>, %arg1: tensor<4x1x28x28xf32>) -> (tensor<4x4092x28x28xf32>, tensor<4x4092x28x28xf32>, tensor<4x4092x28x28xf32>) {
    %1 = "tpu.input"(%arg0) {name = "data0", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<4x4092x28x28xf32>) -> tensor<4x4092x28x28xf32>
    %2 = "tpu.input"(%arg1) {name = "data1", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<4x1x28x28xf32>) -> tensor<4x1x28x28xf32>
    %3 = "tpu.none"() : () -> none
    %4 = "tpu.broadcast_add"(%1, %2, %3, %3, %3, %3) {name = "bcast_add", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<4x4092x28x28xf32>, tensor<4x1x28x28xf32>, none, none, none, none) -> tensor<4x4092x28x28xf32>
    %5 = "tpu.broadcast_sub"(%1, %2, %3, %3, %3, %3) {name = "bcast_sub", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<4x4092x28x28xf32>, tensor<4x1x28x28xf32>, none, none, none, none) -> tensor<4x4092x28x28xf32>
    %6 = "tpu.broadcast_mul"(%1, %2, %3, %3, %3, %3) {name = "bcast_mul", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<4x4092x28x28xf32>, tensor<4x1x28x28xf32>, none, none, none, none) -> tensor<4x4092x28x28xf32>
    return %4,%5,%6 : tensor<4x4092x28x28xf32>, tensor<4x4092x28x28xf32>, tensor<4x4092x28x28xf32>
  }
}
