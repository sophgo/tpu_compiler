

module {
  func @tpu_func(%arg0: tensor<1x1x256x1xf32>, %arg1: tensor<1x4096x256x1xf32>) -> tensor<1x4096xf32> {
    %0 = "tpu.weight_file"() {filename = "model_weight.npz"} : () -> memref<10xf32>
    %1 = "tpu.input"(%arg0) {name = "input_x", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i8}} : (tensor<1x1x256x1xf32>) -> tensor<1x1x256x1xf32>
    %2 = "tpu.input"(%arg1) {name = "input_y", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i8}} : (tensor<1x4096x256x1xf32>) -> tensor<1x4096x256x1xf32>
    %3 = "tpu.none"() : () -> none
    %4 = "tpu.broadcast_sub"(%2, %1, %3, %3, %3, %3) {name = "sub", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i8}} : (tensor<1x4096x256x1xf32>, tensor<1x1x256x1xf32>, none, none, none, none) -> tensor<1x4096x256x1xf32>
    %5 = "tpu.reshape"(%4) {name = "sub_reshape"} : (tensor<1x4096x256x1xf32>) -> tensor<1x4096x16x16xf32>
    %6 = "tpu.quadratic_sum"(%5) {axis = 1 : i32, high_precision = true, name = "out", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i8}} : (tensor<1x4096x16x16xf32>) -> tensor<1x4096x1x1xf32>
    %7 = "tpu.reshape"(%6) {name = "out_reshape"} : (tensor<1x4096x1x1xf32>) -> tensor<1x4096xf32>
    return %7 : tensor<1x4096xf32>
  }
}