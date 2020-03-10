

module {
  func @tpu_func(%arg0: tensor<1x4x3x4xf32>) -> tensor<1x1x6x8xf32> {
    %0 = "tpu.weight_file"() {filename = "test_1_06eeeb7e.npz"} : () -> memref<10xf32>
    %1 = "tpu.input"(%arg0) {name = "data", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32}} : (tensor<1x4x3x4xf32>) -> tensor<1x4x3x4xf32>
    %2 = "tpu.reshape"(%1) {name = "618"} : (tensor<1x4x3x4xf32>) -> tensor<1x1x2x2x3x4xf32>
    %3 = "tpu.pixelshuffle"(%2) {name = "Y", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32}, upscale_factor = 2 : i32} : (tensor<1x1x2x2x3x4xf32>) -> tensor<1x1x6x8xf32>
    return %3 : tensor<1x1x6x8xf32>
  }
}
