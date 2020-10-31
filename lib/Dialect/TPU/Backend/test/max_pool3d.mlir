

module {
  func @tpu_func(%arg0: tensor<1x2x4x6x6xf32>) -> tensor<1x2x2x3x3xf32> {
    %0 = "tpu.weight_file"() {filename = "MaxPool3d_1_06eeeb7e.npz"} : () -> memref<10xf32>
    %1 = "tpu.input"(%arg0) {name = "input", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i8}} : (tensor<1x2x4x6x6xf32>) -> tensor<1x2x4x6x6xf32>
    %2 = "tpu.pool_max_3d"(%1) {name = "output_MaxPool", param = {count_include_pad = false, do_relu = false, kernel_d = 2 : i32, kernel_h = 2 : i32, kernel_w = 2 : i32, padding_b = 0 : i32, padding_d0 = 0 : i32, padding_d1 = 0 : i32, padding_l = 0 : i32, padding_r = 0 : i32, padding_t = 0 : i32, stride_d = 2 : i32, stride_h = 2 : i32, stride_w = 2 : i32}, quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i8}} : (tensor<1x2x4x6x6xf32>) -> tensor<1x2x2x3x3xf32>
    return %2 : tensor<1x2x2x3x3xf32>
  }
}