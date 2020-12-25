

module {
  func @tpu_func(%arg0: tensor<4x128x112x112xf32>) -> tensor<4x128x56x56xf32> {
    %0 = "tpu.weight_file"() {filename = "max_pool_5ab408b04ea65.npz"} : () -> memref<10xf32>
    %1 = "tpu.input"(%arg0) {name = "data", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<4x128x112x112xf32>) -> tensor<4x128x112x112xf32>
    %2 = "tpu.pool_max_2d"(%1) {name = "max_pool", param = {count_include_pad = false, do_relu = false, kernel_h = 2 : i32, kernel_w = 2 : i32, pad_value = 0 : i32, padding_b = 0 : i32, padding_l = 0 : i32, padding_r = 0 : i32, padding_t = 0 : i32, stride_h = 2 : i32, stride_w = 2 : i32}, quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<4x128x112x112xf32>) -> tensor<4x128x56x56xf32>
    return %2 : tensor<4x128x56x56xf32>
  }
}
