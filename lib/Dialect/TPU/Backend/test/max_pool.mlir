

module {
  func @tpu_func(%arg0: tensor<8x128x814x814xf32>) -> tensor<8x128x408x408xf32> {
    %0 = "tpu.weight_file"() {filename = "inceptionv4_2_5ab408b04ea65.npz"} : () -> memref<10xf32>
    %1 = "tpu.input"(%arg0) {layer_id = 0 : i32, name = "data", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<8x128x814x814xf32>) -> tensor<8x128x814x814xf32>
    %2 = "tpu.pool_max_2d"(%1) {layer_id = 149 : i32, name = "reduction_b_pool", param = {count_include_pad = false, do_relu = false, kernel_h = 3 : i32, kernel_w = 3 : i32, padding_b = 1 : i32, padding_l = 2 : i32, padding_r = 1 : i32, padding_t = 2 : i32, stride_h = 2 : i32, stride_w = 2 : i32}, quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<8x128x814x814xf32>) -> tensor<8x128x408x408xf32>
    return %2 : tensor<8x128x408x408xf32>
  }
}
