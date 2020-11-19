

module {
  func @tpu_func(%arg0: tensor<1x3x112x112xf32>) -> tensor<1x64x56x56xf32> {
    %0 = "tpu.weight_file"() {filename = "googlenet_1_06eeeb7e.npz"} : () -> memref<10xf32>
    %1 = "tpu.input"(%arg0) {name = "data", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<1x3x112x112xf32>) -> tensor<1x3x112x112xf32>
    %2 = "tpu.load_weight"(%0) {name = "conv1/7x7_s2_0", storage = "NONE"} : (memref<10xf32>) -> tensor<64x3x7x7xf32>
    %3 = "tpu.load_weight"(%0) {name = "conv1/7x7_s2_1", storage = "NONE"} : (memref<10xf32>) -> tensor<64xf32>
    %4 = "tpu.none"() : () -> none
    %5 = "tpu.conv_2d"(%1, %2, %3, %4, %4, %4, %4) {name = "conv1/7x7_s2", param = {dilation_h = 1 : i32, dilation_w = 1 : i32, do_relu = false, group = 1 : i32, ins = [], is_dw = false, pad_value = 0 : i32, padding = "SAME", padding_b = 3 : i32, padding_l = 3 : i32, padding_r = 3 : i32, padding_t = 3 : i32, stride_h = 2 : i32, stride_w = 2 : i32, with_bias = true}, quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<1x3x112x112xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, none, none, none, none) -> tensor<1x64x56x56xf32>
    %8 = "tpu.lrn_one"(%5) {alpha = 9.99999974E-5 : f32, beta = 7.500000e-01 : f32, k = 1.000000e+00 : f32, local_size = 5 : i32, name = "pool1/norm1_one", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %9 = "tpu.lrn_two"(%8) {alpha = 9.99999974E-5 : f32, beta = 7.500000e-01 : f32, k = 1.000000e+00 : f32, local_size = 5 : i32, name = "pool1/norm1_two", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %10 = "tpu.lrn_three"(%9) {alpha = 9.99999974E-5 : f32, beta = 7.500000e-01 : f32, k = 1.000000e+00 : f32, local_size = 5 : i32, name = "pool1/norm1_three", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %11 = "tpu.none"() : () -> none
    %12 = "tpu.lrn"(%5, %11, %11, %10) {alpha = 9.99999974E-5 : f32, beta = 7.500000e-01 : f32, k = 1.000000e+00 : f32, local_size = 5 : i32, lrn_rshift = 0 : i32, name = "pool1/norm1", norm_region = 0 : i32, quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}, quant_data0 = 0 : i32, quant_data1 = 0 : i32, sum_rshift = 0 : i32} : (tensor<1x64x56x56xf32>, none, none, tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    return %12 : tensor<1x64x56x56xf32>
  }
}