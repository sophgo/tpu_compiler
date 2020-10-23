module {
  func @tpu_func(%arg0: tensor<1x128x64x128xf32>) -> tensor<1x128x64x128xf32> {
    %0 = "tpu.weight_file"() {filename = "erfnet_1_06eeeb7e.npz"} : () -> memref<10xf32>
    %1 = "tpu.input"(%arg0) {name = "data0", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32}} : (tensor<1x128x64x128xf32>) -> tensor<1x128x64x128xf32>
    %2 = "tpu.load_weight"(%0) {name = "NBD12_0", storage = "NONE"} : (memref<10xf32>) -> tensor<128x128x3x1xf32>
    %3 = "tpu.load_weight"(%0) {name = "NBD12_1", storage = "NONE"} : (memref<10xf32>) -> tensor<128xf32>
    %4 = "tpu.none"() : () -> none
    %5 = "tpu.conv_2d"(%1, %2, %3, %4, %4, %4, %4) {name = "NBD12", param = {dilation_h = 16 : i32, dilation_w = 16 : i32, do_relu = false, group = 1 : i32, ins = [], is_dw = false, padding = "SAME", padding_b = 16 : i32, padding_l = 0 : i32, padding_r = 0 : i32, padding_t = 16 : i32, stride_h = 1 : i32, stride_w = 1 : i32, with_bias = true}, quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32}} : (tensor<1x128x64x128xf32>, tensor<128x128x3x1xf32>, tensor<128xf32>, none, none, none, none) -> tensor<1x128x64x128xf32>
    return %5 : tensor<1x128x64x128xf32>
  }
}
