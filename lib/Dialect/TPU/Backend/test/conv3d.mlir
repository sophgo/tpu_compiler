

module {
  func @tpu_func(%arg0: tensor<1x2x4x3x3xf32>) -> tensor<1x4x3x2x2xf32> {
    %0 = "tpu.weight_file"() {filename = "Conv3d_1_06eeeb7e.npz"} : () -> memref<10xf32>
    %1 = "tpu.input"(%arg0) {name = "input", quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<1x2x4x3x3xf32>) -> tensor<1x2x4x3x3xf32>
    %2 = "tpu.load_weight"(%0) {name = "conv_w", storage = "NONE"} : (memref<10xf32>) -> tensor<4x2x2x2x2xf32>
    %3 = "tpu.none"() : () -> none
    %4 = "tpu.conv_3d"(%1, %2, %3, %3, %3, %3, %3) {name = "output_Conv", param = {dilation_d = 1 : i32, dilation_h = 1 : i32, dilation_w = 1 : i32, do_relu = false, group = 1 : i32, ins = [], is_dw = false, padding = "VALID", padding_b = 0 : i32, padding_d0 = 0 : i32, padding_d1 = 0 : i32, padding_l = 0 : i32, padding_r = 0 : i32, padding_t = 0 : i32, stride_d = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32, with_bias = false}, quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<1x2x4x3x3xf32>, tensor<4x2x2x2x2xf32>, none, none, none, none, none) -> tensor<1x4x3x2x2xf32>
    return %4 : tensor<1x4x3x2x2xf32>
  }
}
