

module {
  func @tpu_func(%arg0: tensor<1x512x8x6xf32>) -> tensor<1x1024x16x12xf32> {
    %0 = "tpu.weight_file"() {filename = "alphapose_1_06eeeb7e.npz"} : () -> memref<10xf32>
    %1 = "tpu.input"(%arg0) {name = "input", quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<1x512x8x6xf32>) -> tensor<1x512x8x6xf32>
    %2 = "tpu.load_weight"(%0) {name = "preact.layer4.2.conv3.weight", storage = "NONE"} : (memref<10xf32>) -> tensor<2048x512x1x1xf32>
    %3 = "tpu.none"() : () -> none
    %4 = "tpu.conv_2d"(%1, %2, %3, %3, %3, %3, %3) {name = "613_Conv", param = {dilation_h = 1 : i32, dilation_w = 1 : i32, do_relu = false, group = 1 : i32, ins = [], is_dw = false, pad_value = 0 : i32, padding = "VALID", padding_b = 0 : i32, padding_l = 0 : i32, padding_r = 0 : i32, padding_t = 0 : i32, stride_h = 1 : i32, stride_w = 1 : i32, with_bias = false}, quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<1x512x8x6xf32>, tensor<2048x512x1x1xf32>, none, none, none, none, none) -> tensor<1x2048x8x6xf32>
    %5 = "tpu.pixelshuffle"(%4) {mode = "CRD", name = "619_Transpose", quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}, upscale_factor = 2 : i32} : (tensor<1x2048x8x6xf32>) -> tensor<1x512x16x12xf32>
    %6 = "tpu.load_weight"(%0) {name = "duc1.conv.weight", storage = "NONE"} : (memref<10xf32>) -> tensor<1024x512x3x3xf32>
    %7 = "tpu.none"() : () -> none
    %8 = "tpu.conv_2d"(%5, %6, %7, %7, %7, %7, %7) {name = "622_Conv", param = {dilation_h = 1 : i32, dilation_w = 1 : i32, do_relu = false, group = 1 : i32, ins = [], is_dw = false, pad_value = 0 : i32, padding = "SAME", padding_b = 1 : i32, padding_l = 1 : i32, padding_r = 1 : i32, padding_t = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32, with_bias = false}, quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<1x512x16x12xf32>, tensor<1024x512x3x3xf32>, none, none, none, none, none) -> tensor<1x1024x16x12xf32>
   return %8 : tensor<1x1024x16x12xf32>
  }
}
