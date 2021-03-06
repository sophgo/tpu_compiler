module {
  func @tpu_func(%arg0: tensor<1x64x288x512xf32>) -> tensor<1x1x288x512xf32> {
    %0 = "tpu.weight_file"() {filename = "conv_06eeeb7e.npz"} : () -> memref<10xf32>
    %1 = "tpu.input"(%arg0) {name = "data", quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<1x64x288x512xf32>) -> tensor<1x64x288x512xf32>
    %2 = "tpu.load_weight"(%0) {name = "filter", storage = "NONE"} : (memref<10xf32>) -> tensor<1x64x3x3xf32>
    %3 = "tpu.load_weight"(%0) {name = "bias", storage = "NONE"} : (memref<10xf32>) -> tensor<1xf32>
    %4 = "tpu.none"() : () -> none
    %5 = "tpu.conv_2d"(%1, %2, %3, %4, %4, %4, %4) {name = "cls", param = {dilation_h = 1 : i32, dilation_w = 1 : i32, do_relu = false, group = 1 : i32, ins = [], is_dw = false, pad_value = 0 : i32, padding = "SAME", padding_b = 1 : i32, padding_l = 1 : i32, padding_r = 1 : i32, padding_t = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32, with_bias = true}, quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<1x64x288x512xf32>, tensor<1x64x3x3xf32>, tensor<1xf32>, none, none, none, none) -> tensor<1x1x288x512xf32>
    return %5 : tensor<1x1x288x512xf32>
  }
}
