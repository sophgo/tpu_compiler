module {
  func @tpu_func(%arg0: tensor<1x1x32x8xf32>) -> tensor<1x1x64x4xf32> {
    %0 = "tpu.weight_file"() {filename = "conv_06eeeb7e.npz"} : () -> memref<10xf32>
    %1 = "tpu.input"(%arg0) {name = "data", quant = {mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32}} : (tensor<1x1x32x8xf32>) -> tensor<1x1x32x8xf32>
    %2 = "tpu.load_weight"(%0) {name = "filter", storage = "NONE"} : (memref<10xf32>) -> tensor<1x1x3x3xf32>
    %3 = "tpu.none"() : () -> none
    %4 = "tpu.conv_2d"(%1, %2, %3, %3, %3, %3, %3) {name = "conv", param = {dilation_h = 1 : i32, dilation_w = 1 : i32, do_relu = true, group = 1 : i32, ins = [], is_dw = false, pad_value = 0 : i32, padding = "SAME", padding_b = 1 : i32, padding_l = 1 : i32, padding_r = 1 : i32, padding_t = 1 : i32, stride_h = 2 : i32, stride_w = 2 : i32, with_bias = false}, quant = {mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32}} : (tensor<1x1x32x8xf32>, tensor<1x1x3x3xf32>, none, none, none, none, none) -> tensor<1x1x16x4xf32>
    %5 = "tpu.load_weight"(%0) {name = "table", storage = "NONE"} : (memref<10xf32>) -> tensor<1x1x1000x4xf32>
    %6 = "tpu.embedding"(%4, %5) {name = "embedding", quant = {mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32}} : (tensor<1x1x16x4xf32>, tensor<1x1x1000x4xf32>) -> tensor<1x1x64x4xf32>
    return %6 : tensor<1x1x64x4xf32>
  }
}
