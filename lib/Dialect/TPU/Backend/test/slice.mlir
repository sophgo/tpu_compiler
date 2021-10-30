

module {
  func @tpu_func(%arg0: tensor<4x3x64x64xf32>) -> tensor<4x32x32x32xf32> {
    %0 = "tpu.weight_file"() {filename = "res2net50_1_06eeeb7e.npz"} : () -> memref<10xf32>
    %1 = "tpu.input"(%arg0) {name = "input", quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<4x3x64x64xf32>) -> tensor<4x3x64x64xf32>
    %2 = "tpu.load_weight"(%0) {name = "conv1.weight", storage = "NONE"} : (memref<10xf32>) -> tensor<64x3x7x7xf32>
    %3 = "tpu.none"() : () -> none
    %4 = "tpu.conv_2d"(%1, %2, %3, %3, %3, %3, %3) {name = "321_Conv", param = {dilation_h = 1 : i32, dilation_w = 1 : i32, do_relu = false, group = 1 : i32, ins = [], is_dw = false, padding = "SAME", padding_b = 3 : i32, padding_l = 3 : i32, padding_r = 3 : i32, padding_t = 3 : i32, stride_h = 2 : i32, stride_w = 2 : i32, with_bias = false}, quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<4x3x64x64xf32>, tensor<64x3x7x7xf32>, none, none, none, none, none) -> tensor<4x64x32x32xf32>
    %5 = "tpu.slice"(%4) {axis = 1 : i32, name = "input.7_Split", offset = 15 : i32, quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<4x64x32x32xf32>) -> tensor<4x32x32x32xf32>
   return %5: tensor<4x32x32x32xf32>
  }
}
