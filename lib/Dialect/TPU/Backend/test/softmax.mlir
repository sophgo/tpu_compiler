module  {
  func @tpu_func(%arg0: tensor<4x1x256x512xf32>) -> (tensor<4x1x256x512xf32>, tensor<4x1x256x512xf32>, tensor<4x1x256x512xf32>) {
    %0 = "tpu.weight_file"() {filename = "Softmax_2_12b313b6ab91.npz"} : () -> memref<10xf32>
    %1 = "tpu.input"(%arg0) {name = "input", preprocess = {aligned = false, channel_order = "bgr", keep_aspect_ratio = false, mean = [0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32], pixel_format = "BGR_PLANAR", resize_dims = [256 : i32, 512 : i32], scale = [1.000000e+00 : f32, 1.000000e+00 : f32, 1.000000e+00 : f32]}, quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<4x1x256x512xf32>) -> tensor<4x1x256x512xf32>
    %2 = "tpu.load_weight"(%0) {name = "X0_add_weight_scale", storage = "FP32"} : (memref<10xf32>) -> tensor<1x1x1x1x1xf32>
    %3 = "tpu.load_weight"(%0) {name = "X0_add_bias_scale", storage = "FP32"} : (memref<10xf32>) -> tensor<1xf32>
    %4 = "tpu.none"() : () -> none
    %5 = "tpu.conv_2d"(%1, %2, %3, %4, %4, %4, %4) {name = "X0_Neg", param = {dilation_h = 1 : i32, dilation_w = 1 : i32, do_relu = false, group = 1 : i32, ins = [], is_dw = true, pad_value = 0 : i32, padding = "VALID", padding_b = 0 : i32, padding_l = 0 : i32, padding_r = 0 : i32, padding_t = 0 : i32, stride_h = 1 : i32, stride_w = 1 : i32, with_bias = true}, quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<4x1x256x512xf32>, tensor<1x1x1x1x1xf32>, tensor<1xf32>, none, none, none, none) -> tensor<4x1x256x512xf32>
    %6 = "tpu.none"() : () -> none
    %7 = "tpu.softmax"(%5, %6, %6, %6, %6) {axis = 1 : i32, name = "X1_Softmax", quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<4x1x256x512xf32>, none, none, none, none) -> tensor<4x1x256x512xf32>
    %8 = "tpu.none"() : () -> none
    %9 = "tpu.softmax"(%5, %8, %8, %8, %8) {axis = 2 : i32, name = "X2_Softmax", quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<4x1x256x512xf32>, none, none, none, none) -> tensor<4x1x256x512xf32>
    %10 = "tpu.none"() : () -> none
    %11 = "tpu.softmax"(%5, %10, %10, %10, %10) {axis = 3 : i32, name = "X3_Softmax", quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<4x1x256x512xf32>, none, none, none, none) -> tensor<4x1x256x512xf32>
    return %7, %9, %11 : tensor<4x1x256x512xf32>, tensor<4x1x256x512xf32>, tensor<4x1x256x512xf32>
  }
}

