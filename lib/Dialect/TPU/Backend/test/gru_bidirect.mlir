module  {
  func @tpu_func(%arg0: tensor<175x1x256xf32>) -> tensor<175x2x1x128xf32> attributes {chipname = "cv183x"} {
    %0 = "tpu.weight_file"() {filename = "GRU_1_06eeeb7e.npz"} : () -> memref<10xf32>
    %1 = "tpu.input"(%arg0) {name = "input", preprocess = {aligned = false, channel_order = "bgr", keep_aspect_ratio = false, mean = [0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32], pixel_format = "BGR_PLANAR", resize_dims = [1 : i32, 256 : i32], scale = [1.000000e+00 : f32, 1.000000e+00 : f32, 1.000000e+00 : f32]}, quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<175x1x256xf32>) -> tensor<175x1x256xf32>
    %2 = "tpu.load_weight"(%0) {name = "w", storage = "NONE"} : (memref<10xf32>) -> tensor<1x384x256xf32>
    %3 = "tpu.load_weight"(%0) {name = "w_b", storage = "NONE"} : (memref<10xf32>) -> tensor<1x384x256xf32>
    %4 = "tpu.load_weight"(%0) {name = "r", storage = "NONE"} : (memref<10xf32>) -> tensor<1x384x128xf32>
    %5 = "tpu.load_weight"(%0) {name = "r_b", storage = "NONE"} : (memref<10xf32>) -> tensor<1x384x128xf32>
    %6 = "tpu.load_weight"(%0) {name = "b", storage = "NONE"} : (memref<10xf32>) -> tensor<1x768xf32>
    %7 = "tpu.load_weight"(%0) {name = "b_b", storage = "NONE"} : (memref<10xf32>) -> tensor<1x768xf32>
    %8 = "tpu.none"() : () -> none
    %9 = "tpu.gru"(%1, %2, %4, %6, %8, %8, %8, %8, %8) {bidirectional = false, linear_before_reset = true, name = "output_GRU_forward", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<175x1x256xf32>, tensor<1x384x256xf32>, tensor<1x384x128xf32>, tensor<1x768xf32>, none, none, none, none, none) -> tensor<175x1x1x128xf32>
    %10 = "tpu.reverse"(%1) {axis = 0 : i32, name = "input_reverse", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<175x1x256xf32>) -> tensor<175x1x256xf32>
    %11 = "tpu.none"() : () -> none
    %12 = "tpu.gru"(%10, %3, %5, %7, %11, %11, %11, %11, %11) {bidirectional = false, linear_before_reset = true, name = "output_GRU_backward", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<175x1x256xf32>, tensor<1x384x256xf32>, tensor<1x384x128xf32>, tensor<1x768xf32>, none, none, none, none, none) -> tensor<175x1x1x128xf32>
    %13 = "tpu.reverse"(%12) {axis = 0 : i32, name = "output_GRU_backward_reverse", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<175x1x1x128xf32>) -> tensor<175x1x1x128xf32>
    %14 = "tpu.none"() : () -> none
    %15 = "tpu.concat"(%9, %13, %14, %14, %14, %14) {axis = 1 : i32, name = "output_GRU", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<175x1x1x128xf32>, tensor<175x1x1x128xf32>, none, none, none, none) -> tensor<175x2x1x128xf32>
    return %15 : tensor<175x2x1x128xf32>
  }
}