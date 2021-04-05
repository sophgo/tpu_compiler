module  {
  func @tpu_func(%arg0: tensor<175x4x256xf32>) -> tensor<175x2x4x128xf32> {
    %0 = "tpu.weight_file"() {filename = "GRU_1_06eeeb7e.npz"} : () -> memref<10xf32>
    %1 = "tpu.input"(%arg0) {name = "input", preprocess = {aligned = false, channel_order = "bgr", keep_aspect_ratio = false, mean = [0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32], pixel_format = "BGR_PLANAR", resize_dims = [4 : i32, 256 : i32], scale = [1.000000e+00 : f32, 1.000000e+00 : f32, 1.000000e+00 : f32]}, quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<175x4x256xf32>) -> tensor<175x4x256xf32>
    %2 = "tpu.load_weight"(%0) {name = "w", storage = "NONE"} : (memref<10xf32>) -> tensor<2x384x256xf32>
    %3 = "tpu.load_weight"(%0) {name = "r", storage = "NONE"} : (memref<10xf32>) -> tensor<2x384x128xf32>
    %4 = "tpu.load_weight"(%0) {name = "b", storage = "NONE"} : (memref<10xf32>) -> tensor<2x768xf32>
    %5 = "tpu.none"() : () -> none
    %6 = "tpu.gru"(%1, %2, %3, %4, %5, %5, %5, %5, %5) {bidirectional = true, linear_before_reset = true, name = "output_GRU", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<175x4x256xf32>, tensor<2x384x256xf32>, tensor<2x384x128xf32>, tensor<2x768xf32>, none, none, none, none, none) -> tensor<175x2x4x128xf32>
    return %6 : tensor<175x2x4x128xf32>
  }
}
