module  {
  func @tpu_func(%arg0: tensor<175x2x256xf32>, %arg1: tensor<2x2x128xf32>) -> tensor<175x2x2x128xf32> {
    %0 = "tpu.weight_file"() {filename = "GRU_1_06eeeb7e.npz"} : () -> memref<10xf32>
    %1 = "tpu.input"(%arg0) {name = "input", preprocess = {aligned = false, channel_order = "bgr", keep_aspect_ratio = false, mean = [0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32], pixel_format = "BGR_PLANAR", resize_dims = [2 : i32, 256 : i32], scale = [1.000000e+00 : f32, 1.000000e+00 : f32, 1.000000e+00 : f32]}, quant = {mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32}} : (tensor<175x2x256xf32>) -> tensor<175x2x256xf32>
    %2 = "tpu.load_weight"(%0) {name = "w_FC", storage = "NONE"} : (memref<10xf32>) -> tensor<768x256xf32>
    %3 = "tpu.load_weight"(%0) {name = "b_FC", storage = "NONE"} : (memref<10xf32>) -> tensor<768xf32>
    %4 = "tpu.none"() : () -> none
    %5 = "tpu.fully_connected"(%1, %2, %3, %4, %4, %4, %4) {name = "output_FC", quant = {mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32}} : (tensor<175x2x256xf32>, tensor<768x256xf32>, tensor<768xf32>, none, none, none, none) -> tensor<175x2x768xf32>
    %6 = "tpu.load_weight"(%0) {name = "r", storage = "NONE"} : (memref<10xf32>) -> tensor<2x384x128xf32>
    %7 = "tpu.load_weight"(%0) {name = "b_recurrence", storage = "NONE"} : (memref<10xf32>) -> tensor<2x384xf32>
    %8 = "tpu.none"() : () -> none
    %9 = "tpu.input"(%arg1) {name = "h", quant = {mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32}} : (tensor<2x2x128xf32>) -> tensor<2x2x128xf32>
    %10 = "tpu.gru"(%5, %6, %7, %9, %8, %8, %8, %8) {bidirectional = true, linear_before_reset = true, name = "output_GRU", quant = {mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32}} : (tensor<175x2x768xf32>, tensor<2x384x128xf32>, tensor<2x384xf32>, tensor<2x2x128xf32>, none, none, none, none) -> tensor<175x2x2x128xf32>
    return %10 : tensor<175x2x2x128xf32>
  }
}
