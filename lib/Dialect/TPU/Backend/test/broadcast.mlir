

module {
  func @tpu_func(%arg0: tensor<16x256x32x32xf32>, %arg1: tensor<16x1x32x32xf32>) -> (tensor<16x256x32x32xf32>, tensor<16x256x32x32xf32>, tensor<16x256x32x32xf32>) {
    %0 = "tpu.weight_file"() {filename = "bcast_06eeeb7e.npz"} : () -> memref<10xf32>
    %1 = "tpu.input"(%arg0) {name = "data0", quant = {mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32}} : (tensor<16x256x32x32xf32>) -> tensor<16x256x32x32xf32>
    %2 = "tpu.input"(%arg1) {name = "data1", quant = {mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32}} : (tensor<16x1x32x32xf32>) -> tensor<16x1x32x32xf32>
    %3 = "tpu.none"() : () -> none
    %4 = "tpu.broadcast_add"(%1, %2, %3, %3, %3, %3) {name = "bcast_add", quant = {mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32}} : (tensor<16x256x32x32xf32>, tensor<16x1x32x32xf32>, none, none, none, none) -> tensor<16x256x32x32xf32>
    %5 = "tpu.broadcast_sub"(%1, %2, %3, %3, %3, %3) {name = "bcast_sub", quant = {mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32}} : (tensor<16x256x32x32xf32>, tensor<16x1x32x32xf32>, none, none, none, none) -> tensor<16x256x32x32xf32>
    %6 = "tpu.broadcast_mul"(%1, %2, %3, %3, %3, %3) {name = "bcast_mul", quant = {mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32}} : (tensor<16x256x32x32xf32>, tensor<16x1x32x32xf32>, none, none, none, none) -> tensor<16x256x32x32xf32>
    return %4,%5,%6 : tensor<16x256x32x32xf32>, tensor<16x256x32x32xf32>, tensor<16x256x32x32xf32>
  }
}
