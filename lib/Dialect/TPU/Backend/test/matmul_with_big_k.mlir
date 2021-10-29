

module {
  func @tpu_func(%arg0: tensor<1x16x1x4098xf32>, %arg1: tensor<1x4098x1x4096xf32>) -> tensor<16x4096xf32> {
    %0 = "tpu.weight_file"() {filename = "resnet50_1_06eeeb7e.npz"} : () -> memref<10xf32>
    %1 = "tpu.input"(%arg0) {name = "data0", quant = {mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32}} : (tensor<1x16x1x4098xf32>) -> tensor<1x16x1x4098xf32>
    %2 = "tpu.input"(%arg1) {name = "data1", quant = {mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32}} : (tensor<1x4098x1x4096xf32>) -> tensor<1x4098x1x4096xf32>
    %3 = "tpu.reshape"(%1) {name = "left_Reshape"} : (tensor<1x16x1x4098xf32>) -> tensor<16x4098xf32>
    %4 = "tpu.reshape"(%2) {name = "right_Reshape"} : (tensor<1x4098x1x4096xf32>) -> tensor<4098x4096xf32>
    %5 = "tpu.none"() : () -> none
    %6 = "tpu.matmul"(%3, %4, %5, %5, %5, %5) {do_relu = false, name = "matmul", quant = {mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32}} : (tensor<16x4098xf32>, tensor<4098x4096xf32>, none, none, none, none) -> tensor<16x4096xf32>
    return %6 : tensor<16x4096xf32>
  }
}
