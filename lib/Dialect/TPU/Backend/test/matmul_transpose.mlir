

module {
  func @tpu_func(%arg0: tensor<1x16x1x256xf32>, %arg1: tensor<1x4096x1x256xf32>) -> tensor<16x4096xf32> {
    %0 = "tpu.weight_file"() {filename = "resnet50_1_06eeeb7e.npz"} : () -> memref<10xf32>
    %1 = "tpu.input"(%arg0) {name = "data0", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i8}} : (tensor<1x16x1x256xf32>) -> tensor<1x16x1x256xf32>
    %2 = "tpu.input"(%arg1) {name = "data1", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i8}} : (tensor<1x4096x1x256xf32>) -> tensor<1x4096x1x256xf32>
    %3 = "tpu.reshape"(%1) {name = "left_Reshape"} : (tensor<1x16x1x256xf32>) -> tensor<16x256xf32>
    %4 = "tpu.reshape"(%2) {name = "right_Reshape"} : (tensor<1x4096x1x256xf32>) -> tensor<4096x256xf32>
    %5 = "tpu.matmul"(%3, %4) {do_relu = false, name = "matmul", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i8}} : (tensor<16x256xf32>, tensor<4096x256xf32>) -> tensor<16x4096xf32>
    return %5 : tensor<16x4096xf32>
  }
}
