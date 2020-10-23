

module {
  func @tpu_func(%arg0: tensor<1x1x16x512xf32>, %arg1: tensor<1x1x4096x512xf32>) -> tensor<16x4096xf32> {
    %0 = "tpu.weight_file"() {filename = "matmul-matmul_1_06eeeb7e.npz"} : () -> memref<10xf32>
    %1 = "tpu.input"(%arg0) {name = "data_A", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i8}} : (tensor<1x1x16x512xf32>) -> tensor<1x1x16x512xf32>
    %2 = "tpu.input"(%arg1) {name = "data_B_0", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i8}} : (tensor<1x1x4096x512xf32>) -> tensor<1x1x4096x512xf32>
    %3 = "tpu.reshape"(%1) {name = "matmul_layer_0_reshape_0"} : (tensor<1x1x16x512xf32>) -> tensor<16x512xf32>
    %4 = "tpu.reshape"(%2) {name = "matmul_layer_0_reshape_1"} : (tensor<1x1x4096x512xf32>) -> tensor<4096x512xf32>
    %5 = "tpu.matmul"(%3, %4) {name = "matmul_layer_0", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i8}} : (tensor<16x512xf32>, tensor<4096x512xf32>) -> tensor<16x4096xf32>
    return %5 : tensor<16x4096xf32>
  }
}
