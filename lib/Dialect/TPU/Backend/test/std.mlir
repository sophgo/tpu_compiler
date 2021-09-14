

module {
  func @tpu_func(%arg0: tensor<8x32x32x60xf32>) -> tensor<8x32xf32> {
    %0 = "tpu.weight_file"() {filename = "std_06eeeb7e.npz"} : () -> memref<10xf32>
    %1 = "tpu.input"(%arg0) {name = "input", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<8x32x32x60xf32>) -> tensor<8x32x32x60xf32>
    %2 = "tpu.none"() : () -> none
    %3 = "tpu.load_weight"(%0) {name = "slope", storage = "NONE"} : (memref<10xf32>) -> tensor<1x32x1x1xf32>
    %4 = "tpu.prelu"(%1, %3, %2, %2, %2, %2, %2, %2, %2, %2) {name = "prelu", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<8x32x32x60xf32>, tensor<1x32x1x1xf32>, none, none, none, none, none, none, none, none) -> tensor<8x32x32x60xf32>
    %5 = "tpu.std"(%4, %2, %2) {name = "std", start_dim = 2:i32, unbiased = true, quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<8x32x32x60xf32>, none, none) -> tensor<8x32xf32>
    return %5 : tensor<8x32xf32>
  }
}
