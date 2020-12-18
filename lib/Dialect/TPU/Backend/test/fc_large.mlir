module {
  func @tpu_func(%arg0: tensor<1x512x7x7xf32>) -> tensor<1x4096xf32> {
    %0 = "tpu.weight_file"() {filename = "fc_06eeeb7e.npz"} : () -> memref<10xf32>
    %1 = "tpu.input"(%arg0) {name = "data", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %2 = "tpu.none"() : () -> none
    %3 = "tpu.load_weight"(%0) {name = "filter", storage = "NONE"} : (memref<10xf32>) -> tensor<4096x25088xf32>
    %4 = "tpu.load_weight"(%0) {name = "bias", storage = "NONE"} : (memref<10xf32>) -> tensor<4096xf32>
    %5 = "tpu.fully_connected"(%1, %3, %4, %2, %2, %2, %2) {name = "fc_large", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<1x512x7x7xf32>, tensor<4096x25088xf32>, tensor<4096xf32>, none, none, none, none) -> tensor<1x4096xf32>
   return %5 : tensor<1x4096xf32>
  }
}
