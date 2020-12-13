

module {
  func @tpu_func(%arg0: tensor<1x4096x256x1xf32>) -> tensor<1x4096x256x1xf32> {
    %0 = "tpu.weight_file"() {filename = "resnet50_1_06eeeb7e.npz"} : () -> memref<10xf32>
    %1 = "tpu.input"(%arg0) {name = "data0", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<1x4096x256x1xf32>) -> tensor<1x4096x256x1xf32>
    %2 = "tpu.square"(%1) {name = "res3a", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<1x4096x256x1xf32>) -> tensor<1x4096x256x1xf32>
    return %2 : tensor<1x4096x256x1xf32>
  }
}
