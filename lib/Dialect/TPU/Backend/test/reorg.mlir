module  {
  func @tpu_func(%arg0: tensor<4x64x68x120xf32>) -> tensor<4x256x34x60xf32> {
    %0 = "tpu.weight_file"() {filename = "weight.npz"} : () -> memref<10xf32>
    %1 = "tpu.input"(%arg0) {name = "data", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<4x64x68x120xf32>) -> tensor<4x64x68x120xf32>
    %2 = "tpu.reorg"(%1) {name = "reorg", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}, stride = 2 : i32} : (tensor<4x64x68x120xf32>) -> tensor<4x256x34x60xf32>
    return %2 : tensor<4x256x34x60xf32>
  }
}

