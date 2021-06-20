module  {
  func @tpu_func(%arg0: tensor<1x2x4x4xf32>) -> tensor<1x8x2x2xf32> {
    %0 = "tpu.weight_file"() {filename = "yolov2416_2_35047239d74b.npz"} : () -> memref<10xf32>
    %1 = "tpu.input"(%arg0) {name = "data", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
    %2 = "tpu.reorg"(%1) {name = "reorg1", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}, stride = 2 : i32} : (tensor<1x2x4x4xf32>) -> tensor<1x8x2x2xf32>
    return %2 : tensor<1x8x2x2xf32>
  }
}

