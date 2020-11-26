module {
  func @tpu_func(%arg0: tensor<1x6059x1x1xf32>) -> tensor<1x6059x1x1xf32> {
    %0 = "tpu.weight_file"() {filename = "123.npz"} : () -> memref<10xf32>
    %1 = "tpu.input"(%arg0) {name = "data", preprocess = {color_order = "bgr", input_scale = 1.000000e+00 : f32, mean = [1.040000e+02 : f32, 1.170000e+02 : f32, 1.230000e+02 : f32], raw_scale = 2.550000e+02 : f32, std = [1.000000e+00 : f32, 1.000000e+00 : f32, 1.000000e+00 : f32]}, quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<1x6059x1x1xf32>) -> tensor<1x6059x1x1xf32>
    %2 = "tpu.load_weight"(%0) {name = "NBD12_0", storage = "NONE"} : (memref<10xf32>) -> tensor<128x128x3x1xf32>
    %3 = "tpu.load_weight"(%0) {name = "NBD12_1", storage = "NONE"} : (memref<10xf32>) -> tensor<128xf32>
    %4 = "tpu.none"() : () -> none
    %5 = "tpu.softmax"(%1, %4, %4, %4, %4) {axis = 1 : i32, name = "prob", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<1x6059x1x1xf32>, none, none, none, none) -> tensor<1x6059x1x1xf32>
    return %5 : tensor<1x6059x1x1xf32>
  }
}