

module {
  func @tpu_func(%arg0: tensor<8x32x32x60xf32>) -> tensor<8x32x32x60xf32> {
    %0 = "tpu.weight_file"() {filename = "layernorm_06eeeb7e.npz"} : () -> memref<10xf32>
    %1 = "tpu.input"(%arg0) {name = "input", quant = {mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32}} : (tensor<8x32x32x60xf32>) -> tensor<8x32x32x60xf32>
    %2 = "tpu.none"() : () -> none
    %3 = "tpu.load_weight"(%0) {name = "scale", storage = "NONE"} : (memref<10xf32>) -> tensor<32x60xf32>
    %4 = "tpu.load_weight"(%0) {name = "bias", storage = "NONE"} : (memref<10xf32>) -> tensor<32x60xf32>
    %5 = "tpu.layer_norm"(%1, %3, %4, %2, %2) {name = "layer_norm", normalized_shape = [32:i32,60:i32], eps = 1.0e-5:f32, quant = {mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32}} : (tensor<8x32x32x60xf32>, tensor<32x60xf32>, tensor<32x60xf32>, none, none) -> tensor<8x32x32x60xf32>
    return %5 : tensor<8x32x32x60xf32>
  }
}
