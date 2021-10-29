
module {
  func @tpu_func(%arg0: tensor<32x16x224x224xf32>) -> (tensor<32x16x224x224xf32>,tensor<32x16x224x224xf32>,tensor<32x16x224x224xf32>) {
    %0 = "tpu.weight_file"() {filename = "relu_06eeeb7e.npz"} : () -> memref<10xf32>
    %1 = "tpu.input"(%arg0) {name = "data", quant = {mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32}} : (tensor<32x16x224x224xf32>) -> tensor<32x16x224x224xf32>
    %2 = "tpu.none"() : () -> none
    %3 = "tpu.load_weight"(%0) {name = "slope", storage = "NONE"} : (memref<10xf32>) -> tensor<1x16x1x1xf32>
    %4 = "tpu.relu"(%1) {name = "relu", quant = {mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32}} : (tensor<32x16x224x224xf32>) -> tensor<32x16x224x224xf32>
    %5 = "tpu.leaky_relu"(%1, %2, %2, %2, %2, %2, %2, %2, %2) {name = "leaky_relu", negative_slope = 1.000000e-01 : f32, quant = {mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32}} : (tensor<32x16x224x224xf32>, none, none, none, none, none, none, none, none) -> tensor<32x16x224x224xf32>
    %6 = "tpu.prelu"(%1, %3, %2, %2, %2, %2, %2, %2, %2, %2) {name = "prelu", quant = {mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32}} : (tensor<32x16x224x224xf32>, tensor<1x16x1x1xf32>, none, none, none, none, none, none, none, none) -> tensor<32x16x224x224xf32>
    return %4,%5,%6 : tensor<32x16x224x224xf32>,tensor<32x16x224x224xf32>,tensor<32x16x224x224xf32>
  }
}
