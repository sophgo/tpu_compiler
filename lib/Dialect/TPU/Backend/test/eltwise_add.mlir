

module {
  func @tpu_func(%arg0: tensor<4x512x28x28xf32>, %arg1: tensor<4x512x28x28xf32>) -> tensor<4x512x28x28xf32> {
    %0 = "tpu.weight_file"() {filename = "resnet50_1_06eeeb7e.npz"} : () -> memref<10xf32>
    %1 = "tpu.input"(%arg0) {name = "data0", quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<4x512x28x28xf32>) -> tensor<4x512x28x28xf32>
    %2 = "tpu.input"(%arg1) {name = "data1", quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<4x512x28x28xf32>) -> tensor<4x512x28x28xf32>
    %3 = "tpu.none"() : () -> none
    %4 = "tpu.eltwise_add"(%1, %2, %3, %3, %3, %3) {name = "res3a", quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<4x512x28x28xf32>, tensor<4x512x28x28xf32>, none, none, none, none) -> tensor<4x512x28x28xf32>
    return %4 : tensor<4x512x28x28xf32>
  }
}
