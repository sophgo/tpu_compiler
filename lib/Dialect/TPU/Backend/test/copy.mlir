
module {
  func @tpu_func(%arg0: tensor<1x3x5x9xf32>) -> (tensor<1x5x3x9xf32>) {
    %1 = "tpu.input"(%arg0) {name = "data", quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<1x3x5x9xf32>) -> tensor<1x3x5x9xf32>
    %2 = "tpu.copy"(%1) {name = "copy", shape = [1:i32,3:i32,5:i32,9:i32],input_stride = [135:i32,45:i32,9:i32,1:i32],output_stride = [135:i32,9:i32,27:i32,1:i32], quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<1x3x5x9xf32>) -> tensor<1x5x3x9xf32>
    return %2 : tensor<1x5x3x9xf32>
  }
}
