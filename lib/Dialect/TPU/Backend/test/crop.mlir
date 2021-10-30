
module {
  func @tpu_func(%arg0: tensor<32x16x224x224xf32>) -> (tensor<32x10x220x220xf32>,tensor<16x10x10x10xf32>) {
    %1 = "tpu.input"(%arg0) {name = "data", quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<32x16x224x224xf32>) -> tensor<32x16x224x224xf32>
    %2 = "tpu.crop"(%1) {name = "crop_1", crop_shape = [32:i32,10:i32,220:i32,220:i32],crop_offset = [0:i32,6:i32,4:i32,0:i32], quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<32x16x224x224xf32>) -> tensor<32x10x220x220xf32>
    %3 = "tpu.crop"(%1) {name = "crop_2", crop_shape = [16:i32,10:i32,10:i32,10:i32],crop_offset = [2:i32, 32:i32,10:i32,16:i32], quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<32x16x224x224xf32>) -> tensor<16x10x10x10xf32>
    return %2,%3 : tensor<32x10x220x220xf32>,tensor<16x10x10x10xf32>
  }
}
