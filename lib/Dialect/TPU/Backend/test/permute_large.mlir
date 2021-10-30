
module {
  func @tpu_func(%arg0: tensor<16x32x64x128xf32>) -> (tensor<32x16x64x128xf32>,tensor<32x16x64x128xf32>,tensor<16x64x32x128xf32>,tensor<16x128x32x64xf32>,tensor<16x64x128x32xf32>,tensor<16x128x64x32xf32>) {
    %1 = "tpu.input"(%arg0) {layer_id = 0 : i32, name = "data", quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<16x32x64x128xf32>) -> tensor<16x32x64x128xf32>
    %2 = "tpu.permute"(%1) {name = "permute_1023", order = [  1 : i32, 0 : i32, 2 : i32, 3 : i32 ],quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<16x32x64x128xf32>) -> tensor<32x16x64x128xf32>
    %3 = "tpu.permute"(%2) {name = "permute_0123", order = [  0 : i32, 1 : i32, 2 : i32, 3 : i32 ],quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<32x16x64x128xf32>) -> tensor<32x16x64x128xf32>
    %4 = "tpu.permute"(%1) {name = "permute_0213", order = [  0 : i32, 2 : i32, 1 : i32, 3 : i32 ],quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<16x32x64x128xf32>) -> tensor<16x64x32x128xf32>
    %5 = "tpu.permute"(%1) {name = "permute_0312", order = [  0 : i32, 3 : i32, 1 : i32, 2 : i32 ],quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<16x32x64x128xf32>) -> tensor<16x128x32x64xf32>
    %6 = "tpu.permute"(%1) {name = "permute_0231", order = [  0 : i32, 2 : i32, 3 : i32, 1 : i32 ],quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<16x32x64x128xf32>) -> tensor<16x64x128x32xf32>
    %7 = "tpu.permute"(%1) {name = "permute_0321", order = [  0 : i32, 3 : i32, 2 : i32, 1 : i32 ],quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<16x32x64x128xf32>) -> tensor<16x128x64x32xf32>
    return %2,%3,%4,%5,%6,%7 : tensor<32x16x64x128xf32>,tensor<32x16x64x128xf32>,tensor<16x64x32x128xf32>,tensor<16x128x32x64xf32>,tensor<16x64x128x32xf32>,tensor<16x128x64x32xf32>
  }
}
