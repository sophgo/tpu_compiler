
module {
  func @tpu_func(%arg0: tensor<32x32x32x32xf32>) -> (tensor<96x32x32x32xf32>,tensor<32x96x32x32xf32>,tensor<32x32x96x32xf32>,tensor<32x32x32x96xf32>,tensor<64x32x1024xf32>,tensor<32x64x1024xf32>,tensor<32x32x2048xf32>,tensor<2048x1024xf32>,tensor<1024x2048xf32>) {
    %1 = "tpu.input"(%arg0) {name = "data", quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<32x32x32x32xf32>) -> tensor<32x32x32x32xf32>
    %2 = "tpu.none"() : () -> none
    %3 = "tpu.permute"(%1) {name = "permute_1023", order = [  1 : i32, 0 : i32, 2 : i32, 3 : i32 ],quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<32x32x32x32xf32>) -> tensor<32x32x32x32xf32>
    %5 = "tpu.reshape"(%3) {name = "reshape_1"} : (tensor<32x32x32x32xf32>) -> tensor<32x32x1024xf32>
    %8 = "tpu.reshape"(%3) {name = "reshape_4"} : (tensor<32x32x32x32xf32>) -> tensor<1024x1024xf32>
    %9 = "tpu.concat"(%1, %1, %1, %2, %2, %2, %2) {axis = 0 : i32, name = "concat4_0", do_relu = true, quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<32x32x32x32xf32>, tensor<32x32x32x32xf32>, tensor<32x32x32x32xf32>, none, none, none, none) -> tensor<96x32x32x32xf32>
    %10 = "tpu.concat"(%1, %1, %1, %2, %2, %2, %2) {axis = 1 : i32, name = "concat4_1", do_relu = true, quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<32x32x32x32xf32>, tensor<32x32x32x32xf32>, tensor<32x32x32x32xf32>, none, none, none, none) -> tensor<32x96x32x32xf32>
    %11 = "tpu.concat"(%1, %1, %1, %2, %2, %2, %2) {axis = 2 : i32, name = "concat4_2", do_relu = false, quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<32x32x32x32xf32>, tensor<32x32x32x32xf32>, tensor<32x32x32x32xf32>, none, none, none, none) -> tensor<32x32x96x32xf32>
    %12 = "tpu.concat"(%1, %1, %1, %2, %2, %2, %2) {axis = 3 : i32, name = "concat4_3", quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<32x32x32x32xf32>, tensor<32x32x32x32xf32>, tensor<32x32x32x32xf32>, none, none, none, none) -> tensor<32x32x32x96xf32>
    %13 = "tpu.concat"(%5, %5, %2, %2, %2, %2) {axis = 0 : i32, name = "concat3_0", quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<32x32x1024xf32>, tensor<32x32x1024xf32>, none, none, none, none) -> tensor<64x32x1024xf32>
    %14 = "tpu.concat"(%5, %5, %2, %2, %2, %2) {axis = 1 : i32, name = "concat3_1", quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<32x32x1024xf32>, tensor<32x32x1024xf32>, none, none, none, none) -> tensor<32x64x1024xf32>
    %15 = "tpu.concat"(%5, %5, %2, %2, %2, %2) {axis = 2 : i32, name = "concat3_2", do_relu = true, quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<32x32x1024xf32>, tensor<32x32x1024xf32>, none, none, none, none) -> tensor<32x32x2048xf32>
    %16 = "tpu.concat"(%8, %8, %2, %2, %2, %2) {axis = 0 : i32, name = "concat2_0", quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<1024x1024xf32>, tensor<1024x1024xf32>, none, none, none, none) -> tensor<2048x1024xf32>
    %17 = "tpu.concat"(%8, %8, %2, %2, %2, %2) {axis = 1 : i32, name = "concat2_1", quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<1024x1024xf32>, tensor<1024x1024xf32>, none, none, none, none) -> tensor<1024x2048xf32>
    return %9,%10,%11,%12,%13,%14,%15,%16,%17 : tensor<96x32x32x32xf32>,tensor<32x96x32x32xf32>,tensor<32x32x96x32xf32>,tensor<32x32x32x96xf32>,tensor<64x32x1024xf32>,tensor<32x64x1024xf32>,tensor<32x32x2048xf32>,tensor<2048x1024xf32>,tensor<1024x2048xf32>
  }
}
