
module {
  func @tpu_func(%arg0: tensor<32x64x128x256xf32>) -> (tensor<64x32x128x256xf32>,tensor<64x32x128x256xf32>,tensor<32x128x64x256xf32>,tensor<32x256x64x128xf32>,tensor<32x128x256x64xf32>,tensor<32x256x128x64xf32>) {
    %1 = "tpu.input"(%arg0) {layer_id = 0 : i32, name = "data", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i8}} : (tensor<32x64x128x256xf32>) -> tensor<32x64x128x256xf32>
    %2 = "tpu.permute"(%1) {name = "permute_1023", order0 = 1 : i32, order1 = 0 : i32, order2 = 2 : i32, order3 = 3 : i32, quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i8}} : (tensor<32x64x128x256xf32>) -> tensor<64x32x128x256xf32>
    %3 = "tpu.permute"(%2) {name = "permute_0123", order0 = 0 : i32, order1 = 1 : i32, order2 = 2 : i32, order3 = 3 : i32, quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i8}} : (tensor<64x32x128x256xf32>) -> tensor<64x32x128x256xf32>
    %4 = "tpu.permute"(%1) {name = "permute_0213", order0 = 0 : i32, order1 = 2 : i32, order2 = 1 : i32, order3 = 3 : i32, quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i8}} : (tensor<32x64x128x256xf32>) -> tensor<32x128x64x256xf32>
    %5 = "tpu.permute"(%1) {name = "permute_0312", order0 = 0 : i32, order1 = 3 : i32, order2 = 1 : i32, order3 = 2 : i32, quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i8}} : (tensor<32x64x128x256xf32>) -> tensor<32x256x64x128xf32>
    %6 = "tpu.permute"(%1) {name = "permute_0231", order0 = 0 : i32, order1 = 2 : i32, order2 = 3 : i32, order3 = 1 : i32, quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i8}} : (tensor<32x64x128x256xf32>) -> tensor<32x128x256x64xf32>
    %7 = "tpu.permute"(%1) {name = "permute_0321", order0 = 0 : i32, order1 = 3 : i32, order2 = 2 : i32, order3 = 1 : i32, quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i8}} : (tensor<32x64x128x256xf32>) -> tensor<32x256x128x64xf32>
    return %2,%3,%4,%5,%6,%7 : tensor<64x32x128x256xf32>,tensor<64x32x128x256xf32>,tensor<32x128x64x256xf32>,tensor<32x256x64x128xf32>,tensor<32x128x256x64xf32>,tensor<32x256x128x64xf32>
  }
}