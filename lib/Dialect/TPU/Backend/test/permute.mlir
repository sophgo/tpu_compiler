
module {
  func @tpu_func(%arg0: tensor<16x16x16x16xf32>) -> (tensor<16x16x16x16xf32>,tensor<16x16x16x16xf32>,tensor<16x16x16x16xf32>,tensor<16x16x16x16xf32>,tensor<16x16x16x16xf32>,tensor<16x16x16x16xf32>,tensor<16x16x16x16xf32>,tensor<16x16x16x16xf32>,tensor<16x16x16x16xf32>,tensor<16x16x16x16xf32>) {
    %1 = "tpu.input"(%arg0) {name = "data", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<16x16x16x16xf32>) -> tensor<16x16x16x16xf32>
    %2 = "tpu.permute"(%1) {name = "permute_1023", order0 = 1 : i32, order1 = 0 : i32, order2 = 2 : i32, order3 = 3 : i32, quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<16x16x16x16xf32>) -> tensor<16x16x16x16xf32>
    %3 = "tpu.permute"(%2) {name = "permute_0123", order0 = 0 : i32, order1 = 1 : i32, order2 = 2 : i32, order3 = 3 : i32, quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<16x16x16x16xf32>) -> tensor<16x16x16x16xf32>
    %4 = "tpu.permute"(%1) {name = "permute_0213", order0 = 0 : i32, order1 = 2 : i32, order2 = 1 : i32, order3 = 3 : i32, quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<16x16x16x16xf32>) -> tensor<16x16x16x16xf32>
    %5 = "tpu.permute"(%1) {name = "permute_2013", order0 = 2 : i32, order1 = 0 : i32, order2 = 1 : i32, order3 = 3 : i32, quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<16x16x16x16xf32>) -> tensor<16x16x16x16xf32>
    %6 = "tpu.permute"(%1) {name = "permute_2103", order0 = 2 : i32, order1 = 1 : i32, order2 = 0 : i32, order3 = 3 : i32, quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<16x16x16x16xf32>) -> tensor<16x16x16x16xf32>
    %7 = "tpu.permute"(%1) {name = "permute_1203", order0 = 1 : i32, order1 = 2 : i32, order2 = 0 : i32, order3 = 3 : i32, quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<16x16x16x16xf32>) -> tensor<16x16x16x16xf32>
    %8 = "tpu.permute"(%1) {name = "permute_0312", order0 = 0 : i32, order1 = 3 : i32, order2 = 1 : i32, order3 = 2 : i32, quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<16x16x16x16xf32>) -> tensor<16x16x16x16xf32>
    %9 = "tpu.permute"(%1) {name = "permute_0321", order0 = 0 : i32, order1 = 3 : i32, order2 = 2 : i32, order3 = 1 : i32, quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<16x16x16x16xf32>) -> tensor<16x16x16x16xf32>
    %10 = "tpu.permute"(%1) {name = "permute_0231", order0 = 0 : i32, order1 = 2 : i32, order2 = 3 : i32, order3 = 1 : i32, quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<16x16x16x16xf32>) -> tensor<16x16x16x16xf32>
    %11 = "tpu.permute"(%1) {name = "permute_0132", order0 = 0 : i32, order1 = 1 : i32, order2 = 3 : i32, order3 = 2 : i32, quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<16x16x16x16xf32>) -> tensor<16x16x16x16xf32>
    return %2,%3,%4,%5,%6,%7,%8,%9,%10,%11 : tensor<16x16x16x16xf32>,tensor<16x16x16x16xf32>,tensor<16x16x16x16xf32>,tensor<16x16x16x16xf32>,tensor<16x16x16x16xf32>,tensor<16x16x16x16xf32>,tensor<16x16x16x16xf32>,tensor<16x16x16x16xf32>,tensor<16x16x16x16xf32>,tensor<16x16x16x16xf32>
  }
}
