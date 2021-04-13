module  {
  func @tpu_func(%arg0: tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32> {
    %0 = "tpu.weight_file"() {filename = "MatMul_1_06eeeb7e.npz"} : () -> memref<10xf32>
    %1 = "tpu.input"(%arg0) {name = "input", preprocess = {aligned = false, channel_order = "bgr", keep_aspect_ratio = false, mean = [0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32], pixel_format = "BGR_PLANAR", resize_dims = [40 : i32, 64 : i32], scale = [1.000000e+00 : f32, 1.000000e+00 : f32, 1.000000e+00 : f32]}, quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
    %2 = "tpu.none"() : () -> none
    %4 = "tpu.load_weight"(%0) {name = "slope", storage = "NONE"} : (memref<10xf32>) -> tensor<1x16x1x1xf32>
    %5 = "tpu.prelu"(%1, %4, %2, %2, %2, %2, %2, %2, %2, %2) {name = "prelu", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<1x16x16x16xf32>, tensor<1x16x1x1xf32>, none, none, none, none, none, none, none, none) -> tensor<1x16x16x16xf32>
    %6 = "tpu.permute"(%5) {name = "permute1", order0 = 0 : i32, order1 = 2 : i32, order2 = 1 : i32, order3 = 3 : i32, quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
    %7 = "tpu.matmul"(%1, %6, %2, %2, %2, %2) {name = "output_MatMul", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<1x16x16x16xf32>, tensor<1x16x16x16xf32>, none, none, none, none) -> tensor<1x16x16x16xf32>
    %8 = "tpu.permute"(%7) {name = "permute2", order0 = 0 : i32, order1 = 2 : i32, order2 = 1 : i32, order3 = 3 : i32, quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
    return %8 : tensor<1x16x16x16xf32>
  }
}
