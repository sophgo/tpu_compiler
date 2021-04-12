module  {
  func @tpu_func(%arg0: tensor<1x16x40x64xf32>) -> tensor<1x16x40x40xf32> {
    %0 = "tpu.weight_file"() {filename = "MatMul_1_06eeeb7e.npz"} : () -> memref<10xf32>
    %1 = "tpu.input"(%arg0) {name = "input", preprocess = {aligned = false, channel_order = "bgr", keep_aspect_ratio = false, mean = [0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32], pixel_format = "BGR_PLANAR", resize_dims = [40 : i32, 64 : i32], scale = [1.000000e+00 : f32, 1.000000e+00 : f32, 1.000000e+00 : f32]}, quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<1x16x40x64xf32>) -> tensor<1x16x40x64xf32>
    %2 = "tpu.permute"(%1) {name = "X1_Transpose", order0 = 0 : i32, order1 = 1 : i32, order2 = 3 : i32, order3 = 2 : i32, quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<1x16x40x64xf32>) -> tensor<1x16x64x40xf32>
    %3 = "tpu.none"() : () -> none
    %4 = "tpu.matmul"(%1, %2, %3, %3, %3, %3) {name = "output_MatMul", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<1x16x40x64xf32>, tensor<1x16x64x40xf32>, none, none, none, none) -> tensor<1x16x40x40xf32>
    return %4 : tensor<1x16x40x40xf32>
  }
}
