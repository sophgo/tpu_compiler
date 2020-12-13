

module {
  func @tpu_func(%arg0: tensor<1x1x256x1xf32>, %arg1: tensor<1x4096x256x1xf32>) -> tensor<1x4096xf32> {
    %0 = "tpu.weight_file"() {filename = "resnet50_1_06eeeb7e.npz"} : () -> memref<10xf32>
    %1 = "tpu.input"(%arg0) {name = "data0", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<1x1x256x1xf32>) -> tensor<1x1x256x1xf32>
    %2 = "tpu.input"(%arg1) {name = "data1", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<1x4096x256x1xf32>) -> tensor<1x4096x256x1xf32>
    %3 = "tpu.load_weight"(%0) {name = "x_conv_w"} : (memref<10xf32>) -> tensor<1x1x1x1xf32>
    %4 = "tpu.none"() : () -> none
    %5 = "tpu.conv_2d"(%1, %3, %4, %4, %4, %4, %4) {name = "x1", param = {dilation_h = 1 : i32, dilation_w = 1 : i32, do_relu = false, group = 1 : i32, ins = [], is_dw = false, padding = "VALID", padding_b = 0 : i32, padding_l = 0 : i32, padding_r = 0 : i32, padding_t = 0 : i32, stride_h = 1 : i32, stride_w = 1 : i32, with_bias = false}, quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<1x1x256x1xf32>, tensor<1x1x1x1xf32>, none, none, none, none, none) -> tensor<1x1x256x1xf32>
    %6 = "tpu.load_weight"(%0) {name = "y_conv_w"} : (memref<10xf32>) -> tensor<1x4096x1x1xf32>
    %7 = "tpu.none"() : () -> none
    %8 = "tpu.conv_2d"(%2, %6, %7, %7, %7, %7, %7) {name = "y1", param = {dilation_h = 1 : i32, dilation_w = 1 : i32, do_relu = false, group = 4096 : i32, ins = [], is_dw = false, padding = "VALID", padding_b = 0 : i32, padding_l = 0 : i32, padding_r = 0 : i32, padding_t = 0 : i32, stride_h = 1 : i32, stride_w = 1 : i32, with_bias = false}, quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<1x4096x256x1xf32>, tensor<1x4096x1x1xf32>, none, none, none, none, none) -> tensor<1x4096x256x1xf32>
    %9 = "tpu.none"() : () -> none
    %10 = "tpu.broadcast_sub"(%8, %5, %9, %9, %9, %9) {name = "sub", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<1x4096x256x1xf32>, tensor<1x1x256x1xf32>, none, none, none, none) -> tensor<1x4096x256x1xf32>
    %11 = "tpu.square"(%10) {name = "mul", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<1x4096x256x1xf32>) -> tensor<1x4096x256x1xf32>
    %12 = "tpu.reshape"(%11) {name = "right_Reshape"} : (tensor<1x4096x256x1xf32>) -> tensor<4096x256xf32>
    %13 = "tpu.load_weight"(%0) {name = "fc1_w"} : (memref<10xf32>) -> tensor<1x256xf32>
    %14 = "tpu.none"() : () -> none
    %15 ="tpu.fully_connected"(%12, %13, %14, %14, %14, %14, %14) {name = "out", quant = {is_asymmetric = false, is_perchannel = false, mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32, zero_point = 0 : i32}} : (tensor<4096x256xf32>, tensor<1x256xf32>, none, none, none, none, none) -> tensor<4096x1xf32>
    %16 = "tpu.reshape"(%15) {name = "out_reshape"} : (tensor<4096x1xf32>) -> tensor<1x4096xf32>
    return %16 : tensor<1x4096xf32>
  }
}
