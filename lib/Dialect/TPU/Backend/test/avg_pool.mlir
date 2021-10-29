

module {
  func @tpu_func(%arg0: tensor<1x1024x17x17xf32>) -> tensor<1x1024x17x17xf32> {
    %0 = "tpu.weight_file"() {filename = "inceptionv4_2_5ab52b04ea65.npz"} : () -> memref<10xf32>
    %1 = "tpu.input"(%arg0) {name = "data", quant = {mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32}} : (tensor<1x1024x17x17xf32>) -> tensor<1x1024x17x17xf32>
    %2 = "tpu.none"() : () -> none
    %3 = "tpu.pool_avg_2d"(%1, %2, %2, %2, %2) {name = "inception_b4_pool_ave", param = {count_include_pad = true, do_relu = false, kernel_h = 3 : i32, kernel_w = 3 : i32, pad_value = 0 : i32, padding_b = 1 : i32, padding_l = 1 : i32, padding_r = 1 : i32, padding_t = 1 : i32, stride_h = 1 : i32, stride_w = 1 : i32}, quant = {mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32}} : (tensor<1x1024x17x17xf32>, none, none, none, none) -> tensor<1x1024x17x17xf32>
    return %3 : tensor<1x1024x17x17xf32>
  }
}
