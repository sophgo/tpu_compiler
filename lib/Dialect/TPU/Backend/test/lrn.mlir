module {
  func @tpu_func(%arg0: tensor<4x64x56x56xf32>) -> (tensor<4x64x56x56xf32>) {
    %0 = "tpu.weight_file"() {filename = "lrn_06eeeb7e.npz"} : () -> memref<10xf32>
    %1 = "tpu.input"(%arg0) {name = "data", quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<4x64x56x56xf32>) -> tensor<4x64x56x56xf32>
    %2 = "tpu.none"() : () -> none
    %3 = "tpu.load_weight"(%0) {name = "slope", storage = "NONE"} : (memref<10xf32>) -> tensor<1x64x1x1xf32>
    %4 = "tpu.prelu"(%1, %3, %2, %2, %2, %2, %2, %2, %2, %2) {name = "prelu", quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<4x64x56x56xf32>, tensor<1x64x1x1xf32>, none, none, none, none, none, none, none, none) -> tensor<4x64x56x56xf32>
    %5 = "tpu.relu"(%4) {name = "relu", quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<4x64x56x56xf32>) -> tensor<4x64x56x56xf32>
    %9 = "tpu.lrn"(%5, %2, %2, %2) {alpha = 9.99999974E-5 : f32, beta = 7.500000e-01 : f32, k = 1.000000e+00 : f32, local_size = 5 : i32, lrn_rshift = 0 : i32, name = "lrn_test", norm_region = 0 : i32, quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}, quant_data0 = 0 : i32, quant_data1 = 0 : i32, sum_rshift = 0 : i32} : (tensor<4x64x56x56xf32>, none, none, none) -> tensor<4x64x56x56xf32>
    return %9: tensor<4x64x56x56xf32>
  }
}