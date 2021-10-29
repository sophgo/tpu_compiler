

module {
  func @tpu_func(%arg0: tensor<1x256x1x1xf32>) -> tensor<1x256x31x31xf32> {
    %0 = "tpu.weight_file"() {filename = "tile_1_06eeeb7e.npz"} : () -> memref<10xf32>
    %1 = "tpu.input"(%arg0) {name = "input", quant = {mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32}} : (tensor<1x256x1x1xf32>) -> tensor<1x256x1x1xf32>
    %2 = "tpu.none"() : () -> none
    %3 = "tpu.tile"(%1) {name = "tile_h", quant = {mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32}, axis = 2: i32, tiles = 31 : i32} : (tensor<1x256x1x1xf32>) -> tensor<1x256x31x1xf32>
    %4 = "tpu.tile"(%3) {name = "tile_w", quant = {mode = "NONE", param_type = "NONE", threshold_max = 0.000000e+00 : f32, threshold_min = 0.000000e+00 : f32}, axis = 3: i32, tiles = 31 : i32} : (tensor<1x256x31x1xf32>) -> tensor<1x256x31x31xf32>
   return %4: tensor<1x256x31x31xf32>
  }
}
