
module {
  func @tpu_func(%arg0: tensor<109x87xf32>, %arg1: tensor<80x56xf32>) -> (tensor<1xf32>) {
    %1 = "tpu.input"(%arg0) {name = "data", quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<109x87xf32>) -> tensor<109x87xf32>
    %2 = "tpu.input"(%arg1) {name = "template", quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<80x56xf32>) -> tensor<80x56xf32>
    %3 = "tpu.match_template"(%1,%2) {name = "match", mode = "TM_CCOEFF_NORMED", quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<109x87xf32>, tensor<80x56xf32>) -> tensor<960xf32>
    %5 = "tpu.argmax"(%3) {name = "argmax", axis = 0 : i32, quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}}: (tensor<960xf32>) -> tensor<1xf32>
    return %5 : tensor<1xf32>
  }
}
