
module {
  func @tpu_func(%arg0: tensor<160x180xf32>, %arg1: tensor<80x80xf32>) -> (tensor<1xf32>) {
    %1 = "tpu.input"(%arg0) {name = "data", quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<160x180xf32>) -> tensor<160x180xf32>
    %2 = "tpu.input"(%arg1) {name = "template", quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<80x80xf32>) -> tensor<80x80xf32>
    %3 = "tpu.none"() : () -> none
    %4 = "tpu.none"() : () -> none
    %5 = "tpu.match_template"(%1,%2,%3,%4) {name = "match", mode = "TM_SQDIFF", quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}} : (tensor<160x180xf32>, tensor<80x80xf32>, none, none) -> tensor<81x101xf32>
    %6 = "tpu.reshape"(%5) {name = "6_Reshape"} : (tensor<81x101xf32>) -> tensor<8181xf32>
    %7 = "tpu.argmax"(%6) {name = "argmax", axis = 0 : i32, quant = {mode = "NONE", param_type = "NONE", threshold = 0.000000e+00 : f32}}: (tensor<8181xf32>) -> tensor<1xf32> 
    return %7 : tensor<1xf32>
  }
}
