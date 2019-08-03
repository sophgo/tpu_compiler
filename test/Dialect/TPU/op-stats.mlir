// RUN: mlir-opt -print-op-stats -verify-each=true %s -o=/dev/null 2>&1 | FileCheck %s

func @testConv2d(%arg0: tensor<1x3x28x28xf32>, %arg1: tensor<16x3x3x3xf32>, %arg2: tensor<16xf32>) -> tensor<1x16x28x28xf32> {
  %0 = "tpu.conv_2d"(%arg0, %arg1, %arg2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x3x28x28xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<1x16x28x28xf32>
  return %0 : tensor<1x16x28x28xf32>
}

// CHECK-LABEL: Operations encountered
// CHECK: func , 1
// CHECK: module_terminator , 1
// CHECK: std.return , 1
// CHECK: tpu.conv_2d , 1
