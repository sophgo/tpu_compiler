// RUN: mlir-opt -print-op-stats -verify-each=true %s -o=/dev/null 2>&1 | FileCheck %s
// RUN: mlir-opt -print-tpu-op-stats -verify-each=true %s -o=/dev/null 2>&1 | FileCheck %s --check-prefix=OUTLINING --dump-input=fail

// CHECK-LABEL: module
// CHECK-NOT: error

// OUTLINING-LABEL: Modules:
// OUTLINING-NEXT: -----------------------
// OUTLINING-NEXT:  > std.constant
// OUTLINING-NEXT:  > std.constant
// OUTLINING-NEXT:  > tpu.conv_2d
// OUTLINING-NEXT:  > std.return
// OUTLINING-NEXT: func : test_conv_2d
// OUTLINING-NEXT: (tensor<1x3x28x28xf32>) -> tensor<1x16x28x28xf32>
// OUTLINING-NEXT:  > std.constant
// OUTLINING-NEXT:  > std.constant
// OUTLINING-NEXT:  > std.constant
// OUTLINING-NEXT:  > std.constant
// OUTLINING-NEXT:  > std.constant
// OUTLINING-NEXT:  > std.constant
// OUTLINING-NEXT:  > tpu.conv_2d
// OUTLINING-NEXT:  > tpu.relu
// OUTLINING-NEXT:  > tpu.conv_2d
// OUTLINING-NEXT:  > tpu.relu
// OUTLINING-NEXT:  > tpu.conv_2d
// OUTLINING-NEXT:  > tpu.relu
// OUTLINING-NEXT:  > tpu.average_pool_2d
// OUTLINING-NEXT:  > tpu.reshape
// OUTLINING-NEXT:  > std.return
// OUTLINING-NEXT: func : main
// OUTLINING-NEXT: (tensor<?x1x28x28xf32>) -> tensor<?x10xf32>
// OUTLINING-NEXT:  > module_terminator

// OUTLINING-LABEL: Funcs:
// OUTLINING-NEXT: -----------------------
// OUTLINING-NEXT: test_conv_2d
// OUTLINING-NEXT:  > std.constant
// OUTLINING-NEXT:  > std.constant
// OUTLINING-NEXT:  > tpu.conv_2d
// OUTLINING-NEXT:  > std.return
// OUTLINING-NEXT:  > func
// OUTLINING-NEXT: main
// OUTLINING-NEXT:  > std.constant
// OUTLINING-NEXT:  > std.constant
// OUTLINING-NEXT:  > std.constant
// OUTLINING-NEXT:  > std.constant
// OUTLINING-NEXT:  > std.constant
// OUTLINING-NEXT:  > std.constant
// OUTLINING-NEXT:  > tpu.conv_2d
// OUTLINING-NEXT:  > tpu.relu
// OUTLINING-NEXT:  > tpu.conv_2d
// OUTLINING-NEXT:  > tpu.relu
// OUTLINING-NEXT:  > tpu.conv_2d
// OUTLINING-NEXT:  > tpu.relu
// OUTLINING-NEXT:  > tpu.average_pool_2d
// OUTLINING-NEXT:  > tpu.reshape
// OUTLINING-NEXT:  > std.return
// OUTLINING-NEXT:  > func

// OUTLINING-LABEL: Module walk Conv2DOp:
// OUTLINING-NEXT: -----------------------
// OUTLINING-NEXT:  > tpu.conv_2d
// OUTLINING-NEXT:  >> MAC: 338688, OPs: 677376

// OUTLINING-LABEL: Funcs walk Conv2DOp:
// OUTLINING-NEXT: -----------------------
// OUTLINING-NEXT: test_conv_2d
// OUTLINING-NEXT:  > tpu.conv_2d
// OUTLINING-NEXT: func total MAC: 338688, total OPs: 677376
// OUTLINING-NEXT: main
// OUTLINING-NEXT:  > tpu.conv_2d
// OUTLINING-NEXT:  > tpu.conv_2d
// OUTLINING-NEXT:  > tpu.conv_2d
// OUTLINING-NEXT: func total MAC: 164160, total OPs: 328320

module {
  func @test_conv_2d(%arg0: tensor<1x3x28x28xf32>) -> tensor<1x16x28x28xf32> {
    %cst = constant dense<1.000000e-01> : tensor<16x3x3x3xf32>
    %cst_0 = constant dense<1.000000e-01> : tensor<16xf32>
    %0 = "tpu.conv_2d"(%arg0, %cst, %cst_0) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "RELU", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x3x28x28xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<1x16x28x28xf32>
    return %0 : tensor<1x16x28x28xf32>
  }

  func @main(%arg0: tensor<?x1x28x28xf32>) -> tensor<?x10xf32> {
    %w_0 = constant dense<1.000000e-01> : tensor<16x1x3x3xf32>
    %b_0 = constant dense<1.000000e-01> : tensor<16xf32>
    %w_1 = constant dense<1.000000e-01> : tensor<16x16x3x3xf32>
    %b_1 = constant dense<1.000000e-01> : tensor<16xf32>
    %w_2 = constant dense<1.000000e-01> : tensor<10x16x3x3xf32>
    %b_2 = constant dense<1.000000e-01> : tensor<10xf32>
    %0 = "tpu.conv_2d"(%arg0, %w_0, %b_0) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<?x1x28x28xf32>, tensor<16x1x3x3xf32>, tensor<16xf32>) -> tensor<?x16x14x14xf32>
    %1 = "tpu.relu"(%0) : (tensor<?x16x14x14xf32>) -> tensor<?x16x14x14xf32>
    %2 = "tpu.conv_2d"(%1, %w_1, %b_1) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<?x16x14x14xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>) -> tensor<?x16x7x7xf32>
    %3 = "tpu.relu"(%2) : (tensor<?x16x7x7xf32>) -> tensor<?x16x7x7xf32>
    %4 = "tpu.conv_2d"(%3, %w_2, %b_2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<?x16x7x7xf32>, tensor<10x16x3x3xf32>, tensor<10xf32>) -> tensor<?x10x4x4xf32>
    %5 = "tpu.relu"(%4) : (tensor<?x10x4x4xf32>) -> tensor<?x10x4x4xf32>
    %6 = "tpu.average_pool_2d"(%5) {filter_height = 4 : i32, filter_width = 4 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<?x10x4x4xf32>) -> tensor<?x10x1x1xf32>
    %7 = "tpu.reshape"(%6) : (tensor<?x10x1x1xf32>) -> tensor<?x10xf32>
    return %7 : tensor<?x10xf32>
  }
}
