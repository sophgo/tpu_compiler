// RUN: mlir-opt -print-op-stats -verify-each=true %s -o=/dev/null 2>&1 | FileCheck %s
// RUN: mlir-opt -tpu-ops-outlining -verify-each=true %s -o=/dev/null 2>&1 | FileCheck %s --check-prefix=OUTLINING --dump-input=fail

// CHECK-LABEL: module
// CHECK-NOT: error

// OUTLINING-LABEL: Modules:
// OUTLINING-NEXT: -----------------------
// OUTLINING-NEXT: func
// OUTLINING-NEXT:  > std.constant
// OUTLINING-NEXT:  > std.constant
// OUTLINING-NEXT:  > tpu.conv_2d
// OUTLINING-NEXT:  > std.return
// OUTLINING-NEXT:  > func
// OUTLINING-NEXT: module_terminator
// OUTLINING-NEXT:  > module_terminator

// OUTLINING-LABEL: Funcs:
// OUTLINING-NEXT: -----------------------
// OUTLINING-NEXT: main
// OUTLINING-NEXT:  > std.constant
// OUTLINING-NEXT:  > std.constant
// OUTLINING-NEXT:  > tpu.conv_2d
// OUTLINING-NEXT:  > std.return
// OUTLINING-NEXT:  > func

// OUTLINING-LABEL: Funcs walk Conv2DOp:
// OUTLINING-NEXT: -----------------------
// OUTLINING-NEXT: main
// OUTLINING-NEXT:  > tpu.conv_2d

module {
  func @main(%arg0: tensor<1x3x28x28xf32>) -> tensor<1x16x28x28xf32> {
    %cst = constant dense<1.000000e-01> : tensor<16x3x3x3xf32>
    %cst_0 = constant dense<1.000000e-01> : tensor<16xf32>
    %0 = "tpu.conv_2d"(%arg0, %cst, %cst_0) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "RELU", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x3x28x28xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<1x16x28x28xf32>
    return %0 : tensor<1x16x28x28xf32>
  }
}
