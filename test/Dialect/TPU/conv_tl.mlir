// RUN: mlir-opt -print-op-stats -verify-each=true %s -o=/dev/null 2>&1 | FileCheck %s
// RUN: mlir-opt -print-tpu-op-info -verify-each=true %s -o=/dev/null 2>&1 | FileCheck %s --check-prefix=OUTLINING --dump-input=fail

// CHECK-LABEL: module
// CHECK-NOT: error

// OUTLINING-LABEL: Modules:
// OUTLINING-NEXT: -----------------------
// OUTLINING-NEXT: test_tl_conv_2d

module {

  // filter and bias pass as tensor, function handle weight loading by itself
  // TODO:
  //   1. use subview to alloc lmem (may need to premute to fully express the rank)
  //   2. alloc weight working lmem
  func @test_tl_conv_2d_1(%A: memref<1x3x28x28xi8>, %f: tensor<16x3x3x3xi8>, %b: tensor<16xi8>, %B: memref<1x16x28x28xi8, 0>) {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %tag = alloc() : memref<1 x f32>
    %num_elements = constant 64 : index
    // alloc lmem memref
    %A_lmem = alloc() : memref<1x3x28x28xi8, 1>
    %B_lmem = alloc() : memref<1x16x28x28xi8, 1>
    // dma from A to A_lmem
    dma_start %A[%c0,%c0,%c0,%c0], %A_lmem[%c0,%c0,%c0,%c0], %num_elements, %tag[%c0]
      : memref<1x3x28x28xi8, 0>, memref<1x3x28x28xi8, 1>, memref<1xf32>
    dma_wait %tag[%c0], %num_elements : memref<1 x f32>
    // affine.dma_start %A[%c0,%c0,%c0,%c0], %A_lmem[%c0,%c0,%c0,%c0], %tag[%c0], %num_elements
    //   : memref<1x3x28x28xi8, 0>, memref<1x3x28x28xi8, 1>, memref<1xf32>
    // affine.dma_wait %tag[%c0], %num_elements : memref<1xf32>
    // load to tensor, do conv on tensors, then store to memref
    %A_tensor = tensor_load %A_lmem : memref<1x3x28x28xi8, 1>
    %B_tensor = "tpu.tl_la_conv_2d"(%A_tensor, %f, %b)
      {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "RELU", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32, with_bias = true}
      : (tensor<1x3x28x28xi8>, tensor<16x3x3x3xi8>, tensor<16xi8>) -> tensor<1x16x28x28xi8>
    tensor_store %B_tensor, %B_lmem : memref<1x16x28x28xi8, 1>
    // dma from B_lmem to B
    dma_start %B_lmem[%c0,%c0,%c0,%c0], %B[%c0,%c0,%c0,%c0], %num_elements, %tag[%c0]
      : memref<1x16x28x28xi8, 1>, memref<1x16x28x28xi8, 0>, memref<1xf32>
    dma_wait %tag[%c0], %num_elements : memref<1xf32>
    // dealloc
    dealloc %A_lmem : memref<1x3x28x28xi8, 1>
    dealloc %B_lmem : memref<1x16x28x28xi8, 1>
    return
  }

  func @main(%arg0: tensor<?x1x28x28xf32>) -> tensor<?x10xf32> {
    %w_0 = constant dense<1.000000e-01> : tensor<16x1x3x3xf32>
    %b_0 = constant dense<1.000000e-01> : tensor<16xf32>
    %w_1 = constant dense<1.000000e-01> : tensor<16x16x3x3xf32>
    %b_1 = constant dense<1.000000e-01> : tensor<16xf32>
    %w_2 = constant dense<1.000000e-01> : tensor<10x16x3x3xf32>
    %b_2 = constant dense<1.000000e-01> : tensor<10xf32>
    %0 = "tpu.conv_2d"(%arg0, %w_0, %b_0) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32, with_bias = true} : (tensor<?x1x28x28xf32>, tensor<16x1x3x3xf32>, tensor<16xf32>) -> tensor<?x16x14x14xf32>
    %1 = "tpu.relu"(%0) {negative_slope = 0.000000e+00 : f32} : (tensor<?x16x14x14xf32>) -> tensor<?x16x14x14xf32>
    %2 = "tpu.conv_2d"(%1, %w_1, %b_1) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32, with_bias = true} : (tensor<?x16x14x14xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>) -> tensor<?x16x7x7xf32>
    %3 = "tpu.relu"(%2) {negative_slope = 0.000000e+00 : f32} : (tensor<?x16x7x7xf32>) -> tensor<?x16x7x7xf32>
    %4 = "tpu.conv_2d"(%3, %w_2, %b_2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32, with_bias = true} : (tensor<?x16x7x7xf32>, tensor<10x16x3x3xf32>, tensor<10xf32>) -> tensor<?x10x4x4xf32>
    %5 = "tpu.relu"(%4) {negative_slope = 0.000000e+00 : f32} : (tensor<?x10x4x4xf32>) -> tensor<?x10x4x4xf32>
    %6 = "tpu.pool_2d"(%5) {pool = "AVE", filter_height = 4 : i32, filter_width = 4 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<?x10x4x4xf32>) -> tensor<?x10x1x1xf32>
    %7 = "tpu.reshape"(%6) : (tensor<?x10x1x1xf32>) -> tensor<?x10xf32>
    return %7 : tensor<?x10xf32>
  }
}
