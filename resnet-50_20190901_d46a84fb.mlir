

module {
  func @tpu_func(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32> {
    %0 = "tpu.load_file"() {filename = "ResNet-50-model.weight"} : () -> memref<2147483648xf32>
    %1 = "tpu.load_weight"(%0) {offset = 0 : i64} : (memref<2147483648xf32>) -> tensor<64x3x7x7xf32>
    %2 = "tpu.load_weight"(%0) {offset = 37632 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %3 = "tpu.conv_2d"(%arg0, %1, %2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
    %4 = "tpu.load_weight"(%0) {offset = 37888 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %5 = "tpu.load_weight"(%0) {offset = 38144 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %6 = "tpu.batch_norm"(%3, %4, %5) : (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
    %7 = "tpu.load_weight"(%0) {offset = 38400 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %8 = "tpu.load_weight"(%0) {offset = 38656 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %9 = "tpu.scale"(%6, %7, %8) : (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
    %10 = "tpu.relu"(%9) {negative_slope = 0.000000e+00 : f32} : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %11 = "tpu.max_pool_2d"(%10) {filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x64x112x112xf32>) -> tensor<1x64x55x55xf32>
    %12 = "tpu.load_weight"(%0) {offset = 38912 : i64} : (memref<2147483648xf32>) -> tensor<256x64x1x1xf32>
    %cst = constant dense<0.000000e+00> : tensor<256xf32>
    %13 = "tpu.conv_2d"(%11, %12, %cst) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x64x55x55xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>) -> tensor<1x256x55x55xf32>
    %14 = "tpu.load_weight"(%0) {offset = 104448 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %15 = "tpu.load_weight"(%0) {offset = 105472 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %16 = "tpu.batch_norm"(%13, %14, %15) : (tensor<1x256x55x55xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x55x55xf32>
    %17 = "tpu.load_weight"(%0) {offset = 106496 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %18 = "tpu.load_weight"(%0) {offset = 107520 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %19 = "tpu.scale"(%16, %17, %18) : (tensor<1x256x55x55xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x55x55xf32>
    %20 = "tpu.load_weight"(%0) {offset = 108544 : i64} : (memref<2147483648xf32>) -> tensor<64x64x1x1xf32>
    %cst_0 = constant dense<0.000000e+00> : tensor<64xf32>
    %21 = "tpu.conv_2d"(%11, %20, %cst_0) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x64x55x55xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>) -> tensor<1x64x55x55xf32>
    %22 = "tpu.load_weight"(%0) {offset = 124928 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %23 = "tpu.load_weight"(%0) {offset = 125184 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %24 = "tpu.batch_norm"(%21, %22, %23) : (tensor<1x64x55x55xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x55x55xf32>
    %25 = "tpu.load_weight"(%0) {offset = 125440 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %26 = "tpu.load_weight"(%0) {offset = 125696 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %27 = "tpu.scale"(%24, %25, %26) : (tensor<1x64x55x55xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x55x55xf32>
    %28 = "tpu.relu"(%27) {negative_slope = 0.000000e+00 : f32} : (tensor<1x64x55x55xf32>) -> tensor<1x64x55x55xf32>
    %29 = "tpu.load_weight"(%0) {offset = 125952 : i64} : (memref<2147483648xf32>) -> tensor<64x64x3x3xf32>
    %cst_1 = constant dense<0.000000e+00> : tensor<64xf32>
    %30 = "tpu.conv_2d"(%28, %29, %cst_1) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x64x55x55xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>) -> tensor<1x64x55x55xf32>
    %31 = "tpu.load_weight"(%0) {offset = 273408 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %32 = "tpu.load_weight"(%0) {offset = 273664 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %33 = "tpu.batch_norm"(%30, %31, %32) : (tensor<1x64x55x55xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x55x55xf32>
    %34 = "tpu.load_weight"(%0) {offset = 273920 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %35 = "tpu.load_weight"(%0) {offset = 274176 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %36 = "tpu.scale"(%33, %34, %35) : (tensor<1x64x55x55xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x55x55xf32>
    %37 = "tpu.relu"(%36) {negative_slope = 0.000000e+00 : f32} : (tensor<1x64x55x55xf32>) -> tensor<1x64x55x55xf32>
    %38 = "tpu.load_weight"(%0) {offset = 274432 : i64} : (memref<2147483648xf32>) -> tensor<256x64x1x1xf32>
    %cst_2 = constant dense<0.000000e+00> : tensor<256xf32>
    %39 = "tpu.conv_2d"(%37, %38, %cst_2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x64x55x55xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>) -> tensor<1x256x55x55xf32>
    %40 = "tpu.load_weight"(%0) {offset = 339968 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %41 = "tpu.load_weight"(%0) {offset = 340992 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %42 = "tpu.batch_norm"(%39, %40, %41) : (tensor<1x256x55x55xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x55x55xf32>
    %43 = "tpu.load_weight"(%0) {offset = 342016 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %44 = "tpu.load_weight"(%0) {offset = 343040 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %45 = "tpu.scale"(%42, %43, %44) : (tensor<1x256x55x55xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x55x55xf32>
    %46 = "tpu.eltwise"(%19, %19) : (tensor<1x256x55x55xf32>, tensor<1x256x55x55xf32>) -> tensor<1x256x55x55xf32>
    %47 = "tpu.relu"(%46) {negative_slope = 0.000000e+00 : f32} : (tensor<1x256x55x55xf32>) -> tensor<1x256x55x55xf32>
    %48 = "tpu.load_weight"(%0) {offset = 344064 : i64} : (memref<2147483648xf32>) -> tensor<64x256x1x1xf32>
    %cst_3 = constant dense<0.000000e+00> : tensor<64xf32>
    %49 = "tpu.conv_2d"(%47, %48, %cst_3) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x55x55xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>) -> tensor<1x64x55x55xf32>
    %50 = "tpu.load_weight"(%0) {offset = 409600 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %51 = "tpu.load_weight"(%0) {offset = 409856 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %52 = "tpu.batch_norm"(%49, %50, %51) : (tensor<1x64x55x55xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x55x55xf32>
    %53 = "tpu.load_weight"(%0) {offset = 410112 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %54 = "tpu.load_weight"(%0) {offset = 410368 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %55 = "tpu.scale"(%52, %53, %54) : (tensor<1x64x55x55xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x55x55xf32>
    %56 = "tpu.relu"(%55) {negative_slope = 0.000000e+00 : f32} : (tensor<1x64x55x55xf32>) -> tensor<1x64x55x55xf32>
    %57 = "tpu.load_weight"(%0) {offset = 410624 : i64} : (memref<2147483648xf32>) -> tensor<64x64x3x3xf32>
    %cst_4 = constant dense<0.000000e+00> : tensor<64xf32>
    %58 = "tpu.conv_2d"(%56, %57, %cst_4) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x64x55x55xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>) -> tensor<1x64x55x55xf32>
    %59 = "tpu.load_weight"(%0) {offset = 558080 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %60 = "tpu.load_weight"(%0) {offset = 558336 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %61 = "tpu.batch_norm"(%58, %59, %60) : (tensor<1x64x55x55xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x55x55xf32>
    %62 = "tpu.load_weight"(%0) {offset = 558592 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %63 = "tpu.load_weight"(%0) {offset = 558848 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %64 = "tpu.scale"(%61, %62, %63) : (tensor<1x64x55x55xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x55x55xf32>
    %65 = "tpu.relu"(%64) {negative_slope = 0.000000e+00 : f32} : (tensor<1x64x55x55xf32>) -> tensor<1x64x55x55xf32>
    %66 = "tpu.load_weight"(%0) {offset = 559104 : i64} : (memref<2147483648xf32>) -> tensor<256x64x1x1xf32>
    %cst_5 = constant dense<0.000000e+00> : tensor<256xf32>
    %67 = "tpu.conv_2d"(%65, %66, %cst_5) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x64x55x55xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>) -> tensor<1x256x55x55xf32>
    %68 = "tpu.load_weight"(%0) {offset = 624640 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %69 = "tpu.load_weight"(%0) {offset = 625664 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %70 = "tpu.batch_norm"(%67, %68, %69) : (tensor<1x256x55x55xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x55x55xf32>
    %71 = "tpu.load_weight"(%0) {offset = 626688 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %72 = "tpu.load_weight"(%0) {offset = 627712 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %73 = "tpu.scale"(%70, %71, %72) : (tensor<1x256x55x55xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x55x55xf32>
    %74 = "tpu.eltwise"(%47, %47) : (tensor<1x256x55x55xf32>, tensor<1x256x55x55xf32>) -> tensor<1x256x55x55xf32>
    %75 = "tpu.relu"(%74) {negative_slope = 0.000000e+00 : f32} : (tensor<1x256x55x55xf32>) -> tensor<1x256x55x55xf32>
    %76 = "tpu.load_weight"(%0) {offset = 628736 : i64} : (memref<2147483648xf32>) -> tensor<64x256x1x1xf32>
    %cst_6 = constant dense<0.000000e+00> : tensor<64xf32>
    %77 = "tpu.conv_2d"(%75, %76, %cst_6) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x55x55xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>) -> tensor<1x64x55x55xf32>
    %78 = "tpu.load_weight"(%0) {offset = 694272 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %79 = "tpu.load_weight"(%0) {offset = 694528 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %80 = "tpu.batch_norm"(%77, %78, %79) : (tensor<1x64x55x55xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x55x55xf32>
    %81 = "tpu.load_weight"(%0) {offset = 694784 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %82 = "tpu.load_weight"(%0) {offset = 695040 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %83 = "tpu.scale"(%80, %81, %82) : (tensor<1x64x55x55xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x55x55xf32>
    %84 = "tpu.relu"(%83) {negative_slope = 0.000000e+00 : f32} : (tensor<1x64x55x55xf32>) -> tensor<1x64x55x55xf32>
    %85 = "tpu.load_weight"(%0) {offset = 695296 : i64} : (memref<2147483648xf32>) -> tensor<64x64x3x3xf32>
    %cst_7 = constant dense<0.000000e+00> : tensor<64xf32>
    %86 = "tpu.conv_2d"(%84, %85, %cst_7) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x64x55x55xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>) -> tensor<1x64x55x55xf32>
    %87 = "tpu.load_weight"(%0) {offset = 842752 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %88 = "tpu.load_weight"(%0) {offset = 843008 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %89 = "tpu.batch_norm"(%86, %87, %88) : (tensor<1x64x55x55xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x55x55xf32>
    %90 = "tpu.load_weight"(%0) {offset = 843264 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %91 = "tpu.load_weight"(%0) {offset = 843520 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %92 = "tpu.scale"(%89, %90, %91) : (tensor<1x64x55x55xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x55x55xf32>
    %93 = "tpu.relu"(%92) {negative_slope = 0.000000e+00 : f32} : (tensor<1x64x55x55xf32>) -> tensor<1x64x55x55xf32>
    %94 = "tpu.load_weight"(%0) {offset = 843776 : i64} : (memref<2147483648xf32>) -> tensor<256x64x1x1xf32>
    %cst_8 = constant dense<0.000000e+00> : tensor<256xf32>
    %95 = "tpu.conv_2d"(%93, %94, %cst_8) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x64x55x55xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>) -> tensor<1x256x55x55xf32>
    %96 = "tpu.load_weight"(%0) {offset = 909312 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %97 = "tpu.load_weight"(%0) {offset = 910336 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %98 = "tpu.batch_norm"(%95, %96, %97) : (tensor<1x256x55x55xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x55x55xf32>
    %99 = "tpu.load_weight"(%0) {offset = 911360 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %100 = "tpu.load_weight"(%0) {offset = 912384 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %101 = "tpu.scale"(%98, %99, %100) : (tensor<1x256x55x55xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x55x55xf32>
    %102 = "tpu.eltwise"(%75, %75) : (tensor<1x256x55x55xf32>, tensor<1x256x55x55xf32>) -> tensor<1x256x55x55xf32>
    %103 = "tpu.relu"(%102) {negative_slope = 0.000000e+00 : f32} : (tensor<1x256x55x55xf32>) -> tensor<1x256x55x55xf32>
    %104 = "tpu.load_weight"(%0) {offset = 913408 : i64} : (memref<2147483648xf32>) -> tensor<512x256x1x1xf32>
    %cst_9 = constant dense<0.000000e+00> : tensor<512xf32>
    %105 = "tpu.conv_2d"(%103, %104, %cst_9) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x256x55x55xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %106 = "tpu.load_weight"(%0) {offset = 1437696 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %107 = "tpu.load_weight"(%0) {offset = 1439744 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %108 = "tpu.batch_norm"(%105, %106, %107) : (tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %109 = "tpu.load_weight"(%0) {offset = 1441792 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %110 = "tpu.load_weight"(%0) {offset = 1443840 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %111 = "tpu.scale"(%108, %109, %110) : (tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %112 = "tpu.load_weight"(%0) {offset = 1445888 : i64} : (memref<2147483648xf32>) -> tensor<128x256x1x1xf32>
    %cst_10 = constant dense<0.000000e+00> : tensor<128xf32>
    %113 = "tpu.conv_2d"(%103, %112, %cst_10) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x256x55x55xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %114 = "tpu.load_weight"(%0) {offset = 1576960 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %115 = "tpu.load_weight"(%0) {offset = 1577472 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %116 = "tpu.batch_norm"(%113, %114, %115) : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %117 = "tpu.load_weight"(%0) {offset = 1577984 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %118 = "tpu.load_weight"(%0) {offset = 1578496 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %119 = "tpu.scale"(%116, %117, %118) : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %120 = "tpu.relu"(%119) {negative_slope = 0.000000e+00 : f32} : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %121 = "tpu.load_weight"(%0) {offset = 1579008 : i64} : (memref<2147483648xf32>) -> tensor<128x128x3x3xf32>
    %cst_11 = constant dense<0.000000e+00> : tensor<128xf32>
    %122 = "tpu.conv_2d"(%120, %121, %cst_11) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %123 = "tpu.load_weight"(%0) {offset = 2168832 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %124 = "tpu.load_weight"(%0) {offset = 2169344 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %125 = "tpu.batch_norm"(%122, %123, %124) : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %126 = "tpu.load_weight"(%0) {offset = 2169856 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %127 = "tpu.load_weight"(%0) {offset = 2170368 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %128 = "tpu.scale"(%125, %126, %127) : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %129 = "tpu.relu"(%128) {negative_slope = 0.000000e+00 : f32} : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %130 = "tpu.load_weight"(%0) {offset = 2170880 : i64} : (memref<2147483648xf32>) -> tensor<512x128x1x1xf32>
    %cst_12 = constant dense<0.000000e+00> : tensor<512xf32>
    %131 = "tpu.conv_2d"(%129, %130, %cst_12) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %132 = "tpu.load_weight"(%0) {offset = 2433024 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %133 = "tpu.load_weight"(%0) {offset = 2435072 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %134 = "tpu.batch_norm"(%131, %132, %133) : (tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %135 = "tpu.load_weight"(%0) {offset = 2437120 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %136 = "tpu.load_weight"(%0) {offset = 2439168 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %137 = "tpu.scale"(%134, %135, %136) : (tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %138 = "tpu.eltwise"(%111, %111) : (tensor<1x512x28x28xf32>, tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %139 = "tpu.relu"(%138) {negative_slope = 0.000000e+00 : f32} : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %140 = "tpu.load_weight"(%0) {offset = 2441216 : i64} : (memref<2147483648xf32>) -> tensor<128x512x1x1xf32>
    %cst_13 = constant dense<0.000000e+00> : tensor<128xf32>
    %141 = "tpu.conv_2d"(%139, %140, %cst_13) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x512x28x28xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %142 = "tpu.load_weight"(%0) {offset = 2703360 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %143 = "tpu.load_weight"(%0) {offset = 2703872 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %144 = "tpu.batch_norm"(%141, %142, %143) : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %145 = "tpu.load_weight"(%0) {offset = 2704384 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %146 = "tpu.load_weight"(%0) {offset = 2704896 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %147 = "tpu.scale"(%144, %145, %146) : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %148 = "tpu.relu"(%147) {negative_slope = 0.000000e+00 : f32} : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %149 = "tpu.load_weight"(%0) {offset = 2705408 : i64} : (memref<2147483648xf32>) -> tensor<128x128x3x3xf32>
    %cst_14 = constant dense<0.000000e+00> : tensor<128xf32>
    %150 = "tpu.conv_2d"(%148, %149, %cst_14) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %151 = "tpu.load_weight"(%0) {offset = 3295232 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %152 = "tpu.load_weight"(%0) {offset = 3295744 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %153 = "tpu.batch_norm"(%150, %151, %152) : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %154 = "tpu.load_weight"(%0) {offset = 3296256 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %155 = "tpu.load_weight"(%0) {offset = 3296768 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %156 = "tpu.scale"(%153, %154, %155) : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %157 = "tpu.relu"(%156) {negative_slope = 0.000000e+00 : f32} : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %158 = "tpu.load_weight"(%0) {offset = 3297280 : i64} : (memref<2147483648xf32>) -> tensor<512x128x1x1xf32>
    %cst_15 = constant dense<0.000000e+00> : tensor<512xf32>
    %159 = "tpu.conv_2d"(%157, %158, %cst_15) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %160 = "tpu.load_weight"(%0) {offset = 3559424 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %161 = "tpu.load_weight"(%0) {offset = 3561472 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %162 = "tpu.batch_norm"(%159, %160, %161) : (tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %163 = "tpu.load_weight"(%0) {offset = 3563520 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %164 = "tpu.load_weight"(%0) {offset = 3565568 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %165 = "tpu.scale"(%162, %163, %164) : (tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %166 = "tpu.eltwise"(%139, %139) : (tensor<1x512x28x28xf32>, tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %167 = "tpu.relu"(%166) {negative_slope = 0.000000e+00 : f32} : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %168 = "tpu.load_weight"(%0) {offset = 3567616 : i64} : (memref<2147483648xf32>) -> tensor<128x512x1x1xf32>
    %cst_16 = constant dense<0.000000e+00> : tensor<128xf32>
    %169 = "tpu.conv_2d"(%167, %168, %cst_16) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x512x28x28xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %170 = "tpu.load_weight"(%0) {offset = 3829760 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %171 = "tpu.load_weight"(%0) {offset = 3830272 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %172 = "tpu.batch_norm"(%169, %170, %171) : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %173 = "tpu.load_weight"(%0) {offset = 3830784 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %174 = "tpu.load_weight"(%0) {offset = 3831296 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %175 = "tpu.scale"(%172, %173, %174) : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %176 = "tpu.relu"(%175) {negative_slope = 0.000000e+00 : f32} : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %177 = "tpu.load_weight"(%0) {offset = 3831808 : i64} : (memref<2147483648xf32>) -> tensor<128x128x3x3xf32>
    %cst_17 = constant dense<0.000000e+00> : tensor<128xf32>
    %178 = "tpu.conv_2d"(%176, %177, %cst_17) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %179 = "tpu.load_weight"(%0) {offset = 4421632 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %180 = "tpu.load_weight"(%0) {offset = 4422144 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %181 = "tpu.batch_norm"(%178, %179, %180) : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %182 = "tpu.load_weight"(%0) {offset = 4422656 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %183 = "tpu.load_weight"(%0) {offset = 4423168 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %184 = "tpu.scale"(%181, %182, %183) : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %185 = "tpu.relu"(%184) {negative_slope = 0.000000e+00 : f32} : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %186 = "tpu.load_weight"(%0) {offset = 4423680 : i64} : (memref<2147483648xf32>) -> tensor<512x128x1x1xf32>
    %cst_18 = constant dense<0.000000e+00> : tensor<512xf32>
    %187 = "tpu.conv_2d"(%185, %186, %cst_18) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %188 = "tpu.load_weight"(%0) {offset = 4685824 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %189 = "tpu.load_weight"(%0) {offset = 4687872 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %190 = "tpu.batch_norm"(%187, %188, %189) : (tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %191 = "tpu.load_weight"(%0) {offset = 4689920 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %192 = "tpu.load_weight"(%0) {offset = 4691968 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %193 = "tpu.scale"(%190, %191, %192) : (tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %194 = "tpu.eltwise"(%167, %167) : (tensor<1x512x28x28xf32>, tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %195 = "tpu.relu"(%194) {negative_slope = 0.000000e+00 : f32} : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %196 = "tpu.load_weight"(%0) {offset = 4694016 : i64} : (memref<2147483648xf32>) -> tensor<128x512x1x1xf32>
    %cst_19 = constant dense<0.000000e+00> : tensor<128xf32>
    %197 = "tpu.conv_2d"(%195, %196, %cst_19) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x512x28x28xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %198 = "tpu.load_weight"(%0) {offset = 4956160 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %199 = "tpu.load_weight"(%0) {offset = 4956672 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %200 = "tpu.batch_norm"(%197, %198, %199) : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %201 = "tpu.load_weight"(%0) {offset = 4957184 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %202 = "tpu.load_weight"(%0) {offset = 4957696 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %203 = "tpu.scale"(%200, %201, %202) : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %204 = "tpu.relu"(%203) {negative_slope = 0.000000e+00 : f32} : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %205 = "tpu.load_weight"(%0) {offset = 4958208 : i64} : (memref<2147483648xf32>) -> tensor<128x128x3x3xf32>
    %cst_20 = constant dense<0.000000e+00> : tensor<128xf32>
    %206 = "tpu.conv_2d"(%204, %205, %cst_20) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %207 = "tpu.load_weight"(%0) {offset = 5548032 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %208 = "tpu.load_weight"(%0) {offset = 5548544 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %209 = "tpu.batch_norm"(%206, %207, %208) : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %210 = "tpu.load_weight"(%0) {offset = 5549056 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %211 = "tpu.load_weight"(%0) {offset = 5549568 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %212 = "tpu.scale"(%209, %210, %211) : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %213 = "tpu.relu"(%212) {negative_slope = 0.000000e+00 : f32} : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %214 = "tpu.load_weight"(%0) {offset = 5550080 : i64} : (memref<2147483648xf32>) -> tensor<512x128x1x1xf32>
    %cst_21 = constant dense<0.000000e+00> : tensor<512xf32>
    %215 = "tpu.conv_2d"(%213, %214, %cst_21) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %216 = "tpu.load_weight"(%0) {offset = 5812224 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %217 = "tpu.load_weight"(%0) {offset = 5814272 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %218 = "tpu.batch_norm"(%215, %216, %217) : (tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %219 = "tpu.load_weight"(%0) {offset = 5816320 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %220 = "tpu.load_weight"(%0) {offset = 5818368 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %221 = "tpu.scale"(%218, %219, %220) : (tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %222 = "tpu.eltwise"(%195, %195) : (tensor<1x512x28x28xf32>, tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %223 = "tpu.relu"(%222) {negative_slope = 0.000000e+00 : f32} : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %224 = "tpu.load_weight"(%0) {offset = 5820416 : i64} : (memref<2147483648xf32>) -> tensor<1024x512x1x1xf32>
    %cst_22 = constant dense<0.000000e+00> : tensor<1024xf32>
    %225 = "tpu.conv_2d"(%223, %224, %cst_22) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x512x28x28xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %226 = "tpu.load_weight"(%0) {offset = 7917568 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %227 = "tpu.load_weight"(%0) {offset = 7921664 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %228 = "tpu.batch_norm"(%225, %226, %227) : (tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %229 = "tpu.load_weight"(%0) {offset = 7925760 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %230 = "tpu.load_weight"(%0) {offset = 7929856 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %231 = "tpu.scale"(%228, %229, %230) : (tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %232 = "tpu.load_weight"(%0) {offset = 7933952 : i64} : (memref<2147483648xf32>) -> tensor<256x512x1x1xf32>
    %cst_23 = constant dense<0.000000e+00> : tensor<256xf32>
    %233 = "tpu.conv_2d"(%223, %232, %cst_23) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x512x28x28xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %234 = "tpu.load_weight"(%0) {offset = 8458240 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %235 = "tpu.load_weight"(%0) {offset = 8459264 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %236 = "tpu.batch_norm"(%233, %234, %235) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %237 = "tpu.load_weight"(%0) {offset = 8460288 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %238 = "tpu.load_weight"(%0) {offset = 8461312 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %239 = "tpu.scale"(%236, %237, %238) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %240 = "tpu.relu"(%239) {negative_slope = 0.000000e+00 : f32} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %241 = "tpu.load_weight"(%0) {offset = 8462336 : i64} : (memref<2147483648xf32>) -> tensor<256x256x3x3xf32>
    %cst_24 = constant dense<0.000000e+00> : tensor<256xf32>
    %242 = "tpu.conv_2d"(%240, %241, %cst_24) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %243 = "tpu.load_weight"(%0) {offset = 10821632 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %244 = "tpu.load_weight"(%0) {offset = 10822656 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %245 = "tpu.batch_norm"(%242, %243, %244) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %246 = "tpu.load_weight"(%0) {offset = 10823680 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %247 = "tpu.load_weight"(%0) {offset = 10824704 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %248 = "tpu.scale"(%245, %246, %247) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %249 = "tpu.relu"(%248) {negative_slope = 0.000000e+00 : f32} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %250 = "tpu.load_weight"(%0) {offset = 10825728 : i64} : (memref<2147483648xf32>) -> tensor<1024x256x1x1xf32>
    %cst_25 = constant dense<0.000000e+00> : tensor<1024xf32>
    %251 = "tpu.conv_2d"(%249, %250, %cst_25) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %252 = "tpu.load_weight"(%0) {offset = 11874304 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %253 = "tpu.load_weight"(%0) {offset = 11878400 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %254 = "tpu.batch_norm"(%251, %252, %253) : (tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %255 = "tpu.load_weight"(%0) {offset = 11882496 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %256 = "tpu.load_weight"(%0) {offset = 11886592 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %257 = "tpu.scale"(%254, %255, %256) : (tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %258 = "tpu.eltwise"(%231, %231) : (tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %259 = "tpu.relu"(%258) {negative_slope = 0.000000e+00 : f32} : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %260 = "tpu.load_weight"(%0) {offset = 11890688 : i64} : (memref<2147483648xf32>) -> tensor<256x1024x1x1xf32>
    %cst_26 = constant dense<0.000000e+00> : tensor<256xf32>
    %261 = "tpu.conv_2d"(%259, %260, %cst_26) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %262 = "tpu.load_weight"(%0) {offset = 12939264 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %263 = "tpu.load_weight"(%0) {offset = 12940288 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %264 = "tpu.batch_norm"(%261, %262, %263) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %265 = "tpu.load_weight"(%0) {offset = 12941312 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %266 = "tpu.load_weight"(%0) {offset = 12942336 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %267 = "tpu.scale"(%264, %265, %266) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %268 = "tpu.relu"(%267) {negative_slope = 0.000000e+00 : f32} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %269 = "tpu.load_weight"(%0) {offset = 12943360 : i64} : (memref<2147483648xf32>) -> tensor<256x256x3x3xf32>
    %cst_27 = constant dense<0.000000e+00> : tensor<256xf32>
    %270 = "tpu.conv_2d"(%268, %269, %cst_27) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %271 = "tpu.load_weight"(%0) {offset = 15302656 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %272 = "tpu.load_weight"(%0) {offset = 15303680 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %273 = "tpu.batch_norm"(%270, %271, %272) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %274 = "tpu.load_weight"(%0) {offset = 15304704 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %275 = "tpu.load_weight"(%0) {offset = 15305728 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %276 = "tpu.scale"(%273, %274, %275) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %277 = "tpu.relu"(%276) {negative_slope = 0.000000e+00 : f32} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %278 = "tpu.load_weight"(%0) {offset = 15306752 : i64} : (memref<2147483648xf32>) -> tensor<1024x256x1x1xf32>
    %cst_28 = constant dense<0.000000e+00> : tensor<1024xf32>
    %279 = "tpu.conv_2d"(%277, %278, %cst_28) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %280 = "tpu.load_weight"(%0) {offset = 16355328 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %281 = "tpu.load_weight"(%0) {offset = 16359424 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %282 = "tpu.batch_norm"(%279, %280, %281) : (tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %283 = "tpu.load_weight"(%0) {offset = 16363520 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %284 = "tpu.load_weight"(%0) {offset = 16367616 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %285 = "tpu.scale"(%282, %283, %284) : (tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %286 = "tpu.eltwise"(%259, %259) : (tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %287 = "tpu.relu"(%286) {negative_slope = 0.000000e+00 : f32} : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %288 = "tpu.load_weight"(%0) {offset = 16371712 : i64} : (memref<2147483648xf32>) -> tensor<256x1024x1x1xf32>
    %cst_29 = constant dense<0.000000e+00> : tensor<256xf32>
    %289 = "tpu.conv_2d"(%287, %288, %cst_29) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %290 = "tpu.load_weight"(%0) {offset = 17420288 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %291 = "tpu.load_weight"(%0) {offset = 17421312 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %292 = "tpu.batch_norm"(%289, %290, %291) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %293 = "tpu.load_weight"(%0) {offset = 17422336 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %294 = "tpu.load_weight"(%0) {offset = 17423360 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %295 = "tpu.scale"(%292, %293, %294) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %296 = "tpu.relu"(%295) {negative_slope = 0.000000e+00 : f32} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %297 = "tpu.load_weight"(%0) {offset = 17424384 : i64} : (memref<2147483648xf32>) -> tensor<256x256x3x3xf32>
    %cst_30 = constant dense<0.000000e+00> : tensor<256xf32>
    %298 = "tpu.conv_2d"(%296, %297, %cst_30) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %299 = "tpu.load_weight"(%0) {offset = 19783680 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %300 = "tpu.load_weight"(%0) {offset = 19784704 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %301 = "tpu.batch_norm"(%298, %299, %300) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %302 = "tpu.load_weight"(%0) {offset = 19785728 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %303 = "tpu.load_weight"(%0) {offset = 19786752 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %304 = "tpu.scale"(%301, %302, %303) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %305 = "tpu.relu"(%304) {negative_slope = 0.000000e+00 : f32} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %306 = "tpu.load_weight"(%0) {offset = 19787776 : i64} : (memref<2147483648xf32>) -> tensor<1024x256x1x1xf32>
    %cst_31 = constant dense<0.000000e+00> : tensor<1024xf32>
    %307 = "tpu.conv_2d"(%305, %306, %cst_31) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %308 = "tpu.load_weight"(%0) {offset = 20836352 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %309 = "tpu.load_weight"(%0) {offset = 20840448 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %310 = "tpu.batch_norm"(%307, %308, %309) : (tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %311 = "tpu.load_weight"(%0) {offset = 20844544 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %312 = "tpu.load_weight"(%0) {offset = 20848640 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %313 = "tpu.scale"(%310, %311, %312) : (tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %314 = "tpu.eltwise"(%287, %287) : (tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %315 = "tpu.relu"(%314) {negative_slope = 0.000000e+00 : f32} : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %316 = "tpu.load_weight"(%0) {offset = 20852736 : i64} : (memref<2147483648xf32>) -> tensor<256x1024x1x1xf32>
    %cst_32 = constant dense<0.000000e+00> : tensor<256xf32>
    %317 = "tpu.conv_2d"(%315, %316, %cst_32) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %318 = "tpu.load_weight"(%0) {offset = 21901312 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %319 = "tpu.load_weight"(%0) {offset = 21902336 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %320 = "tpu.batch_norm"(%317, %318, %319) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %321 = "tpu.load_weight"(%0) {offset = 21903360 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %322 = "tpu.load_weight"(%0) {offset = 21904384 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %323 = "tpu.scale"(%320, %321, %322) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %324 = "tpu.relu"(%323) {negative_slope = 0.000000e+00 : f32} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %325 = "tpu.load_weight"(%0) {offset = 21905408 : i64} : (memref<2147483648xf32>) -> tensor<256x256x3x3xf32>
    %cst_33 = constant dense<0.000000e+00> : tensor<256xf32>
    %326 = "tpu.conv_2d"(%324, %325, %cst_33) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %327 = "tpu.load_weight"(%0) {offset = 24264704 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %328 = "tpu.load_weight"(%0) {offset = 24265728 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %329 = "tpu.batch_norm"(%326, %327, %328) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %330 = "tpu.load_weight"(%0) {offset = 24266752 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %331 = "tpu.load_weight"(%0) {offset = 24267776 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %332 = "tpu.scale"(%329, %330, %331) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %333 = "tpu.relu"(%332) {negative_slope = 0.000000e+00 : f32} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %334 = "tpu.load_weight"(%0) {offset = 24268800 : i64} : (memref<2147483648xf32>) -> tensor<1024x256x1x1xf32>
    %cst_34 = constant dense<0.000000e+00> : tensor<1024xf32>
    %335 = "tpu.conv_2d"(%333, %334, %cst_34) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %336 = "tpu.load_weight"(%0) {offset = 25317376 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %337 = "tpu.load_weight"(%0) {offset = 25321472 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %338 = "tpu.batch_norm"(%335, %336, %337) : (tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %339 = "tpu.load_weight"(%0) {offset = 25325568 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %340 = "tpu.load_weight"(%0) {offset = 25329664 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %341 = "tpu.scale"(%338, %339, %340) : (tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %342 = "tpu.eltwise"(%315, %315) : (tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %343 = "tpu.relu"(%342) {negative_slope = 0.000000e+00 : f32} : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %344 = "tpu.load_weight"(%0) {offset = 25333760 : i64} : (memref<2147483648xf32>) -> tensor<256x1024x1x1xf32>
    %cst_35 = constant dense<0.000000e+00> : tensor<256xf32>
    %345 = "tpu.conv_2d"(%343, %344, %cst_35) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %346 = "tpu.load_weight"(%0) {offset = 26382336 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %347 = "tpu.load_weight"(%0) {offset = 26383360 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %348 = "tpu.batch_norm"(%345, %346, %347) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %349 = "tpu.load_weight"(%0) {offset = 26384384 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %350 = "tpu.load_weight"(%0) {offset = 26385408 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %351 = "tpu.scale"(%348, %349, %350) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %352 = "tpu.relu"(%351) {negative_slope = 0.000000e+00 : f32} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %353 = "tpu.load_weight"(%0) {offset = 26386432 : i64} : (memref<2147483648xf32>) -> tensor<256x256x3x3xf32>
    %cst_36 = constant dense<0.000000e+00> : tensor<256xf32>
    %354 = "tpu.conv_2d"(%352, %353, %cst_36) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %355 = "tpu.load_weight"(%0) {offset = 28745728 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %356 = "tpu.load_weight"(%0) {offset = 28746752 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %357 = "tpu.batch_norm"(%354, %355, %356) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %358 = "tpu.load_weight"(%0) {offset = 28747776 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %359 = "tpu.load_weight"(%0) {offset = 28748800 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %360 = "tpu.scale"(%357, %358, %359) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %361 = "tpu.relu"(%360) {negative_slope = 0.000000e+00 : f32} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %362 = "tpu.load_weight"(%0) {offset = 28749824 : i64} : (memref<2147483648xf32>) -> tensor<1024x256x1x1xf32>
    %cst_37 = constant dense<0.000000e+00> : tensor<1024xf32>
    %363 = "tpu.conv_2d"(%361, %362, %cst_37) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %364 = "tpu.load_weight"(%0) {offset = 29798400 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %365 = "tpu.load_weight"(%0) {offset = 29802496 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %366 = "tpu.batch_norm"(%363, %364, %365) : (tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %367 = "tpu.load_weight"(%0) {offset = 29806592 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %368 = "tpu.load_weight"(%0) {offset = 29810688 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %369 = "tpu.scale"(%366, %367, %368) : (tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %370 = "tpu.eltwise"(%343, %343) : (tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %371 = "tpu.relu"(%370) {negative_slope = 0.000000e+00 : f32} : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %372 = "tpu.load_weight"(%0) {offset = 29814784 : i64} : (memref<2147483648xf32>) -> tensor<256x1024x1x1xf32>
    %cst_38 = constant dense<0.000000e+00> : tensor<256xf32>
    %373 = "tpu.conv_2d"(%371, %372, %cst_38) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %374 = "tpu.load_weight"(%0) {offset = 30863360 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %375 = "tpu.load_weight"(%0) {offset = 30864384 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %376 = "tpu.batch_norm"(%373, %374, %375) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %377 = "tpu.load_weight"(%0) {offset = 30865408 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %378 = "tpu.load_weight"(%0) {offset = 30866432 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %379 = "tpu.scale"(%376, %377, %378) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %380 = "tpu.relu"(%379) {negative_slope = 0.000000e+00 : f32} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %381 = "tpu.load_weight"(%0) {offset = 30867456 : i64} : (memref<2147483648xf32>) -> tensor<256x256x3x3xf32>
    %cst_39 = constant dense<0.000000e+00> : tensor<256xf32>
    %382 = "tpu.conv_2d"(%380, %381, %cst_39) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %383 = "tpu.load_weight"(%0) {offset = 33226752 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %384 = "tpu.load_weight"(%0) {offset = 33227776 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %385 = "tpu.batch_norm"(%382, %383, %384) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %386 = "tpu.load_weight"(%0) {offset = 33228800 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %387 = "tpu.load_weight"(%0) {offset = 33229824 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %388 = "tpu.scale"(%385, %386, %387) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %389 = "tpu.relu"(%388) {negative_slope = 0.000000e+00 : f32} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %390 = "tpu.load_weight"(%0) {offset = 33230848 : i64} : (memref<2147483648xf32>) -> tensor<1024x256x1x1xf32>
    %cst_40 = constant dense<0.000000e+00> : tensor<1024xf32>
    %391 = "tpu.conv_2d"(%389, %390, %cst_40) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %392 = "tpu.load_weight"(%0) {offset = 34279424 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %393 = "tpu.load_weight"(%0) {offset = 34283520 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %394 = "tpu.batch_norm"(%391, %392, %393) : (tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %395 = "tpu.load_weight"(%0) {offset = 34287616 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %396 = "tpu.load_weight"(%0) {offset = 34291712 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %397 = "tpu.scale"(%394, %395, %396) : (tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %398 = "tpu.eltwise"(%371, %371) : (tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %399 = "tpu.relu"(%398) {negative_slope = 0.000000e+00 : f32} : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %400 = "tpu.load_weight"(%0) {offset = 34295808 : i64} : (memref<2147483648xf32>) -> tensor<2048x1024x1x1xf32>
    %cst_41 = constant dense<0.000000e+00> : tensor<2048xf32>
    %401 = "tpu.conv_2d"(%399, %400, %cst_41) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x1024x14x14xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %402 = "tpu.load_weight"(%0) {offset = 42684416 : i64} : (memref<2147483648xf32>) -> tensor<2048xf32>
    %403 = "tpu.load_weight"(%0) {offset = 42692608 : i64} : (memref<2147483648xf32>) -> tensor<2048xf32>
    %404 = "tpu.batch_norm"(%401, %402, %403) : (tensor<1x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %405 = "tpu.load_weight"(%0) {offset = 42700800 : i64} : (memref<2147483648xf32>) -> tensor<2048xf32>
    %406 = "tpu.load_weight"(%0) {offset = 42708992 : i64} : (memref<2147483648xf32>) -> tensor<2048xf32>
    %407 = "tpu.scale"(%404, %405, %406) : (tensor<1x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %408 = "tpu.load_weight"(%0) {offset = 42717184 : i64} : (memref<2147483648xf32>) -> tensor<512x1024x1x1xf32>
    %cst_42 = constant dense<0.000000e+00> : tensor<512xf32>
    %409 = "tpu.conv_2d"(%399, %408, %cst_42) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x1024x14x14xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %410 = "tpu.load_weight"(%0) {offset = 44814336 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %411 = "tpu.load_weight"(%0) {offset = 44816384 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %412 = "tpu.batch_norm"(%409, %410, %411) : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %413 = "tpu.load_weight"(%0) {offset = 44818432 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %414 = "tpu.load_weight"(%0) {offset = 44820480 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %415 = "tpu.scale"(%412, %413, %414) : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %416 = "tpu.relu"(%415) {negative_slope = 0.000000e+00 : f32} : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %417 = "tpu.load_weight"(%0) {offset = 44822528 : i64} : (memref<2147483648xf32>) -> tensor<512x512x3x3xf32>
    %cst_43 = constant dense<0.000000e+00> : tensor<512xf32>
    %418 = "tpu.conv_2d"(%416, %417, %cst_43) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x512x7x7xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %419 = "tpu.load_weight"(%0) {offset = 54259712 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %420 = "tpu.load_weight"(%0) {offset = 54261760 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %421 = "tpu.batch_norm"(%418, %419, %420) : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %422 = "tpu.load_weight"(%0) {offset = 54263808 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %423 = "tpu.load_weight"(%0) {offset = 54265856 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %424 = "tpu.scale"(%421, %422, %423) : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %425 = "tpu.relu"(%424) {negative_slope = 0.000000e+00 : f32} : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %426 = "tpu.load_weight"(%0) {offset = 54267904 : i64} : (memref<2147483648xf32>) -> tensor<2048x512x1x1xf32>
    %cst_44 = constant dense<0.000000e+00> : tensor<2048xf32>
    %427 = "tpu.conv_2d"(%425, %426, %cst_44) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x512x7x7xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %428 = "tpu.load_weight"(%0) {offset = 58462208 : i64} : (memref<2147483648xf32>) -> tensor<2048xf32>
    %429 = "tpu.load_weight"(%0) {offset = 58470400 : i64} : (memref<2147483648xf32>) -> tensor<2048xf32>
    %430 = "tpu.batch_norm"(%427, %428, %429) : (tensor<1x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %431 = "tpu.load_weight"(%0) {offset = 58478592 : i64} : (memref<2147483648xf32>) -> tensor<2048xf32>
    %432 = "tpu.load_weight"(%0) {offset = 58486784 : i64} : (memref<2147483648xf32>) -> tensor<2048xf32>
    %433 = "tpu.scale"(%430, %431, %432) : (tensor<1x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %434 = "tpu.eltwise"(%407, %407) : (tensor<1x2048x7x7xf32>, tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %435 = "tpu.relu"(%434) {negative_slope = 0.000000e+00 : f32} : (tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %436 = "tpu.load_weight"(%0) {offset = 58494976 : i64} : (memref<2147483648xf32>) -> tensor<512x2048x1x1xf32>
    %cst_45 = constant dense<0.000000e+00> : tensor<512xf32>
    %437 = "tpu.conv_2d"(%435, %436, %cst_45) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x2048x7x7xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %438 = "tpu.load_weight"(%0) {offset = 62689280 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %439 = "tpu.load_weight"(%0) {offset = 62691328 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %440 = "tpu.batch_norm"(%437, %438, %439) : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %441 = "tpu.load_weight"(%0) {offset = 62693376 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %442 = "tpu.load_weight"(%0) {offset = 62695424 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %443 = "tpu.scale"(%440, %441, %442) : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %444 = "tpu.relu"(%443) {negative_slope = 0.000000e+00 : f32} : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %445 = "tpu.load_weight"(%0) {offset = 62697472 : i64} : (memref<2147483648xf32>) -> tensor<512x512x3x3xf32>
    %cst_46 = constant dense<0.000000e+00> : tensor<512xf32>
    %446 = "tpu.conv_2d"(%444, %445, %cst_46) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x512x7x7xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %447 = "tpu.load_weight"(%0) {offset = 72134656 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %448 = "tpu.load_weight"(%0) {offset = 72136704 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %449 = "tpu.batch_norm"(%446, %447, %448) : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %450 = "tpu.load_weight"(%0) {offset = 72138752 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %451 = "tpu.load_weight"(%0) {offset = 72140800 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %452 = "tpu.scale"(%449, %450, %451) : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %453 = "tpu.relu"(%452) {negative_slope = 0.000000e+00 : f32} : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %454 = "tpu.load_weight"(%0) {offset = 72142848 : i64} : (memref<2147483648xf32>) -> tensor<2048x512x1x1xf32>
    %cst_47 = constant dense<0.000000e+00> : tensor<2048xf32>
    %455 = "tpu.conv_2d"(%453, %454, %cst_47) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x512x7x7xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %456 = "tpu.load_weight"(%0) {offset = 76337152 : i64} : (memref<2147483648xf32>) -> tensor<2048xf32>
    %457 = "tpu.load_weight"(%0) {offset = 76345344 : i64} : (memref<2147483648xf32>) -> tensor<2048xf32>
    %458 = "tpu.batch_norm"(%455, %456, %457) : (tensor<1x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %459 = "tpu.load_weight"(%0) {offset = 76353536 : i64} : (memref<2147483648xf32>) -> tensor<2048xf32>
    %460 = "tpu.load_weight"(%0) {offset = 76361728 : i64} : (memref<2147483648xf32>) -> tensor<2048xf32>
    %461 = "tpu.scale"(%458, %459, %460) : (tensor<1x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %462 = "tpu.eltwise"(%435, %435) : (tensor<1x2048x7x7xf32>, tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %463 = "tpu.relu"(%462) {negative_slope = 0.000000e+00 : f32} : (tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %464 = "tpu.load_weight"(%0) {offset = 76369920 : i64} : (memref<2147483648xf32>) -> tensor<512x2048x1x1xf32>
    %cst_48 = constant dense<0.000000e+00> : tensor<512xf32>
    %465 = "tpu.conv_2d"(%463, %464, %cst_48) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x2048x7x7xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %466 = "tpu.load_weight"(%0) {offset = 80564224 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %467 = "tpu.load_weight"(%0) {offset = 80566272 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %468 = "tpu.batch_norm"(%465, %466, %467) : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %469 = "tpu.load_weight"(%0) {offset = 80568320 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %470 = "tpu.load_weight"(%0) {offset = 80570368 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %471 = "tpu.scale"(%468, %469, %470) : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %472 = "tpu.relu"(%471) {negative_slope = 0.000000e+00 : f32} : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %473 = "tpu.load_weight"(%0) {offset = 80572416 : i64} : (memref<2147483648xf32>) -> tensor<512x512x3x3xf32>
    %cst_49 = constant dense<0.000000e+00> : tensor<512xf32>
    %474 = "tpu.conv_2d"(%472, %473, %cst_49) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x512x7x7xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %475 = "tpu.load_weight"(%0) {offset = 90009600 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %476 = "tpu.load_weight"(%0) {offset = 90011648 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %477 = "tpu.batch_norm"(%474, %475, %476) : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %478 = "tpu.load_weight"(%0) {offset = 90013696 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %479 = "tpu.load_weight"(%0) {offset = 90015744 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %480 = "tpu.scale"(%477, %478, %479) : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %481 = "tpu.relu"(%480) {negative_slope = 0.000000e+00 : f32} : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %482 = "tpu.load_weight"(%0) {offset = 90017792 : i64} : (memref<2147483648xf32>) -> tensor<2048x512x1x1xf32>
    %cst_50 = constant dense<0.000000e+00> : tensor<2048xf32>
    %483 = "tpu.conv_2d"(%481, %482, %cst_50) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x512x7x7xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %484 = "tpu.load_weight"(%0) {offset = 94212096 : i64} : (memref<2147483648xf32>) -> tensor<2048xf32>
    %485 = "tpu.load_weight"(%0) {offset = 94220288 : i64} : (memref<2147483648xf32>) -> tensor<2048xf32>
    %486 = "tpu.batch_norm"(%483, %484, %485) : (tensor<1x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %487 = "tpu.load_weight"(%0) {offset = 94228480 : i64} : (memref<2147483648xf32>) -> tensor<2048xf32>
    %488 = "tpu.load_weight"(%0) {offset = 94236672 : i64} : (memref<2147483648xf32>) -> tensor<2048xf32>
    %489 = "tpu.scale"(%486, %487, %488) : (tensor<1x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %490 = "tpu.eltwise"(%463, %463) : (tensor<1x2048x7x7xf32>, tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %491 = "tpu.relu"(%490) {negative_slope = 0.000000e+00 : f32} : (tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %492 = "tpu.average_pool_2d"(%491) {filter_height = 7 : i32, filter_width = 7 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x2048x7x7xf32>) -> tensor<1x2048x1x1xf32>
    %493 = "tpu.reshape"(%492) : (tensor<1x2048x1x1xf32>) -> tensor<1x2048xf32>
    %494 = "tpu.load_weight"(%0) {offset = 94244864 : i64} : (memref<2147483648xf32>) -> tensor<2048x1000xf32>
    %495 = "tpu.load_weight"(%0) {offset = 102436864 : i64} : (memref<2147483648xf32>) -> tensor<1000xf32>
    %496 = "tpu.fully_connected"(%493, %494, %495) {fused_activation_function = "NONE"} : (tensor<1x2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>) -> tensor<1x1000xf32>
    return %496 : tensor<1x1000xf32>
  }
}
