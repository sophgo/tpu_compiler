

module {
  func @tpu_func(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32> {
    %0 = "tpu.load_file"() {filename = "resnet50.weight"} : () -> memref<2147483648xf32>
    %1 = "tpu.load_weight"(%0) {offset = 0 : i64} : (memref<2147483648xf32>) -> tensor<64x3x7x7xf32>
    %2 = "tpu.load_weight"(%0) {offset = 37632 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %3 = "tpu.conv_2d"(%arg0, %1, %2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
    %4 = "tpu.load_weight"(%0) {offset = 37888 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %5 = "tpu.load_weight"(%0) {offset = 38144 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %6 = "tpu.load_weight"(%0) {offset = 38400 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %7 = "tpu.batch_norm"(%3, %4, %5, %6) : (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<1xf32>) -> tensor<1x64x112x112xf32>
    %8 = "tpu.load_weight"(%0) {offset = 38404 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %9 = "tpu.load_weight"(%0) {offset = 38660 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %10 = "tpu.scale"(%7, %8, %9) : (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
    %11 = "tpu.relu"(%10) {negative_slope = 0.000000e+00 : f32} : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %12 = "tpu.max_pool_2d"(%11) {filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x64x112x112xf32>) -> tensor<1x64x56x56xf32>
    %13 = "tpu.load_weight"(%0) {offset = 38916 : i64} : (memref<2147483648xf32>) -> tensor<256x64x1x1xf32>
    %14 = "tpu.conv_2d"(%12, %13) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<1x256x56x56xf32>
    %15 = "tpu.load_weight"(%0) {offset = 104452 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %16 = "tpu.load_weight"(%0) {offset = 105476 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %17 = "tpu.load_weight"(%0) {offset = 106500 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %18 = "tpu.batch_norm"(%14, %15, %16, %17) : (tensor<1x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1xf32>) -> tensor<1x256x56x56xf32>
    %19 = "tpu.load_weight"(%0) {offset = 106504 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %20 = "tpu.load_weight"(%0) {offset = 107528 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %21 = "tpu.scale"(%18, %19, %20) : (tensor<1x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x56x56xf32>
    %22 = "tpu.load_weight"(%0) {offset = 108552 : i64} : (memref<2147483648xf32>) -> tensor<64x64x1x1xf32>
    %23 = "tpu.conv_2d"(%12, %22) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x64x56x56xf32>, tensor<64x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %24 = "tpu.load_weight"(%0) {offset = 124936 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %25 = "tpu.load_weight"(%0) {offset = 125192 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %26 = "tpu.load_weight"(%0) {offset = 125448 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %27 = "tpu.batch_norm"(%23, %24, %25, %26) : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<1xf32>) -> tensor<1x64x56x56xf32>
    %28 = "tpu.load_weight"(%0) {offset = 125452 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %29 = "tpu.load_weight"(%0) {offset = 125708 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %30 = "tpu.scale"(%27, %28, %29) : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %31 = "tpu.relu"(%30) {negative_slope = 0.000000e+00 : f32} : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %32 = "tpu.load_weight"(%0) {offset = 125964 : i64} : (memref<2147483648xf32>) -> tensor<64x64x3x3xf32>
    %33 = "tpu.conv_2d"(%31, %32) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<1x64x56x56xf32>
    %34 = "tpu.load_weight"(%0) {offset = 273420 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %35 = "tpu.load_weight"(%0) {offset = 273676 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %36 = "tpu.load_weight"(%0) {offset = 273932 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %37 = "tpu.batch_norm"(%33, %34, %35, %36) : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<1xf32>) -> tensor<1x64x56x56xf32>
    %38 = "tpu.load_weight"(%0) {offset = 273936 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %39 = "tpu.load_weight"(%0) {offset = 274192 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %40 = "tpu.scale"(%37, %38, %39) : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %41 = "tpu.relu"(%40) {negative_slope = 0.000000e+00 : f32} : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %42 = "tpu.load_weight"(%0) {offset = 274448 : i64} : (memref<2147483648xf32>) -> tensor<256x64x1x1xf32>
    %43 = "tpu.conv_2d"(%41, %42) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<1x256x56x56xf32>
    %44 = "tpu.load_weight"(%0) {offset = 339984 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %45 = "tpu.load_weight"(%0) {offset = 341008 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %46 = "tpu.load_weight"(%0) {offset = 342032 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %47 = "tpu.batch_norm"(%43, %44, %45, %46) : (tensor<1x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1xf32>) -> tensor<1x256x56x56xf32>
    %48 = "tpu.load_weight"(%0) {offset = 342036 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %49 = "tpu.load_weight"(%0) {offset = 343060 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %50 = "tpu.scale"(%47, %48, %49) : (tensor<1x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x56x56xf32>
    %51 = "tpu.eltwise"(%21, %50) : (tensor<1x256x56x56xf32>, tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %52 = "tpu.relu"(%51) {negative_slope = 0.000000e+00 : f32} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %53 = "tpu.load_weight"(%0) {offset = 344084 : i64} : (memref<2147483648xf32>) -> tensor<64x256x1x1xf32>
    %54 = "tpu.conv_2d"(%52, %53) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tensor<1x64x56x56xf32>
    %55 = "tpu.load_weight"(%0) {offset = 409620 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %56 = "tpu.load_weight"(%0) {offset = 409876 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %57 = "tpu.load_weight"(%0) {offset = 410132 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %58 = "tpu.batch_norm"(%54, %55, %56, %57) : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<1xf32>) -> tensor<1x64x56x56xf32>
    %59 = "tpu.load_weight"(%0) {offset = 410136 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %60 = "tpu.load_weight"(%0) {offset = 410392 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %61 = "tpu.scale"(%58, %59, %60) : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %62 = "tpu.relu"(%61) {negative_slope = 0.000000e+00 : f32} : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %63 = "tpu.load_weight"(%0) {offset = 410648 : i64} : (memref<2147483648xf32>) -> tensor<64x64x3x3xf32>
    %64 = "tpu.conv_2d"(%62, %63) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<1x64x56x56xf32>
    %65 = "tpu.load_weight"(%0) {offset = 558104 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %66 = "tpu.load_weight"(%0) {offset = 558360 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %67 = "tpu.load_weight"(%0) {offset = 558616 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %68 = "tpu.batch_norm"(%64, %65, %66, %67) : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<1xf32>) -> tensor<1x64x56x56xf32>
    %69 = "tpu.load_weight"(%0) {offset = 558620 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %70 = "tpu.load_weight"(%0) {offset = 558876 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %71 = "tpu.scale"(%68, %69, %70) : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %72 = "tpu.relu"(%71) {negative_slope = 0.000000e+00 : f32} : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %73 = "tpu.load_weight"(%0) {offset = 559132 : i64} : (memref<2147483648xf32>) -> tensor<256x64x1x1xf32>
    %74 = "tpu.conv_2d"(%72, %73) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<1x256x56x56xf32>
    %75 = "tpu.load_weight"(%0) {offset = 624668 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %76 = "tpu.load_weight"(%0) {offset = 625692 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %77 = "tpu.load_weight"(%0) {offset = 626716 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %78 = "tpu.batch_norm"(%74, %75, %76, %77) : (tensor<1x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1xf32>) -> tensor<1x256x56x56xf32>
    %79 = "tpu.load_weight"(%0) {offset = 626720 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %80 = "tpu.load_weight"(%0) {offset = 627744 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %81 = "tpu.scale"(%78, %79, %80) : (tensor<1x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x56x56xf32>
    %82 = "tpu.eltwise"(%52, %81) : (tensor<1x256x56x56xf32>, tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %83 = "tpu.relu"(%82) {negative_slope = 0.000000e+00 : f32} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %84 = "tpu.load_weight"(%0) {offset = 628768 : i64} : (memref<2147483648xf32>) -> tensor<64x256x1x1xf32>
    %85 = "tpu.conv_2d"(%83, %84) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tensor<1x64x56x56xf32>
    %86 = "tpu.load_weight"(%0) {offset = 694304 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %87 = "tpu.load_weight"(%0) {offset = 694560 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %88 = "tpu.load_weight"(%0) {offset = 694816 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %89 = "tpu.batch_norm"(%85, %86, %87, %88) : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<1xf32>) -> tensor<1x64x56x56xf32>
    %90 = "tpu.load_weight"(%0) {offset = 694820 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %91 = "tpu.load_weight"(%0) {offset = 695076 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %92 = "tpu.scale"(%89, %90, %91) : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %93 = "tpu.relu"(%92) {negative_slope = 0.000000e+00 : f32} : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %94 = "tpu.load_weight"(%0) {offset = 695332 : i64} : (memref<2147483648xf32>) -> tensor<64x64x3x3xf32>
    %95 = "tpu.conv_2d"(%93, %94) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<1x64x56x56xf32>
    %96 = "tpu.load_weight"(%0) {offset = 842788 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %97 = "tpu.load_weight"(%0) {offset = 843044 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %98 = "tpu.load_weight"(%0) {offset = 843300 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %99 = "tpu.batch_norm"(%95, %96, %97, %98) : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<1xf32>) -> tensor<1x64x56x56xf32>
    %100 = "tpu.load_weight"(%0) {offset = 843304 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %101 = "tpu.load_weight"(%0) {offset = 843560 : i64} : (memref<2147483648xf32>) -> tensor<64xf32>
    %102 = "tpu.scale"(%99, %100, %101) : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
    %103 = "tpu.relu"(%102) {negative_slope = 0.000000e+00 : f32} : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %104 = "tpu.load_weight"(%0) {offset = 843816 : i64} : (memref<2147483648xf32>) -> tensor<256x64x1x1xf32>
    %105 = "tpu.conv_2d"(%103, %104) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<1x256x56x56xf32>
    %106 = "tpu.load_weight"(%0) {offset = 909352 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %107 = "tpu.load_weight"(%0) {offset = 910376 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %108 = "tpu.load_weight"(%0) {offset = 911400 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %109 = "tpu.batch_norm"(%105, %106, %107, %108) : (tensor<1x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1xf32>) -> tensor<1x256x56x56xf32>
    %110 = "tpu.load_weight"(%0) {offset = 911404 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %111 = "tpu.load_weight"(%0) {offset = 912428 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %112 = "tpu.scale"(%109, %110, %111) : (tensor<1x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x56x56xf32>
    %113 = "tpu.eltwise"(%83, %112) : (tensor<1x256x56x56xf32>, tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %114 = "tpu.relu"(%113) {negative_slope = 0.000000e+00 : f32} : (tensor<1x256x56x56xf32>) -> tensor<1x256x56x56xf32>
    %115 = "tpu.load_weight"(%0) {offset = 913452 : i64} : (memref<2147483648xf32>) -> tensor<512x256x1x1xf32>
    %116 = "tpu.conv_2d"(%114, %115) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x256x56x56xf32>, tensor<512x256x1x1xf32>) -> tensor<1x512x28x28xf32>
    %117 = "tpu.load_weight"(%0) {offset = 1437740 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %118 = "tpu.load_weight"(%0) {offset = 1439788 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %119 = "tpu.load_weight"(%0) {offset = 1441836 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %120 = "tpu.batch_norm"(%116, %117, %118, %119) : (tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1xf32>) -> tensor<1x512x28x28xf32>
    %121 = "tpu.load_weight"(%0) {offset = 1441840 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %122 = "tpu.load_weight"(%0) {offset = 1443888 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %123 = "tpu.scale"(%120, %121, %122) : (tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %124 = "tpu.load_weight"(%0) {offset = 1445936 : i64} : (memref<2147483648xf32>) -> tensor<128x256x1x1xf32>
    %125 = "tpu.conv_2d"(%114, %124) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x256x56x56xf32>, tensor<128x256x1x1xf32>) -> tensor<1x128x28x28xf32>
    %126 = "tpu.load_weight"(%0) {offset = 1577008 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %127 = "tpu.load_weight"(%0) {offset = 1577520 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %128 = "tpu.load_weight"(%0) {offset = 1578032 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %129 = "tpu.batch_norm"(%125, %126, %127, %128) : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1xf32>) -> tensor<1x128x28x28xf32>
    %130 = "tpu.load_weight"(%0) {offset = 1578036 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %131 = "tpu.load_weight"(%0) {offset = 1578548 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %132 = "tpu.scale"(%129, %130, %131) : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %133 = "tpu.relu"(%132) {negative_slope = 0.000000e+00 : f32} : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %134 = "tpu.load_weight"(%0) {offset = 1579060 : i64} : (memref<2147483648xf32>) -> tensor<128x128x3x3xf32>
    %135 = "tpu.conv_2d"(%133, %134) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<1x128x28x28xf32>
    %136 = "tpu.load_weight"(%0) {offset = 2168884 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %137 = "tpu.load_weight"(%0) {offset = 2169396 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %138 = "tpu.load_weight"(%0) {offset = 2169908 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %139 = "tpu.batch_norm"(%135, %136, %137, %138) : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1xf32>) -> tensor<1x128x28x28xf32>
    %140 = "tpu.load_weight"(%0) {offset = 2169912 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %141 = "tpu.load_weight"(%0) {offset = 2170424 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %142 = "tpu.scale"(%139, %140, %141) : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %143 = "tpu.relu"(%142) {negative_slope = 0.000000e+00 : f32} : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %144 = "tpu.load_weight"(%0) {offset = 2170936 : i64} : (memref<2147483648xf32>) -> tensor<512x128x1x1xf32>
    %145 = "tpu.conv_2d"(%143, %144) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<1x512x28x28xf32>
    %146 = "tpu.load_weight"(%0) {offset = 2433080 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %147 = "tpu.load_weight"(%0) {offset = 2435128 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %148 = "tpu.load_weight"(%0) {offset = 2437176 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %149 = "tpu.batch_norm"(%145, %146, %147, %148) : (tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1xf32>) -> tensor<1x512x28x28xf32>
    %150 = "tpu.load_weight"(%0) {offset = 2437180 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %151 = "tpu.load_weight"(%0) {offset = 2439228 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %152 = "tpu.scale"(%149, %150, %151) : (tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %153 = "tpu.eltwise"(%123, %152) : (tensor<1x512x28x28xf32>, tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %154 = "tpu.relu"(%153) {negative_slope = 0.000000e+00 : f32} : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %155 = "tpu.load_weight"(%0) {offset = 2441276 : i64} : (memref<2147483648xf32>) -> tensor<128x512x1x1xf32>
    %156 = "tpu.conv_2d"(%154, %155) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<1x128x28x28xf32>
    %157 = "tpu.load_weight"(%0) {offset = 2703420 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %158 = "tpu.load_weight"(%0) {offset = 2703932 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %159 = "tpu.load_weight"(%0) {offset = 2704444 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %160 = "tpu.batch_norm"(%156, %157, %158, %159) : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1xf32>) -> tensor<1x128x28x28xf32>
    %161 = "tpu.load_weight"(%0) {offset = 2704448 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %162 = "tpu.load_weight"(%0) {offset = 2704960 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %163 = "tpu.scale"(%160, %161, %162) : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %164 = "tpu.relu"(%163) {negative_slope = 0.000000e+00 : f32} : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %165 = "tpu.load_weight"(%0) {offset = 2705472 : i64} : (memref<2147483648xf32>) -> tensor<128x128x3x3xf32>
    %166 = "tpu.conv_2d"(%164, %165) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<1x128x28x28xf32>
    %167 = "tpu.load_weight"(%0) {offset = 3295296 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %168 = "tpu.load_weight"(%0) {offset = 3295808 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %169 = "tpu.load_weight"(%0) {offset = 3296320 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %170 = "tpu.batch_norm"(%166, %167, %168, %169) : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1xf32>) -> tensor<1x128x28x28xf32>
    %171 = "tpu.load_weight"(%0) {offset = 3296324 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %172 = "tpu.load_weight"(%0) {offset = 3296836 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %173 = "tpu.scale"(%170, %171, %172) : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %174 = "tpu.relu"(%173) {negative_slope = 0.000000e+00 : f32} : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %175 = "tpu.load_weight"(%0) {offset = 3297348 : i64} : (memref<2147483648xf32>) -> tensor<512x128x1x1xf32>
    %176 = "tpu.conv_2d"(%174, %175) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<1x512x28x28xf32>
    %177 = "tpu.load_weight"(%0) {offset = 3559492 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %178 = "tpu.load_weight"(%0) {offset = 3561540 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %179 = "tpu.load_weight"(%0) {offset = 3563588 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %180 = "tpu.batch_norm"(%176, %177, %178, %179) : (tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1xf32>) -> tensor<1x512x28x28xf32>
    %181 = "tpu.load_weight"(%0) {offset = 3563592 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %182 = "tpu.load_weight"(%0) {offset = 3565640 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %183 = "tpu.scale"(%180, %181, %182) : (tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %184 = "tpu.eltwise"(%154, %183) : (tensor<1x512x28x28xf32>, tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %185 = "tpu.relu"(%184) {negative_slope = 0.000000e+00 : f32} : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %186 = "tpu.load_weight"(%0) {offset = 3567688 : i64} : (memref<2147483648xf32>) -> tensor<128x512x1x1xf32>
    %187 = "tpu.conv_2d"(%185, %186) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<1x128x28x28xf32>
    %188 = "tpu.load_weight"(%0) {offset = 3829832 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %189 = "tpu.load_weight"(%0) {offset = 3830344 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %190 = "tpu.load_weight"(%0) {offset = 3830856 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %191 = "tpu.batch_norm"(%187, %188, %189, %190) : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1xf32>) -> tensor<1x128x28x28xf32>
    %192 = "tpu.load_weight"(%0) {offset = 3830860 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %193 = "tpu.load_weight"(%0) {offset = 3831372 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %194 = "tpu.scale"(%191, %192, %193) : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %195 = "tpu.relu"(%194) {negative_slope = 0.000000e+00 : f32} : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %196 = "tpu.load_weight"(%0) {offset = 3831884 : i64} : (memref<2147483648xf32>) -> tensor<128x128x3x3xf32>
    %197 = "tpu.conv_2d"(%195, %196) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<1x128x28x28xf32>
    %198 = "tpu.load_weight"(%0) {offset = 4421708 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %199 = "tpu.load_weight"(%0) {offset = 4422220 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %200 = "tpu.load_weight"(%0) {offset = 4422732 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %201 = "tpu.batch_norm"(%197, %198, %199, %200) : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1xf32>) -> tensor<1x128x28x28xf32>
    %202 = "tpu.load_weight"(%0) {offset = 4422736 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %203 = "tpu.load_weight"(%0) {offset = 4423248 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %204 = "tpu.scale"(%201, %202, %203) : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %205 = "tpu.relu"(%204) {negative_slope = 0.000000e+00 : f32} : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %206 = "tpu.load_weight"(%0) {offset = 4423760 : i64} : (memref<2147483648xf32>) -> tensor<512x128x1x1xf32>
    %207 = "tpu.conv_2d"(%205, %206) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<1x512x28x28xf32>
    %208 = "tpu.load_weight"(%0) {offset = 4685904 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %209 = "tpu.load_weight"(%0) {offset = 4687952 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %210 = "tpu.load_weight"(%0) {offset = 4690000 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %211 = "tpu.batch_norm"(%207, %208, %209, %210) : (tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1xf32>) -> tensor<1x512x28x28xf32>
    %212 = "tpu.load_weight"(%0) {offset = 4690004 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %213 = "tpu.load_weight"(%0) {offset = 4692052 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %214 = "tpu.scale"(%211, %212, %213) : (tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %215 = "tpu.eltwise"(%185, %214) : (tensor<1x512x28x28xf32>, tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %216 = "tpu.relu"(%215) {negative_slope = 0.000000e+00 : f32} : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %217 = "tpu.load_weight"(%0) {offset = 4694100 : i64} : (memref<2147483648xf32>) -> tensor<128x512x1x1xf32>
    %218 = "tpu.conv_2d"(%216, %217) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<1x128x28x28xf32>
    %219 = "tpu.load_weight"(%0) {offset = 4956244 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %220 = "tpu.load_weight"(%0) {offset = 4956756 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %221 = "tpu.load_weight"(%0) {offset = 4957268 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %222 = "tpu.batch_norm"(%218, %219, %220, %221) : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1xf32>) -> tensor<1x128x28x28xf32>
    %223 = "tpu.load_weight"(%0) {offset = 4957272 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %224 = "tpu.load_weight"(%0) {offset = 4957784 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %225 = "tpu.scale"(%222, %223, %224) : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %226 = "tpu.relu"(%225) {negative_slope = 0.000000e+00 : f32} : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %227 = "tpu.load_weight"(%0) {offset = 4958296 : i64} : (memref<2147483648xf32>) -> tensor<128x128x3x3xf32>
    %228 = "tpu.conv_2d"(%226, %227) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<1x128x28x28xf32>
    %229 = "tpu.load_weight"(%0) {offset = 5548120 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %230 = "tpu.load_weight"(%0) {offset = 5548632 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %231 = "tpu.load_weight"(%0) {offset = 5549144 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %232 = "tpu.batch_norm"(%228, %229, %230, %231) : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<1xf32>) -> tensor<1x128x28x28xf32>
    %233 = "tpu.load_weight"(%0) {offset = 5549148 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %234 = "tpu.load_weight"(%0) {offset = 5549660 : i64} : (memref<2147483648xf32>) -> tensor<128xf32>
    %235 = "tpu.scale"(%232, %233, %234) : (tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %236 = "tpu.relu"(%235) {negative_slope = 0.000000e+00 : f32} : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %237 = "tpu.load_weight"(%0) {offset = 5550172 : i64} : (memref<2147483648xf32>) -> tensor<512x128x1x1xf32>
    %238 = "tpu.conv_2d"(%236, %237) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<1x512x28x28xf32>
    %239 = "tpu.load_weight"(%0) {offset = 5812316 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %240 = "tpu.load_weight"(%0) {offset = 5814364 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %241 = "tpu.load_weight"(%0) {offset = 5816412 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %242 = "tpu.batch_norm"(%238, %239, %240, %241) : (tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1xf32>) -> tensor<1x512x28x28xf32>
    %243 = "tpu.load_weight"(%0) {offset = 5816416 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %244 = "tpu.load_weight"(%0) {offset = 5818464 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %245 = "tpu.scale"(%242, %243, %244) : (tensor<1x512x28x28xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %246 = "tpu.eltwise"(%216, %245) : (tensor<1x512x28x28xf32>, tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %247 = "tpu.relu"(%246) {negative_slope = 0.000000e+00 : f32} : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    %248 = "tpu.load_weight"(%0) {offset = 5820512 : i64} : (memref<2147483648xf32>) -> tensor<1024x512x1x1xf32>
    %249 = "tpu.conv_2d"(%247, %248) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x512x28x28xf32>, tensor<1024x512x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %250 = "tpu.load_weight"(%0) {offset = 7917664 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %251 = "tpu.load_weight"(%0) {offset = 7921760 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %252 = "tpu.load_weight"(%0) {offset = 7925856 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %253 = "tpu.batch_norm"(%249, %250, %251, %252) : (tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1xf32>) -> tensor<1x1024x14x14xf32>
    %254 = "tpu.load_weight"(%0) {offset = 7925860 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %255 = "tpu.load_weight"(%0) {offset = 7929956 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %256 = "tpu.scale"(%253, %254, %255) : (tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %257 = "tpu.load_weight"(%0) {offset = 7934052 : i64} : (memref<2147483648xf32>) -> tensor<256x512x1x1xf32>
    %258 = "tpu.conv_2d"(%247, %257) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x512x28x28xf32>, tensor<256x512x1x1xf32>) -> tensor<1x256x14x14xf32>
    %259 = "tpu.load_weight"(%0) {offset = 8458340 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %260 = "tpu.load_weight"(%0) {offset = 8459364 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %261 = "tpu.load_weight"(%0) {offset = 8460388 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %262 = "tpu.batch_norm"(%258, %259, %260, %261) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1xf32>) -> tensor<1x256x14x14xf32>
    %263 = "tpu.load_weight"(%0) {offset = 8460392 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %264 = "tpu.load_weight"(%0) {offset = 8461416 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %265 = "tpu.scale"(%262, %263, %264) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %266 = "tpu.relu"(%265) {negative_slope = 0.000000e+00 : f32} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %267 = "tpu.load_weight"(%0) {offset = 8462440 : i64} : (memref<2147483648xf32>) -> tensor<256x256x3x3xf32>
    %268 = "tpu.conv_2d"(%266, %267) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<1x256x14x14xf32>
    %269 = "tpu.load_weight"(%0) {offset = 10821736 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %270 = "tpu.load_weight"(%0) {offset = 10822760 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %271 = "tpu.load_weight"(%0) {offset = 10823784 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %272 = "tpu.batch_norm"(%268, %269, %270, %271) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1xf32>) -> tensor<1x256x14x14xf32>
    %273 = "tpu.load_weight"(%0) {offset = 10823788 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %274 = "tpu.load_weight"(%0) {offset = 10824812 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %275 = "tpu.scale"(%272, %273, %274) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %276 = "tpu.relu"(%275) {negative_slope = 0.000000e+00 : f32} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %277 = "tpu.load_weight"(%0) {offset = 10825836 : i64} : (memref<2147483648xf32>) -> tensor<1024x256x1x1xf32>
    %278 = "tpu.conv_2d"(%276, %277) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %279 = "tpu.load_weight"(%0) {offset = 11874412 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %280 = "tpu.load_weight"(%0) {offset = 11878508 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %281 = "tpu.load_weight"(%0) {offset = 11882604 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %282 = "tpu.batch_norm"(%278, %279, %280, %281) : (tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1xf32>) -> tensor<1x1024x14x14xf32>
    %283 = "tpu.load_weight"(%0) {offset = 11882608 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %284 = "tpu.load_weight"(%0) {offset = 11886704 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %285 = "tpu.scale"(%282, %283, %284) : (tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %286 = "tpu.eltwise"(%256, %285) : (tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %287 = "tpu.relu"(%286) {negative_slope = 0.000000e+00 : f32} : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %288 = "tpu.load_weight"(%0) {offset = 11890800 : i64} : (memref<2147483648xf32>) -> tensor<256x1024x1x1xf32>
    %289 = "tpu.conv_2d"(%287, %288) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<1x256x14x14xf32>
    %290 = "tpu.load_weight"(%0) {offset = 12939376 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %291 = "tpu.load_weight"(%0) {offset = 12940400 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %292 = "tpu.load_weight"(%0) {offset = 12941424 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %293 = "tpu.batch_norm"(%289, %290, %291, %292) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1xf32>) -> tensor<1x256x14x14xf32>
    %294 = "tpu.load_weight"(%0) {offset = 12941428 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %295 = "tpu.load_weight"(%0) {offset = 12942452 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %296 = "tpu.scale"(%293, %294, %295) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %297 = "tpu.relu"(%296) {negative_slope = 0.000000e+00 : f32} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %298 = "tpu.load_weight"(%0) {offset = 12943476 : i64} : (memref<2147483648xf32>) -> tensor<256x256x3x3xf32>
    %299 = "tpu.conv_2d"(%297, %298) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<1x256x14x14xf32>
    %300 = "tpu.load_weight"(%0) {offset = 15302772 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %301 = "tpu.load_weight"(%0) {offset = 15303796 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %302 = "tpu.load_weight"(%0) {offset = 15304820 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %303 = "tpu.batch_norm"(%299, %300, %301, %302) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1xf32>) -> tensor<1x256x14x14xf32>
    %304 = "tpu.load_weight"(%0) {offset = 15304824 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %305 = "tpu.load_weight"(%0) {offset = 15305848 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %306 = "tpu.scale"(%303, %304, %305) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %307 = "tpu.relu"(%306) {negative_slope = 0.000000e+00 : f32} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %308 = "tpu.load_weight"(%0) {offset = 15306872 : i64} : (memref<2147483648xf32>) -> tensor<1024x256x1x1xf32>
    %309 = "tpu.conv_2d"(%307, %308) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %310 = "tpu.load_weight"(%0) {offset = 16355448 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %311 = "tpu.load_weight"(%0) {offset = 16359544 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %312 = "tpu.load_weight"(%0) {offset = 16363640 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %313 = "tpu.batch_norm"(%309, %310, %311, %312) : (tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1xf32>) -> tensor<1x1024x14x14xf32>
    %314 = "tpu.load_weight"(%0) {offset = 16363644 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %315 = "tpu.load_weight"(%0) {offset = 16367740 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %316 = "tpu.scale"(%313, %314, %315) : (tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %317 = "tpu.eltwise"(%287, %316) : (tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %318 = "tpu.relu"(%317) {negative_slope = 0.000000e+00 : f32} : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %319 = "tpu.load_weight"(%0) {offset = 16371836 : i64} : (memref<2147483648xf32>) -> tensor<256x1024x1x1xf32>
    %320 = "tpu.conv_2d"(%318, %319) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<1x256x14x14xf32>
    %321 = "tpu.load_weight"(%0) {offset = 17420412 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %322 = "tpu.load_weight"(%0) {offset = 17421436 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %323 = "tpu.load_weight"(%0) {offset = 17422460 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %324 = "tpu.batch_norm"(%320, %321, %322, %323) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1xf32>) -> tensor<1x256x14x14xf32>
    %325 = "tpu.load_weight"(%0) {offset = 17422464 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %326 = "tpu.load_weight"(%0) {offset = 17423488 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %327 = "tpu.scale"(%324, %325, %326) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %328 = "tpu.relu"(%327) {negative_slope = 0.000000e+00 : f32} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %329 = "tpu.load_weight"(%0) {offset = 17424512 : i64} : (memref<2147483648xf32>) -> tensor<256x256x3x3xf32>
    %330 = "tpu.conv_2d"(%328, %329) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<1x256x14x14xf32>
    %331 = "tpu.load_weight"(%0) {offset = 19783808 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %332 = "tpu.load_weight"(%0) {offset = 19784832 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %333 = "tpu.load_weight"(%0) {offset = 19785856 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %334 = "tpu.batch_norm"(%330, %331, %332, %333) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1xf32>) -> tensor<1x256x14x14xf32>
    %335 = "tpu.load_weight"(%0) {offset = 19785860 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %336 = "tpu.load_weight"(%0) {offset = 19786884 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %337 = "tpu.scale"(%334, %335, %336) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %338 = "tpu.relu"(%337) {negative_slope = 0.000000e+00 : f32} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %339 = "tpu.load_weight"(%0) {offset = 19787908 : i64} : (memref<2147483648xf32>) -> tensor<1024x256x1x1xf32>
    %340 = "tpu.conv_2d"(%338, %339) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %341 = "tpu.load_weight"(%0) {offset = 20836484 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %342 = "tpu.load_weight"(%0) {offset = 20840580 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %343 = "tpu.load_weight"(%0) {offset = 20844676 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %344 = "tpu.batch_norm"(%340, %341, %342, %343) : (tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1xf32>) -> tensor<1x1024x14x14xf32>
    %345 = "tpu.load_weight"(%0) {offset = 20844680 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %346 = "tpu.load_weight"(%0) {offset = 20848776 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %347 = "tpu.scale"(%344, %345, %346) : (tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %348 = "tpu.eltwise"(%318, %347) : (tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %349 = "tpu.relu"(%348) {negative_slope = 0.000000e+00 : f32} : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %350 = "tpu.load_weight"(%0) {offset = 20852872 : i64} : (memref<2147483648xf32>) -> tensor<256x1024x1x1xf32>
    %351 = "tpu.conv_2d"(%349, %350) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<1x256x14x14xf32>
    %352 = "tpu.load_weight"(%0) {offset = 21901448 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %353 = "tpu.load_weight"(%0) {offset = 21902472 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %354 = "tpu.load_weight"(%0) {offset = 21903496 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %355 = "tpu.batch_norm"(%351, %352, %353, %354) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1xf32>) -> tensor<1x256x14x14xf32>
    %356 = "tpu.load_weight"(%0) {offset = 21903500 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %357 = "tpu.load_weight"(%0) {offset = 21904524 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %358 = "tpu.scale"(%355, %356, %357) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %359 = "tpu.relu"(%358) {negative_slope = 0.000000e+00 : f32} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %360 = "tpu.load_weight"(%0) {offset = 21905548 : i64} : (memref<2147483648xf32>) -> tensor<256x256x3x3xf32>
    %361 = "tpu.conv_2d"(%359, %360) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<1x256x14x14xf32>
    %362 = "tpu.load_weight"(%0) {offset = 24264844 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %363 = "tpu.load_weight"(%0) {offset = 24265868 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %364 = "tpu.load_weight"(%0) {offset = 24266892 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %365 = "tpu.batch_norm"(%361, %362, %363, %364) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1xf32>) -> tensor<1x256x14x14xf32>
    %366 = "tpu.load_weight"(%0) {offset = 24266896 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %367 = "tpu.load_weight"(%0) {offset = 24267920 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %368 = "tpu.scale"(%365, %366, %367) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %369 = "tpu.relu"(%368) {negative_slope = 0.000000e+00 : f32} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %370 = "tpu.load_weight"(%0) {offset = 24268944 : i64} : (memref<2147483648xf32>) -> tensor<1024x256x1x1xf32>
    %371 = "tpu.conv_2d"(%369, %370) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %372 = "tpu.load_weight"(%0) {offset = 25317520 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %373 = "tpu.load_weight"(%0) {offset = 25321616 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %374 = "tpu.load_weight"(%0) {offset = 25325712 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %375 = "tpu.batch_norm"(%371, %372, %373, %374) : (tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1xf32>) -> tensor<1x1024x14x14xf32>
    %376 = "tpu.load_weight"(%0) {offset = 25325716 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %377 = "tpu.load_weight"(%0) {offset = 25329812 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %378 = "tpu.scale"(%375, %376, %377) : (tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %379 = "tpu.eltwise"(%349, %378) : (tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %380 = "tpu.relu"(%379) {negative_slope = 0.000000e+00 : f32} : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %381 = "tpu.load_weight"(%0) {offset = 25333908 : i64} : (memref<2147483648xf32>) -> tensor<256x1024x1x1xf32>
    %382 = "tpu.conv_2d"(%380, %381) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<1x256x14x14xf32>
    %383 = "tpu.load_weight"(%0) {offset = 26382484 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %384 = "tpu.load_weight"(%0) {offset = 26383508 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %385 = "tpu.load_weight"(%0) {offset = 26384532 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %386 = "tpu.batch_norm"(%382, %383, %384, %385) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1xf32>) -> tensor<1x256x14x14xf32>
    %387 = "tpu.load_weight"(%0) {offset = 26384536 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %388 = "tpu.load_weight"(%0) {offset = 26385560 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %389 = "tpu.scale"(%386, %387, %388) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %390 = "tpu.relu"(%389) {negative_slope = 0.000000e+00 : f32} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %391 = "tpu.load_weight"(%0) {offset = 26386584 : i64} : (memref<2147483648xf32>) -> tensor<256x256x3x3xf32>
    %392 = "tpu.conv_2d"(%390, %391) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<1x256x14x14xf32>
    %393 = "tpu.load_weight"(%0) {offset = 28745880 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %394 = "tpu.load_weight"(%0) {offset = 28746904 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %395 = "tpu.load_weight"(%0) {offset = 28747928 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %396 = "tpu.batch_norm"(%392, %393, %394, %395) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1xf32>) -> tensor<1x256x14x14xf32>
    %397 = "tpu.load_weight"(%0) {offset = 28747932 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %398 = "tpu.load_weight"(%0) {offset = 28748956 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %399 = "tpu.scale"(%396, %397, %398) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %400 = "tpu.relu"(%399) {negative_slope = 0.000000e+00 : f32} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %401 = "tpu.load_weight"(%0) {offset = 28749980 : i64} : (memref<2147483648xf32>) -> tensor<1024x256x1x1xf32>
    %402 = "tpu.conv_2d"(%400, %401) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %403 = "tpu.load_weight"(%0) {offset = 29798556 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %404 = "tpu.load_weight"(%0) {offset = 29802652 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %405 = "tpu.load_weight"(%0) {offset = 29806748 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %406 = "tpu.batch_norm"(%402, %403, %404, %405) : (tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1xf32>) -> tensor<1x1024x14x14xf32>
    %407 = "tpu.load_weight"(%0) {offset = 29806752 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %408 = "tpu.load_weight"(%0) {offset = 29810848 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %409 = "tpu.scale"(%406, %407, %408) : (tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %410 = "tpu.eltwise"(%380, %409) : (tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %411 = "tpu.relu"(%410) {negative_slope = 0.000000e+00 : f32} : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %412 = "tpu.load_weight"(%0) {offset = 29814944 : i64} : (memref<2147483648xf32>) -> tensor<256x1024x1x1xf32>
    %413 = "tpu.conv_2d"(%411, %412) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<1x256x14x14xf32>
    %414 = "tpu.load_weight"(%0) {offset = 30863520 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %415 = "tpu.load_weight"(%0) {offset = 30864544 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %416 = "tpu.load_weight"(%0) {offset = 30865568 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %417 = "tpu.batch_norm"(%413, %414, %415, %416) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1xf32>) -> tensor<1x256x14x14xf32>
    %418 = "tpu.load_weight"(%0) {offset = 30865572 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %419 = "tpu.load_weight"(%0) {offset = 30866596 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %420 = "tpu.scale"(%417, %418, %419) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %421 = "tpu.relu"(%420) {negative_slope = 0.000000e+00 : f32} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %422 = "tpu.load_weight"(%0) {offset = 30867620 : i64} : (memref<2147483648xf32>) -> tensor<256x256x3x3xf32>
    %423 = "tpu.conv_2d"(%421, %422) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<1x256x14x14xf32>
    %424 = "tpu.load_weight"(%0) {offset = 33226916 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %425 = "tpu.load_weight"(%0) {offset = 33227940 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %426 = "tpu.load_weight"(%0) {offset = 33228964 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %427 = "tpu.batch_norm"(%423, %424, %425, %426) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1xf32>) -> tensor<1x256x14x14xf32>
    %428 = "tpu.load_weight"(%0) {offset = 33228968 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %429 = "tpu.load_weight"(%0) {offset = 33229992 : i64} : (memref<2147483648xf32>) -> tensor<256xf32>
    %430 = "tpu.scale"(%427, %428, %429) : (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %431 = "tpu.relu"(%430) {negative_slope = 0.000000e+00 : f32} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %432 = "tpu.load_weight"(%0) {offset = 33231016 : i64} : (memref<2147483648xf32>) -> tensor<1024x256x1x1xf32>
    %433 = "tpu.conv_2d"(%431, %432) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<1x1024x14x14xf32>
    %434 = "tpu.load_weight"(%0) {offset = 34279592 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %435 = "tpu.load_weight"(%0) {offset = 34283688 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %436 = "tpu.load_weight"(%0) {offset = 34287784 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %437 = "tpu.batch_norm"(%433, %434, %435, %436) : (tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1xf32>) -> tensor<1x1024x14x14xf32>
    %438 = "tpu.load_weight"(%0) {offset = 34287788 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %439 = "tpu.load_weight"(%0) {offset = 34291884 : i64} : (memref<2147483648xf32>) -> tensor<1024xf32>
    %440 = "tpu.scale"(%437, %438, %439) : (tensor<1x1024x14x14xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %441 = "tpu.eltwise"(%411, %440) : (tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %442 = "tpu.relu"(%441) {negative_slope = 0.000000e+00 : f32} : (tensor<1x1024x14x14xf32>) -> tensor<1x1024x14x14xf32>
    %443 = "tpu.load_weight"(%0) {offset = 34295980 : i64} : (memref<2147483648xf32>) -> tensor<2048x1024x1x1xf32>
    %444 = "tpu.conv_2d"(%442, %443) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x1024x14x14xf32>, tensor<2048x1024x1x1xf32>) -> tensor<1x2048x7x7xf32>
    %445 = "tpu.load_weight"(%0) {offset = 42684588 : i64} : (memref<2147483648xf32>) -> tensor<2048xf32>
    %446 = "tpu.load_weight"(%0) {offset = 42692780 : i64} : (memref<2147483648xf32>) -> tensor<2048xf32>
    %447 = "tpu.load_weight"(%0) {offset = 42700972 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %448 = "tpu.batch_norm"(%444, %445, %446, %447) : (tensor<1x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<1xf32>) -> tensor<1x2048x7x7xf32>
    %449 = "tpu.load_weight"(%0) {offset = 42700976 : i64} : (memref<2147483648xf32>) -> tensor<2048xf32>
    %450 = "tpu.load_weight"(%0) {offset = 42709168 : i64} : (memref<2147483648xf32>) -> tensor<2048xf32>
    %451 = "tpu.scale"(%448, %449, %450) : (tensor<1x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %452 = "tpu.load_weight"(%0) {offset = 42717360 : i64} : (memref<2147483648xf32>) -> tensor<512x1024x1x1xf32>
    %453 = "tpu.conv_2d"(%442, %452) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x1024x14x14xf32>, tensor<512x1024x1x1xf32>) -> tensor<1x512x7x7xf32>
    %454 = "tpu.load_weight"(%0) {offset = 44814512 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %455 = "tpu.load_weight"(%0) {offset = 44816560 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %456 = "tpu.load_weight"(%0) {offset = 44818608 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %457 = "tpu.batch_norm"(%453, %454, %455, %456) : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1xf32>) -> tensor<1x512x7x7xf32>
    %458 = "tpu.load_weight"(%0) {offset = 44818612 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %459 = "tpu.load_weight"(%0) {offset = 44820660 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %460 = "tpu.scale"(%457, %458, %459) : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %461 = "tpu.relu"(%460) {negative_slope = 0.000000e+00 : f32} : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %462 = "tpu.load_weight"(%0) {offset = 44822708 : i64} : (memref<2147483648xf32>) -> tensor<512x512x3x3xf32>
    %463 = "tpu.conv_2d"(%461, %462) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<1x512x7x7xf32>
    %464 = "tpu.load_weight"(%0) {offset = 54259892 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %465 = "tpu.load_weight"(%0) {offset = 54261940 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %466 = "tpu.load_weight"(%0) {offset = 54263988 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %467 = "tpu.batch_norm"(%463, %464, %465, %466) : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1xf32>) -> tensor<1x512x7x7xf32>
    %468 = "tpu.load_weight"(%0) {offset = 54263992 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %469 = "tpu.load_weight"(%0) {offset = 54266040 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %470 = "tpu.scale"(%467, %468, %469) : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %471 = "tpu.relu"(%470) {negative_slope = 0.000000e+00 : f32} : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %472 = "tpu.load_weight"(%0) {offset = 54268088 : i64} : (memref<2147483648xf32>) -> tensor<2048x512x1x1xf32>
    %473 = "tpu.conv_2d"(%471, %472) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tensor<1x2048x7x7xf32>
    %474 = "tpu.load_weight"(%0) {offset = 58462392 : i64} : (memref<2147483648xf32>) -> tensor<2048xf32>
    %475 = "tpu.load_weight"(%0) {offset = 58470584 : i64} : (memref<2147483648xf32>) -> tensor<2048xf32>
    %476 = "tpu.load_weight"(%0) {offset = 58478776 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %477 = "tpu.batch_norm"(%473, %474, %475, %476) : (tensor<1x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<1xf32>) -> tensor<1x2048x7x7xf32>
    %478 = "tpu.load_weight"(%0) {offset = 58478780 : i64} : (memref<2147483648xf32>) -> tensor<2048xf32>
    %479 = "tpu.load_weight"(%0) {offset = 58486972 : i64} : (memref<2147483648xf32>) -> tensor<2048xf32>
    %480 = "tpu.scale"(%477, %478, %479) : (tensor<1x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %481 = "tpu.eltwise"(%451, %480) : (tensor<1x2048x7x7xf32>, tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %482 = "tpu.relu"(%481) {negative_slope = 0.000000e+00 : f32} : (tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %483 = "tpu.load_weight"(%0) {offset = 58495164 : i64} : (memref<2147483648xf32>) -> tensor<512x2048x1x1xf32>
    %484 = "tpu.conv_2d"(%482, %483) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x2048x7x7xf32>, tensor<512x2048x1x1xf32>) -> tensor<1x512x7x7xf32>
    %485 = "tpu.load_weight"(%0) {offset = 62689468 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %486 = "tpu.load_weight"(%0) {offset = 62691516 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %487 = "tpu.load_weight"(%0) {offset = 62693564 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %488 = "tpu.batch_norm"(%484, %485, %486, %487) : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1xf32>) -> tensor<1x512x7x7xf32>
    %489 = "tpu.load_weight"(%0) {offset = 62693568 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %490 = "tpu.load_weight"(%0) {offset = 62695616 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %491 = "tpu.scale"(%488, %489, %490) : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %492 = "tpu.relu"(%491) {negative_slope = 0.000000e+00 : f32} : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %493 = "tpu.load_weight"(%0) {offset = 62697664 : i64} : (memref<2147483648xf32>) -> tensor<512x512x3x3xf32>
    %494 = "tpu.conv_2d"(%492, %493) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<1x512x7x7xf32>
    %495 = "tpu.load_weight"(%0) {offset = 72134848 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %496 = "tpu.load_weight"(%0) {offset = 72136896 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %497 = "tpu.load_weight"(%0) {offset = 72138944 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %498 = "tpu.batch_norm"(%494, %495, %496, %497) : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1xf32>) -> tensor<1x512x7x7xf32>
    %499 = "tpu.load_weight"(%0) {offset = 72138948 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %500 = "tpu.load_weight"(%0) {offset = 72140996 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %501 = "tpu.scale"(%498, %499, %500) : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %502 = "tpu.relu"(%501) {negative_slope = 0.000000e+00 : f32} : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %503 = "tpu.load_weight"(%0) {offset = 72143044 : i64} : (memref<2147483648xf32>) -> tensor<2048x512x1x1xf32>
    %504 = "tpu.conv_2d"(%502, %503) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tensor<1x2048x7x7xf32>
    %505 = "tpu.load_weight"(%0) {offset = 76337348 : i64} : (memref<2147483648xf32>) -> tensor<2048xf32>
    %506 = "tpu.load_weight"(%0) {offset = 76345540 : i64} : (memref<2147483648xf32>) -> tensor<2048xf32>
    %507 = "tpu.load_weight"(%0) {offset = 76353732 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %508 = "tpu.batch_norm"(%504, %505, %506, %507) : (tensor<1x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<1xf32>) -> tensor<1x2048x7x7xf32>
    %509 = "tpu.load_weight"(%0) {offset = 76353736 : i64} : (memref<2147483648xf32>) -> tensor<2048xf32>
    %510 = "tpu.load_weight"(%0) {offset = 76361928 : i64} : (memref<2147483648xf32>) -> tensor<2048xf32>
    %511 = "tpu.scale"(%508, %509, %510) : (tensor<1x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %512 = "tpu.eltwise"(%482, %511) : (tensor<1x2048x7x7xf32>, tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %513 = "tpu.relu"(%512) {negative_slope = 0.000000e+00 : f32} : (tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %514 = "tpu.load_weight"(%0) {offset = 76370120 : i64} : (memref<2147483648xf32>) -> tensor<512x2048x1x1xf32>
    %515 = "tpu.conv_2d"(%513, %514) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x2048x7x7xf32>, tensor<512x2048x1x1xf32>) -> tensor<1x512x7x7xf32>
    %516 = "tpu.load_weight"(%0) {offset = 80564424 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %517 = "tpu.load_weight"(%0) {offset = 80566472 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %518 = "tpu.load_weight"(%0) {offset = 80568520 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %519 = "tpu.batch_norm"(%515, %516, %517, %518) : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1xf32>) -> tensor<1x512x7x7xf32>
    %520 = "tpu.load_weight"(%0) {offset = 80568524 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %521 = "tpu.load_weight"(%0) {offset = 80570572 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %522 = "tpu.scale"(%519, %520, %521) : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %523 = "tpu.relu"(%522) {negative_slope = 0.000000e+00 : f32} : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %524 = "tpu.load_weight"(%0) {offset = 80572620 : i64} : (memref<2147483648xf32>) -> tensor<512x512x3x3xf32>
    %525 = "tpu.conv_2d"(%523, %524) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<1x512x7x7xf32>
    %526 = "tpu.load_weight"(%0) {offset = 90009804 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %527 = "tpu.load_weight"(%0) {offset = 90011852 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %528 = "tpu.load_weight"(%0) {offset = 90013900 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %529 = "tpu.batch_norm"(%525, %526, %527, %528) : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<1xf32>) -> tensor<1x512x7x7xf32>
    %530 = "tpu.load_weight"(%0) {offset = 90013904 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %531 = "tpu.load_weight"(%0) {offset = 90015952 : i64} : (memref<2147483648xf32>) -> tensor<512xf32>
    %532 = "tpu.scale"(%529, %530, %531) : (tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %533 = "tpu.relu"(%532) {negative_slope = 0.000000e+00 : f32} : (tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %534 = "tpu.load_weight"(%0) {offset = 90018000 : i64} : (memref<2147483648xf32>) -> tensor<2048x512x1x1xf32>
    %535 = "tpu.conv_2d"(%533, %534) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tensor<1x2048x7x7xf32>
    %536 = "tpu.load_weight"(%0) {offset = 94212304 : i64} : (memref<2147483648xf32>) -> tensor<2048xf32>
    %537 = "tpu.load_weight"(%0) {offset = 94220496 : i64} : (memref<2147483648xf32>) -> tensor<2048xf32>
    %538 = "tpu.load_weight"(%0) {offset = 94228688 : i64} : (memref<2147483648xf32>) -> tensor<1xf32>
    %539 = "tpu.batch_norm"(%535, %536, %537, %538) : (tensor<1x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<1xf32>) -> tensor<1x2048x7x7xf32>
    %540 = "tpu.load_weight"(%0) {offset = 94228692 : i64} : (memref<2147483648xf32>) -> tensor<2048xf32>
    %541 = "tpu.load_weight"(%0) {offset = 94236884 : i64} : (memref<2147483648xf32>) -> tensor<2048xf32>
    %542 = "tpu.scale"(%539, %540, %541) : (tensor<1x2048x7x7xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %543 = "tpu.eltwise"(%513, %542) : (tensor<1x2048x7x7xf32>, tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %544 = "tpu.relu"(%543) {negative_slope = 0.000000e+00 : f32} : (tensor<1x2048x7x7xf32>) -> tensor<1x2048x7x7xf32>
    %545 = "tpu.average_pool_2d"(%544) {filter_height = 7 : i32, filter_width = 7 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x2048x7x7xf32>) -> tensor<1x2048x1x1xf32>
    %546 = "tpu.reshape"(%545) : (tensor<1x2048x1x1xf32>) -> tensor<1x2048xf32>
    %547 = "tpu.load_weight"(%0) {offset = 94245076 : i64} : (memref<2147483648xf32>) -> tensor<1000x2048xf32>
    %548 = "tpu.load_weight"(%0) {offset = 102437076 : i64} : (memref<2147483648xf32>) -> tensor<1000xf32>
    %549 = "tpu.fully_connected"(%546, %547, %548) : (tensor<1x2048xf32>, tensor<1000x2048xf32>, tensor<1000xf32>) -> tensor<1x1000xf32>
    %550 = "tpu.softmax"(%549) : (tensor<1x1000xf32>) -> tensor<1x1000xf32>
    return %550 : tensor<1x1000xf32>
  }
}
