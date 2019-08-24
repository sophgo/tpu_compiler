

module {
  func @tpu_func(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32> {
    %cst = constant dense<0.000000e+00> : tensor<64x3x7x7xf32>
    %cst_0 = constant dense<0.000000e+00> : tensor<64xf32>
    %0 = "tpu.conv_2d"(%arg0, %cst, %cst_0) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
    %1 = "tpu.max_pool_2d"(%0) {filter_height = 3 : i32, filter_width = 3 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x64x112x112xf32>) -> tensor<1x64x55x55xf32>
    %cst_1 = constant dense<0.000000e+00> : tensor<256x64x1x1xf32>
    %cst_2 = constant dense<0.000000e+00> : tensor<256xf32>
    %2 = "tpu.conv_2d"(%1, %cst_1, %cst_2) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x64x55x55xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>) -> tensor<1x256x55x55xf32>
    %cst_3 = constant dense<0.000000e+00> : tensor<64x64x1x1xf32>
    %cst_4 = constant dense<0.000000e+00> : tensor<64xf32>
    %3 = "tpu.conv_2d"(%1, %cst_3, %cst_4) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x64x55x55xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>) -> tensor<1x64x55x55xf32>
    %cst_5 = constant dense<0.000000e+00> : tensor<64x64x3x3xf32>
    %cst_6 = constant dense<0.000000e+00> : tensor<64xf32>
    %4 = "tpu.conv_2d"(%3, %cst_5, %cst_6) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x64x55x55xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>) -> tensor<1x64x55x55xf32>
    %cst_7 = constant dense<0.000000e+00> : tensor<256x64x1x1xf32>
    %cst_8 = constant dense<0.000000e+00> : tensor<256xf32>
    %5 = "tpu.conv_2d"(%4, %cst_7, %cst_8) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x64x55x55xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>) -> tensor<1x256x55x55xf32>
    %cst_9 = constant dense<0.000000e+00> : tensor<64x256x1x1xf32>
    %cst_10 = constant dense<0.000000e+00> : tensor<64xf32>
    %6 = "tpu.conv_2d"(%2, %cst_9, %cst_10) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x55x55xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>) -> tensor<1x64x55x55xf32>
    %cst_11 = constant dense<0.000000e+00> : tensor<64x64x3x3xf32>
    %cst_12 = constant dense<0.000000e+00> : tensor<64xf32>
    %7 = "tpu.conv_2d"(%6, %cst_11, %cst_12) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x64x55x55xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>) -> tensor<1x64x55x55xf32>
    %cst_13 = constant dense<0.000000e+00> : tensor<256x64x1x1xf32>
    %cst_14 = constant dense<0.000000e+00> : tensor<256xf32>
    %8 = "tpu.conv_2d"(%7, %cst_13, %cst_14) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x64x55x55xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>) -> tensor<1x256x55x55xf32>
    %cst_15 = constant dense<0.000000e+00> : tensor<64x256x1x1xf32>
    %cst_16 = constant dense<0.000000e+00> : tensor<64xf32>
    %9 = "tpu.conv_2d"(%2, %cst_15, %cst_16) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x55x55xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>) -> tensor<1x64x55x55xf32>
    %cst_17 = constant dense<0.000000e+00> : tensor<64x64x3x3xf32>
    %cst_18 = constant dense<0.000000e+00> : tensor<64xf32>
    %10 = "tpu.conv_2d"(%9, %cst_17, %cst_18) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x64x55x55xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>) -> tensor<1x64x55x55xf32>
    %cst_19 = constant dense<0.000000e+00> : tensor<256x64x1x1xf32>
    %cst_20 = constant dense<0.000000e+00> : tensor<256xf32>
    %11 = "tpu.conv_2d"(%10, %cst_19, %cst_20) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x64x55x55xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>) -> tensor<1x256x55x55xf32>
    %cst_21 = constant dense<0.000000e+00> : tensor<512x256x1x1xf32>
    %cst_22 = constant dense<0.000000e+00> : tensor<512xf32>
    %12 = "tpu.conv_2d"(%2, %cst_21, %cst_22) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x256x55x55xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %cst_23 = constant dense<0.000000e+00> : tensor<128x256x1x1xf32>
    %cst_24 = constant dense<0.000000e+00> : tensor<128xf32>
    %13 = "tpu.conv_2d"(%2, %cst_23, %cst_24) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x256x55x55xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %cst_25 = constant dense<0.000000e+00> : tensor<128x128x3x3xf32>
    %cst_26 = constant dense<0.000000e+00> : tensor<128xf32>
    %14 = "tpu.conv_2d"(%13, %cst_25, %cst_26) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %cst_27 = constant dense<0.000000e+00> : tensor<512x128x1x1xf32>
    %cst_28 = constant dense<0.000000e+00> : tensor<512xf32>
    %15 = "tpu.conv_2d"(%14, %cst_27, %cst_28) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %cst_29 = constant dense<0.000000e+00> : tensor<128x512x1x1xf32>
    %cst_30 = constant dense<0.000000e+00> : tensor<128xf32>
    %16 = "tpu.conv_2d"(%12, %cst_29, %cst_30) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x512x28x28xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %cst_31 = constant dense<0.000000e+00> : tensor<128x128x3x3xf32>
    %cst_32 = constant dense<0.000000e+00> : tensor<128xf32>
    %17 = "tpu.conv_2d"(%16, %cst_31, %cst_32) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %cst_33 = constant dense<0.000000e+00> : tensor<512x128x1x1xf32>
    %cst_34 = constant dense<0.000000e+00> : tensor<512xf32>
    %18 = "tpu.conv_2d"(%17, %cst_33, %cst_34) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %cst_35 = constant dense<0.000000e+00> : tensor<128x512x1x1xf32>
    %cst_36 = constant dense<0.000000e+00> : tensor<128xf32>
    %19 = "tpu.conv_2d"(%12, %cst_35, %cst_36) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x512x28x28xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %cst_37 = constant dense<0.000000e+00> : tensor<128x128x3x3xf32>
    %cst_38 = constant dense<0.000000e+00> : tensor<128xf32>
    %20 = "tpu.conv_2d"(%19, %cst_37, %cst_38) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %cst_39 = constant dense<0.000000e+00> : tensor<512x128x1x1xf32>
    %cst_40 = constant dense<0.000000e+00> : tensor<512xf32>
    %21 = "tpu.conv_2d"(%20, %cst_39, %cst_40) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %cst_41 = constant dense<0.000000e+00> : tensor<128x512x1x1xf32>
    %cst_42 = constant dense<0.000000e+00> : tensor<128xf32>
    %22 = "tpu.conv_2d"(%12, %cst_41, %cst_42) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x512x28x28xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %cst_43 = constant dense<0.000000e+00> : tensor<128x128x3x3xf32>
    %cst_44 = constant dense<0.000000e+00> : tensor<128xf32>
    %23 = "tpu.conv_2d"(%22, %cst_43, %cst_44) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x128x28x28xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>) -> tensor<1x128x28x28xf32>
    %cst_45 = constant dense<0.000000e+00> : tensor<512x128x1x1xf32>
    %cst_46 = constant dense<0.000000e+00> : tensor<512xf32>
    %24 = "tpu.conv_2d"(%23, %cst_45, %cst_46) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x128x28x28xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>) -> tensor<1x512x28x28xf32>
    %cst_47 = constant dense<0.000000e+00> : tensor<1024x512x1x1xf32>
    %cst_48 = constant dense<0.000000e+00> : tensor<1024xf32>
    %25 = "tpu.conv_2d"(%12, %cst_47, %cst_48) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x512x28x28xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %cst_49 = constant dense<0.000000e+00> : tensor<256x512x1x1xf32>
    %cst_50 = constant dense<0.000000e+00> : tensor<256xf32>
    %26 = "tpu.conv_2d"(%12, %cst_49, %cst_50) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x512x28x28xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %cst_51 = constant dense<0.000000e+00> : tensor<256x256x3x3xf32>
    %cst_52 = constant dense<0.000000e+00> : tensor<256xf32>
    %27 = "tpu.conv_2d"(%26, %cst_51, %cst_52) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %cst_53 = constant dense<0.000000e+00> : tensor<1024x256x1x1xf32>
    %cst_54 = constant dense<0.000000e+00> : tensor<1024xf32>
    %28 = "tpu.conv_2d"(%27, %cst_53, %cst_54) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %cst_55 = constant dense<0.000000e+00> : tensor<256x1024x1x1xf32>
    %cst_56 = constant dense<0.000000e+00> : tensor<256xf32>
    %29 = "tpu.conv_2d"(%25, %cst_55, %cst_56) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %cst_57 = constant dense<0.000000e+00> : tensor<256x256x3x3xf32>
    %cst_58 = constant dense<0.000000e+00> : tensor<256xf32>
    %30 = "tpu.conv_2d"(%29, %cst_57, %cst_58) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %cst_59 = constant dense<0.000000e+00> : tensor<1024x256x1x1xf32>
    %cst_60 = constant dense<0.000000e+00> : tensor<1024xf32>
    %31 = "tpu.conv_2d"(%30, %cst_59, %cst_60) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %cst_61 = constant dense<0.000000e+00> : tensor<256x1024x1x1xf32>
    %cst_62 = constant dense<0.000000e+00> : tensor<256xf32>
    %32 = "tpu.conv_2d"(%25, %cst_61, %cst_62) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %cst_63 = constant dense<0.000000e+00> : tensor<256x256x3x3xf32>
    %cst_64 = constant dense<0.000000e+00> : tensor<256xf32>
    %33 = "tpu.conv_2d"(%32, %cst_63, %cst_64) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %cst_65 = constant dense<0.000000e+00> : tensor<1024x256x1x1xf32>
    %cst_66 = constant dense<0.000000e+00> : tensor<1024xf32>
    %34 = "tpu.conv_2d"(%33, %cst_65, %cst_66) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %cst_67 = constant dense<0.000000e+00> : tensor<256x1024x1x1xf32>
    %cst_68 = constant dense<0.000000e+00> : tensor<256xf32>
    %35 = "tpu.conv_2d"(%25, %cst_67, %cst_68) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %cst_69 = constant dense<0.000000e+00> : tensor<256x256x3x3xf32>
    %cst_70 = constant dense<0.000000e+00> : tensor<256xf32>
    %36 = "tpu.conv_2d"(%35, %cst_69, %cst_70) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %cst_71 = constant dense<0.000000e+00> : tensor<1024x256x1x1xf32>
    %cst_72 = constant dense<0.000000e+00> : tensor<1024xf32>
    %37 = "tpu.conv_2d"(%36, %cst_71, %cst_72) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %cst_73 = constant dense<0.000000e+00> : tensor<256x1024x1x1xf32>
    %cst_74 = constant dense<0.000000e+00> : tensor<256xf32>
    %38 = "tpu.conv_2d"(%25, %cst_73, %cst_74) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %cst_75 = constant dense<0.000000e+00> : tensor<256x256x3x3xf32>
    %cst_76 = constant dense<0.000000e+00> : tensor<256xf32>
    %39 = "tpu.conv_2d"(%38, %cst_75, %cst_76) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %cst_77 = constant dense<0.000000e+00> : tensor<1024x256x1x1xf32>
    %cst_78 = constant dense<0.000000e+00> : tensor<1024xf32>
    %40 = "tpu.conv_2d"(%39, %cst_77, %cst_78) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %cst_79 = constant dense<0.000000e+00> : tensor<256x1024x1x1xf32>
    %cst_80 = constant dense<0.000000e+00> : tensor<256xf32>
    %41 = "tpu.conv_2d"(%25, %cst_79, %cst_80) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x1024x14x14xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %cst_81 = constant dense<0.000000e+00> : tensor<256x256x3x3xf32>
    %cst_82 = constant dense<0.000000e+00> : tensor<256xf32>
    %42 = "tpu.conv_2d"(%41, %cst_81, %cst_82) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x14x14xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    %cst_83 = constant dense<0.000000e+00> : tensor<1024x256x1x1xf32>
    %cst_84 = constant dense<0.000000e+00> : tensor<1024xf32>
    %43 = "tpu.conv_2d"(%42, %cst_83, %cst_84) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x256x14x14xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>) -> tensor<1x1024x14x14xf32>
    %cst_85 = constant dense<0.000000e+00> : tensor<2048x1024x1x1xf32>
    %cst_86 = constant dense<0.000000e+00> : tensor<2048xf32>
    %44 = "tpu.conv_2d"(%25, %cst_85, %cst_86) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x1024x14x14xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %cst_87 = constant dense<0.000000e+00> : tensor<512x1024x1x1xf32>
    %cst_88 = constant dense<0.000000e+00> : tensor<512xf32>
    %45 = "tpu.conv_2d"(%25, %cst_87, %cst_88) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 2 : i32, stride_w = 2 : i32} : (tensor<1x1024x14x14xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %cst_89 = constant dense<0.000000e+00> : tensor<512x512x3x3xf32>
    %cst_90 = constant dense<0.000000e+00> : tensor<512xf32>
    %46 = "tpu.conv_2d"(%45, %cst_89, %cst_90) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x512x7x7xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %cst_91 = constant dense<0.000000e+00> : tensor<2048x512x1x1xf32>
    %cst_92 = constant dense<0.000000e+00> : tensor<2048xf32>
    %47 = "tpu.conv_2d"(%46, %cst_91, %cst_92) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x512x7x7xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %cst_93 = constant dense<0.000000e+00> : tensor<512x2048x1x1xf32>
    %cst_94 = constant dense<0.000000e+00> : tensor<512xf32>
    %48 = "tpu.conv_2d"(%44, %cst_93, %cst_94) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x2048x7x7xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %cst_95 = constant dense<0.000000e+00> : tensor<512x512x3x3xf32>
    %cst_96 = constant dense<0.000000e+00> : tensor<512xf32>
    %49 = "tpu.conv_2d"(%48, %cst_95, %cst_96) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x512x7x7xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %cst_97 = constant dense<0.000000e+00> : tensor<2048x512x1x1xf32>
    %cst_98 = constant dense<0.000000e+00> : tensor<2048xf32>
    %50 = "tpu.conv_2d"(%49, %cst_97, %cst_98) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x512x7x7xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %cst_99 = constant dense<0.000000e+00> : tensor<512x2048x1x1xf32>
    %cst_100 = constant dense<0.000000e+00> : tensor<512xf32>
    %51 = "tpu.conv_2d"(%44, %cst_99, %cst_100) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x2048x7x7xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %cst_101 = constant dense<0.000000e+00> : tensor<512x512x3x3xf32>
    %cst_102 = constant dense<0.000000e+00> : tensor<512xf32>
    %52 = "tpu.conv_2d"(%51, %cst_101, %cst_102) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x512x7x7xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>) -> tensor<1x512x7x7xf32>
    %cst_103 = constant dense<0.000000e+00> : tensor<2048x512x1x1xf32>
    %cst_104 = constant dense<0.000000e+00> : tensor<2048xf32>
    %53 = "tpu.conv_2d"(%52, %cst_103, %cst_104) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x512x7x7xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>) -> tensor<1x2048x7x7xf32>
    %54 = "tpu.average_pool_2d"(%44) {filter_height = 7 : i32, filter_width = 7 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x2048x7x7xf32>) -> tensor<1x2048x1x1xf32>
    %55 = "tpu.reshape"(%54) : (tensor<1x2048x1x1xf32>) -> tensor<1x2048xf32>
    %cst_105 = constant dense<0.000000e+00> : tensor<2048x1000xf32>
    %cst_106 = constant dense<0.000000e+00> : tensor<1000xf32>
    %56 = "tpu.fully_connected"(%55, %cst_105, %cst_106) {fused_activation_function = "NONE"} : (tensor<1x2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>) -> tensor<1x1000xf32>
    return %56 : tensor<1x1000xf32>
  }
}
