//===-- TPUOps.td - TPU dialect operation definitions ------*- tablegen -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// Defines some operations of the TPU dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TPU_OPS
#define TPU_OPS

include "mlir/Dialect/TPU/TPUBase.td"
//===----------------------------------------------------------------------===//
// TPU op base class.
//===----------------------------------------------------------------------===//

class TPU_ConvOp<string mnemonic, string opSummary> :
    TPU_Op<mnemonic, [NoSideEffect]> {
  let summary = opSummary # " operator";

  let description = [{
    Performs convolution operation on inputs.

    Inputs:
      `inputs[0]`  : required: the input activation tensor.
      `inputs[1]`  : required: the filter weight tensor.
      `inputs[2~5]`: variadic: other input tensors depends on attributes
          - if `with_bias` is true, and `per_channel_info_is_aggregated` is
            false, a dedicated `bias` tensor will be present.
          - for `INT8` quantization, before assignWeightAddress stage,
            `rshift` tensor will be present.
          - for `INT8_MULTIPLIER`, before assignWeightAddress stage,
            `multiplier` tensor will be present.
          - for `INT8` quantization, after assignWeightAddress, `bias`
            and `rshift` and `multiplier` will be aggregated as one tensor
            therefore `per_channel_info` tensor will be present, and `bias`
            tensor will not be present.
          - if `fused_eltwise_method` is not `NONE`, another input tensor will
            be present as a `eltwise_input` tensor.
          - no matter whether a tensor is present, they present in the order
            we described above
  }];

  let arguments = (
    ins AnyTensor:$input,
    AnyTensor:$filter,
    Variadic<AnyTensor>:$variadic_tensors,
    BoolAttr:$with_bias,
    DefaultValuedAttr<I32Attr, "1">:$dilation_h_factor,
    DefaultValuedAttr<I32Attr, "1">:$dilation_w_factor,
    DefaultValuedAttr<TPU_AFAttr, "NONE">:$fused_activation_function,
    DefaultValuedAttr<TPU_EltwiseAttr, "NONE">:$fused_eltwise_method,
    DefaultValuedAttr<TPU_AFAttr, "NONE">:$fused_activation_function_after_eltwise,
    OptionalAttr<F32Attr>:$threshold_y_before_eltwise,
    TPU_PaddingAttr:$padding,
    I32Attr:$stride_h,
    I32Attr:$stride_w,
    DefaultValuedAttr<I32Attr, "1">:$group,
    DefaultValuedAttr<BoolAttr, "false">:$per_channel_info_is_aggregated,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<NonNegativeI64Attr>:$offset,
    OptionalAttr<F32Attr>:$threshold_y,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant
  );

  let results = (outs AnyTensor:$output);
}

//===----------------------------------------------------------------------===//
// TPU Load/Store op definitions.
//===----------------------------------------------------------------------===//
def TPU_LoadFileOp : TPU_Op<"load_file", [NoSideEffect]> {
  let summary = "load_file operator";

  let description = [{
    Load weight from a file into a memref.
  }];

  let arguments = (
    ins StrAttr:$filename
  );

  let results = (outs AnyMemRef:$memref);
}

def TPU_LoadWeightOp : TPU_Op<"load_weight", [NoSideEffect]> {
  let summary = "load_weight operator";

  let description = [{
    Load weight for a memref into a tensor.

    Inputs:
      `memref` : required: the weight memref
      `name`   : optional: name of the weight
      `storage`: default : type to store to a bin file
  }];

  let arguments = (
    ins AnyMemRef:$memref,
    OptionalAttr<StrAttr>:$name,
    DefaultValuedAttr<TPU_WeightAttr, "FP32">:$storage,
    OptionalAttr<NonNegativeI64Attr>:$offset,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$tensor);
}

//===----------------------------------------------------------------------===//
// TPU op definitions. (in alphabetical order)
//===----------------------------------------------------------------------===//
def TPU_BatchNormOp: TPU_Op<"batch_norm", [NoSideEffect]> {
  let summary = "BatchNorm operator";

  let description = [{
    Normalizes an array across batch and spatial dimensions.
  }];

  let arguments = (
    ins AnyTensor:$x,
    AnyTensor:$mean,
    AnyTensor:$variance,
    AnyTensor:$scale,
    DefaultValuedAttr<F32Attr, "1.0e-5">:$variance_epsilon,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<NonNegativeI64Attr>:$offset,
    OptionalAttr<F32Attr>:$threshold_y,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant
  );

  let results = (outs AnyTensor:$y);
}

def TPU_ConcatOp: TPU_Op<"concat", [NoSideEffect, SameOperandsAndResultElementType]> {
  let summary = "Concat operator";

  let description = [{
    Performs concat operation on inputs.
  }];

  let arguments = (
    ins Variadic<AnyTensor>:$val,
    DefaultValuedAttr<I32Attr, "1">: $dimension,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<NonNegativeI64Attr>:$offset,
    OptionalAttr<F32Attr>:$threshold_y,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant
  );

  let verifier = [{
    auto firstType = getOperand(0)->getType().cast<RankedTensorType>();

    auto firstShape = firstType.getShape();
    int numOperands = getNumOperands();
    for (int i = 1; i < numOperands; i++) {
      auto secondType = getOperand(i)->getType().cast<RankedTensorType>();

      if (firstType.getRank() != secondType.getRank()) {
        return emitOpError() << "operands (0) and" << "(" << i << ")" << "do not match rank.";
      }

      auto secondShape = secondType.getShape();
      for (int d = 0; d < firstType.getRank(); ++d) {
        if (firstShape[d] != secondShape[d] && d != dimension()) {
          return emitOpError() << "operands (0) and (" << "i" << "non-concat dimensions do not match ";
        }
      }
    }
    return success();
  }];

  let results = (outs AnyTensor:$res);
}

def TPU_Conv2DOp : TPU_ConvOp<"conv_2d", "Convolution">;

def TPU_CropOp: TPU_Op<"crop", [NoSideEffect]> {
  let summary = "Crop operator";

  let description = [{
    To crop, elements of the first bottom are selected to fit the dimensions
    of the second, reference bottom.
  }];

  let arguments = (
    ins AnyTensor:$input1,
    AnyTensor:$input2,
    OptionalAttr<I32Attr>:$axis,
    OptionalAttr<I32Attr>:$crop_offset_n,
    OptionalAttr<I32Attr>:$crop_offset_c,
    OptionalAttr<I32Attr>:$crop_offset_h,
    OptionalAttr<I32Attr>:$crop_offset_w,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<NonNegativeI64Attr>:$offset,
    OptionalAttr<F32Attr>:$threshold_y,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant
  );

  let results = (outs AnyTensor:$output);
}

def TPU_DeConv2DOp : TPU_ConvOp<"deconv_2d", "Convolution">;

def TPU_DetectionOutputOp: TPU_Op<"detectionoutput", [NoSideEffect]> {
  let summary = "DetectionOutput operator";

  let description = [{
    Intended for use with MultiBox detection method
  }];

  let arguments = (
    ins Variadic<AnyTensor>:$input,
    I32Attr:$num_classes,
    DefaultValuedAttr<BoolAttr, "true">:$share_location,
    OptionalAttr<NonNegativeI32Attr>:$background_label_id,
    OptionalAttr<F32Attr>:$nms_threshold,
    OptionalAttr<I32Attr>:$top_k,
    TPU_DetectionOutput_Code_typeAttr:$code_type,
    OptionalAttr<I32Attr>:$keep_top_k,
    OptionalAttr<F32Attr>:$confidence_threshold,
    OptionalAttr<F32Attr>:$threshold_y,
    OptionalAttr<NonNegativeI64Attr>:$offset,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant,
    OptionalAttr<StrAttr>:$name
  );

  let results = (outs AnyTensor:$output);
}

def TPU_DivOp: TPU_Op<"div", [NoSideEffect]> {
  let summary = "Division operator";

  let description = [{
    Do division operation.
  }];

  let arguments = (
    ins AnyTensor:$input,
    DefaultValuedAttr<F32Attr, "1.0">:$numerator,
    OptionalAttr<StrAttr>:$name,
    Variadic<AnyTensor>: $table,
    OptionalAttr<BoolAttr>: $has_table,
    OptionalAttr<NonNegativeI64Attr>:$offset,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    OptionalAttr<F32Attr>:$threshold_y,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant
  );

  let results = (outs AnyTensor:$output);
}

def TPU_EltwiseOp: TPU_Op<"eltwise", [NoSideEffect]> {
  let summary = "Eltwise operator";

  let description = [{
    Performs eltwase operation on inputs.
  }];

  let arguments = (
    ins AnyTensor:$x1,
    AnyTensor:$x2,
    Variadic<AnyTensor>:$variadic_tensors,
    DefaultValuedAttr<TPU_EltwiseAttr, "SUM">:$method,
    DefaultValuedAttr<TPU_AFAttr, "NONE">:$fused_activation_function,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<NonNegativeI64Attr>:$offset,
    OptionalAttr<F32Attr>:$threshold_y,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant
  );

  let results = (outs AnyTensor:$y);
}

def TPU_FullyConnectedOp : TPU_Op<"fully_connected", [NoSideEffect]> {
  let summary = "Fully connected operator";

  let arguments = (
    ins AnyTensor:$input,
    AnyTensor:$filter,
    Variadic<AnyTensor>:$bias,
    BoolAttr:$with_bias,
    DefaultValuedAttr<BoolAttr, "false">:$with_transpose,
    DefaultValuedAttr<TPU_AFAttr, "NONE">:$fused_activation_function,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<NonNegativeI64Attr>:$offset,
    OptionalAttr<F32Attr>:$threshold_y,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant
  );

  let results = (outs AnyTensor:$output);
}

def TPU_InputOp: TPU_Op<"input", [NoSideEffect]> {
  let summary = "Input operator";

  let description = [{
    Produces a tensor from function input, primarily for carrying threshold_y.
  }];

  let arguments = (
    ins AnyTensor:$input,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<NonNegativeI64Attr>:$offset,
    OptionalAttr<F32Attr>:$threshold_y,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant
  );

  let results = (outs AnyTensor:$output);
}

def TPU_NormalizeOp : TPU_Op<"normalize", [NoSideEffect]> {
  let summary = "Do Normalization";

  let arguments = (
    ins AnyTensor:$input,
    OptionalAttr<StrAttr>:$name,
    DefaultValuedAttr<BoolAttr, "true">:$across_spatial,
    OptionalAttr<NonNegativeI64Attr>:$offset,
    OptionalAttr<F32Attr>:$threshold_y,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant
  );

  let results = (outs AnyTensor:$output);
}

def TPU_PermuteOp: TPU_Op<"permute", [NoSideEffect]> {
  let summary = "Permute operator";

  let description = [{
    Perform permute on input.
  }];

  let arguments = (
    ins AnyTensor:$input,
    OptionalAttr<StrAttr>:$name,
    NonNegativeI32Attr:$order0,
    NonNegativeI32Attr:$order1,
    NonNegativeI32Attr:$order2,
    NonNegativeI32Attr:$order3,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    OptionalAttr<F32Attr>:$threshold_y,
    OptionalAttr<NonNegativeI64Attr>:$offset,    
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant
  );

  let results = (outs AnyTensor:$output);
}

def TPU_Pool2DOp : TPU_Op<"pool_2d", [NoSideEffect]> {
  let summary = "pool_2d operator";

  let description = [{
    Performs pooling operation on input.

    Inputs:
      `inputs[0]`: required: the input tensor
  }];

  let arguments = (
    ins AnyTensor:$input,
    TPU_PoolAttr:$pool,
    I32Attr:$filter_height,
    I32Attr:$filter_width,
    OptionalAttr<I32Attr>:$pad_top,
    OptionalAttr<I32Attr>:$pad_bottom,
    OptionalAttr<I32Attr>:$pad_left,
    OptionalAttr<I32Attr>:$pad_right,
    // CAUTION - TPU_PaddingAttr is depracated for asymmetric padding
    // Keep it for compatible
    OptionalAttr<TPU_PaddingAttr>:$padding,
    I32Attr:$stride_h,
    I32Attr:$stride_w,
    DefaultValuedAttr<TPU_AFAttr, "NONE">:$fused_activation_function,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<NonNegativeI64Attr>:$offset,
    OptionalAttr<F32Attr>:$threshold_y,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant
  );

  let results = (outs AnyTensor:$output);
}

def TPU_PowerOp: TPU_Op<"power", [NoSideEffect]> {
  let summary = "Power f(x) = (scale * x + shift) ^ power";

  let description = [{
    Power operator
  }];

  let arguments = (
    ins AnyTensor:$x,
    AnyTensor:$scale_table,
    AnyTensor:$shift_table,
    DefaultValuedAttr<F32Attr, "1.0">:$power,
    DefaultValuedAttr<F32Attr, "1.0">:$scale,
    DefaultValuedAttr<F32Attr, "0.0">:$shift,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<F32Attr>:$threshold_y,
    OptionalAttr<BoolAttr>: $has_table,
    OptionalAttr<F32Attr>:$rshift,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    OptionalAttr<NonNegativeI64Attr>:$offset,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant
  );

  let results = (outs AnyTensor:$y);
}

def TPU_PReluOp: TPU_Op<"prelu", [NoSideEffect]> {
  let summary = "PRelu operator";

  let description = [{
    PRelu operator
      x -> max(ax,x)
  }];
  let arguments = (
    ins AnyTensor:$x,
    AnyTensor:$negative_slope,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<F32Attr>:$threshold_y,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    OptionalAttr<NonNegativeI64Attr>:$offset,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant,
    Variadic<AnyTensor>:$useless
  );

  let results = (outs AnyTensor:$y);
}

def TPU_PriorBoxOp: TPU_Op<"priorbox", [NoSideEffect]> {
  let summary = "PriorBox operator";

  let description = [{
    Intended for use with MultiBox detection method to generate prior
  }];

  let arguments = (
    ins Variadic<AnyTensor>:$input,
    F32Attr:$min_size,
    I32Attr:$min_size_size,
    F32Attr:$max_size,
    I32Attr:$max_size_size,
    F32Attr:$aspect_ratio0,
    OptionalAttr<F32Attr>:$aspect_ratio1,
    I32Attr:$aspect_ratios_size,
    DefaultValuedAttr<BoolAttr, "true">:$flip,
    DefaultValuedAttr<BoolAttr, "true">:$clip,
    F32Attr:$variance0,
    F32Attr:$variance1,
    F32Attr:$variance2,
    F32Attr:$variance3,
    F32Attr:$step,
    DefaultValuedAttr<F32Attr, "0.5">:$offset,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    OptionalAttr<F32Attr>:$threshold_y,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant,
    OptionalAttr<StrAttr>:$name
  );

  let results = (outs AnyTensor:$output);
}

def TPU_ReluOp: TPU_Op<"relu", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Relu/LeakyRelu/Relu6 operator";

  let description = [{
    Element-wise Relu operator
      y = max(0, x)
    LeakyRelu, if negative_slope is no 0.0
      y = negative_slope * x for x < 0, x for x >= 0
    Relu6 (maximum = 6) or ReluM, if maximum is present
      y = min(max(x, 0), maximum).
  }];

  let arguments = (
    ins AnyTensor:$x,
    DefaultValuedAttr<F32Attr, "0.0">:$negative_slope,
    OptionalAttr<F32Attr>:$maximum,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<NonNegativeI64Attr>:$offset,
    OptionalAttr<F32Attr>:$threshold_y,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant
  );

  let results = (outs AnyTensor:$y);
}

def TPU_ReshapeOp: TPU_Op<"reshape", [NoSideEffect]> {
  let summary = "Reshape operator";

  let description = [{
    Produces a tensor with the same values but different static shape defined
    by the output type.
  }];

  let arguments = (
    ins AnyTensor:$input,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<NonNegativeI64Attr>:$offset,
    OptionalAttr<F32Attr>:$threshold_y,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant
  );

  let results = (outs AnyTensor:$output);
}

def TPU_ScaleOp: TPU_Op<"scale", [NoSideEffect]> {
  let summary = "Scale operator";

  let description = [{
    Performs scale on input.
  }];

  let arguments = (
    ins AnyTensor:$x,
    AnyTensor:$scale,
    Variadic<AnyTensor>:$bias,
    DefaultValuedAttr<TPU_AFAttr, "NONE">:$fused_activation_function,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<NonNegativeI64Attr>:$offset,
    OptionalAttr<F32Attr>:$threshold_y,
    DefaultValuedAttr<BoolAttr, "false">:$with_bias,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant
  );

  let results = (outs AnyTensor:$y);
}

def TPU_SigmoidOp: TPU_Op<"sigmoid", [NoSideEffect]> {
  let summary = "Sigmoid operator";

  let description = [{
    sigmoid
  }];

  let arguments = (
    ins AnyTensor:$input,
    OptionalAttr<StrAttr>:$name,
    Variadic<AnyTensor>: $table,
    OptionalAttr<BoolAttr>: $has_table,
    OptionalAttr<NonNegativeI64Attr>:$offset,
    OptionalAttr<F32Attr>:$threshold_y,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant
  );

  let results = (outs AnyTensor:$output);
}

def TPU_SliceOp: TPU_Op<"slice", [NoSideEffect]> {
  let summary = "Slice operator";

  let description = [{
    Slices an tensor to multiple output tensors along a given dimension (currently channel only)
    with given slice indices.
  }];

  let arguments = (
    ins AnyTensor:$input,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<I32Attr>:$axis,
    OptionalAttr<I32Attr>:$input_offset,
    OptionalAttr<NonNegativeI64Attr>:$offset,
    OptionalAttr<F32Attr>:$threshold_y,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant
  );

  let results = (outs AnyTensor:$output);
}

def TPU_SoftmaxOp: TPU_Op<"softmax", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Softmax operator";

  let description = [{
    Perform softmax on input.
  }];

  let arguments = (
    ins AnyTensor:$x,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<I32Attr>:$axis,
    OptionalAttr<NonNegativeI64Attr>:$offset,
    OptionalAttr<F32Attr>:$threshold_y,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant
  );

  let results = (outs AnyTensor:$y);
}

def TPU_SqrtOp: TPU_Op<"sqrt", [NoSideEffect]> {
  let summary = "Sqrt operator";

  let description = [{
    Do sqrt operation.
  }];

  let arguments = (
    ins AnyTensor:$input,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<F32Attr>:$threshold_y,
    Variadic<AnyTensor>: $table,
    OptionalAttr<BoolAttr>: $has_table,
    OptionalAttr<NonNegativeI64Attr>:$offset,     
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant

  );

  let results = (outs AnyTensor:$output);
}

def TPU_TanHOp: TPU_Op<"tanh", [NoSideEffect]> {
  let summary = "TanH operator";

  let description = [{
    TanH operator
  }];

  let arguments = (
    ins AnyTensor:$x,
    AnyTensor:$y0_table,
    AnyTensor:$slope,
    DefaultValuedAttr<F32Attr, "-1">:$scale,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<NonNegativeI64Attr>:$offset,
    OptionalAttr<F32Attr>:$threshold_y,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant
  );

  let results = (outs AnyTensor:$y);
}

def TPU_UpsampleOp: TPU_Op<"upsample", [NoSideEffect]> {
  let summary = "Upsample operator";

  let description = [{
    Perform upample on input.
  }];

  let arguments = (
    ins AnyTensor:$x,
    NonNegativeI32Attr:$scale,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<NonNegativeI64Attr>:$offset,
    OptionalAttr<F32Attr>:$threshold_y,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant
  );

  let results = (outs AnyTensor:$y);
}

//===----------------------------------------------------------------------===//
// TPU Quantization op definitions.
//===----------------------------------------------------------------------===//
def TPU_QuantizationOp: TPU_Op<"quantization", [NoSideEffect]> {
  let summary = "Quantization operator";

  let description = [{
    Quantize a activation tensor into int8, according to its threshold value.
      Q(x) = x * 128 / threshold, and saturate to (-128, 127) range
  }];

  let arguments = (
    ins AnyTensor:$input,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<NonNegativeI64Attr>:$offset,
    OptionalAttr<F32Attr>:$threshold,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant
  );

  let results = (outs AnyTensor:$output);
}

def TPU_DequantizationOp: TPU_Op<"dequantization", [NoSideEffect]> {
  let summary = "Dequantization operator";

  let description = [{
    Dequantize a activation tensor from int8 back to float.
      x = Q(x) * threshold / 128
  }];

  let arguments = (
    ins AnyTensor:$input,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<F32Attr>:$threshold,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant
  );

  let results = (outs AnyTensor:$output);
}


//===----------------------------------------------------------------------===//
// Staging
//===----------------------------------------------------------------------===//
def TPU_DummyDataOp: TPU_Op<"dummydata", [NoSideEffect]> {
  let summary = "dummydata";

  let description = [{
    DummyDataLayer fills any number of arbitrarily shaped blobs with random
  }];

  let arguments = (
    ins OptionalAttr<StrAttr>:$name
  );

  let results = (outs AnyTensor:$output);
}

include "mlir/Dialect/TPU/TPUTLOps.td"

#endif // TPU_OPS
