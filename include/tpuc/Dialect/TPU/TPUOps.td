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

include "tpuc/Dialect/TPU/TPUBase.td"
include "tpuc/Dialect/TPU/TPUInterface.td"
include "tpuc/Dialect/TPU/TPUAttribute.td"
//===----------------------------------------------------------------------===//
// TPU op base class.
//===----------------------------------------------------------------------===//

class TPU_ConvOp<string mnemonic, string opSummary> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = opSummary # " operator";

  let description = [{
    Performs convolution operation on inputs.

    Inputs:
      `input`           : required, the input activation tensor.
      `filter`          : required, the filter weight tensor.
      `bias`            : optional, the bias weight tensor.
      `quant_scale`     : optional, the quant scale tensor, should be a fp32
                          value for `per-tensor` quantization, or `oc` fp32
                          values for `per-channel` quantization.
      `quant_zeropoint` : optional, the quant zero_point tensor, should always
                          be a int8 value, we support asymmetric on actvations
                          only, weights are sysmmetric.
      `quant_rshift`    : optional, `quant_scale` can be expressed by a rshift
                          value (`quant.is_rshiftonly` mode), or be deccomposed
                          into a rshift and a multiplier. should be a int8
                          value for `per-tensor` quantization, or `oc` int8
                          values for `per-channel` quantization.
      `quant_multiplier`: optional, see `quant_rshift` comments, should be a
                          int8 value for `per-tensor` quantization, or `oc`
                          int32 values for `per-channel` quantization.

    Attributes:
      `param`           : required, a ConvParam struct attributes, carrying
                          stride, padding, dilation, group, is_dw, with_bias.
      `quant`           : required, a QuantParam struct attributes.
      `do_relu`         : required, a Do_relu_Param attributes.
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    AnyTensor:$filter,
    TPU_TensorOfOrNone<[F32, BF16, I16, I8]>:$bias,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_scale,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_zeropoint,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_rshift,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_multiplier,
    TPU_ConvParamAttr:$param,
    TPU_QuantParamAttr:$quant,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);

}

class TPU_Conv3dOp<string mnemonic, string opSummary> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = opSummary # " operator";

  let description = [{
    Performs convolution operation on inputs.

    Inputs:
      `input`           : required, the input activation tensor.
      `filter`          : required, the filter weight tensor.
      `bias`            : optional, the bias weight tensor.
      `quant_scale`     : optional, the quant scale tensor, should be a fp32
                          value for `per-tensor` quantization, or `oc` fp32
                          values for `per-channel` quantization.
      `quant_zeropoint` : optional, the quant zero_point tensor, should always
                          be a int8 value, we support asymmetric on actvations
                          only, weights are sysmmetric.
      `quant_rshift`    : optional, `quant_scale` can be expressed by a rshift
                          value (`quant.is_rshiftonly` mode), or be deccomposed
                          into a rshift and a multiplier. should be a int8
                          value for `per-tensor` quantization, or `oc` int8
                          values for `per-channel` quantization.
      `quant_multiplier`: optional, see `quant_rshift` comments, should be a
                          int8 value for `per-tensor` quantization, or `oc`
                          int32 values for `per-channel` quantization.

    Attributes:
      `param`           : required, a ConvParam struct attributes, carrying
                          stride, padding, dilation, group, is_dw, with_bias.
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.
      `do_relu`         : required, a Do_relu_Param attributes.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    AnyTensor:$filter,
    TPU_TensorOfOrNone<[F32, BF16, I16, I8]>:$bias,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_scale,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_zeropoint,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_rshift,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_multiplier,
    TPU_Conv3dParamAttr:$param,
    TPU_QuantParamAttr:$quant,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);

}

class TPU_EltwiseOp<string mnemonic, string opSummary> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "eltwise " # opSummary # " operator";

  let description = [{
    Performs eltwise operation on inputs.

    Inputs:
      `inputs`          : required, the variadic input tensors.
      `quant_scale`     : optional, the quant scale tensor. For add/max should
                          be one fp32 value for each input tensor. For mul
                          should be one fp32 value for the output tensor.
      `quant_zeropoint` : optional, the quant zero_point tensor, should be one
                          int8 value for the output tensor.
      `quant_rshift`    : optional, should always be a int8 value for all
                          input tensors.
      `quant_multiplier`: optional, the multiplier. For add/max should be one
                          int8 value for each input tensor. For mul should be
                          one int8 value applied on the output tensor.

    Attributes:
      `do_relu`         : required, if need to preform relu on result.
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantPerTensorOnly`
      `TPU_QuantSupportAsymmetric`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins Variadic<AnyTensor>:$inputs,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_scale,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_zeropoint,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_rshift,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_multiplier,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<BoolAttr, "false">:$do_early_stride,
    DefaultValuedAttr<I32Attr, "1">:$early_stride_h,
    DefaultValuedAttr<I32Attr, "1">:$early_stride_w,
    TPU_QuantParamAttr:$quant,
    OptionalAttr<F32ArrayAttr>:$coeff,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);

  let extraClassDeclaration = [{
    unsigned getNumInputs() { return getNumOperands() - 4; }
  }];
}

def TPU_EltwiseAddOp : TPU_EltwiseOp<"eltwise_add", "Add">;
def TPU_EltwiseMaxOp : TPU_EltwiseOp<"eltwise_max", "Max">;
def TPU_EltwiseMinOp : TPU_EltwiseOp<"eltwise_min", "Min">;
def TPU_EltwiseMulOp : TPU_EltwiseOp<"eltwise_mul", "Mul">;

class TPU_LutOp<string mnemonic,  string opSummary> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {

  let summary = opSummary # " operator";
  let description = [{
    elements of the bottom do activation, reference bottom.
    Performs activation operation on input.
    In Int8, we use lookup table find output

    Inputs:
      `inputs`          : required, variadic input tensors.
      `table`           : optional, the quantize lookup table,which is reused by INT8 and BF16.
      `table_mantissa`  : optional, the quantize lookup table,only used by BF16 .

    Attributes:
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.
      `max_range`       : required, we cut to max_range and outlier saturate it
      `min_range`       : required, we cut to min_range and outlier saturate it
      `layer_id`        : optional, id for profiling.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantPerTensorOnly`
      `TPU_QuantSupportAsymmetric`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins Variadic<AnyTensor>:$inputs,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$table,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$table_mantissa,
    StrAttr:$name,
    DefaultValuedAttr<F32Attr, "8">:$max_range,
    DefaultValuedAttr<F32Attr, "-8">:$min_range,
    DefaultValuedAttr<F32Attr, "0">:$coeff,
    TPU_QuantParamAttr:$quant
  );

  let results = (outs AnyTensor:$output);
}

def TPU_EluOp: TPU_LutOp<"elu", "Elu">;
def TPU_LogOp : TPU_LutOp<"log", "Log">;
def TPU_PowOp : TPU_LutOp<"pow", "Pow">;
def TPU_SwishOp : TPU_LutOp<"swish", "Swish">;
def TPU_TanHOp : TPU_LutOp<"tanh", "TanH">;

//===----------------------------------------------------------------------===//
// TPU op definitions. (in alphabetical order)
//===----------------------------------------------------------------------===//
def TPU_AbsOp: TPU_Op<"abs",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "Abs operator";

  let description = [{
    Element-wise abs operator
      y = max(x * -1, x)

    Inputs:
      `input`           : required, the input activation tensor.

    Attributes:
      `maximum`         : optional, the maximum value for Abs when present.
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantBypass`
      `TPU_QuantPerTensorOnly`
      `TPU_QuantSupportAsymmetric`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def TPU_ArgMaxOp: TPU_Op<"argmax",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "ArgMax operator";

  let description = [{
     Performs argmax operation on inputs.

    Inputs:
      `input`          : required, input tensors.

    Attributes:
      `name`            : required, name for calibration, comparing, or debug.
      `axis`            : required, axis in which to compute the arg indices


    Result:
      `output`          : result tensor.
  }];

  let arguments = (
    ins AnyTensor:$input,
    DefaultValuedAttr<I32Attr, "1">:$axis,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def TPU_BatchNormOp: TPU_Op<"batch_norm",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>]> {
  let summary = "BatchNorm operator";

  let description = [{
    Normalizes an array across batch and spatial dimensions.


    Inputs:
      `input`           : required, the input activation tensor.
      `mean`            : required, the mean weight tensor, per-channel value.
      `variance`        : required, the variance weight tensor, per-channel
                          value.
      `scale`           : required, the scale weight tensor, one value.
      `bias`            : option, pspnet BN layer used

    Attributes:
      `variance_epsilon`: required, eps for variance.
      `name`            : required, name for calibration, comparing, or debug.
      `frozon`          : optional, pspnet BN layer used


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface` : support common TPU Op interface
      No `TpuOpQuantInterface`  : convert to ScaleOp befor quant
      NO `TpuOpLowerInterface`  : convert to ScaleOp befor quant
  }];

  let arguments = (
    ins AnyTensor:$input,
    AnyTensor:$mean,
    AnyTensor:$variance,
    AnyTensor:$scale,
    TPU_TensorOfOrNone<[F32, BF16, I16, I8]>:$bias,
    DefaultValuedAttr<F32Attr, "1.0e-5">:$variance_epsilon,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);

  let hasCanonicalizer = 1;
}

class TPU_BroadcastOp<string mnemonic, string opSummary> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "Broadcast " # opSummary # " operator";

  let description = [{
    Performs scale on input.
    Inputs:
      `inputs`          : required, input tensors.
      `multiplier`      : required, multiplier tensors.
      `quant_scale`     : optional,
      `quant_zeropoint` : optional,
      `quant_rshift`    : optional,
      `quant_multiplier`: optional,

    Attributes:
      `do_relu`         : required, if need to preform relu on result.
      `align_right`     : required, [3,1,3]x[3]=> if true, [3,1,3]x[1,1,3]; else [3, 1, 1]
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins Variadic<AnyTensor>:$inputs,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_scale,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_zeropoint,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_rshift,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_multiplier,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<BoolAttr, "true">:$align_right,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}
def TPU_BroadcastAddOp : TPU_BroadcastOp<"broadcast_add", "Add">;
def TPU_BroadcastSubOp : TPU_BroadcastOp<"broadcast_sub", "Sub">;
def TPU_BroadcastMulOp : TPU_BroadcastOp<"broadcast_mul", "Mul">;

def TPU_ConcatOp: TPU_Op<"concat",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "Concat operator";

  let description = [{
    Performs concat operation on inputs.

    Inputs:
      `inputs`          : required, variadic input tensors.
      `quant_scale`     : optional, the quant scale tensor. one fp32 value
                          for each input tensor.
      `quant_zeropoint` : optional, the quant zero_point tensor, should be one
                          int8 value for the output tensor.
      `quant_rshift`    : optional, should always be a int8 value for all
                          input tensors.
      `quant_multiplier`: optional, the multiplier. one int8 value for each
                          input tensor.

    Attributes:
      `axis`            : required, the axis the concat is applying
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantPerTensorOnly`
      `TPU_QuantSupportAsymmetric`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins Variadic<AnyTensor>:$inputs,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_scale,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_zeropoint,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_rshift,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_multiplier,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<I32Attr, "1">:$axis,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);

  let extraClassDeclaration = [{
    unsigned getNumInputs() { return getNumOperands() - 4; }
  }];

  let verifier = [{
    auto firstType = getOperand(0).getType().cast<RankedTensorType>();

    auto firstShape = firstType.getShape();
    int numOperands = getNumOperands();
    for (int i = 1; i < numOperands - 4; i++) {
      auto secondType = getOperand(i).getType().cast<RankedTensorType>();

      if (firstType.getRank() != secondType.getRank()) {
        return emitOpError() << "operands (0) and" << "(" << i << ")" << "do not match rank.";
      }

      auto secondShape = secondType.getShape();
      for (int d = 0; d < firstType.getRank(); ++d) {
        if (firstShape[d] != secondShape[d] && d != (int)axis()) {
          return emitOpError() << "operands (0) and (" << "i" << "non-concat dimensions do not match ";
        }
      }
    }
    return success();
  }];
}

def TPU_Conv2DOp : TPU_ConvOp<"conv_2d", "Convolution">;
def TPU_Conv3DOp : TPU_Conv3dOp<"conv_3d", "Convolution">;

def TPU_CropOp: TPU_Op<"crop",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "Crop operator";

  let description = [{
     Performs crop operation on inputs.

    Inputs:
      `input`          : required, input tensors.

    Attributes:
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.
      `crop_offset`     : required, crop offset.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantPerTensorOnly`
      `TPU_QuantSupportAsymmetric`
      `TPU_QuantSameInputResultScale` : input and result share same threshold
                                        or scale (bypassed).
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    I32ArrayAttr:$crop_offset,
    OptionalAttr<I32ArrayAttr>:$steps,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name
  );
  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def TPU_ConvFcOp : TPU_Op<"convfc",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "Matrix multiply by conv operator";

  let description = [{
    Inputs:
      `inputs`          : required, input tensors.

    Attributes:
      `do_relu`         : required, if need to preform relu on result.
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    AnyTensor:$filter,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_scale,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_zeropoint,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def TPU_CopyOp: TPU_Op<"copy",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {

  let summary = "Copy operator";

  let arguments = (
    ins AnyTensor:$input,
    I32ArrayAttr:$shape,
    I32ArrayAttr:$input_stride,
    I32ArrayAttr:$output_stride,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def TPU_CscOp: TPU_Op<"csc",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "Color space convert for model's inputs";

  let description = [{
     Performs csc operation on inputs.

    Inputs:
      `input`          : required, input tensors.

    Attributes:
      `pixel_format`    : required, pixel format type.
      `name`            : required, name for calibration, comparing, or debug.
      `y_align`         : width alignment of channel y.
      `w_align`         : width alignment of channel uv.
      `channel_align`   : alignment of channel.

    Result:
      `output`          : result tensor.
  }];

  let arguments = (
    ins AnyTensor:$input,
    StrAttr:$pixel_format,
    TPU_QuantParamAttr:$quant,
    BoolAttr:$aligned,
    I32Attr:$y_align,
    I32Attr:$w_align,
    I32Attr:$channel_align,
    StrAttr:$name
  );
  let results = (outs AnyTensor:$output);
}

def TPU_DeConv2DOp : TPU_ConvOp<"deconv_2d", "Deconvolution">;

def TPU_DetectionOutputOp : TPU_Op<"detectionoutput",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>]> {
  let summary = "DetectionOutput operator";

  let description = [{
    Intended for use with MultiBox detection method

    Inputs:
      `inputs`                  : required, input tensors.
      `num_classes`             : required, class number.
      `share_location`          : required, If true, bounding box are shared among different classes.
      `background_label_id`     : required, background label id.
      `top_k`                   : required, Maximum number of results to be kept.
      `code_type`               : required, Type of coding method for bbox.
      `keep_top_k`              : required, Number of total bboxes to be kept per image after nms step.
      `confidence_threshold`    : required, classification confidence threshold

    Attributes:
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins Variadic<AnyTensor>:$input,
    I32Attr:$num_classes,
    DefaultValuedAttr<BoolAttr, "true">:$share_location,
    I32Attr:$background_label_id,
    F32Attr:$nms_threshold,
    I32Attr:$top_k,
    TPU_DetectionOutput_Code_typeAttr:$code_type,
    I32Attr:$keep_top_k,
    F32Attr:$confidence_threshold,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def TPU_DilateOp: TPU_Op<"dilate",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "Dilate operator";

  let description = [{
    fill with const in dilated activation


    Inputs:
      `input`           : required, the input activation tensor.

    Attributes:
      `name`            : required, name for calibration, comparing, or debug.
      `fill_constant`   : required, fill the dilated value
      `ins`             : required, ins[0] means ins_w, ins[1] means ins_h


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    DefaultValuedAttr<I32Attr, "0">:$fill_constant,
    OptionalAttr<I32ArrayAttr>:$ins,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def TPU_MulConstOp: TPU_Op<"mul_const",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "MulConst operator";

  let description = [{
    Perform MulConst on input.

    Inputs:
      `input`           : required, the input activation tensor.

    Attributes:
      `do_relu`         : required, if need to preform relu on result.
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.
      `const_val`       : operand, to be add or multiply

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    F32Attr:$const_val
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def TPU_FullyConnectedOp : TPU_Op<"fully_connected",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "Fully connected operator";

  let description = [{
    Performs scale on input.

    Inputs:
      `inputs`          : required, input tensors.
      `filter`          : required, filter tensors.
      `bias`            : optional, multiplier tensors.
      `quant_scale`     : optional,
      `quant_zeropoint` : optional,
      `quant_rshift`    : optional,
      `quant_multiplier`: optional,

    Attributes:
      `do_relu`         : required, if need to preform relu on result.
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    AnyTensor:$filter,
    TPU_TensorOfOrNone<[F32, BF16, I16, I8]>:$bias,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_scale,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_zeropoint,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_rshift,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_multiplier,
    DefaultValuedAttr<BoolAttr, "false">:$input_transpose,
    DefaultValuedAttr<BoolAttr, "false">:$output_transpose,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def TPU_ExpOp : TPU_Op<"exp",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {

  let summary = " Exp operator";
  let arguments = (
    ins Variadic<AnyTensor>:$inputs,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$table,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$table_mantissa,
    StrAttr:$name,
    DefaultValuedAttr<F32Attr, "1">:$scale,
    DefaultValuedAttr<F32Attr, "0">:$bias,
    DefaultValuedAttr<F32Attr, "8">:$max_range,
    DefaultValuedAttr<F32Attr, "-8">:$min_range,
    DefaultValuedAttr<I32Attr, "-1">:$layer_id,
    TPU_QuantParamAttr:$quant
  );

  let results = (outs AnyTensor:$output);
}

def TPU_FrcnDetectionOp : TPU_Op<"frcn_detection",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>]> {
  let summary = "Faster rcnn detection operator";

  let description = [{

    Inputs:
      `inputs`          : required, input tensors.

    Attributes:
      `class_num`       : required, detection class num.
      `obj_threshold`   : required, object threshold.
      `nms_threshold`   : required, nms threshold.
      `keep_topk`       : required, keep top k.
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface` : support common TPU Op interface
  }];


  let arguments = (
    ins Variadic<AnyTensor>:$inputs,
    I32Attr:$class_num,
    F32Attr:$obj_threshold,
    F32Attr:$nms_threshold,
    I32Attr:$keep_topk,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def TPU_GruOp: TPU_Op<"gru",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "GRU operator";

  let description = [{
    Performs gru on input.

    Inputs:
      `input`           : required, the input activation tensor.
      `recurrence`       : required, recurrence weight tesnor for update, reset, and hidden gates.
      `bias`            : optional, bias vectors for update, reset, and hidden gates.
      `initial_h`       : optional, initial value of the hidden.

    Attributes:
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    AnyTensor:$recurrence,
    TPU_TensorOfOrNone<[F32, BF16, I16, I8]>:$bias,
    TPU_TensorOfOrNone<[F32, BF16, I16, I8]>:$initial_h,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$sigmoid_table,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$sigmoid_slope_table,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$tanh_table,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$tanh_slope_table,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name,
    DefaultValuedAttr<BoolAttr, "true">:$linear_before_reset,
    DefaultValuedAttr<BoolAttr, "false">:$bidirectional
  );

  let results = (outs AnyTensor:$output);
}

def TPU_InputOp: TPU_Op<"input",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>]> {
  let summary = "Input operator";

  let description = [{
    Produces a tensor from input, primarily for carrying threshold_y.

    Inputs:
      `inputs`          : required, input tensors.

    Attributes:
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.


  }];

  let arguments = (
    ins AnyTensor:$input,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name,
    OptionalAttr<TPU_PreprocessParamAttr>:$preprocess,
    OptionalAttr<I64Attr>:$gaddr
  );

  let results = (outs AnyTensor:$output);
}

def TPU_InterpOp: TPU_Op<"interp",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "Interp operator";

  let description = [{
    Perform interp on input.

    Inputs:
      `input`           : required, the input activation tensor.
      `quant_scale`     : optional,
      `quant_zeropoint` : optional,
      `quant_rshift`    : optional,
      `quant_multiplier`: optional,

    Attributes:
      `quant`           : required, a QuantParam struct attributes, to carrying
                          threshold, although the quantization is bypassed.
      `name`            : required, name for calibration, comparing, or debug.

      `shrink_factor`   : optional, shrink h/w to small one, leverage conv
      `zoom_factor`     : optional, enlarge h/w to larger one, leverage deconv
      `pad_beg`         : optional, padding twice in h/w
      `pad_end`         : optional, padding twice in h/w
      `coordinate_transformation_mode`: optional, This attribute describes how to transform the coordinate in the resized tensor to the coordinate in the original tensor.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantPerTensorOnly`
      `TPU_QuantSupportAsymmetric`
      `TPU_QuantSameInputResultScale` : input and result share same threshold
                                        or scale (bypassed).
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_scale,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_zeropoint,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_rshift,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_multiplier,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name,
    DefaultValuedAttr<I32Attr, "0">:$height,
    DefaultValuedAttr<I32Attr, "0">:$width,
    DefaultValuedAttr<I32Attr, "0">:$shrink_factor,
    DefaultValuedAttr<I32Attr, "0">:$zoom_factor,
    DefaultValuedAttr<I32Attr, "0">:$pad_beg,
    DefaultValuedAttr<I32Attr, "0">:$pad_end,
    DefaultValuedAttr<StrAttr, "align_corners">:$coordinate_transformation_mode
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def TPU_InstanceNormOp: TPU_Op<"instance_norm",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "InstanceNorm operator";

  let description = [{
    Normalizes an array across channel dimensions.
    y = scale * (x - mean) / sqrt(variance + epsilon) + B, where mean and variance are computed per instance per channel.


    Inputs:
      `input`           : required, the input activation tensor.
      `scale`           : required, The input 1-dimensional scale tensor of size C.
      `bias`            : required, The input 1-dimensional bias tensor of size C.

    Attributes:
      `variance_epsilon`: required, eps for variance.
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
  }];

  let arguments = (
    ins AnyTensor:$input,
    AnyTensor:$scale,
    TPU_TensorOfOrNone<[F32, BF16, I16, I8]>:$bias,
    StrAttr:$name,
    DefaultValuedAttr<F32Attr, "1.0e-5">:$variance_epsilon,
    TPU_QuantParamAttr:$quant
  );

  let results = (outs AnyTensor:$output);
}

def TPU_LayerNormOp: TPU_Op<"layer_norm",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "Layer Normalize operator";

  let description = [{
    Layer Normalize operator

    Inputs:
      `input`           : required, the input activation tensor.
      `table`           : required, the quantize lookup table,which is reused by INT8 and BF16.
      `table_mantissa`  : required, the quantize lookup table,only used by BF16 .

    Attributes:
      `normalized_shape`: required, normalize over the last dims.
      `eps`             : optional, default: 1e-5
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantPerTensorOnly`
      `TPU_QuantSupportAsymmetric`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$table,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$mantissa_table,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$scale,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$bias,
    I32ArrayAttr:$normalized_shape,
    DefaultValuedAttr<F32Attr, "1.0e-5">:$eps,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def TPU_LeakyReluOp: TPU_Op<"leaky_relu",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "Leaky Relu operator";

  let description = [{
    Leaky Relu operator
      y = negative_slope * x for x < 0, x for x >= 0

    Inputs:
      `input`               : required, the input activation tensor.
      `quant_pos_scale`     : optional, the quant scale tensor, for positive
                              values, one fp32 value.
      `quant_pos_zeropoint` : optional, the quant zero_point tensor, for
                              positive values, one int8 value.
      `quant_neg_scale`     : optional, the quant scale tensor, for negative
                              values, one fp32 value.
      `quant_neg_zeropoint` : optional, the quant zero_point tensor, for
                              negative values, one int8 value.
      `quant_pos_rshift`    : optional, rshift for positive values, one int32
                              value.
      `quant_pos_multiplier`: optional, multiplier for positive values, one
                              int8 value.
      `quant_neg_rshift`    : optional, rshift for negative values, one int32
                              value.
      `quant_neg_multiplier`: optional, multiplier for negative values, one
                              int8 value.

    Attributes:
      `negative_slope`  : required, the negative_slope.
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantPerTensorOnly`
      `TPU_QuantSupportAsymmetric`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_pos_scale,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_pos_zeropoint,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_neg_scale,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_neg_zeropoint,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_pos_rshift,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_pos_multiplier,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_neg_rshift,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_neg_multiplier,
    F32Attr:$negative_slope,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

class TPU_LrnPartOp<string mnemonic,  string opSummary> : TPU_Op<mnemonic,
    [NoSideEffect,
     DeclareOpInterfaceMethods<TpuOpCommonInterface>,
     DeclareOpInterfaceMethods<TpuOpQuantInterface>]>  {

  let summary = opSummary # " operator";

  let description = [{
    LRN Part on input to calc threshold and will be removed.

    Inputs:
      `input`           : required, the input activation tensor.

    Attributes:
      `local_size`      : optional, local size.
      `alpha`           : optional, alpha.
      `beta`            : optional, beta.
      `k`               : optional, k.
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

  }];

  let arguments = (
    ins AnyTensor:$input,
    DefaultValuedAttr<I32Attr, "5">:$local_size,
    DefaultValuedAttr<F32Attr, "1.0">:$alpha,
    DefaultValuedAttr<F32Attr, "0.75">:$beta,
    DefaultValuedAttr<F32Attr, "1.0">:$k,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def TPU_LrnOneOp : TPU_LrnPartOp<"lrn_one", "lrn part one, for middle data sq">;
def TPU_LrnTwoOp : TPU_LrnPartOp<"lrn_two", "lrn part two, for middle data sum_sq">;
def TPU_LrnThreeOp : TPU_LrnPartOp<"lrn_three", "lrn part three, for middle data scale">;

def TPU_LrnOp: TPU_Op<"lrn",
    [NoSideEffect,
     DeclareOpInterfaceMethods<TpuOpCommonInterface>,
     DeclareOpInterfaceMethods<TpuOpQuantInterface>,
     DeclareOpInterfaceMethods<TpuOpLowerInterface>]>  {

  let summary = "lrn operator";

  let description = [{
    LRN on input.

    Inputs:
      `input`           : required, the input activation tensor.
      `sqr_lut`         : required, sqr lut table used by backend.
      `power_lut`       : required, power lut table used by backend.
      `scale`           : required, scale data

    Attributes:
      `local_size`      : optional, local size.
      `alpha`           : optional, alpha.
      `beta`            : optional, beta.
      `k`               : optional, k.
      `sum_rshift`      : optional, auto generate
      `lrn_rshift`      : optional, auto generate
      `quant_data0`     : optional, auto generate
      `quant_data1`     : optional, auto generate
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantPerTensorOnly`
      `TPU_QuantSupportAsymmetric`
      `TPU_QuantSameInputResultScale` : input and result share same threshold
                                        or scale (bypassed).
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$sqr_lut,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$power_lut,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$scale,
    DefaultValuedAttr<I32Attr, "5">:$local_size,
    DefaultValuedAttr<F32Attr, "1.0">:$alpha,
    DefaultValuedAttr<F32Attr, "0.75">:$beta,
    DefaultValuedAttr<F32Attr, "1.0">:$k,
    OptionalAttr<F32ArrayAttr>:$threshold_parts,
    I32Attr:$sum_rshift,
    I32Attr:$lrn_rshift,
    I32Attr:$quant_data0,
    I32Attr:$quant_data1,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def TPU_LstmOp: TPU_Op<"lstm",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "LSTM operator for onnx";

  let description = [{
    Performs scale on input.

    Inputs:
      `input`           : required, the input activation tensor.
      `recurrence`       : required, recurrence weight tesnor for input, output, forget, and cell gates.
      `bias`            : required, bias vectors for input, output, forget, and cell gates.
      `initial_h`       : required, initial value of the hidden.
      `initial_c`       : required, initial value of the cell.
      `cont`            : optional, for caffe lstm
      `final_h`         : optional, output final cell h
      `final_c`         : optional, output final cell c

    Attributes:
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor with hidden and cell.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    AnyTensor:$recurrence,
    TPU_TensorOfOrNone<[F32, BF16, I16, I8]>:$bias,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$initial_h,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$initial_c,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$cont,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$sigmoid_table,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$sigmoid_slope_table,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$tanh_table,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$tanh_slope_table,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name,
    DefaultValuedAttr<BoolAttr, "false">:$bidirectional,
    DefaultValuedAttr<BoolAttr, "false">:$final_h,
    DefaultValuedAttr<BoolAttr, "false">:$final_c
  );

  let results = (outs AnyTensor:$output);
}

def TPU_MatMulOp : TPU_Op<"matmul",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "Matrix multiply operator";

  let description = [{
    Inputs:
      `inputs`          : required, input tensors.

    Attributes:
      `do_relu`         : required, if need to preform relu on result.
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins Variadic<AnyTensor>:$inputs,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_scale,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_zeropoint,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_rshift,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_multiplier,
    DefaultValuedAttr<BoolAttr, "false">:$left_transpose,
    DefaultValuedAttr<BoolAttr, "false">:$right_transpose,
    DefaultValuedAttr<BoolAttr, "false">:$output_transpose,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);

  let hasCanonicalizer = 1;
}

def TPU_NormalizeOp: TPU_Op<"normalize",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>]> {
  let summary = "Normalize operator";

  let description = [{
    Normalizes an array across batch and spatial dimensions.


    Inputs:
      `input`           : required, the input activation tensor.
      `scale`           : required, the scale weight tensor. even channel_shared is true, extend to tensor.

    Attributes:
      `across_spatial`  : required, normalize cross channel or not.
      `channel_shared`  : required, scale cross channel or not.
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface` : support common TPU Op interface
      No `TpuOpQuantInterface`  : convert to ScaleOp befor quant
      NO `TpuOpLowerInterface`  : convert to ScaleOp befor quant
  }];

  let arguments = (
    ins AnyTensor:$input,
    AnyTensor:$scale,
    DefaultValuedAttr<BoolAttr, "true">:$across_spatial,
    DefaultValuedAttr<BoolAttr, "true">:$channel_shared,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);

  let hasCanonicalizer = 1;
}

def TPU_MishOp : TPU_LutOp<"mish", "yolov4 mish op"> {

  let description = [{
    elements of the bottom do activation, reference bottom.
    Performs activation operation on input.
    In Int8, we use lookup table find output

    Inputs:
      `inputs`          : required, variadic input tensors.
      `table`           : optional, the quantize lookup table,which is reused by INT8 and BF16.
      `table_mantissa`  : optional, the quantize lookup table,only used by BF16 .

    Attributes:
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.
      `max_range`       : required, we cut to max_range and outlier saturate it
      `min_range`       : required, we cut to min_range and outlier saturate it


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantPerTensorOnly`
      `TPU_QuantSupportAsymmetric`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins Variadic<AnyTensor>:$inputs,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$table,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$table_mantissa,
    DefaultValuedAttr<F32Attr, "8">:$max_range,
    DefaultValuedAttr<F32Attr, "-8">:$min_range,
    StrAttr:$name,
    TPU_QuantParamAttr:$quant
  );

  let results = (outs AnyTensor:$output);
}

def TPU_PermuteOp: TPU_Op<"permute",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {

  let summary = "Permute operator";

  let description = [{
      Perform permute on input.

    Inputs:
      `input`               : required, the input activation tensor.

    Attributes:
      `order`    : required, The new order of the axes of data.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    I32ArrayAttr:$order,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);

  let hasCanonicalizer = 1;
}

def TPU_PixelShuffleOp: TPU_Op<"pixelshuffle",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {

  let summary = "pixelshuffle operator";

  let description = [{
      Perform pixelshuffle on input.

    Inputs:
      `input`               : required, the input activation tensor.

    Attributes:
      `upscale_factor`: optional, upscale factor.
      `mode`: optional, string, Use CRD for column-row-depth order, DCR for depth-column-row order re-arrangement.



    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    I32Attr:$upscale_factor,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name,
    DefaultValuedAttr<StrAttr, "CRD">:$mode
  );

  let results = (outs AnyTensor:$output);
}

def TPU_ClipOp: TPU_Op<"clip",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {

  let summary = "clip operator";

  let description = [{
      Perform clip on input.
      clip op comes from onnx, please refer [clip section](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Clip) for more details.

    Inputs:
      `input`               : required, Input tensor whose elements to be clipped

    Attributes:
      `max`: optional, Minimum value, under which element is replaced by min. It must be a scalar(tensor of empty shape).
      `min`: optional, Maximum value, above which element is replaced by max. It must be a scalar(tensor of empty shape).


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_scale,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_zeropoint,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_rshift,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_multiplier,
    DefaultValuedAttr<F32Attr, "-999">:$min,
    DefaultValuedAttr<F32Attr, "999">:$max,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def TPU_PoolAvg2DOp : TPU_Op<"pool_avg_2d",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "average pool_2d operator";

  let description = [{
    Performs average pooling operation on input.

    Inputs:
      `input`           : required, the input activation tensor.
      `quant_scale`     : optional, the quant scale tensor. For average pool
                          only, one fp32 value.
      `quant_zeropoint` : optional, the quant zero_point tensor, should be one
                          int8 value for the output tensor.
      `quant_rshift`    : optional, the rshift. For average pool only, should
                          be a int8 value.
      `quant_multiplier`: optional, the multiplier, one int8 value.

    Attributes:
      `param`           : required, a PoolParam struct attributes, carrying
                          filter size, stride, and padding.
      `do_relu`         : required, do_relu attributes.
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantPerTensorOnly`
      `TPU_QuantSupportAsymmetric`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_scale,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_zeropoint,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_rshift,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_multiplier,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    TPU_PoolParamAttr:$param,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def TPU_PoolMax2DOp : TPU_Op<"pool_max_2d",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "max pool_2d operator";

  let description = [{
    Performs max pooling operation on input.

    Inputs:
      `input`           : required, the input activation tensor.

    Attributes:
      `param`           : required, a PoolParam struct attributes, carrying
                          filter size, stride, and padding.
      `do_relu`         : required, do_relu attributes.
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantPerTensorOnly`
      `TPU_QuantSupportAsymmetric`
      `TPU_QuantSameInputResultScale` : input and result share same threshold
                                        or scale (bypassed).
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    TPU_PoolParamAttr:$param,
    TPU_QuantParamAttr:$quant,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def TPU_PoolMax3DOp : TPU_Op<"pool_max_3d",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "max pool_3d operator";

  let description = [{
    Performs max pooling operation on input.

    Inputs:
      `input`           : required, the input activation tensor.

    Attributes:
      `param`           : required, a PoolParam struct attributes, carrying
                          filter size, stride, and padding.
      `quant`           : required, a QuantParam struct attributes.
      `do_relu`         : required, do_relu attributes.
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantPerTensorOnly`
      `TPU_QuantSupportAsymmetric`
      `TPU_QuantSameInputResultScale` : input and result share same threshold
                                        or scale (bypassed).
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    TPU_Pool3dParamAttr:$param,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def TPU_CustomOp : TPU_Op<"custom_op",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>]> {
  let summary = "generic customer cpu operator";

  let description = [{
    Custom Cpu Op.

    Inputs:
      `input`           : required, the input activation tensor.

    Attributes:
      `param`           : required, a PoolParam struct attributes, carrying
                          filter size, stride, padding, and do_relu.
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.
      `do_quant`         : required, quantize to int8/bf16 or not.
      `threshold_overwrite` : required, overwrite threshold backward/forward or not.


    Result:
      `output`          : result tensor.
  }];

  let arguments = (
    ins Variadic<AnyTensor>:$inputs,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name,
    StrAttr:$operation_name,
    DictionaryAttr:$param,
    DefaultValuedAttr<BoolAttr, "false">:$tpu,
    DefaultValuedAttr<BoolAttr, "false">:$do_quant,
    DefaultValuedAttr<StrAttr, "none">:$threshold_overwrite,
    OptionalAttr<I64Attr>:$gaddr
  );

  let results = (outs AnyTensor:$output);
}

def TPU_PReluOp : TPU_Op<"prelu",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "prelu operator";

  let description = [{
    Performs prelu operation on input.

    Inputs:
      `input`               : required, variadic input tensors.
      `filter`              : required, the filter weight tensor.
      `quant_pos_scale`     : optional, the quant scale tensor, for positive
                              values, one fp32 value.
      `quant_pos_zeropoint` : optional, the quant zero_point tensor, for
                              positive values, one int8 value.
      `quant_neg_scale`     : optional, the quant scale tensor, for negative
                              values, one fp32 value.
      `quant_neg_zeropoint` : optional, the quant zero_point tensor, for
                              negative values, one int8 value.
      `quant_pos_rshift`    : optional, rshift for positive values, one int32
                              value.
      `quant_pos_multiplier`: optional, multiplier for positive values, one
                              int8 value.
      `quant_neg_rshift`    : optional, rshift for negative values, one int32
                              value.
      `quant_neg_multiplier`: optional, multiplier for negative values, one
                              int8 value.


    Attributes:
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantPerTensorOnly`
      `TPU_QuantSupportAsymmetric`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    AnyTensor:$filter,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_pos_scale,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_pos_zeropoint,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_neg_scale,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_neg_zeropoint,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_pos_rshift,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_pos_multiplier,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_neg_rshift,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_neg_multiplier,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def TPU_PriorBoxOp: TPU_Op<"priorbox",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>]> {

  let summary = "Priorbox operator";

  let description = [{
      Intended for use with MultiBox detection method to generate prior.

    Inputs:
      `input`              : required, the input activation tensor.
      `min_size`           : required.
      `max_size`           : required.
      `aspect_ratios`      : required.
      `clip`               : required.
      `variance`           : required.
      `step_h`             : required.
      `step_w`             : required.
      `img_h`              : required.
      `img_w`              : required.
      `offset`             : required.
      `num_priors`         : required.

    Attributes:
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface` : support common TPU Op interface
  }];

  let arguments = (
    ins Variadic<AnyTensor>:$input,
    F32ArrayAttr:$min_size,
    F32ArrayAttr:$max_size,
    F32ArrayAttr:$aspect_ratios,
    F32ArrayAttr:$variance,
    DefaultValuedAttr<BoolAttr, "true">:$clip,
    F32Attr:$step_h,
    F32Attr:$step_w,
    I32Attr:$img_h,
    I32Attr:$img_w,
    DefaultValuedAttr<F32Attr, "0.5">:$offset,
    I32Attr:$num_priors,
    DefaultValuedAttr<BoolAttr, "true">:$use_default_aspect_ratio,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);

  let hasCanonicalizer = 1;
}

def TPU_ProposalOp: TPU_Op<"proposal",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>]> {
  let summary = "Proposal operator";

  let description = [{
    Inputs:
      `inputs`                : required, the input activation tensor.

    Attributes:
      `name`                  : required, name for calibration, comparing, or debug.

      `net_input_h`           : required, net input height
      `net_input_w`           : required, net input width
      `feat_stride`           : required, anchor box stride size
      `anchor_base_size`      : required, anchor box base size
      `rpn_obj_threshold`     : required, obj threshold
      `rpn_nms_threshold`     : required, nms threshold for generate proposal boxes
      `rpn_nms_post_top_n`    : required, keep num boxes after nms

    Result:
      `output`                : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface`  : support common TPU Op interface
  }];

  let arguments = (
    ins Variadic<AnyTensor>:$inputs,
    I32Attr:$net_input_h,
    I32Attr:$net_input_w,
    I32Attr:$feat_stride,
    I32Attr:$anchor_base_size,
    F32Attr:$rpn_obj_threshold,
    F32Attr:$rpn_nms_threshold,
    I32Attr:$rpn_nms_post_top_n,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def TPU_ReflectionPadOp : TPU_Op<"reflection_pad",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "Reflectionpad operator";

  let description = [{
    Inputs:
      `input`           : required, the input activation tensor.

    Attributes:
      `pads`             : required, pad left and right.
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantSupportAsymmetric`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpInterpInterface` : support interperter
      `TpuOpQuantInterface`  : support quantization
  }];

  let arguments = (
    ins AnyTensor:$input,
    TPU_TensorOfOrNone<[F32, BF16, I16, I8]>:$left_select,
    TPU_TensorOfOrNone<[F32, BF16, I16, I8]>:$right_select,
    I32ArrayAttr:$pads,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def TPU_ReluOp: TPU_Op<"relu",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "Relu/Relu6 operator";

  let description = [{
    Element-wise Relu operator
      y = max(0, x)
    Relu6 (maximum = 6) or ReluM, if maximum is present
      y = min(max(x, 0), maximum).

    Inputs:
      `input`           : required, the input activation tensor.

    Attributes:
      `maximum`         : optional, the maximum value for ReluM when present.
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantBypass`
      `TPU_QuantPerTensorOnly`
      `TPU_QuantSupportAsymmetric`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    OptionalAttr<F32Attr>:$maximum,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);

  let hasCanonicalizer = 1;
}

def TPU_ReorgOp : TPU_Op<"reorg",
    [NoSideEffect,
     DeclareOpInterfaceMethods<TpuOpCommonInterface>,
     DeclareOpInterfaceMethods<TpuOpQuantInterface>,
     DeclareOpInterfaceMethods<TpuOpLowerInterface>]>  {

  let summary = "Reorg operator";

  let description = [{
    reorg on input.

    Inputs:
      `input`           : required, the input activation tensor.

    Attributes:
      `stride`          : required, channel / height / width stride.
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantPerTensorOnly`
      `TPU_QuantSupportAsymmetric`
      `TPU_QuantSameInputResultScale` : input and result share same threshold
                                        or scale (bypassed).
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    I32Attr:$stride,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def TPU_RetinaFaceDetectionOp : TPU_Op<"retinaface_detection",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>]> {
  let summary = "RetinaFaceDetection operator";

  let description = [{
    Perform retinaface detection on feature map

    Inputs:
      `inputs`                  : required, input tensors
      `nms_threshold`           : required, nms threshold
      `confidence_threshold`    : required, classification confidence threshold
      `keep_topk`               : required, after nms, keep bbox num

    Attributes:
      `name`            : required, name for calibration, comparing, or debug.
      `quant`           : required, a QuantParam struct attributes.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
  }];

  let arguments = (
    ins Variadic<AnyTensor>:$input,
    F32Attr:$nms_threshold,
    F32Attr:$confidence_threshold,
    I32Attr:$keep_topk,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def TPU_ReverseOp: TPU_Op<"reverse",
    [NoSideEffect,
     DeclareOpInterfaceMethods<TpuOpCommonInterface>,
     DeclareOpInterfaceMethods<TpuOpQuantInterface>,
     DeclareOpInterfaceMethods<TpuOpLowerInterface>]>  {

  let summary = "reverse operator";

  let description = [{
    Reverse on input.

    Inputs:
      `input`           : required, the input activation tensor.

    Attributes:
      `axis`           : required, witch axis to reverse
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantPerTensorOnly`
      `TPU_QuantSupportAsymmetric`
      `TPU_QuantSameInputResultScale` : input and result share same threshold
                                        or scale (bypassed).
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    I32Attr:$axis,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def TPU_ROIPoolingOp : TPU_Op<"roi_pooling",
    [NoSideEffect,
     DeclareOpInterfaceMethods<TpuOpCommonInterface>,
     DeclareOpInterfaceMethods<TpuOpQuantInterface>]>  {

  let summary = "ROIPooling operator";

  let description = [{
    Max pooling on ROI.

    Inputs:
      `inputs`          : required, the input activation tensor.

    Attributes:
      `pooled_h`        : required, pooled output height.
      `pooled_w`        : required, pooled output width
      `spatial_scale`   : required, spatial_scale
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
  }];

  let arguments = (
    ins Variadic<AnyTensor>:$inputs,
    I32Attr:$pooled_h,
    I32Attr:$pooled_w,
    F32Attr:$spatial_scale,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def TPU_ScaleOp: TPU_Op<"scale",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>]> {
  let summary = "Scale operator";

  let description = [{
    Performs scale on input.

    Inputs:
      `input`           : required, the input activation tensor.
      `filter`          : required, the filter weight tensor.
      `bias`            : optional, the bias weight tensor.

    Attributes:
      `do_relu`         : required, if need to preform relu on result.
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface` : support common TPU Op interface
      No `TpuOpQuantInterface`  : convert to DW conv befor quant
      NO `TpuOpLowerInterface`  : convert to DW conv befor quant
  }];

  let arguments = (
    ins AnyTensor:$input,
    AnyTensor:$scale,
    TPU_TensorOfOrNone<[F32, BF16, I16, I8]>:$bias,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);

  let hasCanonicalizer = 1;
}

def TPU_ScaleLutOp: TPU_Op<"scale_lut",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "Scale by lut operator";

  let description = [{
    Performs scale on input, y = input * scale + bias.

    Inputs:
      `input`           : required, the input activation tensor.
      `table`           : required, the quantize lookup table, which is reused by INT8.

    Attributes:
      `channel_order`   : optional, channel swap order, bgr as default
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$table,
    F32ArrayAttr:$scale,
    F32ArrayAttr:$bias,
    StrAttr:$name,
    TPU_QuantParamAttr:$quant
  );

  let results = (outs AnyTensor:$output);
}

def TPU_SoftPlusOp : TPU_LutOp<"softplus", "SoftPlus"> {

  let description = [{
    softplus = lambda x, scale, bias: scale * np.log1p(np.exp(x)) + bias
  }];

  let arguments = (
    ins Variadic<AnyTensor>:$inputs,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$table,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$table_mantissa,
    DefaultValuedAttr<F32Attr, "8">:$max_range,
    DefaultValuedAttr<F32Attr, "-8">:$min_range,
    StrAttr:$name,
    DefaultValuedAttr<F32Attr, "1">:$scale,
    DefaultValuedAttr<F32Attr, "0">:$bias,
    TPU_QuantParamAttr:$quant
  );

  let results = (outs AnyTensor:$output);
}

def TPU_ShuffleChannelOp: TPU_Op<"shuffle_channel",
    [NoSideEffect,
     DeclareOpInterfaceMethods<TpuOpCommonInterface>,
     DeclareOpInterfaceMethods<TpuOpQuantInterface>,
     DeclareOpInterfaceMethods<TpuOpLowerInterface>]>  {

  let summary = "ShuffleChannel operator";

  let description = [{
    Shuffle Channel on input.

    Inputs:
      `input`           : required, the input activation tensor.

    Attributes:
      `group`           : required, channel group.
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantPerTensorOnly`
      `TPU_QuantSupportAsymmetric`
      `TPU_QuantSameInputResultScale` : input and result share same threshold
                                        or scale (bypassed).
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    I32Attr:$group,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def TPU_SigmoidOp : TPU_Op<"sigmoid",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {

  let summary = " Exp operator,  scale * sigmoid + bias";
  let arguments = (
    ins Variadic<AnyTensor>:$inputs,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$table,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$table_mantissa,
    StrAttr:$name,
    DefaultValuedAttr<F32Attr, "1">:$scale,
    DefaultValuedAttr<F32Attr, "0">:$bias,
    DefaultValuedAttr<F32Attr, "8">:$max_range,
    DefaultValuedAttr<F32Attr, "-8">:$min_range,
    DefaultValuedAttr<I32Attr, "-1">:$layer_id,
    TPU_QuantParamAttr:$quant
  );

  let results = (outs AnyTensor:$output);
}

def TPU_SoftmaxOp: TPU_Op<"softmax",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "Softmax operator";

  let description = [{
    Perform softmax on input.

    Inputs:
      `input`           : required, the input activation tensor.
      `exponential_table`           : optional, the exponential table.
      `exponential_slope_table`           : optional, the exponential slope table.
      `reciprocal_table`           : optional, the reciprocal table.
      `reciprocal_mantissa_table`  : optional, the reciprocal mantissa table.

    Attributes:
      `axis`            : optional, the axis for softmax to perform.
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : CPU Op still needs threshold for quant/dequant
                               of the next/prev Ops.
      No `TpuOpLowerInterface`  : CPU Op no lowering to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$exponential_table,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$exponential_slope_table,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$reciprocal_table,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$reciprocal_mantissa_table,
    DefaultValuedAttr<I32Attr, "1">:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$do_log,
    StrAttr:$name,
    TPU_QuantParamAttr:$quant
  );

  let results = (outs AnyTensor:$output);
}

def TPU_SoftmaxCpuOp: TPU_Op<"softmax_cpu",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>]> {
  let summary = "SoftmaxCpu operator";

  let description = [{
    Perform softmax on input.

    Inputs:
      `input`           : required, the input activation tensor.
      `exponential_table`           : optional, the exponential table.
      `exponential_slope_table`           : optional, the exponential slope table.
      `reciprocal_table`           : optional, the reciprocal table.
      `reciprocal_mantissa_table`           : optional, the reciprocal mantissa table.

    Attributes:
      `axis`            : optional, the axis for softmax to perform.
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : CPU Op still needs threshold for quant/dequant
                               of the next/prev Ops.
      No `TpuOpLowerInterface`  : CPU Op no lowering to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    DefaultValuedAttr<I32Attr, "1">:$axis,
    StrAttr:$name,
    TPU_QuantParamAttr:$quant
  );

  let results = (outs AnyTensor:$output);
}

def TPU_QuadraticSumOp: TPU_Op<"quadratic_sum",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "Sum of Squares operator";

  let description = [{
    Perform sum of squares on input.

    Inputs:
      `input`           : required, the input activation tensor.

    Attributes:
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : CPU Op still needs threshold for quant/dequant
                               of the next/prev Ops.
      No `TpuOpLowerInterface`  : CPU Op no lowering to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    StrAttr:$name,
    TPU_QuantParamAttr:$quant,
    OptionalAttr<BoolAttr>:$high_precision,
    DefaultValuedAttr<I32Attr, "1">:$axis
  );

  let results = (outs AnyTensor:$output);
}

def TPU_SwapChannelOp: TPU_Op<"swap_channel",
    [NoSideEffect,
     DeclareOpInterfaceMethods<TpuOpCommonInterface>,
     DeclareOpInterfaceMethods<TpuOpQuantInterface>,
     DeclareOpInterfaceMethods<TpuOpLowerInterface>]>  {

  let summary = "swap channel operator, normally RGB <=> BGR";

  let description = [{
    Swap Channel on input.

    Inputs:
      `input`           : required, the input activation tensor.

    Attributes:
      `channel_order`   : required, channel swap order
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantPerTensorOnly`
      `TPU_QuantSupportAsymmetric`
      `TPU_QuantSameInputResultScale` : input and result share same threshold
                                        or scale (bypassed).
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    I32ArrayAttr:$channel_order,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def TPU_TileOp: TPU_Op<"tile",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "Tile operator";

  let description = [{
    numpy tile operation,
    Construct an array by repeating A the number of times given by reps.
    e.g:
    >>>b = np.array([[1, 2], [3, 4]])
    >>>np.tile(b, [2])
    array([[1, 2, 1, 2],
       [3, 4, 3, 4]])
    >>>np.tile(b, [2, 1])
    array([[1, 2],
       [3, 4],
       [1, 2],
       [3, 4]])

    Inputs:
      `input`           : required, the input activation tensor.
      `axis`            : required, axis to be repeated
      `tiles`           : required, repeat times in axis dim

    Attributes:
      `quant`           : required, a QuantParam struct attributes, to carrying
                          threshold, although the quantization is bypassed.
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantPerTensorOnly`
      `TPU_QuantSupportAsymmetric`
      `TPU_QuantSameInputResultScale` : input and result share same threshold
                                        or scale (bypassed).
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    I32Attr:$axis,
    I32Attr:$tiles,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def TPU_UpsampleOp: TPU_Op<"upsample",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "Upsample operator";

  let description = [{
    Perform upample on input.

    Inputs:
      `input`           : required, the input activation tensor.

    Attributes:
      `scale_h`         : required, scale align h for upsampling.
      `scale_w`         : required, scale align w for upsampling.
      `quant`           : required, a QuantParam struct attributes, to carrying
                          threshold, although the quantization is bypassed.
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantPerTensorOnly`
      `TPU_QuantSupportAsymmetric`
      `TPU_QuantSameInputResultScale` : input and result share same threshold
                                        or scale (bypassed).
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    TPU_TensorOfOrNone<[F32, BF16, I16, I8]>:$mask,
    I32Attr:$scale_h,
    I32Attr:$scale_w,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def TPU_YoloDetectionOp : TPU_Op<"yolo_detection",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>]> {
  let summary = "YoloDetection operator";

  let description = [{
    Perform yolo detection on feature map

    Inputs:
      `inputs`                  : required, input tensors

    Attributes:
      `name`                    : required, name for calibration, comparing, or debug.

      `net_input_h`             : required, net input h
      `net_input_w`             : required, net input w
      `nms_threshold`           : required, nms threshold
      `obj_threshold`           : required, confidence threshold
      `keep_topk`               : required, after nms, keep bbox num

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface` : support common TPU Op interface
  }];

  let arguments = (
    ins Variadic<AnyTensor>:$input,
    I32Attr:$net_input_h,
    I32Attr:$net_input_w,
    F32Attr:$nms_threshold,
    F32Attr:$obj_threshold,
    I32Attr:$keep_topk,
    DefaultValuedAttr<BoolAttr, "false">:$spp_net,
    DefaultValuedAttr<BoolAttr, "false">:$tiny,
    DefaultValuedAttr<BoolAttr, "false">:$yolo_v4,
    DefaultValuedAttr<I32Attr, "80">:$class_num,
    DefaultValuedAttr<StrAttr, "">:$anchors,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def TPU_PadOp: TPU_Op<"pad",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "Pad operator";

  let description = [{
    Perform padding on input.

    Inputs:
      `input`           : required, the input activation tensor.

    Attributes:
      `name`            : required, name for calibration, comparing, or debug.
      `pads`            : tensor containing the number of start and end
                          pad values for axis
      `const_val`       : A scalar value to be used if the mode
                          chosen is `constant` (by default it is 0).
      `mode`            : optional, Supported modes: `constant`(default), `reflect`, `edge`


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name,
    I32ArrayAttr:$pads,
    DefaultValuedAttr<F32Attr, "0">:$const_val,
    DefaultValuedAttr<StrAttr, "constant">:$mode
  );

  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 1;
}

def TPU_PoolMaskOp: TPU_Op<"pool_mask",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "pooling mask operator";

  let description = [{
    Perform padding on input.

    Inputs:
      `input`           : required, the input activation tensor.

    Attributes:
      `scale`           : required, scale for upsampling.
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    I32Attr:$scale,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

class TPU_ReduceOp<string mnemonic> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = mnemonic # " operator";

  let description = [{
    Perform Reduce on input.

    Inputs:
      `input`           : required, the input activation tensor.

    Attributes:
      `name`            : required, name for calibration, comparing, or debug.
      `axes`            : along which axis to do reduce

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_scale,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_zeropoint,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_rshift,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_multiplier,
    I32ArrayAttr:$axes,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def TPU_ReduceMaxOp  : TPU_ReduceOp<"reduce_max">;
def TPU_ReduceMinOp  : TPU_ReduceOp<"reduce_min">;
def TPU_ReduceSumOp  : TPU_ReduceOp<"reduce_sum">;
def TPU_ReduceMeanOp : TPU_ReduceOp<"reduce_mean">;

def TPU_ReduceL2Op: TPU_Op<"reduce_l2",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "Reduce L2 operator";

  let description = [{
  }];

  let arguments = (
    ins AnyTensor:$input,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$table,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$mantissa_table,
    I32ArrayAttr:$axes,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def TPU_EmbeddingOp: TPU_Op<"embedding",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "Embedding operator";

  let description = [{
  }];

  let arguments = (
    ins AnyTensor:$input,
    AnyTensor:$table,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_scale,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_zeropoint,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def TPU_StdOp: TPU_Op<"std",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "std operator";

  let description = [{
    std operator

    Inputs:
      `input`           : required, the input activation tensor.
      `table`           : required, the quantize lookup table,which is reused by INT8 and BF16.
      `table_mantissa`  : required, the quantize lookup table,only used by BF16 .

    Attributes:
      `start_dim`       : required, start dim to do std
      `unbiased`        : required, whether to use Bessel’s correction
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.


    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantPerTensorOnly`
      `TPU_QuantSupportAsymmetric`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$table,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$mantissa_table,
    I32Attr:$start_dim,
    BoolAttr:$unbiased,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name
  );

  let results = (outs AnyTensor:$output);
}

def TPU_ZeroMaskOp: TPU_Op<"zero_mask",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "zero_mask";

  let description = [{
    for example:
    2 3 4 5 0 1 0
    if positive == ture, => 1 1 1 1 0 1 0
    if positive == false, => 0 0 0 0 1 0 1
  }];

  let arguments = (
    ins AnyTensor: $input,
    BoolAttr: $positive,
    TPU_QuantParamAttr: $quant,
    StrAttr: $name
  );

  let results = (outs AnyTensor:$output);
}

def TPU_WeightFileOp: TPU_Op<"weight_file",
      [NoSideEffect]> {
  let summary = "Operator to save weight data";

  let arguments = (
    ins StrAttr:$filename
  );

  let extraClassDeclaration = [{
    TensorFile* get(void) {
      auto *context = getContext();
      auto dialect = context->getLoadedDialect("tpu");
      auto tpuDialect = reinterpret_cast<TPUDialect *>(dialect);
      assert(tpuDialect);
      if (!tpuDialect->getPriv()) {
        auto weightFile = openTensorFile(filename());
        assert(weightFile);
        tpuDialect->setPriv((void *)weightFile.release());
      }
      return (TensorFile *)tpuDialect->getPriv();
    }
  }];

  let printer = [{
    auto *context = getContext();
    auto dialect = context->getLoadedDialect("tpu");
    auto tpuDialect = reinterpret_cast<tpu::TPUDialect *>(dialect);
    assert(tpuDialect);
    TensorFile *weightFile = (TensorFile *)tpuDialect->getPriv();
    if (weightFile) {
      std::string newName;
      int updated = weightFile->keep(true, &newName);
      if (updated) {
        (*this)->setAttr("filename", Builder(context).getStringAttr(newName));
      }
    }
    OpState::print(getOperation(), p);
  }];

  let results = (outs AnyMemRef:$output);
}

include "tpuc/Dialect/TPU/TPUSupportOps.td"
include "tpuc/Dialect/TPU/TPUTGOps.td"
include "tpuc/Dialect/TPU/TPUTLOps_lg.td"

#endif // TPU_OPS
