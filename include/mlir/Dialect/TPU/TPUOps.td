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
include "mlir/Dialect/TPU/TPUInterface.td"
include "mlir/Dialect/TPU/TPUAttribute.td"

//===----------------------------------------------------------------------===//
// TPU op base class.
//===----------------------------------------------------------------------===//

class TPU_ConvOp<string mnemonic, string opSummary> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpInterpInterface>,
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
                          stride, padding, dilation, group, is_dw, with_bias,
                          and do_relu.
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.
      `layer_id`        : optional, id for profiling.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantSupportPerChannel`
      `TPU_QuantSupportAsymmetric`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpInterpInterface` : support interperter
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
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$output);
}

class TPU_EltwiseOp<string mnemonic, string opSummary> : TPU_Op<mnemonic,
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpInterpInterface>,
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
      `layer_id`        : optional, id for profiling.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantPerTensorOnly`
      `TPU_QuantSupportAsymmetric`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpInterpInterface` : support interperter
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
    TPU_QuantParamAttr:$quant,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$output);

  let extraClassDeclaration = [{
    unsigned getNumInputs() { return getNumOperands() - 4; }
  }];
}

//===----------------------------------------------------------------------===//
// TPU op definitions. (in alphabetical order)
//===----------------------------------------------------------------------===//

def TPU_BatchNormOp: TPU_Op<"batch_norm",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpInterpInterface>]> {
  let summary = "BatchNorm operator";

  let description = [{
    Normalizes an array across batch and spatial dimensions.


    Inputs:
      `input`           : required, the input activation tensor.
      `mean`            : required, the mean weight tensor, per-channel value.
      `variance`        : required, the variance weight tensor, per-channel
                          value.
      `scale`           : required, the scale weight tensor, one value.

    Attributes:
      `variance_epsilon`: required, eps for variance.
      `name`            : required, name for calibration, comparing, or debug.
      `layer_id`        : optional, id for profiling.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpInterpInterface` : support interperter
      No `TpuOpQuantInterface`  : convert to ScaleOp befor quant
      NO `TpuOpLowerInterface`  : convert to ScaleOp befor quant
  }];

  let arguments = (
    ins AnyTensor:$input,
    AnyTensor:$mean,
    AnyTensor:$variance,
    AnyTensor:$scale,
    DefaultValuedAttr<F32Attr, "1.0e-5">:$variance_epsilon,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$output);

  let hasCanonicalizer = 1;
}

def TPU_BroadcastMulOp: TPU_Op<"broadcast_mul",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpInterpInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "Broadcast Multiply operator";

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
      `axis`            : required, the axis the multiply is applying
      `do_relu`         : required, if need to preform relu on result.
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.
      `layer_id`        : optional, id for profiling.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantSupportPerChannel`
      `TPU_QuantSupportAsymmetric`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpInterpInterface` : support interperter
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    AnyTensor:$multiplier,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_scale,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_zeropoint,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_rshift,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_multiplier,
    DefaultValuedAttr<I32Attr, "1">:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$output);
}

def TPU_ConcatOp: TPU_Op<"concat",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpInterpInterface>,
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
      `layer_id`        : optional, id for profiling.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantPerTensorOnly`
      `TPU_QuantSupportAsymmetric`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpInterpInterface` : support interperter
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins Variadic<AnyTensor>:$inputs,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_scale,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_zeropoint,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_rshift,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_multiplier,
    DefaultValuedAttr<I32Attr, "1">:$axis,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$output);

  let extraClassDeclaration = [{
    unsigned getNumInputs() { return getNumOperands() - 4; }
  }];

  let verifier = [{
    auto firstType = getOperand(0)->getType().cast<RankedTensorType>();

    auto firstShape = firstType.getShape();
    int numOperands = getNumOperands();
    for (int i = 1; i < numOperands - 4; i++) {
      auto secondType = getOperand(i)->getType().cast<RankedTensorType>();

      if (firstType.getRank() != secondType.getRank()) {
        return emitOpError() << "operands (0) and" << "(" << i << ")" << "do not match rank.";
      }

      auto secondShape = secondType.getShape();
      for (int d = 0; d < firstType.getRank(); ++d) {
        if (firstShape[d] != secondShape[d] && d != axis()) {
          return emitOpError() << "operands (0) and (" << "i" << "non-concat dimensions do not match ";
        }
      }
    }
    return success();
  }];
}

def TPU_Conv2DOp : TPU_ConvOp<"conv_2d", "Convolution">;

def TPU_CropOp: TPU_Op<"crop",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpInterpInterface>,
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
      `crop_shape`      : required, crop shape.
      `crop_offset`     : required, crop offset.
      `layer_id`        : optional, id for profiling.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantPerTensorOnly`
      `TPU_QuantSupportAsymmetric`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpInterpInterface` : support interperter
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    OptionalAttr<I32ArrayAttr>:$crop_shape,
    OptionalAttr<I32ArrayAttr>:$crop_offset,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );
  let results = (outs AnyTensor:$output);
}

def TPU_DeConv2DOp : TPU_ConvOp<"deconv_2d", "Deconvolution">;

def TPU_EltwiseAddOp : TPU_EltwiseOp<"eltwise_add", "Add">;
def TPU_EltwiseMulOp : TPU_EltwiseOp<"eltwise_mul", "Mul">;
def TPU_EltwiseMaxOp : TPU_EltwiseOp<"eltwise_max", "Max">;

def TPU_FullyConnectedOp : TPU_Op<"fully_connected",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpInterpInterface>,
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
      `layer_id`        : optional, id for profiling.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantSupportPerChannel`
      `TPU_QuantSupportAsymmetric`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpInterpInterface` : support interperter
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
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$output);
}

def TPU_LeakyReluOp: TPU_Op<"leaky_relu",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpInterpInterface>,
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
      `layer_id`        : optional, id for profiling.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantPerTensorOnly`
      `TPU_QuantSupportAsymmetric`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpInterpInterface` : support interperter
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
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$output);
}

def TPU_PermuteOp: TPU_Op<"permute",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpInterpInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {

  let summary = "Permute operator";

  let description = [{
      Perform permute on input.

    Inputs:
      `input`               : required, the input activation tensor.

    Attributes:
      `order0`  : required, The new order0 of the axes of data.
      `order1`  : required, The new order1 of the axes of data.
      `order2`  : required, The new order2 of the axes of data.
      `order3`  : optional, The new order3 of the axes of data.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpInterpInterface` : support interperter
      `TpuOpQuantInterface`  : support quantization    
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    NonNegativeI32Attr:$order0,
    NonNegativeI32Attr:$order1,
    NonNegativeI32Attr:$order2,
    NonNegativeI32Attr:$order3,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$output);
}

def TPU_PoolAvg2DOp : TPU_Op<"pool_avg_2d",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpInterpInterface>,
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
                          filter size, stride, padding, and do_relu.
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.
      `layer_id`        : optional, id for profiling.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantPerTensorOnly`
      `TPU_QuantSupportAsymmetric`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpInterpInterface` : support interperter
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_scale,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_zeropoint,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_rshift,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_multiplier,
    TPU_PoolParamAttr:$param,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$output);
}

def TPU_PoolMax2DOp : TPU_Op<"pool_max_2d",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpInterpInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "max pool_2d operator";

  let description = [{
    Performs max pooling operation on input.

    Inputs:
      `input`           : required, the input activation tensor.

    Attributes:
      `param`           : required, a PoolParam struct attributes, carrying
                          filter size, stride, padding, and do_relu.
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.
      `layer_id`        : optional, id for profiling.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantPerTensorOnly`
      `TPU_QuantSupportAsymmetric`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpInterpInterface` : support interperter
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    TPU_PoolParamAttr:$param,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$output);
}

def TPU_PReluOp : TPU_Op<"prelu",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpInterpInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "prelu operator";

  let description = [{
    Performs prelu operation on input.

    Inputs:
      `inputs`              : required, variadic input tensors.
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
      `layer_id`        : optional, id for profiling.
      

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantPerTensorOnly`
      `TPU_QuantSupportAsymmetric`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpInterpInterface` : support interperter
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins Variadic<AnyTensor>:$inputs,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_pos_scale,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_pos_zeropoint,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_neg_scale,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_neg_zeropoint,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_pos_rshift,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_pos_multiplier,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_neg_rshift,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$quant_neg_multiplier,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$output);
}

def TPU_ReluOp: TPU_Op<"relu",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpInterpInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>]> {
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
      `layer_id`        : optional, id for profiling.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantBypass`
      `TPU_QuantPerTensorOnly`
      `TPU_QuantSupportAsymmetric`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpInterpInterface` : support interperter
      `TpuOpQuantInterface`  : support quantization
      No `TpuOpLowerInterface`  : does NOT support lower to TPU TG Ops, always
                                  fuse in frontend
  }];

  let arguments = (
    ins AnyTensor:$input,
    OptionalAttr<F32Attr>:$maximum,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$output);
}

def TPU_ScaleOp: TPU_Op<"scale",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpInterpInterface>]> {
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
      `layer_id`        : optional, id for profiling.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpInterpInterface` : support interperter
      No `TpuOpQuantInterface`  : convert to DW conv befor quant
      NO `TpuOpLowerInterface`  : convert to DW conv befor quant
  }];

  let arguments = (
    ins AnyTensor:$input,
    AnyTensor:$scale,
    TPU_TensorOfOrNone<[F32, BF16, I16, I8]>:$bias,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$output);

  let hasCanonicalizer = 1;
}

def TPU_ShuffleChannelOp: TPU_Op<"shuffle_channel",
    [NoSideEffect,
     DeclareOpInterfaceMethods<TpuOpCommonInterface>,
      DeclareOpInterfaceMethods<TpuOpInterpInterface>,
      DeclareOpInterfaceMethods<TpuOpQuantInterface>,
      DeclareOpInterfaceMethods<TpuOpLowerInterface> ]>  {

  let summary = "ShuffleChannel operator";

  let description = [{
    Shuffle Channel on input.

    Inputs:
      `input`           : required, the input activation tensor.

    Attributes:
      `group`           : required, channel group.
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.
      `layer_id`        : optional, id for profiling.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantPerTensorOnly`
      `TPU_QuantSupportAsymmetric`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpInterpInterface` : support interperter
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    NonNegativeI32Attr:$group,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$output);
}

def TPU_SigmoidOp: TPU_Op<"sigmoid",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpInterpInterface>,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "Sigmoid operator";

  let description = [{
    To sigmoid, elements of the bottom do sigmoid activation, reference bottom.
     Performs sigmoid operation on input.

    Inputs:
      `inputs`          : required, variadic input tensors.
      `table`          : optional, the quantize lookup table .

    Attributes:
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.
      `layer_id`        : optional, id for profiling.
      `has_table`       : optional, if has lookup table.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantPerTensorOnly`
      `TPU_QuantSupportAsymmetric`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpInterpInterface` : support interperter
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins Variadic<AnyTensor>:$inputs,
    TPU_TensorOfOrNone<[F32, BF16, I8]>:$table,
    OptionalAttr<BoolAttr>:$has_table,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    TPU_QuantParamAttr:$quant
  );

  let results = (outs AnyTensor:$output);
}

def TPU_SoftmaxOp: TPU_Op<"softmax",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpInterpInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>]> {
  let summary = "Softmax operator";

  let description = [{
    Perform softmax on input.

    Inputs:
      `input`           : required, the input activation tensor.

    Attributes:
      `axis`            : optional, the axis for softmax to perform.
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.
      `layer_id`        : optional, id for profiling.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpInterpInterface` : support interperter
      `TpuOpQuantInterface`  : CPU Op still needs threshold for quant/dequant
                               of the next/prev Ops.
      No `TpuOpLowerInterface`  : CPU Op no lowering to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    DefaultValuedAttr<I32Attr, "1">:$axis,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$output);
}

def TPU_UpsampleOp: TPU_Op<"upsample",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpInterpInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuOpLowerInterface>]> {
  let summary = "Upsample operator";

  let description = [{
    Perform upample on input.

    Inputs:
      `input`           : required, the input activation tensor.

    Attributes:
      `scale`           : required, scale for upsampling.
      `quant`           : required, a QuantParam struct attributes.
      `name`            : required, name for calibration, comparing, or debug.
      `layer_id`        : optional, id for profiling.

    Result:
      `output`          : result tensor.

    Interfaces or Traits:
      `NoSideEffect`
      `TPU_QuantPerTensorOnly`
      `TPU_QuantSupportAsymmetric`
      `TpuOpCommonInterface` : support common TPU Op interface
      `TpuOpInterpInterface` : support interperter
      `TpuOpQuantInterface`  : support quantization
      `TpuOpLowerInterface`  : support lower to TPU TG Ops
  }];

  let arguments = (
    ins AnyTensor:$input,
    NonNegativeI32Attr:$scale,
    TPU_QuantParamAttr:$quant,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$output);
}














// to be removed
def TPU_DetectionOutputOp: TPU_Op<"detectionoutput",
      [NoSideEffect, DeclareOpInterfaceMethods<TpuInterface>]> {
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
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    OptionalAttr<NonNegativeI64Attr>:$offset,
    OptionalAttr<F32Attr>:$threshold_y,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant
  );

  let results = (outs AnyTensor:$output);
}

def TPU_DivOp: TPU_Op<"div",
      [NoSideEffect, DeclareOpInterfaceMethods<TpuInterface>]> {
  let summary = "Division operator";

  let description = [{
    Do division operation.
  }];

  let arguments = (
    ins AnyTensor:$input,
    DefaultValuedAttr<F32Attr, "1.0">:$numerator,
    Variadic<AnyTensor>:$table,
    OptionalAttr<BoolAttr>:$has_table,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    OptionalAttr<NonNegativeI64Attr>:$offset,
    OptionalAttr<F32Attr>:$threshold_y,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant
  );

  let results = (outs AnyTensor:$output);
}

def TPU_InputOp: TPU_Op<"input",
      [NoSideEffect, DeclareOpInterfaceMethods<TpuInterface>]> {
  let summary = "Input operator";

  let description = [{
    Produces a tensor from function input, primarily for carrying threshold_y.
  }];

  let arguments = (
    ins AnyTensor:$input,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    OptionalAttr<NonNegativeI64Attr>:$offset,
    OptionalAttr<F32Attr>:$threshold_y,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant
  );

  let results = (outs AnyTensor:$output);
}

def TPU_NormalizeOp : TPU_Op<"normalize",
      [NoSideEffect, DeclareOpInterfaceMethods<TpuInterface>]> {
  let summary = "Do Normalization";

  let arguments = (
    ins AnyTensor:$input,
    DefaultValuedAttr<BoolAttr, "true">:$across_spatial,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    OptionalAttr<NonNegativeI64Attr>:$offset,
    OptionalAttr<F32Attr>:$threshold_y,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant
  );

  let results = (outs AnyTensor:$output);
}

def TPU_PowerOp: TPU_Op<"power",
      [NoSideEffect, DeclareOpInterfaceMethods<TpuInterface>]> {
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
    OptionalAttr<BoolAttr>:$has_table,
    OptionalAttr<F32Attr>:$rshift,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    OptionalAttr<NonNegativeI64Attr>:$offset,
    OptionalAttr<F32Attr>:$threshold_y,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant
  );

  let results = (outs AnyTensor:$y);
}


def TPU_PriorBoxOp: TPU_Op<"priorbox",
      [NoSideEffect, DeclareOpInterfaceMethods<TpuInterface>]> {
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
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    OptionalAttr<F32Attr>:$threshold_y,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant
  );

  let results = (outs AnyTensor:$output);
}

def TPU_ReshapeOp: TPU_Op<"reshape",
      [NoSideEffect, DeclareOpInterfaceMethods<TpuInterface>]> {
  let summary = "Reshape operator";

  let description = [{
    Produces a tensor with the same values but different static shape defined
    by the output type.
  }];

  let arguments = (
    ins AnyTensor:$input,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    OptionalAttr<NonNegativeI64Attr>:$offset,
    OptionalAttr<F32Attr>:$threshold_y,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant
  );

  let results = (outs AnyTensor:$output);
}

def TPU_SliceOp: TPU_Op<"slice",
      [NoSideEffect, DeclareOpInterfaceMethods<TpuInterface>]> {
  let summary = "Slice operator";

  let description = [{
    Slices an tensor to multiple output tensors along a given dimension (currently channel only)
    with given slice indices.
  }];

  let arguments = (
    ins AnyTensor:$input,
    OptionalAttr<I32Attr>:$axis,
    OptionalAttr<I32Attr>:$input_offset,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    OptionalAttr<NonNegativeI64Attr>:$offset,
    OptionalAttr<F32Attr>:$threshold_y,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant
  );

  let results = (outs AnyTensor:$output);
}

def TPU_SqrtOp: TPU_Op<"sqrt",
      [NoSideEffect, DeclareOpInterfaceMethods<TpuInterface>]> {
  let summary = "Sqrt operator";

  let description = [{
    Do sqrt operation.
  }];

  let arguments = (
    ins AnyTensor:$input,
    Variadic<AnyTensor>:$table,
    OptionalAttr<BoolAttr>:$has_table,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    OptionalAttr<NonNegativeI64Attr>:$offset,
    OptionalAttr<F32Attr>:$threshold_y,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant
  );

  let results = (outs AnyTensor:$output);
}

def TPU_TanHOp: TPU_Op<"tanh",
      [NoSideEffect, DeclareOpInterfaceMethods<TpuInterface>]> {
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
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    OptionalAttr<NonNegativeI64Attr>:$offset,
    OptionalAttr<F32Attr>:$threshold_y,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant
  );

  let results = (outs AnyTensor:$y);
}

//===----------------------------------------------------------------------===//
// TPU Quantization op definitions.
//===----------------------------------------------------------------------===//
def TPU_QuantizationOp: TPU_Op<"quantization",
      [NoSideEffect, DeclareOpInterfaceMethods<TpuInterface>]> {
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

def TPU_DequantizationOp: TPU_Op<"dequantization",
      [NoSideEffect, DeclareOpInterfaceMethods<TpuInterface>]> {
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

include "mlir/Dialect/TPU/TPUSupportOps.td"
include "mlir/Dialect/TPU/TPUTGOps.td"
include "mlir/Dialect/TPU/TPUTLOps.td"

#endif // TPU_OPS
