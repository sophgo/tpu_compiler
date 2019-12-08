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
// Defines some operations of the GPU dialect.
//
//===----------------------------------------------------------------------===//

#ifdef TPU_OPS
#else
#define TPU_OPS

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

def TPU_Dialect : Dialect {
  let name = "tpu";

  let description = [{
    The TPU dialect.

    This dialect maps to TPU operations.
  }];

  let cppNamespace = "tpu";
}

//===----------------------------------------------------------------------===//
// Quantization methods enum definitions.
//===----------------------------------------------------------------------===//

// Allowed quantization methods cases
def TPU_Quant_None    : StrEnumAttrCase<"NONE">;
def TPU_Quant_Int8    : StrEnumAttrCase<"INT8">;
def TPU_Quant_Int8_PC : StrEnumAttrCase<"INT8_PER_CHANNEL">;
def TPU_Quant_Int8_M  : StrEnumAttrCase<"INT8_MULTIPLIER">;
def TPU_Quant_BF16    : StrEnumAttrCase<"BF16">;

def TPU_QuantAttr : StrEnumAttr<
    "QuantizeFlag", "quantization flag enum", [
      TPU_Quant_None,  TPU_Quant_Int8, TPU_Quant_Int8_PC,
      TPU_Quant_Int8_M, TPU_Quant_BF16
    ]>;

//===----------------------------------------------------------------------===//
// Weight storage type enum definitions.
//===----------------------------------------------------------------------===//

// Allowed weight storage type cases, defines how each weight element is stored
// in the generated weight.bin file
// BF16 is an alias of UINT16
def TPU_Weight_None   : StrEnumAttrCase<"NONE">;
def TPU_Weight_S8     : StrEnumAttrCase<"INT8">;
def TPU_Weight_U8     : StrEnumAttrCase<"UINT8">;
def TPU_Weight_S16    : StrEnumAttrCase<"INT16">;
def TPU_Weight_U16    : StrEnumAttrCase<"UINT16">;
def TPU_Weight_S32    : StrEnumAttrCase<"INT32">;
def TPU_Weight_U32    : StrEnumAttrCase<"UINT32">;
def TPU_Weight_BF16   : StrEnumAttrCase<"BF16">;
def TPU_Weight_FP32   : StrEnumAttrCase<"FP32">;

def TPU_WeightAttr : StrEnumAttr<
    "WeightStorageFlag", "weight storage flag enum", [
      TPU_Weight_None, TPU_Weight_S8, TPU_Weight_U8, TPU_Weight_S16, TPU_Weight_U16,
      TPU_Weight_S32, TPU_Weight_U32, TPU_Weight_BF16, TPU_Weight_FP32
    ]>;

//===----------------------------------------------------------------------===//
// Activation function enum definitions.
//===----------------------------------------------------------------------===//

// Allowed activation function cases
def TPU_AF_None  : StrEnumAttrCase<"NONE">;
def TPU_AF_Relu  : StrEnumAttrCase<"RELU">;
def TPU_AF_Relu1 : StrEnumAttrCase<"RELU_N1_TO_1">;
def TPU_AF_Relu6 : StrEnumAttrCase<"RELU6">;
def TPU_AF_Tanh  : StrEnumAttrCase<"TANH">;
def TPU_AF_Sign  : StrEnumAttrCase<"SIGN_BIT">;

def TPU_AFAttr : StrEnumAttr<
    "ActivationFunctionType", "fused activation enum", [
      TPU_AF_None,  TPU_AF_Relu, TPU_AF_Relu1,
      TPU_AF_Relu6, TPU_AF_Tanh, TPU_AF_Sign
    ]>;

//===----------------------------------------------------------------------===//
// Padding enum definitions.
//===----------------------------------------------------------------------===//

// Allowed padding cases
def TPU_PAD_Same  : StrEnumAttrCase<"SAME">;
def TPU_PAD_Valid : StrEnumAttrCase<"VALID">;

def TPU_PaddingAttr : StrEnumAttr<"Padding", "padding enum", [
      TPU_PAD_Same, TPU_PAD_Valid
    ]>;

//===----------------------------------------------------------------------===//
// Pool method enum definitions.
//===----------------------------------------------------------------------===//

// Allowed pool method cases
def TPU_POOL_AVE  : StrEnumAttrCase<"AVE">;
def TPU_POOL_MAX  : StrEnumAttrCase<"MAX">;

def TPU_PoolAttr : StrEnumAttr<
    "PoolMethodType", "pool method enum", [
      TPU_POOL_AVE,  TPU_POOL_MAX
    ]>;

//===----------------------------------------------------------------------===//
// TPU op base class.
//===----------------------------------------------------------------------===//

class TPU_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<TPU_Dialect, mnemonic, traits>;

class TPU_ConvOp<string mnemonic, string opSummary> :
    TPU_Op<mnemonic, [NoSideEffect]> {
  let summary = opSummary # " operator";

  let description = [{
    Performs convolution operation on inputs.

    Inputs:
      `inputs[0]`: required: the input activation tensor
      `inputs[1]`: required: the filter weight tensor
      `inputs[2]`: optional: the bias tensor
  }];

  let arguments = (
    ins AnyTensor:$input,
    AnyTensor:$filter,
    Variadic<AnyTensor>:$bias,
    DefaultValuedAttr<I32Attr, "1">:$dilation_h_factor,
    DefaultValuedAttr<I32Attr, "1">:$dilation_w_factor,
    DefaultValuedAttr<TPU_AFAttr, "NONE">:$fused_activation_function,
    TPU_PaddingAttr:$padding,
    I32Attr:$stride_h,
    I32Attr:$stride_w,
    DefaultValuedAttr<I32Attr, "1">:$group,
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
// TPU op definitions.
//===----------------------------------------------------------------------===//
def TPU_Conv2DOp : TPU_ConvOp<"conv_2d", "Convolution">;

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
    TPU_PaddingAttr:$padding,
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

def TPU_FullyConnectedOp : TPU_Op<"fully_connected", [NoSideEffect]> {
  let summary = "Fully connected operator";

  let arguments = (
    ins AnyTensor:$input,
    AnyTensor:$filter,
    Variadic<AnyTensor>:$bias,
    DefaultValuedAttr<TPU_AFAttr, "NONE">:$fused_activation_function,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<NonNegativeI64Attr>:$offset,
    OptionalAttr<F32Attr>:$threshold_y,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant
  );

  let results = (outs AnyTensor:$output);
}

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
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<F32Attr>:$threshold_y
  );

  let results = (outs AnyTensor:$y);
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
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<F32Attr>:$threshold_y
  );

  let results = (outs AnyTensor:$y);
}

def TPU_ReluOp: TPU_Op<"relu", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Relu operator";

  let description = [{
    Element-wise Relu operator
      x -> max(0, x)
  }];

  let arguments = (
    ins AnyTensor:$x,
    F32Attr:$negative_slope,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<NonNegativeI64Attr>:$offset,
    OptionalAttr<F32Attr>:$threshold_y,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$y);
}

def TPU_EltwiseOp: TPU_Op<"eltwise", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Eltwise operator";

  let description = [{
    Performs eltwase operation on inputs.
  }];

  let arguments = (
    ins AnyTensor:$x1,
    AnyTensor:$x2,
    DefaultValuedAttr<TPU_AFAttr, "NONE">:$fused_activation_function,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<NonNegativeI64Attr>:$offset,
    OptionalAttr<F32Attr>:$threshold_y,
    OptionalAttr<NonNegativeI32Attr>:$layer_id,
    DefaultValuedAttr<TPU_QuantAttr, "NONE">:$quant
  );

  let results = (outs AnyTensor:$y);
}

def TPU_SoftmaxOp: TPU_Op<"softmax", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Softmax operator";

  let description = [{
    Perform softmax on input.
  }];

  let arguments = (
    ins AnyTensor:$x,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<F32Attr>:$threshold_y
  );

  let results = (outs AnyTensor:$y);
}

def TPU_InputOp: TPU_Op<"input", [NoSideEffect]> {
  let summary = "Input operator";

  let description = [{
    Produces a tensor from function input, primarily for carrying threshold_y.
  }];

  let arguments = (
    ins AnyTensor:$input,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<F32Attr>:$threshold_y
  );

  let results = (outs AnyTensor:$output);
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
    OptionalAttr<F32Attr>:$threshold_y
  );

  let results = (outs AnyTensor:$output);
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

#endif // TPU_OPS
