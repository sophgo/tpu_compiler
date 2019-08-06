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
    AnyTensor:$bias,
    I32Attr:$dilation_h_factor,
    I32Attr:$dilation_w_factor,
    TPU_AFAttr:$fused_activation_function,
    TPU_PaddingAttr:$padding,
    I32Attr:$stride_h,
    I32Attr:$stride_w
  );

  let results = (outs AnyTensor:$output);
}

//===----------------------------------------------------------------------===//
// TPU op definitions.
//===----------------------------------------------------------------------===//
def TPU_AveragePool2DOp :
    TPU_Op<"average_pool_2d", [NoSideEffect]> {
  let summary = "Average_pool_2d operator";

  let description = [{
    Performs average-pooling operation on input.
  }];

  let arguments = (
    ins AnyTensor:$input,
    I32Attr:$filter_height,
    I32Attr:$filter_width,
    TPU_PaddingAttr:$padding,
    I32Attr:$stride_h,
    I32Attr:$stride_w,
    TPU_AFAttr:$fused_activation_function
  );

  let results = (outs AnyTensor:$output);
}

def TPU_MaxPool2DOp : TPU_Op<"max_pool_2d", [NoSideEffect]> {
  let summary = "Max Pool 2D op";

  let description = [{
    Performs max pool 2D on input.

    Inputs:
      `inputs[0]`: required: the input tensor
  }];

  let arguments = (
    ins AnyTensor:$input,
    TPU_PaddingAttr:$padding,
    I32Attr:$stride_w,
    I32Attr:$stride_h,
    I32Attr:$filter_width,
    I32Attr:$filter_height,
    TPU_AFAttr:$fused_activation_function
  );

  let results = (outs AnyTensor:$output);
}

def TPU_Conv2DOp : TPU_ConvOp<"conv_2d", "Convolution">;

def TPU_ReluOp: TPU_Op<"relu", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Relu operator";

  let description = [{
    Element-wise Relu operator
      x -> max(0, x)
  }];

  let arguments = (ins AnyTensor:$x);

  let results = (outs AnyTensor:$y);
}

def TPU_ReshapeOp: TPU_Op<"reshape", [NoSideEffect]> {
  let summary = "Reshape operator";

  let description = [{
    Produces a tensor with the same values but different static shape defined
    by the output type.
  }];

  let arguments = (ins AnyTensor:$input);

  let results = (outs AnyTensor:$output);
}

#endif // TPU_OPS
