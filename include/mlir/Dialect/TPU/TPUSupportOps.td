//===-- TPUTLOps.td - TPU dialect operation definitions ------*- tablegen -*-===//
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

#ifndef TPU_SUPPORT_OPS
#define TPU_SUPPORT_OPS

include "mlir/Dialect/TPU/TPUBase.td"

//===----------------------------------------------------------------------===//
// Quantitzation/Dequantization.
//===----------------------------------------------------------------------===//

def TPU_QuantOp: TPU_Op<"quant",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpInterpInterface>]> {
  let summary = "Quantization/Dequantization operator";

  let description = [{
    Perform quantization or dequantization among different QuantMode.

    - From `None` to `INT8`
      Quantize a activation tensor into int8, according to its threshold value.
      Q(x) = x * 128 / threshold, and saturate to (-128, 127) range

    - From `INT8` to `None`
      Dequantize a activation tensor from int8 back to float.
      x = Q(x) * threshold / 128
  }];

  let arguments = (
    ins AnyTensor:$input,
    TPU_QuantModeAttr:$from,
    TPU_QuantModeAttr:$to,
    OptionalAttr<F32Attr>:$threshold,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$output);
}

//===----------------------------------------------------------------------===//
// Reshape.
//===----------------------------------------------------------------------===//

def TPU_ReshapeOp: TPU_Op<"reshape",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpInterpInterface>,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuOpQuantInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "Reshape operator";

  let description = [{
    Produces a tensor with the same values but different static shape defined
    by the output type.

    Inputs:
      `input`           : required, the input activation tensor.

    Attributes:
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
      `TpuOpInterpInterface` : support interperter
      `TpuOpQuantInterface`  : to return previous Op`s quant info
      No `TpuOpLowerInterface`  : no lower
      `TpuTGOpCodegenInterface` : to return previous Op`s gaddr
  }];

  let arguments = (
    ins AnyTensor:$input,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );

  let results = (outs AnyTensor:$output);
}

//===----------------------------------------------------------------------===//
// TPU Misc op definitions.
//===----------------------------------------------------------------------===//

def TPU_LoadWeightOp : TPU_Op<"load_weight", [NoSideEffect]> {
  let summary = "load_weight operator";

  let description = [{
    This is a virtual operation, to stand for loading a weight tensor,
    with specific tensor shape and type.

    Inputs:
      `weight_file`     : the virtual memref returned by `load_file` op.

    Attributes:
      `storage`         : represent the data type of the weight when save to
                          weight .bin file, setting to `NONE` means the weight
                          tensor is not present in the final weight bin file.
      `name`            : name of the tensor, this is the name showed in the
                          weight npz file, used as the key to retrieve tensor
                          data from the npz file.
      `lowered`         : indicating the weight data has been processed for
                          TPU hardware, eg. packed, transposed, etc.
      `compressed`      : indicating the weight data has been compressed to
                          utilize hardware decompression.
      `offset`          : global address of the weight tensor, which is an
                          offset within `WEIGHT` memory region.
    Result:
      `tensor`          : result tensor of the specific weight.

    Interfaces or Traits:
      `NoSideEffect`
  }];

  let arguments = (
    ins AnyMemRef:$weight_file,
    DefaultValuedAttr<TPU_WeightAttr, "FP32">:$storage,
    OptionalAttr<StrAttr>:$name,
    DefaultValuedAttr<BoolAttr, "false">:$lowered,
    DefaultValuedAttr<BoolAttr, "false">:$compressed,
    OptionalAttr<NonNegativeI64Attr>:$offset
  );

  let results = (outs AnyTensor:$tensor);
}

def TPU_DummyOp: TPU_Op<"dummy", [NoSideEffect]> {
  let summary = "dummy operator";

  let description = [{
    Dummy is used to carry tensor shape, used by some frameworks, and will be
    removed after importing.

    Inputs:

    Attributes:

    Result:
      `tensor`          : result tensor with the specific shape.

  }];

  let results = (outs AnyTensor:$output);
}

def TPU_NoneOp : TPU_Op<"none", [NoSideEffect]> {
  let summary = "none operator";

  let description = [{
    A none Op to return a NoneType.
  }];

  let results = (outs NoneType);
}

#endif // TPU_SUPPORT_OPS
