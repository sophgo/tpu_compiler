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
// TPU Misc op definitions.
//===----------------------------------------------------------------------===//

def TPU_LoadFileOp : TPU_Op<"load_file", [NoSideEffect]> {
  let summary = "load_file operator";

  let description = [{
    This is a virtual operation, to keep a record of the weight file.

    Inputs:
      None

    Attributes:
      `filename`        : weight file name.

    Result:
      `weight_file`     : result memref, this is a virtual result.

    Interfaces or Traits:
      `NoSideEffect`
  }];

  let arguments = (
    ins StrAttr:$filename
  );

  let results = (outs AnyMemRef:$weight);
}

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
