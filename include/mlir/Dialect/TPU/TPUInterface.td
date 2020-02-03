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

#ifndef TPU_INTERFACE
#define TPU_INTERFACE

include "mlir/IR/OpBase.td"

def TpuInterface : OpInterface<"TpuInterface"> {
  let description = [{
    Encodes properties of a TPU Operations.
  }];

  let methods = [
    InterfaceMethod<[{
        Returns name of the Op.
      }],
      "StringRef", "getOpName"
    >,
    InterfaceMethod<[{
        Returns layer_id of the Op.
      }],
      "int", "getLayerId"
    >,
    InterfaceMethod<[{
        Returns result quantization threshold of the Op.
      }],
      "float", "getResultQuantThreshold"
    >,
    InterfaceMethod<[{
        Returns quant method of the Op.
      }],
      "StringRef", "getOpQuant"
    >,
    InterfaceMethod<[{
        Returns assigned result global address of the Op.
      }],
      "APInt", "getResultAddress"
    >,
  ];
}

#endif // TPU_INTERFACE
