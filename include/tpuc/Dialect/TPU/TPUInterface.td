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

//===----------------------------------------------------------------------===//
// TPU interfaces
//===----------------------------------------------------------------------===//

def TpuOpCommonInterface : OpInterface<"TpuOpCommonInterface"> {
  let description = [{
    TPU_Op common interface.
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
  ];
}

def TpuOpQuantInterface : OpInterface<"TpuOpQuantInterface"> {
  let description = [{
    TPU_Op quantization interface.
  }];

  let methods = [
    InterfaceMethod<[{
        Return quant mode.
      }],
      "StringRef", "getOpQuant"
    >,
    InterfaceMethod<[{
        Set quant mode.
      }],
      "LogicalResult", "setOpQuantMode", (ins "StringRef &":$quant)
    >,
    InterfaceMethod<[{
        Return quant param type.
      }],
      "StringRef", "getOpQuantParamType"
    >,
    InterfaceMethod<[{
        Set quant param type.
      }],
      "LogicalResult", "setOpQuantParamType", (ins "StringRef &":$quantParamType)
    >,
    InterfaceMethod<[{
        Return threshold for symmetric quant.
      }],
      "float", "getOpQuantThreshold"
    >,
    InterfaceMethod<[{
        Set threshold for symmetric quant.
      }],
      "LogicalResult", "setOpQuantThreshold", (ins "float ":$threshold)
    >,
    InterfaceMethod<[{
        Quantize Int8.
      }],
      "LogicalResult", "quantizeInt8"
    >,
    InterfaceMethod<[{
        Quantize Bf16.
      }],
      "LogicalResult", "quantizeBf16"
    >,
  ];
}

def TpuOpLowerInterface : OpInterface<"TpuOpLowerInterface"> {
  let description = [{
    TPU_Op lower to TPU_TG_Op interface.
  }];

  let methods = [
    InterfaceMethod<[{
        Do lower.
      }],
      "Value", "convertToTG"
    >,
  ];
}

def TpuTGOpCodegenInterface : OpInterface<"TpuTGOpCodegenInterface"> {
  let description = [{
    TPU_TG_Op genenrate TPU instructions.
  }];

  let methods = [
    InterfaceMethod<[{
        Generate TPU instructions.
      }],
      "LogicalResult", "codegen", (ins "void *":$ctx)
    >,
    InterfaceMethod<[{
        Get result tensor Global Address.
      }],
      "uint64_t", "getGAddr"
    >,
    InterfaceMethod<[{
        Set result tensor Global Address.
      }],
      "LogicalResult", "setGAddr", (ins "uint64_t ":$gaddr)
    >,
    InterfaceMethod<[{
        Set buffer reused.
      }],
      "LogicalResult", "setBufferReused", (ins "bool":$buffer_reused)
    >,
  ];
}

def TpuTLSimpleOpCodegenInterface : OpInterface<"TpuTLSimpleOpCodegenInterface"> {
  let description = [{
    TPU_TL simple genenrate TPU instructions.
  }];

  let methods = [
    InterfaceMethod<[{
        Set load flag.
      }],
      "LogicalResult", "setLoadFlag", (ins "bool":$load_flag)
    >,
    InterfaceMethod<[{
        Get load flag.
      }],
      "bool", "getLoadFlag"
    >,
    InterfaceMethod<[{
        Set store flag.
      }],
      "LogicalResult", "setStoreFlag", (ins "bool":$store_flag)
    >,
    InterfaceMethod<[{
        Get store flag.
      }],
      "bool", "getStoreFlag"
    >,
    InterfaceMethod<[{
    }],
    "StringRef", "getLmLayout"
    >,

    InterfaceMethod<[{
    }],
    "LogicalResult", "setLmLayout", (ins "std::string":$lm_layout)
    >,

  ];
}

def TpuTLOpCodegenInterface : OpInterface<"TpuTLOpCodegenInterface"> {
  let description = [{
    TPU_TL_Op genenrate TPU instructions.
  }];

  let methods = [
    InterfaceMethod<[{
        Generate TPU instructions.
      }],
      "LogicalResult", "codegen", (ins "void *":$ctx)
    >,
    InterfaceMethod<[{
        Get result tensor Global Address.
      }],
      "uint64_t", "getGAddr"
    >,
    InterfaceMethod<[{
        Set result tensor Global Address.
      }],
      "LogicalResult", "setGAddr", (ins "uint64_t ":$gaddr)
    >,
    InterfaceMethod<[{
        Set enable parallel flag.
      }],
      "LogicalResult", "setEnableParallel", (ins "bool ":$flag)
    >,
    InterfaceMethod<[{
        Set disable parallel flag.
      }],
      "LogicalResult", "setDisableParallel", (ins "bool ":$flag)
    >,
    InterfaceMethod<[{
        Get enable parallel flag.
      }],
      "bool", "getEnableParallel"
    >,
    InterfaceMethod<[{
        Get disable parallel flag.
      }],
      "bool", "getDisableParallel"
    >
  ];
}
//===----------------------------------------------------------------------===//
// TPU Traits
//===----------------------------------------------------------------------===//
class TPU_QuantBypass :
  CPred<"$_op.quant().param_type().getValue() == \"NONE\"">;


#endif // TPU_INTERFACE
