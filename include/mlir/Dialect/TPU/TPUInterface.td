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
      "int", "getOpLayerId"
    >,
    InterfaceMethod<[{
        Assign layer_id to the Op.
      }],
      "LogicalResult", "setOpLayerId", (ins "int ":$id)
    >,
  ];
}

def TpuOpInterpInterface : OpInterface<"TpuOpInterpInterface"> {
  let description = [{
    TPU_Op common interface.
  }];

  let methods = [
    InterfaceMethod<[{
        Do interpret.
      }],
      "LogicalResult", "interpret",
          (ins "DenseMap<Value *, std::shared_ptr<std::vector<float> > > &":$valueMapping)
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
        Return if is_perchannel.
      }],
      "bool", "isOpQuantPerchannel"
    >,
    InterfaceMethod<[{
        Set quant as perchannel.
      }],
      "LogicalResult", "setOpQuantPerchannel", (ins "bool ":$is_perchannel)
    >,
    InterfaceMethod<[{
        Return if is_asymmetric.
      }],
      "bool", "isOpQuantAsymmetric"
    >,
    InterfaceMethod<[{
        Set quant as asymmetric.
      }],
      "LogicalResult", "setOpQuantAsymmetric", (ins "bool ":$is_asymmetric)
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
    //InterfaceMethod<[{
    //    Return threshold_max and threshold_min for asymmetric quant.
    //  }],
    //  "void", "getOpQuantThresholdRange", (ins "float ":$threshold_max, "float ":$threshold_min)
    //>,
    //InterfaceMethod<[{
    //    Set threshold_max and threshold_min for asymmetric quant.
    //  }],
    //  "LogicalResult", "setOpQuantThresholdRange", (ins "float &":$threshold_max, "float &":$threshold_min)
    //>,
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
      "Value *", "convertToTG"
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
  ];
}



// to be deprecated
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
      "int", "getOpLayerId"
    >,
    InterfaceMethod<[{
        Returns result quantization threshold of the Op.
      }],
      "float", "getResultQuantThreshold"
    >,
    InterfaceMethod<[{
        Set result quantization threshold of the Op.
      }],
      "LogicalResult", "setResultQuantThreshold", (ins "float ":$threshold)
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
    InterfaceMethod<[{
        Set result global address for the Op.
      }],
      "LogicalResult", "setResultAddress", (ins "uint64_t ":$gaddr)
    >,
  ];
}



//===----------------------------------------------------------------------===//
// TPU Traits
//===----------------------------------------------------------------------===//
class TPU_QuantBypass :
  CPred<"$_op.quant().param_type().getValue() == \"NONE\"">;

class TPU_QuantPerTensorOnly :
  CPred<"$_op.quant().is_perchannel().getValue() == false">;

class TPU_QuantSupportPerChannel<int m> : Or<[
  CPred<"$_op.quant().is_perchannel().getValue() == false">,
  CPred<"$_op.quant().is_perchannel().getValue() == true">
  ]>;

class TPU_QuantSymmetricOnly :
  CPred<"$_op.quant().is_asymmetric().getValue() == false">;

class TPU_QuantSupportAsymmetric<int m> : Or<[
  CPred<"$_op.quant().is_asymmetric().getValue() == false">,
  CPred<"$_op.quant().is_asymmetric().getValue() == true">
  ]>;

#endif // TPU_INTERFACE
