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

#ifndef TPU_BASE
#define TPU_BASE

include "mlir/IR/OpBase.td"

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
      TPU_POOL_AVE, TPU_POOL_MAX
    ]>;

//===----------------------------------------------------------------------===//
// Eltwise method enum definitions.
//===----------------------------------------------------------------------===//

// Allowed eltwise method cases
def TPU_ELTWISE_SUM  : StrEnumAttrCase<"SUM">;
def TPU_ELTWISE_PROD : StrEnumAttrCase<"PROD">;
def TPU_ELTWISE_MAX  : StrEnumAttrCase<"MAX">;

def TPU_EltwiseAttr : StrEnumAttr<
    "EltwiseMethodType", "eltwise method enum", [
      TPU_ELTWISE_SUM, TPU_ELTWISE_PROD, TPU_ELTWISE_MAX
    ]>;

//===----------------------------------------------------------------------===//
// TPU op base class.
//===----------------------------------------------------------------------===//

class TPU_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<TPU_Dialect, mnemonic, traits>;

#endif // TPU_BASE
