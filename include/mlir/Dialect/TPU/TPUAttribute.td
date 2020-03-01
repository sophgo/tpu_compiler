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

#ifndef TPU_ATTRIBUTE
#define TPU_ATTRIBUTE

include "mlir/Dialect/TPU/TPUBase.td"

//===----------------------------------------------------------------------===//
// Quantization methods enum definitions.
//===----------------------------------------------------------------------===//

// Allowed quantization mode cases
def TPU_Quant_None    : StrEnumAttrCase<"NONE">;
def TPU_Quant_Int8    : StrEnumAttrCase<"INT8">;
def TPU_Quant_BF16    : StrEnumAttrCase<"BF16">;

def TPU_QuantModeAttr : StrEnumAttr<
    "QuantizeFlag", "quantization flag enum", [
      TPU_Quant_None,  TPU_Quant_Int8, TPU_Quant_BF16
    ]>;

// Allowed quantization parameter type cases
def TPU_QuantParam_NONE             : StrEnumAttrCase<"NONE">;
def TPU_QuantParam_THRESHOLD        : StrEnumAttrCase<"THRESHOLD">;
def TPU_QuantParam_SCALE            : StrEnumAttrCase<"SCALE">;
def TPU_QuantParam_RSHIFT_ONLY      : StrEnumAttrCase<"RSHIFT_ONLY">;
def TPU_QuantParam_RSHIFT_AND_M_I32 : StrEnumAttrCase<"RSHIFT_AND_M_I32">;
def TPU_QuantParam_RSHIFT_AND_M_I8  : StrEnumAttrCase<"RSHIFT_AND_M_I8">;

def TPU_QuantParamTypeAttr : StrEnumAttr<
    "QuantizeParamTypeFlag", "quantization param type flag enum", [
      TPU_QuantParam_NONE, TPU_QuantParam_THRESHOLD,
      TPU_QuantParam_SCALE, TPU_QuantParam_RSHIFT_ONLY,
      TPU_QuantParam_RSHIFT_AND_M_I32, TPU_QuantParam_RSHIFT_AND_M_I8
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
// Memory region type enum definitions.
//===----------------------------------------------------------------------===//

def TPU_MEM_REGION_INPUT      : StrEnumAttrCase<"INPUT">;
def TPU_MEM_REGION_OUTPUT     : StrEnumAttrCase<"OUTPUT">;
def TPU_MEM_REGION_ACTIVATION : StrEnumAttrCase<"ACTIVATION">;
def TPU_MEM_REGION_WEIGHT     : StrEnumAttrCase<"WEIGHT">;

def TPU_MemRegionAttr : StrEnumAttr<
    "MemoryRegionFlag", "memory region flag enum", [
      TPU_MEM_REGION_INPUT, TPU_MEM_REGION_OUTPUT,
      TPU_MEM_REGION_ACTIVATION, TPU_MEM_REGION_WEIGHT
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
// DetectionOutput_Code_type enum definitions.
//===----------------------------------------------------------------------===//

def TPU_DETECTIONOUTPUT_CODE_TYPE_CORNER  : StrEnumAttrCase<"CORNER">;
def TPU_DETECTIONOUTPUT_CODE_TYPE_CENTER_SIZE : StrEnumAttrCase<"CENTER_SIZE">;
def TPU_DETECTIONOUTPUT_CODE_TYPE_CORNER_SIZE : StrEnumAttrCase<"CORNER_SIZE">;

def TPU_DetectionOutput_Code_typeAttr : StrEnumAttr<"DetectionOutput Code type",
    "DeteictionOutput Code type enum", [
      TPU_DETECTIONOUTPUT_CODE_TYPE_CORNER,
      TPU_DETECTIONOUTPUT_CODE_TYPE_CENTER_SIZE,
      TPU_DETECTIONOUTPUT_CODE_TYPE_CORNER_SIZE
    ]>;

//===----------------------------------------------------------------------===//
// Quantization attributes structure.
//===----------------------------------------------------------------------===//
def TPU_QuantParamAttr : StructAttr<"QuantParam", TPU_Dialect, [
    StructFieldAttr<"mode", TPU_QuantModeAttr>,
    StructFieldAttr<"param_type", TPU_QuantParamTypeAttr>,
    StructFieldAttr<"is_perchannel", BoolAttr>,
    StructFieldAttr<"is_asymmetric", BoolAttr>,
    StructFieldAttr<"threshold_max", F32Attr>,
    StructFieldAttr<"threshold_min", F32Attr>]> {
  let description = "Structure of quantization parameters";
}

//===----------------------------------------------------------------------===//
// Conv dimension parameters structure.
//===----------------------------------------------------------------------===//
def TPU_ConvParamAttr : StructAttr<"ConvParam", TPU_Dialect, [
    StructFieldAttr<"stride_h", I32Attr>,
    StructFieldAttr<"stride_w", I32Attr>,
    StructFieldAttr<"padding", TPU_PaddingAttr>,
    StructFieldAttr<"dilation_h", I32Attr>,
    StructFieldAttr<"dilation_w", I32Attr>,
    StructFieldAttr<"group", I32Attr>,
    StructFieldAttr<"is_dw", BoolAttr>,
    StructFieldAttr<"with_bias", BoolAttr>,
    StructFieldAttr<"do_relu", BoolAttr>]> {
  let description = "Structure of dimension parameters for conv op";
}

//===----------------------------------------------------------------------===//
// Pool dimension parameters structure.
//===----------------------------------------------------------------------===//
def TPU_PoolParamAttr : StructAttr<"PoolParam", TPU_Dialect, [
    StructFieldAttr<"kernel_h", I32Attr>,
    StructFieldAttr<"kernel_w", I32Attr>,
    StructFieldAttr<"padding_t", I32Attr>,
    StructFieldAttr<"padding_b", I32Attr>,
    StructFieldAttr<"padding_l", I32Attr>,
    StructFieldAttr<"padding_r", I32Attr>,
    StructFieldAttr<"stride_h", I32Attr>,
    StructFieldAttr<"stride_w", I32Attr>,
    StructFieldAttr<"do_relu", BoolAttr>]> {
  let description = "Structure of dimension parameters for pool op";
}

#endif // TPU_ATTRIBUTE
