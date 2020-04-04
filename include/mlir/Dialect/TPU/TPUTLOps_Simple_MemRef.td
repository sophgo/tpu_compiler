//===-- TPUTLOps_Simple_MemREf.td - TPU dialect operation definitions -*- tablegen -*-===//
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

#ifndef TPU_TL_MEMREF_OPS
#define TPU_TL_MEMREF_OPS

include "mlir/Dialect/TPU/TPUBase.td"
include "mlir/Dialect/TPU/TPUInterface.td"
include "mlir/Dialect/TPU/TPUTLOps_Simple.td"

//===----------------------------------------------------------------------===//
// TL op definitions.
//===----------------------------------------------------------------------===//

def TPU_TL_MemRef_LA_Conv2DOp : TPU_Op<"tl_memref_la_conv_2d",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TL Load All Convolution operator";

  let description = [{
    [IMPORTANT] : for INT8_MULTIPLIER quantized convolution only.
    Performs TL convolution operation on inputs memref, and output to output
      memref. The input/output/weight are still global memory tensors, so this
      operator has to handle loading input/weight and store output by itself.

    Inputs:
      `inputs[0]`  : required: the input memref.
      `inputs[1]`  : required: the filter memref.
      `inputs[2]`  : required: the perchannel memref.
  }];

  let arguments = (
    ins MemRefOf<[AnyFloat, AnyInteger]>:$input,
    MemRefOf<[AnyFloat, AnyInteger]>:$filter,
    MemRefOf<[I32,I16,I8]>:$pc_info,
    MemRefOf<[AnyFloat, AnyInteger]>:$output,
    TPU_ConvParamAttr:$param,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<NonNegativeI64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );
}

def TPU_TL_MemRef_LW_Conv2DOp : TPU_Op<"tl_memref_lw_conv_2d",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TL Load Weight Convolution operator";

  let description = [{
    [IMPORTANT] : for INT8_MULTIPLIER quantized convolution only.
    Performs TL convolution operation on inputs memref, and output to output
      memref. The weight are still global memory tensors, so this operator has
      to handle loading weight by itself.

    Examples:
       %A = tpu.tl_load(%2) {gaddr = 0x60000000}: memref<?x?xf32, offset: 0, strides: [?, 1]>
       %B = tpu.tl_alloc() : memref<?x?xf32, offset: 0, strides: [?, 1]>
       tpu.tl_lw_conv_2d(%A, %B, %4, %5) { ... } : \
           (memref<1x64x56x56xi8, offset: 0x100, strides: [?, 1]>, \
            memref<1x256x56x56xi8, offset: 0x4000, strides: [?, 1]>, \
            tensor<256x64x3x3xi8>, tensor<256x9xi8>) -> tensor<1x256x56x56xf32>)

    Inputs:
      `inputs[0]`  : required: the input memref.
      `inputs[1]`  : required: the filter tensor.
      `inputs[2]`  : required: the per_channel_info tensor.
  }];

  let arguments = (
    ins MemRefOf<[AnyFloat, AnyInteger]>:$input,
    MemRefOf<[AnyFloat, AnyInteger]>:$filter,
    MemRefOf<[I32,I16,I8]>:$pc_info,
    MemRefOf<[AnyFloat, AnyInteger]>:$output,
    TPU_ConvParamAttr:$param,
    TPU_LMLayoutAttr:$lm_layout,
    I32Attr:$la_input,
    I32Attr:$la_working,
    I32Attr:$la_output,
    BoolAttr:$tl_load_flag,
    BoolAttr:$tl_store_flag,
    OptionalAttr<BoolAttr>:$in_short_path,
    DefaultValuedAttr<BoolAttr, "false">:$do_leaky_relu,
    OptionalAttr<I8Attr>:$rshift_pos,
    OptionalAttr<I8Attr>:$m_i8_pos,
    OptionalAttr<I8Attr>:$rshift_neg,
    OptionalAttr<I8Attr>:$m_i8_neg,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<NonNegativeI64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );
}

def TPU_TL_MemRef_EltwiseAddOp : TPU_Op<"tl_memref_eltwise_add",
      [NoSideEffect,
       DeclareOpInterfaceMethods<TpuOpCommonInterface>,
       DeclareOpInterfaceMethods<TpuTGOpCodegenInterface>]> {
  let summary = "TL Eltwise Add operator";

  let description = [{
  }];

  let arguments = (
    ins MemRefOf<[AnyFloat, AnyInteger]>:$input,
    MemRefOf<[I32,I16,I8]>:$addend,
    MemRefOf<[AnyFloat, AnyInteger]>:$output,
    I8Attr:$rshift,
    I32ArrayAttr:$m_i8_inputs,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    TPU_LMLayoutAttr:$lm_layout,
    I32Attr:$la_input,
    I32Attr:$la_working,
    I32Attr:$la_output,
    BoolAttr:$tl_load_flag,
    BoolAttr:$tl_store_flag,
    DefaultValuedAttr<TPU_MemRegionAttr, "ACTIVATION">:$region,
    OptionalAttr<NonNegativeI64Attr>:$gaddr,
    OptionalAttr<BoolAttr>:$buffer_reused,
    StrAttr:$name,
    OptionalAttr<NonNegativeI32Attr>:$layer_id
  );
}

#endif // TPU_TL_MEMREF_OPS
