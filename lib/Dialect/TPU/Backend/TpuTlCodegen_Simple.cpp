//===- ConvertToBinary.cpp - MLIR SPIR-V module to binary conversion ------===//
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
// This file implements a translation from MLIR SPIR-V ModuleOp to SPIR-V
// binary module.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
#include "mlir/Dialect/TPU/QuantizationArithmetic.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/TensorFile.h"

#include <fstream>

#define DEBUG_TYPE "mlir-to-cmdbuf"

using namespace mlir;

extern int BF16_TABLE_START;
extern int BF16_TABLE_END;

#include "backend/backend_tg_api.h"
#include "backend/backend_tl_api.h"

namespace mlir {


LogicalResult tpu::TL_LA_Conv2DOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, ph, pw, dh, dw;
  parseConvParam(param(), false, input(), output(), filter(),
                 n, ic, ih, iw, oc, oh, ow, g,
                 kh, kw, sh, sw, ph, pw, dh, dw, is_dw, with_bias, do_relu);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  gaddr_t ga_filter = getWeightOpAddress(filter()->getDefiningOp());
  gaddr_t ga_pc_info = getWeightOpAddress(pc_info()->getDefiningOp());
  int layer_id = mlir::getOpLayerId(op);

  LLVM_DEBUG(llvm::errs() << "    TL_LA_Conv2DOp, layer_id = " << layer_id << "\n";);
  cvi_backend_tl_conv_LA(*backend_ctx, layer_id,
      ga_input, ga_output, ga_filter, ga_pc_info,
      n, ic, ih, iw, g, oc, oh, ow, kh, kw,
      dh, dw, ph, ph, pw, pw, sh, sw,
      false, with_bias, do_relu);
  return success();
}

LogicalResult tpu::TL_LW_Conv2DOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, ph, pw, dh, dw;
  parseConvParam(param(), false, input(), output(), filter(),
                 n, ic, ih, iw, oc, oh, ow, g,
                 kh, kw, sh, sw, ph, pw, dh, dw, is_dw, with_bias, do_relu);

  gaddr_t ga_input = getPreviousOpAddress(op);
  gaddr_t ga_output = getOpAddress(op);
  gaddr_t ga_filter = getWeightOpAddress(filter()->getDefiningOp());
  gaddr_t ga_pc_info = getWeightOpAddress(pc_info()->getDefiningOp());
  laddr_t la_input = this->la_input().getLimitedValue();
  laddr_t la_output = this->la_output().getLimitedValue();
  laddr_t la_working = this->la_working().getLimitedValue();
  int layer_id = mlir::getOpLayerId(op);

  llvm::errs() << "    TL_LW_Conv2DOp, layer_id = " << layer_id;
  if (tl_load_flag())
    llvm::errs() << ", LD";
  if (tl_store_flag())
    llvm::errs() << ", ST";
  if (!tl_load_flag() && !tl_store_flag())
    llvm::errs() << ", FUSED";
  llvm::errs() << "\n";

  if (tl_load_flag()) {
    cvi_backend_tl_load(*backend_ctx, layer_id,
        la_input, ga_input, n, ic, ih, iw);
  }
  #if 0
  //
  // V0: Weight Only version, with no parallel for load/store activations
  // (only consider load weight parallel)
  //
  cvi_backend_tl_conv_LW(*backend_ctx, layer_id,
      la_input, la_output, la_working,
      ga_filter, ga_pc_info,
      n, ic, ih, iw, g, oc, oh, ow, kh, kw,
      dh, dw, ph, ph, pw, pw, sh, sw,
      false, with_bias, do_relu);
  if (op.tl_store_flag()) {
    cvi_backend_tl_store(*backend_ctx, layer_id,
        la_output, ga_output, n, oc, oh, ow);
  }
  #endif
  #if 1
  //
  // V1: Weight and Store version
  //   this only do parallel on both Weight load and Activation Store
  //   but load activation is not handled in parallel
  //
  if (tl_store_flag()) {
    cvi_backend_tl_conv_LW(*backend_ctx, layer_id,
        la_input, la_output, la_working,
        ga_filter, ga_pc_info,
        n, ic, ih, iw, g, oc, oh, ow, kh, kw,
        dh, dw, ph, ph, pw, pw, sh, sw,
        false, with_bias, do_relu,
        true, ga_output);
  } else {
    cvi_backend_tl_conv_LW(*backend_ctx, layer_id,
        la_input, la_output, la_working,
        ga_filter, ga_pc_info,
        n, ic, ih, iw, g, oc, oh, ow, kh, kw,
        dh, dw, ph, ph, pw, pw, sh, sw,
        false, with_bias, do_relu);
  }
  #endif
  return success();
}


LogicalResult tpu::TL_EltwiseAddOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  bool do_relu = this->do_relu();

  assert(op->getNumOperands() == 2 && "support 2 inputs only");

  gaddr_t ga_input = getPreviousOpAddress(op, 0);
  gaddr_t ga_addend = getPreviousOpAddress(op, 1);
  gaddr_t ga_output = getOpAddress(op);
  int layer_id = mlir::getOpLayerId(op);

  int8_t rshift = this->rshift().getLimitedValue();
  int8_t m_i8_input[2];
  std::vector<int32_t> m_i8_inputs_array;
  arrayAttrToVector(this->m_i8_inputs(), m_i8_inputs_array);
  assert(m_i8_inputs_array.size() == 2);
  m_i8_input[0] = static_cast<int8_t>(m_i8_inputs_array[0]);
  m_i8_input[1] = static_cast<int8_t>(m_i8_inputs_array[1]);

  // TODO: should change on backend API, rather than doing cast
  gaddr_t ga_inputs[2] = {ga_input, ga_addend};
  int rshift_int;
  int m_int[2];
  if (1) {
    rshift_int = static_cast<int>(rshift);
    m_int[0] = static_cast<int>(m_i8_input[0]);
    m_int[1] = static_cast<int>(m_i8_input[1]);
  }
  const int coeffs[2] = {1, 1};

  bmnet_eltwise_fixed_forward_bmkernel(
      *backend_ctx,
      0,            // stream_id,
      0,            // inst_id,
      layer_id,     // layer_id,
      nullptr,      // depends
      0,            // depends_len
      ga_inputs,    // gaddr_t ga_input[],
      ga_output,    // gaddr_t ga_output,
      2,            // int input_size,
      1,            // int op,  0, prod, 1, sum, 2, max
      n, c, h, w,
      do_relu,      // bool do_relu,
      0.0f,         // float relu_slope,
      rshift_int,   // int right_shift_width,
      m_int,
      coeffs);

  return success();
}

}
