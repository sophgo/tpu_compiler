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

using namespace mlir;

extern int BF16_TABLE_START;
extern int BF16_TABLE_END;

#include "backend/backend_tg_api.h"
#include "backend/backend_tl_api.h"

namespace mlir {

LogicalResult tpu::TL_LG_Conv2DOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();

  bool is_dw, with_bias, do_relu;
  int n, ic, ih, iw, oc, oh, ow, g, kh, kw, sh, sw, ph, pw, dh, dw;
  parseConvParam(param(), false, input(), output(), filter(),
                 n, ic, ih, iw, oc, oh, ow, g,
                 kh, kw, sh, sw, ph, pw, dh, dw, is_dw, with_bias, do_relu);

  laddr_t la_input = this->la_input().getLimitedValue();
  laddr_t la_output = this->la_output().getLimitedValue();
  laddr_t la_weight = this->la_filter().getLimitedValue();
  laddr_t la_perchanel = this->la_bias().getLimitedValue();
  laddr_t la_working = this->la_working().getLimitedValue();

  // pad is not "SAME", can not get from conv param
  int ph_t = this->pad_top_h().getLimitedValue();
  int ph_b = this->pad_bottom_h().getLimitedValue();
  int pw_l = this->pad_left_w().getLimitedValue();
  int pw_r = this->pad_right_w().getLimitedValue();
  int layer_id = mlir::getOpLayerId(op);

  cvi_backend_tl_conv(
    *backend_ctx,
    layer_id,
    la_input, la_output, la_weight, la_working, la_perchanel,
    n, ic, ih, iw,
    g, oc, oh, ow, kh, kw, dh,
    dw, ph_t, ph_b, pw_l, pw_r, sh, sw,
    0,/*result_add*/
    0, /*ctrl*/
    with_bias,
    do_relu,
    0,/*rshift,*/
    oc, /*right_shift_len,*/
    0, /*rshift_pos*/
    0, /*rshift8_neg*/
    0, /*m_i8_pos*/
    0 /*m_i8_neg*/
    );

  return success();
}


LogicalResult tpu::TL_LG_EltwiseAddOp::codegen(void *ctx) {
  llvm::errs() << "TG_codegen: " << getOperationName()
               << " [" << getOpName() << "]\n";

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = mlir::getOpLayerId(op);
  TensorFile *wTF = getWeightTensorFile(op);

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  getTensorShapeAndSize(op->getOperand(0), shape, input_size);
  getNCHW(shape, n, c, h, w);
  bool do_relu = this->do_relu();
  int nInputs = op->getNumOperands();
  assert(op->getNumOperands() == 2 && "support 2 inputs only");

  std::vector<int32_t> la_input_array;
  laddr_t la_input[nInputs];
  arrayAttrToVector(this->la_input().getValue(), la_input_array);
  for (unsigned i = 0; i < nInputs; ++i) {
      la_input[i] = la_input_array[i];
  }

  laddr_t la_output = this->la_output().getLimitedValue();
  laddr_t la_working = this->la_working().getLimitedValue();

  bool do_quant_rescale = false;
  int8_t rshift;
  int8_t m_i8_input[2];
  if (this->rshift().hasValue() && this->m_i8_inputs().hasValue()) {
    do_quant_rescale = true;
    rshift = this->rshift().getValue().getLimitedValue();

    std::vector<int32_t> m_i8_inputs_array;
    arrayAttrToVector(this->m_i8_inputs().getValue(), m_i8_inputs_array);
    assert(m_i8_inputs_array.size() == op->getNumOperands());
    assert(m_i8_inputs_array.size() >= 2);
    m_i8_input[0] = static_cast<int8_t>(m_i8_inputs_array[0]);
    m_i8_input[1] = static_cast<int8_t>(m_i8_inputs_array[1]);
  }


  // op code PROD = 0; SUM = 1; MAX = 2;
  int op_code = 1;
  const int coeffs[2] = {1, 1};

  cvi_backend_tl_eltwise( *backend_ctx,
                          layer_id, /*u32 layer_id,*/
                          la_input,
                          la_output,
                          la_working,
                          n, c, h, w, nInputs,
                          op_code,
                          rshift,
                          m_i8_input,
                          true, /*use_default_coeff,*/
                          do_relu,
                          0, /*relu_slope,*/
                          coeffs,
                          0);

  return success();
}


LogicalResult tpu::TL_LG_LoadNeuronOp::codegen(void *ctx) {
  llvm::errs() << "TL Load Neuron codegen.\n";
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = 0;

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  shape = getTensorShape(op->getOperand(0));
  getNCHW(shape, n, c, h, w);

  int global_c = c;
  int global_h = h;
  int global_w = w;

  shape = getTensorShape(op->getResult(0));
  getNCHW(shape, n, c, h, w);

  int local_n = n;
  int local_c = c;
  int local_h = h;
  int local_w = w;


  gaddr_t src_gaddr = this->gaddr()->getLimitedValue();
  laddr_t dst_laddr = this->laddr()->getLimitedValue();
  bool transpose = this->transpose();
  bool aligned = this->align();
  bool isNeuron = true;

  cvi_backend_tl_load_stride( *backend_ctx,
                              layer_id,
                              src_gaddr,
                              dst_laddr,
                              local_n, local_c, local_h, local_w,
                              global_c, global_h, global_w,
                              transpose, aligned, isNeuron);
  return success();
}

LogicalResult tpu::TL_LG_LoadCoeffOp::codegen(void *ctx) {
  llvm::errs() << "TL Load Coeff codegen.\n";
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = 0;

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  int64_t r_n, r_c, r_h, r_w;
  shape = getTensorShape(op->getResult(0));
  getNCHW(shape, r_n, r_c, r_h, r_w);
  getNCHW(shape, n, c, h, w);

  // load coeff as shape (1, oc, kh * kw, ic/g)
  if (this->tensor_type() == "CONV_COEFF") {
    n = 1;
    c = r_n;
    h = r_h * r_w;
    w = r_c;
    llvm::errs() << "conv coeff load shape(nchw): ( " << n << " ,"
                 << c << " ," << h << ", " << w << ")\n";
  }

  int local_n = n;
  int local_c = c;
  int local_h = h;
  int local_w = w;

  int global_c = c;
  int global_h = h;
  int global_w = w;

  gaddr_t src_gaddr = this->gaddr()->getLimitedValue();
  laddr_t dst_laddr = this->laddr()->getLimitedValue();
  bool transpose = this->transpose();
  bool aligned = this->align();
  bool isNeuron = false;

  cvi_backend_tl_load_stride( *backend_ctx,
                              layer_id,
                              src_gaddr,
                              dst_laddr,
                              local_n, local_c, local_h, local_w,
                              global_c, global_h, global_w,
                              transpose, aligned, isNeuron);
  return success();
}

LogicalResult tpu::TL_LG_StoreOp::codegen(void *ctx) {
  llvm::errs() << "TL Store codegen.\n";
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = 0;

  std::vector<int64_t> shape;
  int64_t input_size, n, c, h, w;
  shape = getTensorShape(op->getOperand(0));
  getNCHW(shape, n, c, h, w);

  int local_n = n;
  int local_c = c;
  int local_h = h;
  int local_w = w;

  shape = getTensorShape(op->getResult(0));
  getNCHW(shape, n, c, h, w);

  int global_c = c;
  int global_h = h;
  int global_w = w;

  gaddr_t src_gaddr = this->gaddr()->getLimitedValue();
  laddr_t dst_laddr = this->laddr()->getLimitedValue();
  bool transpose = this->transpose();
  bool aligned = this->align();
  bool isNeuron = true;

  cvi_backend_tl_store_stride( *backend_ctx,
                                layer_id,
                                src_gaddr,
                                dst_laddr,
                                local_n, local_c, local_h, local_w,
                                global_c, global_h, global_w,
                                transpose, aligned, isNeuron);
  return success();
}

LogicalResult tpu::TL_LG_JoinOp::codegen(void *ctx) {
  llvm::errs() << "TL join codegen.\n";
  return success();
}

LogicalResult tpu::TL_LG_INT8_PoolAvg2DOp::codegen(void *ctx) {
  llvm::errs() << "TL int8 pool avg codegen.\n";

  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = mlir::getOpLayerId(op);
  TensorFile *wTF = getWeightTensorFile(op);

  bool is_global, do_relu;
  int n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr;
  parsePoolParam(param(), input(), output(),
                 n, c, ih, iw, oh, ow,
                 kh, kw, sh, sw, pt, pb, pl, pr,
                 is_global, do_relu);

  laddr_t la_input = this->la_input().getLimitedValue();
  laddr_t la_output = this->la_output().getLimitedValue();

  assert(this->rshift().hasValue());
  int8_t rshift_i8 = this->rshift().getValue().getLimitedValue();
  assert(this->m_i8().hasValue());
  int8_t m_i8 = this->m_i8().getValue().getLimitedValue();

  cvi_backend_tl_pooling( *backend_ctx,
                          layer_id,
                          la_input, la_output,
                          n, c, ih, iw,
                          n, c, oh, ow,
                          kh, kw, sh, sw,
                          pt, pb, pl, pr,
                          true, /*is_avg_pooling,*/
                          rshift_i8,
                          m_i8);

  return success();
}

LogicalResult tpu::TL_LG_INT8_PoolMax2DOp::codegen(void *ctx) {
  llvm::errs() << "TL int8 pool max codegen.\n";
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  Operation *op = this->getOperation();
  int layer_id = mlir::getOpLayerId(op);
  TensorFile *wTF = getWeightTensorFile(op);

  bool is_global, do_relu;
  int n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr;
  parsePoolParam(param(), input(), output(),
                 n, c, ih, iw, oh, ow,
                 kh, kw, sh, sw, pt, pb, pl, pr,
                 is_global, do_relu);

  laddr_t la_input = this->la_input().getLimitedValue();
  laddr_t la_output = this->la_output().getLimitedValue();
  int8_t rshift_i8 = 0, multiplier_i8 = 1;

  cvi_backend_tl_pooling( *backend_ctx,
                          layer_id,
                          la_input, la_output,
                          n, c, ih, iw,
                          n, c, oh, ow,
                          kh, kw, sh, sw,
                          pt, pb, pl, pr,
                          false, /*is_avg_pooling,*/
                          rshift_i8,
                          multiplier_i8);

  return success();
}

}
