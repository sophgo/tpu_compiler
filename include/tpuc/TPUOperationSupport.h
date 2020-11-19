//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
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
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TPU_OPERATION_SUPPORT_H_
#define MLIR_DIALECT_TPU_OPERATION_SUPPORT_H_

#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "tpuc/CustomOpParam.h"

namespace mlir {

void convertAttributesToOpParam(const DictionaryAttr &attrs, cvi::OpParam &param);
void convertOpParamToAttributes(
    mlir::Builder &builder, cvi::OpParam &param,
    std::vector<NamedAttribute> &out);

void arrayAttrToVector(const ArrayAttr &arrayAttr,
                       std::vector<int32_t> &vector);
void arrayAttrToVector(const ArrayAttr &arrayAttr, std::vector<float> &vector);

llvm::StringRef getOpName(Operation *op);
llvm::StringRef getPreviousOpName(Operation *op, uint index = 0);

int getOpLayerId(Operation *op);

llvm::StringRef getChipName(Operation *op);
LogicalResult setChipName(Operation *op, llvm::StringRef chipname);

llvm::StringRef getOpQuant(Operation *op);
LogicalResult setOpQuant(Operation *op, llvm::StringRef mode);

llvm::StringRef getOpQuantParamType(Operation *op);
LogicalResult setOpQuantParamType(Operation *op, llvm::StringRef type);

bool isOpQuantPerchannel(Operation *op);
LogicalResult setOpQuantPerchannel(Operation *op, bool flag);

bool isOpQuantAsymmetric(Operation *op);
LogicalResult setOpQuantAsymmetric(Operation *op, bool flag);

float getOpThreshold(Operation *op);
LogicalResult setOpThreshold(Operation *op, float threshold);
float getPreviousOpThreshold(Operation *op, uint index = 0);

int getOpZeroPoint(Operation *op);
LogicalResult setOpZeroPoint(Operation *op, int zero_point);
int getPreviousOpZeroPoint(Operation *op, uint index = 0);

uint64_t getOpAddress(Operation *op);
LogicalResult setOpAddress(Operation *op, uint64_t gaddr);
uint64_t getPreviousOpAddress(Operation *op, uint index = 0);

uint64_t getWeightOpAddress(Operation *op);

Operation* getNextOp(Operation *op);

void setOpResultType(Value value, Type eltType);

LogicalResult setOpBufferReused(Operation *op, bool flag);

tpu::QuantParam getDefaultQuantParam(Builder &builder);

void parseConvParam(const tpu::ConvParam &p, bool is_deconv,
    Value input, Value output, Value filter,
    int &n, int &ic, int &ih, int &iw, int &oc, int &oh, int &ow, int &g,
    int &kh, int &kw, int &sh, int &sw, int &pt, int &pb, int &pl, int &pr, int &dh, int &dw,
    bool &is_dw, bool &with_bias, bool &do_relu, int &pad_value);

void parseConv3dParam(const tpu::Conv3dParam &p, bool is_deconv,
    Value input, Value output, Value filter,
    int &n, int &ic, int &id, int &ih, int &iw,
    int &oc, int &od, int &oh, int &ow, int &g,
    int &kd, int &kh, int &kw,
    int &sd, int &sh, int &sw,
    int &pd0, int &pd1, int &pt, int &pb, int &pl, int &pr,
    int &dd, int &dh, int &dw,
    bool &is_dw, bool &with_bias, bool &do_relu);

void parsePoolParam(const tpu::PoolParam &p,
    Value input, Value output,
    int &n, int &c, int &ih, int &iw, int &oh, int &ow,
    int &kh, int &kw, int &sh, int &sw, int &pt, int &pb, int &pl, int &pr,
    bool &is_global, bool &do_relu, bool &count_include_pad);

void parsePool3dParam(const tpu::Pool3dParam &p,
    Value input, Value output,
    int &n, int &c, int &id, int &ih, int &iw,
    int &od, int &oh, int &ow,
    int &kd, int &kh, int &kw,
    int &sd, int &sh, int &sw,
    int &pd0, int &pd1, int &pt, int &pb, int &pl, int &pr,
    bool &is_global, bool &do_relu, bool &count_include_pad);

void parseFullyConnectedParam(
    Value input, Value output, Value filter,
    int &m, int &k, int &n);

template<typename OpTy>
void parseLeakyReluParam(Operation *op,
    int8_t &pos_rshift, int8_t &pos_m_i8,
    int8_t &neg_rshift, int8_t &neg_m_i8,
    float &negative_slope);

void parseLstmParam(
    Value input, Value recurrence,
    int &seq_len, int &batch_size, int &input_size, int& hidden_size);

void parseGruParam(
    Value input, Value recurrence,
    int &seq_len, int &batch_size, int &input_size, int& hidden_size);

void parseActCompressParam(const tpu::ActCmprParam &param, int &cmpr_n,
    int &cmpr_c, int &cmpr_h, int64_t &step_size, int64_t &total_size);

bool isBf16Tensor(Value val);

int64_t getTotalCompressedActivationSize(Operation *op);

void getTiledCompressedSize(int n, int c, int h, int w, int n_step, int c_step,
    int h_step, int isBf16, int64_t &stepSize, int64_t &totalSize);

void getTiledCompressedActSize(Operation *op, int n_step, int oc_step,
    int oh_step, int ow, int64_t &stepSize, int64_t &totalSize);

} // namespace mlir

#endif // MLIR_DIALECT_TPU_OPERATION_SUPPORT_H_
