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

#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Dialect/TPU/CustomOpParam.h"

namespace mlir {

void convertAttributesToOpParam(const DictionaryAttr &attrs, cvi::OpParam &param);
void convertOpParamToAttributes(
    mlir::Builder &builder, cvi::OpParam &param,
    std::vector<NamedAttribute> &out);

void arrayAttrToVector(const ArrayAttr &arrayAttr,
                       std::vector<int32_t> &vector);

llvm::StringRef getOpName(Operation *op);
llvm::StringRef getPreviousOpName(Operation *op, uint index = 0);

int getOpLayerId(Operation *op);
LogicalResult setOpLayerId(Operation *op, int id);

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

uint64_t getOpAddress(Operation *op);
LogicalResult setOpAddress(Operation *op, uint64_t gaddr);
uint64_t getPreviousOpAddress(Operation *op, uint index = 0);

uint64_t getWeightOpAddress(Operation *op);

Operation* getNextOp(Operation *op);

void setOpResultType(Operation *op, StandardTypes::Kind kind, int width = 0);

LogicalResult setOpBufferReused(Operation *op, bool flag);

tpu::QuantParam getDefaultQuantParam(Builder &builder);

void parseConvParam(const tpu::ConvParam &p, bool is_deconv,
    Value *input, Value *output, Value *filter,
    int &n, int &ic, int &ih, int &iw, int &oc, int &oh, int &ow, int &g,
    int &kh, int &kw, int &sh, int &sw, int &pt, int &pb, int &pl, int &pr, int &dh, int &dw,
    bool &is_dw, bool &with_bias, bool &do_relu);

void parsePoolParam(const tpu::PoolParam &p,
    Value *input, Value *output,
    int &n, int &c, int &ih, int &iw, int &oh, int &ow,
    int &kh, int &kw, int &sh, int &sw, int &pt, int &pb, int &pl, int &pr,
    bool &is_global, bool &do_relu, bool &count_include_pad);

void parseFullyConnectedParam(
    Value *input, Value *output, Value *filter,
    int &m, int &k, int &n);

void parseLstmParam(
    Value *input, Value *recurrence,
    int &seq_len, int &batch_size, int &input_size, int& hidden_size);

void parseGruParam(
    Value *input, Value *recurrence,
    int &seq_len, int &batch_size, int &input_size, int& hidden_size);

} // namespace mlir

#endif // MLIR_DIALECT_TPU_OPERATION_SUPPORT_H_
