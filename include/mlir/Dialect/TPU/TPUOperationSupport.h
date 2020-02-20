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

namespace mlir {

void arrayAttrToVector(const ArrayAttr &arrayAttr,
                       std::vector<int32_t> &vector);

llvm::StringRef getOpName(Operation *op);
llvm::StringRef getPreviousOpName(Operation *op, uint index = 0);

int getOpLayerId(Operation *op);
LogicalResult setOpLayerId(Operation *op, int id);

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

void parseConvParam(const tpu::ConvParam &p,
    Value *input, Value *output, Value *filter,
    int &n, int &ic, int &ih, int &iw, int &oc, int &oh, int &ow, int &g,
    int &kh, int &kw, int &sh, int &sw, int &ph, int &pw, int &dh, int &dw,
    bool &is_dw, bool &with_bias, bool &do_relu);

void parsePoolParam(const tpu::PoolParam &p,
    Value *input, Value *output,
    int &n, int &c, int &ih, int &iw, int &oh, int &ow,
    int &kh, int &kw, int &sh, int &sw, int &pt, int &pb, int &pl, int &pr,
    bool &is_global, bool &do_relu);

void getDeConv2DOpParam(tpu::DeConv2DOp &op,
    int &n, int &ic, int &ih, int &iw, int &oc, int &oh, int &ow, int &g,
    int &kh, int &kw, int &sh, int &sw, int &ph, int &pw, int &dh, int &dw,
    bool &with_bias);

void getFullyConnectedOpParam(tpu::FullyConnectedOp &op,
    bool &with_transpose, int &m, int &k, int &n,
    bool &with_bias, bool &do_relu);

void getDeConv2DOpVariadicTensors(tpu::DeConv2DOp &op,
    std::vector<std::shared_ptr<std::vector<float> > > &opdT,
    std::shared_ptr<std::vector<float> > &bias,
    std::shared_ptr<std::vector<float> > &rshift,
    std::shared_ptr<std::vector<float> > &multiplier,
    std::shared_ptr<std::vector<float> > &per_channel_info,
    std::shared_ptr<std::vector<float> > &eltwise_input);

void getScaleOpVariadicTensors(
    tpu::ScaleOp &op, std::vector<std::shared_ptr<std::vector<float>>> &opdT,
    std::shared_ptr<std::vector<float>> &bias,
    std::shared_ptr<std::vector<float>> &rshift,
    std::shared_ptr<std::vector<float>> &multiplier);

void getFullyConnectedOpVariadicTensors(
    tpu::FullyConnectedOp &op,
    std::vector<std::shared_ptr<std::vector<float>>> &opdT,
    std::shared_ptr<std::vector<float>> &bias,
    std::shared_ptr<std::vector<float>> &rshift);

void getPReluOpVariadicTensors(tpu::PReluOp &op,
    std::vector<std::shared_ptr<std::vector<float> > > &opdT,
    std::shared_ptr<std::vector<float> > &rshift_pos,
    std::shared_ptr<std::vector<float> > &multiplier_pos,
    std::shared_ptr<std::vector<float> > &rshift_neg);

void getReluOpVariadicTensors(tpu::ReluOp &op,
    std::vector<std::shared_ptr<std::vector<float> > > &opdT,
    std::shared_ptr<std::vector<float> > &rshift_pos,
    std::shared_ptr<std::vector<float> > &multiplier_pos,
    std::shared_ptr<std::vector<float> > &rshift_neg,
    std::shared_ptr<std::vector<float> > &multiplier_neg);

} // namespace mlir

#endif // MLIR_DIALECT_TPU_OPERATION_SUPPORT_H_
