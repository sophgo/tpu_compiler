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

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {

llvm::StringRef getOpName(Operation *op);
llvm::StringRef getPreviousOpName(Operation *op, uint index = 0);
std::string getOpQuant(Operation *op);
float getOpThreshold(Operation *op);
float getPreviousOpThreshold(
    Operation *op, uint index = 0);
uint64_t getPreviousOpAddress(Operation *op, uint index = 0);
uint64_t getWeightOpAddress(Operation *op);

template<typename T>
void getConv2DOpParam(T &op,
    int &n, int &ic, int &ih, int &iw, int &oc, int &oh, int &ow, int &g,
    int &kh, int &kw, int &sh, int &sw, int &ph, int &pw, int &dh, int &dw,
    bool &with_bias, bool &do_relu);
void getDeConv2DOpParam(tpu::DeConv2DOp &op,
    int &n, int &ic, int &ih, int &iw, int &oc, int &oh, int &ow, int &g,
    int &kh, int &kw, int &sh, int &sw, int &ph, int &pw, int &dh, int &dw,
    bool &with_bias);
void getPool2DOpParam(tpu::Pool2DOp &op,
    bool &is_average_pool, int &n, int &c, int &ih, int &iw, int &oh, int &ow,
    int &kh, int &kw, int &sh, int &sw, int &pt, int &pb, int &pl, int &pr, bool &do_relu);
void getFullyConnectedOpParam(tpu::FullyConnectedOp &op,
    bool &with_transpose, int &m, int &k, int &n,
    bool &with_bias, bool &do_relu);

void getConv2DOpVariadicTensors(tpu::Conv2DOp &op,
    std::vector<std::shared_ptr<std::vector<float> > > &opdT,
    std::shared_ptr<std::vector<float> > &bias,
    std::shared_ptr<std::vector<float> > &rshift,
    std::shared_ptr<std::vector<float> > &multiplier,
    std::shared_ptr<std::vector<float> > &per_channel_info,
    std::shared_ptr<std::vector<float> > &eltwise_input);

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

} // namespace mlir

#endif // MLIR_DIALECT_TPU_OPERATION_SUPPORT_H_
