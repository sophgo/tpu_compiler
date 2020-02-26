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

#ifndef MLIR_DIALECT_TPU_TENSOR_SUPPORT_H_
#define MLIR_DIALECT_TPU_TENSOR_SUPPORT_H_

#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Support/TensorFile.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {

/******************************************************************************
 * Tensor helpers
 *****************************************************************************/

bool isTensorNone(Value *);
int64_t getTensorSize(Value *value);
std::vector<int64_t> getTensorShape(Value *value);
void getTensorShapeAndSize(Value *value, std::vector<int64_t> &shape,
                           int64_t &size);
void getNCHW(std::vector<int64_t> &shape, int64_t &n, int64_t &c, int64_t &h,
             int64_t &w);

/******************************************************************************
 * Weight helpers
 *****************************************************************************/

Value* getWeightFileValue(Operation *op);
TensorFile* getWeightTensorFile(Operation *op);

template<typename T>
std::unique_ptr<std::vector<T> > readAndDeleteWeightTensor(
    Value *opd, TensorFile *wTF);

template<typename T>
void addWeightTensorAndUpdateWeightOp(Value* opd,
    StringRef suffix, std::vector<T> &weight, std::vector<int64_t> &shape,
    StringRef storageType, TensorFile *wTF);

template<typename T>
Value* addWeightTensorAndCreateWeightOp(Operation *op,
    StringRef suffix, std::vector<T> &weight,
    std::vector<int64_t> &shape, StringRef storageType,
    TensorFile *wTF, Value *wFV);

} // namespace mlir

#endif // MLIR_DIALECT_TPU_TENSOR_SUPPORT_H_
