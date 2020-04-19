//===- TPUCompress.h - TPU Compression  -------------------------*- C++ -*-===//
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
// This header file defines prototypes that expose compression.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TPU_COMPRESS_UTIL_H_
#define MLIR_DIALECT_TPU_COMPRESS_UTIL_H_

#include "mlir/Support/LLVM.h"

// TPU DMA engine supports compression and decompression during data transfer
// between global memory and local memory.
// Compiler compresses the tiled weight and update the weight file in advanced.
// The backend generates the DMA command which transfers the compressed weight.
// The DMA engine decompresses the weight before writing to the local memory.

namespace mlir {

struct CompressCommandInfo {
  bool signedness;
  bool is_bfloat16;
  uint8_t bias0;
  uint8_t bias1;
  bool zero_guard_en;
};

//
//  dataType
//    0: 8bit
//    1: 16bit
int getCompressedDataSize(int unCompressedDatasize, int dataType);

void getCompressParameter(
    const uint8_t *ibuf, size_t isz, bool signedness, bool isBfloat16,
    CompressCommandInfo *cmd_info);

void compressInt8Data(
    const uint8_t *ibuf, int isz, uint8_t *obuf, int *osz,
    CompressCommandInfo *cmd_info);


// unit test
void testCompress(void);

} // namespace mlir

#endif // MLIR_DIALECT_TPU_COMPRESS_UTIL_H_
