//===- TPUDialect.h - MLIR Dialect for TPU --------------*- C++ -*-===//
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
// This file defines the GPU kernel-related operations and puts them in the
// corresponding dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TPU_TPUDIALECT_H
#define MLIR_DIALECT_TPU_TPUDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
class FuncOp;

namespace tpu {

/// The dialect containing TPU launching operations and related
/// facilities.
class TPUDialect : public Dialect {
public:
  /// Create the dialect in the given `context`.
  TPUDialect(MLIRContext *context);

  /// Get the canonical string name of the dialect.
  static StringRef getDialectName();
};

/// TPU Fusion operation, take tile size as arguments.
//class LaunchOp : public Op<LaunchOp, OpTrait::AtLeastNOperands<6>::Impl,
//                           OpTrait::ZeroResult, OpTrait::IsIsolatedFromAbove> {
//public:
//  using Op::Op;

//  static void build(Builder *builder, OperationState *result, Value *gridSizeX,
//                    Value *gridSizeY, Value *gridSizeZ, Value *blockSizeX,
//                    Value *blockSizeY, Value *blockSizeZ,
//                    ArrayRef<Value *> operands);

  /// Get the kernel region.
//  Region &getBody();

  /// Get the SSA values corresponding to kernel block identifiers.
  //KernelDim3 getBlockIds();
  /// Get the SSA values corresponding to kernel thread identifiers.
  //KernelDim3 getThreadIds();
  /// Get the SSA values corresponding to kernel grid size.
  //KernelDim3 getGridSize();
  /// Get the SSA values corresponding to kernel block size.
  //KernelDim3 getBlockSize();
  /// Get the operand values passed as kernel arguments.
  //operand_range getKernelOperandValues();
  /// Get the operand types passed as kernel arguments.
  //operand_type_range getKernelOperandTypes();

  /// Get the SSA values passed as operands to specify the grid size.
  //KernelDim3 getGridSizeOperandValues();
  /// Get the SSA values passed as operands to specify the block size.
  //KernelDim3 getBlockSizeOperandValues();

  /// Get the SSA values of the kernel arguments.
  //llvm::iterator_range<Block::args_iterator> getKernelArguments();

  //LogicalResult verify();

  /// Custom syntax support.
  //void print(OpAsmPrinter *p);
  //static ParseResult parse(OpAsmParser *parser, OperationState *result);

  //static StringRef getOperationName() { return "gpu.launch"; }

  /// Erase the `index`-th kernel argument.  Both the entry block argument and
  /// the operand will be dropped.  The block argument must not have any uses.
  //void eraseKernelArgument(unsigned index);

  /// Append canonicalization patterns to `results`.
  //static void getCanonicalizationPatterns(OwningRewritePatternList &results,
  //                                        MLIRContext *context);

//private:
  //static StringRef getBlocksKeyword() { return "blocks"; }
  //static StringRef getThreadsKeyword() { return "threads"; }
  //static StringRef getArgsKeyword() { return "args"; }

  /// The number of launch configuration operands, placed at the leading
  /// positions of the operand list.
  //static constexpr unsigned kNumConfigOperands = 6;

  /// The number of region attributes containing the launch configuration,
  /// placed in the leading positions of the argument list.
  //static constexpr unsigned kNumConfigRegionAttributes = 12;
//};

#define GET_OP_CLASSES
#include "mlir/Dialect/TPU/TPUOps.h.inc"

} // end namespace tpu
} // end namespace mlir

#endif // MLIR_DIALECT_TPU_TPUDIALECT_H
