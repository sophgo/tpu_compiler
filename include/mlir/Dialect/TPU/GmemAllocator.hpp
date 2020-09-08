/*
 * Copyright (C) Bitmain Technologies Inc.
 * All Rights Reserved.
 */
#ifndef GMEM_ALLOCATOR_H_
#define GMEM_ALLOCATOR_H_

#include <set>
#include <list>
#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/TPUTensorSupport.h"


namespace mlir {

class GmemBlock {
public:
  int64_t start;
  uint64_t size;
  Operation *op;
};

class GmemAllocator {
public:
  GmemAllocator(
      std::map<Operation *, int64_t> &gaddrMap,
      uint32_t alignment = 16);
  int64_t assignGaddr(
      std::vector<Operation *> &ops,
      std::map<Operation *, std::vector<uint32_t>> &liveRange,
      bool neuronMemoryReuse, int64_t baseGaddr);
  static void markGmemReusedOp(
      std::vector<Operation *> &ops,
      std::map<Operation *, int64_t> &gaddrMap,
      std::set<Operation *> &gmemReusedSet,
      uint32_t alignment);

  std::map<Operation *, int64_t> &gaddrMap;

private:
  uint32_t alignment;
  std::vector<std::list<GmemBlock>> album;

  void reuseGmemBlock(
      std::list<GmemBlock> &snapshot, Operation *op,
      std::map<Operation *, std::vector<uint32_t>> &liveRange);
  void allocGmemBlock(std::list<GmemBlock> &snapshot, Operation *op);
  void mergeFreeGmemBlocks(std::list<GmemBlock> &snapshot);
  void backPropagateToAssignGaddr();
  int64_t updateGmemUsedStatistic(std::vector<Operation *> &ops);
};

} // namespace mlir
#endif
