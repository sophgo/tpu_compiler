/*
 * Copyright (C) Bitmain Technologies Inc.
 * All Rights Reserved.
 */
#ifndef GMEM_ALLOCATOR_H_
#define GMEM_ALLOCATOR_H_

#include <set>
#include <list>
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"
#include "GmemAllocatorMethod.h"

namespace mlir {

class GmemAllocator {
public:
  GmemAllocator(
      std::map<Operation *, int64_t> &gaddrMap,
      uint32_t alignment = 16);
  void registerMethod(std::string method_name, bool reuse);
  void registerAllMethod();
  int64_t assignGaddr(
      std::vector<Operation *> &ops,
      std::map<Operation *, std::vector<uint32_t>> &liveRange,
      bool neuronMemoryReuse, int64_t baseGaddr);
  static void markGmemReusedOp(
      std::vector<Operation *> &ops,
      std::map<Operation *, int64_t> &gaddrMap,
      std::set<Operation *> &gmemReusedSet,
      uint32_t alignment);
  static int64_t assignSpecifiedGmemToOp(
      Operation *op,
      std::map<Operation *, int64_t> &gaddrMap,
      int64_t baseGaddr,
      uint32_t alignment);

  std::map<Operation *, int64_t> &gaddrMap_;
  uint32_t alignment;
private:
  std::vector<std::string> reuse_methods_;
  std::vector<std::string> methods_;
};

} // namespace mlir
#endif
