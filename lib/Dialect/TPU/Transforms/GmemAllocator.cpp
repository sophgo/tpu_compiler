/*
 * Copyright (C) Bitmain Technologies Inc.
 * All Rights Reserved.
 */
#include "llvm/Support/Debug.h"
#include "tpuc/GmemAllocator.hpp"
#include "tpuc/TPUOperationSupport.h"

namespace mlir {

GmemAllocator::GmemAllocator(
    std::map<Operation *, int64_t> &gaddrMap,
    uint32_t alignment)
    : gaddrMap_(gaddrMap),
      alignment(alignment) {
}

void GmemAllocator::markGmemReusedOp(
    std::vector<Operation *> &ops,
    std::map<Operation *, int64_t> &gaddrMap,
    std::set<Operation *> &gmemReusedSet,
    uint32_t alignment) {

  std::vector<Operation *> tmp;
  for (int i = ops.size() - 1; i >= 0; i--) {
    if (gaddrMap.find(ops[i]) == gaddrMap.end())
      continue;

    auto addr_i = gaddrMap[ops[i]];
    auto sz_i = GmemAllocatorMethod::getTensorGmemSize(ops[i], alignment);
    for (int j = 0; j < (int)tmp.size(); j++) {
      auto addr_j = gaddrMap[tmp[j]];
      auto sz_j = GmemAllocatorMethod::getTensorGmemSize(tmp[j], alignment);
      auto start = std::min(addr_i, addr_j);
      auto end = std::max(addr_i + sz_i, addr_j + sz_j);
      // memory overlap
      if (end - start < sz_i + sz_j) {
        gmemReusedSet.insert(ops[i]);
      }
    }
    tmp.push_back(ops[i]);
  }
}

int64_t GmemAllocator::assignSpecifiedGmemToOp(
                                       Operation *op,
                                       std::map<Operation *, int64_t> &gaddrMap,
                                       int64_t baseGaddr,
                                       uint32_t alignment) {
  int64_t size = 0;
  if (auto concatOp = dyn_cast_or_null<tpu::TG_ConcatNOp>(op)) {
    size = GmemAllocatorMethod::getTensorGmemSize(op, alignment);
    gaddrMap[op] = baseGaddr;
  }
  return size;
}


void GmemAllocator::registerMethod(std::string method_name, bool reuse) {
  if (reuse) {
    reuse_methods_.emplace_back(method_name);
  } else {
    methods_.emplace_back(method_name);
  }
}

void GmemAllocator::registerAllMethod() {
  registerMethod("FitFirstAssign", true);
  registerMethod("FitFirstAssign", false);
  registerMethod("LargestFirstAssign", true);
}

int64_t GmemAllocator::assignGaddr(
    std::vector<Operation *> &ops,
    std::map<Operation *, std::vector<uint32_t>> &liveRange,
    bool neuronMemoryReuse, int64_t baseGaddr) {
  if (ops.empty()) {
    llvm::errs() << "Warning input ops is empty!\n";
    return 0;
  }

  if (!reuse_methods_.size() && !methods_.size()) {
    registerAllMethod();
  }

  std::vector<std::string> *cur_methods;
  if (neuronMemoryReuse) {
    cur_methods = &reuse_methods_;
  } else {
    cur_methods = &methods_;
  }

  std::vector<std::unique_ptr<GmemAllocatorMethod> > alloc_methods;
  for (auto& name : *cur_methods) {
    auto p = GmemAllocatorMethodFactory::makeMethod(name, gaddrMap_, alignment);
    if (p) {
      alloc_methods.emplace_back(p);
    } else {
      assert(0);
    }
  }

  int64_t min_gmem_size = 0;
  int idx = 0;
  for (uint32_t i = 0; i < alloc_methods.size(); ++i) {
    int64_t gmem_size = alloc_methods[i]->assignGaddr(ops, liveRange, neuronMemoryReuse, baseGaddr);
    if (gmem_size < min_gmem_size || min_gmem_size == 0) {
      min_gmem_size = gmem_size;
      idx = i;
    }
  }

  llvm::errs() << "GmemAllocator use " << alloc_methods[idx]->getName() << "\n";
  gaddrMap_.swap(alloc_methods[idx]->gaddrMap_);
  return min_gmem_size;
}
}