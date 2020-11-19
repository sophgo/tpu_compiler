/*
 * Copyright (C) Bitmain Technologies Inc.
 * All Rights Reserved.
 */
#include "llvm/Support/Debug.h"
#include "tpuc/GmemAllocator.hpp"
#include "tpuc/TPUOperationSupport.h"

namespace mlir {

#define DEBUG_TYPE "gmem-allocator"

static uint32_t getTensorGmemSize(Operation *op, uint32_t alignment) {
  if (uint32_t size = (uint32_t)getTotalCompressedActivationSize(op))
    return llvm::alignTo(size, alignment);

  uint32_t dsize = 1;
  auto type = op->getResult(0).getType().template cast<TensorType>();
  std::vector<int64_t> shape = type.getShape();
  auto count =
      std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
  auto elementType = type.getElementType();
  if (elementType.isF32()) {
    dsize = sizeof(float);
  } else if (elementType.isInteger(8)) {
    dsize = sizeof(int8_t);
  } else if (elementType.isBF16()) {
    dsize = sizeof(uint16_t);
  } else {
    llvm_unreachable("unsupported data type");
  }
  uint32_t size = (uint32_t)count * dsize;
  // pad to alignment
  if (size % alignment) {
    size = size + alignment - (size % alignment);
  }
  return size;
}

GmemAllocator::GmemAllocator(
    std::map<Operation *, int64_t> &gaddrMap,
    uint32_t alignment)
    : gaddrMap(gaddrMap),
      alignment(alignment) {
  GmemBlock block;
  block.start = 0;
  block.size = 0xFFFFFFFF;
  block.op = nullptr;
  std::list<GmemBlock> snapshot;
  snapshot.push_back(block);
  album.push_back(snapshot);
}

int64_t GmemAllocator::assignGaddr(std::vector<Operation *> &ops,
                                std::map<Operation *, std::vector<uint32_t>> &liveRange,
                                bool neuronMemoryReuse, int64_t baseGaddr) {
  for (auto op : ops) {
    LLVM_DEBUG(llvm::errs() << "loop #" << album.size() - 1 << "\n");
    auto snapshot = album[album.size() - 1];
    allocGmemBlock(snapshot, op);
    if (neuronMemoryReuse) {
      reuseGmemBlock(snapshot, op, liveRange);
    }
    album.push_back(snapshot);
  }

  LLVM_DEBUG(
    int i = 0;
    for (auto snapshot : album) {
      llvm::errs() << "Snapshot idx:" << i++ << "\n";
      int j = 0;
      for (auto &blk : snapshot) {
        llvm::errs() << "\t" << j++ << " "
                    << (blk.op ? blk.op->getName().getStringRef() : llvm::StringRef("null"))
                    << (blk.op ? getOpName(blk.op) : llvm::StringRef("null"))
                    << ", start:" << blk.start << ", size:" << blk.size
                    << ", free:" << (blk.op ? false : true) << "\n";
      }
    }
  );

  backPropagateToAssignGaddr();
  auto totalGmemUsed = updateGmemUsedStatistic(ops);
  // update gaddr map by adding base gaddr.
  for (auto op : ops) {
    gaddrMap[op] += baseGaddr;
  }

  LLVM_DEBUG(
    for (auto op : ops) {
      llvm::errs() << "op:" << op->getName()
                  << ", name:" << getOpName(op)
                  << ", addr:" << gaddrMap[op]
                  << ", baseGaddr:" << baseGaddr
                  << ", size:" << getTensorGmemSize(op, alignment)
                  << ", end:" << gaddrMap[op] + getTensorGmemSize(op, alignment)
                  << ", range:" << liveRange[op][0]
                  << " ~ " << liveRange[op][1]
                  << "\n";
    }
  );
  return totalGmemUsed;
}

void GmemAllocator::reuseGmemBlock(std::list<GmemBlock> &snapshot, Operation *op,
                                   std::map<Operation *, std::vector<uint32_t>> &liveRange) {
  for (auto &blk : snapshot) {
    if (!blk.op) {
      continue;
    }
    // free the block if end position of block's op
    // is same as current op's start position
    if (liveRange[blk.op][1] <= liveRange[op][0]) {
      blk.op = nullptr;
    }
  }
  // merge contiguous free blocks into one
  mergeFreeGmemBlocks(snapshot);
}

void GmemAllocator::allocGmemBlock(std::list<GmemBlock> &snapshot, Operation *op) {
  auto last = --snapshot.end();
  auto selected = last;

  // Policy: just select the free block that has largest size.
  // TODO, we can try other policy here.
  uint32_t max_free_size = 0;
  for (auto iter = snapshot.begin(); iter != last; ++iter) {
    if (!iter->op && iter->size > max_free_size) {
      selected = iter;
      max_free_size = iter->size;
    }
  }

  gaddrMap[op] = -1;
  auto gsize = getTensorGmemSize(op, alignment);

  if (selected->size > gsize) {
    // Occupy this free block firstly.
    // Split the remain memory to anther block,
    // and insert it into snapshot.
    GmemBlock blk;
    blk.start = selected->start + gsize;
    blk.size = selected->size - gsize;
    blk.op = nullptr;

    selected->op = op;
    selected->size = gsize;
    snapshot.insert(++selected, blk);
  } else {
    selected->op = op;
    selected->size = gsize;

    // Enlarge the block to match the size of tensor,
    // and correct the offset of subsequent blocks.
    int64_t offset = selected->start + selected->size;
    while (++selected != snapshot.end()) {
      selected->start = offset;
      offset += selected->size;
    }
  }
}

void GmemAllocator::mergeFreeGmemBlocks(std::list<GmemBlock> &snapshot) {
  auto iter = snapshot.begin();
  while (iter != snapshot.end()) {
    auto cur = iter++;
    while (iter != snapshot.end() && !cur->op && !iter->op) {
      cur->size += iter->size;
      snapshot.erase(iter++);
    }
  }
}

void GmemAllocator::backPropagateToAssignGaddr() {
  for (int i = album.size() - 1; i >= 0; --i) {
    auto &snapshot = album[i];

    int64_t offset = 0;
    for (auto &blk : snapshot) {
      if (!blk.op) {
        blk.start = offset;
        offset += blk.size;
        continue;
      }
      auto op = blk.op;
      // if tensor was allocated, relocate the start offset of block.
      blk.start = (gaddrMap[op] == -1) ? blk.start : gaddrMap[op];

      // if start offset of block is already allocated,
      // relocate it to current offset point.
      if (blk.start <= offset) {
        blk.start = offset;
      }
      offset = blk.start + blk.size;

      if (gaddrMap[op] == -1) {
        gaddrMap[op] = blk.start;
      }
    }
  }
}

int64_t GmemAllocator::updateGmemUsedStatistic(std::vector<Operation *> &ops) {
  int64_t totalNeuronSize = 0;
  int64_t totalGmemUsed = 0;
  for (int i = ops.size() - 1; i >= 0; i--) {
    auto addr_i = gaddrMap[ops[i]];
    auto sz_i = getTensorGmemSize(ops[i], alignment);
    if (totalGmemUsed < addr_i + sz_i) {
      totalGmemUsed = addr_i + sz_i;
    }
    totalNeuronSize += sz_i;
  }

  int32_t reuseRate = 0;
  if (totalNeuronSize) {
    reuseRate = (int32_t)((totalNeuronSize - totalGmemUsed) * 100 / totalNeuronSize);
  }

  llvm::errs() << "Gmem Used: " << totalGmemUsed << "/" << totalNeuronSize
               << ", gmem reused rate:" << reuseRate << "%\n";
  return totalGmemUsed;
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
    auto sz_i = getTensorGmemSize(ops[i], alignment);
    for (int j = 0; j < (int)tmp.size(); j++) {
      auto addr_j = gaddrMap[tmp[j]];
      auto sz_j = getTensorGmemSize(tmp[j], alignment);
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
    int axis = concatOp.axis();
    if (axis != 0) {
      return 0;
    }
    size = getTensorGmemSize(op, alignment);
    gaddrMap[op] = baseGaddr;
  }
  return size;
}

}
