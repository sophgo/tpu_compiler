/*
 * Copyright (C) Bitmain Technologies Inc.
 * All Rights Reserved.
 */
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/TPU/GmemAllocator.hpp"

namespace mlir {

#define DEBUG_TYPE "gmem-allocator"

GmemAllocator::GmemAllocator(uint32_t alignment)
    : alignment(alignment) {
  GmemBlock block;
  block.start = 0;
  block.size = 0xFFFFFFFF;
  block.op = nullptr;
  block.busy = false;
  std::list<GmemBlock> snapshot;
  snapshot.push_back(block);
  album.push_back(snapshot);
}

void GmemAllocator::assignGaddr(std::vector<Operation *> &ops,
                                std::map<Operation *, std::vector<uint32_t>> &liveRange,
                                bool neuronMemoryReuse) {
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
                    << ", busy:" << blk.busy << "\n";
      }
    }
  );

  backPropagateToAssignGaddr();
  updateGmemReusedOpSet(ops);

  LLVM_DEBUG(
    for (auto op : ops) {
      auto reused = (gaddrReusedSet.find(op) != gaddrReusedSet.end());
      llvm::errs() << "op:" << op->getName()
                  << ", name:" << getOpName(op)
                  << ", addr:" << gaddrMap[op]
                  << ", size:" << getTensorGmemSize(op)
                  << ", end:" << gaddrMap[op] + getTensorGmemSize(op)
                  << ", reused:" << reused
                  << ", range:" << liveRange[op][0]
                  << " ~ " << liveRange[op][1]
                  << "\n";
    }
  );
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
      blk.busy = false;
    }
  }
  // merge contiguous free blocks into one
  mergeFreeGmemBlocks(snapshot);
}

uint32_t GmemAllocator::getTensorGmemSize(Operation *op) {
  uint32_t dsize = 1;
  auto type = op->getResult(0)->getType().template cast<TensorType>();
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

void GmemAllocator::allocGmemBlock(std::list<GmemBlock> &snapshot, Operation *op) {
  auto last = --snapshot.end();
  auto selected = last;

  // Policy: just select the free block that has largest size.
  // TODO, we can try other policy here.
  uint32_t max_free_size = 0;
  for (auto iter = snapshot.begin(); iter != last; ++iter) {
    if (!iter->busy && iter->size > max_free_size) {
      selected = iter;
      max_free_size = iter->size;
    }
  }

  gaddrMap[op] = -1;
  auto gsize = getTensorGmemSize(op);

  if (selected->size > gsize) {
    // Occupy this free block firstly.
    // Split the remain memory to anther block,
    // and insert it into snapshot.
    GmemBlock blk;
    blk.start = selected->start + gsize;
    blk.size = selected->size - gsize;
    blk.op = nullptr;
    blk.busy = false;

    selected->op = op;
    selected->size = gsize;
    selected->busy = true;
    snapshot.insert(++selected, blk);
  } else {
    selected->op = op;
    selected->size = gsize;
    selected->busy = true;

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
    while (iter != snapshot.end() && !cur->busy && !iter->busy) {
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
      if (!blk.busy) {
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

void GmemAllocator::updateGmemReusedOpSet(std::vector<Operation *> &ops) {
  int64_t totalNeuronSize = 0;
  int64_t totalNeuronMemorySize = 0;
  std::vector<Operation *> tmp;
  for (int i = ops.size() - 1; i >= 0; i--) {
    auto addr_i = gaddrMap[ops[i]];
    auto sz_i = getTensorGmemSize(ops[i]);
    for (int j = 0; j < (int)tmp.size(); j++) {
      auto addr_j = gaddrMap[tmp[j]];
      auto sz_j = getTensorGmemSize(tmp[j]);
      auto start = std::min(addr_i, addr_j);
      auto end = std::max(addr_i + sz_i, addr_j + sz_j);
      // memory overlap
      if (end - start < sz_i + sz_j) {
        gaddrReusedSet.insert(ops[i]);
      }
    }
    if (totalNeuronMemorySize < addr_i + sz_i) {
      totalNeuronMemorySize = addr_i + sz_i;
    }
    totalNeuronSize += sz_i;

    tmp.push_back(ops[i]);
  }
  llvm::errs() << "Gmem Used: " << totalNeuronMemorySize
               << "/" << totalNeuronSize << ", reused rate:"
               << ((totalNeuronSize - totalNeuronMemorySize) * 100 / totalNeuronSize)
               << "%\n";
}

}