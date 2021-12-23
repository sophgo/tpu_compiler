#include <limits>
#include "tpuc/GmemAllocatorMethod.h"

#define DEBUG_TYPE "gmem-allocator"

namespace mlir {

uint32_t GmemAllocatorMethod::getTensorGmemSize(Operation *op, uint32_t aligment_) {
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
  } else if (elementType.isInteger(16)) {
    dsize = sizeof(int16_t);
  } else if (elementType.isInteger(32)) {
    dsize = sizeof(int32_t);
  } else if (elementType.isBF16()) {
    dsize = sizeof(uint16_t);
  } else {
    llvm_unreachable("unsupported data type");
  }
  uint32_t size = (uint32_t)count * dsize;

  // pad to aligment_
  if (size % aligment_) {
    size = size + aligment_ - (size % aligment_);
  }
  return size;
}

GmemAllocatorMethod::GmemAllocatorMethod(
    std::map<Operation *, int64_t> &gaddrMap, uint32_t aligment)
    : gaddrMap_(gaddrMap), aligment_(aligment) {
  GmemBlock block;
  block.start = 0;
  block.size = 0xFFFFFFFF;
  block.op = nullptr;
  std::list<GmemBlock> snapshot;
  snapshot.emplace_back(block);
  album_.emplace_back(snapshot);
}
GmemAllocatorMethod::~GmemAllocatorMethod() {}

std::string GmemAllocatorMethod::getName() { return name_; }

void GmemAllocatorMethod::reuseGmemBlock(
    std::list<GmemBlock> &snapshot, Operation *op,
    std::map<Operation *, std::vector<uint32_t>> &liveRange) {
  for (auto &blk : snapshot) {
    if (!blk.op) {
      continue;
    }
    // free the block if end position of block's op
    // is same as current op's start position
    if (liveRange[blk.op][1] <= liveRange[op][0] ||
        liveRange[blk.op][0] >= liveRange[op][1]) {
      blk.op = nullptr;
    }
  }
  // merge contiguous free blocks into one
  mergeFreeGmemBlocks(snapshot);
}

int64_t
GmemAllocatorMethod::allocGmemBlock(std::list<GmemBlock> &snapshot,
                                    Operation *op) {
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

  gaddrMap_[op] = -1;
  auto gsize = getTensorGmemSize(op, aligment_);
  auto s_addr = selected->start;

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
  return s_addr;
}

void GmemAllocatorMethod::mergeFreeGmemBlocks(std::list<GmemBlock> &snapshot) {
  auto iter = snapshot.begin();
  while (iter != snapshot.end()) {
    auto cur = iter++;
    while (iter != snapshot.end() && !cur->op && !iter->op) {
      cur->size += iter->size;
      snapshot.erase(iter++);
    }
  }
}

void GmemAllocatorMethod::backPropagateToAssignGaddr() {
  for (int i = album_.size() - 1; i >= 0; --i) {
    auto &snapshot = album_[i];
    int64_t offset = 0;
    for (auto &blk : snapshot) {
      if (!blk.op) {
        blk.start = offset;
        offset += blk.size;
        continue;
      }
      auto op = blk.op;
      // if tensor was allocated, relocate the start offset of block.
      blk.start = (gaddrMap_[op] == -1) ? blk.start : gaddrMap_[op];
      // if start offset of block is already allocated,
      // relocate it to current offset point.
      if (blk.start <= offset) {
        blk.start = offset;
      }
      offset = blk.start + blk.size;

      if (gaddrMap_[op] == -1) {
        gaddrMap_[op] = blk.start;
      }
    }
  }
}

int64_t GmemAllocatorMethod::updateGmemUsedStatistic(std::vector<Operation *> &ops) {
  int64_t totalNeuronSize = 0;
  int64_t totalGmemUsed = 0;
  for (int i = ops.size() - 1; i >= 0; i--) {
    auto addr_i = gaddrMap_[ops[i]];
    auto sz_i = getTensorGmemSize(ops[i], aligment_);
    if (totalGmemUsed < addr_i + sz_i) {
      totalGmemUsed = addr_i + sz_i;
    }
    totalNeuronSize += sz_i;
  }

  int32_t reuseRate = 0;
  if (totalNeuronSize) {
    reuseRate =
        (int32_t)((totalNeuronSize - totalGmemUsed) * 100 / totalNeuronSize);
  }

  llvm::errs() << "GmemAllocMethod:" << name_.c_str() << "  Gmem Used: " << totalGmemUsed << "/" << totalNeuronSize
               << ", gmem reused rate:" << reuseRate << "%\n";
  return totalGmemUsed;
}

GmemAllocLargestFirst::GmemAllocLargestFirst(std::map<Operation *, int64_t> &gaddrMap, uint32_t aligment)
    : GmemAllocatorMethod(gaddrMap, aligment) {
  name_ = "LargestFirstAssign";
}

int64_t GmemAllocLargestFirst::assignGaddr(std::vector<Operation *> &ops,
                    std::map<Operation *, std::vector<uint32_t>> &liveRange,
                    bool neuronMemoryReuse, int64_t baseGaddr) {

  assert(neuronMemoryReuse);
  // sort op:
  // op_set_map : first is the op, second is a vector of ops who's addr relate to
  // the op and order by op's size.
  std::map<Operation *, LineSet> op_set_map;
  LineSet cur_line;
  for (auto &op : ops) {
    uint32_t op_size = getTensorGmemSize(op, aligment_);
    std::shared_ptr<OpAddr> op_addr = std::make_shared<OpAddr>(op, op_size);

    // reuse
    for (auto iter = cur_line.begin(); iter != cur_line.end();) {
      if (liveRange[(*iter)->op][1] <= liveRange[op][0]) {
        cur_line.erase(iter++);
      } else {
        ++iter;
      }
    }
    auto iter = std::find_if(
        cur_line.begin(), cur_line.end(),
        [&op_size](std::shared_ptr<OpAddr> &p) { return p->size < op_size; });
    cur_line.emplace(iter, op_addr);
    op_set_map[op] = cur_line;
  }

  // find largest line
  auto OpSizeSum = [](LineSet &p) {
    uint64_t sum = 0;
    for (auto &op_addr : p) {
      sum += op_addr->size;
    }
    return sum;
  };

  int max_idx = 0;
  uint64_t max_size = 0;
  for (uint32_t i = 0; i < ops.size(); ++i) {
    uint64_t cur_size = OpSizeSum(op_set_map[ops[i]]);
    //llvm::errs() << "line no :" << i << " size:" << cur_size << "\n";
    if (cur_size > max_size) {
      max_size = cur_size;
      max_idx = i;
    }
  }

  // assign addr
  auto snapshot = album_[album_.size() - 1];
  album_.resize(ops.size(), snapshot);

  // first alloc gmem for largest line
  for (auto op_addr : op_set_map[ops[max_idx]]) {
    int64_t s_addr = allocGmemBlock(snapshot, op_addr->op);
    op_addr->start = s_addr;
    op_addr->end = s_addr + op_addr->size;
  }
  album_[max_idx].swap(snapshot);

  auto assign_mem_addr = [&](Operation *op, LineSet &line_set, int32_t idx) {
    LineSet tmp_set;
    for (auto &op_addr : line_set) {
      if (op_addr->end > 0) {
        allocGmemBlock(snapshot, op_addr->op, op_addr->start, op_addr->size);
      } else {
        tmp_set.emplace_back(op_addr);
      }
    }
    for (auto &op_addr : tmp_set) {
      int64_t s_addr = allocGmemBlock(snapshot, op_addr->op);
      op_addr->start = s_addr;
      op_addr->end = s_addr + op_addr->size;
    }
    snapshot.sort([](GmemBlock &a, GmemBlock &b) { return a.start < b.start; });
    album_[idx].swap(snapshot);
  };

  // second alloc gmem from largest line to 0
  for (int32_t i = max_idx - 1; i >= 0; --i) {
    assign_mem_addr(ops[i], op_set_map[ops[i]], i);
  }

  // third alloc gmem for largest line to end
  for (int32_t i = max_idx + 1; i < static_cast<int32_t>(ops.size()); ++i) {
    assign_mem_addr(ops[i], op_set_map[ops[i]], i);
  }

  LLVM_DEBUG(
  int i = 0;
  for (auto snapshot : album_) {
    llvm::errs() << "Snapshot idx:" << i++ << "\n";
    int j = 0;
    for (auto &blk : snapshot) {
      llvm::errs() << "\t" << j++ << " "
                   << (blk.op ? blk.op->getName().getStringRef()
                              : llvm::StringRef("null"))
                   << ":"
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
    gaddrMap_[op] += baseGaddr;
  }

  //LLVM_DEBUG(
  for (auto op : ops) {
    llvm::errs() << "op:" << op->getName() << ", name:" << getOpName(op)
                 << ", addr:" << gaddrMap_[op] << ", baseGaddr:" << baseGaddr
                 << ", size:" << getTensorGmemSize(op, aligment_)
                 << ", end:" << gaddrMap_[op] + getTensorGmemSize(op, aligment_)
                 << ", range:" << liveRange[op][0] << " ~ " << liveRange[op][1]
                 << "\n";
  }
  //);
  return totalGmemUsed;
}

void GmemAllocLargestFirst::allocGmemBlock(std::list<GmemBlock> &snapshot, Operation *op,
                    int64_t s_addr, uint32_t size) {
  auto last = --snapshot.end();
  auto selected = last;

  for (auto iter = snapshot.begin(); iter != last; ++iter) {
    if (iter->start > s_addr) {
      assert(iter->start >= s_addr + size);
      selected = std::prev(iter, 1);
      assert(!selected->op);
      break;
    }
  }
  assert(selected->op != op);

  gaddrMap_[op] = -1;

  uint64_t pre_size = 0;
  if (selected->start < s_addr) {
    GmemBlock blk;
    blk.start = selected->start;
    blk.size = s_addr - selected->start;
    blk.op = nullptr;
    pre_size = blk.size;
    snapshot.insert(selected, blk);
  }

  selected->size -= pre_size;
  selected->start = s_addr;
  selected->op = op;

  if (selected->size > size) {
    GmemBlock blk;
    blk.start = s_addr + size;
    blk.size = selected->size - size;
    blk.op = nullptr;
    selected->size = size;
    snapshot.insert(++selected, blk);
  }
}

int64_t GmemAllocLargestFirst::allocGmemBlock(std::list<GmemBlock> &snapshot, Operation *op) {
  auto last = --snapshot.end();
  auto selected = last;
  auto gsize = getTensorGmemSize(op, aligment_);

  // Policy: just select the free block that has largest size.
  // TODO, we can try other policy here.
  for (auto iter = snapshot.begin(); iter != snapshot.end(); ++iter) {
    if (!iter->op && iter->size >= gsize) {
      selected = iter;
      break;
    }
  }

  gaddrMap_[op] = -1;
  auto s_addr = selected->start;

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
  return s_addr;
}

GmemAllocFitFirst::GmemAllocFitFirst(std::map<Operation *, int64_t> &gaddrMap, uint32_t aligment)
    : GmemAllocatorMethod(gaddrMap, aligment) {
  name_ = "FitFirstAssign";
}

int64_t GmemAllocFitFirst::assignGaddr(std::vector<Operation *> &ops,
                    std::map<Operation *, std::vector<uint32_t>> &liveRange,
                    bool neuronMemoryReuse, int64_t baseGaddr) {
  for (auto op : ops) {
    LLVM_DEBUG(llvm::errs() << "loop #" << album_.size() - 1 << "\n");
    auto snapshot = album_[album_.size() - 1];
    if (neuronMemoryReuse) {
      reuseGmemBlock(snapshot, op, liveRange);
    }
    allocGmemBlock(snapshot, op);
    album_.push_back(snapshot);
  }

  LLVM_DEBUG(
  int i = 0;
  for (auto snapshot : album_) {
    llvm::errs() << "Snapshot idx:" << i++ << "\n";
    int j = 0;
    for (auto &blk : snapshot) {
      llvm::errs() << "\t" << j++ << " "
                   << (blk.op ? blk.op->getName().getStringRef()
                              : llvm::StringRef("null"))
                   << ":"
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
    gaddrMap_[op] += baseGaddr;
  }

  //LLVM_DEBUG(
  for (auto op : ops) {
    llvm::errs() << "op:" << op->getName() << ", name:" << getOpName(op)
                 << ", addr:" << gaddrMap_[op] << ", baseGaddr:" << baseGaddr
                 << ", size:" << getTensorGmemSize(op, aligment_)
                 << ", end:" << gaddrMap_[op] + getTensorGmemSize(op, aligment_)
                 << ", range:" << liveRange[op][0] << " ~ " << liveRange[op][1]
                 << "\n";
  }
  //);
  return totalGmemUsed;
}

GmemAllocLargeOrder::GmemAllocLargeOrder(std::map<Operation *, int64_t> &gaddrMap, uint32_t aligment)
    : GmemAllocatorMethod(gaddrMap, aligment) {
  name_ = "LargeOrderAssign";
}

int64_t GmemAllocLargeOrder::assignGaddr(std::vector<Operation *> &ops,
                    std::map<Operation *, std::vector<uint32_t>> &liveRange,
                    bool neuronMemoryReuse, int64_t baseGaddr) {

  assert(neuronMemoryReuse);
  // sort op:
  // op_set_map : first is the op, second is a vector of ops who's addr relate to
  // the op and order by op's size.
  std::vector<LineSet> op_sets;
  LineSet cur_line;
  for (auto &op : ops) {
    uint32_t op_size = getTensorGmemSize(op, aligment_);
    std::shared_ptr<OpAddr> op_addr = std::make_shared<OpAddr>(op, op_size);

    // reuse
    for (auto iter = cur_line.begin(); iter != cur_line.end();) {
      if (liveRange[(*iter)->op][1] <= liveRange[op][0]) {
        cur_line.erase(iter++);
      } else {
        ++iter;
      }
    }
    auto iter = std::find_if(
        cur_line.begin(), cur_line.end(),
        [&op_size](std::shared_ptr<OpAddr> &p) { return p->size < op_size; });
    cur_line.emplace(iter, op_addr);
    op_sets.emplace_back(cur_line);
  }

  // sort op_sets order by gmem size
  auto OpSizeSum = [](LineSet &p) {
    uint64_t sum = 0;
    for (auto &op_addr : p) {
      sum += op_addr->size;
    }
    return sum;
  };

  std::vector<std::pair<uint32_t, uint32_t> > op_size_vector;

  for (uint32_t i = 0; i < op_sets.size(); ++i) {
    op_size_vector.emplace_back(i, OpSizeSum(op_sets[i]));
  }

  std::sort( op_size_vector.begin(), op_size_vector.end(),
      [](std::pair<uint32_t, uint32_t> &a, std::pair<uint32_t, uint32_t> &b) {
        return a.second > b.second;
      }
  );

  auto assign_mem_addr = [&](LineSet &line_set, int32_t idx,
                            std::list<GmemBlock> &snapshot,
                            std::vector<std::list<GmemBlock>> &album) {
    LineSet tmp_set;
    for (auto &op_addr : line_set) {
      if (op_addr->end > 0) {
        allocGmemBlock(snapshot, op_addr->op, op_addr->start, op_addr->size);
      } else {
        tmp_set.emplace_back(op_addr);
      }
    }
    for (auto &op_addr : tmp_set) {
      int64_t s_addr = allocGmemBlock(snapshot, op_addr->op);
      op_addr->start = s_addr;
      op_addr->end = s_addr + op_addr->size;
    }
    snapshot.sort([](GmemBlock &a, GmemBlock &b) { return a.start < b.start; });
    album[idx].swap(snapshot);
    return album[idx].cbegin()->start;
  };

  uint32_t min_gmem_idx = 0;
  int64_t min_gmem_size = 0;
  for (uint32_t i = 0; i < op_size_vector.size(); ++i) {
    // assign addr
    GmemBlock block;
    block.start = 0;
    block.size = 0xFFFFFFFF;
    block.op = nullptr;
    std::list<GmemBlock> snapshot;
    snapshot.emplace_back(block);
    std::vector<std::list<GmemBlock> > tmp_album;
    tmp_album.resize(op_sets.size(), snapshot);

    uint32_t max_idx = op_size_vector[i].first;
    int64_t cur_size = 0;

    // first alloc gmem from largest line to 0
    for (int32_t j = max_idx; j >= 0; --j) {
      int64_t block_size = assign_mem_addr(op_sets[j], j, snapshot, tmp_album);
      if (block_size > cur_size) {
        cur_size = block_size;
      }
    }

    // second alloc gmem for largest line to end
    for (uint32_t j = max_idx + 1; j < op_sets.size(); ++j) {
      int64_t block_size = assign_mem_addr(op_sets[j], j, snapshot, tmp_album);
      if (block_size > cur_size) {
        cur_size = block_size;
      }
    }

    if (min_gmem_size == 0 || min_gmem_size > cur_size) {
      min_gmem_size = cur_size;
      min_gmem_idx = i;
      album_.swap(tmp_album);
    }
  }

  llvm::errs() << "alloc from " << min_gmem_idx << "th largest\n"; 

  LLVM_DEBUG(
  int i = 0;
  for (auto snapshot : album_) {
    llvm::errs() << "Snapshot idx:" << i++ << "\n";
    int j = 0;
    for (auto &blk : snapshot) {
      llvm::errs() << "\t" << j++ << " "
                   << (blk.op ? blk.op->getName().getStringRef()
                              : llvm::StringRef("null"))
                   << ":"
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
    gaddrMap_[op] += baseGaddr;
  }

  //LLVM_DEBUG(
  for (auto op : ops) {
    llvm::errs() << "op:" << op->getName() << ", name:" << getOpName(op)
                 << ", addr:" << gaddrMap_[op] << ", baseGaddr:" << baseGaddr
                 << ", size:" << getTensorGmemSize(op, aligment_)
                 << ", end:" << gaddrMap_[op] + getTensorGmemSize(op, aligment_)
                 << ", range:" << liveRange[op][0] << " ~ " << liveRange[op][1]
                 << "\n";
  }
  //);
  return totalGmemUsed;
}

void GmemAllocLargeOrder::allocGmemBlock(std::list<GmemBlock> &snapshot, Operation *op,
                    int64_t s_addr, uint32_t size) {
  auto last = --snapshot.end();
  auto selected = last;

  for (auto iter = snapshot.begin(); iter != last; ++iter) {
    if (iter->start > s_addr) {
      assert(iter->start >= s_addr + size);
      selected = std::prev(iter, 1);
      assert(!selected->op);
      break;
    }
  }
  assert(selected->op != op);

  gaddrMap_[op] = -1;

  uint64_t pre_size = 0;
  if (selected->start < s_addr) {
    GmemBlock blk;
    blk.start = selected->start;
    blk.size = s_addr - selected->start;
    blk.op = nullptr;
    pre_size = blk.size;
    snapshot.insert(selected, blk);
  }

  selected->size -= pre_size;
  selected->start = s_addr;
  selected->op = op;

  if (selected->size > size) {
    GmemBlock blk;
    blk.start = s_addr + size;
    blk.size = selected->size - size;
    blk.op = nullptr;
    selected->size = size;
    snapshot.insert(++selected, blk);
  }
}

int64_t GmemAllocLargeOrder::allocGmemBlock(std::list<GmemBlock> &snapshot, Operation *op) {
  auto last = --snapshot.end();
  auto selected = last;
  auto gsize = getTensorGmemSize(op, aligment_);

  // Policy: just select the free block that has largest size.
  // TODO, we can try other policy here.
  for (auto iter = snapshot.begin(); iter != snapshot.end(); ++iter) {
    if (!iter->op && iter->size >= gsize) {
      selected = iter;
      break;
    }
  }

  gaddrMap_[op] = -1;
  auto s_addr = selected->start;

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
  return s_addr;
}

GmemAllocOpSizeOrder::GmemAllocOpSizeOrder(std::map<Operation *, int64_t> &gaddrMap, uint32_t aligment)
    : GmemAllocatorMethod(gaddrMap, aligment) {
  name_ = "OpSizeOrderAssign";
}

int64_t GmemAllocOpSizeOrder::assignGaddr(std::vector<Operation *> &ops,
                    std::map<Operation *, std::vector<uint32_t>> &liveRange,
                    bool neuronMemoryReuse, int64_t baseGaddr) {

  std::list<std::shared_ptr<OpAddr>> op_list;
  std::list<std::shared_ptr<OpAddr>> allocated_op_list;
  assert(neuronMemoryReuse);
  for (auto &op : ops) {
    uint32_t op_size = getTensorGmemSize(op, aligment_);
    std::shared_ptr<OpAddr> op_addr = std::make_shared<OpAddr>(
        op, op_size, liveRange[op][0], liveRange[op][1]);
    op_list.emplace_back(op_addr);
    gaddrMap_[op] = -1;
  }

  op_list.sort([](std::shared_ptr<OpAddr> &a, std::shared_ptr<OpAddr> &b) {
    return a->size >= b->size;
  });


  int64_t total_consumption = 0;
  for (auto &op_addr : op_list) {
    int64_t prev_offset = 0;
    int64_t best_offset = -1;
    int64_t smallest_gap = std::numeric_limits<int64_t>::max();
    for (auto &allocated_op_addr : allocated_op_list) {
      uint32_t max_first_pos = std::max(op_addr->first_pos, allocated_op_addr->first_pos);
      uint32_t min_last_pos = std::min(op_addr->end_pos, allocated_op_addr->end_pos);
      if (max_first_pos < min_last_pos) {
        int64_t gap = allocated_op_addr->start - prev_offset;
        if (gap >= op_addr->size && gap < smallest_gap) {
          smallest_gap = gap;
          best_offset = prev_offset;
        }
        prev_offset = std::max(prev_offset, allocated_op_addr->end);
      }
    }
    if (best_offset == -1) {
      best_offset = prev_offset;
    }
    op_addr->start = best_offset;
    op_addr->end = op_addr->start + op_addr->size;
    total_consumption = std::max(total_consumption, op_addr->end);
    auto iter = std::find_if(
        allocated_op_list.begin(), allocated_op_list.end(),
        [&op_addr](std::shared_ptr<OpAddr> &p) { return p->start >= op_addr->start; });
    allocated_op_list.emplace(iter, op_addr);
  }

  int64_t totalNeuronSize = 0;
  for (auto &op_addr : allocated_op_list) {
    if (gaddrMap_[op_addr->op] == -1) {
      gaddrMap_[op_addr->op] = op_addr->start;
    }
    totalNeuronSize += op_addr->size;
  }

  int32_t reuseRate = 0;
  if (totalNeuronSize) {
    reuseRate =
        (int32_t)((totalNeuronSize - total_consumption) * 100 / totalNeuronSize);
  }

  llvm::errs() << "GmemAllocMethod:" << name_.c_str() << "  Gmem Used: " << total_consumption << "/" << totalNeuronSize
               << ", gmem reused rate:" << reuseRate << "%\n";

  // LLVM_DEBUG(
  for (auto op : ops) {
    llvm::errs() << "op:" << op->getName() << ", name:" << getOpName(op)
                 << ", addr:" << gaddrMap_[op] << ", baseGaddr:" << baseGaddr
                 << ", size:" << getTensorGmemSize(op, aligment_)
                 << ", end:" << gaddrMap_[op] + getTensorGmemSize(op, aligment_)
                 << ", range:" << liveRange[op][0] << " ~ " << liveRange[op][1]
                 << "\n";
  }
  //);

  for (auto &op_addr : allocated_op_list) {
    // update gaddr map by adding base gaddr.
    gaddrMap_[op_addr->op] += baseGaddr;
  }

  return total_consumption;
}
} //namespace mlir