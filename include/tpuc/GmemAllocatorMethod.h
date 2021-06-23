#pragma once

#include <set>
#include <list>
#include "tpuc/Dialect/TPU/TPUDialect.h"
#include "tpuc/TPUOperationSupport.h"
#include "tpuc/TPUTensorSupport.h"

namespace mlir {

class GmemBlock {
public:
  int64_t start;
  uint64_t size;
  Operation *op;
};

class GmemAllocatorMethod {
public:
  GmemAllocatorMethod(std::map<Operation *, int64_t> &gaddrMap, uint32_t aligment);
  virtual ~GmemAllocatorMethod();

  virtual std::string getName();

  virtual int64_t assignGaddr(
      std::vector<Operation *> &ops,
      std::map<Operation *, std::vector<uint32_t>> &liveRange,
      bool neuronMemoryReuse, int64_t baseGaddr) = 0;

  virtual void reuseGmemBlock(
      std::list<GmemBlock> &snapshot, Operation *op,
      std::map<Operation *, std::vector<uint32_t>> &liveRange);

  virtual int64_t allocGmemBlock(std::list<GmemBlock> &snapshot,
                                 Operation *op);

  virtual void mergeFreeGmemBlocks(std::list<GmemBlock> &snapshot);

  virtual void backPropagateToAssignGaddr();

  virtual int64_t updateGmemUsedStatistic(std::vector<Operation *> &ops);

  static uint32_t getTensorGmemSize(Operation *op, uint32_t aligment_);

public:
  std::map<Operation *, int64_t> gaddrMap_;

protected:
  std::string name_;
  uint32_t aligment_;
  std::vector<std::list<GmemBlock> > album_;
};

class GmemAllocLargestFirst : public GmemAllocatorMethod {
public:
  struct OpAddr {
    Operation *op;
    int64_t start = 0;
    int64_t end = 0;
    uint32_t size = 0;
    OpAddr(Operation *_op, uint32_t _size) {
      op = _op;
      size = _size;
    }
  };
  typedef std::list<std::shared_ptr<OpAddr>> LineSet;

public:
  GmemAllocLargestFirst(std::map<Operation *, int64_t> &gaddrMap,
                      uint32_t aligment);

  int64_t assignGaddr(std::vector<Operation *> &ops,
                      std::map<Operation *, std::vector<uint32_t>> &liveRange,
                      bool neuronMemoryReuse, int64_t baseGaddr) override;

  void allocGmemBlock(std::list<GmemBlock> &snapshot, Operation *op,
                      int64_t s_addr, uint32_t size);

  int64_t allocGmemBlock(std::list<GmemBlock> &snapshot,
                                         Operation *op) override;
};

class GmemAllocFitFirst : public GmemAllocatorMethod {
public:
  GmemAllocFitFirst(std::map<Operation *, int64_t> &gaddrMap,
                    uint32_t aligment);

  int64_t assignGaddr(std::vector<Operation *> &ops,
                      std::map<Operation *, std::vector<uint32_t>> &liveRange,
                      bool neuronMemoryReuse, int64_t baseGaddr) override;
};

class GmemAllocatorMethodFactory {
public:
  static GmemAllocatorMethod *
  makeMethod(std::string method_name, std::map<Operation *, int64_t> &gaddrMap,
             uint32_t aligment) {
    if (method_name == "LargestFirstAssign") {
      return static_cast<GmemAllocatorMethod*>(new GmemAllocLargestFirst(gaddrMap, aligment));
    } else if (method_name == "FitFirstAssign") {
      return static_cast<GmemAllocatorMethod*>(new GmemAllocFitFirst(gaddrMap, aligment));
    } else {
      assert(0);
      return nullptr;
    }
  }
};
}
