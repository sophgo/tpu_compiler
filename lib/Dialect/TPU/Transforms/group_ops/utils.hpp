#ifndef CLUSTER_UTILS_H
#define CLUSTER_UTILS_H

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "mlir/Dialect/TPU/TPUDialect.h"
#include "mlir/Dialect/TPU/Passes.h"
#include "mlir/Dialect/TPU/TPUOperationSupport.h"
#include "mlir/Dialect/TPU/TPUTensorSupport.h"
#include "mlir/Dialect/TPU/QuantizationArithmetic.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Support/TensorFile.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/Support/Debug.h>
#include <sstream>
#include <fstream>
#include <math.h>
namespace mlir {
using std::cout;
using std::endl;
using std::list;
using std::make_pair;
using std::make_shared;
using std::map;
using std::move;
using std::pair;
using std::set;
using std::shared_ptr;
using std::string;
using std::vector;

#define u64 uint64_t
#define u32 uint32_t

#define gaddr_t uint64_t
#define laddr_t uint32_t

#define NPU_NUM 32
#define EU_NUM 16


static inline int ceiling_func(int numerator, int denominator)
{
  return (numerator + denominator - 1) / denominator;
}

#define __ALIGN_MASK(x,mask)    (((x)+(mask))&~(mask))
#define ALIGN(x,a)              __ALIGN_MASK(x,(__typeof__(x))(a)-1)

#define DATA_TYPE_SIZE 1
#define bmerr_t int
#define BM_SUCCESS 0               // The operation was successful
#define BM_ERR_AGAIN 1             // Not ready yet
#define BM_ERR_FAILURE 2           // General failure
#define BM_ERR_TIMEOUT 3           // Timeout
#define BM_ERR_UNINITIALIZED 4     // Uninitialzed
#define BM_ERR_INVALID_ARGUMENT 5  // Arguments invalid
#define BM_ERR_NOMEM 6             // Not enough memory
#define BM_ERR_DATA 7              // Data error
#define BM_ERR_BUSY 8              // Busy
#define BM_ERR_NOT_SUPPORTED 9     // Not supported yet
#define LOCAL_BANK_SIZE (1 << 12)
#define LOCAL_MEM_SIZE (1 << 15)

#define GLOBAL_MEM_SIZE  0x100000000


typedef enum {
  S2L = 0,
  L2S = 1,
  S2S = 2,
  L2L = 3,
  S2TSM = 4,
  L2TSM = 5,
  TSM2S = 6,
  TSM2L = 7
}TransportDirection;

typedef enum  {
  NEURON = 0,
  COEFF = 1,
  COEFF_INT8 = 2,
  BIAS_INT8 = 3
}TransportDataType;

typedef enum  {
  PRE = 0,
  CUR = 1,
  POST = 2
}TransportStage;

static inline void printFunction(FuncOp * fn) {
  std::string res;
  llvm::raw_string_ostream os(res);
  fn->walk([&](Operation * op) {
    op->print(os);
    os << "\n";
  });
  llvm::errs() << res;
}

static int getOperandStorageSize(Operation *p) {
  auto op = cast<tpu::LoadWeightOp>(p);

  if (op.storage() == "INT8" || op.storage() == "UINT8" ) {
    return 1;
  } else if (op.storage() == "BF16" || op.storage() == "INT16" ||
             op.storage() == "UINT16" ) {
    return 2;
  } else if(op.storage() == "FP32" || op.storage() == "INT32" ||
            op.storage() == "UINT32") {
    return 4;
  } else {
    assert(0);
  }
}

static Type getElementType(MLIRContext *context, int size) {
  Builder builder(context);
  switch(size){
    case 1:
      return builder.getIntegerType(8);
    case 2:
      return builder.getIntegerType(16);
    case 4:
      return builder.getIntegerType(32);
  }
}

static inline bool isValidTpuOp(Operation *op)
{
  return (!isa<tpu::LoadWeightOp>(op) && !isa<tpu::WeightFileOp>(op) &&
          !isa<tpu::NoneOp>(op) &&
          op->getName().getDialect().str() == "tpu");
}

static inline bool isValidLayerGroupOp(Operation *op) {
  if (isa<tpu::LoadWeightOp>(op)
          || isa<tpu::WeightFileOp>(op)
          || isa<tpu::QuantOp>(op)
          || isa<ReturnOp>(op)
          || isa<tpu::NoneOp>(op)) {
            return false;
  } else {
    return true;
  }
}

static inline int64_t top_size(Operation * op) {
  return op->getNumResults();
}

static inline int64_t bottom_size(Operation * op) {
  return op->getNumOperands();
}

static inline llvm::StringRef top_name(Operation * op, int idx) {
  auto op_top = op->getResult(idx)->getDefiningOp();
  if (op_top && isValidTpuOp(op_top)) {
    auto name = mlir::getOpName(op_top);
    return name;
  }
  else if (auto load_op = dyn_cast<tpu::LoadWeightOp>(op_top))
    return load_op.name().getValue();
  else
    return llvm::StringRef();
}

static inline llvm::StringRef bottom_name(Operation * op, int idx) {
  auto op_bottom = op->getOperand(idx)->getDefiningOp();
  if (op_bottom && isValidTpuOp(op_bottom)) {
    auto name = mlir::getOpName(op_bottom);
    return name;
  }
  else if (auto load_op = dyn_cast<tpu::LoadWeightOp>(op_bottom))
    return load_op.name().getValue();
  else
    return llvm::StringRef();

}

static inline llvm::StringRef name(Operation * op) {
  auto op_name = mlir::getOpName(op);
  return op_name;
}

static inline uint64_t id(Operation * op) {
  uint64_t layer_id = mlir::getOpLayerId(op);
  return layer_id;
}

static inline vector<int64_t> input_shape(Operation *op, int idx) {
  vector<int64_t> shape = op->getOperand(idx)->getType().cast<TensorType>().getShape();
  return shape;
}

static inline vector<int64_t> output_shape(Operation *op, int idx) {
  vector<int64_t> shape = op->getResult(idx)->getType().cast<TensorType>().getShape();
  return shape;
}



}
#endif